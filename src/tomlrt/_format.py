"""Canonicalise Container / Array layout without losing comments.

Pure and idempotent. Shape-preserving: single-line inline values stay
single-line, multi-line stay multi-line.

Canonical layout:

KV slot
    ``pre_eq=" "``, ``post_eq=" "``, ``key_seps="."``, no column-indent
    whitespace, configurable EOL spacing before comments.

Section / AoT-entry header
    ``inner_pre=""``, ``inner_post=""``, otherwise as for KV.

Sibling spacing (within the subtree being formatted)
    0 blank lines between body KVs of the same container; 1 blank line
    between sibling sections / AoT entries, or between body and the
    next section.

Inline arrays / inline tables
    Single-line: ``[a, b, c]`` / ``{ a = 1, b = 2 }``. Multi-line: one
    item per line, with configurable indentation and a configurable
    final comma.

Orphan comment blocks and EOL / leading-attached comments are preserved
in place. Blank-line runs collapse to one; comment text is rewritten to
``# body`` form when ``normalize_comments`` is enabled.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from tomlrt._comma_ops import Boundary
from tomlrt._errors import TOMLError
from tomlrt._slots import KVSlot, StructuralHeaderSlot, ensure_terminator
from tomlrt._trivia import (
    leading_ws,
    retarget_newlines,
    split_above_block,
    split_eol_section,
    split_item_above,
    split_line,
    split_lines,
)
from tomlrt._values import (
    ArrayValue,
    InlineTableEntry,
    InlineTableValue,
    item_has_any_comment,
    set_item_eol_channel,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from tomlrt._slots import Slot
    from tomlrt._values import (
        CommaItem,
        Value,
    )


def _validate_non_negative(value: int, name: str) -> None:
    if value < 0:
        msg = f"{name} must be non-negative"
        raise ValueError(msg)


def _prepare_indent(indent: int) -> str:
    """Validate and prepare indentation before a multiline edit starts."""
    _validate_non_negative(indent, "indent")
    return " " * indent


class FormatOptions:
    """Canonical formatting options shared by all ``format()`` methods.

    ``normalize_comments`` rewrites comment text to canonical ``# body`` form
    and strips trailing whitespace. Layout around comments is canonicalised
    regardless.

    ``indent`` is the number of spaces added at each nested multiline array
    or inline-table level.

    ``eol_comment_spaces`` is the number of spaces inserted before supported
    end-of-line comments.

    ``multiline_trailing_comma`` controls whether the final item in a multiline
    array or inline table has a comma.
    """

    __slots__ = (
        "eol_comment_spaces",
        "indent",
        "multiline_trailing_comma",
        "normalize_comments",
    )

    def __init__(
        self,
        *,
        normalize_comments: bool = True,
        indent: int = 2,
        eol_comment_spaces: int = 1,
        multiline_trailing_comma: bool = True,
    ) -> None:
        _validate_non_negative(indent, "indent")
        _validate_non_negative(eol_comment_spaces, "eol_comment_spaces")
        self.normalize_comments = normalize_comments
        self.indent = indent
        self.eol_comment_spaces = eol_comment_spaces
        self.multiline_trailing_comma = multiline_trailing_comma


_DEFAULT_FORMAT_OPTIONS = FormatOptions()

#: Options for a pure shape change (`set_comma_value_multiline`), which
#: re-lays a value's own rows but is not a request to reformat its text,
#: so comment lexemes are left as the author wrote them.
_SHAPE_ONLY_OPTIONS = FormatOptions(normalize_comments=False)


def _resolve_format_options(
    *,
    options: FormatOptions | None,
    comments: bool | None,
) -> FormatOptions:
    """Resolve public formatting arguments and warn for ``comments=``."""
    if options is not None and comments is not None:
        msg = "cannot specify both options and deprecated comments"
        raise ValueError(msg)
    if comments is not None:
        warnings.warn(
            "comments= is deprecated; use "
            "FormatOptions(normalize_comments=...) instead",
            DeprecationWarning,
            stacklevel=3,
        )
        return FormatOptions(normalize_comments=comments)
    if options is None:
        return _DEFAULT_FORMAT_OPTIONS
    return options


# ---------------------------------------------------------------------------
# Comment text + line cleanup
# ---------------------------------------------------------------------------


def _canon_comment_text(text: str) -> str:
    """Rewrite a comment lexeme to canonical ``# body`` / ``#`` form."""
    assert text.startswith("#"), text
    body = text[1:].rstrip().lstrip(" \t")
    return "# " + body if body else "#"


def _canon_trivia_text(
    t: str,
    *,
    comments: bool,
    comment_indent: str = "",
    first_line_is_eol: bool = False,
) -> str:
    r"""Return ``t`` with its content normalised.

    Strips trailing whitespace from blank lines; full-line comments drop
    pre-comment whitespace, restamped to ``comment_indent`` for
    multi-line inline elements.

    ``first_line_is_eol`` restricts EOL context to the text before the
    first newline, for bracket pads whose opening row holds a row-
    attached EOL comment. Comment text is rewritten via
    :func:`_canon_comment_text` when ``comments`` is true; newline
    retargeting is left to the caller.
    """
    if "\n" not in t and "\r" not in t and "#" not in t:
        # No comment to rewrite, and no line terminator, so no complete
        # line whose trailing whitespace could be stripped: the walk
        # below would rebuild ``t`` unchanged.
        return t
    out: list[str] = []
    in_eol = first_line_is_eol
    previous_line_blank = False
    for line in split_lines(t):
        pre, comment, term = split_line(line)
        if comment:
            if not in_eol:
                pre = comment_indent
            out.append(pre + (_canon_comment_text(comment) if comments else comment))
            out.append(term)
            previous_line_blank = False
        elif term:
            content = pre.rstrip(" \t")
            line_blank = not in_eol and not content
            if not (line_blank and previous_line_blank):
                out.append(content + term)
            previous_line_blank = line_blank
        else:
            out.append(pre)
        in_eol = False
    return "".join(out)


# ---------------------------------------------------------------------------
# Leading trivia of slots
# ---------------------------------------------------------------------------


def _canon_leading(
    slot: Slot,
    *,
    nl: str,
    target_blanks: int | None,
    options: FormatOptions,
    max_preserved_blanks: int | None = None,
) -> None:
    """Rewrite ``slot.leading`` to canonical form.

    Splits into head blanks, middle comment/orphan block, and trailing
    column indent; keeps the middle (with newline/comment cleanup),
    drops the indent, and applies ``target_blanks`` to the head.

    ``target_blanks=None`` preserves preamble/subtree-boundary blanks,
    capped by ``max_preserved_blanks``. When ``middle`` is non-empty,
    clamp the authored head gap to 0/1 but never below the canonical
    target, so comment-block separation intent survives without
    suppressing structural-header spacing.
    """
    lines = split_lines(slot.leading)

    if lines and "\n" not in lines[-1] and "#" not in lines[-1]:
        lines.pop()

    head_count = 0
    while head_count < len(lines) and "#" not in lines[head_count]:
        assert "\n" in lines[head_count]
        head_count += 1
    middle = lines[head_count:]

    middle_t = retarget_newlines("".join(middle), nl)
    middle_t = _canon_trivia_text(middle_t, comments=options.normalize_comments)

    if target_blanks is None:
        n_blanks = head_count
        if max_preserved_blanks is not None:
            n_blanks = min(n_blanks, max_preserved_blanks)
    elif middle:
        n_blanks = max(target_blanks, min(head_count, 1))
    else:
        n_blanks = target_blanks
    slot.leading = nl * n_blanks + middle_t


# ---------------------------------------------------------------------------
# EOL
# ---------------------------------------------------------------------------


def _canon_eol(eol: str, *, nl: str, options: FormatOptions) -> str:
    """Normalise an EOL run: retarget, canonicalise, respace."""
    eol = retarget_newlines(eol, nl)
    if "#" not in eol:
        return eol.lstrip(" \t")
    _pre, comment, term = split_line(eol)
    if options.normalize_comments:
        comment = _canon_comment_text(comment)
    return f"{' ' * options.eol_comment_spaces}{comment}{term}"


# ---------------------------------------------------------------------------
# Key parts
# ---------------------------------------------------------------------------


def _canon_key_equals(node: KVSlot | InlineTableEntry) -> None:
    """Canonicalise the key / ``=`` body of a KV slot or inline-table entry."""
    node.pre_eq = " "
    node.post_eq = " "
    node.key_seps = (".",) * (len(node.key_parts) - 1)


# ---------------------------------------------------------------------------
# Slots — KV / Header
# ---------------------------------------------------------------------------


def _canon_slot(
    slot: Slot,
    *,
    nl: str,
    target_blanks: int | None,
    options: FormatOptions,
    max_preserved_blanks: int | None = None,
) -> None:
    """Canonicalise one slot: its body, then its leading trivia.

    Leading comes last so the caller's blank-line policy
    (``target_blanks`` / ``max_preserved_blanks``) has the final say.
    """
    if isinstance(slot, KVSlot):
        _canon_key_equals(slot)
        _canon_value(slot.value, nl=nl, options=options)
    else:
        assert isinstance(slot, StructuralHeaderSlot), "unknown slot type"
        slot.inner_pre = ""
        slot.inner_post = ""
        slot.key_seps = (".",) * (len(slot.key_parts) - 1)
    slot.eol = _canon_eol(slot.eol, nl=nl, options=options)
    _canon_leading(
        slot,
        nl=nl,
        target_blanks=target_blanks,
        options=options,
        max_preserved_blanks=max_preserved_blanks,
    )


# ---------------------------------------------------------------------------
# Inline values
# ---------------------------------------------------------------------------


def _canon_inline_value(
    v: ArrayValue | InlineTableValue,
    *,
    nl: str,
    options: FormatOptions,
    parent_indent: str = "",
) -> None:
    """Canonicalise inline array/table layout while preserving shape."""
    items = v.items
    multi = v.is_multiline()
    item_indent = parent_indent + (" " * options.indent) if multi else parent_indent

    for it in items:
        if isinstance(it, InlineTableEntry):
            _canon_key_equals(it)
        _canon_value(it.value, nl=nl, options=options, parent_indent=item_indent)

    if not multi:
        _canon_single_line_inline(v)
        return

    _canon_multiline_shape(
        v,
        nl=nl,
        options=options,
        item_indent=item_indent,
        outer_indent=parent_indent,
    )


def _canon_multiline_shape(
    v: ArrayValue | InlineTableValue,
    *,
    nl: str,
    options: FormatOptions,
    item_indent: str,
    outer_indent: str,
) -> None:
    """Apply multi-line canonical shape to ``v``.

    Harvests above-item comments before reshaping their boundaries.
    Comment-free values need no harvest: their pads are rebuilt from
    ``nl`` and indentation alone. Each pad canonicalises any carried
    text as it is composed. Nested values are untouched.
    """
    items = v.items
    above_blocks: list[str] = [""] * len(items)
    if v.has_own_comment():
        for i in range(len(items)):
            boundary = Boundary.capture(v, i)
            above_blocks[i] = _format_above_block(boundary.above)
            boundary.remove_above().restore(v, i)
    last_row_closed = _canon_multi_line_items(
        items,
        above_blocks=above_blocks,
        nl=nl,
        indent=item_indent,
        options=options,
    )
    if items:
        head_eol, _ = split_eol_section(v.header_trivia)
        head_above = above_blocks[0]
        v.header_trivia = _compose_pad(
            head_eol=head_eol,
            above=head_above,
            nl=nl,
            trailing_indent=item_indent,
            comment_indent=item_indent,
            options=options,
        )
        # Unlike ``header_trivia``, ``final_trivia`` has no bracket-EOL first
        # line, so split it as an item boundary rather than treating its
        # leading comment as bracket framing.
        final_above = _format_above(
            v.final_trivia,
            row_already_closed=last_row_closed,
        )
        v.final_trivia = _compose_pad(
            head_eol="",
            above=final_above,
            nl=nl,
            trailing_indent=outer_indent,
            comment_indent=item_indent,
            options=options,
            row_already_closed=last_row_closed,
        )
    else:
        # An empty multi-line value carries all of its trivia
        # (bracket-EOL + above-block + closing pad) in final_trivia;
        # header_trivia is empty by construction.
        final_eol, _ = split_eol_section(v.final_trivia)
        _, final_above = split_above_block(v.final_trivia)
        v.header_trivia = ""
        v.final_trivia = _compose_pad(
            head_eol=final_eol,
            above=final_above,
            nl=nl,
            trailing_indent=outer_indent,
            comment_indent=item_indent,
            options=options,
        )


def _format_above(t: str, *, row_already_closed: bool) -> str:
    """Return an above-item block with any authored separator it owns.

    When the upstream EOL channel already closed the row, the leading
    newline belongs to the blank-line separator rather than the row
    terminator, so keep it with the above block.
    """
    head, above, _ = split_item_above(t)
    if "#" not in above:
        return ""
    return head + above if row_already_closed else above


def _format_above_block(block: str) -> str:
    """Keep an authored logical above-block only when it has comments."""
    return block if "#" in block else ""


def _compose_pad(
    *,
    head_eol: str,
    above: str,
    nl: str,
    trailing_indent: str,
    comment_indent: str,
    options: FormatOptions,
    row_already_closed: bool = False,
) -> str:
    r"""Compose and canonicalise a pad from EOL, above-block, and indent.

    Skips the structural newline when ``head_eol`` or the upstream item
    EOL channel already closed the row. Closing-bracket pads indent their
    comments with the items, independently of the bracket's own indent.
    """
    head = head_eol if head_eol or row_already_closed else nl
    text = head + above + trailing_indent
    if not head_eol and not above:
        return text
    return _canon_trivia_text(
        retarget_newlines(text, nl),
        comments=options.normalize_comments,
        comment_indent=comment_indent,
        first_line_is_eol=bool(head_eol) or not row_already_closed,
    )


def _inner_space(v: ArrayValue | InlineTableValue) -> str:
    """Bracket-inner padding for a single-line inline value.

    Inline tables wear ``{ a = 1 }`` with one space; inline arrays wear
    ``[a, b]`` with none.
    """
    return v._single_line_pad if v.items else ""  # noqa: SLF001


def _canon_single_line_inline(v: ArrayValue | InlineTableValue) -> None:
    v.header_trivia = _inner_space(v)
    v.final_trivia = _inner_space(v)
    items = v.items
    n = len(items)
    for k, it in enumerate(items):
        it.leading = "" if k == 0 else " "
        it.trailing = ""
        it.post_comma_trivia = ""
        it.has_comma = k < n - 1


def _canon_multi_line_items(
    items: Sequence[CommaItem],
    *,
    above_blocks: Sequence[str],
    nl: str,
    indent: str,
    options: FormatOptions,
) -> bool:
    r"""Canonicalise per-item trivia for a multi-line inline value.

    Returns whether the last item's EOL channel closed its row, so the
    caller can avoid adding a duplicate ``final_trivia`` newline.

    Item 0's structural pad lives in ``header_trivia``, so its leading
    is empty. Later items keep their above-item comment block but get
    canonical newline+indent, suppressed when the previous item's
    upstream EOL channel already closed the row.

    ``above_blocks`` is already filtered to the blocks worth keeping,
    so an entry is empty unless it carries a comment.
    """
    previous_row_closed = False
    last_index = len(items) - 1
    trailing_comma = options.multiline_trailing_comma
    # A row with no above-block always wants the same pad, so ask for
    # both of its answers once rather than rebuilding them per item.
    open_pad = _compose_pad(
        head_eol="",
        above="",
        nl=nl,
        trailing_indent=indent,
        comment_indent=indent,
        options=options,
    )
    closed_pad = _compose_pad(
        head_eol="",
        above="",
        nl=nl,
        trailing_indent=indent,
        comment_indent=indent,
        options=options,
        row_already_closed=True,
    )
    for k, it in enumerate(items):
        if k == 0:
            it.leading = ""
        else:
            above = above_blocks[k]
            if above:
                it.leading = _compose_pad(
                    head_eol="",
                    above=above,
                    nl=nl,
                    trailing_indent=indent,
                    comment_indent=indent,
                    options=options,
                    row_already_closed=previous_row_closed,
                )
            else:
                it.leading = closed_pad if previous_row_closed else open_pad
        # Changing comma state may shift comments between ``trailing``
        # and ``post_comma_trivia``; read both before clearing them.
        trailing, post_comma = it.trailing, it.post_comma_trivia
        it.has_comma = k < last_index or trailing_comma
        previous_row_closed = False
        if trailing or post_comma:
            it.trailing = it.post_comma_trivia = ""
            if "#" in trailing or "#" in post_comma:
                _write_item_eol(
                    it,
                    [
                        comment
                        for trivia in (trailing, post_comma)
                        for line in split_lines(trivia)
                        if (comment := split_line(line)[1])
                    ],
                    nl=nl,
                    options=options,
                    indent=indent,
                )
                previous_row_closed = True
    return previous_row_closed


def _write_item_eol(
    item: CommaItem,
    comments: Sequence[str],
    *,
    nl: str,
    options: FormatOptions,
    indent: str,
) -> None:
    r"""Write ``comments`` onto the item's EOL channel, closing its row.

    The first comment stays on the item row; further comments occupy indented
    lines below it. Callers only reach here with comments to write; a row
    with none stays open for the next item's leading pad to terminate.
    """
    separator = " " * options.eol_comment_spaces
    if options.normalize_comments:
        comments = [_canon_comment_text(c) for c in comments]
    set_item_eol_channel(
        item,
        "".join(
            f"{separator if k == 0 else indent}{comment}{nl}"
            for k, comment in enumerate(comments)
        ),
    )


# ---------------------------------------------------------------------------
# Value dispatch
# ---------------------------------------------------------------------------


def _canon_value(
    v: Value,
    *,
    nl: str,
    options: FormatOptions,
    parent_indent: str = "",
) -> None:
    if isinstance(v, (ArrayValue, InlineTableValue)):
        _canon_inline_value(v, nl=nl, options=options, parent_indent=parent_indent)
    # Other value kinds carry no formattable trivia.


def set_comma_value_multiline(
    value: ArrayValue | InlineTableValue,
    *,
    multiline: bool,
    nl: str,
    indent: str,
    host: KVSlot | None,
) -> None:
    """Switch a comma-value between flush single-line and multi-line form.

    Only ``value``'s own rows are re-laid: the items' text, nested values
    and comment lexemes are left verbatim, since this is a shape change
    and not a `format` request.

    Collapsing raises `TOMLError` when a comment would be orphaned. The
    single-line bracket pad is driven by ``value._single_line_pad`` (via
    `_canon_single_line_inline`), so arrays collapse tight (``[1, 2]``)
    while inline tables keep their pad (``{ a = 1 }``).

    ``host`` places the closing bracket -- see `_closing_indent`.
    """
    if multiline:
        _canon_multiline_shape(
            value,
            nl=nl,
            options=_SHAPE_ONLY_OPTIONS,
            item_indent=indent,
            outer_indent=_closing_indent(value, host=host),
        )
    else:
        for it in value.items:
            if item_has_any_comment(it):
                msg = (
                    "cannot collapse to single line: "
                    "items contain EOL or leading comments"
                )
                raise TOMLError(msg)
        if "#" in value.header_trivia or "#" in value.final_trivia:
            msg = (
                "cannot collapse to single line: "
                "header or trailing trivia contains comments"
            )
            raise TOMLError(msg)
        _canon_single_line_inline(value)
    # The explicit single<->multi toggle is the one operation that can flip
    # shape without removing an item; drop the memo so it recomputes.
    value.reset_multiline_cache()


def _extend_row(row: str, text: str) -> str:
    """Append ``text`` to ``row``, keeping only the row it ends on."""
    return (row + text).rsplit("\n", 1)[-1]


def _scan_rows(
    v: Value, target: ArrayValue | InlineTableValue, row: str
) -> tuple[str, str | None]:
    """Advance ``row`` across ``v``'s rendering, spotting ``target`` on the way.

    ``row`` is the text of the physical row rendered so far. Returns the
    row ``v`` ends on, paired with the row ``target`` starts on once
    seen; the first result is meaningless from then on, since every
    caller stops on the second.

    An inline-table entry's ``key =`` prefix is skipped: it carries no
    row break, and a value always renders at least one non-blank
    character, so the row's own indent is unaffected either way.
    """
    if v is target:
        return row, row
    if not isinstance(v, (ArrayValue, InlineTableValue)):
        return _extend_row(row, v.render()), None
    row = _extend_row(row, v._open + v.header_trivia)  # noqa: SLF001
    for it in v.items:
        row, found = _scan_rows(it.value, target, _extend_row(row, it.leading))
        if found is not None:
            return row, found
        row = _extend_row(row, it.render_tail())
    return _extend_row(row, v.final_trivia + v._close), None  # noqa: SLF001


def _value_row_in_slot(
    slot: KVSlot, value: ArrayValue | InlineTableValue
) -> str | None:
    """The physical row ``value`` starts on within ``slot``, or ``None``.

    ``None`` when ``value`` does not appear in ``slot``'s value subtree.
    """
    # A KV slot always starts a row, so its leading ends with one.
    head = slot.leading.rsplit("\n", 1)[-1]
    _, found = _scan_rows(slot.value, value, head)
    return found


def _closing_indent(
    value: ArrayValue | InlineTableValue, *, host: KVSlot | None
) -> str:
    """The indent to place ``value``'s closing bracket at.

    The bracket lines up with the row ``value`` starts on. ``host`` is
    the KV slot whose value subtree contains ``value``, resolved by the
    caller in O(depth). A detached value has no host and no enclosing
    row, so it starts at column zero.
    """
    if host is None:
        return ""
    found = _value_row_in_slot(host, value)
    assert found is not None, "internal: value is not under its host slot"
    return leading_ws(found)


def format_inline_root(
    value: ArrayValue | InlineTableValue,
    *,
    nl: str,
    options: FormatOptions,
    host: KVSlot | None,
) -> None:
    """Canonicalise an inline array/table formatted on its own.

    Indents from the row the value starts on -- see
    :func:`_closing_indent` -- so formatting it in isolation keeps it in
    step with its enclosing document rather than resetting it to column
    zero.
    """
    _canon_inline_value(
        value,
        nl=nl,
        options=options,
        parent_indent=_closing_indent(value, host=host),
    )


# ---------------------------------------------------------------------------
# Slot-run canonicalisation
# ---------------------------------------------------------------------------


def format_slots(
    slots: Iterable[Slot],
    *,
    nl: str,
    options: FormatOptions,
    owns_adjacent_gaps: bool,
    head_blank_cap: int | None,
) -> None:
    """Canonicalise one run of physical slots and the values they hold.

    The caller selects the run and says whether it owns the gaps within
    it: a scope whose slots are spelled inside some other block only
    borrows the lines they sit on, so their separators stay as authored.
    Where the run does own them, a structural header takes one blank
    line and anything else none -- and only between slots that are
    physically adjacent, since a gap spanning a foreign slot belongs to
    whoever owns that.

    ``head_blank_cap`` bounds the blanks the first slot keeps; ``None``
    preserves them, as a nested run's opening boundary is its parent's.
    """
    prev: Slot | None = None
    for slot in slots:
        target: int | None = None
        if owns_adjacent_gaps and prev is not None and slot._prev is prev:  # noqa: SLF001
            target = 1 if isinstance(slot, StructuralHeaderSlot) else 0
        _canon_slot(
            slot,
            nl=nl,
            target_blanks=target,
            options=options,
            max_preserved_blanks=head_blank_cap if prev is None else None,
        )
        # Only the actual document tail may retain no final newline.
        if slot._next is not None:  # noqa: SLF001
            ensure_terminator(slot, nl)
        prev = slot


def format_document_trailing(
    trailing: str,
    *,
    nl: str,
    options: FormatOptions,
) -> str:
    """Canonicalise the trailing trivia of a :class:`Document`.

    Retargets newlines and applies the same blank-line / comment
    cleanup as :func:`_canon_trivia_text`. The preamble/epilogue split
    is unaffected.
    """
    return _canon_trivia_text(
        retarget_newlines(trailing, nl), comments=options.normalize_comments
    )


__all__ = [
    "FormatOptions",
    "_canon_inline_value",
    "_closing_indent",
    "_prepare_indent",
    "_resolve_format_options",
    "format_document_trailing",
    "format_inline_root",
    "format_slots",
    "set_comma_value_multiline",
]
