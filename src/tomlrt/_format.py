"""Canonicalise Container / Array layout without losing comments.

Pure, idempotent slot/trivia/value helpers for ``Container.format`` and
``Array.format``. Inline values are shape-preserving: single-line stays
single-line; multi-line stays multi-line.

Canonical layout enforced here:

KV slot
    pre_eq=" ", post_eq=" ", key_seps=".", strip column-indent WS,
    configurable EOL spacing before comments.

Section / AoT-entry header
    inner_pre="", inner_post="", strip column-indent WS, EOL as for KV.

Sibling spacing (within the subtree of the container being formatted)
    Between body KVs of the same container: 0 blank lines.
    Between sibling sections / AoT entries / between body and next
    section: exactly 1 blank line.

Inline arrays / inline tables
    Single-line: ``[a, b, c]`` / ``{ a = 1, b = 2 }``.
    Multi-line: each item on its own line with configurable recursive
    indentation and a configurable final comma.

Orphan comment blocks (``# …`` runs separated from the slot/header
by a blank line) and EOL / leading-attached comments are preserved
in place. Runs of blank lines collapse to one; comment text is
rewritten to ``# body`` form when ``normalize_comments`` is enabled.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

from tomlrt._comma_ops import Boundary
from tomlrt._errors import TOMLError
from tomlrt._slots import KVSlot, StructuralHeaderSlot, ensure_terminator
from tomlrt._trivia import (
    leading_break,
    retarget_eol_newline,
    retarget_newlines,
    split_above_block,
    split_eol_section,
    split_item_above,
    split_line,
    split_lines,
    trailing_ws,
)
from tomlrt._values import (
    ArrayValue,
    InlineTableEntry,
    InlineTableValue,
    item_eol_channel,
    item_has_any_comment,
    set_item_eol_channel,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tomlrt._slots import AoTEntry, Slot
    from tomlrt._trivia import EolTrivia
    from tomlrt._values import (
        CommaItem,
        Value,
    )


def _validate_non_negative(value: int, name: str) -> None:
    if value < 0:
        msg = f"{name} must be non-negative"
        raise ValueError(msg)


@dataclass(frozen=True, slots=True, kw_only=True)
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

    normalize_comments: bool = True
    indent: int = 2
    eol_comment_spaces: int = 1
    multiline_trailing_comma: bool = True

    def __post_init__(self) -> None:
        _validate_non_negative(self.indent, "indent")
        _validate_non_negative(self.eol_comment_spaces, "eol_comment_spaces")


_DEFAULT_FORMAT_OPTIONS = FormatOptions()


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
    """Rewrite a comment lexeme to canonical ``# body`` / ``#`` form.

    Input starts with ``#``; leading body whitespace collapses to one
    space and trailing whitespace is stripped.
    """
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

    Strips trailing whitespace on blank lines. Full-line comments drop
    pre-comment whitespace, or restamp to ``comment_indent`` for
    multi-line inline element comments.

    ``first_line_is_eol`` treats only the pre-first-newline run as an
    EOL context; this covers bracket pads whose opening row stores a
    row-attached EOL comment. When ``comments`` is true, comment text is
    rewritten via :func:`_canon_comment_text`. Newline text is retargeted
    by callers.
    """
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

    Splits leading trivia into head blanks, middle comment/orphan block,
    and trailing column indent. Canonical form keeps the middle (with
    newline/comment cleanup), drops the column indent, and applies
    ``target_blanks`` to the head.

    ``target_blanks=None`` preserves preamble/subtree-boundary blanks,
    optionally capped by ``max_preserved_blanks``. When ``middle`` is
    non-empty, clamp the authored head gap to 0/1, but never below the
    canonical target, so comment-block separation intent survives without
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

    # Preamble/subtree boundaries keep authored head gaps; attached
    # comment blocks clamp to 0/1 so separation intent survives.
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


def _canon_eol(eol: EolTrivia, *, nl: str, options: FormatOptions) -> None:
    """Normalise an :class:`EolTrivia`.

    Retargets newline to ``nl`` (leaving ``None`` for a no-final-newline
    tail), canonicalises optional comment text, and applies the configured
    separator before comments.
    """
    retarget_eol_newline(eol, nl)
    if not eol.comment:
        eol.trailing_ws = ""
        return
    if options.normalize_comments:
        eol.comment = _canon_comment_text(eol.comment)
    eol.trailing_ws = " " * options.eol_comment_spaces


# ---------------------------------------------------------------------------
# Key parts
# ---------------------------------------------------------------------------


def _canon_key_equals(node: KVSlot | InlineTableEntry) -> None:
    """Canonicalise the key / ``=`` body of a KV slot or inline-table entry."""
    node.pre_eq = " "
    node.post_eq = " "
    node.key_seps = ["."] * (len(node.key_parts) - 1)


# ---------------------------------------------------------------------------
# Slots — KV / Header
# ---------------------------------------------------------------------------


def _canon_kv_slot(slot: KVSlot, *, nl: str, options: FormatOptions) -> None:
    """Normalise a KV slot's body (key/eq/value/eol).

    Leading is handled separately by :func:`_canon_leading` so that
    the subtree-aware blank-line policy can be applied at the walk
    level.
    """
    _canon_key_equals(slot)
    _canon_value(slot.value, nl=nl, options=options)
    _canon_eol(slot.eol, nl=nl, options=options)


def _canon_header_slot(
    slot: StructuralHeaderSlot, *, nl: str, options: FormatOptions
) -> None:
    slot.inner_pre = ""
    slot.inner_post = ""
    slot.key_seps = ["."] * (len(slot.key_parts) - 1)
    _canon_eol(slot.eol, nl=nl, options=options)


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

    Shared by the format walk and ``Array.set_multiline``. Canonicalises
    per-item trivia, restamps bracket pads, then retargets newlines and
    rewrites comment text. The single-line path bypasses this because it
    produces only empty/single-space trivia.
    """
    items = v.items
    above_blocks: list[str] = []
    for i in range(len(items)):
        boundary = Boundary.capture(v, i)
        above_blocks.append(boundary.above)
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
        head_above = _format_above_block(above_blocks[0])
        v.header_trivia = _compose_pad(
            head_eol=head_eol,
            above=head_above,
            nl=nl,
            trailing_indent=item_indent,
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
            row_already_closed=last_row_closed,
        )
        final_eol_first = False
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
        )
        final_eol_first = bool(final_eol)
    _finalise_inline_trivia(
        v,
        nl=nl,
        options=options,
        item_indent=item_indent,
        final_first_line_is_eol=final_eol_first,
        final_row_already_closed=last_row_closed if items else False,
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
    row_already_closed: bool = False,
) -> str:
    r"""Compose a bracket-pad from (row-attached EOL, above-block, indent).

    Layout is ``head_eol`` (already terminated when non-empty), optional
    structural ``\n``, above-block, then trailing indent. Skip the
    structural newline when ``head_eol`` or the upstream item EOL channel
    already closed the row.
    """
    head = head_eol if head_eol or row_already_closed else nl
    return head + above + trailing_indent


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


def _finalise_inline_trivia(
    v: ArrayValue | InlineTableValue,
    *,
    nl: str,
    options: FormatOptions,
    item_indent: str = "",
    final_first_line_is_eol: bool = False,
    final_row_already_closed: bool = False,
) -> None:
    """Retarget newlines + canonicalise comment / blank-WS text across ``v``.

    Runs after shape canonicalisation over bracket pads and all per-item
    trivia. ``item_indent`` keeps full-line comments aligned with
    multi-line items, not stripped to column 0.

    ``final_first_line_is_eol`` covers the empty-value case where the
    opening bracket's row-attached EOL lives in ``final_trivia`` and must
    be treated as EOL context.
    """
    v.header_trivia = retarget_newlines(v.header_trivia, nl)
    v.final_trivia = retarget_newlines(v.final_trivia, nl)
    v.header_trivia = _canon_trivia_text(
        v.header_trivia,
        comments=options.normalize_comments,
        comment_indent=item_indent,
        first_line_is_eol=True,
    )
    v.final_trivia = _canon_trivia_text(
        v.final_trivia,
        comments=options.normalize_comments,
        comment_indent=item_indent,
        first_line_is_eol=(
            final_first_line_is_eol
            or (bool(leading_break(v.final_trivia)) and not final_row_already_closed)
        ),
    )
    for k, it in enumerate(v.items):
        it.leading = retarget_newlines(it.leading, nl)
        previous_row_closed = k > 0 and "\n" in item_eol_channel(v.items[k - 1])
        it.leading = _canon_trivia_text(
            it.leading,
            comments=options.normalize_comments,
            comment_indent=item_indent,
            first_line_is_eol=(
                bool(leading_break(it.leading)) and not previous_row_closed
            ),
        )


def _canon_multi_line_items(
    items: Sequence[CommaItem],
    *,
    above_blocks: Sequence[str],
    nl: str,
    indent: str,
    options: FormatOptions,
) -> bool:
    r"""Canonicalise per-item trivia for a multi-line inline value.

    Returns whether the last item's EOL channel closed its row, so the caller
    can avoid adding a duplicate ``final_trivia`` newline.

    Item 0's structural pad lives in ``header_trivia``, so its leading is
    empty. Later items keep their above-item comment block but get
    canonical newline+indent, suppressed when the previous item's
    upstream EOL channel already closed the row.

    Internal items and, when requested, the final item use commas. Any
    EOL comment moves between ``trailing`` and ``post_comma_trivia`` as
    the comma state changes; both channels are then reduced to the
    canonical EOL section or empty.
    """
    previous_row_closed = False
    last_index = len(items) - 1
    for k, it in enumerate(items):
        if k == 0:
            it.leading = ""
        else:
            above = _format_above_block(above_blocks[k])
            it.leading = _compose_pad(
                head_eol="",
                above=above,
                nl=nl,
                trailing_indent=indent,
                row_already_closed=previous_row_closed,
            )
        # Changing comma state may shift comments between ``trailing``
        # and ``post_comma_trivia``; collect both sides before clearing them.
        comments = [
            comment
            for trivia in (it.trailing, it.post_comma_trivia)
            for line in split_lines(trivia)
            if (comment := split_line(line)[1])
        ]
        it.has_comma = k < last_index or options.multiline_trailing_comma
        it.trailing = ""
        it.post_comma_trivia = ""
        previous_row_closed = _canon_item_eol(
            it,
            comments,
            nl=nl,
            options=options,
            indent=indent,
        )
    return previous_row_closed


def _canon_item_eol(
    item: CommaItem,
    comments: Sequence[str],
    *,
    nl: str,
    options: FormatOptions,
    indent: str,
) -> bool:
    r"""Write ``comments`` onto the item's EOL channel.

    Returns whether they close the row.

    The first comment stays on the item row; further comments occupy indented
    lines below it. With no comments, the row stays open for the next item's
    leading pad to terminate.
    """
    if not comments:
        return False
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
    return True


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
) -> None:
    """Switch a comma-value between flush single-line and multi-line form.

    Shared by `Array.set_multiline` and inline-table ``set_multiline``.
    Collapsing raises `TOMLError` when a comment would be orphaned. The
    single-line bracket pad is driven by ``value._single_line_pad`` (via
    `_canon_single_line_inline`), so arrays collapse tight (``[1, 2]``)
    while inline tables keep their pad (``{ a = 1 }``).
    """
    items = value.items
    outer_indent = _closing_indent(value)
    # The explicit single<->multi toggle is the one operation that can flip
    # shape without removing an item; drop the memo so it recomputes.
    value.reset_multiline_cache()
    if not multiline:
        for it in items:
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
        return
    for it in items:
        _canon_value(
            it.value,
            nl=nl,
            options=_DEFAULT_FORMAT_OPTIONS,
            parent_indent=indent,
        )
    _canon_multiline_shape(
        value,
        nl=nl,
        options=_DEFAULT_FORMAT_OPTIONS,
        item_indent=indent,
        outer_indent=outer_indent,
    )


def _closing_indent(value: ArrayValue | InlineTableValue) -> str:
    """Return the whitespace immediately before a multiline closing bracket."""
    if not value.is_multiline():
        return ""
    return trailing_ws(value.final_trivia)


def format_inline_root(
    value: ArrayValue | InlineTableValue,
    *,
    nl: str,
    options: FormatOptions,
) -> None:
    """Canonicalise an inline array/table formatted on its own.

    Uses the value's existing closing-bracket column as the base indent,
    so formatting it in isolation doesn't reset its position relative to
    its enclosing document.
    """
    _canon_inline_value(
        value, nl=nl, options=options, parent_indent=_closing_indent(value)
    )


# ---------------------------------------------------------------------------
# Subtree walk and orchestration
# ---------------------------------------------------------------------------


def _in_subtree(
    slot: Slot,
    path: tuple[str, ...],
    owner: AoTEntry | None,
) -> bool:
    """Return whether ``slot`` belongs to the ``path`` / ``owner`` subtree.

    ``owner`` disambiguates AoT-entry views (and sections nested inside
    them), where sibling entries share the same path. Matching slots must
    belong to ``owner`` or to a descendant AoT entry under ``owner.path``.
    """
    if isinstance(slot, KVSlot):
        assert slot.host_path[: len(path)] == path
    else:
        assert isinstance(slot, StructuralHeaderSlot)
        if slot.path[: len(path)] != path:
            return False
    if owner is None:
        return True
    slot_owner = slot.owner_aot_entry
    if slot_owner is owner:
        return True
    assert slot_owner is not None
    op = owner.path
    return len(slot_owner.path) > len(op) and slot_owner.path[: len(op)] == op


def format_subtree(
    *,
    start: Slot | None,
    path: tuple[str, ...],
    owner: AoTEntry | None,
    nl: str,
    options: FormatOptions,
    head_blank_cap: int | None = None,
) -> None:
    """Canonicalise every slot in the subtree rooted at ``path``.

    Walks the doc-stream from ``start`` until the first outside slot.
    ``owner`` disambiguates AoT-entry subtrees that share ``path``.

    The first slot's leading head-blanks belong to the parent subtree.
    ``head_blank_cap`` bounds how many survive: ``None`` preserves them
    (a nested subtree owns its opening boundary), while a whole-document
    walk passes the count that leaves one blank line between the document
    start -- or a preamble, which already supplies one -- and the first
    slot. Later slots get the canonical count: 1 blank line before a
    structural header, 0 otherwise.
    """
    prev: Slot | None = None
    slot = start
    while slot is not None and _in_subtree(slot, path, owner):
        # A slot parsed as the file's final line carries
        # ``eol.newline=None``.  If a later mutation (sort, splice)
        # moved it off the tail, restore the terminator so the
        # canonical inter-slot blank line materialises.  The walk's
        # genuinely-final slot is never visited as ``prev``, so its
        # no-final-newline state survives.
        if prev is not None:
            ensure_terminator(prev, nl)
        if isinstance(slot, KVSlot):
            _canon_kv_slot(slot, nl=nl, options=options)
        else:
            assert isinstance(slot, StructuralHeaderSlot), "unknown slot type"
            _canon_header_slot(slot, nl=nl, options=options)
        if prev is None:
            target: int | None = None
        else:
            target = 1 if isinstance(slot, StructuralHeaderSlot) else 0
        _canon_leading(
            slot,
            nl=nl,
            target_blanks=target,
            options=options,
            max_preserved_blanks=head_blank_cap if prev is None else None,
        )
        prev = slot
        slot = slot._next  # noqa: SLF001


def format_document_trailing(
    trailing: str,
    *,
    nl: str,
    options: FormatOptions,
) -> str:
    """Canonicalise the trailing trivia of a :class:`Document`.

    Retargets newlines, collapses blank-line runs, strips blank-line
    trailing whitespace and column-indent whitespace before orphan
    comments, and optionally rewrites comment text. The
    preamble/epilogue split is unaffected.
    """
    return _canon_trivia_text(
        retarget_newlines(trailing, nl), comments=options.normalize_comments
    )


__all__ = [
    "FormatOptions",
    "_canon_header_slot",
    "_canon_inline_value",
    "_canon_kv_slot",
    "_canon_leading",
    "_closing_indent",
    "_resolve_format_options",
    "format_document_trailing",
    "format_inline_root",
    "format_subtree",
    "set_comma_value_multiline",
]
