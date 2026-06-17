"""Canonicalise Container / Array layout without losing comments.

Pure, idempotent slot/trivia/value helpers for ``Container.format`` and
``Array.format``. Inline values are shape-preserving: single-line stays
single-line; multi-line stays multi-line.

Canonical layout enforced here:

KV slot
    pre_eq=" ", post_eq=" ", key_seps=".", strip column-indent WS,
    EOL trailing_ws = " " before any comment.

Section / AoT-entry header
    inner_pre="", inner_post="", strip column-indent WS, EOL as for KV.

Sibling spacing (within the subtree of the container being formatted)
    Between body KVs of the same container: 0 blank lines.
    Between sibling sections / AoT entries / between body and next
    section: exactly 1 blank line.

Inline arrays / inline tables
    Single-line: ``[a, b, c]`` / ``{ a = 1, b = 2 }``.
    Multi-line: each item on its own line, last item carries a
    trailing comma. Indent inherited from the first item if present
    else two spaces.

Orphan comment blocks (``# …`` runs separated from the slot/header
by a blank line) and EOL / leading-attached comments are preserved
in place — comment text is rewritten to ``# body`` form when
``comments=True``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tomlrt._comma_ops import _put_eol, _take_eol
from tomlrt._errors import TOMLError
from tomlrt._slots import KVSlot, StructuralHeaderSlot
from tomlrt._trivia import (
    CommentNode,
    NewlineNode,
    Trivia,
    WhitespaceNode,
    line_has_comment,
    line_has_newline,
    retarget_eol_newline,
    retarget_trivia_newlines,
    split_above_block,
    split_eol_section,
    split_item_above,
    split_lines,
    trivia_has_comment,
)
from tomlrt._values import (
    ArrayValue,
    InlineTableEntry,
    InlineTableValue,
    item_has_any_comment,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tomlrt._slots import AoTEntry, Slot
    from tomlrt._trivia import (
        EolTrivia,
        TriviaPiece,
    )
    from tomlrt._values import (
        CommaItem,
        Value,
    )


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
    t: Trivia,
    *,
    comments: bool,
    strip_pre_comment_ws: bool = True,
    comment_indent: str = "",
    first_line_is_eol: bool = False,
) -> None:
    r"""Normalise trivia content in place.

    Strips trailing whitespace on blank lines. Full-line comments drop
    pre-comment whitespace, or restamp to ``comment_indent`` for
    multi-line inline element comments. Set ``strip_pre_comment_ws=False``
    for EOL trivia, where the caller-owned separator before ``#`` must
    survive.

    ``first_line_is_eol`` treats only the pre-first-newline run as an
    EOL context; this covers bracket pads whose opening row stores a
    row-attached EOL comment. When ``comments`` is true, comment text is
    rewritten via :func:`_canon_comment_text`. Newline text is retargeted
    by callers.
    """
    pieces = t.pieces
    new: list[TriviaPiece] = []
    line: list[TriviaPiece] = []
    in_eol = first_line_is_eol
    for p in pieces:
        if isinstance(p, NewlineNode):
            while line and isinstance(line[-1], WhitespaceNode):
                line.pop()
            new.extend(line)
            new.append(p)
            line = []
            in_eol = False
            continue
        if isinstance(p, CommentNode):
            if strip_pre_comment_ws and not in_eol:
                while line and isinstance(line[-1], WhitespaceNode):
                    line.pop()
                if comment_indent:
                    line.append(WhitespaceNode(comment_indent))
            if comments:
                p.text = _canon_comment_text(p.text)
            new.extend(line)
            new.append(p)
            line = []
            continue
        line.append(p)
    new.extend(line)
    t.pieces = new


# ---------------------------------------------------------------------------
# Leading trivia of slots
# ---------------------------------------------------------------------------


def _canon_leading(
    slot: Slot,
    *,
    nl: str,
    target_blanks: int | None,
    comments: bool,
) -> None:
    """Rewrite ``slot.leading`` to canonical form.

    Splits leading trivia into head blanks, middle comment/orphan block,
    and trailing column indent. Canonical form keeps the middle (with
    newline/comment cleanup), drops the column indent, and applies
    ``target_blanks`` to the head.

    ``target_blanks=None`` preserves preamble/subtree-boundary blanks.
    When ``middle`` is non-empty, clamp the head gap to 0/1 so
    comment-block separation intent survives.
    """
    lines = split_lines(slot.leading.pieces)

    if lines and not line_has_newline(lines[-1]) and not line_has_comment(lines[-1]):
        lines.pop()

    head_count = 0
    while head_count < len(lines) and not line_has_comment(lines[head_count]):
        if not line_has_newline(lines[head_count]):
            break
        head_count += 1
    middle = lines[head_count:]

    middle_t = Trivia([p for line in middle for p in line])
    retarget_trivia_newlines(middle_t, nl)
    _canon_trivia_text(middle_t, comments=comments)

    # Preamble/subtree boundaries keep authored head gaps; attached
    # comment blocks clamp to 0/1 so separation intent survives.
    if target_blanks is None:
        n_blanks = head_count
    elif middle:
        n_blanks = min(head_count, 1)
    else:
        n_blanks = target_blanks
    head_t = Trivia([NewlineNode(nl)] * n_blanks)

    slot.leading.pieces = [*head_t.pieces, *middle_t.pieces]


# ---------------------------------------------------------------------------
# EOL
# ---------------------------------------------------------------------------


def _canon_eol(eol: EolTrivia, *, nl: str, comments: bool) -> None:
    """Normalise an :class:`EolTrivia`.

    Retargets newline to ``nl`` (leaving ``None`` for a no-final-newline
    tail), canonicalises optional comment text, and keeps exactly one
    separator space before comments.
    """
    retarget_eol_newline(eol, nl)
    if eol.comment is None:
        eol.trailing_ws = None
        return
    if comments:
        eol.comment.text = _canon_comment_text(eol.comment.text)
    eol.trailing_ws = WhitespaceNode(" ")


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


def _canon_kv_slot(slot: KVSlot, *, nl: str, comments: bool) -> None:
    """Normalise a KV slot's body (key/eq/value/eol).

    Leading is handled separately by :func:`_canon_leading` so that
    the subtree-aware blank-line policy can be applied at the walk
    level.
    """
    _canon_key_equals(slot)
    _canon_value(slot.value, nl=nl, comments=comments)
    _canon_eol(slot.eol, nl=nl, comments=comments)


def _canon_header_slot(slot: StructuralHeaderSlot, *, nl: str, comments: bool) -> None:
    slot.inner_pre = ""
    slot.inner_post = ""
    slot.key_seps = ["."] * (len(slot.key_parts) - 1)
    _canon_eol(slot.eol, nl=nl, comments=comments)


# ---------------------------------------------------------------------------
# Inline values
# ---------------------------------------------------------------------------


def _canon_inline_value(
    v: ArrayValue | InlineTableValue,
    *,
    nl: str,
    comments: bool,
    parent_indent: str = "",
) -> None:
    """Canonicalise inline array/table layout while preserving shape."""
    items = v.items
    multi = v.is_multiline()
    item_indent = parent_indent + "  " if multi else parent_indent

    for it in items:
        if isinstance(it, InlineTableEntry):
            _canon_key_equals(it)
        _canon_value(it.value, nl=nl, comments=comments, parent_indent=item_indent)

    if not multi:
        _canon_single_line_inline(v)
        return

    _canon_multiline_shape(
        v, nl=nl, comments=comments, item_indent=item_indent, outer_indent=parent_indent
    )


def _canon_multiline_shape(
    v: ArrayValue | InlineTableValue,
    *,
    nl: str,
    comments: bool,
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
    last_line_open = _canon_multi_line_items(items, nl=nl, indent=item_indent)
    if items:
        head_eol, _ = split_eol_section(v.header_trivia)
        _, head_above = split_above_block(v.header_trivia)
        v.header_trivia = _compose_pad(
            head_eol=head_eol,
            above=head_above,
            nl=nl,
            trailing_indent=item_indent,
        )
        # Unlike ``header_trivia``, ``final_trivia`` has no bracket-EOL first
        # line, so use ``split_item_above`` (keeps a leading comment) rather
        # than ``split_above_block`` (would drop it as framing).
        _, final_above, _ = split_item_above(v.final_trivia)
        v.final_trivia = _compose_pad(
            head_eol=Trivia(),
            above=final_above,
            nl=nl,
            trailing_indent=outer_indent,
            line_already_open=last_line_open,
        )
        final_eol_first = False
    else:
        # An empty multi-line value carries all of its trivia
        # (bracket-EOL + above-block + closing pad) in final_trivia;
        # header_trivia is empty by construction.
        final_eol, _ = split_eol_section(v.final_trivia)
        _, final_above = split_above_block(v.final_trivia)
        v.header_trivia = Trivia()
        v.final_trivia = _compose_pad(
            head_eol=final_eol,
            above=final_above,
            nl=nl,
            trailing_indent=outer_indent,
        )
        final_eol_first = bool(final_eol.pieces)
    _finalise_inline_trivia(
        v,
        nl=nl,
        comments=comments,
        item_indent=item_indent,
        final_first_line_is_eol=final_eol_first,
    )


def _compose_pad(
    *,
    head_eol: Trivia,
    above: Trivia,
    nl: str,
    trailing_indent: str,
    line_already_open: bool = False,
) -> Trivia:
    r"""Compose a bracket-pad from (row-attached EOL, above-block, indent).

    Layout is ``head_eol`` (already terminated when non-empty), optional
    structural ``\n``, above-block, then trailing indent. Skip the
    structural newline when ``head_eol`` or upstream
    ``post_comma_trivia`` already closed the line.
    """
    pieces: list[TriviaPiece] = list(head_eol.pieces)
    if not head_eol.pieces and not line_already_open:
        pieces.append(NewlineNode(nl))
    pieces.extend(above.pieces)
    if trailing_indent:
        pieces.append(WhitespaceNode(trailing_indent))
    return Trivia(pieces)


def _inner_space(v: ArrayValue | InlineTableValue) -> Trivia:
    """Bracket-inner padding for a single-line inline value.

    Inline tables wear ``{ a = 1 }`` with one space; inline arrays wear
    ``[a, b]`` with none. Each call returns fresh ``Trivia`` for
    independent ``header_trivia`` / ``final_trivia`` assignment.
    """
    if v.items and v._single_line_pad:  # noqa: SLF001
        return Trivia([WhitespaceNode(v._single_line_pad)])  # noqa: SLF001
    return Trivia()


def _canon_single_line_inline(v: ArrayValue | InlineTableValue) -> None:
    v.header_trivia = _inner_space(v)
    v.final_trivia = _inner_space(v)
    items = v.items
    n = len(items)
    for k, it in enumerate(items):
        it.leading = Trivia() if k == 0 else Trivia([WhitespaceNode(" ")])
        it.trailing = Trivia()
        it.post_comma_trivia = Trivia()
        it.has_comma = k < n - 1


def _finalise_inline_trivia(
    v: ArrayValue | InlineTableValue,
    *,
    nl: str,
    comments: bool,
    item_indent: str = "",
    final_first_line_is_eol: bool = False,
) -> None:
    """Retarget newlines + canonicalise comment / blank-WS text across ``v``.

    Runs after shape canonicalisation over bracket pads and all per-item
    trivia. ``item_indent`` keeps full-line comments aligned with
    multi-line items, not stripped to column 0.

    ``final_first_line_is_eol`` covers the empty-value case where the
    opening bracket's row-attached EOL lives in ``final_trivia`` and must
    be treated as EOL context.
    """
    retarget_trivia_newlines(v.header_trivia, nl)
    retarget_trivia_newlines(v.final_trivia, nl)
    _canon_trivia_text(
        v.header_trivia,
        comments=comments,
        comment_indent=item_indent,
        first_line_is_eol=True,
    )
    _canon_trivia_text(
        v.final_trivia,
        comments=comments,
        comment_indent=item_indent,
        first_line_is_eol=final_first_line_is_eol,
    )
    for it in v.items:
        retarget_trivia_newlines(it.leading, nl)
        retarget_trivia_newlines(it.trailing, nl)
        retarget_trivia_newlines(it.post_comma_trivia, nl)
        _canon_trivia_text(it.leading, comments=comments, comment_indent=item_indent)
        # ``trailing`` and ``post_comma_trivia`` are EOL contexts:
        # the space before ``#`` is the structural separator the
        # callers have already canonicalised to one space.
        _canon_trivia_text(it.trailing, comments=comments, strip_pre_comment_ws=False)
        _canon_trivia_text(
            it.post_comma_trivia, comments=comments, strip_pre_comment_ws=False
        )


def _canon_multi_line_items(
    items: Sequence[CommaItem],
    *,
    nl: str,
    indent: str,
) -> bool:
    r"""Canonicalise per-item trivia for a multi-line inline value.

    Returns whether the last item's ``post_comma_trivia`` closed its row,
    so the caller can avoid adding a duplicate ``final_trivia`` newline.

    Item 0's structural pad lives in ``header_trivia``, so its leading is
    empty. Later items keep their above-item comment block but get
    canonical newline+indent, suppressed when the previous item's
    ``post_comma_trivia`` already closed the row.

    All items use trailing-comma style. If an item lacked a comma, move
    its EOL comment from ``trailing`` to ``post_comma_trivia`` so the
    comma renders before the comment; then shrink ``post_comma_trivia``
    to an EOL section or empty.
    """
    prev_line_open = False
    for k, it in enumerate(items):
        if k == 0:
            it.leading = Trivia()
        else:
            _, above, _ = split_item_above(it.leading)
            it.leading = _compose_pad(
                head_eol=Trivia(),
                above=above,
                nl=nl,
                trailing_indent=indent,
                line_already_open=prev_line_open,
            )
        # Synthesising a comma may shift the EOL row from ``trailing``
        # to ``post_comma_trivia``; the take/put pair preserves the
        # comment across any has_comma flip.
        eol = _take_eol(it)
        it.has_comma = True
        it.trailing = Trivia()
        _put_eol(it, eol)
        prev_line_open = _canon_post_comma_trivia(it, nl=nl)
    return prev_line_open


def _canon_post_comma_trivia(item: CommaItem, *, nl: str) -> bool:
    r"""Canonicalise ``post_comma_trivia``; return True if the line was closed.

    If the first non-whitespace piece is a comment it is preserved as
    an EOL row (``" # comment\\n"``) and we report the line as
    closed; otherwise the channel is cleared and the line stays open
    for the next item's leading pad to terminate.
    """
    first = next(
        (p for p in item.post_comma_trivia.pieces if not isinstance(p, WhitespaceNode)),
        None,
    )
    if isinstance(first, CommentNode):
        item.post_comma_trivia = Trivia([WhitespaceNode(" "), first, NewlineNode(nl)])
        return True
    item.post_comma_trivia = Trivia()
    return False


# ---------------------------------------------------------------------------
# Value dispatch
# ---------------------------------------------------------------------------


def _canon_value(v: Value, *, nl: str, comments: bool, parent_indent: str = "") -> None:
    if isinstance(v, (ArrayValue, InlineTableValue)):
        _canon_inline_value(v, nl=nl, comments=comments, parent_indent=parent_indent)
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
        if trivia_has_comment(value.header_trivia) or trivia_has_comment(
            value.final_trivia
        ):
            msg = (
                "cannot collapse to single line: "
                "header or trailing trivia contains comments"
            )
            raise TOMLError(msg)
        _canon_single_line_inline(value)
        return
    for it in items:
        _canon_value(it.value, nl=nl, comments=True, parent_indent=indent)
    _canon_multiline_shape(
        value, nl=nl, comments=True, item_indent=indent, outer_indent=""
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
        if slot.host_path[: len(path)] != path:
            return False
    else:
        assert isinstance(slot, StructuralHeaderSlot)
        if slot.path[: len(path)] != path:
            return False
    if owner is None:
        return True
    slot_owner = slot.owner_aot_entry
    if slot_owner is owner:
        return True
    if slot_owner is None:
        return False
    op = owner.path
    return len(slot_owner.path) > len(op) and slot_owner.path[: len(op)] == op


def format_subtree(
    *,
    start: Slot | None,
    path: tuple[str, ...],
    owner: AoTEntry | None,
    nl: str,
    comments: bool,
) -> None:
    """Canonicalise every slot in the subtree rooted at ``path``.

    Walks the doc-stream from ``start`` until the first outside slot.
    ``owner`` disambiguates AoT-entry subtrees that share ``path``.

    The first slot's leading head-blanks belong to the document preamble
    or parent subtree and are preserved. Later slots get the canonical
    count: 1 blank line before a structural header, 0 otherwise.
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
            assert isinstance(prev, (KVSlot, StructuralHeaderSlot))
            if prev.eol.newline is None:
                prev.eol.newline = NewlineNode(nl)
        if isinstance(slot, KVSlot):
            _canon_kv_slot(slot, nl=nl, comments=comments)
        elif isinstance(slot, StructuralHeaderSlot):
            _canon_header_slot(slot, nl=nl, comments=comments)
        if prev is None:
            target: int | None = None
        else:
            target = 1 if isinstance(slot, StructuralHeaderSlot) else 0
        _canon_leading(slot, nl=nl, target_blanks=target, comments=comments)
        prev = slot
        slot = slot._next  # noqa: SLF001


def format_document_trailing(
    trailing: Trivia,
    *,
    nl: str,
    comments: bool,
) -> None:
    """Canonicalise the trailing trivia of a :class:`Document`.

    Retargets newlines, strips blank-line trailing whitespace, strips
    column-indent whitespace before orphan comments, and optionally
    rewrites comment text. Structural shape is preserved, so the
    preamble/epilogue split is unaffected.
    """
    retarget_trivia_newlines(trailing, nl)
    _canon_trivia_text(trailing, comments=comments)


__all__ = [
    "_canon_header_slot",
    "_canon_inline_value",
    "_canon_kv_slot",
    "_canon_leading",
    "format_document_trailing",
    "format_subtree",
    "set_comma_value_multiline",
]
