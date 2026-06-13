"""Mutate inline-array / inline-table item lists structurally.

All structural changes to a :class:`tomlrt._values.CommaValue` pass
through this module so the canonical model stays central: per-item
trivia ownership, ``header_trivia`` / ``final_trivia`` bracket-pad
attachment, one-row-break-per-row, EOL section placement, and
trailing-comma policy. :mod:`tomlrt._format` is the counterpart that
canonicalises an existing layout without changing structure.

The small public surface covers append, removal, reorder, comma-state
flips, and EOL-section helpers for arrays and inline tables.

Inline-array and inline-table mutation both drive structural changes
through these helpers, so a future change to the comma-value model only
needs to land here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

from tomlrt._trivia import (
    NewlineNode,
    Trivia,
    WhitespaceNode,
    indent_from_final_trivia,
    join_above_block,
    leading_break_index,
    restamp_bracket_pad_for_first,
    split_above_block,
    split_eol_section,
    split_item_above,
    strip_trailing_indent,
    trivia_has_newline,
)
from tomlrt._values import (
    inter_item_separator,
    item_eol_channel,
    value_is_multiline,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tomlrt._trivia import TriviaPiece
    from tomlrt._values import (
        ArrayValue,
        CommaItem,
        CommaValue,
        InlineTableValue,
    )


_CV_ItemT = TypeVar("_CV_ItemT", bound="CommaItem")


# ---------------------------------------------------------------------------
# Per-item EOL section helpers
# ---------------------------------------------------------------------------


def _row_terminated(item: CommaItem) -> bool:
    """Return whether the item's own row carries its terminating newline.

    Otherwise the break is downstream, in the next item's leading (or
    ``final_trivia`` for the tail). Comma placement never enters into it.
    """
    return trivia_has_newline(item_eol_channel(item))


def _take_eol(item: CommaItem) -> Trivia:
    """Split out and return the item's row-attached EOL section.

    The item keeps only the structural rest in that channel.
    """
    channel = item_eol_channel(item)
    eol, rest = split_eol_section(channel)
    if channel is item.post_comma_trivia:
        item.post_comma_trivia = rest
    else:
        item.trailing = rest
    return eol


def _put_eol(item: CommaItem, eol: Trivia) -> None:
    """Append a previously-taken EOL section onto the item.

    Routes to ``post_comma_trivia`` or ``trailing`` according to the
    item's *current* ``has_comma``, which may differ from the value
    at extraction time.
    """
    if not eol.pieces:
        return
    if item.has_comma:
        item.post_comma_trivia = Trivia([*item.post_comma_trivia.pieces, *eol.pieces])
    else:
        item.trailing = Trivia([*item.trailing.pieces, *eol.pieces])


# ---------------------------------------------------------------------------
# Row-break maintenance
# ---------------------------------------------------------------------------
#
# In a multi-line inline value each item occupies one physical row, and
# each row needs exactly one terminating newline ("row break"). That break
# lives either in the row's own EOL channel (see :func:`_row_terminated`) or
# downstream in the next row's ``leading`` (``final_trivia`` for the tail).
# Any *further* newline at a boundary is a deliberate user blank line.
#
# A mutation can only invalidate the boundaries it actually touches, so the
# helpers below operate per-boundary rather than re-scanning the whole value
# (which would clobber blank lines the edit never went near). Two shapes of
# touched boundary arise:
#
# * a *fresh* boundary, where the mutation introduced a brand-new item — its
#   ``leading`` is a synthesised separator carrying exactly the structural
#   break and no blanks, so a redundant break is just stripped when the
#   predecessor already terminates its own row (:func:`_structural_break`);
# * a *carried* boundary, where an existing downstream region is reused under
#   a predecessor whose termination changed — the rendered blank-line count
#   must survive, so the structural break shifts by ``+1`` / ``-1``
#   (:func:`_shift_carried_boundary`), leaving the blanks intact.


def _structural_break(pred: CommaItem, succ: CommaItem, nl: str) -> None:
    """Strip the redundant downstream break from a fresh boundary.

    A fresh ``succ.leading`` is a synthesised separator carrying exactly
    one newline — its structural row break. When ``pred`` already
    terminates its own row that newline is redundant and is dropped;
    otherwise it is the row break and stays. This is the carried-boundary
    shift of a zero-blank baseline: ``delta == -1`` iff ``pred`` is
    terminated.
    """
    succ.leading = Trivia(
        shift_pieces(succ.leading.pieces, -int(_row_terminated(pred)), nl)
    )


def boundary_break_holder(cv: CommaValue[_CV_ItemT], b: int) -> Trivia:
    """Return the trivia that owns the row break *downstream* of boundary ``b``.

    Boundary ``b`` sits between items ``b-1`` and ``b``; the break for the
    ``b-1`` row, when not carried in that item's own EOL channel, lives in
    ``items[b].leading`` or — for the final boundary (``b == len(items)``) —
    in ``cv.final_trivia``.
    """
    items = cv.items
    return items[b].leading if b < len(items) else cv.final_trivia


def shift_pieces(
    pieces: Sequence[TriviaPiece], delta: int, nl: str
) -> list[TriviaPiece]:
    """Add or drop ``|delta|`` leading newlines, preserving the remainder.

    ``delta > 0`` prepends structural newlines (pushing any existing
    newlines down into blank lines); ``delta < 0`` drops that many leading
    newlines (each skipping an intervening whitespace run). The indent and
    any blank lines below the shifted region are untouched.
    """
    if delta > 0:
        return [*(NewlineNode(text=nl) for _ in range(delta)), *pieces]
    out = list(pieces)
    for _ in range(-delta):
        k = leading_break_index(out)
        if k is None:
            break
        out = out[k + 1 :]
    return out


def _shift_carried_boundary(
    cv: CommaValue[_CV_ItemT], b: int, nl: str, *, old_pred_terminated: bool
) -> None:
    """Shift the structural break at carried boundary ``b``.

    Boundary ``b`` sits between items ``b-1`` and ``b`` (or before the
    closing bracket when ``b == len(items)``). Its predecessor's
    termination has changed from ``old_pred_terminated`` to its current
    value; shift the break by that difference so the rendered blank-line
    count is preserved.
    """
    delta = int(old_pred_terminated) - int(_row_terminated(cv.items[b - 1]))
    if not delta:
        return
    holder = boundary_break_holder(cv, b)
    holder.pieces = shift_pieces(holder.pieces, delta, nl)


# ---------------------------------------------------------------------------
# Comma / terminal-state flips
# ---------------------------------------------------------------------------


def flip_to_internal(item: CommaItem) -> None:
    """Make ``item`` look like an internal (non-last) item.

    The inter-item separator belongs to the next item's leading; this
    only sets comma state and carries any EOL comment across the
    ``trailing`` → ``post_comma_trivia`` channel flip so the comma renders
    before the comment. No-op if the item already has a comma.
    """
    if item.has_comma:
        return
    eol = _take_eol(item)
    item.has_comma = True
    _put_eol(item, eol)


def flip_to_terminal(item: CommaItem, style: CommaStyle) -> None:
    """Make ``item`` look like the terminal item for ``style``.

    Trailing-comma style keeps existing ``post_comma_trivia`` intact;
    no-trailing style moves any EOL section back to ``trailing``.
    """
    if style.trailing_comma:
        if not item.has_comma:
            eol = _take_eol(item)
            item.has_comma = True
            _put_eol(item, eol)
        # When has_comma==True, post_comma_trivia carries any EOL the
        # parser/mutation already filed there; keep it intact.
        return
    # No trailing comma policy: drop the comma; carry any EOL back
    # to trailing.
    if item.has_comma:
        eol = _take_eol(item)
        item.has_comma = False
        item.post_comma_trivia = Trivia()
        _put_eol(item, eol)


# ---------------------------------------------------------------------------
# Style detection
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class CommaStyle:
    """Hold inferred layout policy for inline array/table append paths.

    Carries the inter-item separator, whether the value is multi-line,
    and whether the terminal item should keep a trailing comma.
    """

    is_multiline: bool
    inter_separator: Trivia
    trailing_comma: bool
    trailing_post: Trivia


def detect_style(
    value: ArrayValue | InlineTableValue | None, *, multiline_flag: bool
) -> CommaStyle:
    """Infer a :class:`CommaStyle` for ``value`` (or for a fresh value).

    The inter-item separator is sampled from ``items[1].leading``. A
    multi-line value that cannot sample one (single item, or a comma-first
    peer with empty leading) falls back to :func:`_canonical_separator`.
    """
    if value is None:
        return CommaStyle(
            is_multiline=multiline_flag,
            inter_separator=Trivia([WhitespaceNode(text=" ")]),
            trailing_comma=multiline_flag,
            trailing_post=Trivia(),
        )
    items = value.items
    is_multiline = multiline_flag or value_is_multiline(value)
    inter_sep = inter_item_separator(items)
    if is_multiline and not trivia_has_newline(inter_sep):
        inter_sep = _canonical_separator(value)
    trailing_comma = items[-1].has_comma if items else is_multiline
    pad_ft, _above_ft = split_above_block(value.final_trivia)
    trailing_post = pad_ft if pad_ft.pieces else value.final_trivia.copy()
    if not items and is_multiline and not trailing_post.pieces:
        nl_text = "\n"
        trailing_post = Trivia([NewlineNode(text=nl_text)])
    return CommaStyle(
        is_multiline=is_multiline,
        inter_separator=inter_sep,
        trailing_comma=trailing_comma,
        trailing_post=trailing_post,
    )


def _first_indent_after_newline(trivia: Trivia) -> str:
    pieces = trivia.pieces
    for i, p in enumerate(pieces):
        if (
            isinstance(p, NewlineNode)
            and i + 1 < len(pieces)
            and isinstance(pieces[i + 1], WhitespaceNode)
        ):
            return str(pieces[i + 1].text)
    return ""


def _value_newline(value: CommaValue[Any]) -> str:
    """Return the newline text sampled from ``value`` bracket pads."""
    for trivia in (value.header_trivia, value.final_trivia):
        for p in trivia.pieces:
            if isinstance(p, NewlineNode):
                return str(p.text)
    return "\n"


def _canonical_separator(value: CommaValue[Any]) -> Trivia:
    """Return the fallback inter-item newline plus value indent."""
    indent = (
        _first_indent_after_newline(value.header_trivia)
        or indent_from_final_trivia(value.final_trivia)
        or "    "
    )
    nl = NewlineNode(text=_value_newline(value))
    return Trivia([nl, WhitespaceNode(text=indent)])


def migrate_bracket_above(bracket: Trivia, separator: Trivia) -> tuple[Trivia, Trivia]:
    """Migrate any above-bracket comment block onto a new item's leading.

    An above-block in ``header_trivia`` / ``final_trivia`` conceptually
    belongs to the item below it. Inserting a boundary item moves that
    block from the bracket pad onto the item's leading.

    Returns ``(new_bracket, new_leading)``.
    """
    pad, above = split_above_block(bracket)
    return pad, join_above_block(separator, above)


# ---------------------------------------------------------------------------
# Append / remove / reorder orchestration
# ---------------------------------------------------------------------------


def splice_in(
    cv: CommaValue[_CV_ItemT],
    new_item: _CV_ItemT,
    style: CommaStyle,
    nl: str,
) -> None:
    """Append ``new_item`` to ``cv`` honouring ``style``.

    The caller supplies the concrete item (``ArrayItem`` or
    ``InlineTableEntry``) and style. Empty values reframe bracket pads for
    the first item, preserving canonical single-line inner pad when no
    authored pad survives. Non-empty values migrate above-bracket comments
    to the new item's leading, flip the old tail internal, make the new
    item terminal, and restore row-break invariants.
    """
    items = cv.items
    if not items:
        header, final = restamp_bracket_pad_for_first(cv.final_trivia)
        if cv._single_line_pad and not header.pieces and not final.pieces:  # noqa: SLF001
            header = Trivia([WhitespaceNode(text=cv._single_line_pad)])  # noqa: SLF001
            final = Trivia([WhitespaceNode(text=cv._single_line_pad)])  # noqa: SLF001
        cv.header_trivia, cv.final_trivia = header, final
        items.append(new_item)
        flip_to_terminal(new_item, style)
        return
    cv.final_trivia, new_item.leading = migrate_bracket_above(
        cv.final_trivia, style.inter_separator
    )
    old_tail = items[-1]
    flip_to_internal(old_tail)
    items.append(new_item)
    flip_to_terminal(new_item, style)
    if style.is_multiline:
        # Fresh boundary onto the new item; carried final boundary, whose
        # predecessor changes from the old tail to the new item.
        _structural_break(old_tail, new_item, nl)
        _shift_carried_boundary(
            cv, len(items), nl, old_pred_terminated=_row_terminated(old_tail)
        )


def splice_insert(
    cv: CommaValue[_CV_ItemT],
    new_item: _CV_ItemT,
    index: int,
    style: CommaStyle,
    nl: str,
) -> None:
    """Insert ``new_item`` at ``index`` (``0 <= index < len(items)``).

    Appending is :func:`splice_in`; this covers interior and head inserts.
    A head insert migrates the opening-bracket above-block onto the
    displaced item, keeping item-0 leading empty. The fresh boundary onto
    ``new_item`` gets just its structural break; the carried boundary onto
    the displaced item keeps its blank lines while its predecessor changes
    to ``new_item``.
    """
    items = cv.items
    if index == 0:
        cv.header_trivia, items[0].leading = migrate_bracket_above(
            cv.header_trivia, style.inter_separator
        )
        items.insert(0, new_item)
        if style.is_multiline:
            _structural_break(new_item, items[1], nl)
        return
    pred = items[index - 1]
    pred_terminated = _row_terminated(pred)
    new_item.leading = style.inter_separator.copy()
    items.insert(index, new_item)
    if style.is_multiline:
        _structural_break(pred, new_item, nl)
        _shift_carried_boundary(cv, index + 1, nl, old_pred_terminated=pred_terminated)


def _reterminalise_tail(new_last: CommaItem, *, removed_had_comma: bool) -> None:
    """Re-terminalise the new last item after a tail removal.

    The former internal tail adopts the removed tail's comma policy while
    its row-attached EOL section survives the
    ``post_comma_trivia``/``trailing`` channel flip. This mirrors
    ``splice_in`` in reverse: the item was internal, so its EOL section may
    currently live after its comma; terminalising may move that section
    back before the (now absent) comma.
    """
    eol = _take_eol(new_last)
    new_last.has_comma = removed_had_comma
    new_last.post_comma_trivia = Trivia()
    _put_eol(new_last, eol)


def splice_out_head(
    cv: CommaValue[_CV_ItemT],
    new_first_above: Trivia,
) -> None:
    """Migrate the above-block of the new first item into ``header_trivia``.

    After head deletion, new item 0 was internal and still has separator
    plus above-item comments in leading. The canonical model keeps item 0
    leading empty and stores the above-block in ``header_trivia``; caller
    passes the pre-extracted block.
    """
    if not cv.items:
        return
    head_pad, _drop = split_above_block(cv.header_trivia)
    cv.header_trivia = join_above_block(head_pad, new_first_above)
    cv.items[0].leading = Trivia()


def splice_out(
    cv: CommaValue[_CV_ItemT],
    removed_indices: Sequence[int],
    nl: str,
    *,
    is_multiline: bool,
    strip_when_empty: bool = True,
) -> None:
    """Splice out ``removed_indices`` and run the boundary fixups.

    Shared removal orchestration for inline arrays and inline-table
    entries; callers own index discovery (slice/int normalisation for
    arrays, key/prefix lookup for inline tables) and logical-view updates.
    Head removal snapshots the new-first above-block so the "item 0
    above-block lives in ``header_trivia``" invariant is restored. Tail
    removal transfers the removed tail's comma policy to the new tail.
    Empty values use :func:`strip_trailing_indent` unless the caller
    supplies a different policy.

    ``removed_indices`` must be a list of valid distinct indices into
    ``cv.items`` (not necessarily sorted on input; sorted internally).
    """
    if not removed_indices:
        return
    items = cv.items
    if not items:
        return
    orig_len = len(items)
    sorted_removed = sorted(removed_indices)
    removed_set = set(sorted_removed)
    last_idx = orig_len - 1
    zero_removed = 0 in removed_set
    tail_removed = last_idx in removed_set

    # Snapshot before popping: termination per original position, and the
    # original index of each surviving item (so carried boundaries can shift
    # for a predecessor that was removed).
    term_before = [_row_terminated(it) for it in items]
    survivors = [i for i in range(orig_len) if i not in removed_set]

    new_first_above: Trivia = Trivia()
    if zero_removed:
        for k in range(1, orig_len):
            if k not in removed_set:
                _h, new_first_above, _t = split_item_above(items[k].leading)
                break
    new_terminal_has_comma = items[last_idx].has_comma if tail_removed else False

    for i in reversed(sorted_removed):
        items.pop(i)

    if not items:
        if strip_when_empty:
            strip_trailing_indent(cv.header_trivia, cv.final_trivia)
        return
    if zero_removed:
        splice_out_head(cv, new_first_above)
    if tail_removed:
        _reterminalise_tail(items[-1], removed_had_comma=new_terminal_has_comma)
        if is_multiline:
            _shift_carried_boundary(
                cv, len(items), nl, old_pred_terminated=term_before[last_idx]
            )
    if is_multiline:
        # Each remaining seam — a survivor whose original predecessor was
        # removed — is a carried boundary: keep its blank lines while its
        # predecessor changes to the nearest surviving item.
        for j in range(1, len(items)):
            if survivors[j] - survivors[j - 1] > 1:
                _shift_carried_boundary(
                    cv, j, nl, old_pred_terminated=term_before[survivors[j] - 1]
                )


def reorder_owned(
    cv: CommaValue[_CV_ItemT],
    owned_positions: Sequence[int],
    new_owned: Sequence[_CV_ItemT],
    nl: str,
    *,
    is_multiline: bool,
) -> None:
    """Reorder ``owned_positions`` within ``cv.items`` in place.

    Shared positional-preserve strategy for arrays and dotted-inner
    inline-table navigators. State splits as follows:

    * positional: structural pad, comma state, post-comma trivia, trailing
      trivia;
    * entry-attached: above-item comment blocks and row-attached EOL
      sections.

    ``owned_positions`` is strictly ascending and foreign positions stay
    untouched. ``new_owned`` is the matching permutation. Bracket trivia
    (``header_trivia`` / ``final_trivia``) is untouched, so above-bracket
    comments stay at their structural position. No-op for fewer than two
    owned positions.
    """
    if len(owned_positions) <= 1:
        return
    assert len(owned_positions) == len(new_owned)
    items = cv.items
    # Termination per original position, snapshot before _take_eol mutates it.
    term_before = [_row_terminated(it) for it in items]
    pos_state: dict[int, tuple[Trivia, bool, Trivia, Trivia]] = {}
    above_by_entry: dict[int, Trivia] = {}
    eol_by_entry: dict[int, Trivia] = {}
    for i in owned_positions:
        e = items[i]
        src = cv.header_trivia if i == 0 else e.leading
        pad, above = split_above_block(src)
        eol_by_entry[id(e)] = _take_eol(e)
        pos_state[i] = (pad, e.has_comma, e.post_comma_trivia, e.trailing)
        above_by_entry[id(e)] = above

    new_items = list(items)
    for pos, e in zip(owned_positions, new_owned, strict=True):
        new_items[pos] = e
    items[:] = new_items

    for pos in owned_positions:
        e = items[pos]
        pad, has_comma, post_rest, trail_rest = pos_state[pos]
        e.has_comma = has_comma
        e.post_comma_trivia = post_rest
        e.trailing = trail_rest
        _put_eol(e, eol_by_entry[id(e)])
        above = above_by_entry[id(e)]
        if pos == 0:
            cv.header_trivia = join_above_block(pad, above)
            e.leading = Trivia()
        else:
            e.leading = join_above_block(pad, above)
    # Restore the structural break at the boundaries the reorder touched.
    # An owned item carries its EOL section (hence its termination) with it,
    # so a boundary whose predecessor moved sees a predecessor-termination
    # change; the carried-boundary shift keeps its blank lines intact. Only
    # boundaries whose predecessor moved (``p + 1``) can change.
    if is_multiline:
        for b in {p + 1 for p in owned_positions}:
            _shift_carried_boundary(cv, b, nl, old_pred_terminated=term_before[b - 1])


__all__ = [
    "CommaStyle",
    "boundary_break_holder",
    "detect_style",
    "migrate_bracket_above",
    "reorder_owned",
    "shift_pieces",
    "splice_in",
    "splice_insert",
    "splice_out",
]
