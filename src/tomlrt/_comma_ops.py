"""Structural mutation primitives for inline-array / inline-table values.

This module owns the operations that splice items into or out of a
:class:`tomlrt._values.CommaValue` while keeping the canonical layout
invariants (per-item trivia ownership, one-row-break-per-row, EOL
section attachment, trailing-comma policy). It is the structural
counterpart to :mod:`tomlrt._format`, which owns the *canonicalisation*
of an existing layout to a chosen shape.

The public surface is intentionally small:

* :func:`splice_in` — append one item.
* :func:`splice_out_tail` — promote a previously-internal
  item to terminal after its successor has been spliced out.
* :func:`splice_out_head` — promote a previously-internal
  item to head after the original head has been spliced out.
* :func:`splice_out` — splice out a set of owned
  indices and run the head / tail / inter-item fix-ups.
* :func:`reorder_owned` — permute a set of owned indices
  in place.

Inline-array (`tomlrt._array.Array`) and inline-table
(`tomlrt._inline_ops`) mutation both drive every structural change
through these helpers, so a future change to the canonical model only
needs to land here.

Supporting primitives — :func:`flip_to_internal` / :func:`flip_to_terminal`
and the EOL section helpers :func:`_take_eol` / :func:`_put_eol` /
:func:`_item_has_eol` / :func:`_normalise_row_breaks` — also live here
and are exported for the few :mod:`tomlrt._format` canonicalisation
paths that need them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

from tomlrt._trivia import (
    NewlineNode,
    Trivia,
    WhitespaceNode,
    indent_from_final_trivia,
    join_above_block,
    restamp_bracket_pad_for_first,
    split_above_block,
    split_eol_section,
    split_item_above,
    strip_trailing_indent,
)
from tomlrt._values import (
    inter_item_separator,
    value_is_multiline,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

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


def _item_has_eol(item: CommaItem) -> bool:
    """True if the item carries an inline EOL comment.

    When the item has a comma, the EOL section lives in
    ``post_comma_trivia``; otherwise it lives in ``trailing``.
    """
    target = item.post_comma_trivia if item.has_comma else item.trailing
    eol, _rest = split_eol_section(target)
    return bool(eol.pieces)


def _take_eol(item: CommaItem) -> Trivia:
    """Split out and return the item's row-attached EOL section.

    The EOL section lives in ``post_comma_trivia`` when the item has
    a comma, and in ``trailing`` otherwise. On return the item holds
    only the structural rest in that channel.
    """
    if item.has_comma:
        eol, rest = split_eol_section(item.post_comma_trivia)
        item.post_comma_trivia = rest
    else:
        eol, rest = split_eol_section(item.trailing)
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


def _normalise_row_breaks(
    items: Sequence[CommaItem],
    value: CommaValue[_CV_ItemT],
    nl: str,
    *,
    multiline: bool,
) -> None:
    """Enforce the one-row-break-per-row invariant.

    In a multi-line inline value each item occupies one physical row;
    each row needs exactly one terminating newline. That newline lives
    either in the row's EOL section (``items[i].post_comma_trivia`` or
    ``items[i].trailing``) — or in the next row's leading region
    (``items[i+1].leading`` for inter-item gaps, ``value.final_trivia``
    for the gap before the closing bracket).

    This helper restores the invariant after a mutation has touched
    item count, comma state, or EOL state — call it once at the end.
    Idempotent.
    """
    if not multiline:
        return
    # Inter-item: items[i].leading must start with NL iff items[i-1]
    # carries no EOL.
    for i in range(1, len(items)):
        pred = items[i - 1]
        pieces = items[i].leading.pieces
        has_nl = bool(pieces) and isinstance(pieces[0], NewlineNode)
        if _item_has_eol(pred) and has_nl:
            items[i].leading = Trivia(pieces[1:])
        elif not _item_has_eol(pred) and not has_nl:
            items[i].leading = Trivia([NewlineNode(text=nl), *pieces])
    # Closing-bracket gap: final_trivia must start with NL iff the
    # last item carries no EOL.
    if not items:
        return
    ft = value.final_trivia
    has_nl = bool(ft.pieces) and isinstance(ft.pieces[0], NewlineNode)
    last_has_eol = _item_has_eol(items[-1])
    if last_has_eol and has_nl:
        ft.pieces = list(ft.pieces[1:])
    elif not last_has_eol and not has_nl:
        ft.pieces = [NewlineNode(text=nl), *ft.pieces]


# ---------------------------------------------------------------------------
# Comma / terminal-state flips
# ---------------------------------------------------------------------------


def flip_to_internal(item: CommaItem) -> None:
    """Make ``item`` look like an internal (non-last) item.

    Under the canonical model the inter-item separator lives in the
    NEXT item's leading; this function only ensures the comma is set
    and carries any EOL comment across the channel flip (trailing →
    post_comma_trivia) so the comma stays immediately after the
    value and the comment after the comma. No-op if the item already
    has a comma.

    Shared between the inline-array append path and the inline-table
    append path, both of which transition a previously-terminal item
    into an internal item when a new item is being appended.
    """
    if item.has_comma:
        return
    eol = _take_eol(item)
    item.has_comma = True
    _put_eol(item, eol)


def flip_to_terminal(item: CommaItem, style: CommaStyle) -> None:
    """Make ``item`` look like the terminal (last) item per style."""
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
    """Inferred layout policy for an inline array or inline table.

    Used by both the inline-array and inline-table append paths to
    decide where the new item goes, how the previous-last is flipped
    to internal, and whether the new last carries a trailing comma.
    """

    is_multiline: bool
    inter_separator: Trivia
    trailing_comma: bool
    trailing_post: Trivia


def detect_style(
    value: ArrayValue | InlineTableValue | None, *, multiline_flag: bool
) -> CommaStyle:
    """Infer a :class:`CommaStyle` for ``value`` (or for a fresh value).

    For a single-item multi-line value we synthesise the inter-item
    separator from the bracket-pad indent (no peer to sample from);
    otherwise the separator is the structural pad portion of
    ``items[1].leading``.
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
    if is_multiline and len(items) < 2:
        nl_text = "\n"
        for p in value.header_trivia.pieces:
            if isinstance(p, NewlineNode):
                nl_text = p.text
                break
        else:
            for p in value.final_trivia.pieces:
                if isinstance(p, NewlineNode):
                    nl_text = p.text
                    break
        indent = _first_indent_after_newline(value.header_trivia)
        if not indent:
            indent = indent_from_final_trivia(value.final_trivia) or "    "
        inter_sep = Trivia([NewlineNode(text=nl_text), WhitespaceNode(text=indent)])
    else:
        inter_sep = inter_item_separator(items)
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


def migrate_bracket_above(bracket: Trivia, separator: Trivia) -> tuple[Trivia, Trivia]:
    """Migrate any above-bracket comment block onto a new item's leading.

    An above-block in ``header_trivia`` / ``final_trivia`` (the
    structural row(s) between the bracket and the next/last item)
    conceptually belongs to the item below it. When a new boundary
    item is being inserted or appended, the block migrates from the
    bracket pad onto that item's leading.

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

    Shared orchestration for inline arrays and inline tables. The
    caller is responsible for constructing ``new_item`` of the
    appropriate concrete type (``ArrayItem`` for an array,
    ``InlineTableEntry`` for an inline table) and for computing the
    appropriate ``style`` (typically via :func:`detect_style`).

    Empty case: reframes the bracket pad so the new item sits on its
    own canonical row, then flips it to terminal-per-style.

    Non-empty case: migrates any above-bracket comment block onto the
    new item's leading, flips the previous-last to internal, appends,
    flips the new item to terminal-per-style, and renormalises the
    row-break invariant.
    """
    items = cv.items
    if not items:
        cv.header_trivia, cv.final_trivia = restamp_bracket_pad_for_first(
            cv.final_trivia
        )
        items.append(new_item)
        flip_to_terminal(new_item, style)
        return
    cv.final_trivia, new_item.leading = migrate_bracket_above(
        cv.final_trivia, style.inter_separator
    )
    flip_to_internal(items[-1])
    items.append(new_item)
    flip_to_terminal(new_item, style)
    _normalise_row_breaks(items, cv, nl, multiline=style.is_multiline)


def splice_out_tail(
    cv: CommaValue[_CV_ItemT],
    nl: str,
    *,
    removed_had_comma: bool,
    is_multiline: bool,
) -> None:
    """Re-terminalise the new last item after a tail removal.

    Both inline-array tail-delete and inline-table tail-delete face
    the same transition: the previous tail has been spliced out, and
    the item that was internal is now the new tail.

    Steps (mirroring :func:`splice_in` in reverse):
      * take the entry-attached EOL section off the new tail (it
        currently lives in ``post_comma_trivia`` because the entry
        was internal);
      * adopt the removed entry's trailing-comma policy on the new
        tail and clear its now-stale ``post_comma_trivia``;
      * put the EOL section back, routed to ``trailing`` or
        ``post_comma_trivia`` per the new ``has_comma`` state;
      * renormalise the row-break invariant.

    No-ops when ``cv.items`` is empty (the caller's
    ``strip_trailing_indent`` or equivalent handles the no-item
    canonical form).
    """
    items = cv.items
    if not items:
        return
    new_last = items[-1]
    eol = _take_eol(new_last)
    new_last.has_comma = removed_had_comma
    new_last.post_comma_trivia = Trivia()
    _put_eol(new_last, eol)
    _normalise_row_breaks(items, cv, nl, multiline=is_multiline)


def splice_out_head(
    cv: CommaValue[_CV_ItemT],
    new_first_above: Trivia,
) -> None:
    """Migrate the above-block of the new first item into ``header_trivia``.

    After deleting the original head item(s), the item that is now
    ``items[0]`` was previously internal; its ``leading`` still
    carries the inter-item separator plus any above-item comment
    block that conceptually belonged above it. Under the canonical
    model ``items[0].leading`` is always empty and the above-block
    above the first item lives in ``header_trivia``. This helper
    completes the migration: the caller has already extracted the
    above-block (before deletion, via :func:`split_item_above` on
    the soon-to-be-new-first item's leading); we splice it into
    ``header_trivia`` and reset ``items[0].leading``.
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

    Shared orchestration for inline-array tail/head/internal removal
    (``Array.__delitem__``) and inline-table key / dotted-prefix
    removal (``_inline_ops.delete_entry``). The caller is responsible
    for the kind-specific logic that *finds* the indices to remove
    (slice-vs-int normalisation for arrays, key / prefix lookup for
    inline tables) and for any associated logical-view bookkeeping
    (decoded ``list`` / ``dict`` removal).

    Steps:
      * snapshot the above-block of the soon-to-be-new-first item
        when position 0 is among ``removed_indices`` (so the
        canonical "above-block of item 0 lives in header_trivia"
        invariant can be restored after the splice);
      * capture the about-to-be-removed tail's ``has_comma`` policy
        (so the new tail can adopt it);
      * pop the removed indices from ``cv.items`` in reverse order;
      * if no items remain, call :func:`strip_trailing_indent` to
        canonicalise the empty bracket pad (unless
        ``strip_when_empty`` is False — used by callers that own a
        different empty-canonicalisation policy);
      * else dispatch to :func:`splice_out_head` /
        :func:`splice_out_tail` /
        :func:`_normalise_row_breaks` as appropriate.

    ``removed_indices`` must be a list of valid distinct indices into
    ``cv.items`` (not necessarily sorted on input; sorted internally).
    """
    if not removed_indices:
        return
    items = cv.items
    if not items:
        return
    sorted_removed = sorted(removed_indices)
    removed_set = set(sorted_removed)
    last_idx = len(items) - 1
    zero_removed = 0 in removed_set
    tail_removed = last_idx in removed_set

    new_first_above: Trivia = Trivia()
    if zero_removed:
        for k in range(1, len(items)):
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
        splice_out_tail(
            cv,
            nl,
            removed_had_comma=new_terminal_has_comma,
            is_multiline=is_multiline,
        )
    else:
        _normalise_row_breaks(items, cv, nl, multiline=is_multiline)


def reorder_owned(
    cv: CommaValue[_CV_ItemT],
    owned_positions: Sequence[int],
    new_owned: Sequence[_CV_ItemT],
    nl: str,
    *,
    is_multiline: bool,
) -> None:
    """Reorder ``owned_positions`` within ``cv.items`` in place.

    Positional-preserve strategy shared by inline arrays (where every
    position is owned) and inline tables (where the caller may be a
    dotted-inner navigator with foreign entries interspersed). The
    split between positional state (stays at the position) and
    entry-attached state (travels with the entry) is:

      * **Positional**: structural pad (the part of ``leading`` —
        or ``header_trivia`` for position 0 — before any above-item
        comment block), ``has_comma``, ``post_comma_trivia``,
        ``trailing``.
      * **Entry-attached**: above-item comment block, row-attached
        EOL section.

    ``owned_positions`` is a strictly ascending list of indices into
    ``cv.items`` that are subject to reordering; non-owned positions
    are left untouched. ``new_owned`` is the permutation: the entry
    to place at each owned position, in the same order. Both lists
    must have equal length, and ``new_owned`` must be a permutation
    of ``cv.items[i]`` for ``i in owned_positions``.

    Bracket trivia (``header_trivia`` / ``final_trivia``) is
    untouched, so above-bracket comment blocks survive the reorder
    at their structural position. No-op for fewer than two owned
    positions.
    """
    if len(owned_positions) <= 1:
        return
    assert len(owned_positions) == len(new_owned)
    items = cv.items
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
    # Restore the one-row-break-per-row invariant: an EOL section
    # ends in its own ``\n``, which would otherwise double up with
    # the structural ``\n`` on the next position's pad.
    _normalise_row_breaks(items, cv, nl, multiline=is_multiline)


__all__ = [
    "CommaStyle",
    "detect_style",
    "flip_to_internal",
    "flip_to_terminal",
    "migrate_bracket_above",
    "reorder_owned",
    "splice_in",
    "splice_out",
    "splice_out_head",
    "splice_out_tail",
]
