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
    indent_from_trivia,
    join_above_block,
    leading_break,
    leading_ws,
    restamp_bracket_pad_for_first,
    split_above_block,
    split_eol_section,
    split_lines,
    strip_trailing_indent,
    trailing_ws,
)
from tomlrt._values import (
    inter_item_separator,
    item_breaks_before_comma,
    item_eol_channel,
    set_item_eol_channel,
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


def _take_eol(item: CommaItem) -> str:
    """Split out and return the item's row-attached EOL section.

    The item keeps only the structural rest in that channel.
    """
    eol, rest = split_eol_section(item_eol_channel(item))
    set_item_eol_channel(item, rest)
    return eol


def _put_eol(item: CommaItem, eol: str) -> None:
    """Put a previously-taken EOL section onto the item.

    Routes to ``post_comma_trivia`` or ``trailing`` according to the
    item's *current* ``has_comma``, which may differ from the value
    at extraction time.
    """
    if not eol:
        return
    if item.has_comma:
        item.post_comma_trivia += eol
    else:
        item.trailing += eol


# ---------------------------------------------------------------------------
# Above-item boundary ownership
# ---------------------------------------------------------------------------


def _first_newline_end(t: str) -> int:
    """Return the offset just after the first newline, or zero."""
    return t.find("\n") + 1


@dataclass(slots=True)
class _Lane:
    """Text partitions of one physical boundary lane."""

    head: str = ""
    above: str = ""
    tail: str = ""

    @classmethod
    def capture(cls, trivia: str, start: int) -> _Lane:
        tail = trailing_ws(trivia[start:])
        end = len(trivia) - len(tail)
        return cls(trivia[:start], trivia[start:end], tail)

    def copy(self) -> _Lane:
        return _Lane(self.head, self.above, self.tail)

    def join(self) -> str:
        return self.head + self.above + self.tail


@dataclass(slots=True)
class Boundary:
    """Lossless snapshot of the complete region before an item or bracket."""

    before: _Lane
    after: str
    following: _Lane
    has_comma: bool = False
    is_head: bool = False

    @classmethod
    def capture(cls, cv: CommaValue[Any], i: int) -> Boundary:
        items = cv.items
        if i == 0:
            following = cv.header_trivia
            start = _first_newline_end(following) or len(following)
            return cls(
                _Lane(),
                "",
                _Lane.capture(following, start),
                is_head=True,
            )
        pred = items[i - 1]
        following = cv.final_trivia if i == len(items) else items[i].leading
        before_break = pred.has_comma and "\n" in pred.trailing
        closed_lane = pred.post_comma_trivia if pred.has_comma else pred.trailing
        row_closed = "\n" in closed_lane
        before_start = (
            _first_newline_end(pred.trailing) if before_break else len(pred.trailing)
        )
        following_start = 0 if row_closed else leading_break(following)
        return cls(
            _Lane.capture(pred.trailing, before_start),
            pred.post_comma_trivia,
            _Lane.capture(following, following_start),
            has_comma=pred.has_comma,
        )

    def copy(self) -> Boundary:
        return Boundary(
            self.before.copy(),
            self.after,
            self.following.copy(),
            has_comma=self.has_comma,
            is_head=self.is_head,
        )

    def restore(self, cv: CommaValue[Any], i: int) -> None:
        following = self.following.join()
        if self.is_head:
            cv.header_trivia = following
            cv.items[0].leading = ""
            return
        pred = cv.items[i - 1]
        pred.trailing = self.before.join()
        pred.has_comma = self.has_comma
        pred.post_comma_trivia = self.after
        if i == len(cv.items):
            cv.final_trivia = following
        else:
            cv.items[i].leading = following

    @property
    def break_before_comma(self) -> bool:
        return self.has_comma and "\n" in self.before.head

    @property
    def row_closed(self) -> bool:
        lane = self.after if self.has_comma else self.before.head
        return "\n" in lane

    @property
    def following_break_is_structural(self) -> bool:
        return (
            not self.is_head
            and not self.row_closed
            and bool(leading_break(self.following.head))
        )

    def _eol(self) -> tuple[int | None, str]:
        before = self.before.join()
        if self.break_before_comma:
            eol, _rest = split_eol_section(before)
            if eol or "#" not in self.after:
                return (0 if eol else None), eol
        channel = self.after if self.has_comma else before
        eol, _rest = split_eol_section(channel)
        return (1 if self.has_comma else 0) if eol else None, eol

    @property
    def eol(self) -> str:
        """Left-owned row-attached EOL payload."""
        return self._eol()[1]

    @property
    def eol_lane(self) -> int | None:
        """Physical pre/post-comma lane containing ``eol``."""
        return self._eol()[0]

    @property
    def above_parts(self) -> tuple[str, str]:
        return self.before.above, self.following.above

    @property
    def above(self) -> str:
        return self.before.above + self.following.above

    def target_lane(self) -> int:
        return int(
            self.is_head
            or not self.break_before_comma
            or self.row_closed
            or bool(self.following.head)
            or bool(self.following.above)
        )

    @property
    def attached_lane(self) -> int | None:
        if self.following.above:
            return 1
        if not self.before.above:
            return None
        if self.row_closed or self.following.head:
            return None
        return 0

    @property
    def attached_above(self) -> str | None:
        lane = self.attached_lane
        return None if lane is None else self.above_parts[lane]

    @property
    def target_tail(self) -> str:
        tails = self.before.tail, self.following.tail
        return tails[self.target_lane()]

    @property
    def target_above(self) -> str:
        return self.above_parts[self.target_lane()]

    def replace_lane(self, lane: int, block: str, nl: str, indent: str) -> None:
        lanes = self.before, self.following
        target = lanes[lane]
        head = target.head
        tail = target.tail
        upstream = (
            self.break_before_comma
            if lane == 0
            else (False if self.is_head else self.row_closed)
        )
        if block:
            if not upstream and "\n" not in head:
                head += nl
            if not tail:
                tail = indent
        target.head = head
        target.above = block
        target.tail = tail

    def remove_above(self) -> Boundary:
        self.before.above = ""
        self.following.above = ""
        return self

    def remove_eol(self) -> Boundary:
        lane = self.eol_lane
        break_before_comma = self.break_before_comma
        if lane == 0:
            eol, rest = split_eol_section(self.before.head)
            self.before.head = rest
            if break_before_comma and eol.endswith("\n"):
                nl_text = "\r\n" if eol.endswith("\r\n") else "\n"
                self.before.head = nl_text + self.before.head
        elif lane == 1:
            _eol, rest = split_eol_section(self.after)
            self.after = rest
        return self

    def set_above(self, block: str, nl: str, indent: str) -> Boundary:
        before, following = self.above_parts
        target_lane = self.target_lane()
        self.remove_above()
        if before and following:
            lines = split_lines(block)
            cut = len(split_lines(before))
            self.replace_lane(0, "".join(lines[:cut]), nl, indent)
            self.replace_lane(1, "".join(lines[cut:]), nl, indent)
        else:
            lane = 0 if before else 1 if following else target_lane
            self.replace_lane(lane, block, nl, indent)
        return self

    def set_attached(self, block: str, nl: str, indent: str) -> Boundary:
        lane = self.attached_lane
        target = self.target_lane() if lane is None else lane
        self.replace_lane(target, block, nl, indent)
        return self

    def carry_above_from(
        self,
        source: Boundary,
        nl: str,
        indent: str,
        *,
        preserve_positional: bool = True,
    ) -> Boundary:
        row_closed = self.row_closed
        break_before_comma = self.break_before_comma
        following_head = bool(self.following.head)
        target_lane = self.target_lane()
        if not (preserve_positional and "#" not in self.above):
            self.remove_above()
        if preserve_positional and "#" not in source.above:
            return self
        before, following = source.above_parts

        def append_to(lane: int, block: str) -> None:
            self.replace_lane(lane, self.above_parts[lane] + block, nl, indent)

        before_stays_attached = source.attached_lane != 0 or (
            not row_closed and not following_head
        )
        if before and break_before_comma and before_stays_attached:
            append_to(0, before)
            append_to(1, following)
        else:
            append_to(target_lane, before + following)
        return self

    def put_eol_from(self, source: Boundary, positional: Boundary) -> Boundary:
        """Put ``source``'s EOL into this positional shell."""
        source_eol = source.eol
        if not source_eol:
            return self
        lane = positional.eol_lane
        if lane is None:
            lane = int(
                self.has_comma
                and not (source.eol_lane == 0 and positional.break_before_comma)
            )
        current = self.before.head if lane == 0 else self.after
        target = source_eol + current[leading_break(current) :]
        if lane == 0:
            self.before.head = target
        else:
            self.after = target
        return self

    def shift_carried_from(
        self,
        old: Boundary,
        nl: str,
        indent: str,
        *,
        is_terminal: bool,
    ) -> None:
        """Repair this boundary's structural break after its left item moves."""
        row_closed = self.row_closed
        break_before_comma = self.break_before_comma
        current_structural = (
            False if row_closed else old.row_closed or old.following_break_is_structural
        )
        delta = int(current_structural) - int(old.following_break_is_structural)
        facts_changed = (
            row_closed != old.row_closed or break_before_comma != old.break_before_comma
        )
        if not delta and not facts_changed:
            return
        holder = self.following.join()
        if delta:
            holder = shift_breaks(holder, delta, nl)
        if (
            not is_terminal
            and (row_closed or break_before_comma)
            and not leading_break(holder)
        ):
            holder = (
                reindent_as_leader(holder, indent)
                if row_closed
                else holder.lstrip(" \t")
            )
        following_start = 0 if row_closed else leading_break(holder)
        self.following = _Lane.capture(holder, following_start)


# ---------------------------------------------------------------------------
# Row-break maintenance
# ---------------------------------------------------------------------------
#
# Each multi-line boundary has one structural row break; any further break
# is authored whitespace. Fresh boundaries can drop a redundant separator
# directly, while carried boundaries shift only that structural break.


def _structural_break(pred: CommaItem, succ: CommaItem, nl: str) -> None:
    """Drop a fresh separator when ``pred`` already closes its row."""
    succ.leading = shift_breaks(succ.leading, -int("\n" in item_eol_channel(pred)), nl)


def boundary_break_holder(cv: CommaValue[_CV_ItemT], b: int) -> str:
    """Return boundary ``b``'s downstream row-break owner."""
    items = cv.items
    return items[b].leading if b < len(items) else cv.final_trivia


def set_boundary_break_holder(cv: CommaValue[_CV_ItemT], b: int, text: str) -> None:
    """Write back the run that :func:`boundary_break_holder` reads."""
    items = cv.items
    if b < len(items):
        items[b].leading = text
    else:
        cv.final_trivia = text


def shift_breaks(t: str, delta: int, nl: str) -> str:
    """Shift leading structural newlines without disturbing later trivia."""
    if delta > 0:
        return nl * delta + t
    for _ in range(-delta):
        k = leading_break(t)
        assert k
        t = t[k:]
    return t


def _shift_carried_boundary(
    cv: CommaValue[_CV_ItemT],
    b: int,
    nl: str,
    *,
    old: Boundary,
) -> None:
    """Repair a carried boundary after its predecessor changes."""
    current = Boundary.capture(cv, b)
    current.shift_carried_from(
        old,
        nl,
        _value_indent(cv),
        is_terminal=b >= len(cv.items),
    )
    current.restore(cv, b)


def reindent_as_leader(holder: str, indent: str) -> str:
    """Promote a shared-row follower to a row leader at ``indent``."""
    return indent + holder.lstrip(" \t")


# ---------------------------------------------------------------------------
# Comma / terminal-state flips
# ---------------------------------------------------------------------------


def flip_to_internal(item: CommaItem) -> None:
    """Add an internal comma while keeping the item's EOL attached."""
    if item.has_comma:
        return
    eol = _take_eol(item)
    item.has_comma = True
    _put_eol(item, eol)


def flip_to_terminal(item: CommaItem, style: CommaStyle) -> None:
    """Apply terminal comma policy while keeping the item's EOL attached."""
    if item.has_comma == style.trailing_comma:
        return
    eol = _take_eol(item)
    item.has_comma = style.trailing_comma
    _put_eol(item, eol)


# ---------------------------------------------------------------------------
# Style detection
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class CommaStyle:
    """Hold inferred layout policy for inline array/table append paths.

    Carries the inter-item separator, whether the value is multi-line, and
    whether the terminal item should keep a trailing comma. A non-empty
    ``pre_comma_break`` marks a *break-before-comma* (comma-first) value:
    one that parks each row break in the item's own ``trailing`` ahead of
    its comma rather than downstream in the next item's ``leading``, so
    ``inter_separator`` is just the post-comma pad.
    """

    is_multiline: bool
    inter_separator: str
    trailing_comma: bool
    trailing_post: str
    pre_comma_break: str

    @property
    def break_before_comma(self) -> bool:
        return bool(self.pre_comma_break)


def _pre_comma_break(item: CommaItem) -> str:
    """The row break a break-before-comma ``item`` parks before its comma.

    Sampled so a new internal item matches the authored newline and the
    comma-row indent (the trailing whitespace between break and comma).
    """
    t = item.trailing
    i = t.find("\n")
    return ("\r\n" if t[i - 1 : i] == "\r" else "\n") + trailing_ws(t)


def detect_style(value: ArrayValue | InlineTableValue, *, nl: str) -> CommaStyle:
    """Infer a :class:`CommaStyle` for ``value``.

    Multi-line shape is read from the value's own trivia
    (:meth:`CommaValue.is_multiline`) — the value is the single source of
    truth, so there is no separate "force multi-line" flag. The inter-item
    separator is sampled from ``items[1].leading``; a comma-last multi-line
    value that cannot sample one (single item) falls back to
    :func:`_canonical_separator`. When item 0 parks its break before its comma
    the value is comma-first: the post-comma pad is kept and ``pre_comma_break``
    seeded from that item. ``nl`` is the owning document's newline.
    """
    items = value.items
    is_multiline = value.is_multiline()
    inter_sep = inter_item_separator(items)
    leader = items[0] if items and item_breaks_before_comma(items[0]) else None
    if is_multiline and leader is None and "\n" not in inter_sep:
        inter_sep = _canonical_separator(value, nl)
    trailing_comma = items[-1].has_comma if items else is_multiline
    pad_ft, _above_ft = split_above_block(value.final_trivia)
    trailing_post = pad_ft or value.final_trivia
    return CommaStyle(
        is_multiline=is_multiline,
        inter_separator=inter_sep,
        trailing_comma=trailing_comma,
        trailing_post=trailing_post,
        pre_comma_break=_pre_comma_break(leader) if leader else "",
    )


def _first_indent_after_newline(trivia: str) -> str:
    """Indent of the first line after a newline that carries one."""
    for line in trivia.split("\n")[1:]:
        if ws := leading_ws(line):
            return ws
    return ""


def _value_newline(value: CommaValue[Any], nl: str) -> str:
    """Return the newline text sampled from ``value`` bracket pads, else ``nl``.

    ``nl`` is the owning document's newline: the fallback for a value whose
    bracket pads carry no break to sample (e.g. a multi-line array whose only
    newline lives in an item's EOL section), so a synthesised break matches
    the document instead of defaulting to LF.
    """
    for trivia in (value.header_trivia, value.final_trivia):
        i = trivia.find("\n")
        if i != -1:
            return "\r\n" if trivia[i - 1 : i] == "\r" else "\n"
    return nl


def _value_indent(value: CommaValue[Any]) -> str:
    """Return the row indent sampled from ``value`` (4 spaces if none)."""
    return (
        _first_indent_after_newline(value.header_trivia)
        or indent_from_trivia(value.final_trivia)
        or "    "
    )


def _canonical_separator(value: CommaValue[Any], nl: str) -> str:
    """Return the fallback inter-item newline plus value indent."""
    return _value_newline(value, nl) + _value_indent(value)


def migrate_bracket_above(bracket: str, separator: str) -> tuple[str, str]:
    """Migrate any above-bracket comment block onto a new item's leading.

    An above-block in ``header_trivia`` / ``final_trivia`` conceptually
    belongs to the item below it. Inserting a boundary item moves that
    block from the bracket pad onto the item's leading.

    Returns ``(new_bracket, new_leading)``.
    """
    pad, above = split_above_block(bracket)
    return pad, join_above_block(separator, above)


def _carry_above(
    cv: CommaValue[Any],
    i: int,
    source: Boundary,
    nl: str,
    indent: str,
    *,
    preserve_positional: bool = True,
) -> None:
    Boundary.capture(cv, i).carry_above_from(
        source,
        nl,
        indent,
        preserve_positional=preserve_positional,
    ).restore(cv, i)


def _replace_above(
    cv: CommaValue[Any], i: int, source: Boundary, nl: str, indent: str = ""
) -> None:
    _carry_above(
        cv,
        i,
        source,
        nl,
        indent,
        preserve_positional=False,
    )


# ---------------------------------------------------------------------------
# Append / remove / reorder orchestration
# ---------------------------------------------------------------------------


def splice_in(
    cv: CommaValue[_CV_ItemT],
    new_item: _CV_ItemT,
    style: CommaStyle,
    nl: str,
) -> None:
    """Append ``new_item`` while preserving the inferred comma style."""
    items = cv.items
    if not items:
        header, final = restamp_bracket_pad_for_first(cv.final_trivia)
        if cv._single_line_pad and not header and not final:  # noqa: SLF001
            header = final = cv._single_line_pad  # noqa: SLF001
        cv.header_trivia, cv.final_trivia = header, final
        items.append(new_item)
        flip_to_terminal(new_item, style)
        return
    cv.final_trivia, new_item.leading = migrate_bracket_above(
        cv.final_trivia, style.inter_separator
    )
    old_tail = items[-1]
    old_final = Boundary.capture(cv, len(items))
    if style.break_before_comma:
        # Comma-first: the former tail keeps its EOL comment but yields its
        # terminal break (re-homed before the closing bracket) and gains its
        # own pre-comma break; the new tail needs no trailing break.
        eol = _take_eol(old_tail).rstrip("\r\n")
        old_tail.trailing = eol + style.pre_comma_break
        old_tail.has_comma = True
        old_tail.post_comma_trivia = ""
        if "\n" not in cv.final_trivia:
            cv.final_trivia = nl + cv.final_trivia
        items.append(new_item)
        flip_to_terminal(new_item, style)
        return
    flip_to_internal(old_tail)
    items.append(new_item)
    flip_to_terminal(new_item, style)
    if style.is_multiline:
        # Fresh boundary onto the new item; carried final boundary, whose
        # predecessor changes from the old tail to the new item.
        _structural_break(old_tail, new_item, nl)
        _shift_carried_boundary(
            cv,
            len(items),
            nl,
            old=old_final,
        )


def splice_insert(
    cv: CommaValue[_CV_ItemT],
    new_item: _CV_ItemT,
    index: int,
    style: CommaStyle,
    nl: str,
) -> None:
    """Insert ``new_item`` before the existing item at ``index``."""
    items = cv.items
    displaced = Boundary.capture(cv, index)
    carry_above = "#" in displaced.above
    if carry_above:
        displaced.copy().remove_above().restore(cv, index)
    if style.break_before_comma:
        new_item.trailing = style.pre_comma_break
        new_item.leading = "" if index == 0 else style.inter_separator
        items.insert(index, new_item)
        if index == 0:
            # Item 0 keeps an empty leading; the displaced item takes the pad.
            items[1].leading = style.inter_separator
        if carry_above:
            # A comma-first item sits on its comma's row, so that row's
            # authored indent -- possibly none -- is the one to match.
            _carry_above(
                cv, index + 1, displaced, nl, trailing_ws(style.pre_comma_break)
            )
        return
    if index == 0:
        cv.header_trivia, items[0].leading = migrate_bracket_above(
            cv.header_trivia, style.inter_separator
        )
        items.insert(0, new_item)
        if style.is_multiline:
            _structural_break(new_item, items[1], nl)
        if carry_above:
            _carry_above(cv, 1, displaced, nl, _value_indent(cv))
        return
    pred = items[index - 1]
    new_item.leading = style.inter_separator
    items.insert(index, new_item)
    if style.is_multiline:
        _structural_break(pred, new_item, nl)
        _shift_carried_boundary(
            cv,
            index + 1,
            nl,
            old=displaced,
        )
    if carry_above:
        _carry_above(cv, index + 1, displaced, nl, _value_indent(cv))


def splice_out(
    cv: CommaValue[_CV_ItemT],
    removed_indices: Sequence[int],
    nl: str,
    *,
    is_multiline: bool,
) -> None:
    """Remove valid distinct item indices and repair the surviving seams."""
    assert removed_indices
    items = cv.items
    assert items
    # A single-line value cannot gain a newline by removal, so only a
    # currently multi-line value can flip (the removed item may hold the
    # sole newline, or emptying may collapse the pads).
    if is_multiline:
        cv.reset_multiline_cache()
    orig_len = len(items)
    sorted_removed = sorted(removed_indices)
    removed_set = set(sorted_removed)
    last_idx = orig_len - 1
    zero_removed = 0 in removed_set
    tail_removed = last_idx in removed_set

    survivors = [i for i in range(orig_len) if i not in removed_set]
    seams = [j for j in range(1, len(survivors)) if survivors[j] - survivors[j - 1] > 1]
    needed = {survivors[j] for j in seams}
    if survivors and zero_removed:
        needed.add(survivors[0])
    if survivors and tail_removed:
        needed.update((survivors[-1] + 1, orig_len))
    boundaries_before = {b: Boundary.capture(cv, b) for b in needed}
    new_last_eol = ""
    if survivors and tail_removed:
        left_boundary = survivors[-1] + 1
        boundaries_before[left_boundary].copy().remove_above().restore(
            cv, left_boundary
        )
        new_last_eol = _take_eol(items[survivors[-1]])
    new_terminal_has_comma = items[last_idx].has_comma if tail_removed else False

    for i in reversed(sorted_removed):
        items.pop(i)

    if not items:
        cv.header_trivia, cv.final_trivia = strip_trailing_indent(
            cv.header_trivia, cv.final_trivia
        )
        return
    if tail_removed:
        new_last = items[-1]
        new_last.post_comma_trivia = ""
        new_last.has_comma = new_terminal_has_comma
        if not new_terminal_has_comma and "\n" in new_last.trailing:
            new_last.trailing = ""
        _put_eol(new_last, new_last_eol)
        if is_multiline:
            _shift_carried_boundary(
                cv,
                len(items),
                nl,
                old=boundaries_before[orig_len],
            )
    if is_multiline:
        for j in seams:
            _shift_carried_boundary(
                cv,
                j,
                nl,
                old=boundaries_before[survivors[j]],
            )
    indent = _value_indent(cv)
    if zero_removed:
        _replace_above(cv, 0, boundaries_before[survivors[0]], nl, indent)
    for j in seams:
        _replace_above(cv, j, boundaries_before[survivors[j]], nl, indent)
    if tail_removed:
        _replace_above(cv, len(items), boundaries_before[orig_len], nl)


def reorder_owned(
    cv: CommaValue[_CV_ItemT],
    owned_positions: Sequence[int],
    new_owned: Sequence[_CV_ItemT],
    nl: str,
    *,
    is_multiline: bool,
) -> None:
    """Reorder selected items while boundary shells stay positional."""
    if len(owned_positions) <= 1:
        return
    assert len(owned_positions) == len(new_owned)
    items = cv.items
    if all(
        items[pos] is entry
        for pos, entry in zip(owned_positions, new_owned, strict=True)
    ):
        return
    affected = {b for pos in owned_positions for b in (pos, pos + 1)}
    boundaries = {b: Boundary.capture(cv, b) for b in affected}
    above_by_item = {items[pos]: boundaries[pos] for pos in owned_positions}
    left_by_item = {items[pos]: boundaries[pos + 1] for pos in owned_positions}
    owned = set(owned_positions)
    incoming = dict(zip(owned_positions, new_owned, strict=True))
    indent = _value_indent(cv)
    composed: dict[int, Boundary] = {}
    for b, boundary in boundaries.items():
        shell = boundary.copy()
        if b in owned and "#" in shell.above:
            shell.remove_above()
        if b - 1 in owned:
            shell.remove_eol()
        left = incoming.get(b - 1)
        if left is not None:
            shell.put_eol_from(left_by_item[left], boundary)
        if is_multiline and b - 1 in owned:
            shell.shift_carried_from(
                boundary,
                nl,
                indent,
                is_terminal=b == len(items),
            )
        right = incoming.get(b)
        if right is not None:
            shell.carry_above_from(above_by_item[right], nl, indent)
        composed[b] = shell

    for pos, entry in incoming.items():
        items[pos] = entry
    for b in sorted(affected):
        composed[b].restore(cv, b)


__all__ = [
    "Boundary",
    "CommaStyle",
    "boundary_break_holder",
    "detect_style",
    "migrate_bracket_above",
    "reindent_as_leader",
    "reorder_owned",
    "shift_breaks",
    "splice_in",
    "splice_insert",
    "splice_out",
]
