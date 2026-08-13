"""Flavour-agnostic comment plumbing shared by inline arrays and tables.

Both `Array` (int-keyed) and inline-table (str-keyed) comment views
operate on the same `CommaValue` / `CommaItem` trivia model (see
`CommaValue` for the canonical ownership rules), so the per-item read /
write of EOL comments and above-item comment blocks lives here once.
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from collections.abc import MutableMapping
from typing import TYPE_CHECKING, Generic, TypeVar

if sys.version_info >= (3, 12):
    from typing import override
else:  # pragma: no cover -- backport for Python < 3.12
    from typing_extensions import override

from tomlrt._comma_ops import (
    Boundary,
    _value_indent,
    boundary_break_holder,
    reindent_as_leader,
    set_boundary_break_holder,
    shift_breaks,
)
from tomlrt._comment_text import (
    _encode_comment,
    _extract_leading_comments,
    _line_to_comment,
    _render_comment_lines,
    _split_attached_block,
    _validate_comment_entries,
    _validate_comment_seq,
    _validate_comment_str,
)
from tomlrt._trivia import leading_break, split_eol_section, split_lines
from tomlrt._values import (
    item_eol_channel,
    item_eol_on_trailing,
    set_item_eol_channel,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any

    from tomlrt._values import CommaItem, CommaValue

_ItemT = TypeVar("_ItemT", bound="CommaItem")
_KeyT = TypeVar("_KeyT")
_ValueT = TypeVar("_ValueT")


# ---------------------------------------------------------------------------
# EOL comments
# ---------------------------------------------------------------------------


def _item_eol(item: CommaItem) -> str | None:
    """Decoded EOL comment on ``item``, or None."""
    eol, _rest = split_eol_section(item_eol_channel(item))
    return _line_to_comment(eol) if eol else None


def _set_eol_raw(value: CommaValue[_ItemT], idx: int, raw_text: str, nl: str) -> None:
    """Stamp a raw (already-encoded) EOL comment onto item ``idx``.

    The synthesised EOL section ends with its own newline, so the
    structural newline that previously terminated the item's line must
    be removed to avoid duplication. Depending on layout, it lives:

    * inside ``target`` — handled by the ``rest`` strip below;
    * on the next item's ``leading`` — has-comma, non-tail item;
    * in the value's ``final_trivia`` — tail item (with or without
      a trailing comma).
    """
    item = value.items[idx]
    # Sampled before the write: the row the comment is about to open
    # would otherwise be the first row this value appears to have.
    indent = _value_indent(value)
    existing_eol, rest = split_eol_section(item_eol_channel(item))
    stripped = False
    if not existing_eol and rest.startswith(("\n", "\r\n")):
        rest = rest[2:] if rest[0] == "\r" else rest[1:]
        stripped = True
    set_item_eol_channel(item, f" {raw_text}{nl}{rest}")
    if existing_eol or stripped:
        return
    nxt = boundary_break_holder(value, idx + 1)
    if leading_break(nxt):
        # The next item already starts a fresh line; the comment's own
        # newline replaces that break (a carried -1 boundary shift), along
        # with any stray trailing whitespace (``1,  \n``) that preceded it.
        nxt = shift_breaks(nxt, -1, nl)
    else:
        # The next item shared this row: the comment forces a break, so
        # promote it to a row leader at the value indent.
        nxt = reindent_as_leader(nxt, indent)
    set_boundary_break_holder(value, idx + 1, nxt)


def _del_eol(value: CommaValue[_ItemT], idx: int, nl: str) -> bool:
    """Remove the EOL comment on item ``idx``; return whether one existed."""
    item = value.items[idx]
    eol, rest = split_eol_section(item_eol_channel(item))
    if not eol:
        return False
    if item.has_comma and item_eol_on_trailing(item):
        # The eol section's terminating newline is this row's break and
        # the comma follows it. Keep the break (drop only whitespace +
        # comment) and leave the next item alone.
        item.trailing = nl + rest
        return True
    # Non-comma-first: the row break lived inside the eol section. Drop the
    # whole section and re-home the break (plus any structural rest) onto the
    # downstream holder, mirroring _set_eol_raw in reverse. A bare break left
    # in the item's own channel renders identically but desyncs from a fresh
    # parse: reorder_owned treats that channel as positional and would orphan
    # a later item's EOL comment onto its own line.
    set_item_eol_channel(item, "")
    nxt = boundary_break_holder(value, idx + 1)
    set_boundary_break_holder(value, idx + 1, nl + rest + nxt)
    return True


# ---------------------------------------------------------------------------
# Above-item comment blocks
# ---------------------------------------------------------------------------


def _read_above_block(value: CommaValue[_ItemT], idx: int) -> tuple[str | None, ...]:
    """Decoded full block in item ``idx``'s above-region."""
    boundary = Boundary.capture(value, idx)
    return tuple(_line_to_comment(line) for line in split_lines(boundary.above))


def _read_above_comments(value: CommaValue[_ItemT], idx: int) -> tuple[str, ...]:
    """Decoded attached comment run in item ``idx``'s above-region."""
    block = Boundary.capture(value, idx).attached_above
    if block is None:
        return ()
    return _extract_leading_comments(block)


def _set_above_block(
    value: CommaValue[_ItemT],
    i: int,
    block: tuple[str | None, ...],
    nl: str,
    ind: str,
) -> None:
    """Replace the full block in item ``i``'s above-region."""
    boundary = Boundary.capture(value, i)
    rendered = _render_comment_lines(block, nl, ind)
    boundary.set_above(rendered, nl, ind).restore(value, i)


def _set_attached_comments(
    value: CommaValue[_ItemT],
    i: int,
    comments: tuple[str, ...],
    nl: str,
    ind: str,
) -> None:
    """Replace only the attached comment run, preserving earlier lines."""
    boundary = Boundary.capture(value, i)
    part = boundary.attached_above or boundary.target_above
    above, _attached, _indent = _split_attached_block(part)
    rendered = _render_comment_lines(comments, nl, ind)
    boundary.set_attached(above + rendered, nl, ind).restore(value, i)


def _clear_attached_comments(value: CommaValue[_ItemT], i: int) -> None:
    """Strip the attached comment run while preserving earlier lines."""
    boundary = Boundary.capture(value, i)
    part = boundary.attached_above
    assert part is not None
    above, _attached, _indent = _split_attached_block(part)
    boundary.set_attached(above, "\n", "").restore(value, i)


def _clear_above_block(value: CommaValue[_ItemT], i: int) -> None:
    """Strip the full block from item ``i``'s above-region; keep framing."""
    Boundary.capture(value, i).remove_above().restore(value, i)


# ---------------------------------------------------------------------------
# Keyed mapping views
# ---------------------------------------------------------------------------


class CommaCommentAdapter(ABC, Generic[_KeyT]):
    """Flavour hooks that bind a comma-value comment view to its owner.

    `Array` keys comments by item index; an inline table keys them by
    entry leaf key. Each flavour supplies these few hooks; the generic
    views below own all of the read / write / mapping logic on top.
    """

    __slots__ = ()

    @abstractmethod
    def value(self) -> CommaValue[Any]:
        """The backing comma-value whose items carry the comments."""

    @abstractmethod
    def resolve(self, key: object) -> int | None:
        """Item index for ``key``, or None when no such item exists.

        May raise ``TypeError`` for a key of the wrong type; callers that
        must not propagate it (``__contains__``) catch it.
        """

    @abstractmethod
    def promote(self) -> None:
        """Ensure the value is multi-line so a comment has somewhere to live."""

    @abstractmethod
    def newline(self) -> str:
        """The owning document's newline string."""

    @abstractmethod
    def indexed_candidates(self) -> Iterator[tuple[_KeyT, int]]:
        """The candidate keys and item indices, in item order."""


class _CommaView(MutableMapping[_KeyT, _ValueT]):
    """Shared mapping plumbing over a `CommaCommentAdapter`.

    Subclasses supply ``_get(idx)`` — the current stored value or None
    when absent — and their own writers; this base derives membership,
    iteration, and ``__getitem__`` from that hook.
    """

    __slots__ = ("_a",)

    def __init__(self, adapter: CommaCommentAdapter[_KeyT]) -> None:
        self._a = adapter

    @abstractmethod
    def _get(self, idx: int) -> _ValueT | None:
        """Return the value at ``idx``, or None when absent."""

    def _idx(self, key: _KeyT) -> int:
        idx = self._a.resolve(key)
        if idx is None:
            raise KeyError(key)
        return idx

    @override
    def __contains__(self, key: object) -> bool:
        try:
            idx = self._a.resolve(key)
        except TypeError:
            return False
        return idx is not None and self._get(idx) is not None

    @override
    def __iter__(self) -> Iterator[_KeyT]:
        for key, idx in self._a.indexed_candidates():
            if self._get(idx) is not None:
                yield key

    @override
    def __len__(self) -> int:
        return sum(1 for _ in self)

    @override
    def __getitem__(self, key: _KeyT) -> _ValueT:
        v = self._get(self._idx(key))
        if v is None:
            raise KeyError(key)
        return v

    @override
    def __repr__(self) -> str:
        return repr(dict(self))


class CommaEolView(_CommaView[_KeyT, str]):
    """EOL-comment mapping over a comma-value, keyed by ``_KeyT``."""

    __slots__ = ()

    @override
    def _get(self, idx: int) -> str | None:
        return _item_eol(self._a.value().items[idx])

    @override
    def __setitem__(self, key: _KeyT, value: str) -> None:
        _validate_comment_str(value, "comment text")
        # Resolve before any structural change: a missing key must not
        # leave a partially-promoted single-line value in multi-line form.
        idx = self._idx(key)
        self._a.promote()
        # Promotion preserves item order, so the resolved index is stable.
        _set_eol_raw(self._a.value(), idx, _encode_comment(value), self._a.newline())

    @override
    def __delitem__(self, key: _KeyT) -> None:
        idx = self._idx(key)
        if not _del_eol(self._a.value(), idx, self._a.newline()):
            raise KeyError(key)


class CommaLeadingView(_CommaView[_KeyT, "tuple[str, ...]"]):
    """Above-item comment-block mapping over a comma-value, keyed by ``_KeyT``."""

    __slots__ = ()

    @override
    def _get(self, idx: int) -> tuple[str, ...] | None:
        return _read_above_comments(self._a.value(), idx) or None

    @override
    def __setitem__(self, key: _KeyT, value: tuple[str, ...] | list[str]) -> None:
        idx = self._idx(key)
        seq = _validate_comment_seq(value, "leading_comments")
        if not seq:
            # Empty assignment means "no leading comments" — a
            # delete-if-present. Don't promote: zero comments need no
            # newlines.
            if self._get(idx) is not None:
                _clear_attached_comments(self._a.value(), idx)
            return
        self._a.promote()
        value_obj = self._a.value()
        _set_attached_comments(
            value_obj, idx, seq, self._a.newline(), _value_indent(value_obj)
        )

    @override
    def __delitem__(self, key: _KeyT) -> None:
        idx = self._idx(key)
        if self._get(idx) is None:
            raise KeyError(key)
        _clear_attached_comments(self._a.value(), idx)


class CommaLeadingBlockView(_CommaView[_KeyT, "tuple[str | None, ...]"]):
    """Full above-item block mapping over a comma-value, keyed by ``_KeyT``."""

    __slots__ = ()

    @override
    def _get(self, idx: int) -> tuple[str | None, ...] | None:
        return _read_above_block(self._a.value(), idx) or None

    @override
    def __setitem__(
        self, key: _KeyT, value: tuple[str | None, ...] | list[str | None]
    ) -> None:
        idx = self._idx(key)
        block = _validate_comment_entries(value, "leading_block", allow_none=True)
        if not block:
            if self._get(idx) is not None:
                _clear_above_block(self._a.value(), idx)
            return
        self._a.promote()
        value_obj = self._a.value()
        _set_above_block(
            value_obj, idx, block, self._a.newline(), _value_indent(value_obj)
        )

    @override
    def __delitem__(self, key: _KeyT) -> None:
        idx = self._idx(key)
        if self._get(idx) is None:
            raise KeyError(key)
        _clear_above_block(self._a.value(), idx)


__all__ = [
    "CommaCommentAdapter",
    "CommaEolView",
    "CommaLeadingBlockView",
    "CommaLeadingView",
]
