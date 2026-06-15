"""Flavour-agnostic comment plumbing shared by inline arrays and tables.

Both `Array` (int-keyed) and inline-table (str-keyed) comment views
operate on the same `CommaValue` / `CommaItem` trivia model, so the
per-item read / write of EOL comments and above-item comment blocks
lives here once. The view classes in `_array_comments` /
`_inline_comments` add only the keying (item index vs entry leaf key)
and the multi-line promotion policy.

Per-item trivia ownership (see `CommaValue`):

  - Above-item region for item ``i``:
      ``header_trivia``       if i == 0
      ``items[i].leading``    if i >= 1 (comma-first parks it in the
                              previous item's ``trailing`` after its EOL)
  - EOL for item ``i``: ``item_eol_channel(item)``
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
    boundary_break_holder,
    reindent_as_leader,
    shift_pieces,
)
from tomlrt._comments import (
    _decode_comment,
    _encode_comment,
    _validate_comment_seq,
    _validate_comment_str,
)
from tomlrt._trivia import (
    CommentNode,
    NewlineNode,
    Trivia,
    WhitespaceNode,
    leading_break_index,
    split_above_block,
    split_eol_section,
    split_item_above,
)
from tomlrt._values import (
    item_breaks_before_comma,
    item_eol_channel,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any

    from tomlrt._trivia import TriviaPiece
    from tomlrt._values import CommaItem, CommaValue

_ItemT = TypeVar("_ItemT", bound="CommaItem")
_KeyT = TypeVar("_KeyT")
_ValueT = TypeVar("_ValueT")


# ---------------------------------------------------------------------------
# EOL comments
# ---------------------------------------------------------------------------


def _raw_eol_text(item: CommaItem) -> str | None:
    """Return the raw (still-encoded) EOL comment text on ``item`` or None."""
    for p in item_eol_channel(item).pieces:
        if isinstance(p, NewlineNode):
            return None
        if isinstance(p, CommentNode):
            return p.text
    return None


def _item_eol(item: CommaItem) -> str | None:
    """Decoded EOL comment on ``item``, or None."""
    raw = _raw_eol_text(item)
    return _decode_comment(raw) if raw is not None else None


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
    target = item_eol_channel(item)
    existing_eol, rest = split_eol_section(target)
    stripped = False
    if (
        not existing_eol.pieces
        and rest.pieces
        and isinstance(rest.pieces[0], NewlineNode)
    ):
        rest = Trivia(list(rest.pieces[1:]))
        stripped = True
    new_eol: list[TriviaPiece] = [
        WhitespaceNode(" "),
        CommentNode(raw_text),
        NewlineNode(nl),
    ]
    target.pieces = [*new_eol, *rest.pieces]
    if existing_eol.pieces or stripped:
        return
    nxt = boundary_break_holder(value, idx + 1)
    if leading_break_index(nxt.pieces) is not None:
        # The next item already starts a fresh line; the comment's own
        # newline replaces that break (a carried -1 boundary shift), along
        # with any stray trailing whitespace (``1,  \n``) that preceded it.
        nxt.pieces = shift_pieces(nxt.pieces, -1, nl)
    else:
        # The next item shared this row: the comment forces a break, so
        # promote it to a row leader at the value indent.
        reindent_as_leader(nxt, _value_indent(value))


def _del_eol(value: CommaValue[_ItemT], idx: int, nl: str) -> bool:
    """Remove the EOL comment on item ``idx``; return whether one existed."""
    item = value.items[idx]
    target = item_eol_channel(item)
    eol, rest = split_eol_section(target)
    if not eol.pieces:
        return False
    if item_breaks_before_comma(item):
        # The eol section's terminating newline is this row's break and
        # the comma follows it. Keep the break (drop only whitespace +
        # comment) and leave the next item alone.
        item.trailing = Trivia([NewlineNode(nl), *rest.pieces])
        return True
    # Non-comma-first: the row break lived inside the eol section. Drop the
    # whole section and re-home the break (plus any structural rest) onto the
    # downstream holder, mirroring _set_eol_raw in reverse. A bare break left
    # in the item's own channel renders identically but desyncs from a fresh
    # parse: reorder_owned treats that channel as positional and would orphan
    # a later item's EOL comment onto its own line.
    target.pieces = []
    nxt = boundary_break_holder(value, idx + 1)
    nxt.pieces = [NewlineNode(nl), *rest.pieces, *nxt.pieces]
    return True


# ---------------------------------------------------------------------------
# Above-item comment blocks
# ---------------------------------------------------------------------------


def _above_owner(value: CommaValue[_ItemT], i: int) -> tuple[Trivia, int]:
    """Return ``(owner, prefix_len)`` for item ``i``'s above-region.

    The first ``prefix_len`` owner pieces must be preserved. Comma-first
    parks the block in ``items[i-1].trailing`` after that item's EOL.
    """
    if i == 0:
        return value.header_trivia, 0
    prev = value.items[i - 1]
    if item_breaks_before_comma(prev):
        eol, _rest = split_eol_section(prev.trailing)
        return prev.trailing, len(eol.pieces)
    return value.items[i].leading, 0


def _comments_from_lines(pieces: list[TriviaPiece]) -> tuple[str, ...]:
    return tuple(_decode_comment(p.text) for p in pieces if isinstance(p, CommentNode))


def _read_above_comments(value: CommaValue[_ItemT], idx: int) -> tuple[str, ...]:
    """Decoded comments in item ``idx``'s above-region (all, source order)."""
    owner, prefix_len = _above_owner(value, idx)
    return _comments_from_lines(list(owner.pieces[prefix_len:]))


def _value_indent(value: CommaValue[_ItemT]) -> str:
    """Best-effort indent string for this value's items.

    ``_split_above_frame`` resolves the value-indent uniformly across
    bracket-pad, conventional, and comma-first layouts.
    """
    for i in range(len(value.items)):
        _owner, _prefix, _head, tail = _split_above_frame(value, i)
        # `split_item_above` / the item-0 bracket-pad split guarantee that
        # a non-empty `tail` is a single value-indent WhitespaceNode, so the
        # isinstance check never fails here (it exists only for the type).
        for p in reversed(tail.pieces):
            if isinstance(p, WhitespaceNode):  # pragma: no branch
                return p.text
    # No item carries an indent: the items sit at column zero, so a comment
    # block above them should too.
    return ""


def _render_above_block(
    raw_lines: tuple[str, ...], nl: str, ind: str
) -> list[TriviaPiece]:
    """Render a comment block as ``[WS(ind), Comment, NL] * n``.

    The value-indent WS lives in the surrounding pad, not here.
    """
    out: list[TriviaPiece] = []
    for raw in raw_lines:
        out.extend((WhitespaceNode(ind), CommentNode(raw), NewlineNode(nl)))
    return out


def _split_above_frame(
    value: CommaValue[_ItemT], i: int
) -> tuple[Trivia, Trivia, Trivia, Trivia]:
    """Decompose item ``i``'s above-region into ``(owner, prefix, head, tail)``.

    Rewrites assign ``owner.pieces``. ``prefix`` is preserved verbatim
    (comma-first predecessor EOL, else empty); ``head`` / ``tail`` frame
    the comment block. For item 0, pre-NL bracket comments stay in
    ``head`` rather than becoming above-blocks.
    """
    if i == 0:
        owner = value.header_trivia
        pad, _drop = split_above_block(owner)
        pieces = list(pad.pieces)
        nl_idx = next(
            (k for k, p in enumerate(pieces) if isinstance(p, NewlineNode)),
            -1,
        )
        if nl_idx < 0:
            return owner, Trivia(), Trivia(pieces), Trivia()
        head = Trivia(pieces[: nl_idx + 1])
        tail = Trivia(pieces[nl_idx + 1 :])
        return owner, Trivia(), head, tail
    owner, prefix_len = _above_owner(value, i)
    prefix = Trivia(list(owner.pieces[:prefix_len]))
    head, _drop, tail = split_item_above(Trivia(list(owner.pieces[prefix_len:])))
    return owner, prefix, head, tail


def _set_above_pieces(
    value: CommaValue[_ItemT],
    i: int,
    raw_lines: tuple[str, ...],
    nl: str,
    ind: str,
) -> None:
    """Replace the comment block in item ``i``'s above-region.

    Preserve structural framing and rewrite only the comment block
    between it.
    """
    owner, prefix, head, tail = _split_above_frame(value, i)
    if not head.pieces and not tail.pieces:
        # An empty region needs its own framing. When `prefix` already
        # ends in a row break (the comma-first layout) reuse it rather
        # than stacking a second newline.
        has_break = bool(prefix.pieces) and isinstance(prefix.pieces[-1], NewlineNode)
        head = Trivia([] if has_break else [NewlineNode(nl)])
        tail = Trivia([WhitespaceNode(ind)])
    above = _render_above_block(raw_lines, nl, ind)
    owner.pieces = [*prefix.pieces, *head.pieces, *above, *tail.pieces]


def _clear_above_pieces(value: CommaValue[_ItemT], i: int) -> None:
    """Strip the comment block from item ``i``'s above-region; keep framing."""
    owner, prefix, head, tail = _split_above_frame(value, i)
    owner.pieces = [*prefix.pieces, *head.pieces, *tail.pieces]


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
    def candidates(self) -> Iterator[_KeyT]:
        """The keys that could carry a comment, in item order."""


class _CommaView(MutableMapping[_KeyT, _ValueT]):
    """Shared mapping plumbing over a `CommaCommentAdapter`.

    Subclasses add ``_present`` and the item accessors; this base owns
    membership, iteration, and length.
    """

    __slots__ = ("_a",)

    def __init__(self, adapter: CommaCommentAdapter[_KeyT]) -> None:
        self._a = adapter

    def _present(self, idx: int) -> bool:
        raise NotImplementedError

    @override
    def __contains__(self, key: object) -> bool:
        try:
            idx = self._a.resolve(key)
        except TypeError:
            return False
        return idx is not None and self._present(idx)

    @override
    def __iter__(self) -> Iterator[_KeyT]:
        for key in self._a.candidates():
            idx = self._a.resolve(key)
            if idx is not None and self._present(idx):
                yield key

    @override
    def __len__(self) -> int:
        return sum(1 for _ in self)

    @override
    def __repr__(self) -> str:
        return repr(dict(self))


class CommaEolView(_CommaView[_KeyT, str]):
    """EOL-comment mapping over a comma-value, keyed by ``_KeyT``."""

    __slots__ = ()

    @override
    def _present(self, idx: int) -> bool:
        return _item_eol(self._a.value().items[idx]) is not None

    @override
    def __getitem__(self, key: _KeyT) -> str:
        idx = self._a.resolve(key)
        if idx is None:
            raise KeyError(key)
        eol = _item_eol(self._a.value().items[idx])
        if eol is None:
            raise KeyError(key)
        return eol

    @override
    def __setitem__(self, key: _KeyT, value: str) -> None:
        _validate_comment_str(value, "comment text")
        # Resolve before any structural change: a missing key must not
        # leave a partially-promoted single-line value in multi-line form.
        idx = self._a.resolve(key)
        if idx is None:
            raise KeyError(key)
        self._a.promote()
        # Promotion preserves item order, so the resolved index is stable.
        _set_eol_raw(self._a.value(), idx, _encode_comment(value), self._a.newline())

    @override
    def __delitem__(self, key: _KeyT) -> None:
        idx = self._a.resolve(key)
        if idx is None:
            raise KeyError(key)
        if not _del_eol(self._a.value(), idx, self._a.newline()):
            raise KeyError(key)


class CommaLeadingView(_CommaView[_KeyT, "tuple[str, ...]"]):
    """Above-item comment-block mapping over a comma-value, keyed by ``_KeyT``."""

    __slots__ = ()

    def _read(self, idx: int) -> tuple[str, ...]:
        return _read_above_comments(self._a.value(), idx)

    @override
    def _present(self, idx: int) -> bool:
        return bool(self._read(idx))

    @override
    def __getitem__(self, key: _KeyT) -> tuple[str, ...]:
        idx = self._a.resolve(key)
        if idx is None:
            raise KeyError(key)
        out = self._read(idx)
        if not out:
            raise KeyError(key)
        return out

    @override
    def __setitem__(self, key: _KeyT, value: tuple[str, ...] | list[str]) -> None:
        seq = _validate_comment_seq(value, "leading_comments")
        idx = self._a.resolve(key)
        if idx is None:
            raise KeyError(key)
        if not seq:
            # Empty assignment means "no leading comments" — a
            # delete-if-present. Don't promote: zero comments need no
            # newlines.
            if self._read(idx):
                _clear_above_pieces(self._a.value(), idx)
            return
        self._a.promote()
        value_obj = self._a.value()
        encoded = tuple(_encode_comment(c) for c in seq)
        _set_above_pieces(
            value_obj, idx, encoded, self._a.newline(), _value_indent(value_obj)
        )

    @override
    def __delitem__(self, key: _KeyT) -> None:
        idx = self._a.resolve(key)
        if idx is None:
            raise KeyError(key)
        if not self._read(idx):
            raise KeyError(key)
        _clear_above_pieces(self._a.value(), idx)

    @override
    def __repr__(self) -> str:
        return repr({k: list(v) for k, v in self.items()})


__all__ = [
    "CommaCommentAdapter",
    "CommaEolView",
    "CommaLeadingView",
]
