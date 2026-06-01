"""Comment side-channel views for ``Array``.

``Array.comments`` (EOL comments per item) and
``Array.leading_comments`` (tuple of comment lines above each item)
are implemented here.  Both are indexed by item position and share
the encode/decode helpers and validation rules from ``_comments``.

Per-item trivia ownership (canonical model — see ``ArrayValue``):

  - Above-item region for item ``i``:
      ``header_trivia``       if i == 0
      ``items[i].leading``    if i >= 1
  - EOL for item ``i``:
      ``items[i].post_comma_trivia``  if has_comma
      ``items[i].trailing``           otherwise
"""

from __future__ import annotations

import sys
from collections.abc import MutableMapping
from typing import TYPE_CHECKING, TypeVar

if sys.version_info >= (3, 12):
    from typing import override
else:  # pragma: no cover -- backport for Python < 3.12
    from typing_extensions import override

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
    split_above_block,
    split_eol_section,
    split_item_above,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from tomlrt._array import Array
    from tomlrt._trivia import (
        TriviaPiece,
    )
    from tomlrt._values import ArrayItem, ArrayValue


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _check_index(arr: Array, key: object) -> int:
    if not isinstance(key, int) or isinstance(key, bool):
        msg = f"Array comment indices must be int, not {type(key).__name__}"
        raise TypeError(msg)
    items = arr._value.items  # noqa: SLF001
    n = len(items)
    idx = key if key >= 0 else key + n
    if idx < 0 or idx >= n:
        raise KeyError(key)
    return idx


def _eol_target(item: ArrayItem) -> Trivia:
    return item.post_comma_trivia if item.has_comma else item.trailing


def _raw_eol_text(item: ArrayItem) -> str | None:
    """Return the raw (still-encoded) EOL comment text on ``item`` or None."""
    for p in _eol_target(item).pieces:
        if isinstance(p, NewlineNode):
            return None
        if isinstance(p, CommentNode):
            return p.text
    return None


def _item_eol(item: ArrayItem) -> str | None:
    raw = _raw_eol_text(item)
    return _decode_comment(raw) if raw is not None else None


def _ensure_multiline(arr: Array) -> None:
    if not arr.multiline:
        arr.set_multiline(multiline=True)


def _above_target(value: ArrayValue, i: int) -> Trivia:
    return value.header_trivia if i == 0 else value.items[i].leading


def _comments_from_lines(pieces: list[TriviaPiece]) -> tuple[str, ...]:
    return tuple(_decode_comment(p.text) for p in pieces if isinstance(p, CommentNode))


def _slot_indent(arr: Array) -> str:
    """Best-effort indent string for this array's items."""
    value = arr._value  # noqa: SLF001
    items = value.items
    # Prefer the indent immediately before items[1]'s value (the
    # canonical inter-item separator under the new model), then fall
    # back to header_trivia, then to a reasonable default.
    sources: list[Trivia] = []
    if len(items) >= 2:
        sources.append(items[1].leading)
    sources.append(value.header_trivia)
    sources.extend(it.leading for it in items[2:])
    for src in sources:
        for p in reversed(src.pieces):
            if isinstance(p, WhitespaceNode):
                return p.text
            if isinstance(p, NewlineNode):
                break
    return "  "


# ---------------------------------------------------------------------------
# EOL view
# ---------------------------------------------------------------------------


_T = TypeVar("_T")


class _ArrayIntKeyedView(MutableMapping[int, _T]):
    __slots__ = ("_arr",)

    def __init__(self, arr: Array) -> None:
        self._arr = arr

    def _present(self, idx: int) -> bool:
        raise NotImplementedError

    @override
    def __contains__(self, key: object) -> bool:
        if not isinstance(key, int) or isinstance(key, bool):
            return False
        try:
            idx = _check_index(self._arr, key)
        except KeyError:
            return False
        return self._present(idx)

    @override
    def __iter__(self) -> Iterator[int]:
        items = self._arr._value.items  # noqa: SLF001
        for i in range(len(items)):
            if self._present(i):
                yield i

    @override
    def __len__(self) -> int:
        return sum(1 for _ in self)


def _set_eol_raw(arr: Array, idx: int, raw_text: str) -> None:
    """Stamp a raw (already-encoded) EOL comment onto item ``idx``.

    Maintains the canonical-model invariant that, when the item bears
    a comma, the structural newline lives on the *next* item's
    ``leading`` (or on ``final_trivia`` for the last item). Adding an
    EOL section carries its own newline, so the downstream NL is
    stripped to avoid duplication.
    """
    items = arr._value.items  # noqa: SLF001
    item = items[idx]
    nl = arr._doc_newline  # noqa: SLF001
    target = _eol_target(item)
    existing_eol, rest = split_eol_section(target)
    if (
        not existing_eol.pieces
        and rest.pieces
        and isinstance(rest.pieces[0], NewlineNode)
    ):
        # Replace the structural newline; our synthesised EOL
        # provides its own.
        rest = Trivia(list(rest.pieces[1:]))
    new_eol: list[TriviaPiece] = [
        WhitespaceNode(" "),
        CommentNode(raw_text),
        NewlineNode(nl),
    ]
    target.pieces = [*new_eol, *rest.pieces]
    if item.has_comma and not existing_eol.pieces:
        value = arr._value  # noqa: SLF001
        nxt = items[idx + 1].leading if idx + 1 < len(items) else value.final_trivia
        if nxt.pieces and isinstance(nxt.pieces[0], NewlineNode):
            nxt.pieces = list(nxt.pieces[1:])


class ArrayEolView(_ArrayIntKeyedView[str]):
    __slots__ = ()

    @override
    def _present(self, idx: int) -> bool:
        return _item_eol(self._arr._value.items[idx]) is not None  # noqa: SLF001

    @override
    def __getitem__(self, key: int) -> str:
        idx = _check_index(self._arr, key)
        eol = _item_eol(self._arr._value.items[idx])  # noqa: SLF001
        if eol is None:
            raise KeyError(key)
        return eol

    @override
    def __setitem__(self, key: int, value: str) -> None:
        _validate_comment_str(value, "comment text")
        _ensure_multiline(self._arr)
        idx = _check_index(self._arr, key)
        _set_eol_raw(self._arr, idx, _encode_comment(value))

    @override
    def __delitem__(self, key: int) -> None:
        idx = _check_index(self._arr, key)
        items = self._arr._value.items  # noqa: SLF001
        item = items[idx]
        target = _eol_target(item)
        eol, rest = split_eol_section(target)
        if not eol.pieces:
            raise KeyError(key)
        target.pieces = list(rest.pieces)
        if not item.has_comma:
            return
        # Canonical model inverse: restore the structural NL onto the
        # next item's leading (or final_trivia for the last) so the
        # closing layout still has its row break.
        value = self._arr._value  # noqa: SLF001
        nxt = items[idx + 1].leading if idx + 1 < len(items) else value.final_trivia
        nl = self._arr._doc_newline  # noqa: SLF001
        if not (nxt.pieces and isinstance(nxt.pieces[0], NewlineNode)):
            nxt.pieces = [NewlineNode(nl), *nxt.pieces]

    @override
    def __repr__(self) -> str:
        return repr(dict(self))


# ---------------------------------------------------------------------------
# Leading view
# ---------------------------------------------------------------------------


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


def _split_above_frame(value: ArrayValue, i: int) -> tuple[Trivia, Trivia]:
    """Split item ``i``'s above-region into framing ``(head, tail)``.

    Any current above-block is discarded; mutators that want to
    preserve it must use the underlying splitter directly.

    For ``i == 0`` the target is ``header_trivia`` (bracket-pad
    context): a pre-NL comment is an EOL on ``[`` and stays in
    ``head``. For ``i >= 1`` the target is ``items[i].leading``:
    a pre-NL comment is the item's above-block, possibly hoisted
    from item ``i-1``'s EOL onto its ``post_comma_trivia``.
    """
    target = _above_target(value, i)
    if i >= 1:
        head, _drop, tail = split_item_above(target)
        return head, tail
    pad, _drop = split_above_block(target)
    pieces = list(pad.pieces)
    nl_idx = next(
        (k for k, p in enumerate(pieces) if isinstance(p, NewlineNode)),
        -1,
    )
    if nl_idx < 0:
        return Trivia(pieces), Trivia()
    return Trivia(pieces[: nl_idx + 1]), Trivia(pieces[nl_idx + 1 :])


def _set_above_pieces(
    value: ArrayValue,
    i: int,
    raw_lines: tuple[str, ...],
    nl: str,
    ind: str,
) -> None:
    """Replace the comment block in item ``i``'s above-region.

    Preserves any structural framing already present (leading NL,
    trailing value-indent WS) and only rewrites the comment block
    between.
    """
    head, tail = _split_above_frame(value, i)
    if not head.pieces and not tail.pieces:
        head = Trivia([NewlineNode(nl)])
        tail = Trivia([WhitespaceNode(ind)])
    above = _render_above_block(raw_lines, nl, ind)
    _above_target(value, i).pieces = [*head.pieces, *above, *tail.pieces]


def _clear_above_pieces(value: ArrayValue, i: int) -> None:
    """Strip the comment block from item ``i``'s above-region; keep framing."""
    head, tail = _split_above_frame(value, i)
    _above_target(value, i).pieces = [*head.pieces, *tail.pieces]


class ArrayLeadingView(_ArrayIntKeyedView[tuple[str, ...]]):
    __slots__ = ()

    def _read(self, idx: int) -> tuple[str, ...]:
        return _comments_from_lines(
            list(_above_target(self._arr._value, idx).pieces)  # noqa: SLF001
        )

    @override
    def _present(self, idx: int) -> bool:
        return bool(self._read(idx))

    @override
    def __getitem__(self, key: int) -> tuple[str, ...]:
        idx = _check_index(self._arr, key)
        out = self._read(idx)
        if not out:
            raise KeyError(key)
        return out

    @override
    def __setitem__(self, key: int, value: tuple[str, ...] | list[str]) -> None:
        seq = _validate_comment_seq(value, "leading_comments")
        idx = _check_index(self._arr, key)
        if not seq:
            # Empty assignment means "no leading comments" — semantically
            # a delete-if-present. Don't auto-promote to multi-line: the
            # array doesn't need newlines to carry zero comments.
            self._delete(idx, allow_missing=True)
            return
        _ensure_multiline(self._arr)
        encoded = tuple(_encode_comment(c) for c in seq)
        _set_above_pieces(
            self._arr._value,  # noqa: SLF001
            idx,
            encoded,
            self._arr._doc_newline,  # noqa: SLF001
            _slot_indent(self._arr),
        )

    def _delete(self, idx: int, *, allow_missing: bool) -> None:
        if not self._read(idx):
            if not allow_missing:
                raise KeyError(idx)
            return
        _clear_above_pieces(
            self._arr._value,  # noqa: SLF001
            idx,
        )

    @override
    def __delitem__(self, key: int) -> None:
        idx = _check_index(self._arr, key)
        self._delete(idx, allow_missing=False)

    @override
    def __repr__(self) -> str:
        return repr({k: list(v) for k, v in self.items()})
