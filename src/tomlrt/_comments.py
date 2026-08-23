"""Expose comment side-channel views over slot trivia.

`Container.comments` maps direct keys to decoded EOL comments;
`Container.leading_comments` maps them to attached leading blocks. A
key is exposed on its immediate parent (``b.c = 1`` under ``[a]``
appears on ``a["b"]``). These views are for section-backed containers;
the inline-table equivalents live in `_inline_comments`.
"""

from __future__ import annotations

import sys
from abc import abstractmethod
from collections.abc import MutableMapping
from typing import TYPE_CHECKING, TypeVar

if sys.version_info >= (3, 12):
    from typing import override
else:  # pragma: no cover -- backport for Python < 3.12
    from typing_extensions import override

from tomlrt._comment_text import (
    _decode_comment,
    _encode_comment,
    _extract_leading_comments,
    _line_to_comment,
    _lines_to_comments,
    _render_comment_lines,
    _set_attached_block,
    _split_attached_block,
    _validate_comment_entries,
    _validate_comment_seq,
    _validate_comment_str,
)
from tomlrt._errors import TOMLError
from tomlrt._kind import _Kind
from tomlrt._slots import KVSlot, StructuralHeaderSlot, ensure_terminator
from tomlrt._trivia import split_line, split_lines

if TYPE_CHECKING:
    from collections.abc import Iterator

    from tomlrt._container import Container, Document
    from tomlrt._slots import Slot


def _direct_kv_slot(c: Container, key: str) -> KVSlot | None:
    """Return the primary direct-KV slot for ``key`` in ``c``, or None."""
    refs = c._index.get(key)  # noqa: SLF001
    if not refs:
        return None
    target = (*c._path, key)  # noqa: SLF001
    for ref in refs:
        slot = ref.slot
        if isinstance(slot, KVSlot) and slot.host_path + slot.key == target:
            return slot
    return None


def _require_attached(c: Container) -> None:
    """Reject comment-view mutation on a container with no slot stream.

    Detached containers have nowhere to store comment mutations, so
    raise a clear `TOMLError` instead of an internal ``KeyError``.
    """
    if c._layout_root is None:  # noqa: SLF001
        msg = (
            "comments view is unavailable on a detached container; "
            "attach the container to a Document first (e.g. doc[k] = table) "
            "and then mutate doc[k].comments"
        )
        raise TOMLError(msg)


_T = TypeVar("_T")


class _SlotKeyedView(MutableMapping[str, _T]):
    """Mapping over Container keys whose direct-KV slot carries a value.

    Subclasses answer `_get` -- this view's value for a slot, or ``None``
    when the slot carries none -- and `_clear`, which removes it. Every
    read path derives from `_get`, so presence and value are decided by
    one piece of code rather than by a predicate that a getter then has
    to agree with.
    """

    __slots__ = ("_c",)

    def __init__(self, container: Container) -> None:
        self._c = container

    def _slot(self, key: str) -> KVSlot | None:
        return _direct_kv_slot(self._c, key)

    def _require_slot(self, key: str, *, missing_msg: str | None = None) -> KVSlot:
        slot = self._slot(key)
        if slot is None:
            raise KeyError(key if missing_msg is None else missing_msg)
        return slot

    @abstractmethod
    def _get(self, slot: KVSlot) -> _T | None:
        """This view's value for ``slot``, or ``None`` if it carries none."""

    @abstractmethod
    def _clear(self, slot: KVSlot) -> None:
        """Remove this view's value from ``slot``."""

    @override
    def __repr__(self) -> str:
        return repr(dict(self))

    @override
    def __getitem__(self, key: str) -> _T:
        value = self._get(self._require_slot(key))
        if value is None:
            raise KeyError(key)
        return value

    @override
    def __delitem__(self, key: str) -> None:
        _require_attached(self._c)
        slot = self._require_slot(key)
        if self._get(slot) is None:
            raise KeyError(key)
        self._clear(slot)

    @override
    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        slot = self._slot(key)
        return slot is not None and self._get(slot) is not None

    @override
    def __iter__(self) -> Iterator[str]:
        seen: set[str] = set()
        for ref in self._c._refs:  # noqa: SLF001
            k = ref.local_key
            if k is None or k in seen:
                continue
            slot = self._slot(k)
            if slot is not None and self._get(slot) is not None:
                seen.add(k)
                yield k

    @override
    def __len__(self) -> int:
        return sum(1 for _ in self)


class EolCommentView(_SlotKeyedView[str]):
    __slots__ = ()

    @override
    def _get(self, slot: KVSlot) -> str | None:
        return _read_eol_comment(slot.eol)

    @override
    def _clear(self, slot: KVSlot) -> None:
        slot.eol = _clear_eol_comment(slot.eol)

    @override
    def __setitem__(self, key: str, value: str) -> None:
        _require_attached(self._c)
        slot = self._require_slot(key, missing_msg=f"key {key!r} not in container")
        _validate_comment_str(value, "comment")
        slot.eol = _write_eol_comment(slot.eol, value, self._c._doc_newline)  # noqa: SLF001


def _read_leading_block(slot: Slot) -> tuple[str | None, ...]:
    """Decoded leading block of ``slot``.

    Comment lines decode to text, blank/whitespace-only lines become
    ``None``, and the slot's own trailing indent is excluded.
    """
    above, attached, _indent = _split_attached_block(slot.leading)
    lines = split_lines(above + attached)
    return tuple(_line_to_comment(line) for line in lines)


def _write_leading_block(
    c: Container, slot: Slot, block: tuple[str | None, ...]
) -> None:
    """Replace ``slot``'s leading with ``block``.

    Preserve the slot's column indent; comment lines reuse it and blank
    lines emit a bare newline.
    """
    _above, _attached, indent = _split_attached_block(slot.leading)
    nl = c._doc_newline  # noqa: SLF001
    slot.leading = _render_comment_lines(block, nl, indent) + indent


class LeadingCommentView(_SlotKeyedView[tuple[str, ...]]):
    __slots__ = ()

    @override
    def _get(self, slot: KVSlot) -> tuple[str, ...] | None:
        _above, attached, _indent = _split_attached_block(slot.leading)
        return _lines_to_comments(attached) if "#" in attached else None

    @override
    def _clear(self, slot: KVSlot) -> None:
        above, _attached, indent = _split_attached_block(slot.leading)
        slot.leading = above + indent

    @override
    def __setitem__(self, key: str, value: tuple[str, ...]) -> None:
        _require_attached(self._c)
        slot = self._require_slot(key, missing_msg=f"key {key!r} not in container")
        comments = _validate_comment_seq(value, "leading_comments")
        nl = self._c._doc_newline  # noqa: SLF001
        slot.leading = _set_attached_block(slot.leading, comments, nl)


class LeadingBlockView(_SlotKeyedView[tuple[str | None, ...]]):
    """Mapping view over the full leading-trivia block of each direct key.

    Entries are comment strings in source order, with ``None`` for blank
    lines; the slot's column indent is excluded and reapplied on write.

    The document head preamble lives separately, on
    :attr:`Document.preamble`.
    """

    __slots__ = ()

    @override
    def _get(self, slot: KVSlot) -> tuple[str | None, ...] | None:
        return _read_leading_block(slot) or None

    @override
    def _clear(self, slot: KVSlot) -> None:
        _write_leading_block(self._c, slot, ())

    @override
    def __setitem__(self, key: str, value: tuple[str | None, ...]) -> None:
        _require_attached(self._c)
        slot = self._require_slot(key, missing_msg=f"key {key!r} not in container")
        block = _validate_comment_entries(value, "leading_block", allow_none=True)
        _write_leading_block(self._c, slot, block)


def _header_slot(c: Container) -> StructuralHeaderSlot | None:
    """Return the StructuralHeaderSlot for a section container, or raise.

    Inline tables raise because they have no header; document roots and
    purely implicit containers return ``None``.
    """
    if c._inline:  # noqa: SLF001
        msg = "header comment API is not available on inline tables"
        raise TOMLError(msg)
    if c._kind is not _Kind.SECTION:  # noqa: SLF001
        return None
    hr = c._header_ref  # noqa: SLF001
    assert hr is not None  # implied by SECTION
    slot = hr.slot
    assert isinstance(slot, StructuralHeaderSlot)
    return slot


def _require_header_slot(c: Container, msg: str) -> StructuralHeaderSlot:
    """Return ``c``'s header slot or raise ``TOMLError`` with ``msg``."""
    h = _header_slot(c)
    if h is None:
        raise TOMLError(msg)
    return h


def _header_comment_get(c: Container) -> str | None:
    if (h := _header_slot(c)) is None:
        # No header means no comment to return, mirroring an explicit
        # section. Setters still raise -- silently dropping a write
        # would be a footgun.
        return None
    return _read_eol_comment(h.eol)


def _header_comment_set(c: Container, value: str | None) -> None:
    h = _require_header_slot(c, "container has no header to attach a comment to")
    if value is None:
        h.eol = _clear_eol_comment(h.eol)
        return
    _validate_comment_str(value, "header_comment")
    h.eol = _write_eol_comment(h.eol, value, c._doc_newline)  # noqa: SLF001


def _header_leading_get(c: Container) -> tuple[str, ...]:
    if (h := _header_slot(c)) is None:
        # See _header_comment_get: no header line, no comments above it.
        return ()
    return _extract_leading_comments(h.leading)


def _header_leading_set(c: Container, value: tuple[str, ...]) -> None:
    h = _require_header_slot(c, "container has no header to attach leading comments to")
    comments = _validate_comment_seq(value, "header_leading_comments")
    h.leading = _set_attached_block(h.leading, comments, c._doc_newline)  # noqa: SLF001


def _header_leading_block_get(c: Container) -> tuple[str | None, ...]:
    if (h := _header_slot(c)) is None:
        # See _header_comment_get: no header line, no block above it.
        return ()
    return _read_leading_block(h)


def _header_leading_block_set(c: Container, value: tuple[str | None, ...]) -> None:
    h = _require_header_slot(c, "container has no header to attach a leading block to")
    block = _validate_comment_entries(value, "header_leading_block", allow_none=True)
    _write_leading_block(c, h, block)


def _read_eol_comment(eol: str) -> str | None:
    """The decoded EOL comment in ``eol``, or ``None`` if it carries none."""
    if "#" not in eol:
        return None
    return _decode_comment(split_line(eol)[1])


def _clear_eol_comment(eol: str) -> str:
    """``eol`` without its comment, and without the gap that introduced it.

    The gap goes too, or removing ``# c`` from ``key = 1  # c`` would
    leave a dangling ``key = 1  `` behind. With no comment to remove
    there is no such gap, and whatever whitespace is there was authored
    deliberately, so leave it alone.
    """
    if "#" not in eol:
        return eol
    return split_line(eol)[2]


def _write_eol_comment(eol: str, text: str, nl: str) -> str:
    """``eol`` with ``text`` as its comment, separator and newline assured."""
    pre, _comment, term = split_line(eol)
    return f"{pre or ' '}{_encode_comment(text)}{term or nl}"


def _doc_preamble_get(doc: Document) -> tuple[str, ...]:
    # Lines without a comment -- including the trailing blank-line
    # separator before the first slot -- drop out on their own.
    return _lines_to_comments(doc._preamble)  # noqa: SLF001


def _doc_preamble_set(doc: Document, value: tuple[str, ...]) -> None:
    comments = _validate_comment_seq(value, "preamble")
    nl = doc._newline  # noqa: SLF001
    if not comments:
        doc._preamble = ""  # noqa: SLF001
        return
    rendered = _render_comment_lines(comments, nl)
    # Append a blank-line separator before the first slot.
    if doc._head is not None:  # noqa: SLF001
        rendered += nl
    doc._preamble = rendered  # noqa: SLF001


def _doc_epilogue_get(doc: Document) -> tuple[str | None, ...]:
    # Full fidelity: comment lines decode to text, blank lines become None
    # (including a blank that separates the epilogue from the last slot).
    lines = split_lines(doc._trailing)  # noqa: SLF001
    return tuple(_line_to_comment(line) for line in lines)


def _doc_epilogue_set(doc: Document, value: tuple[str | None, ...]) -> None:
    block = _validate_comment_entries(value, "epilogue", allow_none=True)
    if doc._head is None and block:  # noqa: SLF001
        msg = "cannot set epilogue: document has no structural content"
        raise TOMLError(msg)
    nl = doc._newline  # noqa: SLF001
    if block:
        # The epilogue starts on its own line, which a file whose last
        # line has no terminator does not provide.
        tail = doc._tail  # noqa: SLF001
        assert tail is not None
        ensure_terminator(tail, nl)
    doc._trailing = _render_comment_lines(block, nl)  # noqa: SLF001


__all__ = ["EolCommentView", "LeadingBlockView", "LeadingCommentView"]
