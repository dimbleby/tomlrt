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
from tomlrt._trivia import split_lines

if TYPE_CHECKING:
    from collections.abc import Iterator

    from tomlrt._container import Container, Document
    from tomlrt._slots import Slot
    from tomlrt._trivia import EolTrivia


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
    """Mapping over Container keys whose direct-KV slot satisfies a predicate.

    Subclasses provide ``_present(slot)`` and item methods; the base
    supplies the shared mapping plumbing.
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
    def _present(self, slot: KVSlot) -> bool:
        """Return whether ``slot`` carries this view's value."""

    @override
    def __repr__(self) -> str:
        return repr(dict(self))

    @override
    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        slot = self._slot(key)
        return slot is not None and self._present(slot)

    @override
    def __iter__(self) -> Iterator[str]:
        seen: set[str] = set()
        for ref in self._c._refs:  # noqa: SLF001
            k = ref.local_key
            if k is None or k in seen:
                continue
            slot = self._slot(k)
            if slot is not None and self._present(slot):
                seen.add(k)
                yield k

    @override
    def __len__(self) -> int:
        return sum(1 for _ in self)


class EolCommentView(_SlotKeyedView[str]):
    __slots__ = ()

    @override
    def _present(self, slot: KVSlot) -> bool:
        return bool(slot.eol.comment)

    @override
    def __getitem__(self, key: str) -> str:
        slot = self._require_slot(key)
        if not slot.eol.comment:
            raise KeyError(key)
        return _decode_comment(slot.eol.comment)

    @override
    def __setitem__(self, key: str, value: str) -> None:
        _require_attached(self._c)
        slot = self._require_slot(key, missing_msg=f"key {key!r} not in container")
        _validate_comment_str(value, "comment")
        _write_eol_comment(slot.eol, value, self._c._doc_newline)  # noqa: SLF001

    @override
    def __delitem__(self, key: str) -> None:
        _require_attached(self._c)
        slot = self._require_slot(key)
        if not slot.eol.comment:
            raise KeyError(key)
        slot.eol.comment = ""
        # Also drop the gap-whitespace that preceded the comment so we
        # don't leave a dangling tail like `key = 1   \n`.
        slot.eol.trailing_ws = ""


def _read_leading_block(c: Container, slot: Slot) -> tuple[str | None, ...]:
    """Decoded leading block of ``slot``.

    Comment lines decode to text, blank/whitespace-only lines become
    ``None``, and the slot's own trailing indent is excluded.
    """
    del c
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


def _slot_has_attached_comments(slot: Slot) -> bool:
    leading = slot.leading
    _above, attached, _indent = _split_attached_block(leading)
    return "#" in attached


class LeadingCommentView(_SlotKeyedView[tuple[str, ...]]):
    __slots__ = ()

    @override
    def _present(self, slot: KVSlot) -> bool:
        return _slot_has_attached_comments(slot)

    @override
    def __getitem__(self, key: str) -> tuple[str, ...]:
        slot = self._require_slot(key)
        if not _slot_has_attached_comments(slot):
            raise KeyError(key)
        return _extract_leading_comments(slot.leading)

    @override
    def __setitem__(self, key: str, value: tuple[str, ...]) -> None:
        _require_attached(self._c)
        slot = self._require_slot(key, missing_msg=f"key {key!r} not in container")
        comments = _validate_comment_seq(value, "leading_comments")
        nl = self._c._doc_newline  # noqa: SLF001
        slot.leading = _set_attached_block(slot.leading, comments, nl)

    @override
    def __delitem__(self, key: str) -> None:
        _require_attached(self._c)
        slot = self._require_slot(key)
        if not _slot_has_attached_comments(slot):
            raise KeyError(key)
        above, _attached, indent = _split_attached_block(slot.leading)
        slot.leading = above + indent


class LeadingBlockView(_SlotKeyedView[tuple[str | None, ...]]):
    """Mapping view over the full leading-trivia block of each direct key.

    Entries are comment strings in source order, with ``None`` for blank
    lines; the slot's column indent is excluded and reapplied on write.

    The document head preamble lives separately, on
    :attr:`Document.preamble`.
    """

    __slots__ = ()

    @override
    def _present(self, slot: KVSlot) -> bool:
        return bool(_read_leading_block(self._c, slot))

    @override
    def __getitem__(self, key: str) -> tuple[str | None, ...]:
        slot = self._require_slot(key)
        block = _read_leading_block(self._c, slot)
        if not block:
            raise KeyError(key)
        return block

    @override
    def __setitem__(self, key: str, value: tuple[str | None, ...]) -> None:
        _require_attached(self._c)
        slot = self._require_slot(key, missing_msg=f"key {key!r} not in container")
        block = _validate_comment_entries(value, "leading_block", allow_none=True)
        _write_leading_block(self._c, slot, block)

    @override
    def __delitem__(self, key: str) -> None:
        _require_attached(self._c)
        slot = self._require_slot(key)
        if not _read_leading_block(self._c, slot):
            raise KeyError(key)
        _write_leading_block(self._c, slot, ())


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
    eol = h.eol
    if not eol.comment:
        return None
    return _decode_comment(eol.comment)


def _header_comment_set(c: Container, value: str | None) -> None:
    h = _require_header_slot(c, "container has no header to attach a comment to")
    eol = h.eol
    if value is None:
        if eol.comment:
            eol.comment = ""
            eol.trailing_ws = ""
        return
    _validate_comment_str(value, "header_comment")
    _write_eol_comment(eol, value, c._doc_newline)  # noqa: SLF001


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
    return _read_leading_block(c, h)


def _header_leading_block_set(c: Container, value: tuple[str | None, ...]) -> None:
    h = _require_header_slot(c, "container has no header to attach a leading block to")
    block = _validate_comment_entries(value, "header_leading_block", allow_none=True)
    _write_leading_block(c, h, block)


def _write_eol_comment(eol: EolTrivia, text: str, nl: str) -> None:
    """Set the EOL comment on ``eol``, ensuring a separator and newline."""
    if not eol.trailing_ws:
        eol.trailing_ws = " "
    eol.comment = _encode_comment(text)
    if not eol.newline:
        eol.newline = nl


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
