"""Expose comment side-channel views over slot trivia.

`Container.comments` maps direct keys to decoded EOL comments;
`Container.leading_comments` maps them to attached leading comment
blocks. A slot is exposed under its leaf key on its immediate parent
(``b.c = 1`` under ``[a]`` appears on ``a["b"]``). Inline tables have
no comment view because TOML forbids comments inside them.
"""

from __future__ import annotations

import sys
from abc import abstractmethod
from collections.abc import Iterable, MutableMapping
from typing import TYPE_CHECKING, TypeVar

if sys.version_info >= (3, 12):
    from typing import override
else:  # pragma: no cover -- backport for Python < 3.12
    from typing_extensions import override

from tomlrt._errors import TOMLError
from tomlrt._kind import _Kind
from tomlrt._slots import KVSlot, StructuralHeaderSlot
from tomlrt._trivia import (
    CommentNode,
    NewlineNode,
    WhitespaceNode,
    has_comment,
    has_newline,
    split_lines,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from tomlrt._container import Container, Document
    from tomlrt._slots import Slot
    from tomlrt._trivia import (
        EolTrivia,
        Trivia,
        TriviaPiece,
    )


def _validate_comment_text(text: str) -> None:
    """Reject a comment value that would not round-trip via the parser."""
    if "\n" in text or "\r" in text:
        msg = "comment must be single-line"
        raise ValueError(msg)
    for ch in text:
        cp = ord(ch)
        # TOML comments allow TAB plus printable Unicode; reject other
        # ASCII controls and DEL.
        if cp == 0x09:
            continue
        if cp < 0x20 or cp == 0x7F:
            msg = f"comment may not contain control character U+{cp:04X}"
            raise ValueError(msg)


def _validate_comment_str(value: object, name: str) -> str:
    """Type-check ``value`` is a str and validate its content; return it."""
    if not isinstance(value, str):
        msg = f"{name} must be str, got {type(value).__name__}"
        raise TypeError(msg)
    _validate_comment_text(value)
    return value


def _decode_comment(raw: str) -> str:
    """Strip the leading ``#`` and one optional space from a raw comment."""
    return raw.removeprefix("#").removeprefix(" ")


def _encode_comment(text: str) -> str:
    """Encode a logical comment into a raw ``# ...`` form."""
    if text == "":
        return "#"
    return f"# {text}"


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
        return slot.eol.comment is not None

    @override
    def __getitem__(self, key: str) -> str:
        slot = self._require_slot(key)
        if slot.eol.comment is None:
            raise KeyError(key)
        return _decode_comment(slot.eol.comment.text)

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
        if slot.eol.comment is None:
            raise KeyError(key)
        slot.eol.comment = None
        # Also drop the gap-whitespace that preceded the comment so we
        # don't leave a dangling tail like `key = 1   \n`.
        if slot.eol.trailing_ws is not None:
            slot.eol.trailing_ws = None


def _split_preamble(
    leading: Trivia,
) -> tuple[list[TriviaPiece], list[TriviaPiece]]:
    """Split the head slot's leading at the first blank line into (preamble, rest).

    The dual of :func:`_split_attached_block` (which cuts at the *last* blank).
    The preamble is the opening contiguous comment run, but only when a blank
    line follows it (and travels with it). If that run is attached straight to
    the first construct, or the leading starts with a blank, there is no
    preamble. ``rest`` keeps the construct's own leading block, blanks and all.
    """
    lines = split_lines(leading.pieces)
    i = 0
    while i < len(lines) and has_comment(lines[i]):
        i += 1
    # A separating blank line is a newline with no comment — not the slot's
    # trailing indent (whitespace, no newline).
    if i == 0 or i >= len(lines) or not has_newline(lines[i]):
        return [], list(leading.pieces)
    preamble = [p for line in lines[: i + 1] for p in line]
    rest = [p for line in lines[i + 1 :] for p in line]
    return preamble, rest


def _split_attached_block(
    leading: Trivia,
) -> tuple[list[list[TriviaPiece]], list[list[TriviaPiece]], list[TriviaPiece]]:
    """Split the leading into (above_blank, attached_comment_lines, slot_indent).

    The attached group is the contiguous comment run immediately before
    the slot. Earlier lines are preamble/archived blocks. ``slot_indent``
    is the trailing whitespace-only, newline-less column offset that
    rebuilders must reapply.
    """
    lines = split_lines(leading.pieces)
    indent: list[TriviaPiece] = []
    if lines and not any(isinstance(p, NewlineNode) for p in lines[-1]):
        last = lines[-1]
        if not any(isinstance(p, CommentNode) for p in last):
            indent = last
            lines = lines[:-1]
    i = len(lines)
    while i > 0 and has_comment(lines[i - 1]):
        i -= 1
    above = lines[:i]
    attached = lines[i:]
    return above, attached, indent


def _read_leading_block(c: Container, slot: Slot) -> tuple[str | None, ...]:
    """Decoded leading block of ``slot``.

    Comment lines decode to text, blank/whitespace-only lines become
    ``None``, and the slot's own trailing indent is excluded.
    """
    del c
    above, attached, _indent = _split_attached_block(slot.leading)
    lines = (*above, *attached)
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
    new_pieces = _render_comment_lines(block, nl, indent)
    new_pieces.extend(indent)
    slot.leading.pieces = new_pieces


def _validate_comment_entry(c: str, name: str) -> None:
    """Validate a single non-``None`` comment-sequence entry's content."""
    if "\n" in c or "\r" in c:
        msg = f"{name} entries must not contain a line terminator"
        raise ValueError(msg)
    _validate_comment_text(c)


def _validate_comment_entries(
    value: object, name: str, *, allow_none: bool
) -> tuple[str | None, ...]:
    """Type-check a comment iterable, optionally allowing ``None`` entries."""
    kind = "comment strings or None" if allow_none else "comment strings"
    if isinstance(value, str) or not isinstance(value, Iterable):
        msg = f"{name} must be an iterable of {kind}"
        raise TypeError(msg)
    out: list[str | None] = []
    for c in value:
        if c is None and allow_none:
            out.append(None)
            continue
        if not isinstance(c, str):
            msg = f"{name} entries must be strings{' or None' if allow_none else ''}"
            raise TypeError(msg)
        _validate_comment_entry(c, name)
        out.append(c)
    return tuple(out)


def _extract_leading_comments(leading: Trivia) -> tuple[str, ...]:
    """Return only the *attached* run of comment-bearing lines.

    Comments separated by a blank line are preamble or archived blocks
    and are excluded.
    """
    _above, attached, _indent = _split_attached_block(leading)
    return _lines_to_comments(attached)


def _slot_has_attached_comments(slot: Slot) -> bool:
    leading = slot.leading
    _above, attached, _indent = _split_attached_block(leading)
    return any(has_comment(line) for line in attached)


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
        _set_attached_block(slot.leading, comments, self._c._doc_newline)  # noqa: SLF001

    @override
    def __delitem__(self, key: str) -> None:
        _require_attached(self._c)
        slot = self._require_slot(key)
        if not _slot_has_attached_comments(slot):
            raise KeyError(key)
        above, _attached, indent = _split_attached_block(slot.leading)
        kept: list[TriviaPiece] = []
        for line in above:
            kept.extend(line)
        kept.extend(indent)
        slot.leading.pieces = kept


class LeadingBlockView(_SlotKeyedView[tuple[str | None, ...]]):
    """Mapping view over the full leading-trivia block of each direct key.

    Entries are source-order comment strings and ``None`` for blank
    lines; the slot's own column indent is excluded and re-applied on
    write. This is the full-fidelity peer of :class:`LeadingCommentView`.

    The document head preamble is disjoint and lives on
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
        # No header line means no header comment; mirror the empty
        # state of an explicit section. Setters still raise; silently
        # dropping a write would be a footgun.
        return None
    eol = h.eol
    if eol.comment is None:
        return None
    return _decode_comment(eol.comment.text)


def _header_comment_set(c: Container, value: str | None) -> None:
    h = _require_header_slot(c, "container has no header to attach a comment to")
    eol = h.eol
    if value is None:
        if eol.comment is not None:
            eol.comment = None
            if eol.trailing_ws is not None and eol.trailing_ws.text.strip(" \t") == "":
                eol.trailing_ws = None
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
    _set_attached_block(h.leading, comments, c._doc_newline)  # noqa: SLF001


def _header_leading_block_get(c: Container) -> tuple[str | None, ...]:
    if (h := _header_slot(c)) is None:
        # See _header_comment_get: no header line, no block above it.
        return ()
    return _read_leading_block(c, h)


def _header_leading_block_set(c: Container, value: tuple[str | None, ...]) -> None:
    h = _require_header_slot(c, "container has no header to attach a leading block to")
    block = _validate_comment_entries(value, "header_leading_block", allow_none=True)
    _write_leading_block(c, h, block)


def _validate_comment_seq(value: object, name: str) -> tuple[str, ...]:
    return tuple(
        c
        for c in _validate_comment_entries(value, name, allow_none=False)
        if c is not None
    )


def _line_to_comment(line: Iterable[TriviaPiece]) -> str | None:
    """Decoded comment text for a line, or ``None`` if it has no comment."""
    for p in line:
        if isinstance(p, CommentNode):
            return _decode_comment(p.text)
    return None


def _lines_to_comments(lines: Iterable[Iterable[TriviaPiece]]) -> tuple[str, ...]:
    """Extract one decoded comment per line that contains a CommentNode."""
    return tuple(c for line in lines if (c := _line_to_comment(line)) is not None)


def _set_attached_block(leading: Trivia, comments: tuple[str, ...], nl: str) -> None:
    """Replace the attached comment block on ``leading`` with ``comments``.

    Preserve preamble/archived blocks and reapply the slot's indent to
    each new comment line and the slot itself.
    """
    above, _attached, indent = _split_attached_block(leading)
    kept: list[TriviaPiece] = []
    for line in above:
        kept.extend(line)
    new_pieces = _render_comment_lines(comments, nl, indent)
    new_pieces.extend(indent)
    leading.pieces = [*kept, *new_pieces]


def _write_eol_comment(eol: EolTrivia, text: str, nl: str) -> None:
    """Set the EOL comment on ``eol``, ensuring a separator and newline."""
    if eol.trailing_ws is None:
        eol.trailing_ws = WhitespaceNode(" ")
    eol.comment = CommentNode(_encode_comment(text))
    if eol.newline is None:
        eol.newline = NewlineNode(nl)


def _render_comment_lines(
    block: tuple[str | None, ...],
    nl: str,
    indent: Sequence[TriviaPiece] = (),
) -> list[TriviaPiece]:
    """Render logical comment lines, using ``None`` for blanks."""
    pieces: list[TriviaPiece] = []
    for entry in block:
        if entry is None:
            pieces.append(NewlineNode(nl))
        else:
            pieces.extend(indent)
            pieces.extend((CommentNode(_encode_comment(entry)), NewlineNode(nl)))
    return pieces


def _doc_preamble_get(doc: Document) -> tuple[str, ...]:
    lines = split_lines(doc._preamble.pieces)  # noqa: SLF001
    # Drop the trailing blank-line separator before the first slot.
    while lines and not has_comment(lines[-1]):
        lines.pop()
    return _lines_to_comments(lines)


def _doc_preamble_set(doc: Document, value: tuple[str, ...]) -> None:
    comments = _validate_comment_seq(value, "preamble")
    nl = doc._newline  # noqa: SLF001
    if not comments:
        doc._preamble.pieces = []  # noqa: SLF001
        return
    pieces = _render_comment_lines(comments, nl)
    # Append a blank-line separator before the first slot.
    if doc._head is not None:  # noqa: SLF001
        pieces.append(NewlineNode(nl))
    doc._preamble.pieces = pieces  # noqa: SLF001


def _doc_epilogue_get(doc: Document) -> tuple[str | None, ...]:
    # Full fidelity: comment lines decode to text, blank lines become None
    # (including a blank that separates the epilogue from the last slot).
    lines = split_lines(doc._trailing.pieces)  # noqa: SLF001
    return tuple(_line_to_comment(line) for line in lines)


def _doc_epilogue_set(doc: Document, value: tuple[str | None, ...]) -> None:
    block = _validate_comment_entries(value, "epilogue", allow_none=True)
    if doc._head is None and block:  # noqa: SLF001
        msg = "cannot set epilogue: document has no structural content"
        raise TOMLError(msg)
    nl = doc._newline  # noqa: SLF001
    doc._trailing.pieces = _render_comment_lines(block, nl)  # noqa: SLF001


__all__ = ["EolCommentView", "LeadingBlockView", "LeadingCommentView"]
