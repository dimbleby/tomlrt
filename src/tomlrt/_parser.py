"""Hand-written recursive-descent parser.

Walks source via `_Scanner`, building the physical slot run and
validating declarations against the logical views as they are opened.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tomlrt._build import _apply_header, _apply_kv, _finish_document
from tomlrt._container import Document
from tomlrt._scanner import _Scanner
from tomlrt._slots import AoTEntry, KVSlot, StructuralHeaderSlot, stitch_run
from tomlrt._trivia import leading_has_blank_line, split_eol_section
from tomlrt._validator import check_inline_key_conflict
from tomlrt._values import (
    ArrayItem,
    ArrayValue,
    InlineTableEntry,
    InlineTableValue,
)

if TYPE_CHECKING:
    from tomlrt._container import Container
    from tomlrt._slots import Slot
    from tomlrt._values import CommaItem, Value


class _Parser:
    __slots__ = ("_doc", "_error", "_host", "_sc", "_value_depth")

    _MAX_VALUE_DEPTH = 100

    def __init__(self, src: str) -> None:
        self._sc = _Scanner(src)
        self._error = self._sc.error
        self._doc = Document()
        self._host: Container = self._doc
        self._value_depth = 0

    def parse(self) -> Document:
        slots: list[Slot] = []
        trailing = ""
        sc = self._sc
        src = sc.src
        end = sc.end
        # The document's most recent inter-section gap is the one that
        # decides ``section_blank_separated``, so keep the gap itself
        # and classify it once at the end rather than at every header.
        latest_section_gap: str | None = None
        seen_header = False

        while sc.pos < end:
            leading = sc.scan_trivia()
            pos = sc.pos
            if pos >= end:
                trailing = leading
                break

            ch = src[pos]
            slot: Slot
            if ch == "[":
                slot = self._parse_header(leading)
                if seen_header:
                    latest_section_gap = leading
                seen_header = True
            elif ch == "\r":
                msg = "stray carriage return"
                raise sc.error(msg)
            else:
                slot = self._parse_key_value(leading)
            slots.append(slot)

        stitch_run(None, slots, None)
        _finish_document(
            self._doc,
            slots,
            trailing=trailing,
            newline=sc.detected_newline(),
            section_blank_separated=(
                latest_section_gap is None or leading_has_blank_line(latest_section_gap)
            ),
        )
        return self._doc

    def _parse_header(self, leading: str) -> StructuralHeaderSlot:
        """Parse a ``[a.b]`` / ``[[a.b]]`` header.

        Precondition: cursor is at ``[``.
        """
        sc = self._sc
        src = sc.src
        header_at = sc.pos
        new_entry: AoTEntry | None
        if src.startswith("[[", sc.pos):
            sc.pos += 2
            new_entry = AoTEntry()
            closer, what = "]]", "array-of-tables"
        else:
            sc.pos += 1
            new_entry = None
            closer, what = "]", "table"

        inner_pre = sc.scan_inline_ws_text()
        key_parts, key_seps, inner_post, path = sc.scan_key()

        if not src.startswith(closer, sc.pos):
            msg = f"expected {closer!r} to close {what} header"
            raise sc.error(msg)
        sc.pos += len(closer)

        eol = sc.scan_eol()
        slot = StructuralHeaderSlot(
            leading,
            None,
            eol,
            key_parts,
            key_seps,
            path,
            inner_pre,
            inner_post,
            new_entry,
            synthetic=False,
        )
        self._host = _apply_header(self._doc, slot, self._error, header_at)
        return slot

    def _parse_key_value(self, leading: str) -> KVSlot:
        """Parse a ``key = value`` line.

        Precondition: cursor is at the start of the key.
        """
        sc = self._sc
        key_at = sc.pos
        key_parts, key_seps, pre_eq, key_path = sc.scan_key()
        src = sc.src
        pos = sc.pos
        if pos >= sc.end or src[pos] != "=":
            ch = src[pos] if pos < sc.end else ""
            msg = f"expected '=' after key, got {ch!r}"
            raise sc.error(msg)
        sc.pos = pos + 1
        post_eq = sc.scan_inline_ws_text()
        value = self._parse_value()
        eol = sc.scan_eol()

        host = self._host
        slot = KVSlot(
            leading,
            host._owner_aot_entry,  # noqa: SLF001
            eol,
            host._path,  # noqa: SLF001
            key_parts,
            key_seps,
            key_path,
            pre_eq,
            post_eq,
            value,
        )
        _apply_kv(slot, host, self._error, key_at)
        return slot

    def _parse_value(self) -> Value:
        sc = self._sc
        pos = sc.pos
        ch = sc.src[pos] if pos < sc.end else ""
        if ch in {'"', "'"}:
            return sc.scan_string()
        if ch in ("[", "{"):
            if self._value_depth >= self._MAX_VALUE_DEPTH:
                msg = f"value nesting exceeds maximum depth ({self._MAX_VALUE_DEPTH})"
                raise sc.error(msg)
            self._value_depth += 1
            try:
                if ch == "[":
                    return self._parse_array()
                return self._parse_inline_table()
            finally:
                self._value_depth -= 1
        return sc.scan_value_atom()

    def _finish_gap(self, item: CommaItem, closer: str) -> None:
        """Read one outgoing gap, leaving the closing bracket for the caller."""
        sc = self._sc
        src, end = sc.src, sc.end
        ch = src[sc.pos] if sc.pos < end else ""
        if ch != "," and ch != closer:
            item.before = sc.scan_trivia()
            ch = src[sc.pos] if sc.pos < end else ""
        if ch == ",":
            sc.pos += 1
            item.has_comma = True
            item.after, item.following = split_eol_section(sc.scan_trivia())
        elif ch == closer:
            item.before, item.following = split_eol_section(item.before)
        else:
            kind = "array" if closer == "]" else "inline table"
            msg = f"expected ',' or {closer!r} in {kind}, got {ch!r}"
            raise sc.error(msg)

    def _parse_array(self) -> ArrayValue:
        sc = self._sc
        src, end = sc.src, sc.end
        sc.pos += 1
        node = ArrayValue()
        node.opening = sc.scan_trivia()
        while sc.pos >= end or src[sc.pos] != "]":
            item = ArrayItem(self._parse_value())
            self._finish_gap(item, "]")
            node.items.append(item)
        sc.pos += 1
        return node

    def _parse_inline_table(self) -> InlineTableValue:
        sc = self._sc
        src, end = sc.src, sc.end
        sc.pos += 1
        node = InlineTableValue()
        node.opening = sc.scan_trivia()
        seen_prefixes: set[tuple[str, ...]] = set()
        while sc.pos >= end or src[sc.pos] != "}":
            key_at = sc.pos
            key_parts, key_seps, pre_eq, key_path = sc.scan_key()
            check_inline_key_conflict(
                key_path,
                node._key_index,  # noqa: SLF001
                seen_prefixes,
                self._error,
                at=key_at,
            )
            ch = src[sc.pos] if sc.pos < end else ""
            if ch != "=":
                msg = f"expected '=' in inline table, got {ch!r}"
                raise sc.error(msg)
            sc.pos += 1
            post_eq = sc.scan_inline_ws_text()
            entry = InlineTableEntry(
                self._parse_value(), key_parts, key_seps, key_path, pre_eq, post_eq
            )
            self._finish_gap(entry, "}")
            node.items.append(entry)
            node.record_entry(entry)
        sc.pos += 1
        return node


__all__ = ["_Parser"]
