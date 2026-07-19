"""Hand-written recursive-descent parser.

Walks source via `_Scanner` and emits physical slots plus trailing
trivia. Drives `_Validator` for headers, key/value lines, and
inline-table keys.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from tomlrt._scanner import _Scanner
from tomlrt._slots import KVSlot, StructuralHeaderSlot
from tomlrt._trivia import Trivia, leading_has_blank_line, split_eol_section
from tomlrt._validator import _Validator
from tomlrt._values import ArrayItem, ArrayValue, InlineTableEntry, InlineTableValue

_HeaderKind = Literal["table", "aot-entry"]

if TYPE_CHECKING:
    from tomlrt._slots import Slot
    from tomlrt._values import KeyPart, Value


@dataclass
class ParseResult:
    """The output of `_Parser.parse`.

    `slots` is in physical document order. `trailing` is EOF trivia;
    `newline` is the scanner-detected document-wide line ending.
    """

    slots: list[Slot] = field(default_factory=list)
    trailing: Trivia = field(default_factory=Trivia)
    newline: str = "\n"
    prelude: str = ""
    section_blank_separated: bool = True


class _Parser:
    __slots__ = ("_sc", "_validator", "_value_depth")

    _MAX_VALUE_DEPTH = 100

    def __init__(self, src: str) -> None:
        self._sc = _Scanner(src)
        self._validator = _Validator(self._sc.error)
        self._value_depth = 0

    def parse(self) -> ParseResult:
        result = ParseResult()
        sc = self._sc
        src = sc.src
        end = sc.end
        seen_header = False

        # TOML 1.1 permits a leading UTF-8 BOM only at document start.
        # Store it as Document prelude so slot/trivia mutations cannot
        # silently drop or duplicate it.
        if sc.pos < end and src[sc.pos] == "\ufeff":
            result.prelude = "\ufeff"
            sc.pos += 1

        while sc.pos < end:
            leading = sc.scan_doc_trivia()
            pos = sc.pos
            if pos >= end:
                result.trailing.pieces.extend(leading.pieces)
                break

            ch = src[pos]
            slot: Slot
            if ch == "[":
                slot = self._parse_header(leading)
                if seen_header:
                    result.section_blank_separated = leading_has_blank_line(
                        slot.leading
                    )
                seen_header = True
            else:
                slot = self._parse_key_value(leading)
            result.slots.append(slot)

        # Stitch the physical slot list.
        prev: Slot | None = None
        for slot in result.slots:
            slot._prev = prev  # noqa: SLF001
            if prev is not None:
                prev._next = slot  # noqa: SLF001
            prev = slot

        result.newline = sc.detected_newline()
        return result

    def _parse_header(self, leading: Trivia) -> StructuralHeaderSlot:
        """Parse a ``[a.b]`` / ``[[a.b]]`` header.

        Precondition: cursor is at ``[``.
        """
        sc = self._sc
        kind: _HeaderKind
        if sc.starts_with("[["):
            sc.advance(2)
            kind = "aot-entry"
        else:
            sc.advance(1)
            kind = "table"

        inner_pre = sc.scan_inline_ws_text()
        key_parts, key_seps, inner_post = self._parse_key()

        if kind == "aot-entry":
            if not sc.starts_with("]]"):
                msg = "expected ']]' to close array-of-tables header"
                raise sc.error(msg)
            sc.advance(2)
        else:
            if sc.peek() != "]":
                msg = "expected ']' to close table header"
                raise sc.error(msg)
            sc.advance(1)

        eol = sc.scan_eol()
        path = tuple([p.value for p in key_parts])
        new_entry = self._validator.enter_header(path, kind, at=sc.pos)
        owner = self._validator.current_owner_aot_entry()

        slot = StructuralHeaderSlot(
            leading=leading,
            path=path,
            key_parts=key_parts,
            key_seps=key_seps,
            inner_pre=inner_pre,
            inner_post=inner_post,
            eol=eol,
            owner_aot_entry=owner,
            entry=new_entry,
            synthetic=False,
        )
        if owner is not None:
            owner.entry_slots.append(slot)
        return slot

    def _parse_key(self) -> tuple[list[KeyPart], list[str], str]:
        """Parse a dotted key.

        ``trailing_ws`` is consumed after the last key part and can be
        used directly as ``pre_eq`` / ``inner_post``.
        """
        sc = self._sc
        parts: list[KeyPart] = [sc.scan_key_part()]
        seps: list[str] = []
        while True:
            text, is_sep = sc.scan_key_separator()
            if not is_sep:
                return parts, seps, text
            seps.append(text)
            parts.append(sc.scan_key_part())

    def _parse_key_value(self, leading: Trivia) -> KVSlot:
        sc = self._sc
        key_parts, key_seps, pre_eq = self._parse_key()
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

        key_path = tuple([p.value for p in key_parts])
        self._validator.record_keyvalue(key_path, value, at=sc.pos)
        host_path = self._validator.current_section()
        owner = self._validator.current_owner_aot_entry()
        slot = KVSlot(
            leading=leading,
            host_path=host_path,
            key_parts=key_parts,
            key_seps=key_seps,
            pre_eq=pre_eq,
            post_eq=post_eq,
            value=value,
            eol=eol,
            owner_aot_entry=owner,
        )
        if owner is not None:
            owner.entry_slots.append(slot)
        return slot

    def _parse_value(self) -> Value:
        sc = self._sc
        pos = sc.pos
        ch = sc.src[pos] if pos < sc.end else ""
        if ch == '"' or ch == "'":
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

    def _parse_array(self) -> ArrayValue:
        """Parse a ``[...]`` inline array.

        Precondition: cursor is at ``[``.
        """
        sc = self._sc
        src = sc.src
        sc.pos += 1
        node = ArrayValue()
        head = sc.scan_array_trivia()
        end = sc.end
        if sc.pos < end and src[sc.pos] == "]":
            # Empty array: head trivia is the canonical pre-`]` slot.
            node.final_trivia = head
            sc.pos += 1
            return node
        node.header_trivia = head
        items = node.items
        leading = Trivia()  # items[0].leading is always empty
        while True:
            value = self._parse_value()
            trailing = sc.scan_array_trivia()
            ch = src[sc.pos] if sc.pos < end else ""
            if ch == ",":
                sc.pos += 1
                scanned = sc.scan_array_trivia()
                post_comma, next_leading = split_eol_section(scanned)
                items.append(ArrayItem(leading, value, trailing, True, post_comma))  # noqa: FBT003
                if sc.pos < end and src[sc.pos] == "]":
                    # Trailing-comma terminator: rest is bracket pad.
                    node.final_trivia = next_leading
                    sc.pos += 1
                    return node
                leading = next_leading
            elif ch == "]":
                items.append(ArrayItem(leading, value, trailing, False, Trivia()))  # noqa: FBT003
                # No trailing comma: split item EOL from bracket pad.
                eol, rest = split_eol_section(items[-1].trailing)
                items[-1].trailing = eol
                node.final_trivia = rest
                sc.pos += 1
                return node
            else:
                msg = f"expected ',' or ']' in array, got {ch!r}"
                raise sc.error(msg)

    def _parse_inline_table(self) -> InlineTableValue:
        """Parse a ``{...}`` inline table.

        Precondition: cursor is at ``{``.
        """
        sc = self._sc
        sc.advance(1)
        node = InlineTableValue()
        head = sc.scan_array_trivia()
        if sc.peek() == "}":
            node.final_trivia = head
            sc.advance(1)
            return node
        node.header_trivia = head
        leading = Trivia()  # entries[0].leading is always empty
        seen_values: set[tuple[str, ...]] = set()
        seen_prefixes: set[tuple[str, ...]] = set()
        entries = node.items
        while True:
            key_at = sc.pos
            key_parts, key_seps, pre_eq = self._parse_key()
            key_path = tuple([p.value for p in key_parts])
            self._validator.check_inline_key_conflict(
                key_path, seen_values, seen_prefixes, at=key_at
            )
            seen_values.add(key_path)
            if sc.peek() != "=":
                msg = f"expected '=' in inline table, got {sc.peek()!r}"
                raise sc.error(msg)
            sc.advance(1)
            post_eq = sc.scan_inline_ws_text()
            value = self._parse_value()
            trailing = sc.scan_array_trivia()
            ch = sc.peek()
            if ch == ",":
                sc.advance(1)
                scanned = sc.scan_array_trivia()
                post_comma, next_leading = split_eol_section(scanned)
                entries.append(
                    InlineTableEntry(
                        leading=leading,
                        key_parts=key_parts,
                        key_seps=key_seps,
                        pre_eq=pre_eq,
                        post_eq=post_eq,
                        value=value,
                        trailing=trailing,
                        has_comma=True,
                        post_comma_trivia=post_comma,
                        key_path=key_path,
                    )
                )
                if sc.peek() == "}":
                    node.final_trivia = next_leading
                    sc.advance(1)
                    return node
                leading = next_leading
            elif ch == "}":
                entries.append(
                    InlineTableEntry(
                        leading=leading,
                        key_parts=key_parts,
                        key_seps=key_seps,
                        pre_eq=pre_eq,
                        post_eq=post_eq,
                        value=value,
                        trailing=trailing,
                        has_comma=False,
                        post_comma_trivia=Trivia(),
                        key_path=key_path,
                    )
                )
                eol, rest = split_eol_section(entries[-1].trailing)
                entries[-1].trailing = eol
                node.final_trivia = rest
                sc.advance(1)
                return node
            else:
                msg = f"expected ',' or '}}' in inline table, got {ch!r}"
                raise sc.error(msg)


__all__ = ["ParseResult", "_Parser"]
