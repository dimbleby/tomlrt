"""Value types — the inline-value layer.

Every TOML value (scalar, inline array, inline table) is one of
these. They are pure data: no document/slot stream awareness.

Each scalar carries its source ``lexeme`` together with the
decoded Python value so that re-emission is byte-exact.

`ArrayValue` and `InlineTableValue` carry their full internal
layout (every separator, comment, and whitespace run) for the
same reason.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from tomlrt._trivia import (
    Trivia,
    WhitespaceNode,
    retarget_trivia_newlines,
    split_item_above,
    trivia_has_newline,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import date, datetime, time


StringStyle = Literal["basic", "literal", "ml-basic", "ml-literal"]
IntStyle = Literal["dec", "hex", "oct", "bin"]
DateLikeKind = Literal[
    "offset-datetime",
    "local-datetime",
    "local-date",
    "local-time",
]


@dataclass(slots=True, eq=False)
class StringValue:
    lexeme: str  # including quotes
    value: str
    style: StringStyle

    def render(self) -> str:
        return self.lexeme


@dataclass(slots=True, eq=False)
class IntegerValue:
    lexeme: str
    value: int
    style: IntStyle

    def render(self) -> str:
        return self.lexeme


@dataclass(slots=True, eq=False)
class FloatValue:
    lexeme: str
    value: float

    def render(self) -> str:
        return self.lexeme


@dataclass(slots=True, eq=False)
class BoolValue:
    lexeme: str  # "true" or "false"
    value: bool

    def render(self) -> str:
        return self.lexeme


@dataclass(slots=True, eq=False)
class DateTimeValue:
    lexeme: str
    value: datetime | date | time
    kind: DateLikeKind

    def render(self) -> str:
        return self.lexeme


# ---------------------------------------------------------------------------
# Inline arrays
# ---------------------------------------------------------------------------


@dataclass(slots=True, eq=False)
class ArrayItem:
    """One slot inside an inline array.

    Layout: ``leading value trailing [comma post_comma_trivia]``.
    """

    leading: Trivia
    value: Value
    trailing: Trivia
    has_comma: bool
    post_comma_trivia: Trivia

    def render(self) -> str:
        out = f"{self.leading.render()}{self.value.render()}{self.trailing.render()}"
        if self.has_comma:
            out += f",{self.post_comma_trivia.render()}"
        return out


@dataclass(slots=True, eq=False)
class ArrayValue:
    """Inline array literal (``[ ... ]``).

    Trivia ownership (canonical model):
      - ``header_trivia`` owns the gap immediately after ``[`` and
        before the first item — bracket pad / leading newline / indent
        / interior comments above item 0.
      - ``items[0].leading`` is always empty.
      - ``items[k].leading`` (k >= 1) owns the entire physical gap
        before items[k]: structural newline + indent + above-item
        comment block + indent before the value.
      - ``items[k].post_comma_trivia`` carries only the row-attached
        EOL section: same-line whitespace + EOL comment + the
        terminating newline of that comment row.  Empty if no EOL
        comment.
      - ``final_trivia`` owns the gap before ``]`` (bracket pad /
        trailing newline). For an empty array this is the only place
        interior trivia can live.
    """

    items: list[ArrayItem] = field(default_factory=list)
    header_trivia: Trivia = field(default_factory=Trivia)
    final_trivia: Trivia = field(default_factory=Trivia)

    def render(self) -> str:
        body = "".join([item.render() for item in self.items])
        return f"[{self.header_trivia.render()}{body}{self.final_trivia.render()}]"


# ---------------------------------------------------------------------------
# Inline tables
# ---------------------------------------------------------------------------


@dataclass(slots=True, eq=False)
class KeyPart:
    """A single dotted-key component."""

    raw: str  # source representation including any surrounding quotes
    value: str  # the decoded key string
    kind: Literal["bare", "basic", "literal"]

    def render(self) -> str:
        return self.raw


def quote_basic_key(s: str) -> str:
    """Encode ``s`` as a basic-quoted TOML key (escaping where required)."""
    out = ['"']
    for ch in s:
        c = ord(ch)
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif c < 0x20 or c == 0x7F:
            out.append(f"\\u{c:04X}")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


_RE_BARE_KEY_FULL = re.compile(r"\A[A-Za-z0-9_\-]+\Z")


def make_keypart(name: str) -> KeyPart:
    """Build a ``KeyPart`` for ``name``, choosing bare vs basic-quoted."""
    if _RE_BARE_KEY_FULL.match(name):
        return KeyPart(raw=name, value=name, kind="bare")
    return KeyPart(raw=quote_basic_key(name), value=name, kind="basic")


def make_keyparts(path: tuple[str, ...]) -> list[KeyPart]:
    """Build a list of ``KeyPart``s for each segment of ``path``."""
    return [make_keypart(p) for p in path]


def render_dotted(parts: list[KeyPart], seps: list[str]) -> str:
    """Render a dotted key as ``part0 sep0 part1 sep1 ...``.

    ``seps`` has length ``len(parts) - 1``; each entry is the literal
    whitespace + ``.`` between the surrounding parts (e.g. ``" . "``).
    """
    if len(parts) == 1:
        return parts[0].render()
    out: list[str] = []
    for i, p in enumerate(parts):
        if i:
            out.append(seps[i - 1])
        out.append(p.render())
    return "".join(out)


@dataclass(slots=True, eq=False)
class InlineTableEntry:
    """One ``key = value`` slot inside an inline table."""

    leading: Trivia
    key_parts: list[KeyPart]
    key_seps: list[str]  # length = len(key_parts) - 1
    pre_eq: str
    post_eq: str
    value: Value
    trailing: Trivia
    has_comma: bool
    post_comma_trivia: Trivia

    key_path: tuple[str, ...] = field(kw_only=True)
    """Decoded dotted-key path.

    Set by every construction site (parser, mutation, synthesis). The
    parser passes the tuple it already built for inline-key conflict
    detection; mutation paths pass the path they were given. Read by
    ``_validator._register_inline_table`` and
    ``_build._decode_inline_table`` (also reached via cross-document
    clone of inline-table entries).
    """

    def render_key(self) -> str:
        return render_dotted(self.key_parts, self.key_seps)

    def render(self) -> str:
        out = (
            f"{self.leading.render()}{self.render_key()}"
            f"{self.pre_eq}={self.post_eq}"
            f"{self.value.render()}{self.trailing.render()}"
        )
        if self.has_comma:
            out += f",{self.post_comma_trivia.render()}"
        return out


@dataclass(slots=True, eq=False)
class InlineTableValue:
    """Inline table literal (``{ ... }``).

    Trivia ownership matches :class:`ArrayValue` — see that docstring
    for the canonical model.
    """

    entries: list[InlineTableEntry] = field(default_factory=list)
    header_trivia: Trivia = field(default_factory=Trivia)
    final_trivia: Trivia = field(default_factory=Trivia)

    def render(self) -> str:
        body = "".join([e.render() for e in self.entries])
        return f"{{{self.header_trivia.render()}{body}{self.final_trivia.render()}}}"


Value = (
    StringValue
    | IntegerValue
    | FloatValue
    | BoolValue
    | DateTimeValue
    | ArrayValue
    | InlineTableValue
)

# Per-item shape shared by ArrayItem and InlineTableEntry. The two
# classes stay distinct — see AGENTS.md for the divergence — but
# helpers that walk their per-item trivia uniformly take this union.
CommaItem = ArrayItem | InlineTableEntry


def inter_item_separator(items: Sequence[CommaItem]) -> Trivia:
    """Structural-pad portion of ``items[1].leading``; ``" "`` if ``len < 2``.

    Excludes any above-item comment block, which belongs to the item's
    leading rather than to the separator.
    """
    if len(items) >= 2:
        head, _above, tail = split_item_above(items[1].leading)
        return Trivia([*head.pieces, *tail.pieces])
    return Trivia([WhitespaceNode(text=" ")])


def entries_of(v: ArrayValue | InlineTableValue) -> Sequence[CommaItem]:
    """Common accessor for the item list of an inline array / table.

    ``ArrayValue.items`` and ``InlineTableValue.entries`` carry the
    same per-row shape (`CommaItem`) under different field names —
    this accessor lets shared helpers iterate either without
    isinstance-dispatching at every call site.
    """
    return v.items if isinstance(v, ArrayValue) else v.entries


def value_is_multiline(v: ArrayValue | InlineTableValue) -> bool:
    """Shape detection: any structural ``NewlineNode`` inside means multi-line.

    Inspects every trivia region that can hold a structural newline
    under the canonical model (``header_trivia`` / ``final_trivia`` /
    per-item ``leading`` / ``trailing`` / ``post_comma_trivia``).
    Sufficient and correct independent of which subset of those
    regions actually carries the row breaks in a given layout.
    """
    if trivia_has_newline(v.header_trivia) or trivia_has_newline(v.final_trivia):
        return True
    for it in entries_of(v):
        if (
            trivia_has_newline(it.leading)
            or trivia_has_newline(it.post_comma_trivia)
            or trivia_has_newline(it.trailing)
        ):
            return True
    return False


def retarget_value_newlines(v: Value, target: str) -> None:
    """Recursively rewrite every ``NewlineNode.text`` under ``v`` to ``target``.

    Walks the trivia inside ``ArrayValue`` / ``InlineTableValue``
    containers (header / final / per-item) and recurses into nested
    values. Scalar values have no trivia and are no-ops.

    Multi-line string content lives in ``StringValue.lexeme``, NOT
    in ``NewlineNode``, so this walk preserves any literal CR/LF
    bytes inside string values.
    """
    items: list[ArrayItem] | list[InlineTableEntry]
    if isinstance(v, ArrayValue):
        items = v.items
    elif isinstance(v, InlineTableValue):
        items = v.entries
    else:
        return

    retarget_trivia_newlines(v.header_trivia, target)
    retarget_trivia_newlines(v.final_trivia, target)
    for it in items:
        retarget_trivia_newlines(it.leading, target)
        retarget_trivia_newlines(it.trailing, target)
        retarget_trivia_newlines(it.post_comma_trivia, target)
        retarget_value_newlines(it.value, target)


__all__ = [
    "ArrayItem",
    "ArrayValue",
    "BoolValue",
    "CommaItem",
    "DateLikeKind",
    "DateTimeValue",
    "FloatValue",
    "InlineTableEntry",
    "InlineTableValue",
    "IntStyle",
    "IntegerValue",
    "KeyPart",
    "StringStyle",
    "StringValue",
    "Value",
    "entries_of",
    "inter_item_separator",
    "retarget_value_newlines",
    "value_is_multiline",
]
