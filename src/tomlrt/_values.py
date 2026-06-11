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
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, Generic, TypeVar

if sys.version_info >= (3, 12):
    from typing import override
else:  # pragma: no cover -- backport for Python < 3.12
    from typing_extensions import override

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


@dataclass(slots=True, eq=False)
class StringValue:
    lexeme: str  # including quotes
    value: str

    def render(self) -> str:
        return self.lexeme


@dataclass(slots=True, eq=False)
class IntegerValue:
    lexeme: str
    value: int

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

    def render(self) -> str:
        return self.lexeme


# ---------------------------------------------------------------------------
# Dotted keys
# ---------------------------------------------------------------------------


@dataclass(slots=True, eq=False)
class KeyPart:
    """A single dotted-key component."""

    raw: str  # source representation including any surrounding quotes
    value: str  # the decoded key string

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
        return KeyPart(raw=name, value=name)
    return KeyPart(raw=quote_basic_key(name), value=name)


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


# ---------------------------------------------------------------------------
# Comma-separated values (inline arrays + inline tables)
# ---------------------------------------------------------------------------


@dataclass(slots=True, eq=False)
class CommaItem:
    """One slot inside a comma-separated value.

    Layout: ``leading value trailing [comma post_comma_trivia]``.
    Shared base of the two concrete leaves: `ArrayItem` (bare value,
    used by `ArrayValue`) and `InlineTableEntry` (with a ``key = ``
    prefix, used by `InlineTableValue`). The two leaves are siblings,
    not parent/child — annotate with `CommaItem` when working
    polymorphically over both, and with the concrete leaf when the
    caller is flavour-specific.
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
class ArrayItem(CommaItem):
    """One bare-value slot inside an inline array.

    Adds no fields over `CommaItem`; exists as a distinct concrete
    leaf so `ArrayValue.items: list[ArrayItem]` narrows away
    `InlineTableEntry` at the type level.
    """


@dataclass(slots=True, eq=False)
class InlineTableEntry(CommaItem):
    """One ``key = value`` slot inside an inline table.

    Extends `CommaItem` with the key prefix. The shared trivia +
    comma machinery lives on the base; only the key-prefix fields
    and the keyed-form `render` are added here.
    """

    key_parts: list[KeyPart] = field(kw_only=True)
    key_seps: list[str] = field(kw_only=True)  # len = len(key_parts) - 1
    pre_eq: str = field(kw_only=True)
    post_eq: str = field(kw_only=True)
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

    @override
    def render(self) -> str:
        out = (
            f"{self.leading.render()}{self.render_key()}"
            f"{self.pre_eq}={self.post_eq}"
            f"{self.value.render()}{self.trailing.render()}"
        )
        if self.has_comma:
            out += f",{self.post_comma_trivia.render()}"
        return out


# Any per-item shape — bare `ArrayItem` or keyed `InlineTableEntry`.
# `CommaItem` is the real base class of both; use it at call sites
# that are polymorphic over the two flavours, and use the concrete
# leaf where the code is flavour-specific.


_ItemT = TypeVar("_ItemT", bound=CommaItem)


@dataclass(slots=True, eq=False)
class CommaValue(Generic[_ItemT]):
    """Shared backbone of `ArrayValue` and `InlineTableValue`.

    Trivia ownership (canonical model):
      - ``header_trivia`` owns the gap immediately after the opening
        bracket and before the first item — bracket pad / leading
        newline / indent / interior comments above item 0.
      - ``items[0].leading`` is always empty.
      - ``items[k].leading`` (k >= 1) owns the entire physical gap
        before items[k]: structural newline + indent + above-item
        comment block + indent before the value.
      - ``items[k].post_comma_trivia`` carries only the row-attached
        EOL section: same-line whitespace + EOL comment + the
        terminating newline of that comment row.  Empty if no EOL
        comment.
      - ``final_trivia`` owns the gap before the closing bracket
        (bracket pad / trailing newline). For an empty value this is
        the only place interior trivia can live.

    Concrete subclasses bind ``_ItemT`` to the item flavour they hold
    and set ``_open`` / ``_close`` ClassVars to the appropriate
    brackets.
    """

    items: list[_ItemT] = field(default_factory=list)
    header_trivia: Trivia = field(default_factory=Trivia)
    final_trivia: Trivia = field(default_factory=Trivia)

    _open: ClassVar[str] = ""
    _close: ClassVar[str] = ""

    # Canonical inner bracket padding for the single-line form: one
    # space for inline tables (``{ a = 1 }``), none for inline arrays
    # (``[1, 2]``). An empty value carries no padding regardless.
    _single_line_pad: ClassVar[str] = ""

    def render(self) -> str:
        body = "".join([it.render() for it in self.items])
        return (
            f"{self._open}{self.header_trivia.render()}"
            f"{body}{self.final_trivia.render()}{self._close}"
        )


@dataclass(slots=True, eq=False)
class ArrayValue(CommaValue[ArrayItem]):
    """Inline array literal (``[ ... ]``)."""

    _open: ClassVar[str] = "["
    _close: ClassVar[str] = "]"


@dataclass(slots=True, eq=False)
class InlineTableValue(CommaValue[InlineTableEntry]):
    """Inline table literal (``{ ... }``)."""

    _open: ClassVar[str] = "{"
    _close: ClassVar[str] = "}"
    _single_line_pad: ClassVar[str] = " "


Value = (
    StringValue
    | IntegerValue
    | FloatValue
    | BoolValue
    | DateTimeValue
    | ArrayValue
    | InlineTableValue
)


def item_breaks_before_comma(item: CommaItem) -> bool:
    """True iff the row break (and any EOL comment) precedes the comma.

    The single canonical test for the comma-first / leading-comma
    layout.
    """
    return item.has_comma and trivia_has_newline(item.trailing)


def item_eol_channel(item: CommaItem) -> Trivia:
    """The trivia run that owns the item's row-attached EOL section.

    ``trailing`` for a comma-first or comma-less item, else
    ``post_comma_trivia``. Picking the channel here lets callers read,
    write, and normalise the EOL without branching on the comma.
    """
    if item_breaks_before_comma(item):
        return item.trailing
    return item.post_comma_trivia if item.has_comma else item.trailing


def inter_item_separator(items: Sequence[CommaItem]) -> Trivia:
    """Structural-pad portion of ``items[1].leading``; ``" "`` if ``len < 2``.

    Excludes any above-item comment block, which belongs to the item's
    leading rather than to the separator.
    """
    if len(items) >= 2:
        head, _above, tail = split_item_above(items[1].leading)
        return Trivia([*head.pieces, *tail.pieces])
    return Trivia([WhitespaceNode(text=" ")])


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
    for it in v.items:
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
    if not isinstance(v, CommaValue):
        return
    retarget_trivia_newlines(v.header_trivia, target)
    retarget_trivia_newlines(v.final_trivia, target)
    for it in v.items:
        retarget_trivia_newlines(it.leading, target)
        retarget_trivia_newlines(it.trailing, target)
        retarget_trivia_newlines(it.post_comma_trivia, target)
        retarget_value_newlines(it.value, target)


__all__ = [
    "ArrayItem",
    "ArrayValue",
    "BoolValue",
    "CommaItem",
    "CommaValue",
    "DateTimeValue",
    "FloatValue",
    "InlineTableEntry",
    "InlineTableValue",
    "IntegerValue",
    "KeyPart",
    "StringValue",
    "Value",
    "inter_item_separator",
    "item_breaks_before_comma",
    "item_eol_channel",
    "retarget_value_newlines",
    "value_is_multiline",
]
