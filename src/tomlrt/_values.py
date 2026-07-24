"""Represent byte-exact inline TOML values.

Values are pure data with no slot-stream awareness. Scalars carry their
source ``lexeme``; arrays and inline tables carry every separator,
comment, and whitespace run needed for exact re-emission.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar

if sys.version_info >= (3, 12):
    from typing import override
else:  # pragma: no cover -- backport for Python < 3.12
    from typing_extensions import override

from tomlrt._trivia import (
    Trivia,
    WhitespaceNode,
    has_comment,
    has_newline,
    retarget_trivia_newlines,
    split_eol_section,
    split_item_above,
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
    Shared base of sibling leaves `ArrayItem` and `InlineTableEntry`;
    use `CommaItem` only at polymorphic call sites.
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
    """Represent one bare-value slot inside an inline array."""


@dataclass(slots=True, eq=False)
class InlineTableEntry(CommaItem):
    """One ``key = value`` slot inside an inline table.

    The shared trivia/comma machinery lives on `CommaItem`; this leaf
    adds only the key-prefix fields and keyed rendering.
    """

    key_parts: list[KeyPart] = field(kw_only=True)
    key_seps: list[str] = field(kw_only=True)  # len = len(key_parts) - 1
    pre_eq: str = field(kw_only=True)
    post_eq: str = field(kw_only=True)
    key_path: tuple[str, ...] = field(kw_only=True)
    """Decoded dotted-key path.

    Set by every construction site and read by inline-table validation,
    decoding, and cross-document cloning.
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


_ItemT = TypeVar("_ItemT", bound=CommaItem)


@dataclass(slots=True, eq=False)
class CommaValue(Generic[_ItemT]):
    """Shared backbone of `ArrayValue` and `InlineTableValue`.

    Canonical trivia ownership:
      - ``header_trivia`` owns the gap after the opening bracket and
        before item 0: bracket pad, leading newline, indent, comments.
      - ``items[0].leading`` is always empty.
      - ``items[k].leading`` (k >= 1) owns the physical gap before
        item k, including structural newline, indent, and above-block.
      - ``items[k].post_comma_trivia`` carries only the row-attached
        EOL section: same-line whitespace, comment, and row newline.
      - ``final_trivia`` owns the gap before the closing bracket
        and is the only interior owner for an empty value.

    Concrete subclasses bind ``_ItemT`` and set the bracket ClassVars.
    """

    items: list[_ItemT] = field(default_factory=list)
    header_trivia: Trivia = field(default_factory=Trivia)
    final_trivia: Trivia = field(default_factory=Trivia)

    # Memoised `is_multiline()` result; None means "not computed".
    # Append/insert/sort/reorder preserve multi-line shape, so the hot
    # build path leaves it warm; only item removal and the explicit
    # single<->multi toggle can flip it, and those invalidate it.
    _ml_cache: bool | None = field(default=None, init=False, compare=False, repr=False)

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

    def is_multiline(self) -> bool:
        """Whether this value renders across multiple physical lines.

        Memoised: the scan is O(n) for a single-line value, but the answer
        only flips on item removal or an explicit single<->multi toggle (see
        ``reset_multiline_cache``), so building a value with repeated appends
        stays linear overall.
        """
        if self._ml_cache is None:
            self._ml_cache = _scan_multiline(self)
        return self._ml_cache

    def reset_multiline_cache(self) -> None:
        """Drop the memoised ``is_multiline`` result so it recomputes.

        Call after any mutation that can change the multi-line shape: item
        removal (the removed item may carry the sole newline, or emptying may
        collapse the bracket pads) and the explicit single<->multi toggle.
        Append / insert / sort / reorder preserve the shape and deliberately
        do *not* reset it, keeping the build path warm.
        """
        self._ml_cache = None


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
    """Return whether the row break and any EOL comment precede the comma."""
    return item.has_comma and has_newline(item.trailing.pieces)


def item_eol_channel(item: CommaItem) -> Trivia:
    """The trivia run that owns the item's row-attached EOL section.

    A comma-first item normally uses ``trailing``. If its pre-comma break is
    structural while an EOL comment follows the comma, the post-comma channel
    owns that EOL instead. Picking the channel here lets callers read, write,
    and normalise the EOL without rediscovering that distinction.
    """
    if item_breaks_before_comma(item):
        trailing_eol, _rest = split_eol_section(item.trailing)
        if trailing_eol.pieces or not has_comment(item.post_comma_trivia.pieces):
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


def _scan_multiline(v: CommaValue[Any]) -> bool:
    """Uncached scan: inspect every trivia region that can carry a row break."""
    if has_newline(v.header_trivia.pieces) or has_newline(v.final_trivia.pieces):
        return True
    for it in v.items:
        if (
            has_newline(it.leading.pieces)
            or has_newline(it.post_comma_trivia.pieces)
            or has_newline(it.trailing.pieces)
        ):
            return True
    return False


def value_has_any_comment(v: Value) -> bool:
    """Whether any comment appears anywhere within ``v`` (recursively)."""
    if not isinstance(v, CommaValue):
        return False
    if has_comment(v.header_trivia.pieces) or has_comment(v.final_trivia.pieces):
        return True
    return any(item_has_any_comment(it) for it in v.items)


def item_has_any_comment(item: CommaItem) -> bool:
    """Whether ``item`` carries a comment in its trivia or nested value."""
    if (
        has_comment(item.leading.pieces)
        or has_comment(item.trailing.pieces)
        or has_comment(item.post_comma_trivia.pieces)
    ):
        return True
    return value_has_any_comment(item.value)


def retarget_value_newlines(v: Value, target: str) -> None:
    """Recursively rewrite every ``NewlineNode.text`` under ``v`` to ``target``.

    Scalar values have no trivia. Multi-line string content lives in
    ``StringValue.lexeme``, not ``NewlineNode``, so literal CR/LF bytes
    inside strings are preserved.
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
    "item_has_any_comment",
    "retarget_value_newlines",
]
