"""Represent byte-exact inline TOML values.

Values are pure data with no slot-stream awareness. Scalars carry their
source ``lexeme``; arrays and inline tables carry every separator,
comment, and whitespace run needed for exact re-emission.
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
    retarget_newlines,
    split_eol_section,
    split_item_above,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import date, datetime, time


@dataclass(slots=True, eq=False)
class ScalarValue:
    """Base of the five TOML scalar leaves.

    Every scalar re-emits verbatim from the ``lexeme`` it was parsed
    from -- quotes, radix prefix, digit separators, and case all
    included -- so the rendering is shared here; each subclass adds
    only its own decoded ``value`` field, keeping the
    ``(lexeme, value)`` positional signature and the precise per-type
    ``value`` annotation.
    """

    lexeme: str

    def render(self) -> str:
        return self.lexeme


@dataclass(slots=True, eq=False)
class StringValue(ScalarValue):
    value: str


@dataclass(slots=True, eq=False)
class IntegerValue(ScalarValue):
    value: int


@dataclass(slots=True, eq=False)
class FloatValue(ScalarValue):
    value: float


@dataclass(slots=True, eq=False)
class BoolValue(ScalarValue):
    value: bool


@dataclass(slots=True, eq=False)
class DateTimeValue(ScalarValue):
    value: datetime | date | time


# ---------------------------------------------------------------------------
# Dotted keys
# ---------------------------------------------------------------------------


@dataclass(slots=True, eq=False)
class KeyPart:
    """A single dotted-key component.

    Written but never edited: a slot's ``key_parts`` tuple is replaced
    wholesale, so one slot's parts can be shared with another -- which
    a rebase and `Slot.__deepcopy__` both rely on.
    """

    raw: str  # source representation including any surrounding quotes
    value: str  # the decoded key string


_KEY_ESCAPES: dict[int, str] = {0x22: '\\"', 0x5C: "\\\\"}
for _c in (*range(0x20), 0x7F):
    _KEY_ESCAPES[_c] = f"\\u{_c:04X}"
del _c


def quote_basic_key(s: str) -> str:
    """Encode ``s`` as a basic-quoted TOML key (escaping where required)."""
    return f'"{s.translate(_KEY_ESCAPES)}"'


_RE_BARE_KEY_FULL = re.compile(r"\A[A-Za-z0-9_\-]+\Z")


def make_keypart(name: str) -> KeyPart:
    """Build a ``KeyPart`` for ``name``, choosing bare vs basic-quoted."""
    if _RE_BARE_KEY_FULL.match(name):
        return KeyPart(name, name)
    return KeyPart(quote_basic_key(name), name)


def make_keyparts(path: tuple[str, ...]) -> tuple[KeyPart, ...]:
    """Build a ``KeyPart`` for each segment of ``path``."""
    return tuple([make_keypart(p) for p in path])


def render_dotted(parts: tuple[KeyPart, ...], seps: tuple[str, ...]) -> str:
    """Render a dotted key as ``part0 sep0 part1 sep1 ...``.

    ``seps`` has length ``len(parts) - 1``; each entry is the literal
    whitespace + ``.`` between the surrounding parts (e.g. ``" . "``).
    """
    if len(parts) == 1:
        return parts[0].raw
    out: list[str] = []
    for i, p in enumerate(parts):
        if i:
            out.append(seps[i - 1])
        out.append(p.raw)
    return "".join(out)


# ---------------------------------------------------------------------------
# Comma-separated values (inline arrays + inline tables)
# ---------------------------------------------------------------------------


@dataclass(slots=True, eq=False)
class CommaItem:
    """One slot inside a comma-separated value.

    Layout: ``leading value trailing [comma post_comma_trivia]``.
    Shared base of sibling leaves `ArrayItem` and `InlineTableEntry`;
    use `CommaItem` only at polymorphic call sites. Fields are
    positional, for the reason given on `Slot`.
    """

    leading: str
    value: Value
    trailing: str
    has_comma: bool
    post_comma_trivia: str

    def render_tail(self) -> str:
        """Everything the item renders after its value."""
        if not self.has_comma:
            return self.trailing
        return f"{self.trailing},{self.post_comma_trivia}"

    def render(self) -> str:
        return f"{self.leading}{self.value.render()}{self.render_tail()}"


@dataclass(slots=True, eq=False)
class ArrayItem(CommaItem):
    """Represent one bare-value slot inside an inline array."""


@dataclass(slots=True, eq=False)
class InlineTableEntry(CommaItem):
    """One ``key = value`` slot inside an inline table.

    The shared trivia/comma machinery lives on `CommaItem`; this leaf
    adds only the key-prefix fields and keyed rendering.
    """

    key_parts: tuple[KeyPart, ...]
    key_seps: tuple[str, ...]  # len = len(key_parts) - 1
    pre_eq: str
    post_eq: str
    key_path: tuple[str, ...]
    """Decoded dotted-key path.

    Set by every construction site and read by inline-table validation,
    decoding, and cross-document cloning.
    """

    @override
    def render(self) -> str:
        return (
            f"{self.leading}{render_dotted(self.key_parts, self.key_seps)}"
            f"{self.pre_eq}={self.post_eq}"
            f"{self.value.render()}{self.render_tail()}"
        )


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
    header_trivia: str = ""
    final_trivia: str = ""

    # Memoised `is_multiline()` result; None means "not computed". Mutations
    # that preserve multi-line shape (append/insert/sort/reorder) leave it
    # warm; item removal and the explicit single<->multi toggle invalidate it.
    _ml_cache: bool | None = field(default=None, init=False, compare=False, repr=False)

    _open: ClassVar[str] = ""
    _close: ClassVar[str] = ""

    # Canonical inner bracket padding for the single-line form: one
    # space for inline tables (``{ a = 1 }``), none for inline arrays
    # (``[1, 2]``). An empty value carries no padding regardless.
    _single_line_pad: ClassVar[str] = ""

    def render(self) -> str:
        body = "".join([it.render() for it in self.items])
        return f"{self._open}{self.header_trivia}{body}{self.final_trivia}{self._close}"

    def is_multiline(self) -> bool:
        """Whether this value renders across multiple physical lines.

        Memoised via `_ml_cache`: the first call after a cache-invalidating
        mutation costs an O(n) scan, every other call is O(1).
        """
        if self._ml_cache is None:
            self._ml_cache = _scan_multiline(self)
        return self._ml_cache

    def reset_multiline_cache(self) -> None:
        """Drop the memoised `is_multiline` result so it recomputes.

        Call after any mutation that can change the multi-line shape:
        item removal (the removed item may carry the sole newline, or
        emptying may collapse the bracket pads), or the explicit
        single<->multi toggle. Append / insert / sort / reorder preserve
        the shape and deliberately leave the cache alone.
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
    return item.has_comma and "\n" in item.trailing


def item_eol_on_trailing(item: CommaItem) -> bool:
    """Whether ``trailing`` (rather than ``post_comma_trivia``) owns the EOL.

    A comma-first item normally uses ``trailing``. If its pre-comma break is
    structural while an EOL comment follows the comma, the post-comma channel
    owns that EOL instead. Deciding it here lets callers read, write, and
    normalise the EOL without rediscovering the distinction.

    `tomlrt._comma_ops.Boundary._eol` selects the same channel by the same
    rule, over captured lanes rather than a live item, and layers an
    "is there an EOL at all?" test on top. The two are deliberately not
    shared: expressing the rule once would mean spelling `Boundary`'s
    head/above/tail lane split in this module, and this layer is pure data.
    Change one and you must change the other.
    """
    if item_breaks_before_comma(item):
        trailing_eol, _rest = split_eol_section(item.trailing)
        if trailing_eol or "#" not in item.post_comma_trivia:
            return True
    return not item.has_comma


def item_eol_channel(item: CommaItem) -> str:
    """The trivia run that owns the item's row-attached EOL section."""
    return item.trailing if item_eol_on_trailing(item) else item.post_comma_trivia


def set_item_eol_channel(item: CommaItem, text: str) -> None:
    """Write back the run that :func:`item_eol_channel` reads."""
    if item_eol_on_trailing(item):
        item.trailing = text
    else:
        item.post_comma_trivia = text


def inter_item_separator(items: Sequence[CommaItem]) -> str:
    """Structural-pad portion of ``items[1].leading``; ``" "`` if ``len < 2``.

    Excludes any above-item comment block, which belongs to the item's
    leading rather than to the separator.
    """
    if len(items) >= 2:
        head, _above, tail = split_item_above(items[1].leading)
        return head + tail
    return " "


def _scan_multiline(v: CommaValue[_ItemT]) -> bool:
    """Uncached scan: inspect every trivia region that can carry a row break."""
    if "\n" in v.header_trivia or "\n" in v.final_trivia:
        return True
    for it in v.items:
        if "\n" in it.leading or "\n" in it.post_comma_trivia or "\n" in it.trailing:
            return True
    return False


def value_has_any_comment(v: Value) -> bool:
    """Whether any comment appears anywhere within ``v`` (recursively)."""
    if not isinstance(v, CommaValue):
        return False
    if "#" in v.header_trivia or "#" in v.final_trivia:
        return True
    return any(item_has_any_comment(it) for it in v.items)


def item_has_any_comment(item: CommaItem) -> bool:
    """Whether ``item`` carries a comment in its trivia or nested value."""
    if "#" in item.leading or "#" in item.trailing or "#" in item.post_comma_trivia:
        return True
    return value_has_any_comment(item.value)


def retarget_value_newlines(v: Value, target: str) -> None:
    """Recursively rewrite every line terminator under ``v`` to ``target``.

    Scalar values have no trivia. Multi-line string content lives in
    ``StringValue.lexeme``, not trivia, so literal CR/LF bytes
    inside strings are preserved.
    """
    if not isinstance(v, CommaValue):
        return
    v.header_trivia = retarget_newlines(v.header_trivia, target)
    v.final_trivia = retarget_newlines(v.final_trivia, target)
    for it in v.items:
        it.leading = retarget_newlines(it.leading, target)
        it.trailing = retarget_newlines(it.trailing, target)
        it.post_comma_trivia = retarget_newlines(it.post_comma_trivia, target)
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
    "item_eol_on_trailing",
    "item_has_any_comment",
    "retarget_value_newlines",
    "set_item_eol_channel",
]
