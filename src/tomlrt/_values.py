"""Represent byte-exact inline TOML values.

Values are pure data with no slot-stream awareness. Scalars carry their
source ``lexeme``; arrays and inline tables carry every separator,
comment, and whitespace run needed for exact re-emission.

Records use explicit slotted constructors rather than generating methods
at import time. Fieldless leaves inherit their storage and constructors.
"""

from __future__ import annotations

import copy
import re
import sys
from datetime import date, datetime, time, timedelta, timezone
from typing import TYPE_CHECKING, ClassVar, Generic, TypeVar

if sys.version_info >= (3, 12):
    from typing import override
else:  # pragma: no cover -- backport for Python < 3.12
    from typing_extensions import override

from tomlrt._trivia import retarget_newlines

if TYPE_CHECKING:
    from datetime import tzinfo
    from typing import TypeGuard

    from typing_extensions import Self


_ScalarT = TypeVar("_ScalarT")


def _tzinfo_is_shareable(tz: tzinfo | None) -> bool:
    """Whether a ``tzinfo`` reaches no mutable state.

    `timezone` retains the exact offset and name objects it was built
    from, so a `timedelta` or `str` subclass carrying attributes of its
    own makes an otherwise-immutable instance reachable-mutable.
    Everything the parser produces -- naive, ``timezone.utc``, or
    ``timezone(timedelta(...))`` -- passes.
    """
    if tz is None or tz is timezone.utc:
        # Redundant with the checks below, but a naive or UTC value is
        # much the commonest shape and answering it costs a third.
        return True
    return (
        type(tz) is timezone
        and type(tz.utcoffset(None)) is timedelta
        and type(tz.tzname(None)) is str
    )


def is_shareable_scalar(
    value: object,
) -> TypeGuard[str | int | float | bool | date | time]:
    """Whether a Python scalar is known to be transitively immutable.

    Exact types only: a subclass may carry mutable attributes of its
    own, so it is copied rather than shared. A temporal value has to
    reach nothing mutable through its ``tzinfo`` either.

    Identity comparisons, not a set of types: membership would hash the
    class and compare it with ``==``, so the answer would come from
    whatever its metaclass says.
    """
    kind = type(value)
    if kind is str or kind is int or kind is float or kind is bool or kind is date:
        return True
    if type(value) is datetime or type(value) is time:
        return _tzinfo_is_shareable(value.tzinfo)
    return False


class ScalarValue(Generic[_ScalarT]):
    """Base of the five TOML scalar leaves.

    Every scalar re-emits verbatim from the ``lexeme`` it was parsed
    from -- quotes, radix prefix, digit separators, and case all
    included. The leaves specialize their decoded value type while
    sharing rendering, copying and the positional ``(lexeme, value)``
    constructor.
    """

    __slots__ = ("lexeme", "value")

    def __init__(self, lexeme: str, value: _ScalarT) -> None:
        self.lexeme = lexeme
        self.value = value

    def render(self) -> str:
        return self.lexeme

    @property
    def is_shareable(self) -> bool:
        """Whether both fields are known to be transitively immutable.

        A shareable node is handed to a copy rather than cloned, so the
        two documents then hold the same object. That is sound only
        because a published scalar node is never written in place: a
        mutation rebinds its owner's ``value`` to a fresh node instead.
        The two writers of these fields are the parser, before the node
        is published, and `_copy_payloads`, on a clone that is not.
        """
        return type(self.lexeme) is str and is_shareable_scalar(self.value)

    def __deepcopy__(self, memo: dict[int, object]) -> Self:
        """Share atomic data; copy mutable payloads and subclasses safely."""
        if self.is_shareable:
            return self
        new = type(self)(self.lexeme, self.value)
        memo[id(self)] = new
        new._copy_payloads(memo)  # noqa: SLF001
        return new

    def _copy_payloads(self, memo: dict[int, object]) -> None:
        """Isolate the fields of a fresh, unpublished scalar node."""
        self.lexeme = copy.deepcopy(self.lexeme, memo)
        self.value = copy.deepcopy(self.value, memo)


class StringValue(ScalarValue[str]):
    __slots__ = ()

    value: str


class IntegerValue(ScalarValue[int]):
    __slots__ = ()

    value: int


class FloatValue(ScalarValue[float]):
    __slots__ = ()

    value: float


class BoolValue(ScalarValue[bool]):
    __slots__ = ()

    value: bool


class DateTimeValue(ScalarValue[datetime | date | time]):
    __slots__ = ()

    value: datetime | date | time


# ---------------------------------------------------------------------------
# Dotted keys
# ---------------------------------------------------------------------------


def respell_key_prefix(
    parts: tuple[str, ...],
    seps: tuple[str, ...],
    path: tuple[str, ...],
    drop: int,
    prefix: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Replace a prefix, preserving the spelling of its unchanged suffix."""
    shared = 0
    while (
        shared < min(drop, len(prefix))
        and path[drop - shared - 1] == prefix[-shared - 1]
    ):
        shared += 1
    if shared:
        drop -= shared
        prefix = prefix[:-shared]
    if not drop and not prefix:
        return parts, seps, path
    tail = parts[drop:]
    head = make_keyparts(prefix)
    joins = len(head) - 1 + bool(head and tail)
    return head + tail, (".",) * max(joins, 0) + seps[drop:], prefix + path[drop:]


_KEY_ESCAPES: dict[int, str] = {0x22: '\\"', 0x5C: "\\\\"}
for _c in (*range(0x20), 0x7F):
    _KEY_ESCAPES[_c] = f"\\u{_c:04X}"
del _c


def quote_basic_key(s: str) -> str:
    """Encode ``s`` as a basic-quoted TOML key (escaping where required)."""
    return f'"{s.translate(_KEY_ESCAPES)}"'


_RE_BARE_KEY_FULL = re.compile(r"\A[A-Za-z0-9_\-]+\Z")


def make_keyparts(path: tuple[str, ...]) -> tuple[str, ...]:
    """Spell a path, sharing it when no component needs quoting."""
    parts: list[str] = []
    quoted = False
    for name in path:
        if _RE_BARE_KEY_FULL.match(name):
            parts.append(name)
        else:
            parts.append(quote_basic_key(name))
            quoted = True
    return tuple(parts) if quoted else path


def render_dotted(parts: tuple[str, ...], seps: tuple[str, ...]) -> str:
    """Join verbatim key components and separators."""
    if len(parts) == 1:
        return parts[0]
    out = [""] * (len(parts) + len(seps))
    out[::2] = parts
    out[1::2] = seps
    return "".join(out)


# ---------------------------------------------------------------------------
# Comma-separated values (inline arrays + inline tables)
# ---------------------------------------------------------------------------


class CommaItem:
    """One slot inside a comma-separated value.

    Layout: ``leading value trailing [comma post_comma_trivia]``.
    Shared base of sibling leaves `ArrayItem` and `InlineTableEntry`;
    use `CommaItem` only at polymorphic call sites. Fields are
    positional, for the reason given on `Slot`. Field-adding subclasses
    initialize inherited fields directly to avoid an extra call per item.
    """

    __slots__ = ("has_comma", "leading", "post_comma_trivia", "trailing", "value")

    def __init__(
        self,
        leading: str,
        value: Value,
        trailing: str,
        has_comma: bool,  # noqa: FBT001
        post_comma_trivia: str,
    ) -> None:
        self.leading = leading
        self.value = value
        self.trailing = trailing
        self.has_comma = has_comma
        self.post_comma_trivia = post_comma_trivia

    def __deepcopy__(self, memo: dict[int, object]) -> Self:
        new = object.__new__(type(self))
        memo[id(self)] = new
        new.has_comma = copy.deepcopy(self.has_comma, memo)
        new.leading = copy.deepcopy(self.leading, memo)
        new.post_comma_trivia = copy.deepcopy(self.post_comma_trivia, memo)
        new.trailing = copy.deepcopy(self.trailing, memo)
        new.value = copy.deepcopy(self.value, memo)
        return new

    def render_tail(self) -> str:
        """Everything the item renders after its value."""
        if not self.has_comma:
            return self.trailing
        return f"{self.trailing},{self.post_comma_trivia}"

    def render(self) -> str:
        return f"{self.leading}{self.value.render()}{self.render_tail()}"


class ArrayItem(CommaItem):
    """Represent one bare-value slot inside an inline array."""

    __slots__ = ()


class InlineTableEntry(CommaItem):
    """One ``key = value`` slot inside an inline table.

    The shared trivia/comma machinery lives on `CommaItem`; this leaf
    adds only the key-prefix fields and keyed rendering.
    """

    __slots__ = ("key_parts", "key_path", "key_seps", "post_eq", "pre_eq")

    key_parts: tuple[str, ...]
    key_seps: tuple[str, ...]
    key_path: tuple[str, ...]
    pre_eq: str
    post_eq: str

    def __init__(
        self,
        leading: str,
        value: Value,
        trailing: str,
        has_comma: bool,  # noqa: FBT001
        post_comma_trivia: str,
        key_parts: tuple[str, ...],
        key_seps: tuple[str, ...],
        key_path: tuple[str, ...],
        pre_eq: str,
        post_eq: str,
    ) -> None:
        self.leading = leading
        self.value = value
        self.trailing = trailing
        self.has_comma = has_comma
        self.post_comma_trivia = post_comma_trivia
        self.key_parts = key_parts
        self.key_seps = key_seps
        self.key_path = key_path
        self.pre_eq = pre_eq
        self.post_eq = post_eq

    @override
    def __deepcopy__(self, memo: dict[int, object]) -> Self:
        new = super().__deepcopy__(memo)
        new.key_parts = copy.deepcopy(self.key_parts, memo)
        new.key_path = copy.deepcopy(self.key_path, memo)
        new.key_seps = copy.deepcopy(self.key_seps, memo)
        new.post_eq = copy.deepcopy(self.post_eq, memo)
        new.pre_eq = copy.deepcopy(self.pre_eq, memo)
        return new

    @override
    def render(self) -> str:
        return (
            f"{self.leading}{render_dotted(self.key_parts, self.key_seps)}"
            f"{self.pre_eq}={self.post_eq}"
            f"{self.value.render()}{self.render_tail()}"
        )


_ItemT = TypeVar("_ItemT", bound=CommaItem)


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

    __slots__ = ("_ml_cache", "final_trivia", "header_trivia", "items")

    # Memoised `is_multiline()` result; None means "not computed". Mutations
    # that preserve multi-line shape (append/insert/sort/reorder) leave it
    # warm; item removal and the explicit single<->multi toggle invalidate it.
    _ml_cache: bool | None

    _open: ClassVar[str] = ""
    _close: ClassVar[str] = ""

    # Canonical inner bracket padding for the single-line form: one
    # space for inline tables (``{ a = 1 }``), none for inline arrays
    # (``[1, 2]``). An empty value carries no padding regardless.
    _single_line_pad: ClassVar[str] = ""

    def __init__(
        self,
        items: list[_ItemT] | None = None,
        header_trivia: str = "",
        final_trivia: str = "",
    ) -> None:
        self.items = [] if items is None else items
        self.header_trivia = header_trivia
        self.final_trivia = final_trivia
        self._ml_cache = None

    def __deepcopy__(self, memo: dict[int, object]) -> Self:
        new = object.__new__(type(self))
        memo[id(self)] = new
        new._ml_cache = copy.deepcopy(self._ml_cache, memo)  # noqa: SLF001
        new.final_trivia = copy.deepcopy(self.final_trivia, memo)
        new.header_trivia = copy.deepcopy(self.header_trivia, memo)
        new.items = copy.deepcopy(self.items, memo)
        return new

    def render(self) -> str:
        body = "".join([it.render() for it in self.items])
        return f"{self._open}{self.header_trivia}{body}{self.final_trivia}{self._close}"

    def is_multiline(self) -> bool:
        """Whether this value's own trivia contains a row break.

        Nested values and scalar lexemes do not determine the outer shape.
        Memoised via `_ml_cache`: the first call after a cache-invalidating
        mutation costs an O(n) scan, every other call is O(1).
        """
        if self._ml_cache is None:
            self._ml_cache = self._own_trivia_contains("\n")
        return self._ml_cache

    def has_own_comment(self) -> bool:
        """Whether this value's own trivia carries a comment, without caching.

        Unlike `value_has_any_comment`, excludes comments in nested values.
        """
        return self._own_trivia_contains("#")

    def _own_trivia_contains(self, needle: str) -> bool:
        """Search own-level trivia, excluding nested values and scalar lexemes."""
        if needle in self.header_trivia or needle in self.final_trivia:
            return True
        for it in self.items:
            if (
                needle in it.leading
                or needle in it.post_comma_trivia
                or needle in it.trailing
            ):
                return True
        return False

    def reset_multiline_cache(self) -> None:
        """Drop the memoised `is_multiline` result so it recomputes.

        Call after any mutation that can change the multi-line shape:
        item removal (the removed item may carry the sole newline, or
        emptying may collapse the bracket pads), or the explicit
        single<->multi toggle. Append / insert / sort / reorder preserve
        the shape and deliberately leave the cache alone.
        """
        self._ml_cache = None


class ArrayValue(CommaValue[ArrayItem]):
    """Inline array literal (``[ ... ]``)."""

    __slots__ = ()

    _open: ClassVar[str] = "["
    _close: ClassVar[str] = "]"


class EmptyAoTValue(ArrayValue):
    """Synthetic ``[]`` placeholder retaining an empty array-of-tables' shape."""

    __slots__ = ()


class InlineTableValue(CommaValue[InlineTableEntry]):
    """Inline table literal (``{ ... }``)."""

    __slots__ = ()

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
    "StringValue",
    "Value",
    "item_has_any_comment",
    "retarget_value_newlines",
]
