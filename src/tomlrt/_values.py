"""Represent byte-exact inline TOML values.

Values are pure data with no slot-stream awareness. Scalars carry their
source ``lexeme``; arrays and inline tables carry every separator,
comment, and whitespace run needed for exact re-emission.

Records use explicit slotted constructors rather than generating methods
at import time. Fieldless leaves inherit their storage and constructors.
"""

from __future__ import annotations

import bisect
import copy
import operator
import re
import sys
from datetime import UTC, date, datetime, time, timedelta, timezone
from typing import TYPE_CHECKING, ClassVar, Generic, TypeVar

if sys.version_info >= (3, 12):
    from typing import override
else:  # pragma: no cover -- backport for Python < 3.12
    from typing_extensions import override

from tomlrt._trivia import retarget_newlines

if TYPE_CHECKING:
    from datetime import tzinfo
    from typing import Self, TypeGuard


_ScalarT = TypeVar("_ScalarT")


def _tzinfo_is_shareable(tz: tzinfo | None) -> bool:
    """Whether a ``tzinfo`` reaches no mutable state.

    `timezone` retains the exact offset and name objects it was built
    from, so a `timedelta` or `str` subclass carrying attributes of its
    own makes an otherwise-immutable instance reachable-mutable.
    Everything the parser produces -- naive, ``timezone.utc``, or
    ``timezone(timedelta(...))`` -- passes.
    """
    if tz is None or tz is UTC:
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
    """One payload and its complete outgoing physical gap.

    ``value before [comma after] following`` retains the authored comma
    location. EOL text belongs to this payload, above-blocks to the next
    payload, and blank-only text stays positional.
    """

    __slots__ = ("after", "before", "following", "has_comma", "value")

    def __init__(
        self,
        value: Value,
        before: str = "",
        has_comma: bool = False,  # noqa: FBT001, FBT002
        after: str = "",
        following: str = "",
    ) -> None:
        self.value = value
        self.before = before
        self.has_comma = has_comma
        self.after = after
        self.following = following

    def __deepcopy__(self, memo: dict[int, object]) -> Self:
        new = object.__new__(type(self))
        memo[id(self)] = new
        new.value = copy.deepcopy(self.value, memo)
        new.before = copy.deepcopy(self.before, memo)
        new.has_comma = copy.deepcopy(self.has_comma, memo)
        new.after = copy.deepcopy(self.after, memo)
        new.following = copy.deepcopy(self.following, memo)
        return new

    def render_tail(self) -> str:
        if self.has_comma:
            return f"{self.before},{self.after}{self.following}"
        return f"{self.before}{self.following}"

    def render(self) -> str:
        return f"{self.value.render()}{self.render_tail()}"


class ArrayItem(CommaItem):
    """Represent one bare-value slot inside an inline array."""

    __slots__ = ()


class InlineTableEntry(CommaItem):
    """One ``key = value`` slot inside an inline table.

    Adds the key prefix and an order label for locating its current position.
    """

    __slots__ = ("_order", "key_parts", "key_path", "key_seps", "post_eq", "pre_eq")

    key_parts: tuple[str, ...]
    key_seps: tuple[str, ...]
    key_path: tuple[str, ...]
    pre_eq: str
    post_eq: str

    def __init__(
        self,
        value: Value,
        key_parts: tuple[str, ...],
        key_seps: tuple[str, ...],
        key_path: tuple[str, ...],
        pre_eq: str,
        post_eq: str,
        before: str = "",
        has_comma: bool = False,  # noqa: FBT001, FBT002
        after: str = "",
        following: str = "",
    ) -> None:
        self.value = value
        self.before = before
        self.has_comma = has_comma
        self.after = after
        self.following = following
        self.key_parts = key_parts
        self.key_seps = key_seps
        self.key_path = key_path
        self.pre_eq = pre_eq
        self.post_eq = post_eq
        self._order = 0

    @override
    def __deepcopy__(self, memo: dict[int, object]) -> Self:
        new = super().__deepcopy__(memo)
        new.key_parts = copy.deepcopy(self.key_parts, memo)
        new.key_path = copy.deepcopy(self.key_path, memo)
        new.key_seps = copy.deepcopy(self.key_seps, memo)
        new.post_eq = copy.deepcopy(self.post_eq, memo)
        new.pre_eq = copy.deepcopy(self.pre_eq, memo)
        new._order = self._order  # noqa: SLF001
        return new

    @override
    def render(self) -> str:
        return (
            f"{render_dotted(self.key_parts, self.key_seps)}"
            f"{self.pre_eq}={self.post_eq}"
            f"{self.value.render()}{self.render_tail()}"
        )


_ItemT = TypeVar("_ItemT", bound=CommaItem)


class CommaValue(Generic[_ItemT]):
    """Shared backbone of `ArrayValue` and `InlineTableValue`.

    ``opening`` owns the gap before item zero, or all interior trivia when
    empty. Each item owns one complete outgoing gap, including the final
    item's closing pad. No gap is split across neighboring records.
    Concrete subclasses bind ``_ItemT`` and set the bracket ClassVars.
    """

    __slots__ = ("_ml_cache", "items", "opening")

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
        opening: str = "",
    ) -> None:
        self.items = [] if items is None else items
        self.opening = opening
        self._ml_cache = None

    def __deepcopy__(self, memo: dict[int, object]) -> Self:
        new = object.__new__(type(self))
        memo[id(self)] = new
        new._ml_cache = copy.deepcopy(self._ml_cache, memo)
        new.opening = copy.deepcopy(self.opening, memo)
        new.items = copy.deepcopy(self.items, memo)
        return new

    def render(self) -> str:
        body = "".join([item.render() for item in self.items])
        return f"{self._open}{self.opening}{body}{self._close}"

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
        if needle in self.opening:
            return True
        for item in self.items:
            if (
                needle in item.following
                or needle in item.after
                or needle in item.before
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


_entry_order = operator.attrgetter("_order")


class InlineTableValue(CommaValue[InlineTableEntry]):
    """Inline table literal with direct key bindings and ordered entries.

    Order labels increase along ``items``. Deletion leaves labels intact;
    copying, reordering and key rebasing rebuild the index and labels.
    """

    __slots__ = ("_key_index",)

    _open: ClassVar[str] = "{"
    _close: ClassVar[str] = "}"
    _single_line_pad: ClassVar[str] = " "
    _key_index: dict[tuple[str, ...], InlineTableEntry]

    def __init__(
        self,
        items: list[InlineTableEntry] | None = None,
        opening: str = "",
    ) -> None:
        self.items = [] if items is None else items
        self.opening = opening
        self._ml_cache = None
        self.reindex()

    @override
    def __deepcopy__(self, memo: dict[int, object]) -> Self:
        new = super().__deepcopy__(memo)
        new.reindex()
        return new

    def find_entry(self, path: tuple[str, ...]) -> tuple[int, InlineTableEntry] | None:
        """Resolve a key and locate its entry in physical order."""
        entry = self._key_index.get(path)
        if entry is None:
            return None
        position = bisect.bisect_left(self.items, entry._order, key=_entry_order)  # noqa: SLF001
        assert self.items[position] is entry, "inline entry index is stale"
        return position, entry

    def record_entry(self, entry: InlineTableEntry) -> None:
        """Index an entry just appended to ``items``."""
        entry._order = self.items[-2]._order + 1 if len(self.items) > 1 else 0  # noqa: SLF001
        self._key_index[entry.key_path] = entry

    def reindex(self) -> None:
        """Rebuild key bindings and physical order after copying or reordering."""
        self._key_index = {}
        for position, entry in enumerate(self.items):
            entry._order = position  # noqa: SLF001
            self._key_index[entry.key_path] = entry


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
    if v.has_own_comment():
        return True
    return any(value_has_any_comment(it.value) for it in v.items)


def retarget_value_newlines(v: Value, target: str) -> None:
    """Recursively rewrite every line terminator under ``v`` to ``target``.

    Scalar values have no trivia. Multi-line string content lives in
    ``StringValue.lexeme``, not trivia, so literal CR/LF bytes
    inside strings are preserved.
    """
    if not isinstance(v, CommaValue):
        return
    v.opening = retarget_newlines(v.opening, target)
    for it in v.items:
        it.following = retarget_newlines(it.following, target)
        it.before = retarget_newlines(it.before, target)
        it.after = retarget_newlines(it.after, target)
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
    "retarget_value_newlines",
]
