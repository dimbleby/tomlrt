"""Array views.

`Array(list)` backs inline TOML arrays. `AoT(list[Table])` backs
array-of-tables entries via `AoTEntry` records.
"""

from __future__ import annotations

import operator
import sys
from typing import TYPE_CHECKING, Any, SupportsIndex, TypeVar, overload

if sys.version_info >= (3, 12):
    from typing import override
else:  # pragma: no cover -- backport for Python < 3.12
    from typing_extensions import override

from copy import deepcopy

from tomlrt import _layout_ops
from tomlrt._array_comments import (
    ArrayEolView,
    ArrayLeadingBlockView,
    ArrayLeadingView,
)
from tomlrt._comma_ops import (
    detect_style,
    reorder_owned,
    splice_in,
    splice_insert,
    splice_out,
)
from tomlrt._errors import TOMLError
from tomlrt._format import (
    _resolve_format_options,
    format_inline_root,
    set_comma_value_multiline,
)
from tomlrt._trivia import (
    NewlineNode,
    Trivia,
    WhitespaceNode,
    strip_trailing_indent,
)
from tomlrt._typecheck import _validate_mapping
from tomlrt._values import (
    ArrayItem,
    ArrayValue,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, MutableMapping

    from _typeshed import SupportsRichComparison

    from tomlrt._comma_ops import (
        CommaStyle,
    )
    from tomlrt._format import FormatOptions
    from tomlrt._trivia import TriviaPiece
    from tomlrt._values import (
        Value,
    )

    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self

    from tomlrt._container import Container, Document, Table, TomlInput


_T = TypeVar("_T")


class Array(list[Any]):
    """An inline TOML array.

    `Array` is a `list` subclass, so ``isinstance(arr, list)`` holds and
    it can be passed wherever a `list` or `Sequence` is expected.
    """

    __slots__ = ("_layout_root", "_value")

    def __init__(
        self,
        items: Iterable[TomlInput] = (),
        *,
        multiline: bool = False,
        indent: int = 4,
    ) -> None:
        """Construct a standalone inline array.

        ``Array([1, 2, 3])`` builds an inline array; ``multiline=True``
        lays items out one per line, indented by ``indent`` spaces.
        """
        super().__init__()

        indent_str = " " * indent
        self._value: ArrayValue = ArrayValue()
        self._layout_root: Document | None = None
        items_list = list(items)
        if not items_list:
            if multiline:
                self._value.final_trivia = Trivia(
                    [NewlineNode(text="\n"), WhitespaceNode(text=indent_str)]
                )
            return
        from tomlrt._container import _synth_inline_array  # noqa: PLC0415

        val, arr = _synth_inline_array(items_list, layout_root=None, owner=None)
        self._value = val
        for v in arr:
            list.append(self, v)
        if multiline and val.items:
            indent_pieces: list[TriviaPiece] = [
                NewlineNode(text="\n"),
                WhitespaceNode(text=indent_str),
            ]
            val.header_trivia = Trivia(list(indent_pieces))
            val.final_trivia = Trivia([NewlineNode(text="\n")])
            for k, it in enumerate(val.items):
                it.leading = Trivia() if k == 0 else Trivia(list(indent_pieces))
                it.post_comma_trivia = Trivia()
                it.trailing = Trivia()
                it.has_comma = True

    def to_list(self) -> list[Any]:
        """Materialise a plain-Python ``list`` (recursive)."""
        from tomlrt._container import _to_python  # noqa: PLC0415

        return [_to_python(x) for x in self]

    def __copy__(self) -> Array:
        return Array(self.to_list(), multiline=self.multiline)

    def __deepcopy__(self, memo: dict[int, object]) -> Array:
        return self.__copy__()

    def array(self, index: SupportsIndex) -> Array:
        """Return ``self[index]`` typed as a nested `Array`."""
        return self._typed_item(index, Array, "an Array")

    def table(self, index: SupportsIndex) -> Table:
        """Return ``self[index]`` typed as a `Table`."""
        from tomlrt._container import Table  # noqa: PLC0415

        return self._typed_item(index, Table, "a Table")

    @overload
    def get_array(self, index: SupportsIndex) -> Array | None: ...
    @overload
    def get_array(self, index: SupportsIndex, default: _T) -> Array | _T: ...
    def get_array(self, index: SupportsIndex, default: object = None) -> object:
        """Like `array(index)` but returns ``default`` for out-of-range."""
        try:
            return self.array(index)
        except IndexError:
            return default

    @overload
    def get_table(self, index: SupportsIndex) -> Table | None: ...
    @overload
    def get_table(self, index: SupportsIndex, default: _T) -> Table | _T: ...
    def get_table(self, index: SupportsIndex, default: object = None) -> object:
        """Like `table(index)` but returns ``default`` for out-of-range."""
        try:
            return self.table(index)
        except IndexError:
            return default

    def _typed_item(self, index: SupportsIndex, cls: type[_T], label: str) -> _T:
        v = self[index]
        if not isinstance(v, cls):
            msg = f"item at {index} is {type(v).__name__}, not {label}"
            raise TypeError(msg)
        return v

    @property
    def _attached(self) -> bool:
        """True iff this array is wired to a user-visible document.

        Mirrors :attr:`Container._attached` — see that docstring.
        """
        lr = self._layout_root
        return lr is not None and not lr._is_private  # noqa: SLF001

    @property
    def _doc_newline(self) -> str:
        r"""The active newline of the owning document, or ``"\n"`` if detached.

        Mirrors :attr:`Container._doc_newline`.
        """
        lr = self._layout_root
        return lr._newline if lr is not None else "\n"  # noqa: SLF001

    def _style(self) -> CommaStyle:
        return detect_style(self._value, nl=self._doc_newline)

    @property
    def multiline(self) -> bool:
        """True iff this array is rendered in multi-line form."""
        return self._value.is_multiline()

    @multiline.setter
    def multiline(self, value: bool) -> None:
        if self.multiline == value:
            return
        self.set_multiline(multiline=value)

    @property
    def comments(self) -> MutableMapping[int, str]:
        """EOL comment view, indexed by item position."""
        return ArrayEolView(self)

    @property
    def leading_comments(self) -> MutableMapping[int, tuple[str, ...]]:
        """Attached leading-comment view, indexed by item position."""
        return ArrayLeadingView(self)

    @property
    def leading_block(self) -> MutableMapping[int, tuple[str | None, ...]]:
        """Full leading-block view, indexed by item position.

        Comment lines are strings and blank lines are ``None``.
        """
        return ArrayLeadingBlockView(self)

    def format(
        self,
        *,
        options: FormatOptions | None = None,
        comments: bool | None = None,
    ) -> None:
        """Canonicalise this array's formatting in place.

        Rewrites whitespace, indentation, separators, and newlines
        while preserving shape (single-line stays single-line, multi-line
        stays multi-line) and orphan comment text.

        ``comments=`` is deprecated; use
        ``FormatOptions(normalize_comments=...)`` instead. Supplying both
        arguments raises ``ValueError``.
        """
        resolved = _resolve_format_options(options=options, comments=comments)
        format_inline_root(self._value, nl=self._doc_newline, options=resolved)

    def set_multiline(self, *, multiline: bool, indent: int = 4) -> Array:
        """Switch this array between flush single-line and multi-line form.

        When laying out multi-line, items are indented by ``indent``
        spaces.

        Raises ``TOMLError`` when collapsing a multi-line array that
        carries comments anywhere in it, including inside nested values,
        since they would have nowhere to live on one line.

        Returns ``self`` for chaining.
        """
        set_comma_value_multiline(
            self._value, multiline=multiline, nl=self._doc_newline, indent=" " * indent
        )
        return self

    def _synth_cst(self, value: object) -> tuple[Value, object]:
        from tomlrt._container import _synth_value  # noqa: PLC0415

        return _synth_value(
            value,
            layout_root=self._layout_root,
            parent=None,
            path=(),
            owner=None,
        )

    @override
    def append(self, value: Any) -> None:
        if isinstance(value, AoT):
            msg = "cannot store an array-of-tables inside an inline array"
            raise TOMLError(msg)
        cst, decoded = self._synth_cst(value)
        self._append_with_style(cst, decoded, self._style())

    def _append_with_style(self, cst: Value, decoded: Any, style: CommaStyle) -> None:
        """Append ``cst`` / ``decoded`` using a precomputed ``style``.

        `__imul__` snapshots style before mutating so trailing-comma
        decisions reflect the original layout, not a half-mutated one.
        """
        new_item = _make_item(cst, has_comma=False)
        splice_in(self._value, new_item, style, self._doc_newline)
        list.append(self, decoded)

    @override
    def extend(self, values: Iterable[Any]) -> None:
        # Snapshot so ``arr.extend(arr)`` duplicates once like list does.
        snapshot = list(values)
        if not snapshot:
            return
        if any(isinstance(v, AoT) for v in snapshot):
            msg = "cannot store an array-of-tables inside an inline array"
            raise TOMLError(msg)
        # Reuse one style for every item: re-deriving it per item is O(n)
        # for a single-line array, so doing it n times would be quadratic.
        style = self._style()
        for v in snapshot:
            cst, decoded = self._synth_cst(v)
            self._append_with_style(cst, decoded, style)

    @override
    def clear(self) -> None:
        self._value.items.clear()
        # Drop inter-item trivia; preserve bracket leading in final_trivia.
        strip_trailing_indent(self._value.header_trivia, self._value.final_trivia)
        list.clear(self)
        self._value.reset_multiline_cache()

    @override
    def pop(self, index: SupportsIndex = -1) -> Any:
        i = _norm_index(index, len(self), "pop")
        decoded = self[i]
        del self[i]
        return decoded

    @override
    def remove(self, value: Any) -> None:
        for i, v in enumerate(self):
            if v == value:
                del self[i]
                return
        msg = "Array.remove(x): x not in array"
        raise ValueError(msg)

    @override
    def insert(self, index: SupportsIndex, value: Any) -> None:
        i = _norm_insert_index(index, len(self))
        if i == len(self):
            self.append(value)
            return
        cst, decoded = self._synth_cst(value)
        style = self._style()
        new_item = _make_item(cst, has_comma=True)
        splice_insert(self._value, new_item, i, style, self._doc_newline)
        list.insert(self, i, decoded)

    @override
    def reverse(self) -> None:
        self._reorder(list(reversed(range(len(self)))))

    @override
    def sort(
        self,
        *,
        key: Callable[[Any], SupportsRichComparison] | None = None,
        reverse: bool = False,
    ) -> None:
        n = len(self)
        if key is None:
            sort_key: Callable[[int], Any] = lambda i: self[i]  # noqa: E731
        else:
            key_fn = key
            sort_key = lambda i: key_fn(self[i])  # noqa: E731
        order = sorted(range(n), key=sort_key, reverse=reverse)
        self._reorder(order)

    def _reorder(self, order: list[int]) -> None:
        """Apply index permutation to items, decoded list, and per-item comments."""
        items = self._value.items
        if not items:
            return
        new_items = [items[j] for j in order]
        new_decoded = [self[j] for j in order]
        reorder_owned(
            self._value,
            range(len(items)),
            new_items,
            self._doc_newline,
            is_multiline=self._style().is_multiline,
        )
        list.__init__(self, new_decoded)

    @overload
    def __setitem__(self, index: SupportsIndex, value: Any) -> None: ...
    @overload
    def __setitem__(self, index: slice, value: Iterable[Any]) -> None: ...
    @override
    def __setitem__(
        self,
        index: SupportsIndex | slice,
        value: Any,
    ) -> None:
        if isinstance(index, slice):
            try:
                values = list(value)
            except TypeError as exc:
                msg = "can only assign an iterable"
                raise TypeError(msg) from exc
            if index.step is not None and index.step != 1:
                indices = list(range(*index.indices(len(self))))
                if len(values) != len(indices):
                    msg = (
                        f"attempt to assign sequence of size {len(values)} "
                        f"to extended slice of size {len(indices)}"
                    )
                    raise ValueError(msg)
                # Extended slice positions are unchanged; replace per slot.
                for k, v in zip(indices, values, strict=True):
                    self[k] = v
                return
            # Reuse delete/insert boundary handling for contiguous slices.
            start, stop, _ = index.indices(len(self))
            del self[start:stop]
            for offset, v in enumerate(values):
                self.insert(start + offset, v)
            return
        # int index: just replace the value CST in place.
        # Reject before synthesising or mutating any CST, matching the
        # IndexError that ``list.__setitem__`` raises for a bad index.
        items = self._value.items
        i = _norm_index(index, len(items), "list assignment")
        cst, dec = self._synth_cst(value)
        items[i].value = cst
        list.__setitem__(self, i, dec)

    @override
    def __delitem__(self, index: SupportsIndex | slice) -> None:
        items = self._value.items
        if isinstance(index, slice):
            removed = list(range(*index.indices(len(items))))
            if not removed:
                return
        else:
            removed = [_norm_index(index, len(items), "list assignment")]
        list.__delitem__(self, index if isinstance(index, slice) else removed[0])
        splice_out(
            self._value,
            removed,
            self._doc_newline,
            is_multiline=self._value.is_multiline(),
        )

    @override
    def __iadd__(self, values: Iterable[Any]) -> Self:
        self.extend(values)
        return self

    @override
    def __imul__(self, count: SupportsIndex) -> Self:
        n = operator.index(count)
        if n <= 0:
            self.clear()
            return self
        if n == 1:
            return self
        # Snapshot style and items before appending: `_append_with_style`
        # flips the previous last comma, and decoded inline views must be
        # re-built from cloned CST so they stay wired to their own nodes.
        from tomlrt._build import _decode_value  # noqa: PLC0415

        style = self._style()
        src_items = list(self._value.items)
        for _ in range(n - 1):
            for src in src_items:
                cst = deepcopy(src.value)
                decoded = _decode_value(
                    cst,
                    layout_root=self._layout_root,
                    parent=None,
                    path=(),
                    owner=None,
                )
                self._append_with_style(cst, decoded, style)
        return self


def _norm_insert_index(index: SupportsIndex, n: int) -> int:
    """Return the clamped index used by ``list.insert``."""
    i = operator.index(index)
    return max(0, n + i) if i < 0 else min(i, n)


def _norm_index(index: SupportsIndex, n: int, action: str) -> int:
    """Return non-negative in-range index, or raise IndexError.

    ``action`` is used verbatim in the error message
    (e.g. ``"pop"`` → ``"pop index out of range"``).
    """
    i = operator.index(index)
    if i < 0:
        i += n
    if i < 0 or i >= n:
        msg = f"{action} index out of range"
        raise IndexError(msg)
    return i


def _make_item(cst: Value, *, has_comma: bool) -> ArrayItem:
    """Build a fresh ``ArrayItem`` with empty trivia."""
    return ArrayItem(
        leading=Trivia(),
        value=cst,
        trailing=Trivia(),
        has_comma=has_comma,
        post_comma_trivia=Trivia(),
    )


class AoT(list["Table"]):
    """An Array-of-tables, e.g. ``[[products]]`` repeated.

    `AoT` is a `list[Table]` subclass, so ``isinstance(aot, list)`` holds
    and it can be passed wherever a `list` or `Sequence` is expected.
    """

    __slots__ = ("_layout_root", "_parent", "_path")

    def __init__(self, entries: Iterable[Mapping[str, TomlInput]] = ()) -> None:
        """Construct a standalone array-of-tables."""
        super().__init__()
        self._layout_root: Document | None = None
        self._path: tuple[str, ...] = ()
        self._parent: Container | None = None
        for entry in entries:
            e = _validate_mapping(entry, label="AoT entry")
            list.append(self, _make_unattached_entry(e))

    @property
    def _attached_doc(self) -> Document:
        """The owning ``Document``, asserting this AoT is attached."""
        lr = self._layout_root
        assert lr is not None, "AoT is not attached to a document"
        return lr

    def to_list(self) -> list[dict[str, Any]]:
        """Materialise a list of plain-Python ``dict``s (recursive)."""
        return [t.to_dict() for t in self]

    def __copy__(self) -> AoT:
        return AoT(self.to_list())

    def __deepcopy__(self, memo: dict[int, object]) -> AoT:
        return self.__copy__()

    def add(self, entry: Mapping[str, TomlInput] | None = None) -> Table:
        """Append a fresh ``[[path]]`` entry and return its `Table` view.

        ``entry`` may be initial body content or ``None``. Attached AoTs
        append to the owning document.
        """
        if entry is not None:
            entry = _validate_mapping(entry, label="AoT entry")
        if self._layout_root is None:
            list.append(self, _make_unattached_entry(entry))
            return self[-1]
        return _layout_ops.add_aot_entry(self, entry)

    def _add_entry_attached(self, value: Mapping[str, Any]) -> Table:
        """Dispatch a new attached AoT entry from ``value``.

        Pre: attached AoT. Selects the trivia-preserving clone path for
        attached AoT entries or sections.
        """
        from tomlrt._container import Table as TableType  # noqa: PLC0415

        if isinstance(value, TableType) and value._layout_root is not None:  # noqa: SLF001
            if value._is_own_aot_entry:  # noqa: SLF001
                return _layout_ops.clone_aot_entry(self, value)
            if value._header_ref is not None and not value._inline:  # noqa: SLF001
                return _layout_ops.clone_table_as_aot_entry(self, value)
        return _layout_ops.add_aot_entry(self, value)

    def _replace_entry_attached(self, index: int, value: Mapping[str, Any]) -> None:
        """Dispatch in-place replacement of an attached AoT entry."""
        from tomlrt._container import Table as TableType  # noqa: PLC0415

        if (
            isinstance(value, TableType)
            and value._layout_root is not None  # noqa: SLF001
            and value._is_own_aot_entry  # noqa: SLF001
        ):
            _layout_ops.replace_aot_entry_with_clone(self, index, value)
            return
        _layout_ops.replace_aot_entry(self, index, value)

    # Unsupported list mutators fail closed rather than corrupt the
    # doc-stream via inherited `list` behaviour.

    @override
    def pop(self, index: SupportsIndex = -1) -> Table:
        i = _norm_index(index, len(self), "pop")
        if self._layout_root is None:
            return list.pop(self, i)
        return _layout_ops.remove_aot_entry(self, i)

    @override
    def __delitem__(self, index: SupportsIndex | slice) -> None:
        if isinstance(index, slice):
            if self._layout_root is None:
                list.__delitem__(self, index)
                return
            indices = sorted(set(range(*index.indices(len(self)))))
            if indices:
                _layout_ops.remove_aot_entries(self, indices)
            return
        if self._layout_root is None:
            list.__delitem__(self, index)
            return
        _layout_ops.remove_aot_entry(self, operator.index(index))

    @override
    def clear(self) -> None:
        if self._layout_root is None:
            list.clear(self)
            return
        n = len(self)
        if n:
            _layout_ops.remove_aot_entries(self, range(n))

    @overload
    def __setitem__(
        self, index: SupportsIndex, value: Mapping[str, TomlInput]
    ) -> None: ...
    @overload
    def __setitem__(
        self, index: slice, value: Iterable[Mapping[str, TomlInput]]
    ) -> None: ...
    @override
    def __setitem__(
        self,
        index: SupportsIndex | slice,
        value: Any,
    ) -> None:
        if isinstance(index, slice):
            try:
                values = list(value)
            except TypeError as exc:
                msg = "can only assign an iterable"
                raise TypeError(msg) from exc
            indices = list(range(*index.indices(len(self))))
            if (
                index.step is not None
                and index.step != 1
                and len(values) != len(indices)
            ):
                msg = (
                    f"attempt to assign sequence of size {len(values)} "
                    f"to extended slice of size {len(indices)}"
                )
                raise ValueError(msg)
            # Atomicity preflight: validate everything before mutating.
            typed_values = [_validate_mapping(v, label="AoT entry") for v in values]
            if self._layout_root is None:
                list.__setitem__(
                    self, index, [_make_unattached_entry(v) for v in typed_values]
                )
                return
            # Contiguous replacement: delete, append via dispatcher,
            # then renormalise to the requested order.
            if index.step is None or index.step == 1:
                start = index.indices(len(self))[0]
                for i in sorted(indices, reverse=True):
                    _layout_ops.remove_aot_entry(self, i)
                new_entries = [self._add_entry_attached(v) for v in typed_values]
                cur: list[Table] = list(self)
                cur = cur[: -len(new_entries)] if new_entries else cur
                for off, e in enumerate(new_entries):
                    cur.insert(start + off, e)
                if cur != list(self):
                    _layout_ops.renormalise_aot_order(self, cur)
                return
            # Extended slice with matching length: replace in place.
            for i, v in zip(indices, typed_values, strict=True):
                self._replace_entry_attached(i, v)
            return
        entry = _validate_mapping(value, label="AoT entry")
        if self._layout_root is None:
            list.__setitem__(self, index, _make_unattached_entry(entry))
            return
        self._replace_entry_attached(operator.index(index), entry)

    @override
    def append(self, value: Table | Mapping[str, TomlInput]) -> None:
        # Same semantics as `add(body)` but with no return value (list API).
        entry = _validate_mapping(value, label="AoT entry")
        if self._layout_root is None:
            list.append(self, _make_unattached_entry(entry))
            return
        self._add_entry_attached(entry)

    @override
    def extend(self, values: Iterable[Table | Mapping[str, TomlInput]]) -> None:
        # Snapshot so ``aot.extend(aot)`` duplicates once like list does.
        for v in list(values):
            self.append(v)

    @override
    def insert(
        self, index: SupportsIndex, value: Table | Mapping[str, TomlInput]
    ) -> None:
        entry = _validate_mapping(value, label="AoT entry")
        if self._layout_root is None:
            list.insert(self, index, _make_unattached_entry(entry))
            return
        # Normalise against the pre-append length to match list.insert.
        idx = _norm_insert_index(index, len(self))
        new_entry = self._add_entry_attached(entry)
        new_order: list[Table] = list(self)
        new_order.pop()
        new_order.insert(idx, new_entry)
        if new_order != list(self):
            _layout_ops.renormalise_aot_order(self, new_order)

    @override
    def remove(self, value: Mapping[str, TomlInput]) -> None:
        for i, t in enumerate(self):
            if t is value or t == value:
                del self[i]
                return
        msg = "list.remove(x): x not in list"
        raise ValueError(msg)

    @override
    def reverse(self) -> None:
        if self._layout_root is None:
            list.reverse(self)
            return
        new_order = list(reversed(self))
        _layout_ops.renormalise_aot_order(self, new_order)

    @override
    def sort(  # type: ignore[override]
        self,
        *,
        key: Callable[[Table], SupportsRichComparison],
        reverse: bool = False,
    ) -> None:
        new_order = sorted(self, key=key, reverse=reverse)
        if self._layout_root is None:
            list.__init__(self, new_order)
            return
        _layout_ops.renormalise_aot_order(self, new_order)

    @override
    def __iadd__(self, values: Iterable[Mapping[str, TomlInput]]) -> Self:  # type: ignore[override]
        self.extend(values)
        return self

    @override
    def __imul__(self, count: SupportsIndex) -> Self:
        n = operator.index(count)
        if n <= 0:
            self.clear()
            return self
        if n == 1:
            return self
        if self._layout_root is None:
            # Detached AoT: replicate via `extend` so `_make_unattached_entry`
            # stays the source of truth for document-free entries.
            bodies = self.to_list()
            for _ in range(n - 1):
                self.extend(bodies)
            return self
        originals = list(self)
        for _ in range(n - 1):
            for e in originals:
                _layout_ops.clone_aot_entry(self, e)
        return self


def _make_unattached_entry(body: Mapping[str, TomlInput] | None) -> Table:
    """Build a fresh unattached `Table` view as an AoT-entry placeholder."""
    from tomlrt._container import Table, _populate_unattached  # noqa: PLC0415

    t = Table()
    if body is not None:
        _populate_unattached(t, body)
    return t


__all__ = ["AoT", "Array"]
