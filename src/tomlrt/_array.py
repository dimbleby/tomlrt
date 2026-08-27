"""Array views: `Array` for inline arrays, `AoT` for arrays-of-tables."""

from __future__ import annotations

import operator
import sys
from typing import TYPE_CHECKING, Any, SupportsIndex, TypeVar, overload

if sys.version_info >= (3, 12):
    from typing import override
else:  # pragma: no cover -- backport for Python < 3.12
    from typing_extensions import override

from copy import deepcopy

from tomlrt import _container, _layout_ops
from tomlrt._array_comments import _ArrayAdapter
from tomlrt._comma_comments import (
    CommaEolView,
    CommaLeadingBlockView,
    CommaLeadingView,
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
    strip_trailing_indent,
)
from tomlrt._typecheck import _validate_mapping
from tomlrt._values import (
    ArrayItem,
    ArrayValue,
)
from tomlrt._view import _View

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, MutableMapping

    from _typeshed import SupportsRichComparison

    from tomlrt._comma_ops import (
        CommaStyle,
    )
    from tomlrt._format import FormatOptions
    from tomlrt._values import (
        Value,
    )

    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self

    from tomlrt._container import Container, Document, Table, TomlInput


_T = TypeVar("_T")


class Array(_View, list[Any]):
    """An inline TOML array.

    `Array` is a `list` subclass, so ``isinstance(arr, list)`` holds and
    it can be passed wherever a `list` or `Sequence` is expected.
    """

    __slots__ = ("_host", "_layout_root", "_name", "_value")

    _inline = True

    @classmethod
    def _view(
        cls,
        value: ArrayValue,
        layout_root: Document | None,
        name: str | None = None,
        host: Array | Container | None = None,
    ) -> Array:
        """View over existing CST, holding no items yet.

        The public constructor goes the other way: it synthesises CST from
        Python items. This is for callers that already have CST and fill
        the items in themselves. ``name`` is the key the array is bound
        under and ``host`` is where it is bound -- its containing array
        when it is an element, else its hosting container. A caller that
        does not yet know where the array lands leaves ``host`` unset for
        the value funnel to stamp later (see `_file_host`).
        """
        arr = cls.__new__(cls)
        arr._value = value  # noqa: SLF001
        arr._layout_root = layout_root  # noqa: SLF001
        arr._host = host  # noqa: SLF001
        arr._name = name or ""  # noqa: SLF001
        return arr

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

        val = ArrayValue()
        self._value: ArrayValue = val
        self._layout_root: Document | None = None
        self._host: Array | Container | None = None
        self._name: str = ""
        items_list = list(items)
        if items_list:
            from tomlrt._container import _fill_inline_array  # noqa: PLC0415

            _fill_inline_array(self, items_list, layout_root=None, owner=None)
        if not multiline:
            return
        row_indent = f"\n{' ' * indent}"
        if not val.items:
            val.final_trivia = row_indent
            return
        val.header_trivia = row_indent
        val.final_trivia = "\n"
        for k, it in enumerate(val.items):
            it.leading = "" if k == 0 else row_indent
            it.post_comma_trivia = ""
            it.trailing = ""
            it.has_comma = True

    def to_list(self) -> list[Any]:
        """Materialise a plain-Python ``list`` (recursive)."""
        from tomlrt._container import _to_python  # noqa: PLC0415

        return [_to_python(x) for x in self]

    @override
    def __copy__(self) -> Array:
        return Array(self.to_list(), multiline=self.multiline)

    def array(self, index: SupportsIndex) -> Array:
        """Return ``self[index]`` typed as a nested `Array`."""
        return self._typed_item(index, Array, "an Array")

    def table(self, index: SupportsIndex) -> Table:
        """Return ``self[index]`` typed as a `Table`."""
        return self._typed_item(index, _container.Table, "a Table")

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

    @override
    def _view_children(self) -> Iterable[object]:
        return self

    @override
    def _reset_displaced(self) -> None:
        # ``_value`` stays: it's reused if this view is attached elsewhere.
        self._layout_root = None

    def _style(self) -> CommaStyle:
        return detect_style(self._value)

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
        return CommaEolView(_ArrayAdapter(self))

    @property
    def leading_comments(self) -> MutableMapping[int, tuple[str, ...]]:
        """Attached leading-comment view, indexed by item position."""
        return CommaLeadingView(_ArrayAdapter(self))

    @property
    def leading_block(self) -> MutableMapping[int, tuple[str | None, ...]]:
        """Full leading-block view, indexed by item position.

        Comment lines are strings and blank lines are ``None``.
        """
        return CommaLeadingBlockView(_ArrayAdapter(self))

    def format(
        self,
        *,
        options: FormatOptions | None = None,
        comments: bool | None = None,
    ) -> None:
        """Canonicalise this array's formatting in place.

        Rewrites whitespace, indentation, separators, and newlines
        while preserving shape (single-line stays single-line, multi-line
        stays multi-line) and orphan comment text. A multi-line array's
        closing bracket lines up with the row the array starts on.

        ``comments=`` is deprecated; use
        ``FormatOptions(normalize_comments=...)`` instead. Supplying both
        arguments raises ``ValueError``.
        """
        resolved = _resolve_format_options(options=options, comments=comments)
        from tomlrt._container import _host_kv_slot  # noqa: PLC0415

        format_inline_root(
            self._value,
            nl=self._doc_newline,
            options=resolved,
            host=_host_kv_slot(self),
        )

    def set_multiline(self, *, multiline: bool, indent: int = 4) -> Array:
        """Switch this array between flush single-line and multi-line form.

        When laying out multi-line, items are indented by ``indent``
        spaces and the closing bracket lines up with the row the array
        starts on.

        Raises ``TOMLError`` when collapsing a multi-line array that
        carries comments anywhere in it, including inside nested values,
        since they would have nowhere to live on one line.

        Returns ``self`` for chaining.
        """
        from tomlrt._container import _host_kv_slot  # noqa: PLC0415

        set_comma_value_multiline(
            self._value,
            multiline=multiline,
            nl=self._doc_newline,
            indent=" " * indent,
            host=_host_kv_slot(self),
        )
        return self

    def _synth_item(self, value: object) -> tuple[Value, object]:
        """Synthesise one value accepted by an inline array.

        The synthesised element is uplinked to this array (``array_host``)
        so it can later derive its hosting KV slot: an element has no key
        of its own, so it derives it from the array object (see
        `_host_kv_slot`).
        """
        from tomlrt._container import _synth_value  # noqa: PLC0415

        if isinstance(value, AoT):
            msg = "cannot store an array-of-tables inside an inline array"
            raise TOMLError(msg)
        return _synth_value(
            value,
            layout_root=self._layout_root,
            parent=None,
            name=None,
            owner=None,
            array_host=self,
        )

    def _prepare_values(self, values: list[Any]) -> list[tuple[Value, Any]]:
        """Validate every value before synthesising: synthesis live-attaches."""
        from tomlrt._container import _validate_input  # noqa: PLC0415

        for value in values:
            _validate_input(value, inline_only=True)
        return [self._synth_item(value) for value in values]

    @override
    def append(self, value: Any) -> None:
        cst, decoded = self._synth_item(value)
        self._append_with_style(cst, decoded, self._style())

    def _append_with_style(self, cst: Value, decoded: Any, style: CommaStyle) -> None:
        """Append ``cst`` / ``decoded`` using a precomputed ``style``.

        Precomputing avoids re-deriving style from array state that
        mutation has already changed (e.g. mid-``__imul__``).
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
        prepared = self._prepare_values(snapshot)
        # Reuse one style for every item: re-deriving it per item is O(n)
        # for a single-line array, so doing it n times would be quadratic.
        style = self._style()
        for cst, decoded in prepared:
            self._append_with_style(cst, decoded, style)

    @override
    def clear(self) -> None:
        _layout_ops.reset_displaced_views(*self)
        self._value.items.clear()
        # Drop inter-item trivia; preserve bracket leading in final_trivia.
        self._value.header_trivia, self._value.final_trivia = strip_trailing_indent(
            self._value.header_trivia, self._value.final_trivia
        )
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
        cst, decoded = self._synth_item(value)
        self._insert_synthesised(i, cst, decoded)

    def _insert_synthesised(self, index: int, cst: Value, decoded: Any) -> None:
        """Insert an already-synthesised value."""
        if index == len(self):
            self._append_with_style(cst, decoded, self._style())
            return
        style = self._style()
        new_item = _make_item(cst, has_comma=True)
        splice_insert(self._value, new_item, index, style, self._doc_newline)
        list.insert(self, index, decoded)

    def _replace_synthesised(self, index: int, cst: Value, decoded: Any) -> None:
        """Replace an item with an already-synthesised value."""
        old = self[index]
        # Assigning an item to itself re-uses the very view being
        # replaced, which must stay attached; anything else is displaced.
        if old is not decoded:
            _layout_ops.reset_displaced_views(old)
        self._value.items[index].value = cst
        list.__setitem__(self, index, decoded)

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
                prepared = self._prepare_values(values)
                # Extended slice positions are unchanged; replace per slot.
                for k, (cst, decoded) in zip(indices, prepared, strict=True):
                    self._replace_synthesised(k, cst, decoded)
                return
            prepared = self._prepare_values(values)
            # Reuse delete/insert boundary handling for contiguous slices.
            start, stop, _ = index.indices(len(self))
            del self[start:stop]
            for offset, (cst, decoded) in enumerate(prepared):
                self._insert_synthesised(start + offset, cst, decoded)
            return
        # int index: reject before synthesising or mutating any CST, to
        # match the IndexError ``list.__setitem__`` raises for a bad index.
        i = _norm_index(index, len(self._value.items), "list assignment")
        cst, dec = self._synth_item(value)
        self._replace_synthesised(i, cst, dec)

    @override
    def __delitem__(self, index: SupportsIndex | slice) -> None:
        items = self._value.items
        if isinstance(index, slice):
            removed = list(range(*index.indices(len(items))))
            if not removed:
                return
        else:
            removed = [_norm_index(index, len(items), "list assignment")]
        _layout_ops.reset_displaced_views(*(self[i] for i in removed))
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
                decoded = _decode_value(cst, self._layout_root, None, None, None, self)
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
    return ArrayItem("", cst, "", has_comma, "")


class AoT(_View, list["Table"]):
    """An Array-of-tables, e.g. ``[[products]]`` repeated.

    `AoT` is a `list[Table]` subclass, so ``isinstance(aot, list)`` holds
    and it can be passed wherever a `list` or `Sequence` is expected.
    """

    __slots__ = ("_host", "_layout_root", "_path")

    _inline = False

    @override
    def _view_children(self) -> Iterable[object]:
        return self

    def __init__(self, entries: Iterable[Mapping[str, TomlInput]] = ()) -> None:
        """Construct a standalone array-of-tables."""
        super().__init__()
        self._layout_root: Document | None = None
        self._path: tuple[str, ...] = ()
        self._host: Container | None = None
        for entry in entries:
            e = _validate_mapping(entry, label="AoT entry")
            list.append(self, _make_unattached_entry(e))

    def _unbind_from_document(self) -> None:
        """Detach from the owning document.

        Called once this AoT has left its document, so a caller still
        holding the view can no longer write through it.
        """
        self._layout_root = None
        self._host = None
        self._path = ()

    @property
    def _attached_doc(self) -> Document:
        """The owning ``Document``, asserting this AoT is attached."""
        lr = self._layout_root
        assert lr is not None, "AoT is not attached to a document"
        return lr

    def to_list(self) -> list[dict[str, Any]]:
        """Materialise a list of plain-Python ``dict``s (recursive)."""
        return [t.to_dict() for t in self]

    @override
    def __copy__(self) -> AoT:
        return AoT(self.to_list())

    def add(self, entry: Mapping[str, TomlInput] | None = None) -> Table:
        """Append a fresh ``[[path]]`` entry and return its `Table` view.

        ``entry`` may be initial body content or ``None``. Attached AoTs
        append to the owning document.
        """
        if entry is not None:
            entry = _prepare_aot_entries((entry,))[0]
        if self._layout_root is None:
            list.append(self, _make_unattached_entry(entry))
            return self[-1]
        return _layout_ops.add_aot_entry(self, entry)

    def _add_entry_attached(self, value: Mapping[str, Any]) -> Table:
        """Dispatch a new attached AoT entry from ``value``.

        Precondition: attached AoT. Prefers the trivia-preserving clone
        path for an existing AoT entry or section.
        """
        if isinstance(value, _container.Table) and value._layout_root is not None:  # noqa: SLF001
            if value._is_own_aot_entry:  # noqa: SLF001
                return _layout_ops.clone_aot_entry(self, value)
            if value._header_ref is not None and not value._inline:  # noqa: SLF001
                return _layout_ops.clone_table_as_aot_entry(self, value)
        return _layout_ops.add_aot_entry(self, value)

    def _replace_entry_attached(self, index: int, value: Mapping[str, Any]) -> None:
        """Dispatch in-place replacement of an attached AoT entry."""
        if (
            isinstance(value, _container.Table)
            and value._layout_root is not None  # noqa: SLF001
            and value._is_own_aot_entry  # noqa: SLF001
        ):
            _layout_ops.replace_aot_entry_with_clone(self, index, value)
            return
        _layout_ops.replace_aot_entry(self, index, value)

    # Each of these must route attached vs. detached AoTs differently:
    # inherited `list` behaviour alone would corrupt the doc-stream.

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
            typed_values = _prepare_aot_entries(values)
            if self._layout_root is None:
                list.__setitem__(
                    self, index, [_make_unattached_entry(v) for v in typed_values]
                )
                return
            # Contiguous replacement: delete, append via dispatcher,
            # then renormalise to the requested order.
            if index.step is None or index.step == 1:
                start = index.indices(len(self))[0]
                if indices:
                    _layout_ops.remove_aot_entries(self, indices)
                new_entries = [self._add_entry_attached(v) for v in typed_values]
                cur: list[Table] = list(self)
                cur = cur[: -len(new_entries)] if new_entries else cur
                for off, e in enumerate(new_entries):
                    cur.insert(start + off, e)
                if cur != list(self):
                    _layout_ops.renormalise_aot_order(self, cur)
                return
            # Extended slice: length already matched, so replace in place.
            for i, v in zip(indices, typed_values, strict=True):
                self._replace_entry_attached(i, v)
            return
        entry = _prepare_aot_entries((value,))[0]
        if self._layout_root is None:
            list.__setitem__(self, index, _make_unattached_entry(entry))
            return
        self._replace_entry_attached(operator.index(index), entry)

    @override
    def append(self, value: Table | Mapping[str, TomlInput]) -> None:
        # Same semantics as `add(body)` but with no return value (list API).
        entry = _prepare_aot_entries((value,))[0]
        self._append_validated(entry)

    def _append_validated(self, entry: Mapping[str, TomlInput]) -> None:
        """Append an entry that has passed bulk-mutation preflight."""
        if self._layout_root is None:
            list.append(self, _make_unattached_entry(entry))
            return
        self._add_entry_attached(entry)

    @override
    def extend(self, values: Iterable[Table | Mapping[str, TomlInput]]) -> None:
        # Snapshot so ``aot.extend(aot)`` duplicates once like list does.
        entries = _prepare_aot_entries(values)
        for entry in entries:
            self._append_validated(entry)

    @override
    def insert(
        self, index: SupportsIndex, value: Table | Mapping[str, TomlInput]
    ) -> None:
        entry = _prepare_aot_entries((value,))[0]
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


def _prepare_aot_entries(
    values: Iterable[Any],
) -> list[Mapping[str, TomlInput]]:
    """Snapshot and validate complete AoT entries."""
    from tomlrt._container import _validate_section_values  # noqa: PLC0415

    entries = [_validate_mapping(value, label="AoT entry") for value in list(values)]
    for entry in entries:
        _validate_section_values(entry)
    return entries


def _make_unattached_entry(body: Mapping[str, TomlInput] | None) -> Table:
    """Build a fresh unattached `Table` view as an AoT-entry placeholder."""
    from tomlrt._container import Table, _populate_unattached  # noqa: PLC0415

    t = Table()
    if body is not None:
        _populate_unattached(t, body)
    return t


__all__ = ["AoT", "Array"]
