"""Array views.

`Array(list)` backs an inline-array TOML value (`[1, 2, 3]`). `AoT`
(array-of-tables) is a `list[Table]` whose entries are individually
backed by `AoTEntry` records and whose elements are `Table` views.
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
    ArrayLeadingView,
)
from tomlrt._comma_ops import (
    CommaStyle,
    _normalise_row_breaks,
    detect_style,
    flip_to_internal,
    flip_to_terminal,
    migrate_bracket_above,
    reorder_owned,
    splice_in,
    splice_out,
)
from tomlrt._errors import TOMLError
from tomlrt._format import (
    _canon_inline_value,
    _canon_multiline_shape,
    _canon_value,
)
from tomlrt._trivia import (
    NewlineNode,
    Trivia,
    WhitespaceNode,
    strip_trailing_indent,
    trivia_has_comment,
)
from tomlrt._typecheck import _validate_mapping
from tomlrt._values import (
    ArrayItem,
    ArrayValue,
    CommaValue,
    value_is_multiline,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    from _typeshed import SupportsRichComparison

    from tomlrt._trivia import TriviaPiece
    from tomlrt._values import (
        CommaItem,
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

    `Array` is a `list` subclass, so ``isinstance(arr, list)`` holds
    and it can be passed wherever a `list` or `Sequence` is expected.
    """

    __slots__ = ("_layout_root", "_multiline", "_value")

    def __init__(
        self,
        items: Iterable[TomlInput] = (),
        *,
        multiline: bool = False,
        indent: str = "    ",
    ) -> None:
        """Construct a standalone inline array.

        ``Array([1, 2, 3])`` builds an inline array;
        ``Array([1, 2, 3], multiline=True)`` lays it out one item per
        line with ``indent`` indentation.
        """
        super().__init__()

        self._value: ArrayValue = ArrayValue()
        self._multiline: bool = multiline
        self._layout_root: Document | None = None
        items_list = list(items)
        if not items_list:
            if multiline:
                self._value.final_trivia = Trivia(
                    [NewlineNode(text="\n"), WhitespaceNode(text=indent)]
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
                WhitespaceNode(text=indent),
            ]
            val.header_trivia = Trivia(list(indent_pieces))
            val.final_trivia = Trivia([NewlineNode(text="\n")])
            for k, it in enumerate(val.items):
                it.leading = Trivia() if k == 0 else Trivia(list(indent_pieces))
                it.post_comma_trivia = Trivia()
                it.trailing = Trivia()
                it.has_comma = True
        elif multiline and not val.items:
            # Empty multiline factory: header_trivia stays empty;
            # final_trivia carries the pre-`]` line break + indent so
            # that a subsequent first append slots in correctly.
            val.final_trivia = Trivia(
                [NewlineNode(text="\n"), WhitespaceNode(text=indent)]
            )

    def to_list(self) -> list[Any]:
        """Materialise a plain-Python ``list`` (recursive)."""
        from tomlrt._container import _to_python  # noqa: PLC0415

        return [_to_python(x) for x in self]

    def __copy__(self) -> Array:
        return Array(self.to_list(), multiline=self._multiline)

    def __deepcopy__(self, memo: dict[int, object]) -> Array:
        return Array(self.to_list(), multiline=self._multiline)

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
        i = operator.index(index)
        if i < -len(self) or i >= len(self):
            return default
        return self.array(index)

    @overload
    def get_table(self, index: SupportsIndex) -> Table | None: ...
    @overload
    def get_table(self, index: SupportsIndex, default: _T) -> Table | _T: ...
    def get_table(self, index: SupportsIndex, default: object = None) -> object:
        """Like `table(index)` but returns ``default`` for out-of-range."""
        i = operator.index(index)
        if i < -len(self) or i >= len(self):
            return default
        return self.table(index)

    def _typed_item(self, index: SupportsIndex, cls: type[_T], label: str) -> _T:
        v = self[index]
        if not isinstance(v, cls):
            msg = f"item at {index} is {type(v).__name__}, not {label}"
            raise TypeError(msg)
        return v

    # ---- mutation -----------------------------------------------------

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
        return detect_style(self._value, multiline_flag=self._multiline)

    @property
    def multiline(self) -> bool:
        """True iff this array is rendered in multi-line form."""
        return self._style().is_multiline

    @multiline.setter
    def multiline(self, value: bool) -> None:
        if self.multiline == value:
            return
        self.set_multiline(multiline=value)

    @property
    def comments(self) -> ArrayEolView:
        """EOL comment view, indexed by item position."""
        return ArrayEolView(self)

    @property
    def leading_comments(self) -> ArrayLeadingView:
        """Leading-comment view, indexed by item position."""
        return ArrayLeadingView(self)

    def format(self, *, comments: bool = True) -> None:
        """Canonicalise this array's formatting in place.

        Rewrites whitespace, indentation, separators, and newlines to a
        canonical layout, while preserving the array's overall shape
        (single-line stays single-line, multi-line stays multi-line)
        and the *content* of any orphan comment blocks above items.

        When ``comments`` is true (the default), comment text is also
        normalised: ``#foo`` and ``#   foo`` both become ``# foo``, and
        trailing whitespace inside comments is stripped.
        """
        _canon_inline_value(self._value, nl=self._doc_newline, comments=comments)

    def set_multiline(self, *, multiline: bool, indent: str = "    ") -> Array:
        """Switch this array between flush single-line and multi-line form.

        Raises ``TOMLError`` when collapsing a multi-line array that
        carries comments (anywhere in per-item trivia, header_trivia
        or final_trivia, or in nested inline values), since those
        would have nowhere to live on a single line.

        Returns ``self`` for chaining.
        """
        ind = indent
        value = self._value
        items = value.items
        if not multiline:
            for it in items:
                if _item_has_any_comment(it):
                    msg = (
                        "cannot collapse multi-line array: "
                        "items contain EOL or leading comments"
                    )
                    raise TOMLError(msg)
            if trivia_has_comment(value.header_trivia) or trivia_has_comment(
                value.final_trivia
            ):
                msg = (
                    "cannot collapse multi-line array: "
                    "header or trailing trivia contains comments"
                )
                raise TOMLError(msg)
            value.header_trivia = Trivia()
            value.final_trivia = Trivia()
            for k, it in enumerate(items):
                it.leading = Trivia() if k == 0 else Trivia([WhitespaceNode(" ")])
                it.post_comma_trivia = Trivia()
                it.trailing = Trivia()
            self._multiline = False
            flush_style = CommaStyle(
                is_multiline=False,
                inter_separator=Trivia([WhitespaceNode(text=" ")]),
                trailing_comma=False,
                trailing_post=Trivia(),
            )
            _renormalise_commas(items, flush_style)
            return self
        self._multiline = True
        nl = self._doc_newline
        for it in items:
            _canon_value(it.value, nl=nl, comments=True, parent_indent=ind)
        _canon_multiline_shape(
            value, nl=nl, comments=True, item_indent=ind, outer_indent=""
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
            msg = "Cannot store an array-of-tables inside an inline array"
            raise TOMLError(msg)
        cst, decoded = self._synth_cst(value)
        self._append_with_style(cst, decoded, self._style())

    def _append_with_style(self, cst: Value, decoded: Any, style: CommaStyle) -> None:
        """Append ``cst`` / ``decoded`` using a precomputed ``style``.

        Shared between `append` (which derives style fresh) and
        `__imul__` (which must snapshot style before mutating, so
        the closing trailing-comma decision reflects the array's
        original layout — not the half-mutated state).
        """
        new_item = _make_item(cst, has_comma=False)
        splice_in(self._value, new_item, style, self._doc_newline)
        list.append(self, decoded)

    @override
    def extend(self, values: Iterable[Any]) -> None:
        for v in values:
            self.append(v)

    @override
    def clear(self) -> None:
        self._value.items.clear()
        # Drop any inter-item trivia clutter; preserve the bracket
        # leading captured in final_trivia.
        strip_trailing_indent(self._value.header_trivia, self._value.final_trivia)
        list.clear(self)

    @override
    def pop(self, index: SupportsIndex = -1) -> Any:
        n = len(self)
        i = int(index)
        if i < 0:
            i += n
        if i < 0 or i >= n:
            msg = "pop index out of range"
            raise IndexError(msg)
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
        cst, decoded = self._synth_cst(value)
        i = int(index)
        n = len(self)
        if i < 0:
            i = max(0, n + i)
        i = min(i, n)
        items = self._value.items
        if i == n:
            self.append(value)
            return
        style = self._style()
        if i == 0:
            # Item-owned semantics: any above-block currently inside
            # header_trivia conceptually belongs to *the item that
            # appears below it*. On insert(0), that item becomes
            # items[1]; the above-block migrates to its leading. The
            # new item gets bare leading; header_trivia retains only
            # its structural pad.
            new_item = _make_item(cst, has_comma=True)
            self._value.header_trivia, items[0].leading = migrate_bracket_above(
                self._value.header_trivia, style.inter_separator
            )
            items.insert(0, new_item)
        else:
            # Internal insert: new item with leading = inter_sep; the
            # item that was at position i (now at i+1) keeps its old
            # leading (which already carries inter_sep + its own
            # above-block).
            new_item = _make_item(
                cst, leading=style.inter_separator.copy(), has_comma=True
            )
            items.insert(i, new_item)
        _normalise_row_breaks(
            items, self._value, self._doc_newline, multiline=style.is_multiline
        )
        list.insert(self, i, decoded)

    @override
    def reverse(self) -> None:
        self._reorder(list(reversed(range(len(self)))))

    @override
    def sort(
        self,
        *,
        key: Callable[[Any], object] | None = None,
        reverse: bool = False,
    ) -> None:
        n = len(self)
        if key is None:
            sort_key: Callable[[int], Any] = lambda i: self[i]  # noqa: E731
        else:
            key_fn = key
            sort_key = lambda i: key_fn(self[i])  # noqa: E731
        order = sorted(range(n), key=sort_key, reverse=reverse)  # ty: ignore[no-matching-overload]
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
        list.clear(self)
        for v in new_decoded:
            list.append(self, v)

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
                # Extended slice: positions are unchanged, only values
                # change. Delegate to the int-index path per slot.
                for k, v in zip(indices, values, strict=True):
                    self[k] = v
                return
            # Contiguous slice: realise as delete + insert so the
            # boundary-handling that ``__delitem__`` and ``insert``
            # already encode is reused, rather than duplicated here.
            start, stop, _ = index.indices(len(self))
            del self[start:stop]
            for offset, v in enumerate(values):
                self.insert(start + offset, v)
            return
        # int index: just replace the value CST in place.
        i = int(index)
        cst, dec = self._synth_cst(value)
        items = self._value.items
        if i < 0:
            i += len(items)
        items[i].value = cst
        list.__setitem__(self, index, dec)

    @override
    def __delitem__(self, index: SupportsIndex | slice) -> None:
        items = self._value.items
        if not items:
            list.__delitem__(self, index)  # propagate IndexError
            return
        # Normalise to a list of removed positions so the shared
        # comma-list orchestration can run uniformly.
        if isinstance(index, slice):
            removed = list(range(*index.indices(len(items))))
        else:
            i = int(index)
            if i < 0:
                i += len(items)
            if i < 0 or i >= len(items):
                # Let list.__delitem__ raise the canonical IndexError.
                list.__delitem__(self, index)
                return
            removed = [i]
        if not removed:
            list.__delitem__(self, index)
            return
        list.__delitem__(self, index)
        splice_out(
            self._value,
            removed,
            self._doc_newline,
            is_multiline=self._multiline or value_is_multiline(self._value),
        )

    @override
    def __iadd__(self, values: Iterable[Any]) -> Self:
        self.extend(values)
        return self

    @override
    def __imul__(self, count: SupportsIndex) -> Self:
        n = int(count)
        if n <= 0:
            self.clear()
            return self
        if n == 1:
            return self
        # Snapshot style and the source-item list before any append:
        # ``_append_with_style`` flips the previous-last item's
        # ``has_comma``, so re-detecting style on a half-mutated array
        # would spuriously promote a no-trailing-comma array to one with
        # a trailing comma. The items list is snapshot because we append
        # to it inside the loop. Each iteration deepcopies the source
        # ``Value`` and re-decodes so the appended logical view is wired
        # to its own CST clone (``deepcopy`` of the decoded view for an
        # inline ``Table`` / ``Array`` yields a detached object not wired
        # to any CST).
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


# ---------------------------------------------------------------------------
# Array-specific helpers (style detection + canonical pads live in _format)
# ---------------------------------------------------------------------------


def _make_item(
    cst: Value, *, leading: Trivia | None = None, has_comma: bool
) -> ArrayItem:
    """Build a fresh ``ArrayItem`` with empty trailing/post_comma.

    Most call-sites only vary ``leading`` and ``has_comma``; this helper
    centralises the boilerplate so the policy stays in one place.
    """
    return ArrayItem(
        leading=leading if leading is not None else Trivia(),
        value=cst,
        trailing=Trivia(),
        has_comma=has_comma,
        post_comma_trivia=Trivia(),
    )


def _value_has_any_comment(val: Value) -> bool:
    if not isinstance(val, CommaValue):
        return False
    if trivia_has_comment(val.header_trivia) or trivia_has_comment(val.final_trivia):
        return True
    return any(_item_has_any_comment(it) for it in val.items)


def _item_has_any_comment(item: CommaItem) -> bool:
    if (
        trivia_has_comment(item.leading)
        or trivia_has_comment(item.trailing)
        or trivia_has_comment(item.post_comma_trivia)
    ):
        return True
    return _value_has_any_comment(item.value)


def _renormalise_commas(items: list[ArrayItem], style: CommaStyle) -> None:
    """Reset has_comma + post_comma_trivia across ``items`` per style."""
    if not items:
        return
    for it in items[:-1]:
        flip_to_internal(it)
    flip_to_terminal(items[-1], style)


class AoT(list["Table"]):
    """An Array-of-tables, e.g. ``[[products]]`` repeated.

    `AoT` is a `list[Table]` subclass, so ``isinstance(aot, list)``
    holds and it can be passed wherever a `list` or `Sequence` is
    expected.
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
        """The owning ``Document``, asserting this AoT is attached.

        Mirror of :attr:`Container._attached_doc` — see that docstring.
        """
        lr = self._layout_root
        assert lr is not None, "AoT is not attached to a document"
        return lr

    def to_list(self) -> list[dict[str, Any]]:
        """Materialise a list of plain-Python ``dict``s (recursive)."""
        return [t.to_dict() for t in self]

    def __copy__(self) -> AoT:
        return AoT(self.to_list())

    def __deepcopy__(self, memo: dict[int, object]) -> AoT:
        return AoT(self.to_list())

    def add(self, entry: Mapping[str, TomlInput] | None = None) -> Table:
        """Append a fresh ``[[path]]`` entry and return its `Table` view.

        ``entry`` may be a Mapping (initial body content) or ``None``
        (empty entry). The AoT must be attached to a document.
        """
        if entry is not None:
            entry = _validate_mapping(entry, label="AoT entry")
        if self._layout_root is None:
            list.append(self, _make_unattached_entry(entry))
            return self[-1]
        return _layout_ops.add_aot_entry(self, entry)

    def _add_entry_attached(self, value: Mapping[str, Any]) -> Table:
        """Dispatch a new attached AoT entry from ``value``.

        Pre: ``self._layout_root is not None``. Selects the trivia-
        preserving clone path when ``value`` is itself an attached
        AoT entry or attached standard section, otherwise falls
        through to ``add_aot_entry``.
        """
        from tomlrt._container import Table as TableType  # noqa: PLC0415

        if isinstance(value, TableType) and value._layout_root is not None:  # noqa: SLF001
            if value._owner_aot_entry is not None:  # noqa: SLF001
                return _layout_ops.clone_aot_entry(self, value)
            if value._header_ref is not None and not value._inline:  # noqa: SLF001
                return _layout_ops.clone_table_as_aot_entry(self, value)
        return _layout_ops.add_aot_entry(self, value)

    def _replace_entry_attached(
        self, index: int, value: Mapping[str, Any] | None
    ) -> None:
        """Dispatch in-place replacement of an attached AoT entry."""
        from tomlrt._container import Table as TableType  # noqa: PLC0415

        if (
            isinstance(value, TableType)
            and value._layout_root is not None  # noqa: SLF001
            and value._owner_aot_entry is not None  # noqa: SLF001
        ):
            _layout_ops.replace_aot_entry_with_clone(self, index, value)
            return
        _layout_ops.replace_aot_entry(self, index, value)

    # Supported list-mutator surface. Anything not implemented here
    # is overridden below to fail closed rather than corrupt the
    # doc-stream via inherited `list` behaviour.

    @override
    def pop(self, index: SupportsIndex = -1) -> Table:
        idx = int(index)
        n = len(self)
        if not -n <= idx < n:
            msg = "pop index out of range"
            raise IndexError(msg)
        if self._layout_root is None:
            return list.pop(self, idx)
        return _layout_ops.remove_aot_entry(self, idx)

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
        _layout_ops.remove_aot_entry(self, int(index))

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
        value: Mapping[str, TomlInput] | Iterable[Mapping[str, TomlInput]],
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
            # Validate every assigned value is a Mapping/Table BEFORE
            # mutating the AoT (atomicity preflight).
            typed_values = [_validate_mapping(v, label="AoT entry") for v in values]
            if self._layout_root is None:
                list.__setitem__(
                    self, index, [_make_unattached_entry(v) for v in typed_values]
                )
                return
            # For contiguous step == 1: replace by delete-range then
            # insert at the start index. Order matters: build new
            # entries via the dispatcher (appended to end), then
            # renormalise.
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
            # Extended slice with step != 1 and matching length:
            # replace each entry in place.
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
        for v in values:
            self.append(v)

    @override
    def insert(
        self, index: SupportsIndex, value: Table | Mapping[str, TomlInput]
    ) -> None:
        entry = _validate_mapping(value, label="AoT entry")
        if self._layout_root is None:
            list.insert(self, index, _make_unattached_entry(entry))
            return
        # Normalise against the pre-append length so semantics match
        # list.insert (e.g. insert(-1, x) inserts BEFORE the last entry,
        # not in place of it).
        n_before = len(self)
        new_entry = self._add_entry_attached(entry)
        idx = int(index)
        idx = max(0, n_before + idx) if idx < 0 else min(idx, n_before)
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
            list.clear(self)
            for t in new_order:
                list.append(self, t)
            return
        _layout_ops.renormalise_aot_order(self, new_order)

    @override
    def __iadd__(self, values: Iterable[Mapping[str, TomlInput]]) -> Self:  # type: ignore[override]
        self.extend(values)
        return self

    @override
    def __imul__(self, count: SupportsIndex) -> Self:
        n = int(count)
        if n <= 0:
            self.clear()
            return self
        if n == 1:
            return self
        if self._layout_root is None:
            # Detached AoT: replicate entries through `extend`, so the
            # detached append path (`_make_unattached_entry`) is the
            # one source of truth for "how do we add an entry without
            # touching the document".
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
