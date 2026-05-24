"""Array views.

`Array(list)` backs an inline-array TOML value (`[1, 2, 3]`). `AoT`
(array-of-tables) is a `list[Table]` whose entries are individually
backed by `AoTEntry` records and whose elements are `Table` views.
"""

from __future__ import annotations

import operator
import sys
from collections.abc import Mapping
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
    apply_comments,
    clear_all_comments,
    snapshot_comments,
)
from tomlrt._errors import TOMLError
from tomlrt._trivia import (
    CommentNode,
    NewlineNode,
    Trivia,
    WhitespaceNode,
    clone_trivia,
    join_above_block,
    split_above_block,
    split_eol_section,
    trivia_has_comment,
    trivia_has_newline,
)
from tomlrt._values import (
    ArrayItem,
    ArrayValue,
    InlineTableValue,
    inter_item_separator,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from _typeshed import SupportsRichComparison

    from tomlrt._trivia import TriviaPiece
    from tomlrt._values import (
        CommaItem,
        InlineTableEntry,
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

    def _style(self) -> _ArrayStyle:
        return _detect_style(self._value, multiline_flag=self._multiline)

    @property
    def multiline(self) -> bool:
        """True iff this array is rendered in multi-line form."""
        return self._style().is_multiline

    @multiline.setter
    def multiline(self, value: bool) -> None:
        self.set_multiline(multiline=value)

    @property
    def comments(self) -> ArrayEolView:
        """EOL comment view, indexed by item position."""
        return ArrayEolView(self)

    @property
    def leading_comments(self) -> ArrayLeadingView:
        """Leading-comment view, indexed by item position."""
        return ArrayLeadingView(self)

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
            flush_style = _ArrayStyle(
                is_multiline=False,
                inter_separator=Trivia([WhitespaceNode(text=" ")]),
                trailing_comma=False,
                trailing_post=Trivia(),
            )
            _renormalise_commas(items, flush_style)
            return self
        self._multiline = True
        nl = self._doc_newline
        if not items:
            value.final_trivia = Trivia(
                [NewlineNode(text=nl), WhitespaceNode(text=ind)]
            )
            return self
        indent_pieces: list[TriviaPiece] = [
            NewlineNode(text=nl),
            WhitespaceNode(text=ind),
        ]
        value.header_trivia = Trivia(list(indent_pieces))
        value.final_trivia = Trivia([NewlineNode(text=nl)])
        for k, it in enumerate(items):
            it.leading = Trivia() if k == 0 else Trivia(list(indent_pieces))
            it.post_comma_trivia = Trivia()
            it.trailing = Trivia()
        ml_style = _ArrayStyle(
            is_multiline=True,
            inter_separator=Trivia(list(indent_pieces)),
            trailing_comma=True,
            trailing_post=Trivia([NewlineNode(text=nl)]),
        )
        _renormalise_commas(items, ml_style)
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

    def _append_with_style(self, cst: Value, decoded: Any, style: _ArrayStyle) -> None:
        """Append ``cst`` / ``decoded`` using a precomputed ``style``.

        Shared between `append` (which derives style fresh) and
        `__imul__` (which must snapshot style before mutating, so
        the closing trailing-comma decision reflects the array's
        original layout — not the half-mutated state).
        """
        items = self._value.items
        if not items:
            # Empty array → first append. The interior trivia (which
            # may contain a comment block) currently lives in
            # final_trivia. Reframe it under the canonical model:
            #   header_trivia = (everything up through the indent that
            #                    will sit before the new item)
            #   final_trivia  = (just the line break before `]`)
            self._restamp_for_first_append()
            new_item = _make_item(cst, has_comma=False)
            items.append(new_item)
            _flip_to_terminal(new_item, style)
            list.append(self, decoded)
            return
        # Non-empty: any above-`]` comment block in final_trivia
        # logically belongs ABOVE the new item we're about to add.
        ft_pad, ft_above = split_above_block(self._value.final_trivia)
        new_leading = join_above_block(style.inter_separator, ft_above)
        _flip_to_internal(items[-1])
        new_item = _make_item(cst, leading=new_leading, has_comma=False)
        if ft_above.pieces:
            self._value.final_trivia = ft_pad
        items.append(new_item)
        _flip_to_terminal(new_item, style)
        # Under the canonical multiline model the row break + bracket
        # pad before ``]`` lives in ``final_trivia``. The previous last
        # item's "before ]" gap was already in ``final_trivia`` (or
        # was empty when its post_comma carried an EOL whose NL
        # absorbed the gap). After append, the new last item still
        # needs that row break before ``]`` — synthesise one when
        # final_trivia opens without a newline.
        if style.is_multiline:
            ft = self._value.final_trivia
            if not (ft.pieces and isinstance(ft.pieces[0], NewlineNode)):
                ft.pieces = [NewlineNode(self._doc_newline), *ft.pieces]
        list.append(self, decoded)

    def _restamp_for_first_append(self) -> None:
        """Reframe an empty array's final_trivia into header_trivia + tail.

        Called from the first ``append`` onto an empty multiline array.
        Splits ``final_trivia`` so that:
          * ``header_trivia`` carries the bracket-line break, any
            above-item comments, and the indent for the about-to-be-
            inserted value;
          * ``final_trivia`` carries just the line break that puts
            ``]`` on its own line.
        For single-line empty arrays (``[ ]``), mirrors the inner space
        on both sides so both bracket faces stay padded.
        """
        ft = self._value.final_trivia
        if not ft.pieces:
            return
        if not trivia_has_newline(ft):
            self._value.header_trivia = clone_trivia(ft)
            return
        pieces = list(ft.pieces)
        last_nl = -1
        for i, p in enumerate(pieces):
            if isinstance(p, NewlineNode):
                last_nl = i
        # Pieces up to and including the last NL form the "above the
        # new value" region. The indent for the value itself is either
        # the WS that already follows last_nl in the existing trivia,
        # or the indent of the most recent comment line, or four spaces.
        head_pieces = pieces[: last_nl + 1]
        tail_pieces = pieces[last_nl + 1 :]
        if tail_pieces and isinstance(tail_pieces[0], WhitespaceNode):
            value_indent = str(tail_pieces[0].text)
        else:
            value_indent = _indent_from_final_trivia(ft) or "    "
        self._value.header_trivia = Trivia(
            [*head_pieces, WhitespaceNode(text=value_indent)]
        )
        self._value.final_trivia = Trivia([NewlineNode(text=pieces[last_nl].text)])

    @override
    def extend(self, values: Iterable[Any]) -> None:
        for v in values:
            self.append(v)

    @override
    def clear(self) -> None:
        self._value.items.clear()
        # Drop any inter-item trivia clutter; preserve the bracket
        # leading captured in final_trivia.
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
            pad, above = split_above_block(self._value.header_trivia)
            self._value.header_trivia = pad
            new_item = _make_item(cst, has_comma=True)
            old_first = items[0]
            old_first.leading = join_above_block(style.inter_separator, above)
            items.insert(0, new_item)
        else:
            # Internal insert: new item with leading = inter_sep; the
            # item that was at position i (now at i+1) keeps its old
            # leading (which already carries inter_sep + its own
            # above-block).
            new_item = _make_item(
                cst, leading=clone_trivia(style.inter_separator), has_comma=True
            )
            items.insert(i, new_item)
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
        # Capture style BEFORE swapping items so inter_sep reflects
        # the original layout.
        style = self._style()
        # Snapshot pad of header_trivia (everything bracket-side of the
        # above-pos-0 block) so it stays put across the reorder.
        head_pad, _head_above = split_above_block(self._value.header_trivia)
        leadings, eols = snapshot_comments(self)
        new_items = [items[j] for j in order]
        new_decoded = [self[j] for j in order]
        new_leadings = [leadings[j] for j in order]
        new_eols = [eols[j] for j in order]
        items[:] = new_items
        list.clear(self)
        for v in new_decoded:
            list.append(self, v)
        # Re-stamp leadings and post_comma_trivia per canonical model.
        # Above-blocks will be re-applied by ``apply_comments`` from
        # the snapshot.
        self._value.header_trivia = clone_trivia(head_pad)
        _restamp_canonical_leadings(items, style)
        for it in items:
            it.post_comma_trivia = Trivia()
            it.trailing = Trivia()
        _renormalise_commas(items, style)
        clear_all_comments(self)
        apply_comments(self, new_leadings, new_eols)

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
            new_csts = []
            new_decoded = []
            for v in values:
                cst, dec = self._synth_cst(v)
                new_csts.append(cst)
                new_decoded.append(dec)
            items = self._value.items
            style = self._style()
            new_segment = [_make_item(cst, has_comma=False) for cst in new_csts]
            items[index] = new_segment
            list.__setitem__(self, index, new_decoded)
            _restamp_canonical_leadings(items, style)
            _renormalise_commas(items, style)
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
        # Normalise to a sorted list of removed positions so we can
        # reason uniformly about delete-at-0 and tail removal.
        if isinstance(index, slice):
            removed = sorted(range(*index.indices(len(items))))
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
        last_idx = len(items) - 1
        zero_removed = 0 in removed
        tail_removed = last_idx in removed
        survivors_after_zero = [
            j for j in range(len(items)) if j > 0 and j not in removed
        ]
        # Capture per-item snapshots needed for boundary fix-ups.
        # When position 0 is removed, the new items[0] inherits a
        # leading whose above-block currently lives in items[k].leading
        # for the smallest surviving k > 0.
        new_first_above: Trivia = Trivia()
        if zero_removed and survivors_after_zero:
            k = survivors_after_zero[0]
            _new_pad, new_first_above = split_above_block(items[k].leading)
        # When the tail is removed, the surviving last item's
        # post_comma_trivia (if it had a comma) currently encodes a
        # post-comma bracket pad we need to recompute from style.
        is_multiline = (
            self._multiline
            or trivia_has_newline(self._value.header_trivia)
            or trivia_has_newline(self._value.final_trivia)
        )
        new_terminal_has_comma = False
        new_terminal_post: Trivia = Trivia()
        new_terminal_trailing: Trivia | None = None
        if tail_removed:
            new_last_idx = max(
                (j for j in range(len(items)) if j not in removed), default=-1
            )
            if new_last_idx >= 0:
                orig_terminal = items[last_idx]
                new_terminal_has_comma = orig_terminal.has_comma
                if orig_terminal.has_comma:
                    new_terminal_post = _structural_trailing_post(
                        orig_terminal.post_comma_trivia, multiline=is_multiline
                    )
                else:
                    new_terminal_trailing = clone_trivia(orig_terminal.trailing)
        del items[index]
        list.__delitem__(self, index)
        if not items:
            return
        if zero_removed:
            # The new items[0] absorbs nothing into its leading
            # (canonical empty); the above-block that previously sat
            # before it migrates into header_trivia.
            head_pad, _drop = split_above_block(self._value.header_trivia)
            self._value.header_trivia = join_above_block(head_pad, new_first_above)
            items[0].leading = Trivia()
        if tail_removed:
            items[-1].has_comma = new_terminal_has_comma
            items[-1].post_comma_trivia = (
                clone_trivia(new_terminal_post) if new_terminal_has_comma else Trivia()
            )
            if new_terminal_trailing is not None:
                items[-1].trailing = new_terminal_trailing

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
# Style detection + ArrayItem builders
# ---------------------------------------------------------------------------


class _ArrayStyle:
    """Inferred separator + trailing-comma policy for an Array."""

    __slots__ = ("inter_separator", "is_multiline", "trailing_comma", "trailing_post")

    def __init__(
        self,
        *,
        is_multiline: bool,
        inter_separator: Trivia,
        trailing_comma: bool,
        trailing_post: Trivia,
    ) -> None:
        self.is_multiline = is_multiline
        self.inter_separator = inter_separator
        self.trailing_comma = trailing_comma
        self.trailing_post = trailing_post


def _detect_style(value: ArrayValue | None, *, multiline_flag: bool) -> _ArrayStyle:

    if value is None:
        return _ArrayStyle(
            is_multiline=multiline_flag,
            inter_separator=Trivia([WhitespaceNode(text=" ")]),
            trailing_comma=multiline_flag,
            trailing_post=Trivia(),
        )
    items = value.items
    # Under the canonical model, multiline-ness is visible in
    # header_trivia / final_trivia / items[k>=1].leading. Sample bounded
    # canonical locations.
    has_newline = (
        trivia_has_newline(value.header_trivia)
        or trivia_has_newline(value.final_trivia)
        or (len(items) >= 2 and trivia_has_newline(items[1].leading))
        or (bool(items) and trivia_has_newline(items[-1].leading))
    )
    is_multiline = has_newline or multiline_flag
    # Inter-item separator: structural pad portion of items[1].leading
    # (the part before any above-item-1 comment block). For single-item
    # multiline arrays, synthesise from header_trivia indent — there's
    # no peer to sample from. (Inline tables can't be multiline so this
    # branch is array-specific; the shared helper handles the rest.)
    if is_multiline and len(items) < 2:
        nl_text = "\n"
        for p in value.header_trivia.pieces:
            if isinstance(p, NewlineNode):
                nl_text = p.text
                break
        else:
            for p in value.final_trivia.pieces:
                if isinstance(p, NewlineNode):
                    nl_text = p.text
                    break
        indent = _first_indent_after_newline(value.header_trivia)
        if not indent:
            indent = _indent_from_final_trivia(value.final_trivia) or "    "
        inter_sep = Trivia([NewlineNode(text=nl_text), WhitespaceNode(text=indent)])
    else:
        inter_sep = inter_item_separator(items)
    # Trailing-comma policy: the last item's has_comma if any.
    trailing_comma = items[-1].has_comma if items else is_multiline
    # Trailing post: pad portion of final_trivia (drops any
    # above-`]` comment block; that's an above-block belonging to a
    # would-be next item, not bracket pad).
    pad_ft, _above_ft = split_above_block(value.final_trivia)
    trailing_post = pad_ft if pad_ft.pieces else clone_trivia(value.final_trivia)
    if not items and is_multiline and not trailing_post.pieces:
        nl_text = "\n"
        trailing_post = Trivia([NewlineNode(text=nl_text)])
    return _ArrayStyle(
        is_multiline=is_multiline,
        inter_separator=inter_sep,
        trailing_comma=trailing_comma,
        trailing_post=trailing_post,
    )


def _first_indent_after_newline(trivia: Trivia) -> str:
    pieces = trivia.pieces
    for i, p in enumerate(pieces):
        if (
            isinstance(p, NewlineNode)
            and i + 1 < len(pieces)
            and isinstance(pieces[i + 1], WhitespaceNode)
        ):
            return str(pieces[i + 1].text)
    return ""


def _indent_from_final_trivia(ft: Trivia) -> str:
    """Extract a logical indent from `final_trivia` pieces.

    Prefers the indent of the last comment line (so a varied-indent
    or blank-line-prefixed comment block aligns the new item with
    the *most recent* commented line). Falls back to the indent of
    the last whitespace-after-newline block, then to "".
    """
    pieces = ft.pieces
    last_comment_indent: str | None = None
    last_ws_after_nl: str | None = None
    j = 0
    while j < len(pieces):
        if isinstance(pieces[j], NewlineNode) and j + 1 < len(pieces):
            ws = ""
            if isinstance(pieces[j + 1], WhitespaceNode):
                ws = str(pieces[j + 1].text)
                last_ws_after_nl = ws
                k = j + 2
            else:
                k = j + 1
            if k < len(pieces) and isinstance(pieces[k], CommentNode):
                last_comment_indent = ws
        j += 1
    if last_comment_indent is not None:
        return last_comment_indent
    return last_ws_after_nl or ""


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


def _restamp_canonical_leadings(items: list[ArrayItem], style: _ArrayStyle) -> None:
    """Reset every item's ``leading`` to the canonical-model pad.

    Under the canonical model ``items[0].leading`` is empty (its pad
    lives in ``header_trivia``) and ``items[k>=1].leading`` carries
    the inter-item separator. This helper is the structural reset
    used after a reorder or whole-segment slice replacement; comments
    are reapplied separately by the caller.
    """
    for k, it in enumerate(items):
        it.leading = Trivia() if k == 0 else clone_trivia(style.inter_separator)


def _migrate_eol_trailing_to_post_comma(item: ArrayItem) -> None:
    """If item.trailing carries an EOL section, move it to post_comma_trivia.

    Used when flipping from terminal (no comma) to internal (comma added):
    the EOL row that previously sat between the value and the closing `]`
    now logically follows the new comma.
    """
    eol, rest = split_eol_section(item.trailing)
    if eol.pieces:
        item.post_comma_trivia = eol
        item.trailing = rest


def _migrate_eol_post_comma_to_trailing(item: ArrayItem) -> None:
    """Inverse of _migrate_eol_trailing_to_post_comma."""
    eol, rest = split_eol_section(item.post_comma_trivia)
    if eol.pieces:
        item.trailing = Trivia(list(item.trailing.pieces) + list(eol.pieces))
        item.post_comma_trivia = rest


def _flip_to_internal(item: ArrayItem) -> None:
    """Make ``item`` look like an internal (non-last) item.

    Under the canonical model the inter-item separator lives in the
    NEXT item's leading; this function only ensures the comma is set
    and migrates any EOL comment from `trailing` into `post_comma_trivia`.
    """
    if not item.has_comma:
        _migrate_eol_trailing_to_post_comma(item)
        item.has_comma = True


def _flip_to_terminal(item: ArrayItem, style: _ArrayStyle) -> None:
    """Make ``item`` look like the terminal (last) item per style."""
    if style.trailing_comma:
        if not item.has_comma:
            _migrate_eol_trailing_to_post_comma(item)
            item.has_comma = True
        # When has_comma==True, post_comma_trivia carries any EOL the
        # parser/mutation already filed there; keep it intact.
        return
    # No trailing comma policy: drop the comma; migrate any EOL back
    # to trailing.
    if item.has_comma:
        _migrate_eol_post_comma_to_trailing(item)
        item.has_comma = False
        item.post_comma_trivia = Trivia()


def _structural_trailing_post(tp: Trivia, *, multiline: bool) -> Trivia:
    """Extract the structural trailing-post from a comma-terminated item.

    Mirrors `_detect_style`'s trailing_post computation but operates on
    a single ``post_comma_trivia`` so callers don't have to scan the
    whole array. In a multiline layout, drops any pre-newline comment
    block (which conceptually belonged to the popped item's row) and
    keeps the structural newline + bracket-pad indent.
    """
    if not multiline:
        return clone_trivia(tp)
    pieces_list = list(tp.pieces)
    last_nl = -1
    for j, p in enumerate(pieces_list):
        if isinstance(p, NewlineNode):
            last_nl = j
    if last_nl >= 0:
        return Trivia(list(pieces_list[last_nl:]))
    return clone_trivia(tp)


def _value_has_any_comment(val: Value) -> bool:
    items: list[ArrayItem] | list[InlineTableEntry]
    if isinstance(val, ArrayValue):
        items = val.items
    elif isinstance(val, InlineTableValue):
        items = val.entries
    else:
        return False
    if trivia_has_comment(val.header_trivia) or trivia_has_comment(val.final_trivia):
        return True
    return any(_item_has_any_comment(it) for it in items)


def _item_has_any_comment(item: CommaItem) -> bool:
    if (
        trivia_has_comment(item.leading)
        or trivia_has_comment(item.trailing)
        or trivia_has_comment(item.post_comma_trivia)
    ):
        return True
    return _value_has_any_comment(item.value)


def _renormalise_commas(items: list[ArrayItem], style: _ArrayStyle) -> None:
    """Reset has_comma + post_comma_trivia across ``items`` per style."""
    if not items:
        return
    for it in items[:-1]:
        _flip_to_internal(it)
    _flip_to_terminal(items[-1], style)


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
        for e in entries:
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
            typed_values: list[Mapping[str, Any]] = []
            for v in values:
                if not isinstance(v, Mapping):
                    msg = f"AoT entries must be Mapping/Table; got {type(v).__name__}"
                    raise TypeError(msg)
                typed_values.append(v)  # ty: ignore[invalid-argument-type]
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
        assert isinstance(value, Mapping)
        if self._layout_root is None:
            assert isinstance(value, dict)
            list.__setitem__(self, index, _make_unattached_entry(value))  # ty: ignore[invalid-argument-type]
            return
        self._replace_entry_attached(operator.index(index), value)  # ty: ignore[invalid-argument-type]

    @override
    def append(self, value: Table | Mapping[str, TomlInput]) -> None:
        # Same semantics as `add(body)` but with no return value (list API).
        if self._layout_root is None:
            assert isinstance(value, dict)
            list.append(self, _make_unattached_entry(value))
            return
        self._add_entry_attached(value)

    @override
    def extend(self, values: Iterable[Table | Mapping[str, TomlInput]]) -> None:
        for v in values:
            self.append(v)

    @override
    def insert(
        self, index: SupportsIndex, value: Table | Mapping[str, TomlInput]
    ) -> None:
        if self._layout_root is None:
            assert isinstance(value, dict)
            list.insert(self, index, _make_unattached_entry(value))
            return
        new_entry = self._add_entry_attached(value)
        idx = int(index)
        n = len(self)
        if idx < 0:
            idx = max(0, n + idx)
        idx = min(idx, n - 1)
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
    def sort(  # type: ignore[override]  # ty: ignore[invalid-method-override]
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
    from tomlrt._container import Table  # noqa: PLC0415

    t = Table()
    if body is not None:
        for k, v in body.items():
            dict.__setitem__(t, k, v)
    return t


__all__ = ["AoT", "Array"]
