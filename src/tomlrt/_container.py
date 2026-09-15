"""Logical container layer.

`Container(dict)` backs `Document` and `Table`. Dict storage follows
doc-stream first-occurrence order; mutations update the slot stream
through `_index`, `_refs`, `_header_ref`, and `_body_tail`.
"""

from __future__ import annotations

import contextlib
import copy
import sys
import warnings
from collections.abc import Mapping
from datetime import date, datetime, time
from typing import (
    TYPE_CHECKING,
    Any,
    Final,
    Literal,
    TypeAlias,
    TypeGuard,
    TypeVar,
    overload,
)

if sys.version_info >= (3, 12):
    from typing import Self, override
else:  # pragma: no cover -- backport for Python < 3.12
    from typing_extensions import override

from tomlrt import _inline_ops, _layout_ops
from tomlrt._comma_comments import (
    CommaEolView,
    CommaLeadingBlockView,
    CommaLeadingView,
)
from tomlrt._comments import (
    EolCommentView,
    LeadingBlockView,
    LeadingCommentView,
    _direct_kv_slot,
    _doc_epilogue_get,
    _doc_epilogue_set,
    _doc_preamble_get,
    _doc_preamble_set,
    _header_comment_get,
    _header_comment_set,
    _header_leading_block_get,
    _header_leading_block_set,
    _header_leading_get,
    _header_leading_set,
)
from tomlrt._errors import TOMLError
from tomlrt._format import (
    _prepare_indent,
    _resolve_format_options,
    format_document_trailing,
    format_inline_root,
    format_slots,
)
from tomlrt._inline_comments import _InlineAdapter
from tomlrt._kind import _Kind
from tomlrt._paths import validate_path
from tomlrt._render import render
from tomlrt._scalar import (
    CHECKED_SCALARS,
    PLAIN_SCALARS,
    coerce_scalar,
    is_scalar,
    validate_scalar,
)
from tomlrt._slots import KVSlot, StructuralHeaderSlot
from tomlrt._trivia import split_line
from tomlrt._typecheck import (
    _mapping_items,
    _require_mapping,
    _validate_key,
    _validate_mapping,
)
from tomlrt._values import (
    ArrayItem,
    ArrayValue,
    InlineTableEntry,
    InlineTableValue,
    is_shareable_scalar,
    make_keypart,
    retarget_value_newlines,
)
from tomlrt._view import _View, is_inline_value

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, MutableMapping, Sequence

    from _typeshed import SupportsKeysAndGetItem, SupportsRichComparison
    from typing_extensions import Self

    from tomlrt._format import FormatOptions
    from tomlrt._scalar import Scalar
    from tomlrt._slots import AoTEntry, Slot, SlotRef
    from tomlrt._values import Value


_T = TypeVar("_T")

_MISSING = object()


class Container(_View, dict[str, Any]):
    """Dict-typed base for `Document` and `Table` views.

    Reads are pure dict operations. Section mutations use the
    per-container cache (`_index`, `_refs`, `_header_ref`,
    `_body_tail`). Inline tables keep those caches empty and mutate
    the backing `InlineTableValue` in `_value`.
    """

    __slots__ = (
        "_body_tail",
        "_header_ref",
        "_host",
        "_index",
        "_inline",
        "_layout_root",
        "_owner_aot_entry",
        "_path",
        "_refs",
        "_value",
    )

    @override
    def _view_children(self) -> Iterable[object]:
        return self.values()

    @override
    def _reset_displaced(self) -> None:
        # The displacement root decides which internal bindings survive.
        # Per-node reset only forgets document-level attachment.
        assert self._inline
        _clear_inline_document_binding(self)

    def __init__(self) -> None:
        super().__init__()
        self._layout_root: Document | None = None
        self._path: tuple[str, ...] = ()
        self._inline = False
        self._host: Array | AoT | Container | None = None
        self._owner_aot_entry: AoTEntry | None = None
        self._index: dict[str, list[SlotRef]] = {}
        self._refs: list[SlotRef] = []
        self._header_ref: SlotRef | None = None
        self._body_tail: Slot | None = None
        self._value: InlineTableValue | None = None

    @property
    def _parent(self) -> Container | None:
        """The path-parent container, or ``None`` for an inline array element.

        An AoT entry's immediate host is its array; its path parent is
        the container holding that array. Assign :attr:`_host`, not this
        read-only projection.
        """
        host = self._host
        if isinstance(host, Container):
            return host
        return host._host if isinstance(host, AoT) else None  # noqa: SLF001

    @property
    def _kind(self) -> _Kind:
        """The shape this container is in. See :class:`_Kind`."""
        if self._inline:
            if self._value is not None:
                return _Kind.INLINE_ROOT
            if self._host is None:
                return _Kind.INLINE_FACTORY
            return _Kind.INLINE_DOTTED_INNER
        if self._header_ref is not None:
            return _Kind.SECTION
        return _Kind.IMPLICIT_SECTION

    @property
    def _needs_layout(self) -> bool:
        """Whether this is a factory still holding its data in dict storage.

        Such a container owns no slots and no inline value, so it renders
        nothing and structural edits are pure dict edits. Layout appears
        when it is attached to a document, or materialised in place by
        the first comment written on it.
        """
        if self._inline:
            return self._kind is _Kind.INLINE_FACTORY
        return self._layout_root is None

    @property
    def comments(self) -> MutableMapping[str, str]:
        """Mapping view of EOL comments on this container's direct keys.

        Comments may be set before attachment. Section tables expose direct
        key/value entries; inline tables expose direct leaf entries. Adding
        comments to a single-line inline table makes it multi-line.
        """
        if self._inline:
            return CommaEolView(_InlineAdapter(self))
        return EolCommentView(self)

    @property
    def leading_comments(self) -> MutableMapping[str, tuple[str, ...]]:
        """Mapping view of leading-comment blocks on this container's direct keys.

        Returns only the *attached* comment run immediately above each key
        (no blank line between). For the full block, including any
        above-blank groups and the blank-line structure between them, see
        [`leading_block`][tomlrt.Table.leading_block].

        Comments may be set before attachment. Inline tables expose direct
        leaf entries with the same attached-run semantics. Adding comments
        makes a single-line inline table multi-line.
        """
        if self._inline:
            return CommaLeadingView(_InlineAdapter(self))
        return LeadingCommentView(self)

    @property
    def leading_block(self) -> MutableMapping[str, tuple[str | None, ...]]:
        """Mapping view of full leading-trivia blocks on direct keys.

        Each entry is a ``tuple[str | None, ...]`` of comment strings
        interleaved with ``None`` (one per blank line), in source order;
        the slot's own column indent is implicit and re-applied on write.

        For the document's first key, the opening comment paragraph is the
        [`Document.preamble`][tomlrt.Document.preamble] and is omitted here;
        this block starts after the first blank line.

        Blocks may be set before attachment. Inline tables expose direct
        leaf entries; opening-bracket EOL comments are framing and are not
        part of the first entry's block.
        """
        if self._inline:
            return CommaLeadingBlockView(_InlineAdapter(self))
        return LeadingBlockView(self)

    @property
    def header_comment(self) -> str | None:
        """The EOL comment on this container's section header, or None.

        Section factories may be annotated before attachment.
        Document roots and implicit sections opened only by a nested
        ``[a.b]`` header read as ``None``. Setting on such a container raises
        [`TOMLError`][tomlrt.TOMLError]; inline tables also raise.
        """
        return _header_comment_get(self)

    @header_comment.setter
    def header_comment(self, value: str | None) -> None:
        _header_comment_set(self, value)

    @header_comment.deleter
    def header_comment(self) -> None:
        _header_comment_set(self, None)

    @property
    def header_leading_comments(self) -> tuple[str, ...]:
        """The attached comment block immediately above this container's header.

        Section factories may be annotated before attachment.
        Document roots and implicit sections opened only by a nested
        ``[a.b]`` header read as ``()``.
        Setting on such a container raises [`TOMLError`][tomlrt.TOMLError];
        inline tables also raise.

        Excludes any above-blank groups — those are visible via
        [`header_leading_block`][tomlrt.Table.header_leading_block].
        """
        return _header_leading_get(self)

    @header_leading_comments.setter
    def header_leading_comments(self, value: tuple[str, ...]) -> None:
        _header_leading_set(self, value)

    @header_leading_comments.deleter
    def header_leading_comments(self) -> None:
        _header_leading_set(self, ())

    @property
    def header_leading_block(self) -> tuple[str | None, ...]:
        """The full leading-trivia block above this container's header.

        A ``tuple[str | None, ...]`` of comment strings interleaved with
        ``None`` (one per blank line), in source order. Section factories
        may be annotated before attachment. Document roots and implicit
        sections opened only by a nested ``[a.b]`` header read as ``()``.
        Setting on such a container raises [`TOMLError`][tomlrt.TOMLError];
        inline tables also raise.

        For the document's first section, the opening comment paragraph is
        the [`Document.preamble`][tomlrt.Document.preamble] and is omitted
        here; this block starts after the first blank line.
        """
        return _header_leading_block_get(self)

    @header_leading_block.setter
    def header_leading_block(self, value: tuple[str | None, ...]) -> None:
        _header_leading_block_set(self, value)

    @header_leading_block.deleter
    def header_leading_block(self) -> None:
        _header_leading_block_set(self, ())

    def format(
        self,
        *,
        options: FormatOptions | None = None,
        comments: bool | None = None,
    ) -> None:
        """Canonicalise this container's formatting in place.

        Rewrites this subtree to the canonical layout:

        * Keys, ``=`` spacing, and header brackets use canonical
          whitespace.
        * Sibling key/value slots have no blank line between them;
          section / array-of-tables headers get one.
        * Orphan comment blocks above slots are preserved, with each
          blank-line run collapsed to one.
        * Inline values keep their shape (single-line stays single-line,
          multi-line stays multi-line), and a multi-line one closes on
          the row it starts on.
        * Newlines use the owning document's style.

        ``comments=`` is deprecated; use
        ``FormatOptions(normalize_comments=...)`` instead. Supplying both
        arguments raises ``ValueError``.

        Factory-style containers without layout yet (``Table.section()`` /
        ``Table.inline()``) and inline dotted
        navigators are unsupported and raise `TOMLError`.
        """
        resolved = _resolve_format_options(options=options, comments=comments)
        kind = self._kind
        nl = self._doc_newline
        if kind is _Kind.INLINE_ROOT:
            assert self._value is not None
            format_inline_root(
                self._value, nl=nl, options=resolved, host=_host_kv_slot(self)
            )
            return
        if kind in (_Kind.INLINE_FACTORY, _Kind.INLINE_DOTTED_INNER):
            msg = "format() is not supported on detached inline-table views"
            raise TOMLError(msg)
        doc = self._layout_root
        if doc is None:
            msg = "format() requires the container to have document layout"
            raise TOMLError(msg)
        whole_document = kind is _Kind.DOCUMENT
        # A preamble already supplies the document's opening separator.
        head_blank_cap = None
        if whole_document:
            assert isinstance(self, Document)
            head_blank_cap = 0 if self._preamble else 1
        for slots, owns_adjacent_gaps in self._format_scopes():
            format_slots(
                slots,
                nl=nl,
                options=resolved,
                owns_adjacent_gaps=owns_adjacent_gaps,
                head_blank_cap=head_blank_cap,
            )
        if whole_document:
            assert isinstance(self, Document)
            self._preamble = format_document_trailing(
                self._preamble, nl=nl, options=resolved
            )
            self._trailing = format_document_trailing(
                self._trailing, nl=nl, options=resolved
            )

    def _format_scopes(self) -> Iterator[tuple[list[Slot], bool]]:
        """The disjoint slot runs `format` canonicalises, and who owns each gap.

        A header-bearing receiver is one run: its own block, complete
        with its subtree. An implicit one owns only the outer-hosted
        dotted keys that spell it -- lines in someone else's block, so
        it does not own the gaps between them -- and contributes its
        first header-bearing descendants as runs of their own.
        """
        implicit = self._kind is _Kind.IMPLICIT_SECTION
        if not implicit:
            yield _layout_ops.owned_slots(self), True
            return
        yield _layout_ops.implicit_body_slots(self), False
        pending: list[Container] = [self]
        while pending:
            for child in pending.pop().values():
                if _is_section(child):
                    if child._header_ref is None:  # noqa: SLF001
                        pending.append(child)
                    else:
                        yield _layout_ops.owned_slots(child), True
                elif isinstance(child, AoT):
                    for entry in child:
                        yield _layout_ops.owned_slots(entry), True

    @property
    def _attached_doc(self) -> Document:
        """The owning ``Document``, asserting the container is attached.

        Most ``_layout_ops`` primitives require an attached target
        because they mutate the doc-stream linked list. This accessor
        narrows ``_layout_root`` from ``Document | None``.
        """
        lr = self._layout_root
        assert lr is not None, "container is not attached to a document"
        return lr

    def _wire(
        self,
        *,
        layout_root: Document | None,
        parent: Container | AoT | None,
        path: tuple[str, ...],
        owner: AoTEntry | None,
    ) -> None:
        """Set the four common attachment fields shared by every Container.

        Inline-specific bits (``_inline``, ``_value``) and section-specific
        bits (``_header_ref``, ``_body_tail``) are not touched — callers
        set them explicitly so the table's flavour is visible at the call
        site.
        """
        self._layout_root = layout_root
        self._host = parent
        self._path = path
        self._owner_aot_entry = owner

    # ------------------------------------------------------------------
    # Typed accessors
    # ------------------------------------------------------------------

    def table(self, key: str | Sequence[str]) -> Table:
        """Return the value at ``key`` typed as a `Table`.

        ``key`` may be a single name, a dotted-string path, or a
        sequence of names.
        """
        return self._typed_entry(key, Table, "a Table")

    def array(self, key: str | Sequence[str]) -> Array:
        """Return the value at ``key`` typed as an `Array`."""
        return self._typed_entry(key, Array, "an Array")

    def aot(self, key: str | Sequence[str]) -> AoT:
        """Return the value at ``key`` typed as an array-of-tables (`AoT`)."""
        return self._typed_entry(key, AoT, "an AoT")

    @overload
    def get_table(self, key: str | Sequence[str]) -> Table | None: ...
    @overload
    def get_table(self, key: str | Sequence[str], default: _T) -> Table | _T: ...
    def get_table(self, key: str | Sequence[str], default: object = None) -> object:
        """Like `table(key)` but returns ``default`` if the key is missing."""
        return self._typed_entry_or(key, Table, "a Table", default)

    @overload
    def get_array(self, key: str | Sequence[str]) -> Array | None: ...
    @overload
    def get_array(self, key: str | Sequence[str], default: _T) -> Array | _T: ...
    def get_array(self, key: str | Sequence[str], default: object = None) -> object:
        """Like `array(key)` but returns ``default`` if the key is missing."""
        return self._typed_entry_or(key, Array, "an Array", default)

    @overload
    def get_aot(self, key: str | Sequence[str]) -> AoT | None: ...
    @overload
    def get_aot(self, key: str | Sequence[str], default: _T) -> AoT | _T: ...
    def get_aot(self, key: str | Sequence[str], default: object = None) -> object:
        """Like `aot(key)` but returns ``default`` if the key is missing."""
        return self._typed_entry_or(key, AoT, "an AoT", default)

    def _typed_entry(self, key: str | Sequence[str], cls: type[_T], label: str) -> _T:
        v = self.entry(key)
        if not isinstance(v, cls):
            msg = f"value at {key!r} is {type(v).__name__}, not {label}"
            raise TypeError(msg)
        return v

    def _typed_entry_or(
        self, key: str | Sequence[str], cls: type[_T], label: str, default: object
    ) -> _T | object:
        try:
            return self._typed_entry(key, cls, label)
        except KeyError:
            return default

    def entry(self, key: str | Sequence[str]) -> Any:
        """Resolve a (possibly dotted) key path; raises ``KeyError`` if missing.

        Raises ``TypeError`` if descent passes through a non-table, and
        ``ValueError`` for an empty path or a path with empty segments.
        """
        parts = validate_path(key)
        cur: object = self
        for i, p in enumerate(parts):
            if not isinstance(cur, Container):
                msg = f"cannot descend into {parts[i - 1]!r}: not a table"
                raise TypeError(msg)
            if p not in cur:
                raise KeyError(p)
            cur = dict.__getitem__(cur, p)
        return cur

    def get_entry(self, key: str | Sequence[str], default: Any = None) -> Any:
        """Like `entry(key)` but returns ``default`` if the path is missing."""
        try:
            return self.entry(key)
        except KeyError:
            return default

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Materialise independent plain-Python data (recursive)."""
        out = _to_python(self)
        assert isinstance(out, dict)
        return out

    def _synth_local_value(self, key: str, value: TomlInput) -> tuple[Value, object]:
        """Synthesise ``value`` for direct storage under ``key`` on ``self``."""
        return _synth_value(
            value,
            layout_root=self._layout_root,
            parent=self,
            name=key,
            owner=self._owner_aot_entry,
        )

    def _materialise_layout(self, *, preserve_header: bool = False) -> None:
        """Give a factory backing layout without changing its public identity."""
        assert self._needs_layout
        _validate_input(self, inline_only=self._inline)
        if self._inline:
            _synth_value(self, layout_root=None, parent=None, name=None, owner=None)
        else:
            assert isinstance(self, Table)
            _layout_ops.materialise_section(self, preserve_header=preserve_header)

    def _prepare_comment_write(self, key: str, *, materialize: bool) -> bool:
        """Ready this container's layout for a comment write on ``key``.

        A factory materialises on its first substantive write. Returns
        ``False`` when it has none and the caller wants none: an empty
        assignment then has nothing to clear. Keys that carry no comment
        of their own — a nested section, a populated AoT, or no such key
        at all — are rejected before any layout is built.
        """
        if not self._needs_layout:
            return True
        value = dict.__getitem__(self, key)
        if _is_section(value) or (isinstance(value, AoT) and value):
            raise KeyError(key)
        if not materialize:
            return False
        self._materialise_layout()
        return True

    def _require_promotable_entry(self, key: str, *, action: str) -> object:
        """Return ``self[key]`` after the shared promotion pre-checks."""
        if self._inline:
            msg = f"{action} is not supported on inline tables"
            raise TOMLError(msg)
        if key not in self:
            msg = f"key {key!r} not in table"
            raise KeyError(msg)
        return dict.__getitem__(self, key)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    @override
    def __setitem__(self, key: str, value: Any) -> None:
        # Reject a bad key or value before they reach the layout
        # pipeline and fail later with an opaque error, or — worse —
        # after a structural overwrite has already torn down the old
        # binding.
        _validate_key(key)
        _validate_input(value, inline_only=self._inline, key=key)
        self._setitem_validated(key, value)

    def _setitem_validated(self, key: str, value: TomlInput) -> None:
        """Bind ``key`` to a ``value`` already checked for this container.

        The body of `__setitem__` below the validation boundary.
        `_validate_input` walks a value recursively, so re-entering
        `__setitem__` from a path that has already validated costs the
        whole walk again — quadratically so for the `_layout_ops`
        population loops, which re-enter once per structural child.
        """
        if key in self and self[key] is value:
            return
        if self._needs_layout:
            dict.__setitem__(self, key, value)
            return
        if self._inline:
            self._inline_setitem(key, value)
            return
        if key in self:
            self._overwrite_existing(key, value)
            return
        self._insert_new(key, value)

    def _overwrite_existing(self, key: str, value: TomlInput) -> None:
        """Replace the value at an already-bound key.

        Uses in-place swaps when the existing binding is a single KV.
        Structural values take the delete + reinsert + move-to-anchor
        path so the new view's Python identity becomes the live one.
        """
        current = dict.__getitem__(self, key)
        if is_scalar(current) and is_scalar(value):
            self._replace_scalar(key, value)
            return
        # Single-direct-KV-slot current → any synth-able value
        # (scalar or inline). The slot's `value` field is swapped
        # in place; ordering, comments, key spelling are preserved.
        if (
            is_scalar(current)
            or _is_inline_table(current)
            or isinstance(current, Array)
        ) and (is_scalar(value) or _is_synth_inline(value)):
            self._inline_typed_replace(key, value)
            return
        # Structural overwrite keeps the doc-stream anchor but detaches
        # old user references from the live doc: every value
        # `_validate_input` accepts and the branches above declined is
        # structural.
        value = _snapshot_for_overlapping_install(self, key, value)
        _layout_ops.reposition_install(self, key, value)

    def _insert_new(
        self,
        key: str,
        value: TomlInput,
        *,
        reinstall_as_dotted: bool = False,
    ) -> None:
        """Bind ``key`` for the first time at the document tail."""
        if is_scalar(value):
            # Deliberately not routed through `append_synth_kv`: that
            # costs three extra call frames and a duplicated `is_scalar`
            # test, measured at +6.5% on a scalar insert — the commonest
            # mutation there is.
            _layout_ops.append_direct_kv(
                self,
                key,
                coerce_scalar(value),
                reinstall_as_dotted=reinstall_as_dotted,
            )
            dict.__setitem__(self, key, value)
            return
        if _is_synth_inline(value):
            _layout_ops.append_synth_kv(
                self,
                key,
                value,
                reinstall_as_dotted=reinstall_as_dotted,
            )
            return
        if isinstance(value, AoT):
            self._attach_aot(key, value)
            return
        # `_validate_input` leaves only a section table for this branch.
        assert isinstance(value, Container)
        self._attach_section(key, value)

    def _attach_aot(self, key: str, value: AoT) -> None:
        """Install ``value`` (an AoT) under ``key``.

        Public sources are copied. Private layout moves with its live
        views; entries without layout are synthesized in place.
        """
        src_root = value._layout_root  # noqa: SLF001
        if src_root is not None and not _can_adopt_from(src_root, self._attached_doc):
            _layout_ops.clone_aot(self, key, value)
            return
        snapshot = _snapshot_for_overlapping_install(self, key, value)
        if snapshot is not value:
            assert isinstance(snapshot, AoT)
            _layout_ops.clone_aot(self, key, snapshot)
            return
        emptied = value._host  # noqa: SLF001
        existing_entries: list[Table] = list(value)
        _layout_ops.detach_aot_from_orphan(value)
        list.clear(value)
        attached = _layout_ops.attach_empty_aot(self, key, value)
        dict.__setitem__(self, key, attached)
        for entry_table in existing_entries:
            source_doc = entry_table._layout_root  # noqa: SLF001
            if source_doc is None:
                _layout_ops.add_aot_entry(value, None, rehome=entry_table)
            elif _can_adopt_from(source_doc, self._attached_doc):
                source_parent = (
                    emptied if src_root is not None else entry_table._parent  # noqa: SLF001
                )
                _layout_ops.adopt_private_entry(
                    value,
                    entry_table,
                    preserve_source_separator=src_root is not None,
                )
                _layout_ops.synthesise_header_for_emptied(source_parent)
            else:
                _layout_ops.add_aot_entry(value, entry_table)
        _layout_ops.synthesise_header_for_emptied(emptied)

    def _attach_section(self, key: str, source: Container) -> None:
        """Install ``source`` (a section-flavoured Table) under ``key``.

        As with AoTs, synthesize factories, adopt private layout, and clone
        public sources. Overlapping sources are snapshotted before moving.
        """
        snapshot = _snapshot_for_overlapping_install(self, key, source)
        assert isinstance(snapshot, Container)
        value: Container = snapshot
        src_root = value._layout_root
        if src_root is None:
            assert isinstance(value, Table), "a section factory must be a Table"
            _layout_ops.attach_section_at(self, (key,), value)
        elif _can_adopt_from(src_root, self._attached_doc):
            assert value._refs, "a private section owns slots"
            emptied = value._parent
            if value._header_ref is not None:
                _layout_ops.adopt_private_section(self, key, value)
            else:
                _layout_ops.adopt_private_implicit(self, key, value)
            _layout_ops.synthesise_header_for_emptied(emptied)
        elif value._header_ref is not None or isinstance(value, Document):
            _layout_ops.clone_section(self, key, value)
        else:
            _layout_ops.clone_implicit_section(self, key, value)

    def _replace_scalar(self, key: str, value: Scalar) -> None:
        """Replace a scalar while preserving its existing KV slot."""
        refs = self._index.get(key)
        assert refs is not None, "scalar value must have a slot"
        slot = refs[0].slot
        assert isinstance(slot, KVSlot), "scalar value must have a KV slot"
        slot.value = coerce_scalar(value)
        dict.__setitem__(self, key, value)

    def _inline_typed_replace(self, key: str, value: TomlInput) -> None:
        """Swap an existing direct-KV slot's value to a synthesised inline value.

        Works for any existing scalar / inline-table / inline-array
        binding backed by a single direct-KV slot (dotted or not).

        If the displaced value is itself a typed view (inline Table,
        Array), its attachment state is cleared so a later assignment
        elsewhere re-attaches live instead of cloning.
        """
        refs = self._index.get(key)
        assert refs is not None, "inline value must have a slot"
        assert len(refs) == 1, "inline value must be backed by exactly one slot"
        primary = refs[0]
        slot = primary.slot
        assert isinstance(slot, KVSlot), "inline value must be backed by a KV slot"
        old = dict.__getitem__(self, key)
        cst, decoded = self._synth_local_value(key, value)
        slot.value = cst
        dict.__setitem__(self, key, decoded)
        # Free the displaced root while preserving any CST-owning subtree
        # beneath it. Safe because `_setitem_validated` has already
        # returned if the new value *is* the old one.
        _layout_ops.reset_displaced_views(old)

    @override
    def __delitem__(self, key: str) -> None:
        if self._needs_layout:
            dict.__delitem__(self, key)
            return
        if self._inline:
            self._inline_delitem(key)
            return
        _layout_ops.delete_key(self, key, materialise_empty=True)

    # ------------------------------------------------------------------
    # Dict-method overrides — route through ``__setitem__`` /
    # ``__delitem__`` so inline / section / headerless dispatch is uniform.
    # ------------------------------------------------------------------

    @override
    def clear(self) -> None:
        for k in list(dict.keys(self)):
            del self[k]

    @override
    def pop(self, key: object, default: Any = _MISSING) -> Any:
        if key in self and isinstance(key, str):
            value = dict.__getitem__(self, key)
            del self[key]
            return value
        if default is _MISSING:
            raise KeyError(key)
        return default

    @override
    def popitem(self) -> tuple[str, Any]:
        try:
            key = next(reversed(self))
        except StopIteration:
            msg = "dictionary is empty"
            raise KeyError(msg) from None
        value = dict.__getitem__(self, key)
        del self[key]
        return key, value

    @override
    def update(self, *args: Any, **kwargs: Any) -> None:
        if len(args) > 1:
            msg = f"update expected at most 1 argument, got {len(args)}"
            raise TypeError(msg)
        items: list[tuple[str, object]] = []
        if args:
            other = args[0]
            if hasattr(other, "keys"):
                items = [(k, other[k]) for k in other.keys()]  # noqa: SIM118
            else:
                items = list(other)
        items.extend(kwargs.items())
        # Updating from a mapping must not consume it, and reading it
        # once up front keeps an install that unbinds one of its keys
        # from cutting the read short.
        with _sources_kept_intact(self._layout_root, (v for _, v in items)):
            for k, v in items:
                self[k] = v

    @override
    def setdefault(self, key: str, default: Any = None) -> Any:
        if key in self:
            return dict.__getitem__(self, key)
        self[key] = default
        return dict.__getitem__(self, key)

    def sort(
        self,
        *,
        key: Callable[[str], SupportsRichComparison] | None = None,
        reverse: bool = False,
    ) -> None:
        """Sort direct child keys in place, preserving per-key trivia.

        Mirrors ``list.sort``: keyword-only ``key`` / ``reverse``,
        stable, in-place. Structural keys (children bound to an ``AoT``
        or to a section ``Table``, i.e. one rendered with a ``[header]``)
        are always placed after bare keys; otherwise a bare key after a
        section header would re-bind under it. ``key`` and ``reverse``
        apply within, never across, the partitions. Implicit sections
        built from dotted keys (e.g. ``a.x = 1``) sort as bare keys.

        Inline containers have no structural children, so the partition
        is a no-op and ``key`` / ``reverse`` behave as on a plain dict.

        See [`has_header`][tomlrt.Table.has_header] for the predicate
        that defines the partition; a custom ``key`` function can call it
        to decide which side of the split a given child sits on.
        """
        current = list(dict.keys(self))
        if len(current) <= 1:
            return
        if self._inline:
            new_order = sorted(current, key=key, reverse=reverse)
        else:
            # A key's leaf content must precede every section header.
            # Keep mixed keys between pure leaves and pure sections so
            # their leaf part stays ahead of all sections. Each key is
            # classified in one pass; `has_header` is only consulted for
            # keys that have a leaf, since it is only needed to tell
            # pure leaves from mixed ones.
            pure_leaves: list[str] = []
            mixed: list[str] = []
            pure_sections: list[str] = []
            for k in current:
                has_leaf = False
                has_header = False
                for ref in self._index.get(k, ()):
                    if isinstance(ref.slot, KVSlot):
                        has_leaf = True
                    else:
                        assert isinstance(ref.slot, StructuralHeaderSlot)
                        has_header = True
                    if has_leaf and has_header:
                        break
                if not has_leaf:
                    pure_sections.append(k)
                elif has_header:
                    mixed.append(k)
                else:
                    pure_leaves.append(k)
            for group in (pure_leaves, mixed, pure_sections):
                group.sort(key=key, reverse=reverse)
            new_order = pure_leaves + mixed + pure_sections
        if new_order == current:
            return
        if not self._needs_layout:
            if self._inline:
                _inline_ops.reorder_inline(self, new_order)
            else:
                _layout_ops.reorder_container(self, new_order)
        _reorder_dict_storage(self, new_order)

    def has_header(self, key: str) -> bool:
        """Whether ``key``'s rendered block contains a structural header.

        This describes the whole block, not the child table itself: with
        ``[a.b]``, ``doc.has_header("a")`` is true although ``a`` is implicit
        and only ``b`` owns the header. Structural headers are ``[header]``
        sections and ``[[header]]`` array-of-tables entries. Returns ``False``
        for bare ``key = value`` leaves, inline tables, implicit sections built
        entirely from dotted keys (e.g. ``a.x = 1``), and missing keys.
        """
        refs = self._index.get(key, ())
        # A plain loop measurably beats any()+generator here; sort()
        # calls this once per key, so it's hot for wide containers.
        for r in refs:  # noqa: SIM110
            if isinstance(r.slot, StructuralHeaderSlot):
                return True
        return False

    # Broader than dict's union signature by design: mutation accepts the same
    # keys/getitem objects and key-value iterables as dict.update().
    @override
    def __ior__(  # type: ignore[override]
        self,
        other: SupportsKeysAndGetItem[str, Any] | Iterable[tuple[str, Any]],
        /,
    ) -> Self:
        self.update(other)
        return self

    @override
    def __copy__(self) -> Container:
        _validate_input(self, inline_only=self._inline)
        cloned = _copy_input(self)
        assert isinstance(cloned, Container)
        return cloned

    # ------------------------------------------------------------------
    # Inline-table dispatch
    # ------------------------------------------------------------------

    def _inline_setitem(self, key: str, value: TomlInput) -> None:
        # ``__setitem__`` has already rejected values an inline host
        # cannot store (``AoT``, sections, non-coerceable types).
        cst, decoded = self._synth_local_value(key, value)
        if key in self:
            # Overwriting displaces the old value, so a view of it must
            # stop resolving against the entry it no longer owns.
            # ``__setitem__`` has already returned if the new value *is*
            # the old one.
            _layout_ops.reset_displaced_views(dict.__getitem__(self, key))
            _inline_ops.overwrite_entry(self, key, cst)
        else:
            _inline_ops.append_entry(self, key, cst)
        dict.__setitem__(self, key, decoded)

    def _inline_delitem(self, key: str) -> None:
        if key not in self:
            raise KeyError(key)
        # A held view of the deleted entry must stop resolving against
        # the CST it no longer owns, or a later write through it would
        # resurrect the key beside whatever replaced it.
        _layout_ops.reset_displaced_views(dict.__getitem__(self, key))
        _inline_ops.delete_entry(self, key)
        dict.__delitem__(self, key)
        # Empty dotted-prefix navigators have no backing CST entry; prune
        # them from the dict chain too.
        cur: Container | None = self
        while (
            cur is not None
            and cur._kind is _Kind.INLINE_DOTTED_INNER  # noqa: SLF001
            and len(cur) == 0
            and cur._path  # noqa: SLF001
        ):
            parent = cur._parent  # noqa: SLF001
            assert parent is not None  # implied by INLINE_DOTTED_INNER
            my_key = cur._path[-1]  # noqa: SLF001
            # `cur` is only ever filed under `parent` by this same prune
            # loop or by initial build, both of which keep the two in
            # lockstep, so `my_key` is always still present here.
            dict.__delitem__(parent, my_key)
            cur = parent

    def install(self, path: str | Sequence[str], value: TomlInput) -> Any:
        """Set ``value`` at the (possibly dotted) ``path``.

        Existing parents retain their form for scalar and inline values.
        Installing a section-style container or `AoT` from an attached section
        or document promotes inline ancestors as needed. An inline receiver
        or detached inline ancestor cannot be promoted this way.

        Returns the stored value or live view. Rejected paths and values
        leave the document unchanged.
        """
        parts = validate_path(path)
        cur, i = _walk_existing_tables(self, parts[:-1], action="install")
        promote = (
            cur._inline  # noqa: SLF001
            and not self._inline
            and self._layout_root is not None
            and (_is_section(value) or isinstance(value, AoT))
        )
        _validate_input(
            value,
            inline_only=cur._inline and not promote,  # noqa: SLF001
            key=parts[-1],
        )
        if promote:
            prefix = parts[:i]
            _check_table_promotions(self, prefix)
            value = _layout_ops._capture_input(value, [cur], {})  # noqa: SLF001
            cur = _promote_tables(self, prefix)
        if i == len(parts) - 1:
            host = cur
        elif cur._inline:  # noqa: SLF001
            first, host = _make_inline_chain(parts[i + 1 : -1])
            # Read the source before publishing any new parents, even
            # when it contains the inline table being extended.
            dict.__setitem__(host, parts[-1], value)
            cur._setitem_validated(parts[i], first)  # noqa: SLF001
            return host[parts[-1]]
        elif self._layout_root is not None and (
            _is_section(value) or isinstance(value, AoT)
        ):
            # Only the installed value needs an explicit header.
            host = _layout_ops.ensure_implicit_chain(cur, tuple(parts[i:-1]))
        else:
            host = cur._create_section_chain(parts[i:-1])  # noqa: SLF001
        host._setitem_validated(parts[-1], value)  # noqa: SLF001
        return host[parts[-1]]

    def ensure_table(
        self, key: str | Sequence[str], *, promote_inline: bool | None = None
    ) -> Table:
        """Return the table at ``key``, creating it if missing.

        Existing section and inline tables are traversed without changing
        their representation. A missing child of an inline table is created
        inline; elsewhere, missing intermediate components stay implicit and
        only the deepest component gets an explicit ``[a.b.c]`` header.
        Raises `TOMLError` if an existing component is an array-of-tables or
        non-table value.

        ``promote_inline`` is deprecated and ignored. Use `promote_inline()`
        to request conversion explicitly.
        """
        parts = validate_path(key)
        if promote_inline is not None:
            warnings.warn(
                "ensure_table(promote_inline=...) is deprecated and ignored; "
                "use promote_inline() for explicit conversion",
                DeprecationWarning,
                stacklevel=2,
            )
        cur, i = _walk_existing_tables(self, parts, action="ensure_table")
        if i == len(parts):
            assert isinstance(cur, Table)
            return cur
        if cur._inline:  # noqa: SLF001
            first, deepest = _make_inline_chain(parts[i + 1 :])
            cur._setitem_validated(parts[i], first)  # noqa: SLF001
            return deepest
        return cur._create_section_chain(parts[i:])  # noqa: SLF001

    def _create_section_chain(self, parts: Sequence[str]) -> Table:
        """Create a wholly missing section path beneath this section."""
        if self._layout_root is None:
            cur: Container = self
            for p in parts:
                child = Table.section()
                dict.__setitem__(cur, p, child)
                cur = child
            assert isinstance(cur, Table)
            return cur
        attached = _layout_ops.attach_section_at(self, tuple(parts), Table.section())
        assert isinstance(attached, Table)
        return attached

    def promote_inline(self, key: str) -> Table:
        """Convert an inline-table entry at ``key`` into a section header.

        Returns the live view at ``key`` after promotion. Raises
        ``KeyError`` if the key is missing, or `TOMLError` if it
        doesn't refer to an inline-style table.
        """
        cur = self._require_promotable_entry(key, action="inline-table promotion")
        if not _is_inline_table(cur):
            msg = f"{key!r} is not an inline table"
            raise TOMLError(msg)
        _check_inline_promotable(cur, key)
        return self._promote_inline_entry(key, cur)

    def _promote_inline_entry(self, key: str, cur: Container) -> Table:
        """Promote already-checked inline table ``cur`` at ``key``."""
        value = cur._value
        assert isinstance(value, InlineTableValue)
        entries = _layout_ops.prepare_promoted_inline_entries(value.items)
        # Transfer the existing KV slot's leading + eol to the header.
        saved_leading, saved_eol = _direct_kv_trivia(self, key)
        _layout_ops.delete_key(self, key)
        self[key] = Table.section()
        result = dict.__getitem__(self, key)
        assert isinstance(result, Table)
        _layout_ops.populate_promoted_inline_entries(result, entries)
        header_ref = result._header_ref  # noqa: SLF001
        assert header_ref is not None
        new_header = header_ref.slot
        assert isinstance(new_header, StructuralHeaderSlot)
        _layout_ops.restore_captured_leading(new_header, saved_leading, from_kv=True)
        new_header.eol = saved_eol
        return result

    def promote_array(self, key: str) -> AoT:
        """Convert an array-of-inline-tables at ``key`` into an AoT.

        Returns the live AoT view at ``key``. Raises ``KeyError`` if the
        key is missing, or `TOMLError` if it refers to a non-array, an
        empty array, or an array with non-inline-table elements.
        """
        cur = self._require_promotable_entry(key, action="array-of-tables promotion")
        if not isinstance(cur, Array):
            msg = f"{key!r} is not an array"
            raise TOMLError(msg)
        if len(cur) == 0:
            msg = f"cannot promote empty array {key!r}"
            raise TOMLError(msg)
        for el in cur:
            if not (_is_inline_table(el)):
                msg = f"{key!r} contains a non-inline-table element"
                raise TOMLError(msg)
        if cur._value.has_own_comment():  # noqa: SLF001
            msg = f"cannot promote {key!r}: array has comments that would be lost"
            raise TOMLError(msg)
        for entry_view in cur:
            ev = entry_view._value  # noqa: SLF001
            if ev is not None and ev.has_own_comment():
                msg = (
                    f"cannot promote {key!r}: array entry has inner "
                    f"comments that would be lost"
                )
                raise TOMLError(msg)
        value = cur._value  # noqa: SLF001
        assert isinstance(value, ArrayValue)
        entries: list[list[tuple[InlineTableEntry, Value]]] = []
        for item in value.items:
            assert isinstance(item.value, InlineTableValue)
            entries.append(
                _layout_ops.prepare_promoted_inline_entries(item.value.items)
            )
        # Carry the original KV slot's leading/eol onto the promoted AoT.
        saved_leading, saved_eol = _direct_kv_trivia(self, key)
        _layout_ops.delete_key(self, key)
        self[key] = AoT()
        result = dict.__getitem__(self, key)
        assert isinstance(result, AoT)
        for body in entries:
            entry = _layout_ops.add_aot_entry(result, None)
            _layout_ops.populate_promoted_inline_entries(entry, body)
        # Apply saved leading to the first entry header and saved eol to
        # the last entry's last slot.
        first_record = result[0]._owner_aot_entry  # noqa: SLF001
        assert first_record is not None
        _layout_ops.restore_captured_leading(
            first_record.header, saved_leading, from_kv=True
        )
        last_slot = result[-1]._body_tail  # noqa: SLF001
        assert last_slot is not None
        if "#" in saved_eol and "#" not in last_slot.eol:
            pre, comment, _term = split_line(saved_eol)
            last_slot.eol = f"{pre}{comment}{split_line(last_slot.eol)[2]}"
        return result


def _reorder_dict_storage(c: Container, new_key_order: list[str]) -> None:
    """Reorder ``c``'s dict storage in place to match ``new_key_order``.

    Bypasses ``Container.__setitem__`` so no validation, slot rebuild,
    or attach paths fire. ``new_key_order`` is trusted to be a
    permutation of ``dict.keys(c)``.
    """
    values = [(k, dict.__getitem__(c, k)) for k in new_key_order]
    dict.clear(c)
    for k, v in values:
        dict.__setitem__(c, k, v)


def _direct_kv_trivia(c: Container, key: str) -> tuple[str, str]:
    """Return the direct-KV slot's leading/EOL trivia for ``key``.

    Both fields are non-Optional on `KVSlot`, so the result is never
    ``None`` in either component.
    """
    slot = _direct_kv_slot(c, key)
    assert slot is not None, "inline promotion source must have a direct KV slot"
    return slot.leading, slot.eol


def _check_table_promotions(start: Container, parts: Sequence[str]) -> None:
    """Check promotion of an existing table path without changing its views."""
    cur: Container = start
    for p in parts:
        cur = dict.__getitem__(cur, p)
        if cur._inline:  # noqa: SLF001
            _check_inline_promotable(cur, p)


def _promote_tables(start: Container, parts: Sequence[str]) -> Container:
    """Promote a checked path after the caller has captured any input values."""
    # Promotion can replace descendant views, so resolve each step again.
    cur = start
    for p in parts:
        nxt = dict.__getitem__(cur, p)
        cur = cur._promote_inline_entry(p, nxt) if nxt._inline else nxt  # noqa: SLF001
    return cur


def _walk_existing_tables(
    start: Container,
    parts: Sequence[str],
    *,
    action: str,
) -> tuple[Container, int]:
    """Walk an existing prefix through section and inline tables."""
    cur = start
    for i, p in enumerate(parts):
        if p not in cur:
            return cur, i
        nxt = dict.__getitem__(cur, p)
        if isinstance(nxt, AoT):
            msg = (
                f"cannot {action} through array-of-tables at {p!r}: "
                "no addressable target inside an AoT"
            )
            raise TOMLError(msg)
        if not isinstance(nxt, Container):
            msg = f"cannot {action} through {p!r}: existing value is not a table"
            raise TOMLError(msg)
        cur = nxt
    return cur, len(parts)


def _make_inline_chain(parts: Sequence[str]) -> tuple[Table, Table]:
    """Build an unpublished inline root and its deepest descendant."""
    root = Table.inline()
    deepest = root
    for p in parts:
        child = Table.inline()
        dict.__setitem__(deepest, p, child)
        deepest = child
    return root, deepest


def _populate_unattached(t: Container, mapping: Mapping[str, TomlInput]) -> None:
    """Populate an unattached ``Container`` whose keys are already validated."""
    for k, v in _mapping_items(mapping):
        dict.__setitem__(t, k, v)


class Table(Container):
    """A logical TOML table.

    Every nested mapping in a document is a [`Table`][tomlrt.Table].
    `Table` is a `dict` subclass, so ``isinstance(t, dict)`` holds
    and it can be passed wherever a `dict` or `Mapping` is expected.

    The same `Table` class backs both standard ``[section]`` blocks
    and inline ``{x = 1}`` tables. Use [`is_inline`][tomlrt.Table.is_inline]
    to tell them apart when walking a parsed document.
    """

    __slots__ = ()

    @property
    def is_inline(self) -> bool:
        """True for inline ``{...}`` tables, False for ``[section]`` blocks."""
        return self._inline

    def _require_inline_root(self, action: str) -> Container:
        """Return the `INLINE_ROOT` backing this table, else raise.

        ``multiline`` is a property of the whole physical inline value, so
        a dotted-key navigator view is rejected in favour of calling on
        the table itself.
        """
        kind = self._kind
        if kind is _Kind.INLINE_ROOT:
            return self
        if not self._inline:
            msg = f"{action} is only available on inline tables"
            raise TOMLError(msg)
        if kind is _Kind.INLINE_FACTORY:
            msg = (
                f"{action} is unavailable on a detached inline table; "
                "attach it to a Document first"
            )
            raise TOMLError(msg)
        msg = (
            f"{action} applies to the whole inline table; call it on the "
            "table itself, not on a dotted-key view"
        )
        raise TOMLError(msg)

    @property
    def multiline(self) -> bool:
        """Whether this inline table is laid out across multiple lines.

        Raises [`TOMLError`][tomlrt.TOMLError] on a non-inline table.
        """
        root = self._require_inline_root("multiline")
        assert root._value is not None  # noqa: SLF001
        return root._value.is_multiline()  # noqa: SLF001

    @multiline.setter
    def multiline(self, value: bool) -> None:
        if self.multiline == value:
            return
        self.set_multiline(multiline=value)

    def set_multiline(self, *, multiline: bool, indent: int = 4) -> Table:
        """Switch this inline table between single-line and multi-line form.

        When laying out multi-line, entries are indented by ``indent``
        spaces and the closing brace lines up with the row the table
        starts on.

        Raises [`TOMLError`][tomlrt.TOMLError] on a non-inline table, and
        when collapsing a multi-line table that carries comments anywhere
        in it (they would have nowhere to live on one line).

        Returns ``self`` for chaining.
        """
        root = self._require_inline_root("set_multiline")
        _inline_ops.set_inline_multiline(
            root,
            multiline=multiline,
            indent=_prepare_indent(indent) if multiline else "",
        )
        return self

    @classmethod
    def _factory(
        cls, mapping: Mapping[str, TomlInput] | None, *, inline: bool, label: str
    ) -> Table:
        t = cls()
        t._inline = inline
        if mapping is not None:
            mapping = _validate_mapping(mapping, label=label)
            _populate_unattached(t, mapping)
        return t

    @classmethod
    def section(cls, mapping: Mapping[str, TomlInput] | None = None) -> Table:
        """Return a standard-section table, optionally populated from ``mapping``.

        Assign the result to install a ``[k]`` block:

            doc[k] = Table.section({"x": 1})
        """
        return cls._factory(mapping, inline=False, label="Table.section argument")

    @classmethod
    def inline(cls, mapping: Mapping[str, TomlInput] | None = None) -> Table:
        """Return an inline table, optionally populated from ``mapping``.

        Assign the result to install a ``{x = 1}`` value:

            doc[k] = Table.inline({"x": 1})
        """
        return cls._factory(mapping, inline=True, label="Table.inline argument")


DEFAULT_NEWLINE: Final = "\n"
"""Line ending a document uses until a parse tells it otherwise."""


class Document(Container):
    """Top-level TOML document.

    A [`Document`][tomlrt.Document] is the root of a parsed TOML
    file. It is a `dict` subclass and can be passed wherever a
    `dict` or `Mapping` is expected.
    """

    __slots__ = (
        "_head",
        "_install_recorders",
        "_is_private",
        "_newline",
        "_preamble",
        "_prelude",
        "_protected_source_roots",
        "_section_blank_separated",
        "_tail",
        "_trailing",
    )

    def __init__(self, data: Mapping[str, Any] | None = None) -> None:
        """Return a fresh empty document, optionally populated from ``data``.

        With a mapping:

        * nested mappings become standard ``[section]`` blocks (not
          inline tables);
        * lists of mappings become ``[[array.of.tables]]`` blocks;
        * everything else is set with ordinary key-value assignment.

        Constructing copies. Every value in ``data`` is copied into the
        new document, so later mutations through your own references
        are not visible in it -- and a rejected ``data`` leaves them all
        alone. A [`Table`][tomlrt.Table] / [`Array`][tomlrt.Array] /
        [`AoT`][tomlrt.AoT] contributes its contents and its shape
        (section or inline, array or array-of-tables). Available comments
        and spacing are preserved, including on entries of lists and
        standalone arrays-of-tables.

        To have the document keep your object, assign it instead:
        ``doc[k] = table`` *attaches live*. See
        [Editing documents](editing.md#live-vs-snapshot).
        """
        super().__init__()
        self._head: Slot | None = None
        self._tail: Slot | None = None
        self._trailing: str = ""
        self._preamble: str = ""
        self._newline: str = DEFAULT_NEWLINE
        self._prelude: str = ""
        self._is_private: bool = False
        self._protected_source_roots: dict[int, Document] | None = None
        self._install_recorders: (
            tuple[
                list[Slot],
                list[tuple[Slot, str, Slot | None]],
            ]
            | None
        ) = None
        self._section_blank_separated = True
        self._layout_root = self
        if data is None:
            return
        if isinstance(data, Document):
            from tomlrt._build import populate_cloned_document  # noqa: PLC0415

            populate_cloned_document(self, data)
            return
        if _has_extractable_layout(data):
            from tomlrt._build import populate_extracted_document  # noqa: PLC0415

            populate_extracted_document(self, data)
            return
        from tomlrt._synth import populate  # noqa: PLC0415

        populate(self, data)

    @property
    @override
    def _kind(self) -> _Kind:
        return _Kind.DOCUMENT

    def render(self) -> str:
        """Serialize the document back to a TOML string.

        Equivalent to `tomlrt.dumps(self)`.
        """
        return render(self)

    @property
    def preamble(self) -> tuple[str, ...]:
        """The document's opening comment paragraph, as bare comment texts.

        The preamble is the run of ``# …`` lines before the first blank
        line. Comments below that blank line belong to the first key or
        section (its `leading_comments` / `leading_block`), not the
        preamble.

        Setting replaces the preamble with a sequence of comment texts
        (without the leading ``#``); assign ``()`` to remove. Line
        terminators within a comment are rejected.
        """
        return _doc_preamble_get(self)

    @preamble.setter
    def preamble(self, value: tuple[str, ...]) -> None:
        _doc_preamble_set(self, value)

    @preamble.deleter
    def preamble(self) -> None:
        _doc_preamble_set(self, ())

    @property
    def epilogue(self) -> tuple[str | None, ...]:
        """Comment block at the very end of the document.

        The trailing comments that follow all structural content, as bare
        comment texts (without the leading ``#``) with ``None`` for each
        blank line, in source order. With no structural content everything
        is `preamble` instead.

        Setting replaces the epilogue with the same shape; assign ``()`` to
        remove. Line terminators within a comment are rejected.

        Raises [`TOMLError`][tomlrt.TOMLError] if called with a non-empty
        value on a document with no structural content.
        """
        return _doc_epilogue_get(self)

    @epilogue.setter
    def epilogue(self, value: tuple[str | None, ...]) -> None:
        _doc_epilogue_set(self, value)

    @epilogue.deleter
    def epilogue(self) -> None:
        _doc_epilogue_set(self, ())

    @override
    def __copy__(self) -> Document:
        return Document(self)


def _check_inline_promotable(v: Container, key: str) -> None:
    """Raise `TOMLError` if promoting ``v`` (bound to ``key``) would lose comments.

    Callers are expected to have already confirmed ``v`` is an inline
    table (e.g. via `_is_inline_table`).
    """
    value = v._value  # noqa: SLF001
    if value is not None and value.has_own_comment():
        msg = (
            f"cannot promote {key!r}: inline table has inner "
            f"comments that would be lost"
        )
        raise TOMLError(msg)


def _detached_inline_value(v: Container | Array) -> Value | None:
    """An independent copy of ``v``'s backing inline value, if it has one.

    An inline root or an `Array` owns its value outright. A dotted
    navigator — the ``a`` in ``{a.b = 1}`` — owns a slice of its
    outermost table's value, and that slice is extracted here. A factory
    has no value yet, so callers build one from its items instead.
    """
    if v._value is not None:  # noqa: SLF001
        return copy.deepcopy(v._value)  # noqa: SLF001
    # Arrays always own a value, so only a Table reaches here.
    if v._kind is _Kind.INLINE_DOTTED_INNER:  # noqa: SLF001
        return _inline_ops.copy_dotted_table(v)
    return None


def _clone_private_layout(value: Container | AoT) -> Table | AoT:
    """Copy structural layout into a private holder without taking its source."""
    holder = Document()
    holder._is_private = True  # noqa: SLF001
    holder._newline = value._doc_newline  # noqa: SLF001
    with _sources_kept_intact(holder, (value,)):
        holder._setitem_validated("", value)  # noqa: SLF001
    result = dict.__getitem__(holder, "")
    assert isinstance(result, (Table, AoT))
    return result


def _copy_input(value: TomlInput, memo: dict[int, object] | None = None) -> TomlInput:
    """Deep-copy validated input without creating layout where none exists."""
    if is_shareable_scalar(value):
        return value
    if memo is None:
        memo = {}
    if is_scalar(value):
        return copy.deepcopy(value, memo)
    if is_inline_value(value):
        cst = _detached_inline_value(value)
        if cst is not None:
            from tomlrt._build import _decode_value  # noqa: PLC0415

            cloned = _decode_value(cst, None, None, None, None)
            assert isinstance(cloned, (Table, Array))
            return cloned
    if (
        isinstance(value, (Container, AoT))
        and value._layout_root is not None  # noqa: SLF001
        and not value._inline  # noqa: SLF001
    ):
        return _clone_private_layout(value)
    if isinstance(value, Container):
        table = Table.inline() if value._inline else Table.section()  # noqa: SLF001
        for key, child in _mapping_items(value):
            dict.__setitem__(table, key, _copy_input(child, memo))
        return table
    if isinstance(value, AoT):
        aot = AoT()
        for original in value:
            entry = _copy_input(original, memo)
            assert isinstance(entry, Table)
            list.append(aot, entry)
        return aot
    if isinstance(value, Mapping):
        return {key: _copy_input(child, memo) for key, child in _mapping_items(value)}
    assert isinstance(value, list), "validated compound input expected"
    return [_copy_input(child, memo) for child in value]


def _clear_inline_document_binding(t: Container) -> None:
    """Forget document metadata while preserving inline CST ownership."""
    t._layout_root = None  # noqa: SLF001
    t._owner_aot_entry = None  # noqa: SLF001


def _to_python(v: object) -> object:
    """Export independent plain data from views and unmaterialized payloads."""
    return _to_python_value(v, {})


def _to_python_value(v: object, memo: dict[int, object]) -> object:
    """Walk an export unit without recursing or copying for atomic values."""
    if isinstance(v, (dict, Mapping)):
        return {
            key: value if is_shareable_scalar(value) else _to_python_value(value, memo)
            for key, value in _mapping_items(v)
        }
    if isinstance(v, list):
        return [x if is_shareable_scalar(x) else _to_python_value(x, memo) for x in v]
    return copy.deepcopy(v, memo) if is_scalar(v) else v


def _is_section(v: object) -> TypeGuard[Container]:
    """True iff ``v`` is a non-inline (section-style) Container."""
    return isinstance(v, Container) and not v._inline  # noqa: SLF001


def _is_inline_table(v: object) -> TypeGuard[Container]:
    """True iff ``v`` is an inline Container."""
    return isinstance(v, Container) and v._inline  # noqa: SLF001


def _snapshot_for_overlapping_install(
    parent: Container, key: str, value: TomlInput
) -> TomlInput:
    """Snapshot ``value`` if an overlapping install cannot safely read it.

    A source the write site lives in would grow while it is read.
    During overwrite, a headerless grandchild also depends on implicit
    intermediates that deletion resets. Other descendants retain
    independent slot anchors and use the trivia-preserving
    private-orphan adopt path.

    The snapshot is the same view in a byte-exact copy of the document,
    so it is read exactly as the source would have been.
    """
    if not isinstance(value, (Container, AoT)):
        return value
    root = parent._layout_root  # noqa: SLF001
    if root is None or value._layout_root is not root:  # noqa: SLF001
        return value
    dest_path = (*parent._path, key)  # noqa: SLF001
    value_path = value._path  # noqa: SLF001
    headerless_value = not isinstance(value, AoT) and value._header_ref is None  # noqa: SLF001
    descendant_overlap = (
        headerless_value
        and len(value_path) - len(dest_path) >= 2
        and value_path[: len(dest_path)] == dest_path
    )
    if not (_layout_ops.hosts_site(value, parent) or descendant_overlap):
        return value
    return _layout_ops.stable_snapshot(value)


def _collect_private_roots(value: object, found: dict[int, Document]) -> None:
    """Record the private documents any view within ``value`` belongs to.

    A source can be reached through plain mappings and lists that are
    only wrappers, so the whole shape is walked rather than its top
    level: those wrappers are rebuilt on the way in, but the views
    inside them are installed as they are.

    Not only the popped subtree, which the top level finds anyway:
    popping re-roots its inline descendants onto the orphan too, so an
    `Array` or inline `Table` taken out of one is privately rooted and
    can sit inside an ordinary list or dict.
    """
    if isinstance(value, _View):
        root = value._layout_root  # noqa: SLF001
        if root is not None and root._is_private:  # noqa: SLF001
            found[id(root)] = root
    if isinstance(value, Mapping):
        for _, sub in _mapping_items(value):
            _collect_private_roots(sub, found)
    elif isinstance(value, list):
        for sub in value:
            _collect_private_roots(sub, found)


def _can_adopt_from(source: Document, destination: Document) -> bool:
    """Whether this destination may consume the source root's layout."""
    protected = destination._protected_source_roots  # noqa: SLF001
    return source._is_private and (protected is None or id(source) not in protected)  # noqa: SLF001


@contextlib.contextmanager
def _sources_kept_intact(
    destination: Document | None, values: Iterable[object]
) -> Iterator[None]:
    """Protect initial source roots during writes to this destination.

    Newly orphaned roots remain adoptable. Strong references keep the
    captured identity keys valid until the scope exits.
    """
    roots: dict[int, Document] = {}
    for value in values:
        _collect_private_roots(value, roots)
    if destination is None:
        yield
        return
    previous = destination._protected_source_roots  # noqa: SLF001
    if previous is not None:
        roots.update(previous)
    destination._protected_source_roots = roots or None  # noqa: SLF001
    try:
        yield
    finally:
        destination._protected_source_roots = previous  # noqa: SLF001


def _has_extractable_layout(data: Mapping[str, object]) -> TypeGuard[Table]:
    """True when ``data`` is a `Table` that owns section layout.

    A section-backed table — implicit ones and ``[[aot]]`` entries
    included — owns slots that can be cloned and re-rooted at a document
    of their own. An inline table has no section layout, and a detached
    factory table has none yet.
    """
    return (
        isinstance(data, Table)
        and data._layout_root is not None  # noqa: SLF001
        and data._kind in {_Kind.SECTION, _Kind.IMPLICIT_SECTION}  # noqa: SLF001
    )


# `_array` depends on `Container` for `Table`, so the import is at the
# bottom to avoid a circular import. The `Array` / `AoT` symbols are
# re-exported for convenience.
from tomlrt._array import AoT, Array  # noqa: E402

TomlInput: TypeAlias = (
    str
    | int
    | float
    | bool
    | datetime
    | date
    | time
    | Array
    | AoT
    | Table
    | Mapping[str, Any]
    | list[Any]
)
"""Values accepted by mutators and factories.

Includes [`Table`][tomlrt.Table], [`Array`][tomlrt.Array],
[`AoT`][tomlrt.AoT], any TOML scalar (`str`, `int`, `float`, `bool`,
`datetime`, `date`, `time`), and plain `Mapping[str, Any]` /
`list[Any]`.

The nested `list` / `Mapping` arms intentionally use `Any` for
elements: tightening to a recursive alias would trip over Python's
invariant container generics (a `list[int]` is not assignable to
`list[TomlInput]`). Invalid elements are rejected at runtime when
the value is assigned.
"""


# ---------------------------------------------------------------------------
# Plain-Python value synthesis.
# ---------------------------------------------------------------------------


def _is_synth_inline(v: object) -> bool:
    """True iff ``v`` is a value we can synthesise to an inline TOML value.

    Accepts any ``Mapping``, inline ``Container``, ``list``, or
    ``Array`` (deep-copy semantics); rejects everything else (tuple,
    bytes, sets, AoT, section Container, …) so the caller can route to
    a stronger error.
    """
    if isinstance(v, AoT):
        return False
    if isinstance(v, Container):
        # Section containers need real live-attach; only inline ones
        # round-trip through value-synthesis safely.
        return v._inline  # noqa: SLF001
    if isinstance(v, Array):
        return True
    if isinstance(v, Mapping):
        return True
    # `list` (or subclass) only — `tuple` is intentionally not accepted
    # (TOML has no tuple, and accepting it would mask user typos). Array,
    # a list subclass, was already accepted above.
    return isinstance(v, list)


def _validate_input(
    v: object,
    *,
    inline_only: bool,
    key: str | None = None,
    inline_kind: Literal["array", "table"] = "table",
) -> None:
    """Validate a value recursively for an inline or section context.

    ``key`` names the entry ``v`` is bound to, if it has one. The
    recursion passes each child's own key, so a rejection deep inside a
    nested mapping still reports where it came from. Types TOML cannot
    represent are named in the terminal branch rather than tested for up
    front, so the accepting paths never pay for them.

    The scalar arms are spelled out rather than delegated to
    `is_scalar` + `validate_scalar`: splitting the ladder where
    behaviour actually differs classifies and checks in one pass.
    """
    if isinstance(v, PLAIN_SCALARS):
        return
    if isinstance(v, CHECKED_SCALARS):
        validate_scalar(v)
        return
    if isinstance(v, AoT):
        if inline_only:
            msg = f"cannot store an array-of-tables inside an inline {inline_kind}"
            raise TOMLError(msg)
        for entry in v:
            mapping = _require_mapping(entry, label="AoT entry")
            _validate_mapping_items(mapping, inline_only=False)
        return
    # Before the mapping arms: a list is never a mapping, and asking a
    # concrete type is far cheaper than asking the `Mapping` ABC.
    if isinstance(v, list):
        for child in v:
            _validate_input(child, inline_only=True, inline_kind="array")
        return
    if _is_section(v):
        if inline_only:
            msg = "cannot store a section-style table inside an inline-style table"
            raise TOMLError(msg)
        _validate_mapping_items(v, inline_only=False)
        return
    if isinstance(v, Mapping):
        _validate_mapping_items(v, inline_only=True)
        return
    raise TypeError(_unrepresentable_message(v, key))


def _unrepresentable_message(v: object, key: str | None) -> str:
    """Explain why ``v`` cannot be stored, naming ``key`` when known."""
    at = f" to TOML key {key!r}" if key is not None else ""
    if isinstance(v, tuple):
        return f"cannot assign tuple{at}; use a list"
    if isinstance(v, (bytes, bytearray)):
        return f"cannot assign bytes{at}; use a string"
    return f"cannot convert {type(v).__name__} to a TOML value"


def _validate_mapping_items(
    mapping: Mapping[Any, object], *, inline_only: bool
) -> None:
    """Validate each mapping key and its value in one pass."""
    for raw_key, value in _mapping_items(mapping):
        key = _validate_key(raw_key)
        _validate_input(value, inline_only=inline_only, key=key)


def _file_host(
    view: Array | Container, parent: Container | None, array_host: Array | None
) -> None:
    """Stamp where ``view`` is bound.

    ``array_host`` is the array ``view`` is an element of, or ``None``
    when ``view`` is key-hosted under ``parent``. Called from the single
    tail of the ``_synth_value`` / ``_decode_value`` funnels, so no
    construction site has to remember the choice. Synthesis reaches
    this point with either a free view or a clone, so an existing
    materialised binding is never overwritten.
    """
    view._host = array_host if array_host is not None else parent  # noqa: SLF001


def _is_adoptable_inline(view: Array | Container) -> bool:
    """Whether ``view`` has no current document or materialised owner."""
    return view._layout_root is None and view._host is None  # noqa: SLF001


def _host_kv_slot(view: Array | Container) -> KVSlot | None:
    """The KV slot whose value subtree contains ``view``, or ``None``.

    Climbs ``_host`` -- the one field naming whichever view holds this
    one -- to the outermost value view held by a section/document
    container, then reads that container's index in O(depth). ``None``
    only when ``view`` is detached.
    """
    if view._layout_root is None:  # noqa: SLF001
        return None
    cur: Array | Container = view
    up = cur._host  # noqa: SLF001
    while is_inline_value(up):
        cur = up
        up = cur._host  # noqa: SLF001
    assert isinstance(up, Container), "internal: attached value has no host container"
    leaf = cur._name if isinstance(cur, Array) else cur._path[-1]  # noqa: SLF001
    kv = _direct_kv_slot(up, leaf)
    assert kv is not None, "internal: host key is absent from its container index"
    return kv


def _synth_value(
    v: TomlInput,
    *,
    layout_root: Document | None,
    parent: Container | None,
    name: str | None,
    owner: AoTEntry | None,
    array_host: Array | None = None,
) -> tuple[Value, object]:
    """Synthesise a validated (CST value, decoded view) pair from ``v``.

    Plain ``dict`` / ``Mapping`` → ``InlineTableValue`` + inline ``Table``.
    ``list`` / ``Array`` view → ``ArrayValue`` + ``Array``.

    ``parent``/``name`` are the container and key ``v`` is bound under,
    driving a key-hosted view's binding and name. ``array_host`` is the
    array ``v`` is an element of, if any; the resulting view's binding is
    filed at the single funnel tail via `_file_host`, so no site
    has to remember.
    """
    if is_scalar(v):
        return coerce_scalar(v), v
    # Adopt only a free inline value. A view already owned by any
    # materialised container is cloned, even when neither belongs to a
    # document: one view cannot represent two CST occurrences.
    cst: Value
    view: Array | Container
    if is_inline_value(v) and _is_adoptable_inline(v):
        if isinstance(v, Container) and v._value is None:  # noqa: SLF001
            cst, view = _populate_inline_table(
                v,
                list(_mapping_items(v)),
                layout_root=layout_root,
                parent=parent,
                name=name,
                owner=owner,
            )
        else:
            owned = v._value  # noqa: SLF001
            assert owned is not None
            cst = owned
            _retarget_to_doc(cst, layout_root)
            if isinstance(v, Array):
                v._name = name or ""  # noqa: SLF001
            elif parent is None:
                v._path = ()  # noqa: SLF001
            else:
                assert name is not None, "name is required with a parent"
                v._path = (*parent._path, name)  # noqa: SLF001
            _attach_inline_view(v, layout_root, owner)
            view = v
    # Cross-document / same-doc live inline values clone CST so source
    # formatting survives; plain Mapping / list inputs have none.
    elif is_inline_value(v) and (own := _detached_inline_value(v)) is not None:
        from tomlrt._build import _decode_value  # noqa: PLC0415

        _retarget_to_doc(own, layout_root)
        cst = own
        decoded = _decode_value(own, layout_root, parent, name, owner)
        assert isinstance(decoded, (Array, Container)), "inline CST decodes to a view"
        view = decoded
    elif isinstance(v, Mapping):
        cst, view = _populate_inline_table(
            Table(),
            list(_mapping_items(v)),
            layout_root=layout_root,
            parent=parent,
            name=name,
            owner=owner,
        )
    else:
        assert isinstance(v, list), "validated inline value expected"
        val = ArrayValue()
        arr = Array._view(val, layout_root, name)  # noqa: SLF001
        _fill_inline_array(
            arr,
            v,
            layout_root=layout_root,
            owner=owner,
        )
        cst, view = val, arr
    _file_host(view, parent, array_host)
    return cst, view


def _retarget_to_doc(val: Value, layout_root: Document | None) -> None:
    r"""Retarget ``val``'s baked-in newlines to ``layout_root``'s line ending.

    Called whenever pre-existing inline CST is dragged into a
    destination doc. Without this, cross-document grafts and unattached
    ``Array(multiline=True)`` factories can dump mixed ``\n`` /
    ``\r\n`` newlines.
    """
    if layout_root is not None:
        retarget_value_newlines(val, layout_root._newline)  # noqa: SLF001


def _attach_inline_view(
    value: Array | Container,
    layout_root: Document | None,
    owner: AoTEntry | None,
) -> None:
    """Record document attachment throughout a materialised inline tree."""
    if isinstance(value, Array):
        value._layout_root = layout_root  # noqa: SLF001
        for child in value:
            if is_inline_value(child):
                _file_inline_child(child, value, None)
                _attach_inline_view(child, layout_root, owner)
    else:
        value._layout_root = layout_root  # noqa: SLF001
        value._owner_aot_entry = owner  # noqa: SLF001
        for key, child in value.items():
            if is_inline_value(child):
                _file_inline_child(child, value, key)
                _attach_inline_view(child, layout_root, owner)


def _file_inline_child(
    child: Array | Container,
    host: Array | Container,
    name: str | None,
) -> None:
    """Rebuild one child binding inside a preserved inline CST tree."""
    if isinstance(child, Array):
        child._host = host  # noqa: SLF001
        child._name = name or ""  # noqa: SLF001
    else:
        child._host = host  # noqa: SLF001
        if isinstance(host, Array):
            child._path = ()  # noqa: SLF001
        else:
            assert name is not None, "table-hosted child requires a key"
            child._path = (*host._path, name)  # noqa: SLF001


def _populate_inline_table(
    table: Container,
    items: Sequence[tuple[str, TomlInput]],
    *,
    layout_root: Document | None,
    parent: Container | None,
    name: str | None,
    owner: AoTEntry | None,
) -> tuple[InlineTableValue, Container]:
    """Wire ``table`` as an inline view and populate its entries.

    The live-attach path passes a user-supplied ``Table.inline()`` so
    identity is preserved; the plain-Mapping synth path passes a fresh
    ``Table()``. Entries use canonical single-line spacing.
    """
    if parent is None:
        path: tuple[str, ...] = ()
    else:
        assert name is not None, "name is required whenever parent is given"
        path = (*parent._path, name)  # noqa: SLF001
    val = InlineTableValue()
    table._wire(  # noqa: SLF001
        layout_root=layout_root, parent=parent, path=path, owner=owner
    )
    table._inline = True  # noqa: SLF001
    table._value = val  # noqa: SLF001

    last = len(items) - 1
    for i, (k, sub) in enumerate(items):
        sub_cst, sub_dec = _synth_value(
            sub,
            layout_root=layout_root,
            parent=table,
            name=k,
            owner=owner,
        )
        entry = InlineTableEntry(
            "" if i == 0 else " ",
            sub_cst,
            "",
            i != last,
            "",
            (make_keypart(k),),
            (),
            " ",
            " ",
            (k,),
        )
        val.items.append(entry)
        dict.__setitem__(table, k, sub_dec)
    if items:
        val.header_trivia = val._single_line_pad  # noqa: SLF001
        val.final_trivia = val._single_line_pad  # noqa: SLF001
    return val, table


def _fill_inline_array(
    arr: Array,
    items: Sequence[TomlInput],
    *,
    layout_root: Document | None,
    owner: AoTEntry | None,
) -> None:
    """Append ``items`` to ``arr`` and to the `ArrayValue` behind it.

    ``arr`` is already a view over the value to fill; the items are laid
    out with canonical single-line spacing.
    """
    val = arr._value  # noqa: SLF001
    last = len(items) - 1
    for i, sub in enumerate(items):
        sub_cst, sub_dec = _synth_value(
            sub,
            layout_root=layout_root,
            parent=None,
            name=None,
            owner=owner,
            array_host=arr,
        )
        # Under the canonical model, inter-item separators live in the
        # NEXT item's leading; items[0].leading is always empty;
        # post_comma_trivia carries only EOL sections (empty here).
        item = ArrayItem(
            "" if i == 0 else " ",
            sub_cst,
            "",
            i != last,
            "",
        )
        val.items.append(item)
        list.append(arr, sub_dec)


__all__ = ["AoT", "Array", "Container", "Document", "Table", "TomlInput"]
