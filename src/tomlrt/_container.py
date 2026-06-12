"""Logical container layer.

`Container(dict)` backs `Document` and `Table`. Dict storage follows
doc-stream first-occurrence order; mutations update the slot stream
through `_index`, `_refs`, `_header_ref`, and `_body_tail`.
"""

from __future__ import annotations

import copy
import sys
from collections.abc import Mapping
from datetime import date, datetime, time
from typing import TYPE_CHECKING, Any, TypeAlias, TypeGuard, TypeVar, overload

if sys.version_info >= (3, 12):
    from typing import Self, override
else:  # pragma: no cover -- backport for Python < 3.12
    from typing_extensions import override

from tomlrt import _inline_ops, _layout_ops
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
    _canon_header_slot,
    _canon_inline_value,
    _canon_kv_slot,
    _canon_leading,
    format_document_trailing,
    format_subtree,
)
from tomlrt._kind import _Kind
from tomlrt._paths import split_path, validate_path
from tomlrt._render import render
from tomlrt._scalar import (
    coerce_scalar,
    is_scalar,
)
from tomlrt._slots import KVSlot, StructuralHeaderSlot
from tomlrt._trivia import (
    NewlineNode,
    Trivia,
    WhitespaceNode,
    retarget_trivia_newlines,
    trivia_has_comment,
)
from tomlrt._typecheck import _validate_key, _validate_mapping
from tomlrt._values import (
    ArrayItem,
    ArrayValue,
    InlineTableEntry,
    InlineTableValue,
    make_keypart,
    retarget_value_newlines,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from _typeshed import SupportsKeysAndGetItem, SupportsRichComparison
    from typing_extensions import Self

    from tomlrt._slots import AoTEntry, Slot, SlotRef
    from tomlrt._trivia import TriviaPiece
    from tomlrt._values import (
        CommaItem,
        CommaValue,
        Value,
    )


_T = TypeVar("_T")
_ItemT = TypeVar("_ItemT", bound="CommaItem")

_MISSING: Any = object()


class Container(dict[str, Any]):
    """Dict-typed base for `Document` and `Table` views.

    Reads are pure dict operations. Section mutations use the
    per-container cache (`_index`, `_refs`, `_header_ref`,
    `_body_tail`). Inline tables keep those caches empty and mutate
    the backing `InlineTableValue` in `_value`.
    """

    __slots__ = (
        "_body_tail",
        "_header_ref",
        "_index",
        "_inline",
        "_layout_root",
        "_owner_aot_entry",
        "_parent",
        "_path",
        "_refs",
        "_value",
    )

    def __init__(self) -> None:
        super().__init__()
        self._layout_root: Document | None = None
        self._path: tuple[str, ...] = ()
        self._inline: bool = False
        self._parent: Container | None = None
        self._owner_aot_entry: AoTEntry | None = None
        self._index: dict[str, list[SlotRef]] = {}
        self._refs: list[SlotRef] = []
        self._header_ref: SlotRef | None = None
        self._body_tail: Slot | None = None
        self._value: InlineTableValue | None = None

    @property
    def _kind(self) -> _Kind:
        """The shape this container is in. See :class:`_Kind`."""
        if self._inline:
            if self._value is not None:
                return _Kind.INLINE_ROOT
            if self._layout_root is None:
                return _Kind.INLINE_FACTORY
            return _Kind.INLINE_DOTTED_INNER
        if self._header_ref is not None:
            return _Kind.SECTION
        return _Kind.IMPLICIT_SECTION

    @property
    def comments(self) -> EolCommentView:
        """Mapping view of EOL comments on this container's direct keys.

        Mutating the view requires the container to be attached to a
        `Document`; mutate via the attached view, not on a detached
        ``Table.section()`` / ``Table.inline()``.
        """
        if self._inline:
            msg = "comment API is not available on inline tables"
            raise TOMLError(msg)
        return EolCommentView(self)

    @property
    def leading_comments(self) -> LeadingCommentView:
        """Mapping view of leading-comment blocks on this container's direct keys.

        Returns only the *attached* comment run immediately above each key
        (no blank line between). For the full block, including any
        above-blank groups and the blank-line structure between them, see
        [`leading_block`][tomlrt.Table.leading_block].

        Mutating the view requires the container to be attached to a
        `Document`.
        """
        if self._inline:
            msg = "comment API is not available on inline tables"
            raise TOMLError(msg)
        return LeadingCommentView(self)

    @property
    def leading_block(self) -> LeadingBlockView:
        """Mapping view of full leading-trivia blocks on direct keys.

        Each entry is a ``tuple[str | None, ...]`` of comment strings
        interleaved with ``None`` (one per blank line), in source order;
        the slot's own column indent is implicit and re-applied on write.

        At the document's head slot,
        [`Document.preamble`][tomlrt.Document.preamble] is disjoint from
        this view: reading omits the preamble lines, and writing
        preserves them.

        Mutating the view requires the container to be attached to a
        `Document`.
        """
        if self._inline:
            msg = "comment API is not available on inline tables"
            raise TOMLError(msg)
        return LeadingBlockView(self)

    @property
    def header_comment(self) -> str | None:
        """The EOL comment on this container's section header, or None.

        Containers without a header line (the document root, or implicit
        sections opened only by a nested ``[a.b]`` header) read as
        ``None``. Setting on such a container raises
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

        Containers without a header line (the document root, or implicit
        sections opened only by a nested ``[a.b]`` header) read as ``()``.
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
        ``None`` (one per blank line), in source order. Containers without
        a header line (the document root, or implicit sections opened only
        by a nested ``[a.b]`` header) read as ``()``. Setting on such a
        container raises [`TOMLError`][tomlrt.TOMLError]; inline tables
        also raise.

        For the first section in a document,
        [`Document.preamble`][tomlrt.Document.preamble] is disjoint from
        this view: reading omits the preamble lines, and writing preserves
        them.
        """
        return _header_leading_block_get(self)

    @header_leading_block.setter
    def header_leading_block(self, value: tuple[str | None, ...]) -> None:
        _header_leading_block_set(self, value)

    @header_leading_block.deleter
    def header_leading_block(self) -> None:
        _header_leading_block_set(self, ())

    @property
    def _doc_newline(self) -> str:
        r"""The active newline of the owning document, or ``"\n"`` if detached."""
        lr = self._layout_root
        return lr._newline if lr is not None else "\n"  # noqa: SLF001

    def format(self, *, comments: bool = True) -> None:
        """Canonicalise this container's formatting in place.

        Rewrites this subtree to the canonical layout:

        * Keys, ``=`` spacing, and header brackets use canonical
          whitespace.
        * Sibling key/value slots have no blank line between them;
          section / array-of-tables headers get one.
        * Orphan comment blocks above slots are preserved verbatim.
        * Inline values keep their shape (single-line stays single-line,
          multi-line stays multi-line).
        * Newlines use the owning document's style.

        With ``comments=True``, ``#foo`` and ``#   foo`` both become
        ``# foo`` and trailing comment whitespace is stripped.

        Detached factory-style containers (``Table.section()`` /
        ``Table.inline()`` not yet assigned anywhere) and inline dotted
        navigators are unsupported and raise `TOMLError`.
        """
        kind = self._kind
        nl = self._doc_newline

        if kind is _Kind.INLINE_ROOT:
            assert self._value is not None
            _canon_inline_value(self._value, nl=nl, comments=comments)
            return

        if kind in (_Kind.INLINE_FACTORY, _Kind.INLINE_DOTTED_INNER):
            msg = "format() is not supported on detached inline-table views"
            raise TOMLError(msg)

        if kind is _Kind.DOCUMENT:
            assert isinstance(self, Document)
            format_subtree(
                start=self._head,
                path=(),
                owner=None,
                nl=nl,
                comments=comments,
            )
            format_document_trailing(self._preamble, nl=nl, comments=comments)
            format_document_trailing(self._trailing, nl=nl, comments=comments)
            return

        if kind is _Kind.SECTION:
            assert self._header_ref is not None
            format_subtree(
                start=self._header_ref.slot,
                path=self._path,
                owner=self._owner_aot_entry,
                nl=nl,
                comments=comments,
            )
            return

        # IMPLICIT_SECTION slots are not contiguous, so canonicalise each
        # owned slot and recurse through dict storage.
        if not self._attached:
            msg = "format() requires the container to be attached to a Document"
            raise TOMLError(msg)
        for ref in list(self._refs):
            slot = ref.slot
            if isinstance(slot, KVSlot):
                _canon_kv_slot(slot, nl=nl, comments=comments)
            elif isinstance(slot, StructuralHeaderSlot):
                _canon_header_slot(slot, nl=nl, comments=comments)
            _canon_leading(slot, nl=nl, target_blanks=None, comments=comments)
        for value in self.values():
            if isinstance(value, (Container, Array)):
                value.format(comments=comments)
            elif isinstance(value, AoT):
                for entry in value:
                    entry.format(comments=comments)

    @property
    def _attached(self) -> bool:
        """True iff this container is attached to a live document root.

        Attached means the layout root is a user-visible document: not
        ``None`` (factory mode) and not a private orphan root. Mirrors
        :attr:`Array._attached` for cross-document live-attach dispatch.
        """
        lr = self._layout_root
        return lr is not None and not lr._is_private  # noqa: SLF001

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
        parent: Container | None,
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
        self._parent = parent
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

        Raises ``TypeError`` if descent passes through a non-table.
        """
        parts = split_path(key)
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
        """Materialise a plain-Python ``dict`` (recursive)."""
        out: dict[str, Any] = {}
        for k, v in self.items():
            out[k] = _to_python(v)
        return out

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    @override
    def __setitem__(self, key: str, value: Any) -> None:
        # Reject non-str keys before they reach the layout pipeline and
        # fail later with an opaque TypeError.
        _validate_key(key)
        if key in self and self[key] is value:
            return
        # Reject types we explicitly do not coerce.
        if isinstance(value, tuple):
            msg = f"cannot assign tuple to TOML key {key!r}; use a list"
            raise TypeError(msg)
        if isinstance(value, (bytes, bytearray)):
            msg = f"cannot assign bytes to TOML key {key!r}; use a string"
            raise TypeError(msg)
        # Inline tables cannot host section-shaped values; fail at the
        # assignment site, for both attached and detached factories.
        if self._inline:
            if isinstance(value, AoT):
                msg = "Cannot store an array-of-tables inside an inline table"
                raise TOMLError(msg)
            if _is_section(value):
                msg = "Cannot store a section-style table inside an inline-style table"
                raise TOMLError(msg)
        # Unattached factory mode: dict-only storage until attach.
        if self._layout_root is None:
            dict.__setitem__(self, key, value)
            return
        if self._inline:
            self._inline_setitem(key, value)
            return
        if key in self:
            self._overwrite_existing(key, value)
            return
        self._insert_new(key, value)

    def _overwrite_existing(self, key: str, value: Any) -> None:
        """Replace the value at an already-bound key.

        Uses in-place swaps when the existing binding is a single KV.
        Structural values take the delete + reinsert + move-to-anchor
        path so the new view's Python identity becomes the live one.
        """
        current = dict.__getitem__(self, key)
        # Fast-path: pure scalar → scalar (cheap, no synth alloc).
        if is_scalar(current) and is_scalar(value):
            self._scalar_replace(key, value)
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
        # old user references from the live doc.
        if (
            is_scalar(value)
            or _is_synth_inline(value)
            or isinstance(value, AoT)
            or _is_section(value)
            or isinstance(value, Mapping)
        ):
            self._structural_overwrite(key, value)
            return
        # Unsupported value type — TypeError, not NIE.
        msg = (
            f"Cannot convert value of type {type(value).__name__!r} "
            f"for TOML key {key!r}"
        )
        raise TypeError(msg)

    def _structural_overwrite(self, key: str, value: Any) -> None:
        """Replace ``key`` by deleting then reinstalling at the saved anchor."""
        _layout_ops.reposition_install(self, key, value)

    def _insert_new(self, key: str, value: Any) -> None:
        """Bind ``key`` for the first time at the document tail."""
        if is_scalar(value):
            _layout_ops.append_direct_kv(self, key, coerce_scalar(value))
            dict.__setitem__(self, key, value)
            return
        if _is_synth_inline(value):
            cst, decoded = _synth_value(
                value,
                layout_root=self._layout_root,
                parent=self,
                path=(*self._path, key),
                owner=self._owner_aot_entry,
            )
            _layout_ops.append_direct_kv(self, key, cst)
            dict.__setitem__(self, key, decoded)
            return
        if isinstance(value, AoT):
            self._attach_aot(key, value)
            return
        if _is_section(value):
            self._attach_section(key, value)
            return
        # Unsupported types get the canonical TypeError.
        msg = f"Cannot convert {type(value).__name__} to a TOML value"
        raise TypeError(msg)

    def _attach_aot(self, key: str, value: AoT) -> None:
        """Install ``value`` (an AoT) under ``key``.

        Live-attached sources or private orphans with intact entry
        slots route through :func:`clone_aot` to preserve per-entry
        trivia and nested sub-sections. Detached AoTs without preserved
        slots are rehomed entry-by-entry.
        """
        src_root = value._layout_root  # noqa: SLF001
        if src_root is not None and not src_root._is_private:  # noqa: SLF001
            if key in self:
                _layout_ops.delete_key(self, key)
            _layout_ops.clone_aot(self, key, value)
            return
        # Private orphans may still own intact entry_slots from a
        # structural-overwrite detach; clone those to keep per-KV
        # trivia, nested sub-sections, and inter-entry separators.
        # The generic add_aot_entry(rehome=) path rebuilds from dict
        # storage and drops that CST.
        existing_entries: list[Table] = list(value)
        preserved_entries: list[AoTEntry | None] = [
            et._owner_aot_entry  # noqa: SLF001
            if et._owner_aot_entry is not None  # noqa: SLF001
            and et._owner_aot_entry.entry_slots  # noqa: SLF001
            else None
            for et in existing_entries
        ]
        can_clone = bool(preserved_entries) and all(
            e is not None for e in preserved_entries
        )
        for et in existing_entries:
            _reset_table_for_rehome(et)
        list.clear(value)
        value._layout_root = None  # noqa: SLF001
        value._parent = None  # noqa: SLF001
        value._path = ()  # noqa: SLF001
        attached = _layout_ops.attach_empty_aot(self, key, value)
        dict.__setitem__(self, key, attached)
        if can_clone:
            # Clone CST from intact orphan slots. Entry-table identities
            # are replaced; the AoT object's identity is preserved.
            for src_entry in preserved_entries:
                assert src_entry is not None
                _layout_ops.clone_aot_entry(
                    value, src_entry, preserve_source_separator=True
                )
        else:
            for entry_table in existing_entries:
                _layout_ops.add_aot_entry(value, None, rehome=entry_table)

    def _attach_section(self, key: str, value: Container) -> None:
        """Install ``value`` (a section-flavoured Table) under ``key``.

        AoT-entry sources clone as a table so trivia survives and the
        head normalises from ``[[..]]`` to ``[..]``. Attached
        header-bearing sections clone slots; a whole attached
        ``Document`` clones its body under a synthesised header (so body
        comments and inline pad survive); attached implicit sources
        recurse via ``_install_attached_subtree``; detached/private
        sources rehome in place.
        """
        src_root = value._layout_root
        live_source = src_root is not None and not src_root._is_private  # noqa: SLF001
        if live_source:
            if key in self:
                _layout_ops.delete_key(self, key)
            if value._owner_aot_entry is not None and self._layout_root is not None:
                _layout_ops.clone_aot_entry_as_table(self, key, value)
            elif value._header_ref is not None:
                _layout_ops.clone_section_as_section(self, key, value)
            elif isinstance(value, Document):
                _layout_ops.clone_document_as_section(self, key, value)
            else:
                # Implicit source: tuple-path install preserves structural
                # children and implicit chains.
                _install_attached_subtree(self, (key,), value)
            return
        if src_root is not None and src_root._is_private:  # noqa: SLF001
            _reset_table_for_rehome(value)
        _layout_ops.attach_section_at(self, (key,), value)

    def _scalar_replace(self, key: str, value: Any) -> None:
        refs = self._index.get(key)
        if not refs:  # pragma: no cover  -- view/CST drift invariant guard
            msg = f"internal: key {key!r} present in dict but missing from _index"
            raise AssertionError(msg)
        primary = refs[0]
        slot = primary.slot
        if not isinstance(slot, KVSlot):  # pragma: no cover  -- type invariant guard
            msg = "internal: scalar replace expects KVSlot"
            raise AssertionError(msg)  # noqa: TRY004
        slot.value = coerce_scalar(value)
        dict.__setitem__(self, key, value)

    def _inline_typed_replace(self, key: str, value: Any) -> None:
        """Swap an existing direct-KV slot's value to a synthesised inline value.

        Works for any existing scalar / inline-table / inline-array
        binding bound by a single direct-KV slot. Dotted KV slots are
        also fine: the new value is just an inline value at the same
        leaf position.

        If the displaced value is itself a typed view (inline Table,
        Array), its attachment state is cleared so a subsequent
        assignment of that view elsewhere re-attaches live with
        identity preserved (rather than going through the
        cross-doc clone path).
        """
        refs = self._index.get(key)
        if not refs or len(refs) != 1:
            msg = "structural overwrite (multiple contributing refs) is not supported"
            raise NotImplementedError(msg)
        primary = refs[0]
        slot = primary.slot
        if not isinstance(slot, KVSlot):
            msg = "structural overwrite of header-bound binding is not supported"
            raise NotImplementedError(msg)
        old = dict.__getitem__(self, key)
        cst, decoded = _synth_value(
            value,
            layout_root=self._layout_root,
            parent=self,
            path=(*self._path, key),
            owner=self._owner_aot_entry,
        )
        slot.value = cst
        dict.__setitem__(self, key, decoded)
        # Detach the displaced view so it can be reattached live.
        if _is_inline_table(old) and old is not decoded:
            _reset_inline_for_rehome(old)
        elif isinstance(old, Array) and old is not decoded:
            _reset_array_for_rehome(old)

    @override
    def __delitem__(self, key: str) -> None:
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
    def pop(self, key: str, default: Any = _MISSING) -> Any:
        if key in self:
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
        if args:
            other = args[0]
            if hasattr(other, "keys"):
                for k in other.keys():  # noqa: SIM118
                    self[k] = other[k]
            else:
                for k, v in other:
                    self[k] = v
        for k, v in kwargs.items():
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
            # their leaf part stays ahead of all sections.
            pure_leaves = [
                k for k in current if self._has_leaf(k) and not self.has_header(k)
            ]
            mixed = [k for k in current if self._has_leaf(k) and self.has_header(k)]
            pure_sections = [k for k in current if not self._has_leaf(k)]
            for group in (pure_leaves, mixed, pure_sections):
                group.sort(key=key, reverse=reverse)
            new_order = pure_leaves + mixed + pure_sections
        if new_order == current:
            return
        if self._inline:
            _inline_ops.reorder_inline(self, new_order)
        elif self._layout_root is not None:
            _layout_ops.reorder_container(self, new_order)
        _reorder_dict_storage(self, new_order)

    def has_header(self, key: str) -> bool:
        """Whether ``key`` is bound to a child with a ``[header]`` line.

        Returns ``True`` when this container's block for ``key`` contains
        a structural header slot — i.e. a ``[header]`` section or a
        ``[[header]]`` array-of-tables entry. Returns ``False`` for bare
        ``key = value`` leaves, inline tables, implicit sections built
        entirely from dotted keys (e.g. ``a.x = 1``, where ``a`` has no
        header line of its own), and missing keys.
        """
        refs = self._index.get(key, ())
        return any(isinstance(r.slot, StructuralHeaderSlot) for r in refs)

    def _has_leaf(self, key: str) -> bool:
        """Whether ``key``'s block contains a leaf ``key = value`` slot.

        A key can have both a leaf and a sub-section header, in which
        case both this and `has_header` return True.
        """
        refs = self._index.get(key, ())
        return any(isinstance(r.slot, KVSlot) for r in refs)

    @override
    def __ior__(  # type: ignore[override]
        self,
        other: SupportsKeysAndGetItem[str, Any] | Iterable[tuple[str, Any]],
        /,
    ) -> Self:
        self.update(other)
        return self

    def __copy__(self) -> Container:
        # Equivalent to deepcopy: returns an independent detached
        # container preserving nested typed views, so .table() etc.
        # continue to work on the copy.
        return _deep_clone(self)

    def __deepcopy__(self, memo: dict[int, object]) -> Container:
        return _deep_clone(self)

    # ------------------------------------------------------------------
    # Inline-table dispatch
    # ------------------------------------------------------------------

    def _inline_setitem(self, key: str, value: Any) -> None:
        # ``__setitem__`` has already rejected ``AoT`` / section values
        # for inline hosts. Non-coerceable types (``set``, custom
        # classes, …) reach ``_synth_value`` for the canonical TypeError.
        if is_scalar(value):
            cst: Value = coerce_scalar(value)
            decoded: object = value
        else:
            cst, decoded = _synth_value(
                value,
                layout_root=self._layout_root,
                parent=self,
                path=(*self._path, key),
                owner=self._owner_aot_entry,
            )
        if key in self and isinstance(dict.__getitem__(self, key), Container):
            # Stay on the CST side when replacing a dotted-prefix
            # navigator; dict-level delete would prune the parent chain.
            _inline_ops.overwrite_entry(self, key, cst)
        elif key in self:
            ok = _inline_ops.replace_entry_value(self, key, cst)
            if not ok:  # pragma: no cover  -- view/CST drift invariant guard
                msg = (
                    f"internal: key {key!r} present on inline view but no "
                    "matching entry in the backing InlineTableValue"
                )
                raise AssertionError(msg)
        else:
            _inline_ops.append_entry(self, key, cst)
        dict.__setitem__(self, key, decoded)

    def _inline_delitem(self, key: str) -> None:
        if key not in self:
            raise KeyError(key)
        ok = _inline_ops.delete_entry(self, key)
        if not ok:  # pragma: no cover  -- view/CST drift invariant guard
            msg = (
                f"internal: key {key!r} present on inline view but no "
                "matching entry in the backing InlineTableValue"
            )
            raise AssertionError(msg)
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
            if my_key in parent:
                dict.__delitem__(parent, my_key)
            cur = parent

    def install(self, path: str | Sequence[str], value: TomlInput) -> Any:
        """Set ``value`` at the (possibly dotted) ``path``.

        Intermediate sections are created as needed via `ensure_table`.
        Returns the live view stored at the leaf.
        """
        parts = validate_path(path)
        if self._inline and len(parts) > 1:
            msg = "cannot install dotted path into an inline-style table"
            raise TOMLError(msg)
        # Section / AoT values need the multi-component attach path so
        # intermediate components stay implicit.
        is_section = isinstance(value, Table) and not value._inline  # noqa: SLF001
        is_aot = isinstance(value, AoT)
        if (is_section or is_aot) and len(parts) > 1 and self._layout_root is not None:
            # Walk existing prefix; whatever's left is created with
            # implicit intermediates plus the final explicit binding.
            cur: Container = self
            i = 0
            while i < len(parts) - 1:
                p = parts[i]
                if p not in cur:
                    break
                nxt = dict.__getitem__(cur, p)
                if isinstance(nxt, AoT):
                    msg = (
                        f"cannot install through array-of-tables at {p!r}: "
                        "no addressable target inside an AoT"
                    )
                    raise TOMLError(msg)
                if not isinstance(nxt, Container) or nxt._inline:  # noqa: SLF001
                    break
                cur = nxt
                i += 1
            # Drop conflicting inline-tail keys before installing the
            # section, or a stale inline value would shadow it.
            if (
                i < len(parts) - 1
                and parts[i] in cur
                and isinstance(dict.__getitem__(cur, parts[i]), Container)
                and dict.__getitem__(cur, parts[i])._inline  # noqa: SLF001
            ):
                inline_holder: Container = dict.__getitem__(cur, parts[i])
                # Delete the next inline path component in order.
                tail = parts[i + 1 :]
                if tail and tail[0] in inline_holder:
                    del inline_holder[tail[0]]
                    if len(inline_holder) == 0:
                        _layout_ops.delete_key(cur, parts[i])
            # Leaf already present: use normal overwrite dispatch.
            if i == len(parts) - 1:
                cur[parts[-1]] = value
                return cur[parts[-1]]
            # Then use normal Container dispatch; it already handles
            # attached, detached, section, and AoT source states.
            anchor = _layout_ops.ensure_implicit_chain(cur, tuple(parts[i:-1]))
            anchor[parts[-1]] = value
            return anchor[parts[-1]]
        host = self if len(parts) == 1 else self.ensure_table(parts[:-1])
        host[parts[-1]] = value
        return host[parts[-1]]

    def ensure_table(self, key: str | Sequence[str]) -> Table:
        """Return the table at ``key``, creating it if missing.

        If any prefix already exists as a section, descent continues
        from there. Intermediate components missing entirely are left
        implicit; only the deepest component gets an explicit
        ``[a.b.c]`` header. An existing non-table at any component
        raises ``TypeError``.
        """
        parts = validate_path(key)
        cur: Container = self
        i = 0
        while i < len(parts):
            p = parts[i]
            if p not in cur:
                break
            nxt = dict.__getitem__(cur, p)
            if isinstance(nxt, AoT):
                msg = (
                    f"cannot ensure_table through array-of-tables at {p!r}: "
                    "no addressable target inside an AoT"
                )
                raise TOMLError(msg)
            if not isinstance(nxt, Container):
                msg = (
                    f"existing value at {p!r} is not section-backed "
                    "(is an inline table or non-table value)"
                )
                raise TOMLError(msg)
            if nxt._inline and i < len(parts) - 1:  # noqa: SLF001
                msg = (
                    f"existing value at {p!r} is not section-backed "
                    "(is an inline table or non-table value)"
                )
                raise TOMLError(msg)
            cur = nxt
            i += 1
        if i == len(parts):
            assert isinstance(cur, Table)
            return cur
        if cur._inline:  # noqa: SLF001
            msg = "cannot create section table inside an inline-style table"
            raise TOMLError(msg)
        if cur._layout_root is None:  # noqa: SLF001
            # Detached: build nested Table.section()s purely in dict
            # storage. No layout ops.
            for p in parts[i:]:
                child = Table.section()
                dict.__setitem__(cur, p, child)
                cur = child
            assert isinstance(cur, Table)
            return cur
        new_section = Table.section()
        attached = _layout_ops.attach_section_at(cur, parts[i:], new_section)
        assert isinstance(attached, Table)
        return attached

    def promote_inline(self, key: str) -> Table:
        """Convert an inline-table entry at ``key`` into a section header.

        Returns the live view at ``key`` after promotion. Raises
        ``TOMLError`` if the key is missing or doesn't refer to an
        inline-style table. If the value is already a section table,
        returns it unchanged.
        """
        if self._inline:
            msg = "inline-table promotion is not supported on inline tables"
            raise TOMLError(msg)
        if key not in self:
            msg = f"key {key!r} not in table"
            raise KeyError(msg)
        cur = dict.__getitem__(self, key)
        if not (_is_inline_table(cur)):
            msg = f"{key!r} is not an inline table"
            raise TOMLError(msg)
        if _inline_value_has_inner_comments(cur._value):  # noqa: SLF001
            msg = (
                f"cannot promote {key!r}: inline table has inner "
                f"comments that would be lost"
            )
            raise TOMLError(msg)
        # Transfer the existing KV slot's leading + eol to the header.
        old_slot = _direct_kv_slot(self, key)
        saved_leading = old_slot.leading if old_slot is not None else None
        saved_eol = old_slot.eol if old_slot is not None else None
        snapshot = cur.to_dict()
        _layout_ops.delete_key(self, key)
        self[key] = Table.section(snapshot)
        result = dict.__getitem__(self, key)
        assert isinstance(result, Table)
        new_header = result._header_ref.slot if result._header_ref else None  # noqa: SLF001
        if isinstance(new_header, StructuralHeaderSlot):
            if saved_leading is not None:
                new_header.leading = saved_leading
            if saved_eol is not None:
                new_header.eol = saved_eol
            # A promoted KV becomes a section header and needs a visual
            # separator from the parent's direct entries.
            if (
                self._body_tail is not None
                and new_header._prev is self._body_tail  # noqa: SLF001
                and not _layout_ops._leading_has_blank_line(new_header.leading)  # noqa: SLF001
            ):
                layout_root = self._layout_root
                nl = layout_root._newline if layout_root else "\n"  # noqa: SLF001
                new_header.leading.pieces.insert(0, NewlineNode(text=nl))
        return result

    def promote_array(self, key: str) -> AoT:
        """Convert an array-of-inline-tables at ``key`` into an AoT.

        Returns the live AoT view at ``key``. If the value is already
        an AoT, returns it unchanged. Raises ``TOMLError`` if the key
        is missing, refers to a non-array, an empty array, or an array
        with non-inline-table elements.
        """
        if self._inline:
            msg = "array-of-tables promotion is not supported on inline tables"
            raise TOMLError(msg)
        if key not in self:
            msg = f"key {key!r} not in table"
            raise KeyError(msg)
        cur = dict.__getitem__(self, key)
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
        if cur._value is not None:  # noqa: SLF001
            if _array_value_has_outer_comments(cur._value):  # noqa: SLF001
                msg = f"cannot promote {key!r}: array has comments that would be lost"
                raise TOMLError(msg)
            for entry_view in cur:
                ev = entry_view._value  # noqa: SLF001
                if ev is not None and _inline_value_has_inner_comments(ev):
                    msg = (
                        f"cannot promote {key!r}: array entry has inner "
                        f"comments that would be lost"
                    )
                    raise TOMLError(msg)
        snapshot = cur.to_list()
        # Carry the original KV slot's leading/eol onto the promoted AoT.
        old_slot = _direct_kv_slot(self, key)
        saved_leading = old_slot.leading if old_slot is not None else None
        saved_eol = old_slot.eol if old_slot is not None else None
        _layout_ops.delete_key(self, key)
        self[key] = AoT(snapshot)
        result = dict.__getitem__(self, key)
        assert isinstance(result, AoT)
        # Apply saved leading to the first entry header and saved eol to
        # the last entry's last slot.
        if saved_leading is not None and len(result) > 0:
            first_entry = result[0]
            entry_record = first_entry._owner_aot_entry  # noqa: SLF001
            if entry_record is not None and entry_record.entry_slots:
                first_slot = entry_record.entry_slots[0]
                if isinstance(first_slot, StructuralHeaderSlot):
                    # Preserve any separator already placed on the header.
                    first_slot.leading.pieces = [
                        *saved_leading.pieces,
                        *first_slot.leading.pieces,
                    ]
        if saved_eol is not None and len(result) > 0:
            last_entry = result[-1]
            entry_record = last_entry._owner_aot_entry  # noqa: SLF001
            if entry_record is not None and entry_record.entry_slots:
                last_slot = _layout_ops._entry_last_slot(  # noqa: SLF001
                    entry_record, self._attached_doc
                )
                if (
                    isinstance(last_slot, (KVSlot, StructuralHeaderSlot))
                    and saved_eol.comment is not None
                    and last_slot.eol.comment is None
                ):
                    last_slot.eol.comment = saved_eol.comment
                    if saved_eol.trailing_ws is not None:
                        last_slot.eol.trailing_ws = saved_eol.trailing_ws
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


def _populate_unattached(t: Container, mapping: Mapping[str, Any]) -> None:
    """Bulk-populate an unattached ``Container`` from a validated mapping.

    Bypasses ``Container.__setitem__`` (and its key typecheck) so the
    layout pipeline never re-pays validation work that the caller has
    already done.
    """
    for k, v in mapping.items():
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

    @classmethod
    def section(cls, mapping: Mapping[str, TomlInput] | None = None) -> Table:
        """Return a standard-section table, optionally populated from ``mapping``.

        Assign the result to install a ``[k]`` block:

            doc[k] = Table.section({"x": 1})
        """
        t = cls()
        if mapping is not None:
            mapping = _validate_mapping(mapping, label="Table.section argument")
            _populate_unattached(t, mapping)
        return t

    @classmethod
    def inline(cls, mapping: Mapping[str, TomlInput] | None = None) -> Table:
        """Return an inline table, optionally populated from ``mapping``.

        Assign the result to install a ``{x = 1}`` value:

            doc[k] = Table.inline({"x": 1})
        """
        t = cls()
        t._inline = True
        if mapping is not None:
            mapping = _validate_mapping(mapping, label="Table.inline argument")
            _populate_unattached(t, mapping)
        return t


class Document(Container):
    """Top-level TOML document.

    A [`Document`][tomlrt.Document] is the root of a parsed TOML
    file. It is a `dict` subclass and can be passed wherever a
    `dict` or `Mapping` is expected.
    """

    __slots__ = (
        "_displaced_recorder",
        "_head",
        "_install_recorder",
        "_is_private",
        "_newline",
        "_preamble",
        "_prelude",
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

        Existing [`Table`][tomlrt.Table] / [`AoT`][tomlrt.AoT] /
        [`Array`][tomlrt.Array] views are deep-cloned, so the returned
        document shares no mutable state with ``data``.
        """
        super().__init__()
        self._head: Slot | None = None
        self._tail: Slot | None = None
        self._trailing: Trivia = Trivia()
        self._preamble: Trivia = Trivia()
        self._newline: str = "\n"
        self._prelude: str = ""
        self._is_private: bool = False
        self._install_recorder: list[Slot] | None = None
        self._displaced_recorder: (
            list[tuple[Slot, list[TriviaPiece], Slot | None]] | None
        ) = None
        self._layout_root = self
        if data is not None:
            validated = _validate_mapping(data, label="Document data argument")
            for k, v in validated.items():
                self[k] = _coerce_for_document_init(v)

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
        """Comment block at the top of the document.

        A "preamble" is the run of ``# …`` lines that opens the file
        and is blank-line-separated from anything below. Comments
        directly above the first key (no blank line) are that key's
        `leading_comments`. With no structural content, the opening
        comment block is preamble.

        Setter accepts a sequence of bare comment texts (without the
        leading ``#``) and replaces the current preamble; assign ``()``
        to remove. Newlines inside any line are rejected.

        When the first slot is a section header or bare key, the
        preamble is stored as the above-blank prefix of that slot's
        leading trivia — the same storage that
        [`Table.header_leading_block`][tomlrt.Table.header_leading_block] /
        [`Document.leading_block`][tomlrt.Document.leading_block] expose
        for the first slot. Writes through either path are visible
        through the other.
        """
        return _doc_preamble_get(self)

    @preamble.setter
    def preamble(self, value: tuple[str, ...]) -> None:
        _doc_preamble_set(self, value)

    @preamble.deleter
    def preamble(self) -> None:
        _doc_preamble_set(self, ())

    @property
    def epilogue(self) -> tuple[str, ...]:
        """Comment block at the very end of the document.

        Returns the trailing run of ``# …`` lines that follows all
        structural content; with no structural content everything is
        `preamble`.

        Setter accepts a sequence of bare comment texts and replaces
        the current epilogue. Assign ``()`` to remove.

        Raises [`TOMLError`][tomlrt.TOMLError] if called with a
        non-empty value on a document with no structural content.
        """
        return _doc_epilogue_get(self)

    @epilogue.setter
    def epilogue(self, value: tuple[str, ...]) -> None:
        _doc_epilogue_set(self, value)

    @epilogue.deleter
    def epilogue(self) -> None:
        _doc_epilogue_set(self, ())

    @override
    def __copy__(self) -> Document:
        # Round-trip via dumps/loads: preserves bytes exactly.
        from tomlrt._public import loads  # noqa: PLC0415

        return loads(self.render())

    @override
    def __deepcopy__(self, memo: dict[int, object]) -> Document:
        from tomlrt._public import loads  # noqa: PLC0415

        return loads(self.render())


def _inline_value_has_inner_comments(v: object) -> bool:
    """Return True iff the inline-table value carries inner comments.

    Used to refuse ``promote_inline`` on inline tables whose comments
    would have nowhere to live in the promoted form.
    """
    return isinstance(v, InlineTableValue) and _comma_value_has_outer_comments(v)


def _array_value_has_outer_comments(v: object) -> bool:
    """Return True iff the array carries item-level or final comments.

    "Outer" here means comments at the array layer itself; nested
    inline-value comments are tested separately (and produce a
    different error message).
    """
    return isinstance(v, ArrayValue) and _comma_value_has_outer_comments(v)


def _comma_value_has_outer_comments(v: CommaValue[_ItemT]) -> bool:
    if trivia_has_comment(v.header_trivia) or trivia_has_comment(v.final_trivia):
        return True
    return any(
        trivia_has_comment(p.leading)
        or trivia_has_comment(p.trailing)
        or trivia_has_comment(p.post_comma_trivia)
        for p in v.items
    )


def _deep_clone(c: Container) -> Container:
    """Build a detached deep clone of ``c`` as a lossy snapshot.

    The clone preserves the source's inline vs section shape (so a
    deepcopy of an inline table returns an inline view) and recurses
    into nested ``Container`` / ``AoT`` values. Render-level formatting
    is not preserved; for byte-exact preservation of an entire document,
    use ``Document``'s ``__deepcopy__``.
    """
    out = Table.inline() if c._inline else Table.section()  # noqa: SLF001
    for k, v in c.items():
        if isinstance(v, Container):
            dict.__setitem__(out, k, _deep_clone(v))
        elif isinstance(v, AoT):
            dict.__setitem__(out, k, AoT([_deep_clone(e) for e in v]))
        else:
            dict.__setitem__(out, k, _to_python(v))
    return out


def _reset_table_for_rehome(t: Container, *, recurse: bool = False) -> None:
    """Clear a Table's slot infrastructure so it can be reattached.

    Preserves dict storage (so post-detach mutations survive) but
    drops `_layout_root` / `_path` / `_parent` / `_owner_aot_entry`
    / `_refs` / `_index` / `_header_ref` / `_body_tail` so the
    standard attach path treats `t` as if freshly constructed.

    With ``recurse=True``, also resets nested non-inline ``Container`` /
    ``AoT`` children from the same detached subtree. Children pointing
    at a different doc are left for the cross-doc clone path. Recursion
    is opt-in so single-table rehomes avoid a subtree walk.

    Used when re-installing a held view that was detached into a
    private orphan ``Document``.
    """
    old_root = t._layout_root  # noqa: SLF001
    t._layout_root = None  # noqa: SLF001
    t._path = ()  # noqa: SLF001
    t._parent = None  # noqa: SLF001
    t._owner_aot_entry = None  # noqa: SLF001
    t._refs = []  # noqa: SLF001
    t._index = {}  # noqa: SLF001
    t._header_ref = None  # noqa: SLF001
    t._body_tail = None  # noqa: SLF001

    if not recurse:
        return
    for child in dict.values(t):
        if _is_section(child):
            if child._layout_root is old_root:  # noqa: SLF001
                _reset_table_for_rehome(child, recurse=True)
        elif isinstance(child, AoT) and child._layout_root is old_root:  # noqa: SLF001
            for entry in list.__iter__(child):
                _reset_table_for_rehome(entry, recurse=True)
            child._layout_root = None  # noqa: SLF001
            child._parent = None  # noqa: SLF001
            child._path = ()  # noqa: SLF001


def _reset_inline_for_rehome(t: Container) -> None:
    """Clear an inline Table's slot infrastructure so it can be reattached.

    Inline tables are slot-less from the doc-stream perspective but
    keep an ``InlineTableValue`` in their ``_value`` field once
    attached. Drop that and the layout-root pointer so the standard
    inline-attach path treats ``t`` as if freshly constructed.
    """
    t._layout_root = None  # noqa: SLF001
    t._parent = None  # noqa: SLF001
    t._owner_aot_entry = None  # noqa: SLF001
    t._value = None  # noqa: SLF001


def _reset_array_for_rehome(a: Array) -> None:
    """Clear an Array view's attachment so it can be reattached live.

    Leaves ``_value`` (the displaced ``ArrayValue``) intact so the
    next attach can reuse it.
    """
    a._layout_root = None  # noqa: SLF001


def _install_attached_subtree(
    dst_parent: Container, dst_path: tuple[str, ...], src_table: Container
) -> None:
    """Recursively install an attached implicit / Document source.

    Section / AoT children clone via tuple-path :meth:`Container.install`
    so headers and slots survive. Direct entries at this implicit level
    are written as dotted KVs hosted by ``dst_parent``'s nearest
    header-bearing ancestor, preserving dotted form per key.

    Note: bucketing into directs vs structurals can reorder relative
    to ``src_table.items()`` — at a given implicit level all dotted
    leaves emit before any subsection. TOML is insensitive to that
    order; the dotted-form preservation is the win.
    """
    direct_kvs: list[tuple[str, object]] = []
    structural: list[tuple[str, object]] = []
    for k, v in src_table.items():
        if isinstance(v, AoT) or (_is_section(v)):
            structural.append((k, v))
        else:
            direct_kvs.append((k, v))

    if direct_kvs:
        _install_dotted_direct_kvs(dst_parent, dst_path, direct_kvs, src_table)

    for k, v in structural:
        sub_path = (*dst_path, k)
        if isinstance(v, AoT) or (
            isinstance(v, Container) and v._header_ref is not None  # noqa: SLF001
        ):
            dst_parent.install(sub_path, v)
        elif isinstance(v, Container):
            _install_attached_subtree(dst_parent, sub_path, v)


def _install_dotted_direct_kvs(
    dst_parent: Container,
    dst_path: tuple[str, ...],
    direct_kvs: list[tuple[str, object]],
    src_table: Container,
) -> None:
    """Emit each ``(k, v)`` in ``direct_kvs`` as a dotted KV under host.

    ``host`` is the nearest header-bearing ancestor at-or-above
    ``dst_parent`` (or the doc / AoT-entry root). Creates implicit
    intermediates as needed. Each value's CST and leading trivia are
    deep-cloned from the corresponding source slot so string/number
    style, inline-array pad, and standalone comments survive — a
    re-synthesis from the logical value would drop all of those.
    """
    from tomlrt._build import _decode_value  # noqa: PLC0415

    doc = dst_parent._attached_doc  # noqa: SLF001
    host: Container = dst_parent
    while host._header_ref is None and host._parent is not None:  # noqa: SLF001
        host = host._parent  # noqa: SLF001
    parent_to_host = dst_parent._path[len(host._path) :]  # noqa: SLF001
    owner = host._owner_aot_entry  # noqa: SLF001
    for k, _v in direct_kvs:
        leaf_keypath = (*parent_to_host, *dst_path, k)
        leaf_parent = _layout_ops.ensure_implicit_chain(host, leaf_keypath[:-1])
        path = (*host._path, *leaf_keypath)  # noqa: SLF001
        # A direct (non-structural) key of an attached source is always
        # backed by a single KVSlot; clone its value + leading so style
        # and standalone comments survive (re-synthesis would drop them).
        src_slot = src_table._index[k][0].slot  # noqa: SLF001
        assert isinstance(src_slot, KVSlot)
        cst = copy.deepcopy(src_slot.value)
        _retarget_to_doc(cst, doc)
        leading = copy.deepcopy(src_slot.leading)
        retarget_trivia_newlines(leading, doc._newline)  # noqa: SLF001
        decoded = _decode_value(
            cst, layout_root=doc, parent=leaf_parent, path=path, owner=owner
        )
        _layout_ops.install_dotted_kv_slot(
            host, leaf_keypath, cst, leaf_parent=leaf_parent, leading=leading
        )
        dict.__setitem__(leaf_parent, k, decoded)


def _to_python(v: Any) -> Any:
    """Recursively materialise a tomlrt view into plain Python values."""
    if isinstance(v, Container):
        return v.to_dict()
    if isinstance(v, AoT):
        return [t.to_dict() for t in v]
    if isinstance(v, Array):
        return [_to_python(x) for x in v]
    return v


# ---------------------------------------------------------------------------
# Scalar coercion
# ---------------------------------------------------------------------------


def _is_section(v: object) -> TypeGuard[Container]:
    """True iff ``v`` is a non-inline (section-style) Container."""
    return isinstance(v, Container) and not v._inline  # noqa: SLF001


def _is_inline_table(v: object) -> TypeGuard[Container]:
    """True iff ``v`` is an inline Container."""
    return isinstance(v, Container) and v._inline  # noqa: SLF001


def _coerce_for_document_init(v: Any) -> Any:
    """Pick a sensible structural shape for ``Document(data=...)`` values.

    * Mapping → section ``Table.section`` (recursively coerced).
    * Plain ``list`` of mappings (non-empty) → ``AoT`` of section tables.
    * Anything else passes through unchanged.

    A user-supplied ``Array`` (even one carrying mappings) is *not*
    coerced — the caller has explicitly chosen inline-array shape.
    """
    if isinstance(v, AoT):
        return v
    if isinstance(v, Container):
        return v
    if isinstance(v, Array):
        return v
    if isinstance(v, Mapping):
        return Table.section(
            {k: _coerce_for_document_init(sub) for k, sub in v.items()}
        )
    if isinstance(v, list) and v and all(isinstance(x, Mapping) for x in v):
        return AoT(
            [{k: _coerce_for_document_init(sub) for k, sub in m.items()} for m in v]
        )
    return v


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

    Accepts:
    - any ``Mapping`` or inline ``Container`` view (deep-copy semantics)
    - ``list`` or ``Array`` views (deep-copy semantics)

    Rejects everything else (tuple, bytes, sets, AoT, section
    Container, …) so the caller can route to a stronger error.
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
    # `list` only — `tuple` is intentionally not accepted (TOML has no
    # tuple, and accepting it would mask user typos).
    return type(v) is list or (isinstance(v, list) and not isinstance(v, Array))


def _synth_value(
    v: object,
    *,
    layout_root: Document | None,
    parent: Container | None,
    path: tuple[str, ...],
    owner: AoTEntry | None,
) -> tuple[Value, object]:
    """Synthesise a (CST value, decoded view) pair from ``v``.

    Plain ``dict`` / ``Mapping`` → ``InlineTableValue`` + inline ``Table``.
    ``list`` / ``Array`` view → ``ArrayValue`` + ``Array``.
    Section ``Container`` / ``AoT`` raise ``TOMLError`` — those can't
    live as inline values. Anything else raises the canonical
    ``TypeError``.
    """
    if is_scalar(v):
        return coerce_scalar(v), v
    if isinstance(v, AoT):
        msg = "Cannot store an array-of-tables inside an inline table"
        raise TOMLError(msg)
    if _is_section(v):
        msg = "Cannot store a section-style table inside an inline-style table"
        raise TOMLError(msg)
    # Live-attach unattached inline values so user identity is preserved.
    # Displaced inline tables need a reset before reattach; arrays do not.
    if (_is_inline_table(v) or isinstance(v, Array)) and not v._attached:  # noqa: SLF001
        if isinstance(v, Array):
            _retarget_to_doc(v._value, layout_root)  # noqa: SLF001
            _attach_array_view(v, layout_root, owner)
            return v._value, v  # noqa: SLF001
        if v._layout_root is not None:  # noqa: SLF001
            _reset_inline_for_rehome(v)
        return _populate_inline_table(
            v,
            list(v.items()),
            layout_root=layout_root,
            parent=parent,
            path=path,
            owner=owner,
        )
    # Cross-document / same-doc live inline values clone CST so source
    # formatting survives. Plain Mapping / list inputs have no CST.
    if _is_inline_table(v) or isinstance(v, Array):
        from tomlrt._build import _decode_value  # noqa: PLC0415

        src_val = v._value  # noqa: SLF001
        assert src_val is not None
        cloned = copy.deepcopy(src_val)
        _retarget_to_doc(cloned, layout_root)
        new = _decode_value(
            cloned, layout_root=layout_root, parent=parent, path=path, owner=owner
        )
        return cloned, new
    # Plain ``Mapping`` → inline table (synthesise from items).
    if isinstance(v, Mapping):
        return _populate_inline_table(
            Table(),
            list(v.items()),
            layout_root=layout_root,
            parent=parent,
            path=path,
            owner=owner,
        )
    # Plain ``list`` → inline array (synthesise from items).
    if isinstance(v, list):
        return _synth_inline_array(v, layout_root=layout_root, owner=owner)
    msg = f"Cannot convert {type(v).__name__} to a TOML value"
    raise TypeError(msg)


def _retarget_to_doc(val: Value, layout_root: Document | None) -> None:
    r"""Retarget ``val``'s baked-in newlines to ``layout_root``'s line ending.

    Called whenever pre-existing inline CST is dragged into a
    destination doc. Without this, cross-document grafts and unattached
    ``Array(multiline=True)`` factories can dump mixed ``\n`` /
    ``\r\n`` newlines.
    """
    if layout_root is not None:
        retarget_value_newlines(val, layout_root._newline)  # noqa: SLF001


def _attach_array_view(
    arr: Array, layout_root: Document | None, owner: AoTEntry | None
) -> None:
    """Record the document attachment on an Array and its inline children."""
    arr._layout_root = layout_root  # noqa: SLF001
    for child in arr:
        _attach_inline_child_view(child, layout_root, owner)


def _attach_inline_child_view(
    value: object, layout_root: Document | None, owner: AoTEntry | None
) -> None:
    if isinstance(value, Array):
        _attach_array_view(value, layout_root, owner)
    elif _is_inline_table(value):
        value._layout_root = layout_root  # noqa: SLF001
        value._owner_aot_entry = owner  # noqa: SLF001
        for child in value.values():
            _attach_inline_child_view(child, layout_root, owner)


def _populate_inline_table(
    table: Table | Container,
    items: list[tuple[object, object]],
    *,
    layout_root: Document | None,
    parent: Container | None,
    path: tuple[str, ...],
    owner: AoTEntry | None,
) -> tuple[InlineTableValue, Container]:
    """Wire ``table`` as an inline view and populate its entries.

    The live-attach path passes a user-supplied ``Table.inline()`` so
    identity is preserved; the plain-Mapping synth path passes a fresh
    ``Table()``. Entries use canonical single-line spacing.
    """
    val = InlineTableValue()
    table._wire(  # noqa: SLF001
        layout_root=layout_root, parent=parent, path=path, owner=owner
    )
    table._inline = True  # noqa: SLF001
    table._value = val  # noqa: SLF001

    for i, (raw_k, sub) in enumerate(items):
        k = _validate_key(raw_k)
        sub_cst, sub_dec = _synth_value(
            sub,
            layout_root=layout_root,
            parent=table,
            path=(*path, k),
            owner=owner,
        )
        is_last = i == len(items) - 1
        entry = InlineTableEntry(
            leading=Trivia() if i == 0 else Trivia([WhitespaceNode(text=" ")]),
            key_parts=[make_keypart(k)],
            key_seps=[],
            pre_eq=" ",
            post_eq=" ",
            value=sub_cst,
            trailing=Trivia(),
            has_comma=not is_last,
            post_comma_trivia=Trivia(),
            key_path=(k,),
        )
        val.items.append(entry)
        dict.__setitem__(table, k, sub_dec)
    if items:
        pad = Trivia([WhitespaceNode(text=val._single_line_pad)])  # noqa: SLF001
        val.header_trivia = pad.copy()
        val.final_trivia = pad
    return val, table


def _synth_inline_array(
    items: Sequence[object],
    *,
    layout_root: Document | None,
    owner: AoTEntry | None,
) -> tuple[ArrayValue, Array]:
    val = ArrayValue()
    arr = Array()
    arr._value = val  # noqa: SLF001
    arr._layout_root = layout_root  # noqa: SLF001

    for i, sub in enumerate(items):
        sub_cst, sub_dec = _synth_value(
            sub,
            layout_root=layout_root,
            parent=None,
            path=(),
            owner=owner,
        )
        is_last = i == len(items) - 1
        # Under the canonical model, inter-item separators live in the
        # NEXT item's leading; items[0].leading is always empty;
        # post_comma_trivia carries only EOL sections (empty here).
        item = ArrayItem(
            leading=Trivia() if i == 0 else Trivia([WhitespaceNode(text=" ")]),
            value=sub_cst,
            trailing=Trivia(),
            has_comma=not is_last,
            post_comma_trivia=Trivia(),
        )
        val.items.append(item)
        list.append(arr, sub_dec)
    return val, arr


__all__ = ["AoT", "Array", "Container", "Document", "Table", "TomlInput"]
