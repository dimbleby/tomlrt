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
    _canon_kv_slot,
    _canon_leading,
    _resolve_format_options,
    format_document_trailing,
    format_inline_root,
    format_subtree,
)
from tomlrt._inline_comments import (
    InlineEolView,
    InlineLeadingBlockView,
    InlineLeadingView,
)
from tomlrt._kind import _Kind
from tomlrt._paths import validate_path
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
    has_comment,
    leading_has_blank_line,
    retarget_trivia_newlines,
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
    from collections.abc import Callable, Iterable, MutableMapping, Sequence

    from _typeshed import SupportsKeysAndGetItem, SupportsRichComparison
    from typing_extensions import Self

    from tomlrt._format import FormatOptions
    from tomlrt._scalar import Scalar
    from tomlrt._slots import AoTEntry, Slot, SlotRef
    from tomlrt._trivia import EolTrivia, TriviaPiece
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
    def comments(self) -> MutableMapping[str, str]:
        """Mapping view of EOL comments on this container's direct keys.

        For a section-backed container the view is keyed by direct key
        and requires attachment to a `Document` to mutate. For an inline
        table the view is keyed by direct leaf key over the backing
        inline value; setting a comment promotes a single-line table to
        multi-line (a single line has nowhere to hold a comment).
        """
        if self._inline:
            return InlineEolView(self)
        return EolCommentView(self)

    @property
    def leading_comments(self) -> MutableMapping[str, tuple[str, ...]]:
        """Mapping view of leading-comment blocks on this container's direct keys.

        Returns only the *attached* comment run immediately above each key
        (no blank line between). For the full block, including any
        above-blank groups and the blank-line structure between them, see
        [`leading_block`][tomlrt.Table.leading_block].

        Mutating the view requires the container to be attached to a
        `Document`. On an inline table the view is keyed by direct leaf
        key over the backing inline value; it has the same attached-run
        semantics as a section-backed table, and setting comments promotes
        a single-line table to multi-line.
        """
        if self._inline:
            return InlineLeadingView(self)
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

        Mutating the view requires the container to be attached to a
        `Document`. On an inline table the view is keyed by direct leaf
        key over the backing inline value; an opening-bracket EOL comment
        is framing and is not part of the first entry's block.
        """
        if self._inline:
            return InlineLeadingBlockView(self)
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

    @property
    def _doc_newline(self) -> str:
        r"""The active newline of the owning document, or ``"\n"`` if detached."""
        lr = self._layout_root
        return lr._newline if lr is not None else "\n"  # noqa: SLF001

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
          multi-line stays multi-line).
        * Newlines use the owning document's style.

        ``comments=`` is deprecated; use
        ``FormatOptions(normalize_comments=...)`` instead. Supplying both
        arguments raises ``ValueError``.

        Detached factory-style containers (``Table.section()`` /
        ``Table.inline()`` not yet assigned anywhere) and inline dotted
        navigators are unsupported and raise `TOMLError`.
        """
        resolved = _resolve_format_options(options=options, comments=comments)
        kind = self._kind
        nl = self._doc_newline

        if kind is _Kind.INLINE_ROOT:
            assert self._value is not None
            format_inline_root(self._value, nl=nl, options=resolved)
            return

        if kind in (_Kind.INLINE_FACTORY, _Kind.INLINE_DOTTED_INNER):
            msg = "format() is not supported on detached inline-table views"
            raise TOMLError(msg)

        if kind is _Kind.DOCUMENT:
            assert isinstance(self, Document)
            # A non-empty preamble already ends in one blank-line
            # separator, so the first slot must contribute none;
            # otherwise the document opens with at most one blank line.
            format_subtree(
                start=self._head,
                path=(),
                owner=None,
                nl=nl,
                options=resolved,
                head_blank_cap=0 if self._preamble.pieces else 1,
            )
            format_document_trailing(self._preamble, nl=nl, options=resolved)
            format_document_trailing(self._trailing, nl=nl, options=resolved)
            return

        if kind is _Kind.SECTION:
            assert self._header_ref is not None
            format_subtree(
                start=self._header_ref.slot,
                path=self._path,
                owner=self._owner_aot_entry,
                nl=nl,
                options=resolved,
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
                _canon_kv_slot(slot, nl=nl, options=resolved)
            elif isinstance(slot, StructuralHeaderSlot):
                _canon_header_slot(slot, nl=nl, options=resolved)
            _canon_leading(slot, nl=nl, target_blanks=None, options=resolved)
        for value in self.values():
            if isinstance(value, (Container, Array)):
                value.format(options=resolved)
            elif isinstance(value, AoT):
                for entry in value:
                    entry.format(options=resolved)

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
    def _is_own_aot_entry(self) -> bool:
        """True iff this container *is* an AoT entry, not merely nested inside one.

        ``_owner_aot_entry`` is set on the entry's own table *and* on
        every section, inline table, or implicit descendant physically
        nested inside it — only the entry's own table has a header whose
        ``entry`` matches ``_owner_aot_entry``.
        """
        if self._owner_aot_entry is None or self._header_ref is None:
            return False
        header = self._header_ref.slot
        assert isinstance(header, StructuralHeaderSlot)
        return header.entry is self._owner_aot_entry

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
        """Materialise a plain-Python ``dict`` (recursive)."""
        out: dict[str, Any] = {}
        for k, v in self.items():
            out[k] = _to_python(v)
        return out

    def _synth_local_value(self, key: str, value: object) -> tuple[Value, object]:
        """Synthesise ``value`` for direct storage under ``key`` on ``self``."""
        return _synth_value(
            value,
            layout_root=self._layout_root,
            parent=self,
            path=(*self._path, key),
            owner=self._owner_aot_entry,
        )

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
                msg = "cannot store an array-of-tables inside an inline table"
                raise TOMLError(msg)
            if _is_section(value):
                msg = "cannot store a section-style table inside an inline-style table"
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
        # old user references from the live doc.
        if (
            is_scalar(value)
            or _is_synth_inline(value)
            or isinstance(value, AoT)
            or _is_section(value)
            or isinstance(value, Mapping)
        ):
            value = _snapshot_for_overlapping_install(self, key, value)
            _layout_ops.reposition_install(self, key, value)
            return
        # Unsupported value type — TypeError, not NIE.
        msg = (
            f"cannot convert value of type {type(value).__name__!r} "
            f"for TOML key {key!r}"
        )
        raise TypeError(msg)

    def _insert_new(
        self,
        key: str,
        value: Any,
        *,
        reinstall_as_dotted: bool = False,
    ) -> None:
        """Bind ``key`` for the first time at the document tail."""
        if is_scalar(value):
            _layout_ops.append_direct_kv(
                self,
                key,
                coerce_scalar(value),
                reinstall_as_dotted=reinstall_as_dotted,
            )
            dict.__setitem__(self, key, value)
            return
        if _is_synth_inline(value):
            cst, decoded = self._synth_local_value(key, value)
            _layout_ops.append_direct_kv(
                self,
                key,
                cst,
                reinstall_as_dotted=reinstall_as_dotted,
            )
            dict.__setitem__(self, key, decoded)
            return
        if isinstance(value, AoT):
            self._attach_aot(key, value)
            return
        if _is_section(value):
            self._attach_section(key, value)
            return
        # Unsupported types get the canonical TypeError.
        msg = f"cannot convert {type(value).__name__} to a TOML value"
        raise TypeError(msg)

    def _attach_aot(self, key: str, value: AoT) -> None:
        """Install ``value`` (an AoT) under ``key``.

        Live-attached sources or private orphans with intact entry
        slots route through :func:`clone_aot_entry` to preserve per-entry
        trivia and nested sub-sections. Detached AoTs without preserved
        slots are rehomed entry-by-entry.
        """
        src_root = value._layout_root  # noqa: SLF001
        if src_root is not None and not src_root._is_private:  # noqa: SLF001
            _layout_ops.clone_aot(self, key, value)
            return
        # Private orphans may still own intact entry_slots from a
        # structural-overwrite detach; clone those to keep per-KV
        # trivia, nested sub-sections, and inter-entry separators.
        # The generic add_aot_entry(rehome=) path rebuilds from dict
        # storage and drops that CST.
        existing_entries: list[Table] = list(value)
        list.clear(value)
        value._layout_root = None  # noqa: SLF001
        value._parent = None  # noqa: SLF001
        value._path = ()  # noqa: SLF001
        attached = _layout_ops.attach_empty_aot(self, key, value)
        dict.__setitem__(self, key, attached)
        for entry_table in existing_entries:
            owner = entry_table._owner_aot_entry  # noqa: SLF001
            preserve_cst = owner is not None and bool(owner.entry_slots)
            if preserve_cst:
                # Gathering includes nested AoTs and requires the live view.
                _layout_ops.clone_aot_entry(
                    value,
                    entry_table,
                    dst_path=value._path,  # noqa: SLF001
                    preserve_source_separator=True,
                )
            _reset_table_for_rehome(entry_table)
            if not preserve_cst:
                _layout_ops.add_aot_entry(value, None, rehome=entry_table)

    def _attach_section(self, key: str, value: Container) -> None:
        """Install ``value`` (a section-flavoured Table) under ``key``.

        AoT-entry sources clone as a table so trivia survives and the
        head normalises from ``[[..]]`` to ``[..]``. Attached
        header-bearing sections clone slots; a whole attached
        ``Document`` clones its body under a synthesised header (so body
        comments and inline pad survive); attached implicit sources
        recurse via ``_install_attached_subtree``. A private orphan is
        rehomed in place — its slots move into the document, preserving
        identity and trivia; a truly detached source (no slots) is
        synthesised.
        """
        src_root = value._layout_root
        live_source = src_root is not None and not src_root._is_private  # noqa: SLF001
        if live_source:
            if value._is_own_aot_entry and self._layout_root is not None:
                _layout_ops.clone_aot_entry_as_table(self, key, value)
            elif value._header_ref is not None:
                _layout_ops.clone_section_as_section(self, key, value)
            elif isinstance(value, Document):
                _layout_ops.clone_document_as_section(self, key, value)
            else:
                value = _snapshot_for_overlapping_install(self, key, value)
                _install_attached_subtree(self, (key,), value)
            return
        if src_root is not None and src_root._is_private:  # noqa: SLF001
            # Private orphan with intact slots: move the slots into the
            # document so identity and trivia both survive. A header-bearing
            # section (an AoT entry is normalised to a plain section) moves
            # its block; a header-less implicit section moves its dotted KVs.
            if value._header_ref is not None:  # noqa: SLF001
                _layout_ops.adopt_private_section(self, key, value)
            else:
                _layout_ops.adopt_private_implicit(self, key, value)
            return
        # A Container with no layout root can't be a Document (which is
        # always its own root), so it must be an unattached Table.
        assert isinstance(value, Table), (
            "internal: detached section source expected to be a Table"
        )
        _layout_ops.attach_section_at(self, (key,), value)

    def _replace_scalar(self, key: str, value: Scalar) -> None:
        """Replace a scalar while preserving its existing KV slot."""
        refs = self._index.get(key)
        assert refs is not None, "scalar value must have a slot"
        slot = refs[0].slot
        assert isinstance(slot, KVSlot), "scalar value must have a KV slot"
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
        assert refs is not None, "inline value must have a slot"
        assert len(refs) == 1, "inline value must be backed by exactly one slot"
        primary = refs[0]
        slot = primary.slot
        assert isinstance(slot, KVSlot), "inline value must be backed by a KV slot"
        old = dict.__getitem__(self, key)
        cst, decoded = self._synth_local_value(key, value)
        slot.value = cst
        dict.__setitem__(self, key, decoded)
        # Detach the displaced view so it can be reattached live.
        if _is_inline_table(old) and old is not decoded:
            _reset_inline_for_rehome(old)
        elif isinstance(old, Array) and old is not decoded:
            _reset_array_for_rehome(old)

    @override
    def __delitem__(self, key: str) -> None:
        # Unattached factory mode: dict-only storage until attach.
        if self._layout_root is None:
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
        if key in self:
            assert isinstance(key, str)
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
        """Whether ``key``'s rendered block contains a structural header.

        This describes the whole block, not the child table itself: with
        ``[a.b]``, ``doc.has_header("a")`` is true although ``a`` is implicit
        and only ``b`` owns the header. Structural headers are ``[header]``
        sections and ``[[header]]`` array-of-tables entries. Returns ``False``
        for bare ``key = value`` leaves, inline tables, implicit sections built
        entirely from dotted keys (e.g. ``a.x = 1``), and missing keys.
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
        return self.__copy__()

    # ------------------------------------------------------------------
    # Inline-table dispatch
    # ------------------------------------------------------------------

    def _inline_setitem(self, key: str, value: Any) -> None:
        # ``__setitem__`` has already rejected ``AoT`` / section values
        # for inline hosts. Non-coerceable types (``set``, custom
        # classes, …) reach ``_synth_value`` for the canonical TypeError.
        cst, decoded = self._synth_local_value(key, value)
        if key in self and isinstance(dict.__getitem__(self, key), Container):
            # Stay on the CST side when replacing a dotted-prefix
            # navigator; dict-level delete would prune the parent chain.
            _inline_ops.overwrite_entry(self, key, cst)
        elif key in self:
            _inline_ops.replace_entry_value(self, key, cst)
        else:
            _inline_ops.append_entry(self, key, cst)
        dict.__setitem__(self, key, decoded)

    def _inline_delitem(self, key: str) -> None:
        if key not in self:
            raise KeyError(key)
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
            cur, i = _walk_existing_sections(
                self, parts, limit=len(parts) - 1, stop_on_non_section=True
            )
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
        ``[a.b.c]`` header. Raises `TOMLError` if a component cannot be
        descended through: an existing array-of-tables, or a non-table
        value or inline table that would have to be traversed.
        """
        parts = validate_path(key)
        cur, i = _walk_existing_sections(
            self, parts, limit=len(parts), stop_on_non_section=False
        )
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
        ``KeyError`` if the key is missing, or ``TypeError`` if it
        doesn't refer to an inline-style table.
        """
        cur = self._require_promotable_entry(key, action="inline-table promotion")
        if not (_is_inline_table(cur)):
            msg = f"{key!r} is not an inline table"
            raise TypeError(msg)
        if _inline_value_has_inner_comments(cur._value):  # noqa: SLF001
            msg = (
                f"cannot promote {key!r}: inline table has inner "
                f"comments that would be lost"
            )
            raise TOMLError(msg)
        value = cur._value  # noqa: SLF001
        assert isinstance(value, InlineTableValue)
        entries = _layout_ops.prepare_promoted_inline_entries(value.items)
        # Transfer the existing KV slot's leading + eol to the header.
        saved_leading, saved_eol = _direct_kv_trivia(self, key)
        _layout_ops.delete_key(self, key)
        self[key] = Table.section()
        result = dict.__getitem__(self, key)
        assert isinstance(result, Table)
        _layout_ops.populate_promoted_inline_entries(result, entries)
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
                and not leading_has_blank_line(new_header.leading)
            ):
                layout_root = self._layout_root
                nl = layout_root._newline if layout_root else "\n"  # noqa: SLF001
                new_header.leading.pieces.insert(0, NewlineNode(text=nl))
        return result

    def promote_array(self, key: str) -> AoT:
        """Convert an array-of-inline-tables at ``key`` into an AoT.

        Returns the live AoT view at ``key``. Raises ``KeyError`` if the
        key is missing, ``TypeError`` if it refers to a non-array, or
        ``TOMLError`` for an empty array or an array with
        non-inline-table elements.
        """
        cur = self._require_promotable_entry(key, action="array-of-tables promotion")
        if not isinstance(cur, Array):
            msg = f"{key!r} is not an array"
            raise TypeError(msg)
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
            last_slot = _layout_ops._body_anchor(last_entry)  # noqa: SLF001
            assert last_slot is not None
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


def _direct_kv_trivia(c: Container, key: str) -> tuple[Trivia | None, EolTrivia | None]:
    """Return the direct-KV slot's leading/EOL trivia for ``key``, if any."""
    slot = _direct_kv_slot(c, key)
    assert slot is not None, "inline promotion source must have a direct KV slot"
    return slot.leading, slot.eol


def _walk_existing_sections(
    start: Container,
    parts: Sequence[str],
    *,
    limit: int,
    stop_on_non_section: bool,
) -> tuple[Container, int]:
    """Walk the existing section-backed prefix of ``parts`` under ``start``."""
    cur: Container = start
    i = 0
    while i < limit:
        p = parts[i]
        if p not in cur:
            break
        nxt = dict.__getitem__(cur, p)
        if isinstance(nxt, AoT):
            action = "install" if stop_on_non_section else "ensure_table"
            msg = (
                f"cannot {action} through array-of-tables at {p!r}: "
                "no addressable target inside an AoT"
            )
            raise TOMLError(msg)
        if not isinstance(nxt, Container) or (nxt._inline and i < len(parts) - 1):  # noqa: SLF001
            if stop_on_non_section:
                break
            msg = (
                f"existing value at {p!r} is not section-backed "
                "(is an inline table or non-table value)"
            )
            raise TOMLError(msg)
        cur = nxt
        i += 1
    return cur, i


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
        spaces.

        Raises [`TOMLError`][tomlrt.TOMLError] on a non-inline table, and
        when collapsing a multi-line table that carries comments anywhere
        in it (they would have nowhere to live on one line).

        Returns ``self`` for chaining.
        """
        root = self._require_inline_root("set_multiline")
        _inline_ops.set_inline_multiline(root, multiline=multiline, indent=" " * indent)
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

        A [`Table`][tomlrt.Table] / [`AoT`][tomlrt.AoT] /
        [`Array`][tomlrt.Array] view that is not already attached to a
        document *attaches live*: the returned document keeps your
        object, so later mutations through either reference are visible
        on the other. A view already attached to another document is
        deep-cloned instead, so the two never share state. Plain
        ``dict`` / ``list`` values are always snapshot-copied.
        """
        super().__init__()
        self._head: Slot | None = None
        self._tail: Slot | None = None
        self._trailing: Trivia = Trivia()
        self._preamble: Trivia = Trivia()
        self._newline: str = "\n"
        self._prelude: str = ""
        self._is_private: bool = False
        self._install_recorders: (
            tuple[
                list[Slot],
                list[tuple[Slot, list[TriviaPiece], Slot | None]],
            ]
            | None
        ) = None
        self._section_blank_separated = True
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
        # Round-trip via dumps/loads: preserves bytes exactly.
        from tomlrt._public import loads  # noqa: PLC0415

        return loads(self.render())

    @override
    def __deepcopy__(self, memo: dict[int, object]) -> Document:
        return self.__copy__()


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
    if has_comment(v.header_trivia.pieces) or has_comment(v.final_trivia.pieces):
        return True
    return any(
        has_comment(p.leading.pieces)
        or has_comment(p.trailing.pieces)
        or has_comment(p.post_comma_trivia.pieces)
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


def _reset_table_for_rehome(t: Container) -> None:
    """Clear a Table's slot infrastructure so it can be reattached.

    Preserves dict storage (so post-detach mutations survive) but
    drops `_layout_root` / `_path` / `_parent` / `_owner_aot_entry`
    / `_refs` / `_index` / `_header_ref` / `_body_tail` so the
    standard attach path treats `t` as if freshly constructed.

    Also resets nested non-inline ``Container`` / ``AoT`` children from
    the same detached subtree. Children pointing at a different doc are
    left for the cross-doc clone path.

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

    for child in dict.values(t):
        if _is_section(child):
            if child._layout_root is old_root:  # noqa: SLF001
                _reset_table_for_rehome(child)
        elif isinstance(child, AoT) and child._layout_root is old_root:  # noqa: SLF001
            for entry in list.__iter__(child):
                _reset_table_for_rehome(entry)
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
            # Bypass Container.install()'s tuple-path validation here:
            # `k` is a key already known valid on a live source Container
            # (an empty string is a legal — if unusual — TOML key), not a
            # human-supplied dotted path where an empty segment signals a
            # typo. install() rejects the latter; ensure_implicit_chain +
            # a direct assignment only validates `k` as a single key.
            leaf_parent = _layout_ops.ensure_implicit_chain(dst_parent, sub_path[:-1])
            leaf_parent[sub_path[-1]] = v
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
    destination = _layout_ops.ensure_implicit_chain(dst_parent, dst_path)
    host = destination
    while host._header_ref is None and host._parent is not None:  # noqa: SLF001
        host = host._parent  # noqa: SLF001
    owner = host._owner_aot_entry  # noqa: SLF001
    destination_to_host = destination._path[len(host._path) :]  # noqa: SLF001
    destination_path = destination._path  # noqa: SLF001
    for k, _v in direct_kvs:
        leaf_keypath = (*destination_to_host, k)
        path = (*destination_path, k)
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
            cst, layout_root=doc, parent=destination, path=path, owner=owner
        )
        _layout_ops.install_dotted_kv_slot(
            host,
            leaf_keypath,
            cst,
            leaf_parent=destination,
            leading=leading,
        )
        dict.__setitem__(destination, k, decoded)


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


def _snapshot_for_overlapping_install(parent: Container, key: str, value: Any) -> Any:
    """Snapshot ``value`` if an overlapping install cannot safely read it.

    An ancestor source would grow while it is read. During overwrite, a
    headerless grandchild also depends on implicit intermediates that
    deletion resets. Other descendants retain independent slot anchors
    and use the trivia-preserving private-orphan adopt path.
    """
    if not isinstance(value, (Container, AoT)):
        return value
    root = parent._layout_root  # noqa: SLF001
    if root is None or value._layout_root is not root:  # noqa: SLF001
        return value
    dest_path = (*parent._path, key)  # noqa: SLF001
    value_path = value._path  # noqa: SLF001
    vlen, dlen = len(value_path), len(dest_path)
    ancestor_overlap = vlen < dlen and dest_path[:vlen] == value_path
    headerless_value = not isinstance(value, AoT) and value._header_ref is None  # noqa: SLF001
    descendant_overlap = (
        headerless_value and vlen - dlen >= 2 and value_path[:dlen] == dest_path
    )
    if not (ancestor_overlap or descendant_overlap):
        return value
    if isinstance(value, AoT):
        return AoT(value.to_list())
    return Document(data=value.to_dict())


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
    # `list` (or subclass) only — `tuple` is intentionally not accepted
    # (TOML has no tuple, and accepting it would mask user typos). Array,
    # a list subclass, was already accepted above.
    return isinstance(v, list)


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
        msg = "cannot store an array-of-tables inside an inline table"
        raise TOMLError(msg)
    if _is_section(v):
        msg = "cannot store a section-style table inside an inline-style table"
        raise TOMLError(msg)
    # Live-attach unattached inline values so user identity is preserved.
    # For an inline Table, `_populate_inline_table` fully re-wires state
    # (including a fresh `_value`), so no separate reset is needed.
    if (_is_inline_table(v) or isinstance(v, Array)) and not v._attached:  # noqa: SLF001
        if isinstance(v, Array):
            _retarget_to_doc(v._value, layout_root)  # noqa: SLF001
            _attach_inline_view(v, layout_root, owner)
            return v._value, v  # noqa: SLF001
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
        src_val = v._value  # noqa: SLF001
        if src_val is not None:
            from tomlrt._build import _decode_value  # noqa: PLC0415

            cloned = copy.deepcopy(src_val)
            _retarget_to_doc(cloned, layout_root)
            new = _decode_value(
                cloned, layout_root=layout_root, parent=parent, path=path, owner=owner
            )
            return cloned, new
        # A dotted-key navigator view (`_Kind.INLINE_DOTTED_INNER`) owns
        # no CST of its own — it's a live projection over an ancestor's
        # inline value. Arrays always own `_value`, so only a Table
        # falls through here, which the generic Mapping branch below
        # handles by synthesising fresh from its logical items.
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
    msg = f"cannot convert {type(v).__name__} to a TOML value"
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


def _attach_inline_view(
    value: object, layout_root: Document | None, owner: AoTEntry | None
) -> None:
    """Record document attachment on an Array or inline Table, recursively.

    Arrays have no `_owner_aot_entry` field; inline Tables do and
    inherit ``owner`` from the outermost host. An Array reattach reuses
    its preserved `ArrayValue` wholesale rather than resynthesising each
    item, so a nested inline-table item may still carry the `_value =
    None` that `_reset_inline_for_rehome` set on delete (on the
    assumption of a full rebuild on reattach, which this path skips) -
    re-link it from the corresponding array item before recursing.
    """
    if isinstance(value, Array):
        value._layout_root = layout_root  # noqa: SLF001
        for item, child in zip(value._value.items, value, strict=True):  # noqa: SLF001
            if _is_inline_table(child) and child._value is None:  # noqa: SLF001
                assert isinstance(item.value, InlineTableValue)
                child._value = item.value  # noqa: SLF001
            _attach_inline_view(child, layout_root, owner)
    elif _is_inline_table(value):
        value._layout_root = layout_root  # noqa: SLF001
        value._owner_aot_entry = owner  # noqa: SLF001
        for child in value.values():
            _attach_inline_view(child, layout_root, owner)


def _populate_inline_table(
    table: Container,
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
