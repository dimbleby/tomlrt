"""Build logical containers while parsing or replaying a trusted slot run.

Builds `Document`, `Table`, `Array`, and `AoT` views from a parsed
slot stream in doc-stream first-occurrence order. This is the one place
that derives implicit containers from slot paths.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tomlrt._array import AoT, Array
from tomlrt._comment_text import _split_preamble
from tomlrt._container import Table, _file_host, _reorder_dict_storage
from tomlrt._layout_ops import (
    _bind_aot,
    _clone_entry_slots,
    append_parsed_kv,
    append_parsed_ref,
    extract_subtree_slots,
    file_own_header,
    owned_slots,
    record_slot,
)
from tomlrt._slots import KVSlot, StructuralHeaderSlot, stitch_run
from tomlrt._validator import check_header, dotted_conflict, key_conflict
from tomlrt._values import (
    ArrayValue,
    EmptyAoTValue,
    InlineTableValue,
)

if TYPE_CHECKING:
    from tomlrt._container import Container, DecodedValue, Document
    from tomlrt._slots import AoTEntry, Slot
    from tomlrt._validator import ErrorBuilder
    from tomlrt._values import (
        Value,
    )


def _build_containers(root: Container, slots: list[Slot]) -> None:
    """Walk ``slots`` and populate ``root`` and its descendants.

    The run is already linked, and its paths and owners are known to be
    valid. Each KV's host is threaded through from the most recent header.

    Structural slots introduce descendants of ``root``; its own header,
    if any, is already filed.
    """
    current_host = root
    for slot in slots:
        if isinstance(slot, StructuralHeaderSlot):
            current_host = _apply_header(root, slot)
        else:
            assert isinstance(slot, KVSlot)
            assert slot.host_path == current_host._path, (  # noqa: SLF001
                f"KV host_path {slot.host_path} != current section {current_host._path}"  # noqa: SLF001
            )
            _apply_kv(slot, current_host)


# ---------------------------------------------------------------------------
# Header handling
# ---------------------------------------------------------------------------


def _apply_header(
    root: Container,
    slot: StructuralHeaderSlot,
    error: ErrorBuilder | None = None,
    at: int = 0,
) -> Table:
    """Validate (when parsing) and open a table header or fresh AoT entry."""
    path = slot.key_path
    root_path = root._path  # noqa: SLF001
    assert path[: len(root_path)] == root_path
    file_ref = record_slot if error is None else append_parsed_ref
    parent = root
    file_ref(parent, slot)
    for depth, name in enumerate(path[len(root_path) : -1], len(root_path) + 1):
        child = parent.get(name)
        if child is None:
            child = _make_table(
                parent,
                path[:depth],
                owner=parent._owner_aot_entry,  # noqa: SLF001
            )
            dict.__setitem__(parent, name, child)
        elif isinstance(child, AoT):
            child = child[-1]
        elif not isinstance(child, Table) or child._inline:  # noqa: SLF001
            assert error is not None
            joined = ".".join(path[:depth])
            msg = f"cannot use {joined!r} as a table: already defined as a value"
            raise error(msg, at=at)
        parent = child
        file_ref(parent, slot)

    entry = slot.entry
    name = path[-1]
    existing = parent.get(name)
    if error is not None:
        if existing is not None and not (
            entry is not None and isinstance(existing, AoT)
        ):
            check_header(existing, path, entry, error, at=at)
        slot.owner_aot_entry = entry if entry is not None else parent._owner_aot_entry  # noqa: SLF001
    if entry is None:
        if existing is None:
            table = _make_table(parent, path, owner=slot.owner_aot_entry)
            dict.__setitem__(parent, name, table)
        else:
            assert isinstance(existing, Table)
            table = existing
    else:
        if existing is None:
            existing = AoT()
            _bind_aot(parent, name, existing)
            dict.__setitem__(parent, name, existing)
        assert isinstance(existing, AoT)
        table = _make_table(existing, path, owner=entry)
        list.append(existing, table)
    file_own_header(table, slot, parsed=error is not None)
    return table


def _resolve_inline_child(parent: Table, name: str, owner: AoTEntry | None) -> Table:
    sub = parent.get(name)
    if sub is None:
        child = _make_table(
            parent,
            (*parent._path, name),  # noqa: SLF001
            owner=owner,
        )
        child._inline = True  # noqa: SLF001
        dict.__setitem__(parent, name, child)
        return child
    assert isinstance(sub, Table)
    assert sub._inline  # noqa: SLF001
    return sub


def _make_table(
    parent: Container | AoT, path: tuple[str, ...], *, owner: AoTEntry | None
) -> Table:
    table = Table()
    table._wire(  # noqa: SLF001
        layout_root=parent._layout_root,  # noqa: SLF001
        parent=parent,
        path=path,
        owner=owner,
    )
    return table


# ---------------------------------------------------------------------------
# KV slot handling
# ---------------------------------------------------------------------------


def _apply_kv(
    slot: KVSlot,
    host: Container,
    error: ErrorBuilder | None = None,
    at: int = 0,
) -> None:
    """Validate (when parsing) and bind a KV into its current host.

    Refs propagate **only** along the slot's logical path starting at
    the host container, not from the document root. A KV under ``[a]``
    contributes to ``a._refs``, not ``doc._refs``.

    ``host`` is the pre-resolved container for ``slot.host_path``,
    threaded from the most recent header; decoded value attachment
    cascades through ``host._layout_root``.
    """
    path = slot.key_path
    target = host
    if len(path) > 1:
        target = _walk_dotted_kv(slot, host, error, at)
    name = path[-1]
    if error is not None:
        if name in target:
            raise key_conflict(target[name], slot.host_path + path, error, at=at)
        append_parsed_kv(target, slot, name)
    else:
        assert name not in target
        record_slot(target, slot)
    dict.__setitem__(
        target,
        name,
        _decode_value(
            slot.value,
            target._layout_root,  # noqa: SLF001
            target,
            name,
            target._owner_aot_entry,  # noqa: SLF001
        ),
    )


def _walk_dotted_kv(
    slot: KVSlot, target: Container, error: ErrorBuilder | None, at: int
) -> Container:
    """Resolve and file the ancestors of a dotted KV's binding."""
    path = slot.key_path
    for depth, name in enumerate(path[:-1], 1):
        child = target.get(name)
        if child is None:
            child = _make_table(
                target,
                (*target._path, name),  # noqa: SLF001
                owner=slot.owner_aot_entry,
            )
            dict.__setitem__(target, name, child)
        elif error is not None:
            if isinstance(child, AoT) or (
                isinstance(child, Table) and child._header is not None  # noqa: SLF001
            ):
                raise dotted_conflict(child, slot.host_path, path, depth, error, at=at)
            if not isinstance(child, Table) or child._inline:  # noqa: SLF001
                joined = ".".join(slot.host_path + path[:depth])
                msg = f"key {joined!r} already defined as a value"
                raise error(msg, at=at)
        assert isinstance(child, Table)
        if error is None:
            record_slot(target, slot)
        else:
            append_parsed_kv(target, slot)
        target = child
    return target


# ---------------------------------------------------------------------------
# Value decoding
# ---------------------------------------------------------------------------


def _decode_value(
    value: Value,
    layout_root: Document | None,
    parent: Container | None,
    name: str | None,
    owner: AoTEntry | None,
    array_host: Array | None = None,
) -> DecodedValue:
    """Decode any TOML value to its Python representation.

    ``parent``/``name`` -- the container and key ``value`` is bound
    under -- give a key-hosted ``Array`` / ``Table`` view a real binding
    and name for its own navigation and host lookup. ``array_host`` is
    the array ``value`` is an element of, if any; the decoded view's
    binding is filed here in one place so a decode site cannot forget,
    and it derives its host from the array object. Pass ``parent=None``
    (with ``name=None``) for a value with no key binding -- either an
    array element or a wholly detached value.

    An `Array` takes its binding at construction rather than by a stamp
    afterwards: this runs once per element of every array in the
    document, so the funnel hands `Array._view` the two fields it would
    otherwise write twice. The decode helpers take their arguments
    positionally for the same reason -- a keyword argument costs more
    per call than a positional one, and this is the hottest path in the
    build.
    """
    if isinstance(value, ArrayValue):
        if isinstance(value, EmptyAoTValue):
            assert not value.items
            assert parent is not None
            assert not parent._inline  # noqa: SLF001
            assert name is not None
            assert array_host is None
            aot = AoT()
            _bind_aot(parent, name, aot)
            return aot
        return _decode_array(
            value,
            layout_root,
            name,
            owner,
            array_host if array_host is not None else parent,
        )
    if isinstance(value, InlineTableValue):
        if parent is None:
            path: tuple[str, ...] = ()
        else:
            assert name is not None, "name is required whenever parent is given"
            path = (*parent._path, name)  # noqa: SLF001
        table = _decode_inline_table(value, layout_root, parent, path, owner)
        _file_host(table, parent, array_host)
        return table
    return value.value


def _decode_array(
    value: ArrayValue,
    layout_root: Document | None,
    name: str | None,
    owner: AoTEntry | None,
    host: Array | Container | None,
) -> Array:
    arr = Array._view(value, layout_root, name, host)  # noqa: SLF001
    for item in value.items:
        list.append(
            arr,
            _decode_value(item.value, layout_root, None, None, owner, arr),
        )
    return arr


def _decode_inline_table(
    value: InlineTableValue,
    layout_root: Document | None,
    parent: Container | None,
    path: tuple[str, ...],
    owner: AoTEntry | None,
) -> Table:
    table = Table()
    table._wire(  # noqa: SLF001
        layout_root=layout_root, parent=parent, path=path, owner=owner
    )
    table._inline = True  # noqa: SLF001
    table._value = value  # noqa: SLF001
    for entry in value.items:
        decoded_key = entry.key_path
        cur = table
        for name in decoded_key[:-1]:
            cur = _resolve_inline_child(cur, name, owner)
        leaf = decoded_key[-1]
        assert leaf not in cur, (
            f"duplicate inline-table key {leaf!r} reached builder; validator drift"
        )
        dict.__setitem__(
            cur,
            leaf,
            _decode_value(entry.value, layout_root, cur, leaf, owner),
        )
    return table


def populate_extracted_document(doc: Document, src_table: Container) -> None:
    """Wire ``doc`` around a clone of ``src_table``'s subtree, re-rooted.

    Line ending and section-spacing convention are inherited from the
    source document; its file-level preamble and epilogue are not.
    """
    slots, promoted = extract_subtree_slots(src_table)
    root = src_table._layout_root  # noqa: SLF001
    assert root is not None
    if slots:
        slots[0].leading = promoted + slots[0].leading
    _assemble_document(
        doc,
        slots,
        trailing="" if slots else promoted,
        newline=root._newline,  # noqa: SLF001
        section_blank_separated=root._section_blank_separated,  # noqa: SLF001
    )


def populate_cloned_document(doc: Document, source: Document) -> None:
    """Clone a whole document's CST and its already-separated file envelope."""
    slots, _head = _clone_entry_slots(
        owned_slots(source),
        new_entry=None,
        body_owner=None,
        src_prefix=(),
        target_prefix=(),
        dst_newline=None,
    )
    stitch_run(None, slots, None)
    _assemble_document(
        doc,
        slots,
        trailing=source._trailing,  # noqa: SLF001
        newline=source._newline,  # noqa: SLF001
        section_blank_separated=source._section_blank_separated,  # noqa: SLF001
        preamble=source._preamble,  # noqa: SLF001
    )
    keys = list(source)
    if list(doc) != keys:
        _reorder_dict_storage(doc, keys)


def _assemble_document(
    doc: Document,
    slots: list[Slot],
    *,
    trailing: str,
    newline: str,
    section_blank_separated: bool,
    preamble: str | None = None,
) -> Document:
    """Wire ``doc`` around a linked slot run and build its views."""
    _finish_document(
        doc,
        slots,
        trailing=trailing,
        newline=newline,
        section_blank_separated=section_blank_separated,
        preamble=preamble,
    )
    _build_containers(doc, slots)
    return doc


def _finish_document(
    doc: Document,
    slots: list[Slot],
    *,
    trailing: str,
    newline: str,
    section_blank_separated: bool,
    preamble: str | None = None,
) -> None:
    """Set the document's endpoints and file-level trivia."""
    doc._head = slots[0] if slots else None  # noqa: SLF001
    doc._tail = slots[-1] if slots else None  # noqa: SLF001
    doc._trailing = trailing  # noqa: SLF001
    doc._newline = newline  # noqa: SLF001
    doc._section_blank_separated = section_blank_separated  # noqa: SLF001
    if preamble is not None:
        doc._preamble = preamble  # noqa: SLF001
    elif slots:
        # The opening comment paragraph is the document preamble; the rest of
        # the head slot's leading stays as the first construct's block.
        head = slots[0]
        preamble, rest = _split_preamble(head.leading)
        if preamble:
            doc._preamble = preamble  # noqa: SLF001
            head.leading = rest
    else:
        # Comment-only source: the parser put everything onto
        # ``trailing``; that's the preamble, not the epilogue.
        doc._preamble = trailing  # noqa: SLF001
        doc._trailing = ""  # noqa: SLF001


__all__ = [
    "populate_cloned_document",
    "populate_extracted_document",
]
