"""Initial logical-container build.

Builds `Document`, `Table`, `Array`, and `AoT` views from a parsed
slot stream in doc-stream first-occurrence order. This is the one place
that derives implicit containers from slot paths.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tomlrt._array import AoT, Array
from tomlrt._comments import _split_preamble
from tomlrt._container import Container, Document, Table
from tomlrt._layout_ops import maybe_advance_body_tail, record_ref
from tomlrt._slots import KVSlot, StructuralHeaderSlot
from tomlrt._trivia import Trivia
from tomlrt._values import (
    ArrayValue,
    InlineTableValue,
)

if TYPE_CHECKING:
    from tomlrt._parser import ParseResult
    from tomlrt._slots import AoTEntry, Slot
    from tomlrt._values import (
        Value,
    )


def _build_containers(root: Container, slots: list[Slot]) -> None:
    """Walk ``slots`` and populate ``root`` and its descendants.

    Threads the current header's host container through the loop so
    each KV skips a root re-walk. The validator guarantees
    ``slot.host_path`` equals the most recent header path (or the
    supplied root's path for a cloned subtree).

    A header whose path equals ``root``'s own can appear mid-body
    rather than as ``slots[0]`` (see :func:`tomlrt._layout_ops._owned_slots_ordered`
    for why); it cannot be reopened via ``_apply_header`` (``root`` is
    already wired, not a fresh child to create/find), so it is handled
    as a pure host-context reset back to ``root``.
    """
    current_host = root
    for slot in slots:
        if isinstance(slot, StructuralHeaderSlot):
            if slot.path == root._path:  # noqa: SLF001
                assert root._header_ref is None  # noqa: SLF001
                own_ref = record_ref(root, slot)
                root._header_ref = own_ref  # noqa: SLF001
                root._body_tail = slot  # noqa: SLF001
                current_host = root
            else:
                current_host = _apply_header(root, slot)
        else:
            assert isinstance(slot, KVSlot)
            assert slot.host_path == current_host._path, (  # noqa: SLF001
                f"KV host_path {slot.host_path} != current section "
                f"{current_host._path}; validator drift"  # noqa: SLF001
            )
            _apply_kv(slot, host=current_host)


# ---------------------------------------------------------------------------
# Cache primitives
# ---------------------------------------------------------------------------


# ``record_ref`` and ``maybe_advance_body_tail`` are shared with
# mutation paths, keeping cache maintenance in one place.


# ---------------------------------------------------------------------------
# Header handling
# ---------------------------------------------------------------------------


def _apply_header(root: Container, slot: StructuralHeaderSlot) -> Table:
    if slot.kind == "aot-entry":
        assert slot.entry is not None
        return _open_aot_entry(root, slot, slot.entry)
    return _open_table(root, slot)


def _resolve_parent(
    root: Container, path: tuple[str, ...], header: StructuralHeaderSlot
) -> tuple[Container, str]:
    """Resolve ``path``'s parent + final component, recording ancestor refs."""
    root_path = root._path  # noqa: SLF001
    assert path[: len(root_path)] == root_path
    parent = root
    record_ref(parent, header)
    for name in path[len(root_path) : -1]:
        parent = _resolve_table_child(parent, name, descend_aot=True)
        record_ref(parent, header)
    return parent, path[-1]


def _finish_opened_table(table: Table, header: StructuralHeaderSlot) -> Table:
    """Own-header ref + body-tail reset for a freshly opened ``table``."""
    own_ref = record_ref(table, header)
    table._header_ref = own_ref  # noqa: SLF001
    table._body_tail = header  # noqa: SLF001
    return table


def _open_table(root: Container, header: StructuralHeaderSlot) -> Table:
    """Open ``[a.b.c]`` — return the `Table` view for ``path``.

    Creates implicit ancestors as needed. A non-table intermediate is
    validator drift and raises.
    """
    path = header.path
    parent, name = _resolve_parent(root, path, header)
    existing = parent.get(name)
    if existing is None:
        table = _make_table(parent, path, owner=header.owner_aot_entry)
        dict.__setitem__(parent, name, table)
    else:
        assert isinstance(existing, Table), (
            f"header [{'.'.join(path)}] reopens a non-table at "
            f"{name!r} (got {type(existing).__name__}); validator drift"
        )
        table = existing
    return _finish_opened_table(table, header)


def _open_aot_entry(
    root: Container,
    header: StructuralHeaderSlot,
    entry: AoTEntry,
) -> Table:
    """Open ``[[a.b]]`` — append a fresh `Table` to the AoT at ``path``."""
    path = header.path
    parent, name = _resolve_parent(root, path, header)
    aot = parent.get(name)
    if aot is None:
        aot = AoT()
        aot._layout_root = root._layout_root  # noqa: SLF001
        aot._path = path  # noqa: SLF001
        aot._parent = parent  # noqa: SLF001
        dict.__setitem__(parent, name, aot)
    assert isinstance(aot, AoT), (
        f"AoT header [[{'.'.join(path)}]] collides with non-AoT at "
        f"{name!r} (got {type(aot).__name__}); validator drift"
    )
    table = _make_table(parent, path, owner=entry)
    list.append(aot, table)
    return _finish_opened_table(table, header)


def _resolve_table_child(
    parent: Container,
    name: str,
    *,
    owner: AoTEntry | None = None,
    inline: bool = False,
    descend_aot: bool = False,
) -> Table:
    """Resolve or create one table step beneath ``parent``.

    Header paths descend through the latest AoT entry and inherit its owner.
    Dotted KV and inline-table paths reject AoTs and use ``owner``.
    """
    sub = parent.get(name)
    if sub is None:
        child = _make_table(
            parent,
            (*parent._path, name),  # noqa: SLF001
            owner=parent._owner_aot_entry if descend_aot else owner,  # noqa: SLF001
        )
        child._inline = inline  # noqa: SLF001
        dict.__setitem__(parent, name, child)
        return child
    if isinstance(sub, Table):
        return sub
    if descend_aot and isinstance(sub, AoT):
        assert sub, "validator should have rejected empty-AoT prefix"
        return sub[-1]
    msg = (
        f"path component {name!r} is bound to "
        f"{type(sub).__name__}, not a table (validator drift)"
    )
    raise AssertionError(msg)


def _make_table(
    parent: Container, path: tuple[str, ...], *, owner: AoTEntry | None
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


def _apply_kv(slot: KVSlot, *, host: Container) -> None:
    """Bind a `key = value` slot into its host container.

    Refs propagate **only** along the slot's logical path starting at
    the host container, not from the document root. A KV under ``[a]``
    contributes to ``a._index["x"]``, not ``doc._index["a"]``.

    ``host`` is the pre-resolved container for ``slot.host_path``,
    threaded from the most recent header; decoded value attachment
    cascades through ``host._layout_root``.
    """
    parts = slot.key_parts
    target = host
    record_ref(target, slot)
    maybe_advance_body_tail(target, slot)
    for part in parts[:-1]:
        target = _resolve_table_child(
            target,
            part.value,
            owner=slot.owner_aot_entry,
        )
        record_ref(target, slot)
        maybe_advance_body_tail(target, slot)
    name = parts[-1].value
    assert name not in target, (
        f"duplicate key {name!r} reached builder under {target._path}; "  # noqa: SLF001
        "validator drift"
    )
    dict.__setitem__(
        target,
        name,
        _decode_value(
            slot.value,
            layout_root=target._layout_root,  # noqa: SLF001
            parent=target,
            path=(*target._path, name),  # noqa: SLF001
            owner=target._owner_aot_entry,  # noqa: SLF001
        ),
    )


# ---------------------------------------------------------------------------
# Value decoding
# ---------------------------------------------------------------------------


def _decode_value(
    value: Value,
    *,
    layout_root: Document | None,
    parent: Container | None,
    path: tuple[str, ...],
    owner: AoTEntry | None,
) -> object:
    """Decode any TOML value to its Python representation."""
    if isinstance(value, ArrayValue):
        return _decode_array(value, layout_root=layout_root, owner=owner)
    if isinstance(value, InlineTableValue):
        return _decode_inline_table(
            value,
            layout_root=layout_root,
            parent=parent,
            path=path,
            owner=owner,
        )
    return value.value


def _decode_array(
    value: ArrayValue,
    *,
    layout_root: Document | None,
    owner: AoTEntry | None,
) -> Array:
    arr = Array()
    arr._value = value  # noqa: SLF001
    arr._layout_root = layout_root  # noqa: SLF001
    for item in value.items:
        list.append(
            arr,
            _decode_value(
                item.value,
                layout_root=layout_root,
                parent=None,
                path=(),
                owner=owner,
            ),
        )
    return arr


def _decode_inline_table(
    value: InlineTableValue,
    *,
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
            cur = _resolve_table_child(cur, name, owner=owner, inline=True)
        leaf = decoded_key[-1]
        assert leaf not in cur, (
            f"duplicate inline-table key {leaf!r} reached builder; validator drift"
        )
        dict.__setitem__(
            cur,
            leaf,
            _decode_value(
                entry.value,
                layout_root=layout_root,
                parent=cur,
                path=(*cur._path, leaf),  # noqa: SLF001
                owner=owner,
            ),
        )
    return table


def build_from_parse(result: ParseResult) -> Document:
    """One-shot: parse-result → fully constructed `Document`."""
    doc = Document.__new__(Document)
    Container.__init__(doc)
    doc._head = result.slots[0] if result.slots else None  # noqa: SLF001
    doc._tail = result.slots[-1] if result.slots else None  # noqa: SLF001
    doc._trailing = result.trailing  # noqa: SLF001
    doc._preamble = Trivia()  # noqa: SLF001
    doc._newline = result.newline  # noqa: SLF001
    doc._prelude = result.prelude  # noqa: SLF001
    doc._is_private = False  # noqa: SLF001
    doc._install_recorders = None  # noqa: SLF001
    doc._section_blank_separated = result.section_blank_separated  # noqa: SLF001
    doc._layout_root = doc  # noqa: SLF001
    if result.slots:
        # The opening comment paragraph is the document preamble; the rest of
        # the head slot's leading stays as the first construct's block.
        head = result.slots[0]
        preamble, rest = _split_preamble(head.leading)
        if preamble:
            doc._preamble = Trivia(preamble)  # noqa: SLF001
            head.leading = Trivia(rest)
    else:
        # Comment-only source: the parser put everything onto
        # ``trailing``; that's the preamble, not the epilogue.
        doc._preamble = result.trailing  # noqa: SLF001
        doc._trailing = Trivia()  # noqa: SLF001
    _build_containers(doc, result.slots)
    return doc


__all__ = ["build_from_parse"]
