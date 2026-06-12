"""Initial logical-container build.

Builds `Document`, `Table`, `Array`, and `AoT` views from a parsed
slot stream in doc-stream first-occurrence order. This is the one place
that derives implicit containers from slot paths.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tomlrt._array import AoT, Array
from tomlrt._comments import _split_attached_block
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


def build_initial_containers(doc: Document, slots: list[Slot]) -> None:
    """Walk the slot stream and populate ``doc`` and its descendants.

    Threads the current header's host container through the loop so
    each KV skips a doc-root re-walk. The validator guarantees
    ``slot.host_path`` equals the most recent header path (or ``()``).
    """
    current_host: Container = doc
    for slot in slots:
        if isinstance(slot, StructuralHeaderSlot):
            current_host = _apply_header(doc, slot)
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


def _apply_header(doc: Document, slot: StructuralHeaderSlot) -> Table:
    if slot.kind == "aot-entry":
        assert slot.entry is not None
        return _open_aot_entry(doc, slot, slot.entry)
    return _open_table(doc, slot)


def _open_table(doc: Document, header: StructuralHeaderSlot) -> Table:
    """Open ``[a.b.c]`` — return the `Table` view for ``path``.

    Creates implicit ancestors as needed. A non-table intermediate is
    validator drift and raises.
    """
    path = header.path
    parent_chain = _resolve_chain(doc, path[:-1])
    parent = parent_chain[-1]
    name = path[-1]
    # Ancestor binding refs (chain[:i] -> child step path[i]).
    for ancestor in parent_chain:
        record_ref(ancestor, header)
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
    # Own-header ref + body-tail reset for this container.
    own_ref = record_ref(table, header)
    table._header_ref = own_ref  # noqa: SLF001
    table._body_tail = header  # noqa: SLF001
    return table


def _open_aot_entry(
    doc: Document,
    header: StructuralHeaderSlot,
    entry: AoTEntry,
) -> Table:
    """Open ``[[a.b]]`` — append a fresh `Table` to the AoT at ``path``."""
    path = header.path
    parent_chain = _resolve_chain(doc, path[:-1])
    parent = parent_chain[-1]
    name = path[-1]
    # Ancestor binding refs to the [[..]] header slot.
    for ancestor in parent_chain:
        record_ref(ancestor, header)
    aot = parent.get(name)
    if aot is None:
        aot = AoT()
        aot._layout_root = doc  # noqa: SLF001
        aot._path = path  # noqa: SLF001
        aot._parent = parent  # noqa: SLF001
        dict.__setitem__(parent, name, aot)
    assert isinstance(aot, AoT), (
        f"AoT header [[{'.'.join(path)}]] collides with non-AoT at "
        f"{name!r} (got {type(aot).__name__}); validator drift"
    )
    table = _make_table(parent, path, owner=entry)
    list.append(aot, table)
    own_ref = record_ref(table, header)
    table._header_ref = own_ref  # noqa: SLF001
    table._body_tail = header  # noqa: SLF001
    return table


def _resolve_chain(doc: Document, prefix: tuple[str, ...]) -> list[Container]:
    """Return the container chain ``[doc, doc.a, doc.a.b, ...]`` for prefix.

    Creates implicit containers as needed. For an AoT prefix, descends
    into the most recent entry. The returned list always has length
    ``len(prefix) + 1``.
    """
    chain: list[Container] = [doc]
    cur: Container = doc
    for i, name in enumerate(prefix):
        sub = cur.get(name)
        if sub is None:
            child_path = prefix[: i + 1]
            child = _make_table(cur, child_path, owner=cur._owner_aot_entry)  # noqa: SLF001
            dict.__setitem__(cur, name, child)
            cur = child
        elif isinstance(sub, Table):
            cur = sub
        elif isinstance(sub, AoT):
            assert sub, "validator should have rejected empty-AoT prefix"
            cur = sub[-1]
        else:
            msg = (
                f"path component {name!r} is bound to "
                f"{type(sub).__name__}, not a Table/AoT (validator drift)"
            )
            raise AssertionError(msg)
        chain.append(cur)
    return chain


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
    decoded = slot.key
    # Logical chain for dotted-KV intermediates: host -> ... -> leaf parent.
    leaf_chain: list[Container] = [host]
    cur = host
    for step in decoded[:-1]:
        sub = cur.get(step)
        if sub is None:
            child_path = (*cur._path, step)  # noqa: SLF001
            child = _make_table(cur, child_path, owner=slot.owner_aot_entry)
            dict.__setitem__(cur, step, child)
            cur = child
        else:
            assert isinstance(sub, Table), (
                f"dotted-key step {step!r} hits {type(sub).__name__} (validator drift)"
            )
            cur = sub
        leaf_chain.append(cur)
    target = leaf_chain[-1]
    name = decoded[-1]
    assert name not in target, (
        f"duplicate key {name!r} reached builder under {target._path}; "  # noqa: SLF001
        "validator drift"
    )
    for ancestor in leaf_chain:
        record_ref(ancestor, slot)
        maybe_advance_body_tail(ancestor, slot)
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
        cur: Container = table
        for step in decoded_key[:-1]:
            sub = cur.get(step)
            if sub is None:
                inner = Table()
                inner._wire(  # noqa: SLF001
                    layout_root=layout_root,
                    parent=cur,
                    path=(*cur._path, step),  # noqa: SLF001
                    owner=owner,
                )
                inner._inline = True  # noqa: SLF001
                # Dotted inline-table navigators have no backing value.
                dict.__setitem__(cur, step, inner)
                cur = inner
            else:
                assert isinstance(sub, Table)
                cur = sub
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
    doc._install_recorder = None  # noqa: SLF001
    doc._displaced_recorder = None  # noqa: SLF001
    doc._layout_root = doc  # noqa: SLF001
    if result.slots:
        # Move the head slot's above-blank prefix onto the document
        # preamble; leave only attached comments + indent on the slot.
        head = result.slots[0]
        above, attached, indent = _split_attached_block(head.leading)
        if above:
            doc._preamble = Trivia([p for line in above for p in line])  # noqa: SLF001
            head.leading = Trivia(
                [*(p for line in attached for p in line), *indent],
            )
    else:
        # Comment-only source: the parser put everything onto
        # ``trailing``; that's the preamble, not the epilogue.
        doc._preamble = result.trailing  # noqa: SLF001
        doc._trailing = Trivia()  # noqa: SLF001
    build_initial_containers(doc, result.slots)
    return doc


__all__ = ["build_from_parse", "build_initial_containers"]
