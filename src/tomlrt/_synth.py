"""Build a document from Python data by synthesising its slots.

The layout of a document built from a mapping is fixed the moment the
mapping is known, so this writes it out directly rather than editing it
into place one key at a time. `_build` then turns the slots into views,
exactly as it does for a parse, and both constructors share one linear
builder.

Constructing copies: a `Table` / `Array` / `AoT` contributes its
contents and its shape, not itself. One holding a block of source
layout -- a section or array-of-tables still in a document, or popped
out of one -- cannot be rebuilt from its data without losing the
comments and spacing that live in its slots, so its block is cloned
and written out with the rest.

Two passes over the mapping, because the two orders differ. `_plan`
walks it in its own order, so the first thing wrong with it is the
first thing reported. `_emit` then writes the slots in document order,
where a section's own keys precede its subsections.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from tomlrt._array import AoT, Array
from tomlrt._build import _assemble_document
from tomlrt._container import (
    DEFAULT_NEWLINE,
    Container,
    Document,
    Table,
    _has_extractable_layout,
    _is_inline_table,
    _is_section,
    _reorder_dict_storage,
    _unrepresentable_message,
    _validate_input,
)
from tomlrt._errors import TOMLError
from tomlrt._kind import _Kind
from tomlrt._layout_ops import _retarget_separator, clone_graft_slots
from tomlrt._render import render_run
from tomlrt._scalar import coerce_scalar, is_scalar
from tomlrt._slots import (
    AoTEntry,
    KVSlot,
    StructuralHeaderSlot,
    ensure_terminator,
    stitch_run,
)
from tomlrt._typecheck import _require_mapping, _validate_key
from tomlrt._values import (
    ArrayItem,
    ArrayValue,
    InlineTableEntry,
    InlineTableValue,
    make_keypart,
    make_keyparts,
    retarget_value_newlines,
)

if TYPE_CHECKING:
    from tomlrt._slots import Slot
    from tomlrt._values import Value


def _spells_own_key(slot: Slot, depth: int) -> bool:
    """Whether ``slot`` is a dotted key of the section at ``depth``.

    A header-less section has no line of its own: its keys are written
    as ``a.b = 1`` in the body of whichever section hosts them, one
    step above. Everything else in its block is hosted at its depth or
    below. True of a clone as well as a source, since cloning re-hosts
    exactly these keys and rebases every other path onto the target.
    """
    return isinstance(slot, KVSlot) and len(slot.host_path) < depth


def _graft_regions(v: object) -> tuple[bool, bool] | None:
    """Which regions ``v``'s block occupies, or ``None`` if it has none.

    A value has a block when it is still in a document, or popped out
    of one: its comments and spacing live in slots there, so only a
    clone can carry them. An inline value keeps all of that in the one
    `Value` it owns, which `_inline_value` copies.

    A header-less section is spelled by its descendants -- its own keys
    in the body above, each sub-section as a block -- so it can occupy
    either region or both. Anything else is all block.
    """
    if not isinstance(v, (Container, AoT)) or v._layout_root is None:  # noqa: SLF001
        return None
    if not isinstance(v, Container) or v._kind is not _Kind.IMPLICIT_SECTION:  # noqa: SLF001
        return False, True
    depth = len(v._path)  # noqa: SLF001
    body = blocks = False
    for ref in v._refs:  # noqa: SLF001
        if _spells_own_key(ref.slot, depth):
            body = True
        else:
            blocks = True
    assert body or blocks, "an implicit section is spelled by its own slots"
    return body, blocks


def _wants_section(v: object) -> bool:
    """Whether ``v`` becomes a ``[section]`` where a section may go.

    Any mapping does, except a `Table` that says it is inline.
    """
    # `dict` covers every `Table` and nearly every plain mapping; the
    # ABC is three times dearer to ask, and only an exotic one needs it.
    if not isinstance(v, dict):
        return isinstance(v, Mapping)
    return not _is_inline_table(v)


def _aot_entries(v: list[Any]) -> list[Mapping[str, object]] | None:
    """``v`` as the tables of an ``[[aot]]``, when that is what it is.

    An `AoT` says so itself, even an empty one -- which keeps its key
    as ``k = []``, the same placeholder the mutation layer uses. A bare
    list qualifies when it holds nothing but section-shaped mappings;
    an `Array` never does, having been asked for explicitly, and nor
    does the empty list, which is just an empty array value.
    """
    if isinstance(v, AoT):
        return list(v)
    if isinstance(v, Array) or not v:
        return None
    entries: list[Mapping[str, object]] = []
    for item in v:
        if not _wants_section(item):
            return None
        entries.append(item)
    return entries


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


class _Node:
    """One planned key: its spelling, and what to make of its value."""

    __slots__ = (
        "blocks",
        "body",
        "entries",
        "graft",
        "key",
        "raw",
        "table",
        "value",
    )

    def __init__(self, key: str, raw: object) -> None:
        self.key = key
        self.raw = raw
        self.table: _Plan | None = None
        self.entries: list[_Plan] | None = None
        self.value: Value | None = None
        self.graft = False
        # A graft's cloned slots, split into the two regions they are
        # written to; `_emit` fills both in.
        self.body: list[Slot] = []
        self.blocks: list[Slot] = []


class _Plan:
    """A section's planned contents: its own keys, then its subsections."""

    __slots__ = ("any_grafts", "grafts", "in_order", "keys", "structural", "values")

    def __init__(self) -> None:
        self.values: list[_Node] = []
        self.structural: list[_Node] = []
        self.grafts: list[_Node] = []
        self.keys: list[str] = []
        # Whether this section or anything under it has a graft.
        self.any_grafts = False
        # Whether the mapping's order is the one the slots will produce
        # anyway -- true unless a key follows a subsection, or a graft
        # is re-filed as it installs.
        self.in_order = True

    def add_value(self, node: _Node) -> None:
        """Record a key held by a KV of the section's own body."""
        if self.structural:
            # The body is written before any subsection, so this key
            # will not come back where the mapping put it.
            self.in_order = False
        self.values.append(node)

    def add_graft(self, node: _Node, regions: tuple[bool, bool]) -> None:
        """Record a key whose block is cloned in from another document.

        Its slots are written where the mapping asks for them, so it
        occupies the same two regions any other key does -- and a
        header-less section occupies both.
        """
        node.graft = True
        self.grafts.append(node)
        self.any_grafts = True
        body, blocks = regions
        if body:
            self.add_value(node)
        if blocks:
            self.structural.append(node)

    @property
    def needs_header(self) -> bool:
        """Whether this section is worth a ``[path]`` line of its own.

        One with keys must have somewhere to put them; one with nothing
        at all would otherwise leave no trace. A section that only holds
        subsections is spelled by their headers alone.
        """
        return bool(self.values) or not self.structural


def _plan(mapping: Mapping[str, object], nl: str) -> _Plan:
    """Check and classify ``mapping``, keeping its own order.

    Anything TOML cannot hold raises the error the caller should see;
    a view holding source layout is set aside for `_settle`.
    """
    if not isinstance(mapping, dict):
        # A `Mapping` is free to hand out the same key twice, or to
        # disagree with its own ``__iter__``. Take one reading of it, so
        # the plan's keys are the document's keys.
        mapping = dict(mapping.items())
    plan = _Plan()
    for raw_key, raw in mapping.items():
        key = _validate_key(raw_key)
        node = _Node(key, raw)
        plan.keys.append(key)
        # Dispatch on the value's shape, so each arm is asked only
        # what it alone can answer. A list is a list, whatever else it
        # may also claim to be.
        if isinstance(raw, list):
            _plan_list(plan, node, nl)
        elif _wants_section(raw):
            _plan_section(plan, node, nl)
        else:
            node.value = _inline_value(raw, nl, key=key)
            plan.add_value(node)
    return plan


def _plan_list(plan: _Plan, node: _Node, nl: str) -> None:
    """Plan a list: an array-of-tables if that is what it holds."""
    raw = node.raw
    assert isinstance(raw, list)
    entries = _aot_entries(raw)
    if entries is None:
        node.value = _inline_value(raw, nl, key=node.key)
        plan.add_value(node)
    elif not entries:
        # No entries, so no headers: the key is held by an empty array,
        # which is what a mutation parks there too.
        node.value = ArrayValue()
        plan.add_value(node)
    elif (regions := _graft_regions(raw)) is not None:
        plan.add_graft(node, regions)
    else:
        node.entries = [_plan(entry, nl) for entry in entries]
        plan.any_grafts |= any(e.any_grafts for e in node.entries)
        plan.structural.append(node)


def _plan_section(plan: _Plan, node: _Node, nl: str) -> None:
    """Plan a mapping that becomes a ``[section]``."""
    raw = node.raw
    if (regions := _graft_regions(raw)) is not None:
        plan.add_graft(node, regions)
        return
    assert isinstance(raw, Mapping)
    node.table = _plan(raw, nl)
    plan.any_grafts |= node.table.any_grafts
    plan.structural.append(node)


def _inline_value(v: object, nl: str, *, key: str | None = None) -> Value:
    """Validate and build the TOML value for ``v``, laid out on one line.

    Mirrors the spacing `_fill_inline_array` and `_populate_inline_table`
    give a synthesised value: items separated by ``", "``, brackets
    padded only when there is something between them.
    """
    if is_scalar(v):
        return coerce_scalar(v)
    if isinstance(v, AoT):
        msg = "cannot store an array-of-tables inside an inline table"
        raise TOMLError(msg)
    if _is_section(v):
        msg = "cannot store a section-style table inside an inline-style table"
        raise TOMLError(msg)
    own = v._value if isinstance(v, (Array, Table)) else None  # noqa: SLF001
    if own is not None:
        # An `Array` or inline `Table` already holds the value it wants
        # written, including any shape it was given or parsed with; copy
        # that rather than rebuild it from the items alone.
        _validate_input(v, inline_only=True, key=key)
        cloned = copy.deepcopy(own)
        retarget_value_newlines(cloned, nl)
        return cloned
    if isinstance(v, list):
        array = ArrayValue()
        last = len(v) - 1
        for i, sub in enumerate(v):
            array.items.append(
                ArrayItem(
                    "" if i == 0 else " ", _inline_value(sub, nl), "", i != last, ""
                )
            )
        return array
    if isinstance(v, Mapping):
        # One reading of it, as `_plan` takes: a `Mapping` is free to
        # repeat a key or to disagree with its own ``__len__``, and the
        # separators are counted from what we take.
        if not isinstance(v, dict):
            v = dict(v.items())
        items = [(_validate_key(raw_key), sub) for raw_key, sub in v.items()]
        table = InlineTableValue()
        last = len(items) - 1
        for i, (child_key, sub) in enumerate(items):
            table.items.append(
                InlineTableEntry(
                    "" if i == 0 else " ",
                    _inline_value(sub, nl, key=child_key),
                    "",
                    i != last,
                    "",
                    (make_keypart(child_key),),
                    (),
                    " ",
                    " ",
                    (child_key,),
                )
            )
        if items:
            table.header_trivia = table._single_line_pad  # noqa: SLF001
            table.final_trivia = table._single_line_pad  # noqa: SLF001
        return table
    raise TypeError(_unrepresentable_message(v, key))


# ---------------------------------------------------------------------------
# Emit
# ---------------------------------------------------------------------------


def _emit(
    plan: _Plan,
    path: tuple[str, ...],
    owner: AoTEntry | None,
    out: list[Slot],
    nl: str,
    *,
    header: bool,
) -> None:
    """Write ``plan``'s slots, in document order, onto ``out``."""
    if header and plan.needs_header:
        out.append(_header_slot(path, "" if not out else nl, owner, None, nl))

    for node in plan.grafts:
        node.body, node.blocks = _graft_segments(node, path, owner, nl)

    for node in plan.values:
        if node.graft:
            out.extend(node.body)
            continue
        value = node.value
        assert value is not None, "a value node has no synthesised value"
        out.append(
            KVSlot(
                "",
                owner,
                nl,
                path,
                (make_keypart(node.key),),
                (),
                " ",
                " ",
                value,
            )
        )

    for node in plan.structural:
        if node.graft:
            # The clone brings its own comments; only the blank line
            # that positions it here is the destination's to say.
            _retarget_separator(node.blocks[0], "" if not out else nl)
            out.extend(node.blocks)
            continue
        sub = (*path, node.key)
        if node.table is not None:
            _emit(node.table, sub, owner, out, nl, header=True)
            continue
        assert node.entries, "a structural node is a table or a non-empty AoT"
        for entry in node.entries:
            entry_owner = AoTEntry()
            entry_header = _header_slot(
                sub, "" if not out else nl, entry_owner, entry_owner, nl
            )
            entry_owner.bind_header(entry_header)
            out.append(entry_header)
            _emit(entry, sub, entry_owner, out, nl, header=False)


def _header_slot(
    path: tuple[str, ...],
    leading: str,
    owner: AoTEntry | None,
    entry: AoTEntry | None,
    nl: str,
) -> StructuralHeaderSlot:
    """A synthesised ``[path]`` / ``[[path]]`` header line."""
    return StructuralHeaderSlot(
        leading,
        owner,
        nl,
        make_keyparts(path),
        (".",) * (len(path) - 1),
        "",
        "",
        entry,
        synthetic=True,
    )


def _graft_segments(
    node: _Node, path: tuple[str, ...], owner: AoTEntry | None, nl: str
) -> tuple[list[Slot], list[Slot]]:
    """``node``'s block, cloned under its key and split into its regions.

    A whole document has no header of its own, and takes one
    synthesised for the key it is bound to.
    """
    view = node.raw
    assert isinstance(view, (Container, AoT)), "only a view is grafted"
    target = (*path, node.key)
    cloned = clone_graft_slots(
        view, target_path=target, host_path=path, owner=owner, nl=nl
    )
    depth = len(target)
    body: list[Slot] = []
    blocks: list[Slot] = []
    for slot in cloned:
        (body if _spells_own_key(slot, depth) else blocks).append(slot)
    if isinstance(view, Document):
        blocks.insert(0, _header_slot(target, "", owner, None, nl))
    assert (bool(body), bool(blocks)) == _graft_regions(view), (
        "cloned regions disagree with the ones the plan filed the key under"
    )
    return body, blocks


def _reorder(container: Container, plan: _Plan) -> None:
    """Put ``container``'s keys, and its descendants', in mapping order.

    Slots are written body-first, so a key that follows a subsection in
    the mapping is built after it and dict storage comes out in neither
    the mapping's order nor the document's. `Document(mapping)` keeps
    the mapping's, as ``dict(mapping)`` does.

    A graft's own children came from a clone rather than from a plan,
    so there is nothing below it to reorder.
    """
    if not plan.in_order:
        _reorder_dict_storage(container, plan.keys)
    for node in plan.structural:
        if node.graft:
            continue
        child = dict.__getitem__(container, node.key)
        if node.table is not None:
            assert isinstance(child, Container)
            _reorder(child, node.table)
            continue
        assert node.entries, "a structural node is a table or a non-empty AoT"
        assert isinstance(child, AoT)
        for entry_plan, entry_table in zip(node.entries, child, strict=True):
            _reorder(entry_table, entry_plan)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _slot_run(data: Mapping[str, object], nl: str) -> tuple[_Plan, list[Slot]]:
    """Check ``data`` and write it out as a linked run of slots.

    Everything a document built from a mapping physically is. What is
    made of it afterwards -- views, or just text -- is the caller's.
    """
    _require_mapping(data, label="Document data argument")
    plan = _plan(data, nl)
    slots: list[Slot] = []
    _emit(plan, (), None, slots, nl, header=False)
    if plan.any_grafts:
        # A synthesised slot always ends its line, but a cloned one
        # taken from the end of its source file need not, and anything
        # written after it would run into it.
        for slot in slots[:-1]:
            ensure_terminator(slot, nl)
    stitch_run(None, slots, None)
    return plan, slots


def populate(doc: Document, data: Mapping[str, object]) -> None:
    """Populate ``doc`` from ``data``."""
    nl = doc._newline  # noqa: SLF001
    plan, slots = _slot_run(data, nl)
    _assemble_document(
        doc,
        slots,
        trailing="",
        newline=nl,
        prelude="",
        section_blank_separated=doc._section_blank_separated,  # noqa: SLF001
    )
    _reorder(doc, plan)


def render_mapping(data: Mapping[str, object]) -> str:
    """The text `Document` would render ``data`` as, without its views.

    Asked for text, `dumps` needs the slots and nothing built on top of
    them: no `Table` / `Array` / `AoT`, no refs, no dict storage. The
    run comes from the same `_slot_run` a `Document` is built from, so
    there is one synthesiser and one rendering walk, not two of either.

    A `Table` that owns section layout is the exception. `Document`
    clones and re-roots its slots rather than rebuilding them from its
    data, which keeps the comments and spacing they carry, so that one
    is built and rendered.
    """
    if _has_extractable_layout(data):
        return Document(data).render()
    _unused, slots = _slot_run(data, DEFAULT_NEWLINE)
    # The preamble split `_assemble_document` performs is byte-neutral:
    # it only decides which side of the join the opening comments are
    # rendered from.
    return render_run("", "", slots[0] if slots else None, "")


__all__ = ["populate", "render_mapping"]
