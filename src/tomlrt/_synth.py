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

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from tomlrt._array import AoT, Array
from tomlrt._build import _assemble_document
from tomlrt._container import (
    DEFAULT_NEWLINE,
    Container,
    Document,
    Table,
    _detached_inline_value,
    _has_extractable_layout,
    _is_inline_table,
    _is_section,
    _reorder_dict_storage,
    _unrepresentable_message,
    _validate_input,
)
from tomlrt._errors import TOMLError
from tomlrt._kind import _Kind
from tomlrt._layout_ops import (
    _retarget_separator,
    _spells_own_key,
    clone_aot_entry_layout,
    clone_graft_slots,
    split_subtree_slots,
)
from tomlrt._render import render_run
from tomlrt._scalar import coerce_scalar, is_scalar
from tomlrt._slots import (
    AoTEntry,
    KVSlot,
    StructuralHeaderSlot,
    ensure_terminator,
    stitch_run,
)
from tomlrt._typecheck import _mapping_items, _require_mapping, _validate_key
from tomlrt._values import (
    ArrayItem,
    ArrayValue,
    EmptyAoTValue,
    InlineTableEntry,
    InlineTableValue,
    is_shareable_scalar,
    make_keyparts,
    retarget_value_newlines,
)

if TYPE_CHECKING:
    from tomlrt._slots import Slot
    from tomlrt._values import Value

_KeyT = TypeVar("_KeyT")


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
    for slot in v._refs:  # noqa: SLF001
        if _spells_own_key(slot, depth):
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


def _aot_entries(v: list[object]) -> list[Mapping[Any, object]] | None:
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
    entries: list[Mapping[Any, object]] = []
    for item in v:
        if not _wants_section(item):
            return None
        assert isinstance(item, Mapping)
        entries.append(item)
    return entries


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


class _Graft:
    """A source-layout copy deferred until input validation has completed."""

    __slots__ = ("blocks", "body", "key", "source")

    def __init__(self, key: str, source: Container | AoT) -> None:
        self.key = key
        self.source = source
        self.body: list[Slot] = []
        self.blocks: list[Slot] = []


class _AoTPlan:
    """The planned or source-layout entries of one array-of-tables."""

    __slots__ = ("entries", "path")

    def __init__(self, path: tuple[str, ...], entries: list[_Plan | Container]) -> None:
        self.path = path
        self.entries = entries

    @property
    def key(self) -> str:
        return self.path[-1]


class _Plan:
    """A section's prepared slots, then its subsections.

    Own KV slots are private and unlinked; emission completes their key
    spelling after every input has been validated.
    """

    __slots__ = (
        "any_grafts",
        "grafts",
        "in_order",
        "keys",
        "owner",
        "path",
        "structural",
        "values",
    )

    def __init__(self, path: tuple[str, ...], owner: AoTEntry | None) -> None:
        self.path = path
        self.owner = owner
        self.values: list[KVSlot | _Graft] = []
        self.structural: list[_Plan | _AoTPlan | _Graft] = []
        self.grafts: list[_Graft] = []
        self.keys: list[str] = []
        # Whether this section or anything under it has a graft.
        self.any_grafts = False
        # Whether the mapping's order is the one the slots will produce
        # anyway -- true unless a key follows a subsection, or a graft
        # is re-filed as it installs.
        self.in_order = True

    @property
    def key(self) -> str:
        return self.path[-1]

    def add_value(self, node: KVSlot | _Graft) -> None:
        """Record a key held by a KV of the section's own body."""
        if self.structural:
            # The body is written before any subsection, so this key
            # will not come back where the mapping put it.
            self.in_order = False
        self.values.append(node)

    def add_graft(
        self, key: str, source: Container | AoT, regions: tuple[bool, bool]
    ) -> None:
        """Record a key whose block is cloned in from another document.

        Its slots are written where the mapping asks for them, so it
        occupies the same two regions any other key does -- and a
        header-less section occupies both.
        """
        node = _Graft(key, source)
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


def _plan(
    mapping: Mapping[_KeyT, object],
    nl: str,
    scalar_memo: dict[int, object] | None,
    path: tuple[str, ...],
    owner: AoTEntry | None,
) -> _Plan:
    """Check and classify ``mapping``, keeping its own order.

    Anything TOML cannot hold raises the error the caller should see;
    a view holding source layout is set aside for `_emit`.
    """
    plan = _Plan(path, owner)
    for raw_key, raw in _mapping_items(mapping):
        key = _validate_key(raw_key)
        plan.keys.append(key)
        # Dispatch on the value's shape, so each arm is asked only
        # what it alone can answer. A list is a list, whatever else it
        # may also claim to be.
        if isinstance(raw, list):
            entries = _aot_entries(raw)
            if entries is None:
                value = _inline_value(raw, nl, scalar_memo, key=key)
            elif not entries:
                value = EmptyAoTValue()
            elif (regions := _graft_regions(raw)) is not None:
                assert isinstance(raw, AoT)
                plan.add_graft(key, raw, regions)
                continue
            else:
                entry_path = (*path, key)
                entry_plans: list[_Plan | Container] = [
                    entry
                    if _is_section(entry) and entry._layout_root is not None  # noqa: SLF001
                    else _plan(entry, nl, scalar_memo, entry_path, AoTEntry())
                    for entry in entries
                ]
                plan.any_grafts |= any(
                    isinstance(entry, Container) or entry.any_grafts
                    for entry in entry_plans
                )
                plan.structural.append(_AoTPlan(entry_path, entry_plans))
                continue
        elif _wants_section(raw):
            if (regions := _graft_regions(raw)) is not None:
                assert isinstance(raw, Container)
                plan.add_graft(key, raw, regions)
            else:
                assert isinstance(raw, Mapping)
                child = _plan(raw, nl, scalar_memo, (*path, key), owner)
                plan.any_grafts |= child.any_grafts
                plan.structural.append(child)
            continue
        else:
            value = _inline_value(raw, nl, scalar_memo, key=key)
        # Key spelling stays in emission: it can invoke str-subclass hooks.
        plan.add_value(KVSlot("", owner, nl, path, (), (), (key,), " ", " ", value))
    return plan


def _inline_value(
    v: object,
    nl: str,
    scalar_memo: dict[int, object] | None,
    *,
    key: str | None = None,
) -> Value:
    """Validate and build the TOML value for ``v``, laid out on one line.

    Mirrors the spacing `_fill_inline_array` and `_populate_inline_table`
    give a synthesised value: items separated by ``", "``, brackets
    padded only when there is something between them.

    Construction supplies ``scalar_memo`` to isolate user payloads while
    retaining scalar aliases. Serialization supplies ``None``: rendering
    plain data must not invoke its scalar deepcopy hooks. Views already
    clone their own CST once per occurrence, independently of this memo.
    """
    if is_shareable_scalar(v):
        return coerce_scalar(v)
    if is_scalar(v):
        value = coerce_scalar(v)
        if scalar_memo is not None:
            value._copy_payloads(scalar_memo)  # noqa: SLF001
        return value
    if isinstance(v, AoT):
        msg = "cannot store an array-of-tables inside an inline table"
        raise TOMLError(msg)
    if _is_section(v):
        msg = "cannot store a section-style table inside an inline-style table"
        raise TOMLError(msg)
    own = _detached_inline_value(v) if isinstance(v, (Array, Table)) else None
    if own is not None:
        # An `Array` or inline `Table` already holds the value it wants
        # written, including any shape it was given or parsed with; copy
        # that rather than rebuild it from the items alone.
        _validate_input(v, inline_only=True, key=key)
        retarget_value_newlines(own, nl)
        return own
    if isinstance(v, list):
        array = ArrayValue()
        last = len(v) - 1
        for i, sub in enumerate(v):
            array.items.append(
                ArrayItem(
                    "" if i == 0 else " ",
                    _inline_value(sub, nl, scalar_memo),
                    "",
                    i != last,
                    "",
                )
            )
        return array
    if isinstance(v, Mapping):
        items = [(_validate_key(raw_key), sub) for raw_key, sub in _mapping_items(v)]
        table = InlineTableValue()
        last = len(items) - 1
        for i, (child_key, sub) in enumerate(items):
            key_path = (child_key,)
            table.items.append(
                InlineTableEntry(
                    "" if i == 0 else " ",
                    _inline_value(sub, nl, scalar_memo, key=child_key),
                    "",
                    i != last,
                    "",
                    make_keyparts(key_path),
                    (),
                    key_path,
                    " ",
                    " ",
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
    out: list[Slot],
    nl: str,
    *,
    header: bool,
) -> None:
    """Write ``plan``'s slots, in document order, onto ``out``."""
    path, owner = plan.path, plan.owner
    if header and plan.needs_header:
        out.append(_header_slot(path, "" if not out else nl, owner, None, nl))

    for graft in plan.grafts:
        graft.body, graft.blocks = _graft_segments(graft, path, owner, nl)

    for value in plan.values:
        if isinstance(value, _Graft):
            out.extend(value.body)
        else:
            assert not value.key_parts, "a planned key is spelled only once"
            value.key_parts = make_keyparts(value.key_path)
            out.append(value)

    for node in plan.structural:
        if isinstance(node, _Graft):
            # The clone brings its own comments; only the blank line
            # that positions it here is the destination's to say.
            _retarget_separator(node.blocks[0], "" if not out else nl)
            out.extend(node.blocks)
            continue
        if isinstance(node, _Plan):
            _emit(node, out, nl, header=True)
            continue
        sub = node.path
        for entry in node.entries:
            entry_owner: AoTEntry | None
            entry_header = None
            body = None
            if isinstance(entry, Container):
                entry_owner = AoTEntry()
                entry_header, body = clone_aot_entry_layout(
                    entry, path=sub, owner=entry_owner, nl=nl
                )
            else:
                entry_owner = entry.owner
                assert entry_owner is not None
            if entry_header is None:
                entry_header = _header_slot(sub, "", entry_owner, entry_owner, nl)
                entry_owner.bind_header(entry_header)
            _retarget_separator(entry_header, "" if not out else nl)
            out.append(entry_header)
            if body is None:
                assert isinstance(entry, _Plan)
                _emit(entry, out, nl, header=False)
            else:
                out.extend(body)


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
        path,
        "",
        "",
        entry,
        synthetic=True,
    )


def _graft_segments(
    node: _Graft, path: tuple[str, ...], owner: AoTEntry | None, nl: str
) -> tuple[list[Slot], list[Slot]]:
    """``node``'s block, cloned under its key and split into its regions.

    A whole document has no header of its own, and takes one
    synthesised for the key it is bound to.
    """
    view = node.source
    target = (*path, node.key)
    cloned = clone_graft_slots(
        view, target_path=target, host_path=path, owner=owner, nl=nl
    )
    body, blocks = split_subtree_slots(cloned, len(target))
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
        if isinstance(node, _Graft):
            continue
        child = dict.__getitem__(container, node.key)
        if isinstance(node, _Plan):
            assert isinstance(child, Container)
            _reorder(child, node)
            continue
        assert isinstance(child, AoT)
        for entry_plan, entry_table in zip(node.entries, child, strict=True):
            if isinstance(entry_plan, Container):
                _reorder_dict_storage(entry_table, list(entry_plan))
            else:
                _reorder(entry_table, entry_plan)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _slot_run(
    data: Mapping[str, object],
    nl: str,
    scalar_memo: dict[int, object] | None = None,
) -> tuple[_Plan, list[Slot]]:
    """Check ``data`` and write it out as a linked run of slots.

    Everything a document built from a mapping physically is. What is
    made of it afterwards -- views, or just text -- is the caller's.
    """
    _require_mapping(data, label="Document data argument")
    plan = _plan(data, nl, scalar_memo, (), None)
    slots: list[Slot] = []
    _emit(plan, slots, nl, header=False)
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
    plan, slots = _slot_run(data, nl, {})
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
