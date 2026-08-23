"""Build a document from Python data by synthesising its slots.

The layout of a document built from a mapping is fixed the moment the
mapping is known, so this writes it out directly rather than editing it
into place one key at a time. `_build` then turns the slots into views,
exactly as it does for a parse, and both constructors share one linear
builder.

Constructing copies: a `Table` / `Array` / `AoT` contributes its
contents and its shape, not itself. One holding a block of source
layout -- a section or array-of-tables still in a document, or popped
out of one -- keeps its place here with an empty header, and
`_settle` has the mutation layer clone it in afterwards.

Two passes over the mapping, because the two orders differ. `_plan`
walks it in its own order, so the first thing wrong with it is the
first thing reported. `_emit` then writes the slots in document order,
where a section's own keys precede its subsections.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import TYPE_CHECKING

from tomlrt._array import AoT, Array
from tomlrt._build import _assemble_document
from tomlrt._container import (
    Container,
    Document,
    Table,
    _is_inline_table,
    _is_section,
    _reorder_dict_storage,
    _sources_kept_intact,
    _validate_input,
)
from tomlrt._layout_ops import (
    first_block_slot,
    leading_comment_block,
    set_leading_comment_block,
)
from tomlrt._scalar import coerce_scalar, is_scalar
from tomlrt._slots import AoTEntry, KVSlot, StructuralHeaderSlot, stitch_run
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
    from collections.abc import Iterator

    from tomlrt._slots import Slot
    from tomlrt._values import Value


def _needs_cloning(v: object) -> bool:
    """Whether ``v`` is a block of source layout only a clone can carry.

    A section or array-of-tables still in a document, or popped out of
    one, spans slots whose comments and spacing live in the document it
    came from; rebuilding it from its data would drop them. An inline
    value keeps all of that in the one `Value` it owns, which
    `_inline_value` copies, and an entry-less array-of-tables spans no
    slots at all -- both rebuild exactly.
    """
    if not isinstance(v, (Container, AoT)) or v._layout_root is None:  # noqa: SLF001
        return False
    return (isinstance(v, AoT) and bool(v)) or _is_section(v)


def _wants_section(v: object) -> bool:
    """Whether ``v`` becomes a ``[section]`` where a section may go.

    Any mapping does, except a `Table` that says it is inline.
    """
    return isinstance(v, Mapping) and not _is_inline_table(v)


def _aot_entries(v: object) -> list[Mapping[str, object]] | None:
    """``v`` as the tables of an ``[[aot]]``, when that is what it is.

    An `AoT` says so itself, even an empty one -- which keeps its key
    as ``k = []``, the same placeholder the mutation layer uses. A bare
    list qualifies when it holds nothing but section-shaped mappings;
    an `Array` never does, having been asked for explicitly, and nor
    does the empty list, which is just an empty array value.
    """
    if isinstance(v, AoT):
        return list(v)
    if isinstance(v, Array) or not isinstance(v, list) or not v:
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

    __slots__ = ("carries_comments", "entries", "graft", "key", "raw", "table")

    def __init__(self, key: str, raw: object) -> None:
        self.key = key
        self.raw = raw
        self.table: _Plan | None = None
        self.entries: list[_Plan] | None = None
        self.graft = False
        # Whether the comments above the source's first slot have to be
        # carried across; only a block that replaces a placeholder does.
        self.carries_comments = False


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

    def add_graft(self, node: _Node) -> None:
        """Record a key whose block `_settle` has to clone in.

        One that arrives as dotted keys is body content and needs no
        placeholder; anything else keeps its place with an empty header
        that the install replaces, and so has to carry its comments.
        """
        node.table = _Plan()
        node.graft = True
        self.grafts.append(node)
        self.any_grafts = True
        # Installing the clone re-files the key.
        self.in_order = False
        if _installs_as_body(node.raw):
            self.values.append(node)
            return
        node.carries_comments = not isinstance(node.raw, Document)
        self.structural.append(node)

    @property
    def needs_header(self) -> bool:
        """Whether this section is worth a ``[path]`` line of its own.

        One with keys must have somewhere to put them; one with nothing
        at all would otherwise leave no trace. A section that only holds
        subsections is spelled by their headers alone.
        """
        return bool(self.values) or not self.structural


def _plan(mapping: Mapping[str, object]) -> _Plan:
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
        if _needs_cloning(raw):
            # Only a clone can carry this one's layout, so it keeps its
            # place with an empty header and the mutation layer fills it
            # in once the rest of the document is standing.
            plan.add_graft(node)
            continue
        if (entries := _aot_entries(raw)) is not None:
            if not entries:
                # No entries, so no headers: the key is held by an empty
                # array, which is what a mutation parks there too.
                plan.add_value(node)
                continue
            node.entries = [_plan(entry) for entry in entries]
            plan.any_grafts |= any(e.any_grafts for e in node.entries)
            plan.structural.append(node)
        elif _wants_section(raw):
            assert isinstance(raw, Mapping)
            node.table = _plan(raw)
            plan.any_grafts |= node.table.any_grafts
            plan.structural.append(node)
        else:
            _validate_input(raw, inline_only=True, key=key)
            plan.add_value(node)
    return plan


def _inline_value(v: object, nl: str) -> Value:
    """The TOML value for ``v``, laid out on one line.

    Mirrors the spacing `_fill_inline_array` and `_populate_inline_table`
    give a synthesised value: items separated by ``", "``, brackets
    padded only when there is something between them.
    """
    if is_scalar(v):
        return coerce_scalar(v)
    own = v._value if isinstance(v, (Array, Table)) else None  # noqa: SLF001
    if own is not None:
        # An `Array` or inline `Table` already holds the value it wants
        # written, including any shape it was given or parsed with; copy
        # that rather than rebuild it from the items alone.
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
    assert isinstance(v, Mapping), "the plan pass accepted only inline values"
    if not isinstance(v, dict):
        # One reading of it, as `_plan` takes: a `Mapping` is free to
        # repeat a key or to disagree with its own ``__len__``, and the
        # separators are counted from what we take.
        v = dict(v.items())
    table = InlineTableValue()
    last = len(v) - 1
    for i, (raw_key, sub) in enumerate(v.items()):
        key = _validate_key(raw_key)
        table.items.append(
            InlineTableEntry(
                "" if i == 0 else " ",
                _inline_value(sub, nl),
                "",
                i != last,
                "",
                (make_keypart(key),),
                (),
                " ",
                " ",
                (key,),
            )
        )
    if last >= 0:
        table.header_trivia = table._single_line_pad  # noqa: SLF001
        table.final_trivia = table._single_line_pad  # noqa: SLF001
    return table


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
        out.append(
            StructuralHeaderSlot(
                "" if not out else nl,
                owner,
                nl,
                make_keyparts(path),
                (".",) * (len(path) - 1),
                "",
                "",
                None,
                synthetic=True,
            )
        )

    out.extend(
        KVSlot(
            "",
            owner,
            nl,
            path,
            (make_keypart(node.key),),
            (),
            " ",
            " ",
            _inline_value(node.raw, nl),
        )
        for node in plan.values
        if not node.graft
    )

    for node in plan.structural:
        sub = (*path, node.key)
        if node.graft:
            assert node.table is not None, "a graft is planned with an empty table"
            _emit(node.table, sub, owner, out, nl, header=True)
            continue
        if node.table is not None:
            _emit(node.table, sub, owner, out, nl, header=True)
            continue
        assert node.entries, "a structural node is a table or a non-empty AoT"
        for entry in node.entries:
            entry_owner = AoTEntry()
            entry_header = StructuralHeaderSlot(
                "" if not out else nl,
                entry_owner,
                nl,
                make_keyparts(sub),
                (".",) * (len(sub) - 1),
                "",
                "",
                entry_owner,
                synthetic=True,
            )
            entry_owner.bind_header(entry_header)
            out.append(entry_header)
            _emit(entry, sub, entry_owner, out, nl, header=False)


def _installs_as_body(v: object) -> bool:
    """Whether a graft arrives as dotted KVs rather than as a header block.

    A section spelled only by its descendants' dotted keys is body
    content, and takes none of the blank line a header would. A whole
    `Document` is never that: it installs under a header synthesised
    for the key it is bound to, whatever its own first slot is.
    """
    return not isinstance(v, Document) and isinstance(
        first_block_slot(_as_view(v)), KVSlot
    )


def _as_view(v: object) -> Container | AoT:
    """``v`` narrowed to the view a graft always is."""
    assert isinstance(v, (Container, AoT)), "only a view is grafted"
    return v


def _settle(container: Container, plan: _Plan) -> None:
    """Finish ``container``: clone in its grafts, then order its keys.

    A graft replaces the empty header `_emit` left in its place, so the
    block lands where the mapping asked for it. Ordering comes after,
    because that install re-files the key; slots are written body-first
    too, so dict storage is in neither case the mapping's order, which
    is the one `Document(mapping)` keeps -- as ``dict(mapping)`` does.

    A graft's own children came from the clone, not from a plan, and
    its node holds an empty one; descending into it would reorder its
    container to nothing.
    """
    for node in plan.grafts:
        # A block that keeps its place replaces the header `_emit` left
        # there, and the install takes that slot's leading trivia -- so
        # the comments above it have to be carried across. One that
        # installs as dotted keys has no placeholder and keeps its own.
        block = (
            leading_comment_block(first_block_slot(_as_view(node.raw)))
            if node.carries_comments
            else ""
        )
        container._setitem_validated(node.key, node.raw)  # noqa: SLF001
        if node.carries_comments:
            installed = _as_view(dict.__getitem__(container, node.key))
            set_leading_comment_block(first_block_slot(installed), block)
    if not plan.in_order:
        _reorder_dict_storage(container, plan.keys)
    for node in plan.structural:
        if node.graft:
            continue
        child = dict.__getitem__(container, node.key)
        if node.table is not None:
            assert isinstance(child, Container)
            _settle(child, node.table)
            continue
        assert node.entries, "a structural node is a table or a non-empty AoT"
        assert isinstance(child, AoT)
        for entry_plan, entry_table in zip(node.entries, child, strict=True):
            _settle(entry_table, entry_plan)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def populate(doc: Document, data: Mapping[str, object]) -> None:
    """Populate ``doc`` from ``data``."""
    _require_mapping(data, label="Document data argument")
    plan = _plan(data)
    slots: list[Slot] = []
    _emit(plan, (), None, slots, doc._newline, header=False)  # noqa: SLF001
    stitch_run(None, slots, None)
    _assemble_document(
        doc,
        slots,
        trailing="",
        newline=doc._newline,  # noqa: SLF001
        prelude="",
        section_blank_separated=doc._section_blank_separated,  # noqa: SLF001
    )
    grafts = (node.raw for node in _all_grafts(plan)) if plan.any_grafts else ()
    with _sources_kept_intact(grafts):
        _settle(doc, plan)


def _all_grafts(plan: _Plan) -> Iterator[_Node]:
    """Every graft in ``plan``, at any depth."""
    yield from plan.grafts
    for node in plan.structural:
        if node.graft:
            continue
        if node.table is not None:
            yield from _all_grafts(node.table)
            continue
        assert node.entries, "a structural node is a table or a non-empty AoT"
        for entry in node.entries:
            yield from _all_grafts(entry)


__all__ = ["populate"]
