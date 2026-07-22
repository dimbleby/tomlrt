"""Mutate section-side layout.

This module owns linked-list and per-container cache updates for
direct KV insert and leaf delete. Inline-table mutation lives in
``_inline_ops.py``.

Design notes:

* The doc-stream linked list is the single source of physical ordering;
  inserts splice exactly one slot at an explicit anchor.
* ``c._refs`` mirrors the doc-stream subset referenced by ``c``. For a
  direct KV insert, the new ref belongs immediately after the anchor's
  ref (or at the front), not blindly at the tail where child-section
  refs may already sit.
* ``c._body_tail`` is incremental: O(1) on insert, O(len(c._refs)) only
  when deleting the current tail.
* A non-dotted direct KV files exactly one ref on its host container;
  ancestors are unaffected.
"""

from __future__ import annotations

import contextlib
import copy
import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

from tomlrt._comments import _split_attached_block
from tomlrt._kind import _Kind
from tomlrt._scalar import is_scalar
from tomlrt._slots import (
    AoTEntry,
    KVSlot,
    SlotRef,
    StructuralHeaderSlot,
    retarget_slot_newlines,
)
from tomlrt._trivia import (
    CommentNode,
    EolTrivia,
    NewlineNode,
    Trivia,
    WhitespaceNode,
    has_comment,
    leading_has_blank_line,
)
from tomlrt._values import (
    ArrayValue,
    InlineTableValue,
    make_keyparts,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence

    from tomlrt._array import AoT, Array
    from tomlrt._container import Container, Document, Table
    from tomlrt._slots import Slot
    from tomlrt._trivia import TriviaPiece
    from tomlrt._values import InlineTableEntry, KeyPart, Value


# ---------------------------------------------------------------------------
# Pure linked-list ops
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _record_install(
    doc: Document,
) -> Iterator[tuple[list[Slot], list[tuple[Slot, list[TriviaPiece], Slot | None]]]]:
    """Record slots installed and existing slots displaced by the transaction.

    The three insertion primitives (:func:`insert_after`,
    :func:`insert_before`, :func:`insert_before_head`) — the sole points
    at which a slot is linked into the document — append to the first
    yielded list. The second captures existing slots whose leading trivia
    was temporarily rewritten by synthetic-header insertion.

    The delete runs before this transaction and the reinstall never moves
    pre-existing slots, so the record distinguishes materialisation from
    movement. ``reposition_install`` uses both lists to move the installed
    block and restore surviving seams. Nested contexts stack; only the
    innermost is active.
    """
    prev = doc._install_recorders  # noqa: SLF001
    installed: list[Slot] = []
    displaced: list[tuple[Slot, list[TriviaPiece], Slot | None]] = []
    doc._install_recorders = (installed, displaced)  # noqa: SLF001
    try:
        yield installed, displaced
    finally:
        doc._install_recorders = prev  # noqa: SLF001


def _record_new_slot(doc: Document, slot: Slot) -> None:
    """Append ``slot`` to ``doc``'s active install recorder, if any."""
    recorder = doc._install_recorders  # noqa: SLF001
    if recorder is not None:
        recorder[0].append(slot)


def _slot_is_linked(slot: Slot, doc: Document) -> bool:
    """Return whether ``slot`` is currently in ``doc``'s linked list."""
    return (
        slot is doc._head  # noqa: SLF001
        or slot._prev is not None  # noqa: SLF001
        or slot._next is not None  # noqa: SLF001
    )


def _effective_header_path_before(anchor: Slot | None) -> tuple[str, ...] | None:
    """The path of the header governing a bare KV placed right after ``anchor``.

    Walks backward from ``anchor`` to the nearest preceding
    ``StructuralHeaderSlot`` — a bare KV's scope comes from whichever
    header most recently opened, not necessarily from ``anchor`` itself
    (``anchor`` may be a KV physically inside some other table's own
    body). Returns ``None`` for doc-root scope (no header precedes it).
    """
    cur = anchor
    while cur is not None:
        if isinstance(cur, StructuralHeaderSlot):
            return tuple(cur.path)
        cur = cur._prev  # noqa: SLF001
    return None


def reposition_install(parent: Container, key: str, value: Any) -> None:
    """Replace ``parent[key]`` while preserving its physical position.

    The binding is deleted, reinstalled via ``parent[key] = value``,
    captured with ``_record_install``, then moved back to the saved anchor.

    A surviving neighbour keeps its pre-op leading iff, after the move,
    it sits immediately after the slot that legitimately precedes it:
    the relocated block tail for the original successor, or the original
    predecessor ``R`` for a sibling temporarily displaced by synthetic
    header insertion. The expected predecessor is unique, so a slot that
    is both successor and displaced sibling is restored at most once.

    A header-less new binding (scalar / synth-inline) is left where
    ``_insert_new`` placed it when the captured anchor lies outside
    ``parent``'s body region — moving it there would silently
    re-parent it. (A new binding that brings its own header carries
    its scope with it and is always safe to reposition.)

    Precondition: ``key`` is currently bound under ``parent``.
    """
    primary_ref = _binding_primary_ref(parent, key)
    old_primary = primary_ref.slot
    saved_anchor_prev, successor_slot = _binding_run_neighbours(parent, key)
    saved_leading_pieces = list(old_primary.leading.pieces)
    successor_leading = (
        list(successor_slot.leading.pieces) if successor_slot is not None else None
    )
    # The header-less safety check reads the doc-stream around the
    # captured anchor, so evaluate it before ``del`` perturbs the
    # links. The header-bearing check is done after install, against
    # the actual installed slots.
    in_body = _anchor_in_parent_direct_body(parent, saved_anchor_prev)
    # If the binding being replaced was itself a dotted key (its primary
    # slot is a KVSlot, not a header), keep the dotted form when the new
    # value re-emits into an emptied implicit container — replacing
    # ``a.b.c = 1`` with a scalar should yield ``a.b = "str"``, not a new
    # ``[a]`` header. A binding whose primary was a header keeps a header.
    reinstall_as_dotted = isinstance(old_primary, KVSlot)
    delete_key(parent, key)
    doc = parent._attached_doc  # noqa: SLF001
    with _record_install(doc) as (new_slots, displaced):
        parent._insert_new(  # noqa: SLF001
            key,
            value,
            reinstall_as_dotted=reinstall_as_dotted,
        )
    # Header demotion during reinstall can invalidate the saved anchor.
    if saved_anchor_prev is not None and not _slot_is_linked(saved_anchor_prev, doc):
        return
    installed = _recorded_install_span(new_slots, doc)
    if installed is None:
        return
    if not _anchor_accepts_install(
        installed, saved_anchor_prev, in_parent_body=in_body, doc=doc
    ):
        return
    _move_slots_to_anchor(parent, installed, saved_anchor_prev, saved_leading_pieces)
    # Unified neighbour-leading restore (see docstring): restore each
    # perturbed neighbour's pre-op leading iff the move left it directly
    # after the predecessor that makes that leading correct.
    restores: list[tuple[Slot, list[TriviaPiece], Slot | None]] = list(displaced)
    if successor_slot is not None and successor_leading is not None:
        restores.append((successor_slot, successor_leading, installed[-1]))
    for slot, original, expected_pred in restores:
        if slot._prev is expected_pred:  # noqa: SLF001
            slot.leading.pieces = list(original)


def _recorded_install_span(recorded: list[Slot], doc: Document) -> list[Slot] | None:
    """Return the linked recorded slots in order, if they form one span.

    Slots unlinked again during the transaction are ignored. Implicit
    sources can install direct KVs and structural children into separate
    regions; those records return ``None`` and are not repositioned.
    """
    survivors = list(dict.fromkeys(s for s in recorded if _slot_is_linked(s, doc)))
    if not survivors:
        return None
    ids = {id(s) for s in survivors}
    heads = [
        s
        for s in survivors
        if s._prev is None or id(s._prev) not in ids  # noqa: SLF001
    ]
    if len(heads) != 1:
        return None
    ordered: list[Slot] = []
    cur: Slot | None = heads[0]
    while cur is not None and id(cur) in ids:
        ordered.append(cur)
        cur = cur._next  # noqa: SLF001
    assert len(ordered) == len(survivors), "linked slot span must be contiguous"
    return ordered


def _anchor_accepts_install(
    slots: list[Slot],
    anchor: Slot | None,
    *,
    in_parent_body: bool,
    doc: Document,
) -> bool:
    """Return whether moving ``slots`` after ``anchor`` preserves TOML scope."""
    if not any(isinstance(s, StructuralHeaderSlot) for s in slots):
        return in_parent_body

    installed_ids = {id(s) for s in slots}
    successor = anchor._next if anchor is not None else doc._head  # noqa: SLF001
    while successor is not None and id(successor) in installed_ids:
        successor = successor._next  # noqa: SLF001
    if isinstance(successor, KVSlot):
        return False

    first = slots[0]
    return not (
        isinstance(first, KVSlot)
        and _effective_header_path_before(anchor) != first.host_path
    )


def _ancestor_chain(c: Container | AoT) -> list[Container]:
    """Ancestors from ``c._parent`` up to (and including) the document root."""
    out: list[Container] = []
    cur = c._parent  # noqa: SLF001
    while cur is not None:
        out.append(cur)
        cur = cur._parent  # noqa: SLF001
    return out


def _resort_and_recompute_tails(c: Container, doc: Document) -> None:
    """Repair ancestor ref order and cached tails after moving slots."""
    position: dict[int, int] = {}
    cur = doc._head  # noqa: SLF001
    while cur is not None:
        position[id(cur)] = len(position)
        cur = cur._next  # noqa: SLF001

    for cn in [c, *_ancestor_chain(c)]:
        cn._refs.sort(key=lambda ref: position[id(ref.slot)])  # noqa: SLF001
        for refs in cn._index.values():  # noqa: SLF001
            refs.sort(key=lambda ref: position[id(ref.slot)])
        if cn._body_tail is not None:  # noqa: SLF001
            cn._body_tail = _recompute_body_tail(cn)  # noqa: SLF001


def _anchor_in_parent_direct_body(parent: Container, anchor_prev: Slot | None) -> bool:
    """True iff a direct KV spliced after ``anchor_prev`` would belong to ``parent``.

    For an implicit (header-less, non-root) container the binding is
    emitted as a dotted key hosted by the nearest header-bearing
    ancestor, so its scope is that host's. Every KV records that physical
    scope in ``host_path``; a header records it directly in ``path``.
    """
    host = _nearest_header_host(parent)
    host_header_ref = host._header_ref  # noqa: SLF001
    host_header = host_header_ref.slot if host_header_ref else None
    if anchor_prev is None:
        return host_header is None
    if isinstance(anchor_prev, StructuralHeaderSlot):
        return anchor_prev is host_header
    assert isinstance(anchor_prev, KVSlot)
    return (
        anchor_prev.host_path == host._path  # noqa: SLF001
        and anchor_prev.owner_aot_entry is host._owner_aot_entry  # noqa: SLF001
    )


def _file_ref_at_tail(c: Container, ref: SlotRef) -> None:
    """Append ``ref`` to ``c._refs`` and (when keyed) ``c._index``."""
    c._refs.append(ref)  # noqa: SLF001
    local_key = ref.local_key
    if local_key is not None:
        c._index.setdefault(local_key, []).append(ref)  # noqa: SLF001


def record_ref(c: Container, slot: Slot) -> SlotRef:
    """Create a `SlotRef(slot, c)` and file it at the tail of ``c``'s caches.

    The ``_index`` key is :attr:`SlotRef.local_key`, derived from
    ``(slot, container)`` geometry, so callers cannot file under a
    disagreeing key. Used by ``_build`` and section-clone installers.
    """
    ref = SlotRef(slot, c)
    _file_ref_at_tail(c, ref)
    return ref


def maybe_advance_body_tail(c: Container, slot: Slot) -> None:
    """Advance ``c._body_tail`` if ``slot`` is a body-region KV of ``c``.

    For a header-bearing (SECTION) container, the body is restricted to
    its own host_path; for a header-less container (document root or
    implicit table) any KV with a matching owner counts. Used by
    ``_build``'s initial population pass; mutation-time appends set
    ``_body_tail`` directly because they know they are appending a body
    slot by construction.
    """
    if (
        isinstance(slot, KVSlot)
        and slot.owner_aot_entry is c._owner_aot_entry  # noqa: SLF001
        and (c._kind is not _Kind.SECTION or slot.host_path == c._path)  # noqa: SLF001
    ):
        c._body_tail = slot  # noqa: SLF001


def _host_tail_predecessors(
    chain: list[Container],
    host: Container,
) -> list[Slot | None]:
    """Derive filed predecessors for a header appended after ``host``."""
    host_index = next((i for i, c in enumerate(chain) if c is host), None)
    if host_index is None:
        host_ancestors = _ancestor_chain(host)
        assert len(chain) <= len(host_ancestors)
        assert all(c is a for c, a in zip(chain, host_ancestors, strict=False))
        tail_count = 0
    else:
        tail_count = host_index + 1

    last = host._refs[-1].slot if host._refs else None  # noqa: SLF001
    header_ref = host._header_ref  # noqa: SLF001
    host_predecessor: Slot | None
    if isinstance(last, StructuralHeaderSlot):
        host_predecessor = last
    elif header_ref is not None:
        host_predecessor = header_ref.slot
    else:
        host_predecessor = None
    predecessor_container_ids = (
        {id(ref.container) for ref in host_predecessor._refs}  # noqa: SLF001
        if host_predecessor is not None
        else set()
    )

    predecessors: list[Slot | None] = []
    for i, c in enumerate(chain):
        if i < tail_count:
            refs = c._refs  # noqa: SLF001
            predecessors.append(refs[-1].slot if refs else None)
        else:
            predecessors.append(
                host_predecessor if id(c) in predecessor_container_ids else None
            )
    return predecessors


def _file_header_binding_chain(
    deepest: Container,
    header: StructuralHeaderSlot,
    *,
    host: Container | None = None,
) -> None:
    """File ``header`` in doc order on ``deepest`` and every ancestor.

    ``host`` identifies the subtree whose physical tail immediately
    precedes ``header``. Its cached projection supplies the predecessors
    without a doc-stream walk; arbitrary insertion positions omit it.
    """
    chain = [deepest, *_ancestor_chain(deepest)]
    predecessors = (
        _nearest_filed_predecessors(chain, header)
        if host is None
        else _host_tail_predecessors(chain, host)
    )
    for c, predecessor in zip(chain, predecessors, strict=True):
        _file_ordered_ref(c, header, predecessor=predecessor)


def _extend_header_bindings_to_root(
    parent: Container,
    slots: Iterable[Slot],
    *,
    host: Container | None = None,
) -> None:
    """Extend headers through ``parent``'s ancestors in physical order.

    ``host`` applies only to the first header; later headers are interior
    to the installed block and use their actual doc-stream predecessors.
    """
    for s in slots:
        if isinstance(s, StructuralHeaderSlot):
            _file_header_binding_chain(parent, s, host=host)
            host = None


def _file_synthetic_header_and_kv(
    c: Container,
    *,
    header_slot: StructuralHeaderSlot,
    key: str,
    value: Value,
    doc: Document,
    owner: AoTEntry | None,
    header_ref_index: int,
) -> KVSlot:
    """Common tail of the two header-synthesis paths.

    Files ``c``'s own-header ref, inserts ``key = value`` directly
    after ``header_slot``, files the KV ref, and updates
    ``c._header_ref`` / ``c._index[key]`` / ``c._body_tail``.

    Anchoring and ancestor binding-ref filing stay explicit in callers;
    both are highly position-sensitive and not safe to share.
    """
    own_header_ref = SlotRef(slot=header_slot, container=c)
    c._refs.insert(header_ref_index, own_header_ref)  # noqa: SLF001
    c._header_ref = own_header_ref  # noqa: SLF001

    new_kv = _new_kv_slot(
        host_path=c._path,  # noqa: SLF001
        key=(key,),
        value=value,
        doc=doc,
        owner=owner,
        leading=Trivia(),
    )
    insert_after(header_slot, new_kv, doc)
    kv_ref = SlotRef(slot=new_kv, container=c)
    c._refs.insert(header_ref_index + 1, kv_ref)  # noqa: SLF001
    c._index.setdefault(key, []).append(kv_ref)  # noqa: SLF001
    c._body_tail = new_kv  # noqa: SLF001
    return new_kv


def _wire_section_container(
    c: Container,
    *,
    doc: Document,
    path: tuple[str, ...],
    parent: Container,
    owner: AoTEntry | None,
) -> None:
    """Initialise a freshly-built section container's attachment fields."""
    c._wire(layout_root=doc, parent=parent, path=path, owner=owner)  # noqa: SLF001


def _file_own_header(c: Container, header: StructuralHeaderSlot) -> SlotRef:
    """File a freshly-wired section's own header."""
    ref = SlotRef(slot=header, container=c)
    c._refs.append(ref)  # noqa: SLF001
    c._header_ref = ref  # noqa: SLF001
    return ref


def _init_implicit_table(
    doc: Document,
    path: tuple[str, ...],
    parent: Container,
    owner: AoTEntry | None,
) -> Table:
    """Build an implicit (header-less) Table wired into ``doc`` at ``path``."""
    from tomlrt._container import Table  # noqa: PLC0415

    child = Table()
    child._wire(layout_root=doc, parent=parent, path=path, owner=owner)  # noqa: SLF001
    return child


def ensure_implicit_chain(
    parent: Container,
    sub_path: tuple[str, ...],
) -> Container:
    """Navigate or create implicit Tables along ``sub_path`` under ``parent``.

    Returns the deepest container. Missing components become new
    implicit (header-less) tables wired into ``parent._attached_doc``;
    existing components must already be ``Container`` instances.
    """
    from tomlrt._container import Container  # noqa: PLC0415

    doc = parent._attached_doc  # noqa: SLF001
    owner = parent._owner_aot_entry  # noqa: SLF001
    cur: Container = parent
    for j, comp in enumerate(sub_path):
        if comp in cur:
            nxt = dict.__getitem__(cur, comp)
            if not isinstance(nxt, Container):
                msg = f"intermediate {comp!r} is not a table"
                raise TypeError(msg)
            cur = nxt
            continue
        implicit = _init_implicit_table(
            doc,
            (*parent._path, *sub_path[: j + 1]),  # noqa: SLF001
            cur,
            owner,
        )
        dict.__setitem__(cur, comp, implicit)
        cur = implicit
    return cur


def _rebuild_index_for_key(c: Container, local_key: str) -> None:
    """Restore ``c._index[local_key]`` as the doc-stream subset of ``c._refs``.

    The invariant is that ``_index[k]`` equals the in-order list of refs
    in ``_refs`` whose ``local_key == k``. Rebuild after any mid-stream
    insertion under ``k`` rather than blindly appending — appending
    would be wrong when the new ref is followed by other contributors
    sharing the same key (e.g. a later ``[a.b]`` header).
    """
    c._index[local_key] = [  # noqa: SLF001
        r
        for r in c._refs  # noqa: SLF001
        if r.local_key == local_key
    ]


def _default_eol(doc: Document) -> EolTrivia:
    """A bare-newline `EolTrivia` for a freshly synthesised slot."""
    return EolTrivia(
        trailing_ws=None,
        comment=None,
        newline=NewlineNode(text=doc._newline),  # noqa: SLF001
    )


def _body_anchor(c: Container) -> Slot | None:
    """Return the end of ``c``'s direct body, including its header fallback."""
    if c._body_tail is not None:  # noqa: SLF001
        return c._body_tail  # noqa: SLF001
    header_ref = c._header_ref  # noqa: SLF001
    return header_ref.slot if header_ref is not None else None


def _insert_between(
    prev: Slot | None, new_slot: Slot, nxt: Slot | None, doc: Document
) -> None:
    """Splice ``new_slot`` between ``prev`` and ``nxt`` in ``doc``."""
    new_slot._prev = prev  # noqa: SLF001
    new_slot._next = nxt  # noqa: SLF001
    if prev is not None:
        prev._next = new_slot  # noqa: SLF001
    else:
        doc._head = new_slot  # noqa: SLF001
    if nxt is not None:
        nxt._prev = new_slot  # noqa: SLF001
    else:
        doc._tail = new_slot  # noqa: SLF001
    _record_new_slot(doc, new_slot)


def insert_after(anchor: Slot, new_slot: Slot, doc: Document) -> None:
    """Splice ``new_slot`` immediately after ``anchor`` in ``doc``."""
    _insert_between(anchor, new_slot, anchor._next, doc)  # noqa: SLF001


def insert_before(anchor: Slot, new_slot: Slot, doc: Document) -> None:
    """Splice ``new_slot`` immediately before ``anchor`` in ``doc``."""
    _insert_between(anchor._prev, new_slot, anchor, doc)  # noqa: SLF001


def insert_before_head(new_slot: Slot, doc: Document) -> None:
    """Splice ``new_slot`` at the start of ``doc``'s linked list.

    Purely mechanical; does not touch ``doc._trailing``. Callers
    inserting the very first slot into an empty doc that may carry
    preamble trivia in ``_trailing`` (e.g. set via
    :attr:`Document.preamble` or parsed from a comment-only source)
    should follow up with :func:`_promote_trailing_to_preamble`.
    """
    _insert_between(None, new_slot, doc._head, doc)  # noqa: SLF001


def _promote_trailing_to_preamble(doc: Document) -> None:
    """Ensure the doc preamble carries a blank-line separator before the head.

    Called on the empty-to-non-empty transition (first slot insert).
    The preamble itself lives on ``doc._preamble`` and is unchanged;
    we only need to ensure it ends with a blank-line gap (two NLs in a
    row) before the new first slot. Idempotent and no-op when preamble
    is empty.
    """
    pieces = doc._preamble.pieces  # noqa: SLF001
    if not pieces:
        return
    nl_count = 0
    for p in reversed(pieces):
        if isinstance(p, NewlineNode):
            nl_count += 1
        else:
            break
    while nl_count < 2:
        pieces.append(NewlineNode(doc._newline))  # noqa: SLF001
        nl_count += 1


def unlink_slot(
    slot: Slot, doc: Document, *, strip_new_head_leading: bool = True
) -> None:
    """Remove ``slot`` from ``doc``'s linked list.

    When ``strip_new_head_leading`` is True (default), if the unlink
    promotes a successor to be the new doc head, leading blank-line
    pieces on that successor are stripped — what was a separator from
    the removed first slot must not show up as a stray blank at the
    top of the file. Pass False for transient unlinks (e.g. AoT
    renormalise that re-splices the same slots) where the leading
    must be preserved.
    """
    p = slot._prev  # noqa: SLF001
    n = slot._next  # noqa: SLF001
    if p is not None:
        p._next = n  # noqa: SLF001
    else:
        doc._head = n  # noqa: SLF001
        if n is not None and strip_new_head_leading:
            _strip_leading_blank_lines(n)
    if n is not None:
        n._prev = p  # noqa: SLF001
    else:
        doc._tail = p  # noqa: SLF001
    slot._prev = None  # noqa: SLF001
    slot._next = None  # noqa: SLF001


def _strip_leading_blank_lines(slot: Slot) -> None:
    """Drop leading newline-only pieces from ``slot.leading``.

    Comment pieces are preserved (we don't want to silently drop user
    comments). Stops at the first non-newline piece.
    """
    pieces = slot.leading.pieces
    i = 0
    while i < len(pieces) and isinstance(pieces[i], NewlineNode):
        i += 1
    if i:
        del pieces[:i]


# ---------------------------------------------------------------------------
# Higher-level ops
# ---------------------------------------------------------------------------


def _splice_body_slot(
    new_slot: Slot,
    *,
    anchor_body_tail: Slot | None,
    anchor_header_ref: SlotRef | None,
    doc: Document,
) -> bool:
    """Splice ``new_slot`` into the doc-stream at the canonical body anchor.

    Anchor preference: body tail > header > head-of-doc seam > empty doc.
    Shared by the direct-KV and dotted-KV insert paths. Returns ``True``
    iff ``new_slot`` became the new doc head ahead of an existing head
    (the seam case), where ancestor refs must go at index 0.
    """
    if anchor_body_tail is not None:
        _ensure_terminator(anchor_body_tail, doc)
        insert_after(anchor_body_tail, new_slot, doc)
        return False
    if anchor_header_ref is not None:
        _ensure_terminator(anchor_header_ref.slot, doc)
        insert_after(anchor_header_ref.slot, new_slot, doc)
        return False
    if doc._head is not None:  # noqa: SLF001
        # Section-only doc: splice before the first slot, separating it.
        old_head = doc._head  # noqa: SLF001
        insert_before_head(new_slot, doc)
        _ensure_leading_blank_line(old_head, doc)
        return True
    # Empty doc: splice in as head, hoisting any preamble trivia.
    insert_before_head(new_slot, doc)
    _promote_trailing_to_preamble(doc)
    return False


def _project_bucket_index(
    refs: list[SlotRef], bucket: list[SlotRef], insert_idx: int
) -> int:
    """Position in ``bucket`` for a ref inserted into ``refs`` at ``insert_idx``.

    ``bucket`` is the same-key ordered subsequence of ``refs`` (a
    ``Container._index`` entry). The new ref belongs immediately after its
    nearest predecessor that is also in ``bucket``, or at the front if it has
    none. A fresh key (empty bucket), an append past the last ref, and — the
    sequential-append common case — an append directly after the last same-key
    ref all resolve in O(1); only an insert that lands *between* two same-key
    refs builds the position map and scans left for the predecessor.
    """
    if not bucket:
        return 0
    if insert_idx >= len(refs):
        return len(bucket)
    if insert_idx > 0 and refs[insert_idx - 1] is bucket[-1]:
        return len(bucket)
    pos = {r: i for i, r in enumerate(bucket)}
    for k in range(insert_idx - 1, -1, -1):
        i = pos.get(refs[k])
        if i is not None:
            return i + 1
    # No same-key predecessor: the new ref precedes every existing ref of its
    # key, so it heads the bucket. Reached e.g. when overwriting a dotted
    # sub-table whose key also heads a later sub-section header.
    return 0


def _nearest_filed_predecessors(
    ancestors: Sequence[Container], slot: Slot
) -> list[Slot | None]:
    """Find each ancestor's nearest filed predecessor in one backward walk."""
    remaining = {id(a) for a in ancestors}
    anchors: dict[int, Slot] = {}
    cur = slot._prev  # noqa: SLF001
    while cur is not None and remaining:
        for r in cur._refs:  # noqa: SLF001
            cid = id(r.container)
            if cid in remaining:
                anchors[cid] = cur
                remaining.discard(cid)
        cur = cur._prev  # noqa: SLF001
    return [anchors.get(id(a)) for a in ancestors]


def _file_ordered_ref(
    c: Container,
    slot: Slot,
    *,
    predecessor: Slot | None,
) -> SlotRef:
    """File ``slot`` on ``c`` immediately after its nearest filed predecessor.

    ``c._index[local_key]`` is updated as the same-key projection of
    ``c._refs``. With no predecessor the ref belongs at the front.
    """
    new_ref = SlotRef(slot=slot, container=c)
    local_key = new_ref.local_key
    assert local_key is not None
    refs = c._refs  # noqa: SLF001
    bucket = c._index.setdefault(local_key, [])  # noqa: SLF001
    if refs and refs[-1].slot is predecessor:
        # Common case: sequential tail append. Skip the index search
        # and bucket projection below.
        refs.append(new_ref)
        bucket.append(new_ref)
        return new_ref
    insert_idx = (
        _find_ref_index_by_slot(c, predecessor) + 1 if predecessor is not None else 0
    )
    bucket_idx = _project_bucket_index(refs, bucket, insert_idx)
    refs.insert(insert_idx, new_ref)
    bucket.insert(bucket_idx, new_ref)
    return new_ref


def append_direct_kv(
    c: Container,
    key: str,
    value: Value,
    *,
    reinstall_as_dotted: bool = False,
    key_parts: Sequence[KeyPart] | None = None,
    key_seps: Sequence[str] | None = None,
) -> None:
    """Append a fresh direct (non-dotted) KV to ``c``.

    Updates ``c._refs`` / ``_index`` / ``_body_tail`` and dict storage.
    Implicit headerless containers route through dotted-KV synthesis;
    AoT-entry sub-table bodies are not yet supported.
    """
    if c._kind is _Kind.IMPLICIT_SECTION:  # noqa: SLF001
        # Implicit / headerless non-root container. A fresh
        # ``host_path = c._path`` slot would render in whatever scope
        # the previous header (or the doc root) established, not in
        # ``c``'s logical scope — semantic mismatch. Insert via a
        # dotted KV under the nearest header-bearing ancestor instead.
        if c._body_tail is None and not reinstall_as_dotted:  # noqa: SLF001
            # ``c`` has no dotted body to anchor a dotted KV. Promote it
            # to an explicit ``[c]`` header: before its first descendant
            # header when it has one (``[a.b]`` ⇒ synthesise ``[a]``), or
            # as a fresh header when fully empty. The exception is a
            # structural overwrite that is replacing a dotted binding;
            # there the original form stays dotted rather than gaining
            # a header (see ``reposition_install``).
            _synthesise_header_then_insert_kv(c, key, value)
            return
        host = _nearest_header_host(c)
        install_dotted_kv_slot(
            host,
            (*c._path[len(host._path) :], key),  # noqa: SLF001
            value,
            leaf_parent=c,
        )
        return
    doc = c._attached_doc  # noqa: SLF001
    # Capture the anchor *before* mutating any cache.
    body_tail = c._body_tail  # noqa: SLF001
    header_ref = c._header_ref  # noqa: SLF001

    new_slot = _build_kv_slot(
        c,
        key,
        value,
        doc,
        key_parts=key_parts,
        key_seps=key_seps,
    )

    _splice_body_slot(
        new_slot,
        anchor_body_tail=body_tail,
        anchor_header_ref=header_ref,
        doc=doc,
    )
    anchor_slot: Slot | None = body_tail or (
        header_ref.slot if header_ref is not None else None
    )
    _file_ordered_ref(
        c,
        new_slot,
        predecessor=anchor_slot,
    )
    c._body_tail = new_slot  # noqa: SLF001
    _extend_entry_slots(c._owner_aot_entry, new_slot)  # noqa: SLF001


def _invalidate_body_tail_chain(
    start: Container | None,
    owned_slot_ids: set[int],
    *,
    min_depth: int = 0,
) -> None:
    """Recompute invalidated ``_body_tail`` values on the path to root.

    For each container ``cc`` along the chain whose existing
    ``_body_tail`` slot is in ``owned_slot_ids``, recompute the tail.

    Walks until either the chain is exhausted or
    ``len(cc._path) < min_depth``. The depth bound is a
    correctness short-circuit: an ancestor at depth ``d`` cannot
    have its body_tail point at a slot whose minimum bottom-depth
    exceeds ``d``. Common-case leaf-KV deletes never walk past
    ``c`` itself.
    """
    cur = start
    while cur is not None and len(cur._path) >= min_depth:  # noqa: SLF001
        if (
            cur._body_tail is not None  # noqa: SLF001
            and id(cur._body_tail) in owned_slot_ids  # noqa: SLF001
        ):
            cur._body_tail = _recompute_body_tail(cur)  # noqa: SLF001
        cur = cur._parent  # noqa: SLF001


def _nearest_header_host(c: Container) -> Container:
    """The closest ancestor (or ``c``) owning a header, else the doc root."""
    host = c
    while host._parent is not None and host._header_ref is None:  # noqa: SLF001
        host = host._parent  # noqa: SLF001
    return host


def _dotted_chain(host: Container, leaf: Container) -> list[Container]:
    """The container chain ``[host, ..., leaf]`` in doc-stream order."""
    chain: list[Container] = []
    cur: Container | None = leaf
    while cur is not host:
        assert cur is not None
        chain.append(cur)
        cur = cur._parent  # noqa: SLF001
    chain.append(host)
    chain.reverse()
    return chain


def _replace_primary_in_place(
    new_slot: KVSlot | StructuralHeaderSlot,
    primary: KVSlot | StructuralHeaderSlot,
    doc: Document,
) -> None:
    """Splice ``new_slot`` into the doc-stream where ``primary`` sits.

    The caller is materialising a replacement for an about-to-be-deleted
    binding whose doc-stream-first slot is ``primary``. ``new_slot`` takes
    ``primary``'s position — inheriting its leading and eol — and is
    inserted *before* it, so the later unlink of ``primary`` leaves
    ``new_slot`` exactly where ``primary`` was. Because ``new_slot`` is in
    place before the unlink, head-occupancy is preserved for free: if
    ``primary`` was the doc head, ``new_slot`` becomes the head and the
    unlink never strips the following separator.
    """
    new_slot.leading.pieces = list(primary.leading.pieces)
    new_slot.eol = primary.eol
    if primary._prev is None:  # noqa: SLF001
        insert_before_head(new_slot, doc)
    else:
        insert_after(primary._prev, new_slot, doc)  # noqa: SLF001


def _new_owned_section_header(
    c: Container, *, leading: Trivia, doc: Document
) -> StructuralHeaderSlot:
    return _new_section_header(
        c._path,  # noqa: SLF001
        leading=leading,
        doc=doc,
        owner_aot_entry=c._owner_aot_entry,  # noqa: SLF001
    )


def _extend_entry_slots(owner: AoTEntry | None, *slots: Slot) -> None:
    if owner is not None:
        owner.entry_slots.extend(slots)


def _transfer_stale_owner(
    slot: Slot, stale_owner: AoTEntry | None, new_owner: AoTEntry | None
) -> None:
    if stale_owner is None or slot.owner_aot_entry is not stale_owner:
        return
    slot.owner_aot_entry = new_owner
    _extend_entry_slots(new_owner, slot)
    if isinstance(slot, StructuralHeaderSlot) and slot.entry is stale_owner:
        slot.entry = None


def _materialise_empty_section_header(
    c: Container,
    primary: StructuralHeaderSlot,
    doc: Document,
) -> None:
    """Re-materialise a header for a now-empty header-origin section.

    The emptied section's physical presence was a descendant *header*
    (``a`` in ``[a.b]`` once ``b`` is removed). A ``[c._path]`` header
    replaces it in place: a header re-parents the KVs that follow it, but
    everything after ``primary`` up to the next header belonged to the
    deleted descendant, so nothing survives there to be wrongly
    re-parented. The empty section therefore keeps rendering — as ``[a]``
    — exactly where the descendant header was.
    """
    parent = c._parent  # noqa: SLF001
    assert parent is not None
    owner = c._owner_aot_entry  # noqa: SLF001
    header = _new_owned_section_header(c, leading=_build_section_leading(doc), doc=doc)
    own_ref = SlotRef(slot=header, container=c)
    c._refs.append(own_ref)  # noqa: SLF001
    c._header_ref = own_ref  # noqa: SLF001
    _replace_primary_in_place(header, primary, doc)
    _extend_entry_slots(owner, header)
    _file_header_binding_chain(parent, header)
    c._body_tail = header  # noqa: SLF001


def _materialise_empty_inline_table(
    c: Container,
    primary: KVSlot,
    doc: Document,
) -> None:
    """Re-materialise an empty inline table for a now-empty dotted section.

    The emptied section's physical presence was a descendant dotted *KV*
    (``a`` in ``a.b.x = 1`` once ``b`` is removed). Unlike a header, an
    inline-table binding re-parents nothing, so it can take ``primary``'s
    exact position even when sibling KVs survive around it — the section
    stays put and renders as ``a = {}`` (or, under an implicit ancestor,
    the dotted ``a.b = {}``). ``c`` flips from an implicit section to an
    inline-root table backed by the new (empty) ``InlineTableValue``.

    Must run *before* the scrub: the new binding's chain refs are filed
    immediately ahead of ``primary``'s own refs (which the scrub then
    removes), so they inherit ``primary``'s doc-stream position in each
    ancestor's ``_refs``.
    """
    parent = c._parent  # noqa: SLF001
    assert parent is not None
    owner = c._owner_aot_entry  # noqa: SLF001

    host = _nearest_header_host(c)
    key_path = c._path[len(host._path) :]  # noqa: SLF001

    val = InlineTableValue()
    kv = _new_kv_slot(
        host_path=host._path,  # noqa: SLF001
        key=key_path,
        value=val,
        doc=doc,
        owner=owner,
        leading=Trivia(),
    )
    _replace_primary_in_place(kv, primary, doc)

    # File the binding chain ``[host, ..., parent]``, slipping each new
    # ref in just before ``primary``'s ref so it lands at ``primary``'s
    # doc-stream position; the scrub removes ``primary``'s refs next.
    chain = _dotted_chain(host, parent)
    for i, anc in enumerate(chain):
        idx = _find_ref_index_by_slot(anc, primary)
        anc._refs.insert(idx, SlotRef(slot=kv, container=anc))  # noqa: SLF001
        _rebuild_index_for_key(anc, key_path[i])

    # ``c`` becomes an inline-root table, which keeps no ``_refs`` /
    # ``_index`` of its own (the binding lives on the parent chain, and
    # entries live in ``val.items``). Unfile its remaining
    # descendant-binding refs first — this also unregisters their slot
    # back-pointers, so the later scrub no longer reaches ``c``.
    for ref in list(c._refs):  # noqa: SLF001
        unfile_ref(ref)
    c._inline = True  # noqa: SLF001
    c._value = val  # noqa: SLF001
    c._body_tail = None  # noqa: SLF001

    _extend_entry_slots(owner, kv)


def delete_key(c: Container, key: str, *, materialise_empty: bool = False) -> None:
    """Delete ``key`` from ``c`` — scalar, inline, section, AoT, or dotted-subtree.

    Owned slots are scrubbed from live refs/indexes via slot
    back-pointers, body tails and ``AoTEntry.entry_slots`` are repaired,
    then the slots are unlinked. Cascade-prune is intentionally *not*
    performed: ``del c[k]`` removes exactly ``k`` and leaves any
    now-emptied implicit ancestor chain reachable as nested empty
    ``Table`` views.

    ``materialise_empty`` is opt-in for the public delete API: if the
    removal leaves ``c`` itself an empty, header-less section (its only
    physical presence was the deleted descendant, as for ``a`` in
    ``[a.b]`` once ``b`` goes), a synthetic ``[c._path]`` header is
    materialised so an empty-but-live table still renders. Internal
    delete-then-reinstall callers leave it ``False`` — the container is
    repopulated immediately, so a transient empty state must not grow a
    spurious header.

    Deleted structural views are transplanted to a private orphan document,
    preserving safe mutation and later reattachment without touching the live
    document.
    """
    val = dict.__getitem__(c, key)  # raises KeyError if absent
    doc = c._attached_doc  # noqa: SLF001

    # Re-materialisation bookkeeping for the public delete API. When the
    # removal empties ``c`` into a live, header-less, non-inline section,
    # that section must still render, so a ``[c._path]`` header is
    # synthesised by ``_materialise_empty_section_header`` (below, before
    # the unlink loop, while the descendant's primary slot is still in
    # place). ``len(c) == 1`` with ``key in c`` means ``c`` empties to
    # zero keys.
    will_materialise = (
        materialise_empty
        and bool(c._path)  # noqa: SLF001
        and not c._inline  # noqa: SLF001
        and c._header_ref is None  # noqa: SLF001
        and len(c) == 1
    )
    mat_primary: KVSlot | StructuralHeaderSlot | None = None
    if will_materialise:
        # ``_index[key]`` is scrubbed below; grab the removed descendant's
        # doc-stream-first slot now (its trivia / position is read at
        # materialise time, while it is still linked).
        primary = c._index[key][0].slot  # noqa: SLF001
        assert isinstance(primary, (KVSlot, StructuralHeaderSlot))
        mat_primary = primary

    # Owned-slot identity set + retained slot objects (for unlink).
    owned_ids: set[int] = set()
    owned_slots: list[Slot] = []

    def _add_slot(s: Slot) -> None:
        if id(s) in owned_ids:
            return
        owned_ids.add(id(s))
        owned_slots.append(s)

    for r in c._index.get(key, []):  # noqa: SLF001
        _add_slot(r.slot)

    # Subtree containers + AoTs + descendant owned slots.
    subtree_containers: list[Container] = []
    subtree_aots: list[AoT] = []
    _collect_subtree(val, subtree_containers, subtree_aots, _add_slot)

    # Synthesise the now-empty section's physical presence while the
    # descendant's primary slot is still linked, so the replacement takes
    # its position in place. The descendant's *origin* picks the form: a
    # header-origin section (``[a.b]``) re-materialises as a header
    # ``[a]``; a dotted-origin section (``a.b.x = 1``) re-materialises as
    # an inline table ``a = {}``, which — unlike a header — re-parents
    # nothing and so can sit among surviving sibling KVs untouched.
    if will_materialise:
        assert mat_primary is not None
        if isinstance(mat_primary, StructuralHeaderSlot):
            _materialise_empty_section_header(c, mat_primary, doc)
        else:
            _materialise_empty_inline_table(c, mat_primary, doc)

    # Slot-driven scrub via back-pointers, *skipping* subtree containers:
    # those move to a fresh Document and keep their internal caches.
    skip_ids = frozenset(id(sc) for sc in subtree_containers)
    _scrub_owned_slots_via_backptrs(owned_slots, skip_container_ids=skip_ids)

    # Body-tail recompute on the ancestor chain. `min_owned_depth`
    # short-circuits the common leaf-KV case.
    min_owned_depth = len(c._path)  # noqa: SLF001
    for s in owned_slots:
        d = len(s.host_path) if isinstance(s, KVSlot) else 0
        if d < min_owned_depth:
            min_owned_depth = d
    _invalidate_body_tail_chain(c, owned_ids, min_depth=min_owned_depth)

    # Unlink owned slots; transplant user-referenced subtrees to an
    # orphan Document. Keep entry_slots for AoTEntries whose AoT moves
    # with them, so clone/re-install can still read the full CST.
    moving_aot_entry_ids: set[int] = set()
    for ao in subtree_aots:
        for entry_table in list.__iter__(ao):
            owner_e = entry_table._owner_aot_entry  # noqa: SLF001
            if owner_e is not None:
                moving_aot_entry_ids.add(id(owner_e))

    candidate_owners: set[int] = set()
    for slot in owned_slots:
        owner = slot.owner_aot_entry
        if owner is not None and id(owner) not in moving_aot_entry_ids:
            candidate_owners.add(id(owner))
    surviving_aot_entries = (
        _surviving_aot_entries(doc, candidate_owners) if candidate_owners else set()
    )
    # Capture doc-stream order *before* the unlink loop severs the linked
    # list. ``owned_slots`` is in collection order (every ref bound under
    # ``key`` in ``c`` first — which front-loads nested headers — then the
    # subtree body), not doc-stream order; transplanting in that order
    # would corrupt the orphan's linked list. ``owned_slots[0]`` (the
    # binding's primary slot, ``c._index[key][0]``) anchors the walk; see
    # :func:`_owned_slots_ordered` for why that's usually but not always
    # doc-stream-first.
    transplanting = bool(subtree_containers or subtree_aots)
    ordered_for_transplant = (
        _owned_slots_ordered(owned_slots[0], owned_ids)
        if transplanting and owned_slots
        else owned_slots
    )
    for slot in reversed(owned_slots):
        owner = slot.owner_aot_entry
        if (
            owner is not None
            and id(owner) in surviving_aot_entries
            and id(owner) not in moving_aot_entry_ids
        ):
            with contextlib.suppress(ValueError):
                _pop_or_remove(owner.entry_slots, slot)
    # Unlink in *reverse* doc-stream order (see remove_aot_entry /
    # remove_aot_entries for the same idiom): unlinking a doc-stream-first
    # owned slot promotes its successor to be the new doc head, stripping
    # that successor's leading blank line. If that successor is itself
    # about to be unlinked too, the strip is wasted on a slot that's
    # leaving anyway, and the *actually* surviving new head never gets
    # stripped. Working back-to-front unlinks every later owned slot
    # first, so the doc-stream-first one goes last and any head-promotion
    # strip lands on the true surviving successor.
    for slot in reversed(ordered_for_transplant):
        unlink_slot(slot, doc)

    displaced_inlines: list[Container] = []
    displaced_arrays: list[Array] = []
    _collect_displaced_inline_views(val, displaced_inlines, displaced_arrays)
    if transplanting:
        from tomlrt._container import Document  # noqa: PLC0415

        orphan = Document()
        orphan._newline = doc._newline  # noqa: SLF001
        orphan._is_private = True  # noqa: SLF001
        _splice_block_after(ordered_for_transplant, None, orphan)
        for sc in subtree_containers:
            sc._layout_root = orphan  # noqa: SLF001
        for ao in subtree_aots:
            ao._layout_root = orphan  # noqa: SLF001
        # Keep inline descendants live against the orphan: their backing
        # CST lives inside a transplanted KV, so re-pointing (rather than
        # resetting) lets edits through a held reference flow into the
        # orphaned slot value, which a later rehome moves intact.
        for it in displaced_inlines:
            it._layout_root = orphan  # noqa: SLF001
        for ar in displaced_arrays:
            ar._layout_root = orphan  # noqa: SLF001
    else:
        # No orphan (e.g. a top-level inline value): reset by hand so a
        # held reference reports detached and can re-attach cleanly.
        from tomlrt._container import (  # noqa: PLC0415
            _reset_array_for_rehome,
            _reset_inline_for_rehome,
        )

        for it in displaced_inlines:
            if it._layout_root is not None:  # noqa: SLF001
                _reset_inline_for_rehome(it)
        for ar in displaced_arrays:
            if ar._attached:  # noqa: SLF001
                _reset_array_for_rehome(ar)

    # Drop the dict entry.
    dict.__delitem__(c, key)


def _walk_view_tree(val: object, visit: Callable[[object], None]) -> None:
    """Visit every Container / AoT / Array node in a view subtree.

    The three node kinds and their recursion (Container -> values,
    AoT -> entries, Array -> items) are the shared spine of the
    delete-side displacement walk (:func:`_collect_displaced_inline_views`)
    and the adopt-side rehome walk (:func:`_rehome_view_tree`); each
    caller supplies the per-node action. Scalars are inert.
    """
    from tomlrt._array import AoT, Array  # noqa: PLC0415
    from tomlrt._container import Container  # noqa: PLC0415

    if isinstance(val, Container):
        visit(val)
        for child in val.values():
            _walk_view_tree(child, visit)
    elif isinstance(val, AoT):
        visit(val)
        for entry in val:
            _walk_view_tree(entry, visit)
    elif isinstance(val, Array):
        visit(val)
        for item in val:
            _walk_view_tree(item, visit)


def _collect_displaced_inline_views(
    val: object,
    inlines_out: list[Container],
    arrays_out: list[Array],
) -> None:
    """Walk an about-to-be-displaced subtree, gathering inline views.

    Section Containers and AoTs are handled by ``_collect_subtree``
    + the orphan-rehome step. This walker complements that by
    reaching into inline tables and inline arrays — which carry no
    doc-stream slots of their own but do hold ``_layout_root`` /
    ``_attached`` state that goes stale when their hosting KV is
    deleted.
    """
    from tomlrt._array import Array  # noqa: PLC0415
    from tomlrt._container import Container  # noqa: PLC0415

    def visit(node: object) -> None:
        if isinstance(node, Container):
            if node._inline:  # noqa: SLF001
                inlines_out.append(node)
        elif isinstance(node, Array):
            arrays_out.append(node)

    _walk_view_tree(val, visit)


def _collect_subtree(
    val: object,
    containers_out: list[Container],
    aots_out: list[AoT],
    add_slot: Callable[[Slot], None],
) -> None:
    """Walk ``val``'s container subtree, collecting containers, AoTs and owned slots."""
    from tomlrt._array import AoT  # noqa: PLC0415
    from tomlrt._container import Container  # noqa: PLC0415

    if isinstance(val, Container):
        if val._inline:  # noqa: SLF001
            return
        containers_out.append(val)
        for r in val._refs:  # noqa: SLF001
            add_slot(r.slot)
        for child in val.values():
            _collect_subtree(child, containers_out, aots_out, add_slot)
    elif isinstance(val, AoT):
        aots_out.append(val)
        placeholder = _empty_aot_placeholder_ref(val)
        if placeholder is not None:
            add_slot(placeholder.slot)
        for entry in val:
            _collect_subtree(entry, containers_out, aots_out, add_slot)


def _owned_slots_ordered(start: Slot, owned_ids: set[int]) -> list[Slot]:
    """Collect ``owned_ids`` in true doc-stream order, anchored at ``start``.

    ``start`` is typically a binding's own header/primary slot, and
    usually — but not always — the owned set's doc-stream-first slot: a
    nested descendant's header or dotted KV may have been written
    physically *earlier* (legal, spec-conformant TOML, e.g. a sub-table
    ``[a.b]`` followed later by its parent's own ``[a]``). Walking
    forward from ``start`` finds every owned slot that follows it; any
    owned slot forward can't reach must instead precede it, so the walk
    back from ``start`` (via ``_prev``, needing no separate document-head
    reference) collects exactly the shortfall. Interleaved foreign slots
    are skipped either way (a binding's slots need not be contiguous —
    ``[[a]] … [b] … [[a]]`` is legal).

    Same cost as a plain forward walk in the common case (no backward
    step at all once forward has found everything); the rare shortfall
    only walks as far back as needed to find the missing slots, not the
    whole document.
    """
    forward: list[Slot] = []
    seen: set[int] = set()
    cur: Slot | None = start
    while cur is not None and len(seen) < len(owned_ids):
        if id(cur) in owned_ids:
            forward.append(cur)
            seen.add(id(cur))
        cur = cur._next  # noqa: SLF001
    missing = len(owned_ids) - len(seen)
    if not missing:
        return forward
    backward: list[Slot] = []
    cur = start._prev  # noqa: SLF001
    while cur is not None and len(backward) < missing:
        if id(cur) in owned_ids:
            backward.append(cur)
        cur = cur._prev  # noqa: SLF001
    assert len(backward) == missing, "owned slot unreachable from start"
    backward.reverse()
    return backward + forward


def _surviving_aot_entries(doc: Document, candidates: set[int]) -> set[int]:
    """Return ``id(AoTEntry)`` values from ``candidates`` still reachable in ``doc``.

    Bails out as soon as every candidate has been spotted.
    """
    from tomlrt._array import AoT  # noqa: PLC0415
    from tomlrt._container import Container  # noqa: PLC0415

    surviving: set[int] = set()
    remaining = set(candidates)

    def visit(v: object) -> None:
        if not remaining:
            return
        if isinstance(v, Container):
            owner = v._owner_aot_entry  # noqa: SLF001
            if owner is not None:
                oid = id(owner)
                if oid in remaining:
                    surviving.add(oid)
                    remaining.discard(oid)
            if not v._inline:  # noqa: SLF001
                for child in v.values():
                    if not remaining:
                        return
                    visit(child)
        elif isinstance(v, AoT):
            for entry in v:
                if not remaining:
                    return
                visit(entry)

    visit(doc)
    return surviving


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _last_kv(
    c: Container, *, direct: bool = False, body: bool = False
) -> KVSlot | None:
    """Reverse-walk ``c._refs`` for the last matching KV slot.

    Default: matches any KV with ``host_path == c._path`` and matching owner.
    ``direct=True``: further restrict to single-key-part.
    ``body=True``: header-bearing (SECTION) container's body is
    ``host_path == c._path``; otherwise (document root, implicit table)
    any KV with a matching owner counts.
    """
    owner = c._owner_aot_entry  # noqa: SLF001
    path = c._path  # noqa: SLF001
    lax_body = body and c._kind is not _Kind.SECTION  # noqa: SLF001
    for ref in reversed(c._refs):  # noqa: SLF001
        s = ref.slot
        if not isinstance(s, KVSlot) or s.owner_aot_entry is not owner:
            continue
        if not lax_body and s.host_path != path:
            continue
        if direct and len(s.key_parts) != 1:
            continue
        return s
    return None


def _is_direct_kv(c: Container, s: Slot) -> bool:
    """True iff ``s`` is a direct (single-key-part, host=c) KV of ``c``."""
    return (
        isinstance(s, KVSlot)
        and s.host_path == c._path  # noqa: SLF001
        and len(s.key_parts) == 1
        and s.owner_aot_entry is c._owner_aot_entry  # noqa: SLF001
    )


def _last_direct_kv(c: Container) -> KVSlot | None:
    """Return the most-recent direct KV slot of ``c`` in doc-stream order.

    Fast path: ``c._body_tail`` is by construction the latest body-region
    slot, and on every direct-KV append it IS the new direct KV — so for
    the typical "just-appended" case this is O(1). Otherwise (body_tail
    is a header or a dotted KV) reverse-walk ``c._refs``.
    """
    body_tail = c._body_tail  # noqa: SLF001
    if body_tail is not None and _is_direct_kv(c, body_tail):
        assert isinstance(body_tail, KVSlot)
        return body_tail
    return _last_kv(c, direct=True)


def _extract_indent(leading: Trivia) -> str:
    """Return indent (whitespace after the last newline) of ``leading``."""
    pieces = leading.pieces
    last_nl = -1
    for i, p in enumerate(pieces):
        if isinstance(p, NewlineNode):
            last_nl = i
    text = ""
    for p in pieces[last_nl + 1 :]:
        if isinstance(p, WhitespaceNode):
            text += p.text
        else:
            break
    return text


def _aot_sibling_last_kv(c: Container) -> KVSlot | None:
    """Return the last direct KV of the most recent prior AoT sibling.

    Used to inherit indent when ``c`` is an AoT entry root with no
    direct KVs of its own yet.
    """
    from tomlrt._array import AoT  # noqa: PLC0415

    owner = c._owner_aot_entry  # noqa: SLF001
    if owner is None:
        return None
    parent = c._parent  # noqa: SLF001
    if parent is None:
        return None
    key = c._path[-1] if c._path else None  # noqa: SLF001
    if key is None or key not in parent:
        return None
    aot = dict.__getitem__(parent, key)
    if not isinstance(aot, AoT):
        return None
    found_self = False
    for entry_table in reversed(aot):
        if entry_table is c:
            found_self = True
            continue
        if not found_self:
            continue
        sib = _last_direct_kv(entry_table)
        if sib is not None:
            return sib
    return None


def _peer_separator(prev_leading: Trivia | None, doc: Document) -> Trivia:
    """Mirror a peer's blank-gap when emitting a new structural sibling.

    Returns a single-newline ``Trivia`` (one blank line of separation)
    iff ``prev_leading`` itself contains a blank line, or when there
    is no peer to mirror (the conventional default for the first
    sibling of its kind). Otherwise returns empty ``Trivia``.

    This is the shared "match the last peer" rule used by KV append,
    section-header insertion, and AoT-entry append; each caller wraps
    it with kind-specific peer lookup and any extra decoration (e.g.
    KV indent).
    """
    if prev_leading is None or leading_has_blank_line(prev_leading):
        return Trivia([NewlineNode(text=doc._newline)])  # noqa: SLF001
    return Trivia()


def _kv_leading_after(
    prev: KVSlot | None, doc: Document, fallback_indent: str = ""
) -> Trivia:
    """Build leading trivia for a new KV slot following ``prev``.

    Inherits indent from ``prev`` and mirrors its blank-gap so the
    new KV continues the user's most recent spacing convention. With
    no prior sibling, falls back to a bare ``fallback_indent``.
    """
    if prev is None:
        if fallback_indent:
            return Trivia([WhitespaceNode(text=fallback_indent)])
        return Trivia()
    pieces: list[TriviaPiece] = list(_peer_separator(prev.leading, doc).pieces)
    indent_text = _extract_indent(prev.leading)
    if indent_text:
        pieces.append(WhitespaceNode(text=indent_text))
    return Trivia(pieces)


def _kv_separator_leading(c: Container, doc: Document) -> Trivia:
    """Pick leading trivia for a new direct-KV slot in container ``c``.

    For an AoT entry with no own KVs yet, falls back to inheriting
    indent (only) from the previous sibling entry's last KV.
    """
    last = _last_direct_kv(c)
    if last is not None:
        return _kv_leading_after(last, doc)
    sibling = _aot_sibling_last_kv(c)
    fallback = _extract_indent(sibling.leading) if sibling is not None else ""
    return _kv_leading_after(None, doc, fallback_indent=fallback)


def _new_kv_slot(
    *,
    host_path: tuple[str, ...],
    key: tuple[str, ...],
    value: Value,
    doc: Document,
    owner: AoTEntry | None,
    leading: Trivia,
    key_parts: Sequence[KeyPart] | None = None,
    key_seps: Sequence[str] | None = None,
) -> KVSlot:
    """Synthesise a fresh KV slot (recorded when spliced, not here).

    By default ``key_parts`` and ``key_seps`` use canonical synthetic
    spelling. Callers moving an existing value may supply source spelling.
    """
    return KVSlot(
        leading=leading,
        host_path=host_path,
        key_parts=make_keyparts(key) if key_parts is None else list(key_parts),
        key_seps=["."] * (len(key) - 1) if key_seps is None else list(key_seps),
        pre_eq=" ",
        post_eq=" ",
        value=value,
        eol=_default_eol(doc),
        owner_aot_entry=owner,
    )


def _build_kv_slot(
    c: Container,
    key: str,
    value: Value,
    doc: Document,
    *,
    key_parts: Sequence[KeyPart] | None = None,
    key_seps: Sequence[str] | None = None,
) -> KVSlot:
    """Synthesise a new ``KVSlot`` carrying default trivia + style."""
    return _new_kv_slot(
        host_path=c._path,  # noqa: SLF001
        key=(key,),
        value=value,
        doc=doc,
        owner=c._owner_aot_entry,  # noqa: SLF001
        leading=_kv_separator_leading(c, doc),
        key_parts=key_parts,
        key_seps=key_seps,
    )


def install_dotted_kv_slot(
    host: Container,
    leaf_keypath: tuple[str, ...],
    value: Value,
    *,
    leaf_parent: Container,
    leading: Trivia | None = None,
    key_parts: Sequence[KeyPart] | None = None,
    key_seps: Sequence[str] | None = None,
) -> None:
    """Insert a single dotted-KV slot hosted by ``host``.

    Files refs on ``host`` and every implicit intermediate in
    ``[host, ..., leaf_parent]``, updates ``_body_tail`` along the
    chain, and maintains ``AoTEntry.entry_slots`` membership. The caller
    owns dict storage at ``leaf_parent``.

    ``leading`` overrides the synthesised separator/indent — used by the
    trivia-preserving graft to carry a cloned source slot's leading
    (standalone comments) onto the new slot.

    Pre-conditions (checked by caller):
    ``host`` can own the KV scope; the implicit chain already exists;
    ``leaf_parent`` is that chain's leaf; ``leaf_keypath[-1]`` is
    unbound; and ``len(leaf_keypath) >= 2``.
    """
    assert len(leaf_keypath) >= 2
    doc = host._attached_doc  # noqa: SLF001

    # Build chain [host, ..., leaf_parent] via _parent walk + reverse.
    chain = _dotted_chain(host, leaf_parent)
    assert len(chain) == len(leaf_keypath)

    body_tail = leaf_parent._body_tail or host._body_tail  # noqa: SLF001
    header_ref = host._header_ref  # noqa: SLF001
    owner = host._owner_aot_entry  # noqa: SLF001

    new_slot = _new_kv_slot(
        host_path=host._path,  # noqa: SLF001
        key=leaf_keypath,
        value=value,
        doc=doc,
        owner=owner,
        leading=leading
        if leading is not None
        else _kv_leading_after(_last_kv(host), doc),
        key_parts=key_parts,
        key_seps=key_seps,
    )

    _splice_body_slot(
        new_slot,
        anchor_body_tail=body_tail,
        anchor_header_ref=header_ref,
        doc=doc,
    )

    # Each ancestor needs its own physical predecessor; cached tails can
    # lie on either side of this newly-spliced slot.
    anchors = _nearest_filed_predecessors(chain, new_slot)
    for i, (anc, anchor) in enumerate(zip(chain, anchors, strict=True)):
        ref = _file_ordered_ref(
            anc,
            new_slot,
            predecessor=anchor,
        )
        assert ref.local_key == leaf_keypath[i]
        if anchor is anc._body_tail or anc._body_tail is None:  # noqa: SLF001
            anc._body_tail = new_slot  # noqa: SLF001

    # ``entry_slots`` is membership + header-first order only (doc
    # order is derived on demand), so a plain append is enough.
    _extend_entry_slots(owner, new_slot)


def _synthesise_header_then_insert_kv(c: Container, key: str, value: Value) -> None:
    """Promote a purely-implicit container ``c`` to an explicit section.

    When ``c`` has a descendant, inserts ``[c._path]`` immediately before
    it and transfers the existing seam to the header. Otherwise appends the
    new block inside the owning AoT entry or at document tail; structural
    replacement moves it to the caller's saved anchor afterward.

    Pre-condition: ``c`` is non-root, header-less, and body-less.
    """
    doc = c._attached_doc  # noqa: SLF001
    owner = c._owner_aot_entry  # noqa: SLF001
    anchor_slot = c._refs[0].slot if c._refs else None  # noqa: SLF001
    host: Container | None = None

    if anchor_slot is not None:
        adopted_leading = anchor_slot.leading
        original_pred = anchor_slot._prev  # noqa: SLF001
        new_descendant_leading = _build_section_leading(doc)
        header_slot = _new_owned_section_header(c, leading=adopted_leading, doc=doc)
        insert_before(anchor_slot, header_slot, doc)
        recorder = doc._install_recorders  # noqa: SLF001
        if recorder is not None:
            recorder[1].append(
                (anchor_slot, list(anchor_slot.leading.pieces), original_pred)
            )
        anchor_slot.leading = new_descendant_leading
    else:
        host, host_tail = _header_host_and_tail(c)
        header_slot = _new_owned_section_header(
            c, leading=_build_section_leading(doc), doc=doc
        )
        # Keep an anchorless promoted header inside its nearest
        # header-bearing host, not under an unrelated document tail.
        _splice_block_after([header_slot], host_tail, doc)
        if isinstance(header_slot._prev, StructuralHeaderSlot):  # noqa: SLF001
            header_slot.leading = Trivia()

    parent = c._parent  # noqa: SLF001
    assert parent is not None
    _file_header_binding_chain(parent, header_slot, host=host)

    new_kv = _file_synthetic_header_and_kv(
        c,
        header_slot=header_slot,
        key=key,
        value=value,
        doc=doc,
        owner=owner,
        header_ref_index=0,
    )

    # ``entry_slots`` is membership + header-first order, not doc order.
    _extend_entry_slots(owner, header_slot, new_kv)


def _ensure_terminator(slot: Slot, doc: Document) -> None:
    """Give ``slot`` a trailing newline if it lacks one (no-final-newline doc)."""
    if isinstance(slot, (KVSlot, StructuralHeaderSlot)) and slot.eol.newline is None:
        slot.eol = EolTrivia(
            trailing_ws=slot.eol.trailing_ws,
            comment=slot.eol.comment,
            newline=NewlineNode(text=doc._newline),  # noqa: SLF001
        )


def _terminate_unless_tail(slot: Slot, doc: Document) -> None:
    """Ensure ``slot`` has a trailing newline, unless it is now the doc tail.

    A slot cloned or moved from a no-final-newline source (it was
    previously the very last slot there) can arrive with no trailing
    newline of its own. That's fine if it lands at this doc's own tail
    too, but anywhere else it now runs into whatever follows on the
    same line.
    """
    if slot is not doc._tail:  # noqa: SLF001
        _ensure_terminator(slot, doc)


def _ensure_leading_blank_line(slot: Slot, doc: Document) -> None:
    """Ensure ``slot.leading`` begins with a blank line.

    A run of ``pieces`` starts with a blank line when the first
    non-whitespace piece is a ``NewlineNode``. If a comment appears
    first, prepend a fresh ``NewlineNode`` so the comment block stays
    visually detached from the slot.
    """
    pieces = slot.leading.pieces
    for p in pieces:
        if isinstance(p, NewlineNode):
            return
        if isinstance(p, CommentNode):
            break
    pieces.insert(0, NewlineNode(text=doc._newline))  # noqa: SLF001


def _find_ref_index_by_slot(c: Container, slot: Slot) -> int:
    """Locate ``slot``'s ref in ``c._refs``, scanning from both ends.

    Callers pass body-tail / anchor slots whose position in ``c._refs``
    is either near the end (typical: body sits before a few trailing
    sub-section header refs) or near the start (e.g. doc-root with a
    handful of top-level KVs preceding many section headers, as in
    bulk ``doc[k] = inline`` patterns). A two-pronged scan converges
    in O(min(P, N-P)) instead of always degrading to O(N) at one end.
    """
    refs = c._refs  # noqa: SLF001
    lo, hi = 0, len(refs) - 1
    while lo <= hi:
        if refs[hi].slot is slot:
            return hi
        if refs[lo].slot is slot:
            return lo
        lo += 1
        hi -= 1
    msg = "internal: anchor slot not found in c._refs"
    raise AssertionError(msg)


def _recompute_body_tail(c: Container) -> Slot | None:
    """Last body-region ref's slot in ``c._refs`` (mirrors invariants rule)."""
    found = _last_kv(c, body=True)
    if found is not None:
        return found
    if c._header_ref is not None:  # noqa: SLF001
        # Header-only container falls back to its own header.
        return c._header_ref.slot  # noqa: SLF001
    return None


# ---------------------------------------------------------------------------
# Structural attach — section / AoT synthesis
# ---------------------------------------------------------------------------


def _new_section_header(
    path: tuple[str, ...],
    *,
    leading: Trivia,
    doc: Document,
    entry: AoTEntry | None = None,
    owner_aot_entry: AoTEntry | None = None,
) -> StructuralHeaderSlot:
    return StructuralHeaderSlot(
        leading=leading,
        path=path,
        key_parts=make_keyparts(path),
        key_seps=["."] * (len(path) - 1),
        eol=_default_eol(doc),
        entry=entry,
        owner_aot_entry=owner_aot_entry,
        synthetic=True,
    )


def _belongs_to_parent_extent(
    slot: Slot,
    base_path: tuple[str, ...],
    base_owner: AoTEntry | None,
) -> bool:
    """Is ``slot`` within the physical subtree rooted at ``base_path``?

    The container is identified by ``(base_path, base_owner)``: an
    AoT-entry table has the same path as its sibling entries, so the
    owner is needed to disambiguate same-level slots.

    Rule: a slot whose path strictly extends ``base_path`` is always
    in-extent (it's a descendant section / KV / nested AoT entry, and
    its own ``owner_aot_entry`` is irrelevant). A slot at exactly
    ``base_path`` is in-extent only if it shares ``base_owner`` —
    otherwise it is a sibling AoT entry at the same level.

    The descendant rule is valid only while walking a physically
    contiguous doc-stream region. It must not filter an ``_index``
    bucket, which can interleave descendants from sibling AoT entries.
    """
    if isinstance(slot, KVSlot):
        path = slot.host_path
    else:
        assert isinstance(slot, StructuralHeaderSlot)
        path = slot.path
    n = len(base_path)
    if path[:n] != base_path:
        return False
    if len(path) > n:
        return True
    return slot.owner_aot_entry is base_owner


def _parent_subtree_tail(parent: Container) -> Slot | None:
    """Return the last slot in ``parent``'s physical subtree.

    Walks forward in the doc-stream linked list from ``parent._refs[-1]``
    while subsequent slots still belong to ``parent``'s extent (see
    :func:`_belongs_to_parent_extent` for the precise predicate).
    """
    refs = parent._refs  # noqa: SLF001
    if not refs:
        return None
    base_path = parent._path  # noqa: SLF001
    base_owner = parent._owner_aot_entry  # noqa: SLF001
    cur = refs[-1].slot
    while cur._next is not None:  # noqa: SLF001
        nxt = cur._next  # noqa: SLF001
        if not _belongs_to_parent_extent(nxt, base_path, base_owner):
            break
        cur = nxt
    return cur


def _header_host_and_tail(c: Container) -> tuple[Container, Slot | None]:
    """Return ``c``'s nearest header-bearing host and its subtree tail."""
    host = _nearest_header_host(c)
    return host, _parent_subtree_tail(host)


def _safe_header_anchor(anchor: Slot | None) -> Slot | None:
    """Extend a subtree-tail anchor past any immediately-following bare KVs.

    A dotted key inherits its scope from physical position rather than
    from any header of its own, so a KV belonging to an unrelated
    sibling (one that happens to sit right after ``parent``'s own
    extent, e.g. another root-level implicit table's key) does not
    bound a safe insertion point for a *header*-bearing block — landing
    one there would recapture that KV under the new header on re-parse.
    Skip forward past any such run to the next structural header, or
    doc end, where insertion is unambiguous.
    """
    while anchor is not None and isinstance(anchor._next, KVSlot):  # noqa: SLF001
        anchor = anchor._next  # noqa: SLF001
    return anchor


def _child_header_anchor(parent: Container) -> Slot | None:
    """Return a safe anchor for a new header-backed child of ``parent``."""
    return _safe_header_anchor(
        _parent_subtree_tail(parent) or _header_host_and_tail(parent)[1]
    )


def _splice_at_end(slot: Slot, doc: Document) -> None:
    """Insert ``slot`` at the end of the doc-stream."""
    anchor = doc._tail  # noqa: SLF001
    if anchor is None:
        # Empty doc: this is also the first slot, so any preamble
        # parked in `_trailing` migrates onto its leading.
        insert_before_head(slot, doc)
        _promote_trailing_to_preamble(doc)
    else:
        _ensure_terminator(anchor, doc)
        insert_after(anchor, slot, doc)


def _splice_block_after(slots: list[Slot], anchor: Slot | None, doc: Document) -> None:
    """Splice a contiguous, internally terminated block after ``anchor``."""
    if not slots:
        return
    if anchor is None:
        _splice_at_end(slots[0], doc)
    else:
        _ensure_terminator(anchor, doc)
        insert_after(anchor, slots[0], doc)
    prev = slots[0]
    for s in slots[1:]:
        _ensure_terminator(prev, doc)
        insert_after(prev, s, doc)
        prev = s
    _terminate_unless_tail(prev, doc)


def _maybe_demote_synthetic_empty_header(parent: Container) -> None:
    """Drop ``parent``'s header if it is synthetic and has no direct KV body.

    Used after attaching a child header under ``parent``: if ``parent``
    was synthesised as an empty placeholder (e.g.
    ``doc["tool"] = Table.section({})``) and the new child gives it a
    dotted-implicit anchor (``[tool.poetry]``), the placeholder header
    is redundant and is removed.
    """
    hdr_ref = parent._header_ref  # noqa: SLF001
    if hdr_ref is None:
        return
    header = hdr_ref.slot
    assert isinstance(header, StructuralHeaderSlot)
    if not header.synthetic or header.kind != "table":
        return
    # The header's physical body extends to the next structural
    # header or EOF. Every Slot is either a KVSlot or a
    # StructuralHeaderSlot, so the body is non-empty iff the very
    # next slot is a KVSlot.
    successor = header._next  # noqa: SLF001
    if isinstance(successor, KVSlot):
        return
    layout_root = parent._layout_root  # noqa: SLF001
    from tomlrt._container import Document  # noqa: PLC0415

    assert isinstance(layout_root, Document)
    doc = layout_root
    # Remove the header from the doc stream and from all caches.
    # Hand the demoted header's leading trivia (its separation-from-above,
    # plus the file preamble / comments it carries when it sits at doc
    # head) off to the successor so nothing is silently dropped on
    # promotion to implicit. The successor's own leading was a separator
    # *from the header* — now redundant — so strip it first, otherwise
    # the transfer stacks a second blank line before the successor.
    unlink_slot(header, doc, strip_new_head_leading=True)
    if successor is not None:
        _strip_leading_blank_lines(successor)
        if header.leading.pieces:
            successor.leading.pieces = [
                *header.leading.pieces,
                *successor.leading.pieces,
            ]
    parent._body_tail = None  # noqa: SLF001
    # Canonical bulk-scrub via the header's back-pointer list: drops
    # ``hdr_ref`` from ``parent._refs`` (and clears ``parent._header_ref``
    # as a side effect of ``unfile_ref``'s own-header branch), drops the
    # binding refs from every ancestor's ``_refs`` / ``_index``, and
    # empties ``header._refs`` itself so the orphaned slot leaves no
    # stale back-pointers behind.
    _scrub_owned_slots_via_backptrs([header])
    # Owner aot-entry, if any, also drops it.
    owner = header.owner_aot_entry
    if owner is not None:
        with contextlib.suppress(ValueError):
            owner.entry_slots.remove(header)


def _split_at_remainder(
    leading: Trivia,
    remainder_lines: Iterable[Iterable[TriviaPiece]],
    indent: Iterable[TriviaPiece],
) -> tuple[Trivia, Trivia]:
    """Build ``(positional-prefix, remainder)`` from already-classified lines.

    Concatenates ``remainder_lines`` and ``indent`` to form the
    remainder; the positional prefix is whatever's left of
    ``leading`` once ``len(remainder)`` pieces have been peeled off
    the tail. Callers are responsible for choosing which lines
    travel with the slot.
    """
    pieces = leading.pieces
    remainder: list[TriviaPiece] = []
    for line in remainder_lines:
        remainder.extend(line)
    remainder.extend(indent)
    cut = len(pieces) - len(remainder)
    return Trivia(list(pieces[:cut])), Trivia(remainder)


def _split_leading_structural(leading: Trivia) -> tuple[Trivia, Trivia]:
    """Split a leading-trivia stream into (above-blank prefix, slot-remainder).

    The slot-remainder is the attached comment block (immediately above
    the slot, with no blank line between) plus the slot's own column-
    offset indent. The positional prefix is everything that came before
    it in the leading.

    Used by reorder paths to decide which prefix travels with the slot
    under move and which is positional (separator) trivia at the seam.
    """
    _above, attached, indent = _split_attached_block(leading)
    return _split_at_remainder(leading, attached, indent)


def _split_leading_for_reorder(slot: Slot) -> tuple[Trivia, Trivia]:
    """Reorder-aware leading split: disjoint comment blocks travel with the slot.

    Per the public ownership model (``Table.header_leading_block``,
    ``Container.leading_block``), an above-blank comment block that
    immediately precedes a slot is part of that slot's leading and
    must travel with it under reorder. For every slot the positional
    prefix is the run of pure-blank lines before the first comment;
    the remainder is everything from the first comment line onward.
    """
    above, attached, indent = _split_attached_block(slot.leading)
    i = 0
    while i < len(above) and not has_comment(above[i]):
        i += 1
    return _split_at_remainder(slot.leading, [*above[i:], *attached], indent)


def _retarget_header_separator(
    header: StructuralHeaderSlot,
    new_separator: Trivia,
) -> None:
    """Replace ``header.leading``'s positional prefix with ``new_separator``.

    See :func:`_split_leading_structural`: the slot's attached
    comments and own indent are kept; the source's positional
    prefix is dropped.
    """
    _positional, remainder = _split_leading_structural(header.leading)
    header.leading = Trivia([*new_separator.pieces, *remainder.pieces])


def _build_section_leading(doc: Document) -> Trivia:
    """Trivia for a fresh section header.

    Empty doc → no leading; otherwise use the document's stable
    structural-header spacing convention learned when it was parsed
    (or the canonical blank-separated default for a fresh document).
    """
    if doc._head is None:  # noqa: SLF001
        return Trivia()
    if doc._section_blank_separated:  # noqa: SLF001
        return Trivia([NewlineNode(text=doc._newline)])  # noqa: SLF001
    return Trivia()


def attach_empty_aot(parent: Container, key: str, source_aot: AoT) -> AoT:
    """Bind an empty AoT under ``parent[key]``.

    The AoT has no entries, so its physical presence is a single
    ``key = []`` placeholder KVSlot (an empty inline array) filed in
    ``parent``'s body. The first ``aot.add(...)`` consumes that
    placeholder and materialises the first ``[[path]]`` header in its
    stead. The ``source_aot`` is rehomed in place (identity preserved).
    """
    assert len(source_aot) == 0, "non-empty AoT live-attach has its own routing"
    # Rehome the orphan AoT into this parent's logical scope.
    source_aot._layout_root = parent._layout_root  # noqa: SLF001
    source_aot._path = (*parent._path, key)  # noqa: SLF001
    source_aot._parent = parent  # noqa: SLF001
    _materialise_empty_aot(source_aot)
    return source_aot


def _materialise_empty_aot(aot: AoT) -> None:
    """Splice a ``key = []`` placeholder for a now-empty attached AoT.

    The placeholder is a normal direct KV (empty ``ArrayValue``) under
    the AoT's parent, so it lands in the parent's body region rather
    than at a header position a re-parse would misattribute. Dict
    storage at ``parent[key]`` is left as the AoT — only the physical
    slot is created.
    """
    parent = aot._parent  # noqa: SLF001
    assert parent is not None
    assert len(aot) == 0
    key = aot._path[-1]  # noqa: SLF001
    append_direct_kv(parent, key, ArrayValue())


def _empty_aot_placeholder_ref(aot: AoT) -> SlotRef | None:
    """Return the ``key = []`` placeholder ref backing an empty AoT, if any.

    Derived from the parent's ``_index[key]``: an empty AoT's only
    physical presence is one ``KVSlot`` whose value is an empty
    ``ArrayValue``. Returns ``None`` when the AoT is non-empty or carries
    no placeholder yet (e.g. a fresh AoT mid-clone, before its first
    entry or placeholder lands).
    """
    if len(aot) != 0:
        return None
    parent = aot._parent  # noqa: SLF001
    assert parent is not None
    key = aot._path[-1]  # noqa: SLF001
    bucket = parent._index.get(key)  # noqa: SLF001
    if not bucket:
        return None
    ref = bucket[0]
    slot = ref.slot
    assert isinstance(slot, KVSlot), "empty AoT placeholder must be a KV slot"
    assert isinstance(slot.value, ArrayValue), (
        "empty AoT key must be bound to an array placeholder"
    )
    return ref


def _consume_first_entry_placeholder(aot: AoT, ordinal: int) -> None:
    """Drop the ``key = []`` placeholder before the AoT's first entry lands.

    No-op past entry 0 or when the AoT carries no placeholder (the
    fresh-AoT clone path). The first ``[[path]]`` header takes the AoT's
    structural position (after the parent body), not the placeholder's
    in-body position, which a re-parse could otherwise misattribute.
    Runs before the append anchor is computed and before any synthetic
    parent header is demoted.

    Mirrors the focused leaf-delete sequence in `delete_key`: scrub
    every ref via slot back-pointers, recompute body tails on the
    parent chain, drop any ``entry_slots`` membership, then unlink.
    """
    if ordinal != 0:
        return
    ref = _empty_aot_placeholder_ref(aot)
    if ref is None:
        return
    parent = aot._parent  # noqa: SLF001
    assert parent is not None
    doc = aot._attached_doc  # noqa: SLF001
    slot = ref.slot
    _scrub_owned_slots_via_backptrs([slot])
    min_depth = len(slot.host_path) if isinstance(slot, KVSlot) else 0
    _invalidate_body_tail_chain(parent, {id(slot)}, min_depth=min_depth)
    owner = slot.owner_aot_entry
    if owner is not None:
        with contextlib.suppress(ValueError):
            owner.entry_slots.remove(slot)
    unlink_slot(slot, doc)


def _aot_separator(aot: AoT, doc: Document) -> Trivia:
    """Pick the leading-trivia for a newly-appended AoT entry header.

    Mirrors the most recent entry's blank-gap; for the first entry
    (or an empty/zero-slot last entry), defaults to one blank line.
    """
    if len(aot) <= 1:
        return _peer_separator(None, doc)
    last_entry = aot[-1]._owner_aot_entry  # noqa: SLF001
    if last_entry is None or not last_entry.entry_slots:
        return _peer_separator(None, doc)
    return _peer_separator(last_entry.entry_slots[0].leading, doc)


def add_aot_entry(
    aot: AoT, body: Mapping[str, Any] | None, *, rehome: Table | None = None
) -> Table:
    """Append a ``[[path]]`` entry to ``aot`` and return its `Table` view.

    If ``rehome`` is supplied (must be an unattached ``Table``), it is
    used as the entry view so the caller can preserve identity for a
    user reference. ``body`` is then ignored — ``rehome``'s own dict
    storage is used as the source body and is cleared/repopulated
    in place.
    """
    from tomlrt._container import (  # noqa: PLC0415
        Table,
        _is_synth_inline,
        _synth_value,
    )

    parent = aot._parent  # noqa: SLF001
    doc = aot._attached_doc  # noqa: SLF001
    path = aot._path  # noqa: SLF001
    assert parent is not None
    assert path

    ordinal = len(aot)
    # Consume the ``key = []`` placeholder before computing the append
    # anchor or demoting the parent header: the first ``[[path]]`` header
    # takes the AoT's structural position (after the parent body), not
    # the placeholder's in-body position, which could otherwise capture
    # trailing parent KVs on re-parse.
    _consume_first_entry_placeholder(aot, ordinal)
    entry = AoTEntry()
    leading = _build_section_leading(doc) if ordinal == 0 else _aot_separator(aot, doc)
    header = _new_section_header(
        path,
        leading=leading,
        doc=doc,
        entry=entry,
        owner_aot_entry=entry,
    )
    entry.entry_slots.append(header)

    # Build entry-root container (or rehome an existing one).
    body_items: list[tuple[str, object]]
    if rehome is not None:
        assert isinstance(rehome, Table)
        assert rehome._layout_root is None  # noqa: SLF001
        entry_table = rehome
        body_items = list(rehome.items())
        dict.clear(entry_table)
    else:
        entry_table = Table()
        body_items = list(body.items()) if body is not None else []
    _wire_section_container(
        entry_table,
        doc=doc,
        path=path,
        parent=parent,
        owner=entry,
    )
    _file_own_header(entry_table, header)

    append_host, append_anchor = _aot_append_position(aot)
    _splice_block_after([header], append_anchor, doc)
    _file_header_binding_chain(parent, header, host=append_host)
    list.append(aot, entry_table)

    # First entry under a synthetic placeholder section makes the
    # parent header redundant — the dotted-implicit anchor lives
    # entirely in `[[tool.list]]`.
    if ordinal == 0:
        _maybe_demote_synthetic_empty_header(parent)

    for k, v in body_items:
        if not (is_scalar(v) or _is_synth_inline(v)):
            entry_table[k] = v
            continue
        cst, dec = _synth_value(
            v,
            layout_root=doc,
            parent=entry_table,
            path=(*path, k),
            owner=entry,
        )
        append_direct_kv(entry_table, k, cst)
        dict.__setitem__(entry_table, k, dec)
    return entry_table


def prepare_promoted_inline_entries(
    entries: Sequence[InlineTableEntry],
) -> list[tuple[InlineTableEntry, Value]]:
    """Capture inline entries for section-side installation.

    Scalar value nodes are immutable after construction and can be shared
    safely. Composite values are copied so held views displaced by promotion
    remain detached from the new section.
    """
    return [
        (
            entry,
            copy.deepcopy(entry.value)
            if isinstance(entry.value, (ArrayValue, InlineTableValue))
            else entry.value,
        )
        for entry in entries
    ]


def populate_promoted_inline_entries(
    target: Container,
    entries: Sequence[tuple[InlineTableEntry, Value]],
) -> None:
    """Install captured inline entries as section-backed KV slots."""
    from tomlrt._build import _decode_value  # noqa: PLC0415

    doc = target._attached_doc  # noqa: SLF001
    owner = target._owner_aot_entry  # noqa: SLF001
    for source, value in entries:
        key_path = source.key_path
        leaf_parent = (
            target
            if len(key_path) == 1
            else ensure_implicit_chain(target, key_path[:-1])
        )
        leaf = key_path[-1]
        decoded = _decode_value(
            value,
            layout_root=doc,
            parent=leaf_parent,
            path=(*leaf_parent._path, leaf),  # noqa: SLF001
            owner=owner,
        )
        if len(key_path) == 1:
            append_direct_kv(
                target,
                leaf,
                value,
                key_parts=source.key_parts,
                key_seps=source.key_seps,
            )
        else:
            install_dotted_kv_slot(
                target,
                key_path,
                value,
                leaf_parent=leaf_parent,
                key_parts=source.key_parts,
                key_seps=source.key_seps,
            )
        dict.__setitem__(leaf_parent, leaf, decoded)


def clone_aot_entry(
    aot: AoT,
    src: Container,
    *,
    dst_path: tuple[str, ...] | None = None,
    preserve_source_separator: bool = False,
) -> Table:
    """Append a deep CST clone of ``src`` (a live attached entry) to ``aot``.

    The live view supplies its complete subtree, including nested AoT
    entries. Paths and newlines are retargeted; source trivia is kept.
    The destination separator is used unless
    ``preserve_source_separator`` is true.
    """
    owner = src._owner_aot_entry  # noqa: SLF001
    assert owner is not None, "source entry has no owning AoTEntry"
    src_slots = _gather_subtree_slots(src)

    target_path = dst_path if dst_path is not None else aot._path  # noqa: SLF001
    return _install_cloned_aot_entry(
        aot,
        src_slots,
        owner.path,
        target_path=target_path,
        rewrite_separator=not preserve_source_separator,
    )


def _install_cloned_structural_block(
    table: Table,
    *,
    parent: Container,
    doc: Document,
    target_path: tuple[str, ...],
    owner: AoTEntry | None,
    cloned_slots: list[Slot],
    anchor: Slot | None,
    host: Container | None = None,
) -> None:
    """Wire ``table``'s own-header ref, splice its slots, then build views.

    ``cloned_slots`` includes ``table``'s own header — not necessarily at
    index 0, since doc-stream order may put a forward-declared nested
    descendant's header first — so the full list, header included, goes
    to :func:`_populate_entry_views`; ``_build_containers`` recognises
    and files ``table``'s own header wherever it falls rather than
    reopening it as a child.
    """
    _wire_section_container(
        table,
        doc=doc,
        path=target_path,
        parent=parent,
        owner=owner,
    )
    _splice_block_after(cloned_slots, anchor, doc)
    _populate_entry_views(
        entry_table=table,
        cloned_slots=cloned_slots,
        target_prefix=target_path,
        doc=doc,
    )
    _extend_header_bindings_to_root(parent, cloned_slots, host=host)
    _maybe_demote_synthetic_empty_header(parent)


def _install_cloned_aot_entry(
    aot: AoT,
    src_slots: list[Slot],
    src_prefix: tuple[str, ...],
    *,
    target_path: tuple[str, ...],
    rewrite_separator: bool,
) -> Table:
    """Common installer for appending a cloned aot-entry to ``aot``.

    Deep-clones ``src_slots`` (head becomes an aot-entry), wires a fresh
    entry container, splices after the AoT's last slot, files the parent
    binding ref, and populates child views.

    ``src_slots[0]`` must be the entry's own header — an array entry's
    content can never physically precede its own ``[[..]]``/``[..]``
    header (unlike a plain table's, which a nested descendant can
    forward-declare), so this holds for every caller without reordering.

    ``rewrite_separator``: if True, the source's structural leading
    is replaced with destination-style preamble (entry 0) or the
    AoT's inter-entry separator (entry > 0). :func:`clone_aot` sets this
    False past entry 0 so source separators survive.
    """
    from tomlrt._container import Table  # noqa: PLC0415

    parent = aot._parent  # noqa: SLF001
    doc = aot._attached_doc  # noqa: SLF001
    assert parent is not None
    assert target_path

    ordinal = len(aot)
    _consume_first_entry_placeholder(aot, ordinal)
    new_entry = AoTEntry()

    cloned_slots, cloned_header = _clone_entry_slots(
        src_slots,
        new_entry=new_entry,
        body_owner=new_entry,
        src_prefix=src_prefix,
        target_prefix=target_path,
        dst_newline=doc._newline,  # noqa: SLF001
        head=src_slots[0],
    )
    assert cloned_header is not None

    if ordinal == 0:
        _retarget_header_separator(cloned_header, _build_section_leading(doc))
    elif rewrite_separator:
        _retarget_header_separator(cloned_header, _aot_separator(aot, doc))
    # else: keep source leading verbatim (bulk-clone past entry 0).

    append_host, append_anchor = _aot_append_position(aot)
    entry_table = Table()
    _install_cloned_structural_block(
        entry_table,
        parent=parent,
        doc=doc,
        target_path=target_path,
        owner=new_entry,
        cloned_slots=cloned_slots,
        anchor=append_anchor,
        host=append_host,
    )

    list.append(aot, entry_table)
    return entry_table


def _install_cloned_section(
    parent: Container,
    key: str,
    src_slots: list[Slot],
    src_prefix: tuple[str, ...],
    head: Slot,
) -> Table:
    """Common installer for ``parent[key] = <cloned section>``.

    Deep-clones ``src_slots`` (rewriting ``head`` — the source's own
    boundary header, identified by identity since doc-stream order may
    put it after a forward-declared nested descendant — from ``[..]`` /
    ``[[..]]`` to ``[<key>]``, rebasing paths from ``src_prefix`` to
    ``parent._path + (key,)``), wires the section container, splices at
    the parent's subtree anchor, and populates child views.
    """
    layout_root = parent._layout_root  # noqa: SLF001
    assert layout_root is not None, (
        "cloned-section install requires parent attached to a document"
    )
    doc = layout_root
    target_path = (*parent._path, key)  # noqa: SLF001

    cloned_slots, cloned_head = _clone_entry_slots(
        src_slots,
        new_entry=None,
        body_owner=parent._owner_aot_entry,  # noqa: SLF001
        src_prefix=src_prefix,
        target_prefix=target_path,
        dst_newline=doc._newline,  # noqa: SLF001
        head=head,
    )
    assert cloned_head is not None

    # The slot physically first in the spliced block is the one being
    # detached from its original doc-stream predecessor, so it needs a
    # destination-appropriate separator. That may be a forward-declared
    # nested descendant's header rather than ``cloned_head`` itself, but
    # either way it must be a header: a bare/dotted KV can never precede
    # its own table's header in valid TOML.
    first = cloned_slots[0]
    assert isinstance(first, StructuralHeaderSlot)
    _retarget_header_separator(first, _build_section_leading(doc))
    return _finish_cloned_section(
        parent,
        key,
        doc=doc,
        target_path=target_path,
        cloned_slots=cloned_slots,
    )


def _finish_cloned_section(
    parent: Container,
    key: str,
    *,
    doc: Document,
    target_path: tuple[str, ...],
    cloned_slots: list[Slot],
) -> Table:
    """Wire a cloned section block under ``parent[key]`` and store it.

    Shared tail of :func:`_install_cloned_section` (clones an existing
    header) and :func:`clone_document_as_section` (synthesises one).
    """
    from tomlrt._container import Table  # noqa: PLC0415

    # Fresh implicit parents have no extent, so fall back to their
    # nearest header-bearing host. A header must also clear any
    # immediately-following KVs it would otherwise capture.
    section = Table.section()
    _install_cloned_structural_block(
        section,
        parent=parent,
        doc=doc,
        target_path=target_path,
        owner=parent._owner_aot_entry,  # noqa: SLF001
        cloned_slots=cloned_slots,
        anchor=_child_header_anchor(parent),
    )
    dict.__setitem__(parent, key, section)
    return section


def clone_aot_entry_as_table(
    parent: Container,
    key: str,
    src_entry_table: Container,
) -> Table:
    """Install an AoT entry under ``parent[key]`` as a standard ``[key]`` table.

    Deep-clones the source entry's slots, rewriting the head from
    ``[[..]]`` to ``[..]`` and rebasing paths from the source AoT prefix
    to ``parent._path + (key,)``.
    """
    src_entry = src_entry_table._owner_aot_entry  # noqa: SLF001
    assert src_entry is not None, "source entry has no owning AoTEntry"
    assert src_entry_table._header_ref is not None  # noqa: SLF001
    # _gather_subtree_slots, not entry.entry_slots, for true doc-stream
    # order (entry_slots is membership order only) and to pull in
    # nested ``[[a.x]]`` entries physically inside this entry's body.
    src_slots = _gather_subtree_slots(src_entry_table)
    return _install_cloned_section(
        parent,
        key,
        src_slots,
        src_entry.path,
        src_entry_table._header_ref.slot,  # noqa: SLF001
    )


def _owned_slots_from(root: Container, start: Slot) -> list[Slot]:
    """Collect ``root``'s owned slots in true doc-stream order.

    ``start`` should be ``root``'s own header/first slot; see
    :func:`_owned_slots_ordered` for why that's usually but not always
    doc-stream-first, and how it's handled either way.
    """
    return _owned_slots_ordered(start, _owned_slot_ids(root))


def _owned_slot_ids(root: Container) -> set[int]:
    """Return the identities of every slot owned by ``root``'s subtree."""
    owned: set[int] = set()
    _collect_subtree(root, [], [], lambda s: owned.add(id(s)))
    return owned


def _gather_subtree_slots(src_table: Container) -> list[Slot]:
    """Collect a container subtree's owned slots in doc-stream order.

    Includes the container's own header, direct/dotted KVs, nested
    sub-sections, and nested ``[[a.x]]`` AoT entries — the entire
    physical body of ``src_table``.
    """
    assert src_table._header_ref is not None  # noqa: SLF001
    return _owned_slots_from(src_table, src_table._header_ref.slot)  # noqa: SLF001


def _gather_headered_subtree_slots(
    src_table: Container,
) -> tuple[StructuralHeaderSlot, list[Slot]]:
    """Collect ``src_table``'s subtree slots plus its own header, by identity.

    ``src_table._header_ref.slot`` — not ``src_slots[0]`` — is the
    container's own header: doc-stream order may put a forward-declared
    nested descendant's header earlier in the returned list.
    """
    src_slots = _gather_subtree_slots(src_table)
    assert src_table._header_ref is not None  # noqa: SLF001
    head = src_table._header_ref.slot  # noqa: SLF001
    assert isinstance(head, StructuralHeaderSlot)
    return head, src_slots


def _hoist_own_slots_first(slots: list[Slot], root_path: tuple[str, ...]) -> list[Slot]:
    """Stable-partition ``slots`` so ``root_path``'s own slots precede nested ones.

    A plain table's own header and direct/dotted keys may legally be
    interleaved with a forward-declared nested descendant's block (see
    :func:`_owned_slots_ordered`) — a table can always be reopened. An
    array-of-tables entry can never be reopened this way (a later plain
    ``[..]`` for an already-``[[..]]``-opened path is invalid TOML), so
    installing a clone as a new AoT entry (:func:`clone_table_as_aot_entry`)
    must gather all of the root's own slots to the front, ahead of any
    nested descendant's content, unlike a plain-section install which
    keeps true doc-stream order.
    """

    def is_own(s: Slot) -> bool:
        return (
            s.host_path == root_path
            if isinstance(s, KVSlot)
            else isinstance(s, StructuralHeaderSlot) and s.path == root_path
        )

    own = [s for s in slots if is_own(s)]
    if own == slots[: len(own)]:
        return slots
    nested = [s for s in slots if not is_own(s)]
    return own + nested


def clone_table_as_aot_entry(
    aot: AoT,
    src_table: Container,
) -> Table:
    """Append ``src_table`` (a standard ``[k]`` section) to ``aot`` as an entry.

    Deep-clones the source section's slots, rewriting the head from
    ``[k]`` to ``[[aot._path]]`` and rebasing paths to ``aot._path``.
    Preserves per-slot leading / EOL / lexeme bytes.
    """
    head, src_slots = _gather_headered_subtree_slots(src_table)
    assert head.kind == "table", (
        "clone_table_as_aot_entry source must be a standard section"
    )
    return _install_cloned_aot_entry(
        aot,
        _hoist_own_slots_first(src_slots, src_table._path),  # noqa: SLF001
        src_table._path,  # noqa: SLF001
        target_path=aot._path,  # noqa: SLF001
        rewrite_separator=True,
    )


def clone_section_as_section(
    parent: Container,
    key: str,
    src_table: Container,
) -> Table:
    """Install a deep clone of a standard section under ``parent[key]``.

    Used for cross-doc assignment and same-doc clone of an attached
    ``[k]`` section. Preserves per-slot trivia and nested sub-sections
    while rebasing paths under ``parent._path + (key,)``.
    """
    head, src_slots = _gather_headered_subtree_slots(src_table)
    return _install_cloned_section(parent, key, src_slots, src_table._path, head)  # noqa: SLF001


def clone_document_as_section(
    parent: Container,
    key: str,
    src_doc: Document,
) -> Table:
    """Install a whole ``Document``'s body under ``parent[key]`` as a section.

    The document is header-less, so a fresh ``[parent.key]`` header is
    synthesised and the document's entire slot stream is cloned verbatim
    beneath it (rebasing paths from the document root to the target). This
    preserves body trivia — standalone comments and inline-array pad —
    that re-synthesising from logical values would drop. The document's
    file-level preamble / epilogue belong to no key and are not carried.
    """
    doc = parent._attached_doc  # noqa: SLF001
    target_path = (*parent._path, key)  # noqa: SLF001

    src_slots: list[Slot] = []
    s = src_doc._head  # noqa: SLF001
    while s is not None:
        src_slots.append(s)
        s = s._next  # noqa: SLF001

    header = _new_section_header(
        target_path,
        leading=_build_section_leading(doc),
        doc=doc,
        owner_aot_entry=parent._owner_aot_entry,  # noqa: SLF001
    )
    # Unlike a cloned header, this one is synthesised here rather than
    # produced by `_clone_entry_slots` (which registers `entry_slots`
    # membership for every slot it clones) — file it explicitly, or an
    # AoT-entry owner never learns this header is part of it.
    _extend_entry_slots(parent._owner_aot_entry, header)  # noqa: SLF001
    cloned_body, _ = _clone_entry_slots(
        src_slots,
        new_entry=None,
        body_owner=parent._owner_aot_entry,  # noqa: SLF001
        src_prefix=src_doc._path,  # noqa: SLF001
        target_prefix=target_path,
        dst_newline=doc._newline,  # noqa: SLF001
    )
    return _finish_cloned_section(
        parent,
        key,
        doc=doc,
        target_path=target_path,
        cloned_slots=[header, *cloned_body],
    )


def _unfile_stale_same_orphan_ancestors(
    value: Container, target_slots: Iterable[Slot]
) -> None:
    """Drop ``value``'s bindings from its old same-orphan ancestor chain.

    A detached orphan retains its internal refs. Moving a nested value
    out must scrub those refs up to, but not beyond, the orphan root.
    Slot back-pointers avoid scanning every ancestor's complete cache.
    """
    old_parent = value._parent  # noqa: SLF001
    if old_parent is None or old_parent._layout_root is not value._layout_root:  # noqa: SLF001
        return  # `value` is the orphan's own root; nothing to clean up.
    direct_key = value._path[len(old_parent._path) :]  # noqa: SLF001
    if len(direct_key) == 1 and direct_key[0] in old_parent:
        dict.__delitem__(old_parent, direct_key[0])

    stale_container_ids: set[int] = set()
    node: Container | None = old_parent
    while node is not None and node._layout_root is value._layout_root:  # noqa: SLF001
        stale_container_ids.add(id(node))
        node = node._parent  # noqa: SLF001
    for slot in target_slots:
        for ref in list(slot._refs):  # noqa: SLF001
            if id(ref.container) in stale_container_ids:
                unfile_ref(ref)


def adopt_private_section(
    dest_parent: Container,
    key: str,
    value: Container,
) -> Container:
    """Rehome a private-orphan section under ``dest_parent[key]`` in place.

    Moves and rebases the existing slot/view subtree, preserving identity
    and trivia. An AoT-entry orphan becomes a plain section; slots owned
    by its stale entry are transferred to the destination's entry.

    Pre-condition (checked by the caller): ``value`` is a header-bearing
    section attached to a private orphan with intact slots.
    """
    doc = dest_parent._attached_doc  # noqa: SLF001
    old_prefix = value._path  # noqa: SLF001
    new_prefix = (*dest_parent._path, key)  # noqa: SLF001
    # The orphan's stale owner: non-None whenever it was itself an AoT
    # entry, *or* nested inside one (e.g. a plain section that lived in
    # the body of an ``[[a]]`` entry now being removed).
    stale_owner = value._owner_aot_entry  # noqa: SLF001
    new_owner = dest_parent._owner_aot_entry  # noqa: SLF001

    assert value._header_ref is not None  # noqa: SLF001
    _, slots = _gather_headered_subtree_slots(value)
    # Nested headers also retain bindings to the orphan's old ancestors.
    nested_headers = [s for s in slots if isinstance(s, StructuralHeaderSlot)]
    _unfile_stale_same_orphan_ancestors(
        value,
        [value._header_ref.slot, *nested_headers],  # noqa: SLF001
    )
    for s in slots:
        _retarget_slot_paths(s, old_prefix, new_prefix, doc._newline)  # noqa: SLF001
        _transfer_stale_owner(s, stale_owner, new_owner)
    _rehome_view_tree(
        value, dest_parent, old_prefix, new_prefix, doc, stale_owner=stale_owner
    )

    assert isinstance(value._header_ref.slot, StructuralHeaderSlot)  # noqa: SLF001
    # A forward-declared descendant may physically precede value's header.
    first = slots[0]
    assert isinstance(first, StructuralHeaderSlot)
    _retarget_header_separator(first, _build_section_leading(doc))
    _splice_block_after(slots, _child_header_anchor(dest_parent), doc)
    _extend_header_bindings_to_root(dest_parent, slots)
    dict.__setitem__(dest_parent, key, value)
    _maybe_demote_synthetic_empty_header(dest_parent)
    return value


def _retarget_slot_paths(
    s: Slot, src_prefix: tuple[str, ...], target_prefix: tuple[str, ...], nl: str
) -> None:
    """Rebase a slot's host / header paths + header render keys, retarget newlines.

    Shared by the deep-clone (:func:`_clone_entry_slots`) and move-in-place
    rehome paths. Owner / AoT-entry handling differs between the two and
    stays at the call site.
    """
    retarget_slot_newlines(s, nl)
    if isinstance(s, KVSlot):
        s.host_path = _rebase_path(s.host_path, src_prefix, target_prefix)
    elif isinstance(s, StructuralHeaderSlot):
        s.path = _rebase_path(s.path, src_prefix, target_prefix)
        s.key_parts = make_keyparts(s.path)
        s.key_seps = ["."] * (len(s.key_parts) - 1)


def _rehome_view_tree(
    root: Container,
    dest_parent: Container,
    old_prefix: tuple[str, ...],
    new_prefix: tuple[str, ...],
    doc: Document,
    *,
    stale_owner: AoTEntry | None = None,
) -> None:
    """Re-point ``root``'s existing view subtree at ``doc`` with rebased paths.

    Slot-backed caches remain valid because their slots are rebased in
    parallel. Views owned by ``stale_owner`` transfer to the destination
    entry; nested AoT entries retain their own owners.
    """
    from tomlrt._array import AoT, Array  # noqa: PLC0415
    from tomlrt._container import Container  # noqa: PLC0415

    new_owner = dest_parent._owner_aot_entry  # noqa: SLF001

    def visit(node: object) -> None:
        if isinstance(node, (Container, AoT)):
            node._layout_root = doc  # noqa: SLF001
            node._path = _rebase_path(node._path, old_prefix, new_prefix)  # noqa: SLF001
            if (
                isinstance(node, Container)
                and stale_owner is not None
                and node._owner_aot_entry is stale_owner  # noqa: SLF001
            ):
                node._owner_aot_entry = new_owner  # noqa: SLF001
        elif isinstance(node, Array):
            node._layout_root = doc  # noqa: SLF001

    root._parent = dest_parent  # noqa: SLF001
    _walk_view_tree(root, visit)


def adopt_private_implicit(
    dest_parent: Container,
    key: str,
    value: Container,
) -> Container:
    """Rehome a header-less (dotted) private-orphan section in place.

    The orphan has no header of its own — its content lives in dotted KVs
    hosted by an ancestor — so it is moved (not rebuilt) under
    ``dest_parent[key]``, re-hosted at the destination's nearest header
    and with its dotted-key prefix rebased, preserving identity and trivia
    (dotted shape, comments, value style). Nested sub-section / AoT headers
    keep their shape.
    """
    doc = dest_parent._attached_doc  # noqa: SLF001
    old_prefix = value._path  # noqa: SLF001
    new_prefix = (*dest_parent._path, key)  # noqa: SLF001
    host = _nearest_header_host(dest_parent)
    host_path = host._path  # noqa: SLF001
    stale_owner = value._owner_aot_entry  # noqa: SLF001
    new_owner = dest_parent._owner_aot_entry  # noqa: SLF001

    # An implicit table always owns at least one slot (an emptied one
    # materialises to an inline table and never reaches here).
    assert value._refs, "implicit orphan has no slots"  # noqa: SLF001
    slots = _owned_slots_from(value, value._refs[0].slot)  # noqa: SLF001
    _unfile_stale_same_orphan_ancestors(value, slots)

    nl = doc._newline  # noqa: SLF001
    for s in slots:
        _rebase_implicit_slot_in_place(s, old_prefix, new_prefix, host_path, nl)
        _transfer_stale_owner(s, stale_owner, new_owner)
    _rehome_view_tree(
        value,
        dest_parent,
        old_prefix,
        new_prefix,
        doc,
        stale_owner=stale_owner,
    )

    # Dotted KVs inherit scope from position, so anchor at the host's
    # direct body rather than its descendant-inclusive subtree.
    anchor = host._body_tail or (  # noqa: SLF001
        host._header_ref.slot if host._header_ref is not None else None  # noqa: SLF001
    )
    if anchor is None and doc._head is not None:  # noqa: SLF001
        # Appending would inherit the scope of an unrelated later header.
        old_head = doc._head  # noqa: SLF001
        insert_before_head(slots[0], doc)
        for prev, s in itertools.pairwise(slots):
            insert_after(prev, s, doc)
        _ensure_leading_blank_line(old_head, doc)
        _terminate_unless_tail(slots[-1], doc)
    else:
        _splice_block_after(slots, anchor, doc)
    # value's own subtree refs travelled intact; re-file only the ancestor
    # binding refs the delete scrubbed: dotted KVs hosted at ``host``
    # propagate up the ``host``-to-``dest_parent`` chain, nested headers
    # propagate all the way to the document root (mirroring the parser's
    # "every ancestor gets a ref" invariant for sections), and KVs under a
    # nested sub-section stay filed within value's subtree.
    chain = _dotted_chain(host, dest_parent)
    for s in slots:
        if isinstance(s, StructuralHeaderSlot):
            _file_header_binding_chain(dest_parent, s)
            continue
        if isinstance(s, KVSlot) and s.host_path != host_path:
            continue
        predecessors = _nearest_filed_predecessors(chain, s)
        for anc, predecessor in zip(chain, predecessors, strict=True):
            _file_ordered_ref(anc, s, predecessor=predecessor)
            maybe_advance_body_tail(anc, s)
    dict.__setitem__(dest_parent, key, value)
    return value


def _rebase_implicit_slot_in_place(
    s: Slot,
    old_prefix: tuple[str, ...],
    new_prefix: tuple[str, ...],
    host_path: tuple[str, ...],
    nl: str,
) -> None:
    """Rebase a header-less section's slot in place.

    A dotted KV hosted *above* the section (its ``host_path`` does not
    start with the section prefix) is re-hosted at ``host_path`` with its
    dotted-key prefix rebased; the within-section key parts keep their
    spelling. KVs under a nested sub-header and nested headers rebase by
    path, exactly as for a header-bearing section.
    """
    if isinstance(s, KVSlot) and s.host_path[: len(old_prefix)] != old_prefix:
        retarget_slot_newlines(s, nl)
        within = (*s.host_path, *s.key)[len(old_prefix) :]
        new_key = (*new_prefix, *within)[len(host_path) :]
        head_n = len(new_key) - len(within)
        s.host_path = host_path
        s.key_parts = (
            make_keyparts(new_key[:head_n])
            + s.key_parts[len(s.key_parts) - len(within) :]
        )
        s.key_seps = ["."] * (len(new_key) - 1)
    else:
        _retarget_slot_paths(s, old_prefix, new_prefix, nl)


def clone_aot(
    parent: Container,
    key: str,
    src_aot: AoT,
) -> AoT:
    """Install ``src_aot`` (an attached AoT) under ``parent[key]``.

    Each entry is deep-cloned with path-rebasing so any nested
    sub-sections stay logically inside the new key.
    """
    from tomlrt._array import AoT  # noqa: PLC0415

    layout_root = parent._attached_doc  # noqa: SLF001
    target_path = (*parent._path, key)  # noqa: SLF001

    new_aot = AoT()
    new_aot._layout_root = layout_root  # noqa: SLF001
    new_aot._path = target_path  # noqa: SLF001
    new_aot._parent = parent  # noqa: SLF001

    dict.__setitem__(parent, key, new_aot)
    for src_entry_table in list(src_aot):
        clone_aot_entry(
            new_aot,
            src_entry_table,
            dst_path=target_path,
            preserve_source_separator=True,
        )
    if len(new_aot) == 0:
        _materialise_empty_aot(new_aot)
    return new_aot


def _clone_entry_slots(
    src_slots: list[Slot],
    *,
    new_entry: AoTEntry | None,
    body_owner: AoTEntry | None,
    src_prefix: tuple[str, ...],
    target_prefix: tuple[str, ...],
    dst_newline: str,
    head: Slot | None = None,
) -> tuple[list[Slot], StructuralHeaderSlot | None]:
    r"""Deep-clone an entry's slot list with path/owner rebasing.

    ``head``, if given, identifies ``src_slots``' own boundary header —
    by identity, not position, since it need not be ``src_slots[0]``
    (see :func:`_owned_slots_ordered`). Its cloned counterpart is
    returned as the second element, with its ``entry`` set to
    ``new_entry`` — so passing ``new_entry=None`` converts an aot-entry
    header to a table header, and passing a non-None ``new_entry`` does
    the inverse. ``head=None`` means the list is body-only (used by
    in-place body replacement that keeps the destination header) and
    the second element is ``None``.

    ``body_owner`` is written to every slot's ``owner_aot_entry`` so
    cloning under another AoT entry keeps physical ownership coherent.
    ``new_entry`` is the AoTEntry the cloned slots are *logically*
    owned by.

    Nested aot-entry headers inside the body keep their AoT shape:
    a fresh `AoTEntry` is allocated per unique source entry found
    in the body, cloned slots are repointed to it, and the
    discriminator (`StructuralHeaderSlot.entry`) is preserved so
    ``_populate_entry_views`` can rebuild the AoT view. Without
    this, cross-doc whole-section copy would downgrade nested
    ``[[a.x]]`` to a duplicated ``[a.x]`` (issue #108).

    ``dst_newline`` is the destination document's line ending; every
    cloned slot's structural-newline trivia is retargeted so a
    cross-document graft does not leave alien line endings behind.
    """
    nested_entry_map: dict[int, AoTEntry] = {}
    if head is not None and new_entry is not None:
        assert isinstance(head, StructuralHeaderSlot)
        if head.entry is not None:
            nested_entry_map[id(head.entry)] = new_entry
    for s in src_slots:
        if s is head or not isinstance(s, StructuralHeaderSlot) or s.entry is None:
            continue
        if id(s.entry) in nested_entry_map:
            continue
        nested_entry_map[id(s.entry)] = AoTEntry()

    cloned: list[Slot] = []
    cloned_head: StructuralHeaderSlot | None = None
    for s in src_slots:
        c: Slot = copy.deepcopy(s)
        c._prev = None  # noqa: SLF001
        c._next = None  # noqa: SLF001
        _retarget_slot_paths(c, src_prefix, target_prefix, dst_newline)
        src_owner = s.owner_aot_entry
        mapped = nested_entry_map.get(id(src_owner)) if src_owner else None
        owner_for_slot = mapped if mapped is not None else body_owner
        c.owner_aot_entry = owner_for_slot
        if isinstance(c, StructuralHeaderSlot):
            assert isinstance(s, StructuralHeaderSlot)
            if s is head:
                # head's kind always comes from new_entry, not from
                # source-entry lookup (which is None for a plain table).
                c.entry = new_entry
            elif s.entry is not None:
                c.entry = nested_entry_map.get(id(s.entry))
        cloned.append(c)
        if s is head:
            assert isinstance(c, StructuralHeaderSlot)
            cloned_head = c
        # Whichever AoT entry ends up owning this slot (``owner_for_slot``,
        # mirroring ``c.owner_aot_entry`` above) must also list it in its
        # own ``entry_slots`` membership — callers like
        # ``remove_aot_entries`` enumerate an entry's owned slots via
        # ``entry_slots``, not by scanning for ``owner_aot_entry``.
        if owner_for_slot is not None:
            owner_for_slot.entry_slots.append(c)

    return cloned, cloned_head


def _rebase_path(
    p: tuple[str, ...],
    src_prefix: tuple[str, ...],
    target_prefix: tuple[str, ...],
) -> tuple[str, ...]:
    """Replace a leading ``src_prefix`` in ``p`` with ``target_prefix``."""
    if src_prefix == target_prefix:
        return p
    if p[: len(src_prefix)] == src_prefix:
        return target_prefix + p[len(src_prefix) :]
    return p


def _populate_entry_views(
    *,
    entry_table: Container,
    cloned_slots: list[Slot],
    target_prefix: tuple[str, ...],
    doc: Document,
) -> None:
    """Build child views from a cloned subtree's non-root slots.

    The same root-relative slot builder handles initial parses and cloned
    structural blocks, so path creation, AoT descent, ref filing, and value
    decoding have one implementation.
    """
    from tomlrt._build import _build_containers  # noqa: PLC0415

    assert entry_table._path == target_prefix  # noqa: SLF001
    assert entry_table._layout_root is doc  # noqa: SLF001
    _build_containers(entry_table, cloned_slots)


def attach_section_at(
    parent: Container,
    sub_path: tuple[str, ...] | list[str],
    source: Table,
) -> Table:
    """Synthesise ``[parent_path.sub_path]`` (multi-component) at end-of-doc.

    Intermediate components in ``sub_path[:-1]`` become implicit tables;
    the deepest component gets the explicit header. ``source`` (always an
    unattached `Table`) is rehomed in place.
    """
    from tomlrt._container import (  # noqa: PLC0415
        _is_synth_inline,
        _synth_value,
    )

    sub = tuple(sub_path)
    assert sub, "attach_section_at requires a non-empty sub_path"
    assert source._layout_root is None, "attach_section_at requires a detached source"  # noqa: SLF001

    doc = parent._attached_doc  # noqa: SLF001
    full_path = (*parent._path, *sub)  # noqa: SLF001

    leading = _build_section_leading(doc)
    owner = parent._owner_aot_entry  # noqa: SLF001
    header = _new_section_header(
        full_path,
        leading=leading,
        doc=doc,
        owner_aot_entry=owner,
    )

    # Build implicit chain: intermediates become header-less Tables
    # living in dict storage; the deepest is where the new explicit
    # header is filed.
    deepest_parent = ensure_implicit_chain(parent, sub[:-1])

    section = source
    pending: list[tuple[str, object]] = list(source.items())
    dict.clear(section)

    _wire_section_container(
        section,
        doc=doc,
        path=full_path,
        parent=deepest_parent,
        owner=owner,
    )
    _file_own_header(section, header)

    # Anchor past the whole subtree of the nearest header-bearing
    # ancestor: a header re-parents everything after it, so landing it
    # mid-section would capture that host's trailing KVs (e.g. a ``d = 4``
    # sibling of an implicit ``parent``) under the new header on re-parse.
    host, anchor = _header_host_and_tail(parent)
    _splice_block_after([header], anchor, doc)
    # Own the new header on the AoT entry so a later delete of the
    # entry takes the promoted section with it.
    _extend_entry_slots(owner, header)

    # File the binding ref under the deepest implicit parent and
    # propagate ancestor-prefix bindings up to the doc root.
    _file_header_binding_chain(deepest_parent, header, host=host)
    dict.__setitem__(deepest_parent, sub[-1], section)

    _maybe_demote_synthetic_empty_header(parent)

    # Process scalars (and synth-inlines) before nested structural
    # children. TOML semantics require all direct KVs of a section to
    # appear before any sub-section header — re-opening a section
    # after a child header is illegal. The ordering is also a defence
    # against an interaction with header demotion: the recursive
    # ``section[k] = v`` path may demote ``section``'s synthetic
    # empty header on its first sub-section attach. Processing
    # scalars first ensures the section's KV body is fully populated
    # (and the header therefore non-empty / not demote-eligible)
    # before any sub-section attach can run.
    scalars: list[tuple[str, object]] = []
    structurals: list[tuple[str, object]] = []
    for k, v in pending:
        if is_scalar(v) or _is_synth_inline(v):
            scalars.append((k, v))
        else:
            structurals.append((k, v))
    for k, v in scalars:
        cst, dec = _synth_value(
            v,
            layout_root=doc,
            parent=section,
            path=(*full_path, k),
            owner=owner,
        )
        append_direct_kv(section, k, cst)
        dict.__setitem__(section, k, dec)
    for k, v in structurals:
        section[k] = v
    return section


def _aot_append_position(aot: AoT) -> tuple[Container, Slot | None]:
    """Return the host and anchor for a newly-appended ``[[path]]`` entry.

    Non-empty AoTs anchor after the last entry's complete subtree,
    including nested AoTs. Empty AoTs anchor in their nearest
    header-bearing host rather than at an unrelated document tail.
    """
    for entry_table in reversed(aot):
        e = entry_table._owner_aot_entry  # noqa: SLF001
        if e is None or not e.entry_slots:
            continue
        return entry_table, _parent_subtree_tail(entry_table)
    parent = aot._parent  # noqa: SLF001
    assert parent is not None, "attached AoT must have a parent"
    # A document-tail anchor could place the first entry under a later sibling.
    return _header_host_and_tail(parent)


_PopT = TypeVar("_PopT")


def _pop_or_remove(lst: list[_PopT], item: _PopT) -> None:
    """O(1) pop if ``item`` is at the tail; else C-level ``list.remove``.

    Both branches are C-implemented; the tail check avoids an
    O(N) scan when the caller is consuming a list in reverse
    (the common case for batched scrubs).
    """
    if lst[-1] is item:
        lst.pop()
    else:
        lst.remove(item)


def unfile_ref(ref: SlotRef) -> None:
    """Remove ``ref`` from its container's ``_refs``/``_index`` and from ``slot._refs``.

    Each affected list uses the tail-fast-path via
    `_pop_or_remove`. Also clears ``container._header_ref`` if the
    ref was the container's own-header ref.
    """
    c = ref.container
    assert not c._inline, "inline containers do not file refs"  # noqa: SLF001
    _pop_or_remove(c._refs, ref)  # noqa: SLF001
    local_key = ref.local_key
    if local_key is None:
        assert c._header_ref is ref  # noqa: SLF001
        c._header_ref = None  # noqa: SLF001
    else:
        bucket = c._index[local_key]  # noqa: SLF001
        _pop_or_remove(bucket, ref)
        if not bucket:
            del c._index[local_key]  # noqa: SLF001
    _pop_or_remove(ref.slot._refs, ref)  # noqa: SLF001


def _scrub_owned_slots_via_backptrs(
    owned: Iterable[Slot],
    *,
    skip_container_ids: frozenset[int] = frozenset(),
) -> None:
    """Remove every live ref to each slot in ``owned`` via slot back-pointers.

    Walks ``slot._refs`` directly (length ≤ path depth, bounded
    independent of doc size) instead of scanning ancestor containers'
    ``_index``/``_refs`` lists.

    ``skip_container_ids`` names containers whose internal refs to
    owned slots should be left in place — the typical caller is
    `delete_key`, which transplants the deleted subtree to a fresh
    orphan doc and needs the subtree containers' internal
    structure intact. The default (empty) is correct for AoT removal,
    which discards the popped entries' containers entirely.
    """
    for s in owned:
        # Snapshot — unfile_ref mutates slot._refs.
        for ref in list(s._refs):  # noqa: SLF001
            if id(ref.container) in skip_container_ids:
                continue
            unfile_ref(ref)


def _norm_aot_index(aot: AoT, index: int) -> int:
    """Normalise ``index`` to non-negative; raise IndexError if out of range."""
    n = len(aot)
    if not -n <= index < n:
        msg = f"AoT index {index} out of range (len {n})"
        raise IndexError(msg)
    return index + n if index < 0 else index


def remove_aot_entry(aot: AoT, index: int) -> Table:
    """Remove ``aot[index]``, unlink its slots, and return it detached.

    Returns the popped entry ``Table`` itself (not a fresh copy), reset
    so it behaves as an unattached, freshly-constructed container —
    mirroring `delete_key`'s orphan-transplant model. User-held
    references remain the same object and are reusable.
    """
    return remove_aot_entries(aot, [_norm_aot_index(aot, index)])[0]


def remove_aot_entries(aot: AoT, indices: Iterable[int]) -> list[Table]:
    """Remove ``aot[i]`` for each ``i`` in ``indices`` in one batch.

    The indices must already be **non-negative, in-range, distinct,
    and ascending**; callers are responsible for normalising. Returns
    the reset popped entry ``Table``s in the same order as ``indices``.

    Scrubbing the union in reverse document order keeps clear and
    slice-delete linear rather than quadratic in sibling count.
    """
    from tomlrt._container import _reset_table_for_rehome  # noqa: PLC0415

    idx_list = list(indices)
    if not idx_list:
        return []
    doc = aot._attached_doc  # noqa: SLF001
    parent = aot._parent  # noqa: SLF001
    assert parent is not None

    # Per-entry: collect the whole subtree in doc-stream order and
    # capture the entry table itself for return / reset.
    owned_per_entry: list[list[Slot]] = []
    popped_entries: list[Table] = []
    union_owned: set[Slot] = set()
    union_owned_ordered: list[Slot] = []  # in doc-stream order

    for i in idx_list:
        entry_table = aot[i]
        owned_ordered = _gather_subtree_slots(entry_table)
        for s in owned_ordered:
            if s not in union_owned:
                union_owned.add(s)
                union_owned_ordered.append(s)
        owned_per_entry.append(owned_ordered)
        popped_entries.append(entry_table)

    # Reverse order lets unfile_ref use the tail fast path in each cache.
    _scrub_owned_slots_via_backptrs(reversed(union_owned_ordered))

    # Body-tail invalidation on the parent chain. Use the same
    # min-depth bound as `delete_key`: the popped slots' min
    # bottom-depth is 0 (every popped AoT entry includes a
    # header), so this walks all the way to the doc root —
    # exactly what we want, since a binding ref to an AoT entry
    # header lives at every prefix container.
    union_owned_ids = {id(s) for s in union_owned}
    _invalidate_body_tail_chain(parent, union_owned_ids)

    for owned in owned_per_entry:
        # Unlink in reverse order so the entry's leftmost slot (the
        # ``[[a]]`` header) goes last — see remove_aot_entry's
        # original comment for the trivia-promotion hazard.
        for slot in reversed(owned):
            unlink_slot(slot, doc)

    # Drop entries from the logical list in reverse so earlier
    # indices stay valid as we go.
    for i in reversed(idx_list):
        list.pop(aot, i)

    # Reset each popped entry in place so it presents as a freshly-
    # constructed unattached Table (matches `delete_key`'s orphan-
    # transplant model). `_layout_root` is set to `doc` momentarily so
    # the recurse-filter in `_reset_table_for_rehome` knows which
    # children belong to this subtree.
    for entry_table in popped_entries:
        entry_table._layout_root = doc  # noqa: SLF001
        _reset_table_for_rehome(entry_table)

    last_key = aot._path[-1]  # noqa: SLF001
    if len(aot) == 0 and not parent._index.get(last_key):  # noqa: SLF001
        parent._index.pop(last_key, None)  # noqa: SLF001
        # An empty AoT still lives in dict storage; a ``key = []``
        # placeholder gives it a physical presence so the document keeps
        # the same semantic shape as the dict view.
        _materialise_empty_aot(aot)

    return popped_entries


def replace_aot_entry_with_clone(
    aot: AoT,
    index: int,
    src_entry_table: Container,
) -> None:
    """Replace ``aot[index]`` with a deep clone of ``src_entry_table``.

    Preserves the *destination* entry header's leading trivia (and
    any pre-header comment block) while replacing the body with a clone
    of the source entry's slots, preserving source per-KV trivia.

    Both entries must be attached AoT-entry tables.
    """
    index = _norm_aot_index(aot, index)

    doc = aot._attached_doc  # noqa: SLF001
    path = aot._path  # noqa: SLF001
    assert path  # invariant of _attached_doc

    dst_entry_table = aot[index]
    if dst_entry_table is src_entry_table:
        return

    dst_entry = dst_entry_table._owner_aot_entry  # noqa: SLF001
    src_entry = src_entry_table._owner_aot_entry  # noqa: SLF001
    assert dst_entry is not None
    assert src_entry is not None

    src_slots = _gather_subtree_slots(src_entry_table)
    # src_slots[0] is guaranteed src_entry_table's own header: an array
    # entry's content can never physically precede its own header (unlike
    # a plain table's, which a nested descendant can forward-declare), so
    # unlike _gather_headered_subtree_slots this never needs identity
    # lookup.
    src_prefix = src_entry.path

    # Save the destination header (we keep it in place, only its body
    # changes). The header's leading carries any pre-header comment
    # block — that's the trivia the test pins.
    dst_header = dst_entry.entry_slots[0]
    assert isinstance(dst_header, StructuralHeaderSlot)

    # Pre-clone source body before any destructive cleanup, so a clone
    # failure can't leave the destination half-emptied. Also covers
    # the source-inside-destination case (e.g. self-nested clone).
    prev_count = len(dst_entry.entry_slots)
    cloned_body = (
        _clone_entry_slots(
            src_slots[1:],
            new_entry=dst_entry,
            body_owner=dst_entry,
            src_prefix=src_prefix,
            target_prefix=path,
            dst_newline=doc._newline,  # noqa: SLF001
        )[0]
        if len(src_slots) > 1
        else []
    )
    # ``_clone_entry_slots`` files dst-owned slots onto
    # ``dst_entry.entry_slots`` straight away. Defer them until after
    # ``clear()`` so it sees the pre-clone state of the entry.
    new_dst_slots = dst_entry.entry_slots[prev_count:]
    del dst_entry.entry_slots[prev_count:]

    # Reuse the structural-delete path to tear down the destination's
    # body: orphans held sub-sections / AoTs into a PrivateRoot,
    # unlinks all body slots from the doc, recomputes tails, and
    # cleans up nested AoTEntry membership. The destination header
    # stays in place because it is not a dict-storage entry.
    dst_entry_table.clear()
    # After clear(), dst_entry.entry_slots should be [dst_header].
    assert dst_entry.entry_slots == [dst_header]

    _splice_block_after(cloned_body, dst_header, doc)
    dst_entry.entry_slots.extend(new_dst_slots)

    # Rebuild views / dict storage from the cloned body.
    _populate_entry_views(
        entry_table=dst_entry_table,
        cloned_slots=cloned_body,
        target_prefix=path,
        doc=doc,
    )


def replace_aot_entry(aot: AoT, index: int, body: Mapping[str, Any]) -> None:
    """Replace ``aot[index]`` in place.

    Keeps the entry's header slot and live `Table` view; just clears
    the body and re-populates from ``body``.

    O(m) in the size of ``body``, independent of AoT length and
    document size. Header position and `_refs` ordering are preserved
    because no slot splicing is involved.
    """
    entry_table = aot[_norm_aot_index(aot, index)]
    items = list(body.items())
    entry_table.clear()
    for k, v in items:
        entry_table[k] = v


def renormalise_aot_order(aot: AoT, new_logical_order: Sequence[Table]) -> None:
    """Re-order an attached AoT's entries to ``new_logical_order``.

    Normalises on reorder: snapshot the slot before the AoT's first
    owned slot, unlink every owned entry slot, then reinsert entries in
    the new order as contiguous blocks.

    ``new_logical_order`` must be a permutation of the AoT's current
    entries (same set of `Table` objects, possibly reordered).
    """
    if len(aot) <= 1:
        # Reverse / sort on 0 or 1 elements is a no-op.
        list.clear(aot)
        for t in new_logical_order:
            list.append(aot, t)
        return
    doc = aot._attached_doc  # noqa: SLF001

    # Collect every non-empty entry's full physical block, in current
    # logical order (which equals physical doc-stream order for AoT
    # entries), and map each surviving Table identity back to its
    # block. Slotless entries have no CST representation and don't
    # participate in physical layout, so they're skipped.
    #
    # A block spans the entry's whole subtree, not just its own
    # header + KV slots: a nested ``[[a.x]]`` lives in its own
    # AoTEntry (outside the parent's ``entry_slots``) and must travel
    # with its parent entry, so that reordering doesn't strand it and
    # re-parent it onto whichever entry lands at its old position.
    physical_blocks: list[list[Slot]] = []
    phys_idx_by_id: dict[int, int] = {}
    for entry_table in aot:
        e = entry_table._owner_aot_entry  # noqa: SLF001
        assert e is not None
        if not e.entry_slots:
            continue
        phys_idx_by_id[id(entry_table)] = len(physical_blocks)
        physical_blocks.append(_gather_subtree_slots(entry_table))

    if physical_blocks:
        region_predecessor = physical_blocks[0][0]._prev  # noqa: SLF001
        region_successor = physical_blocks[-1][-1]._next  # noqa: SLF001
        old_region_slots = _slots_between(doc, region_predecessor, region_successor)

        new_order_indices = [
            phys_idx_by_id[id(t)] for t in new_logical_order if id(t) in phys_idx_by_id
        ]
        output_blocks = [physical_blocks[phys_idx] for phys_idx in new_order_indices]
        movable_slots = [slot for block in physical_blocks for slot in block]
        placements = _peer_placements(physical_blocks, output_blocks)
        _splice_blocks_in_order(doc, movable_slots, placements)
        _finish_region_permutation(
            doc,
            predecessor=region_predecessor,
            successor=region_successor,
            old_slots=old_region_slots,
        )

    # Reflect the new order in the AoT's own list view.
    list.clear(aot)
    for t in new_logical_order:
        list.append(aot, t)


def _slots_between(
    doc: Document,
    predecessor: Slot | None,
    successor: Slot | None,
) -> list[Slot]:
    """Return the open linked-list interval between two stable boundary slots."""
    out: list[Slot] = []
    cur = predecessor._next if predecessor is not None else doc._head  # noqa: SLF001
    while cur is not successor:
        assert cur is not None, "slot interval successor is unreachable"
        out.append(cur)
        cur = cur._next  # noqa: SLF001
    return out


def _ref_projections(
    slots: list[Slot],
) -> dict[int, list[SlotRef]]:
    """Project a doc-ordered slot interval onto every container referencing it."""
    projections: dict[int, list[SlotRef]] = {}
    for slot in slots:
        for ref in slot._refs:  # noqa: SLF001
            projections.setdefault(id(ref.container), []).append(ref)
    return projections


def _replace_ref_projection(
    refs: list[SlotRef],
    old: list[SlotRef],
    new: list[SlotRef],
) -> None:
    """Replace one reordered physical-region projection in doc order."""
    assert old
    assert old != new
    assert len(old) == len(new)
    start = refs.index(old[0])
    end = start + len(old)
    assert refs[start:end] == old, "region projection must be contiguous"
    refs[start:end] = new


def _changed_key_projections(
    old_refs: list[SlotRef],
    new_refs: list[SlotRef],
) -> tuple[dict[str, list[SlotRef]], dict[str, list[SlotRef]]]:
    """Return only keyed projections whose relative order changed."""
    first_by_key: dict[str, SlotRef] = {}
    old_multiple: dict[str, list[SlotRef]] = {}
    for ref in old_refs:
        local_key = ref.local_key
        if local_key is None:
            continue
        first = first_by_key.get(local_key)
        if first is None:
            first_by_key[local_key] = ref
            continue
        old_multiple.setdefault(local_key, [first]).append(ref)

    if not old_multiple:
        return {}, {}

    new_multiple: dict[str, list[SlotRef]] = {key: [] for key in old_multiple}
    for ref in new_refs:
        local_key = ref.local_key
        if local_key is not None and local_key in new_multiple:
            new_multiple[local_key].append(ref)

    old_changed: dict[str, list[SlotRef]] = {}
    new_changed: dict[str, list[SlotRef]] = {}
    for key, old_key_refs in old_multiple.items():
        new_key_refs = new_multiple[key]
        if old_key_refs != new_key_refs:
            old_changed[key] = old_key_refs
            new_changed[key] = new_key_refs
    return old_changed, new_changed


def _reorder_region_refs(
    old_slots: list[Slot],
    new_slots: list[Slot],
) -> None:
    """Apply a physical region's permutation to its ref projections."""
    old_by_container = _ref_projections(old_slots)
    new_by_container = _ref_projections(new_slots)
    assert old_by_container.keys() == new_by_container.keys()

    for container_id, old_refs in old_by_container.items():
        c = old_refs[0].container
        new_refs = new_by_container[container_id]
        if old_refs == new_refs:
            continue
        _replace_ref_projection(
            c._refs,  # noqa: SLF001
            old_refs,
            new_refs,
        )
        old_by_key, new_by_key = _changed_key_projections(old_refs, new_refs)
        for key, old_key_refs in old_by_key.items():
            _replace_ref_projection(
                c._index[key],  # noqa: SLF001
                old_key_refs,
                new_by_key[key],
            )


def _finish_region_permutation(
    doc: Document,
    *,
    predecessor: Slot | None,
    successor: Slot | None,
    old_slots: list[Slot],
) -> None:
    """Validate a physical permutation and apply it to every ref projection."""
    new_slots = _slots_between(doc, predecessor, successor)
    assert {id(s) for s in new_slots} == {id(s) for s in old_slots}
    _reorder_region_refs(old_slots, new_slots)


@dataclass(slots=True)
class _ReorderUnit:
    """One independently sortable slot block and its leading-trivia state."""

    slots: list[Slot]
    key_rank: int
    structural: bool
    mixed: bool
    prefix: Trivia
    remainder: Trivia
    physical_position: int


def _peer_placements(
    physical_blocks: list[list[Slot]], output_blocks: list[list[Slot]]
) -> list[tuple[list[Slot], Trivia]]:
    """Pair peer blocks with positional prefixes and attached remainders."""
    prefixes: list[Trivia] = []
    remainder_by_head: dict[int, Trivia] = {}
    for block in physical_blocks:
        prefix, remainder = _split_leading_for_reorder(block[0])
        prefixes.append(prefix)
        remainder_by_head[id(block[0])] = remainder
    return [
        (
            block,
            Trivia(
                list(prefixes[position].pieces)
                + list(remainder_by_head[id(block[0])].pieces)
            ),
        )
        for position, block in enumerate(output_blocks)
    ]


def _splice_blocks_in_order(
    doc: Document,
    movable_slots: list[Slot],
    placements: list[tuple[list[Slot], Trivia]],
) -> None:
    """Reorder movable layout blocks within the doc-stream.

    ``movable_slots`` is in original physical order. ``placements`` is
    the block grouping, order, and head leading to reinsert; callers may
    split an original logical block when its binding order must change.

    The helper permutes the doc-stream linked list and terminates the
    former final movable slot if it moves into the middle. Other trivia
    policy (positional vs slot-attached) is the caller's responsibility
    — see ``renormalise_aot_order`` (peer-block model) and
    ``reorder_container`` (region-marker model) for the two existing
    flavours.
    """
    if not movable_slots:
        return

    anchor_prev = movable_slots[0]._prev  # noqa: SLF001
    former_region_tail = movable_slots[-1]
    for slot in movable_slots:
        unlink_slot(slot, doc, strip_new_head_leading=False)

    insert_after_slot = anchor_prev
    for block, leading in placements:
        block[0].leading = Trivia(list(leading.pieces))
        for slot in block:
            if insert_after_slot is None:
                insert_before_head(slot, doc)
            else:
                insert_after(insert_after_slot, slot, doc)
            insert_after_slot = slot

    _terminate_unless_tail(former_region_tail, doc)


def _slot_binding_root(slot: Slot) -> tuple[str, ...]:
    """Return the direct binding path represented by ``slot``."""
    if isinstance(slot, StructuralHeaderSlot):
        return tuple(slot.path)
    assert isinstance(slot, KVSlot)
    return (*slot.host_path, slot.key_parts[0].value)


def _binding_run_neighbours(
    parent: Container, key: str
) -> tuple[Slot | None, Slot | None]:
    """Return the slots immediately outside ``parent[key]``'s first physical run.

    Slots in the run can be absent from ``parent._index[key]`` when a
    dotted KV is hosted below ``parent``, so both boundaries are found
    by path rather than by treating the first indexed ref as the run head.
    """
    path_prefix = (*parent._path, key)  # noqa: SLF001
    plen = len(path_prefix)
    primary = _binding_primary_ref(parent, key).slot

    predecessor = primary._prev  # noqa: SLF001
    while (
        predecessor is not None
        and _slot_binding_root(predecessor)[:plen] == path_prefix
    ):
        predecessor = predecessor._prev  # noqa: SLF001

    succ: Slot | None = primary
    while succ is not None and _slot_binding_root(succ)[:plen] == path_prefix:
        succ = succ._next  # noqa: SLF001
    return predecessor, succ


def _binding_primary_ref(parent: Container, key: str) -> SlotRef:
    """Return a binding's direct ref, or its first descendant when implicit."""
    refs = parent._index.get(key)  # noqa: SLF001
    assert refs, "bound key must have refs"
    path = (*parent._path, key)  # noqa: SLF001
    return next(
        (ref for ref in refs if _slot_binding_root(ref.slot) == path),
        refs[0],
    )


def _move_slots_to_anchor(
    parent: Container,
    slots: list[Slot],
    saved_anchor_prev: Slot | None,
    saved_leading_pieces: list[TriviaPiece],
) -> None:
    """Move ``slots`` to ``saved_anchor_prev`` in the doc-stream.

    Splices the contiguous block immediately after ``saved_anchor_prev``
    (or to doc head), applies ``saved_leading_pieces`` to the new head,
    and resorts affected ancestor ``_refs``. Used by the
    ``Container.__setitem__`` position-preserving structural replace
    path.

    ``slots`` must be in doc-stream order and contiguous in the
    linked list (i.e. ``slots[i]._next is slots[i + 1]`` for all i).
    """
    doc = parent._layout_root  # noqa: SLF001
    if doc is None:
        return
    if not slots:
        # Empty AoT or other slotless binding — nothing to move.
        return
    head = slots[0]
    tail = slots[-1]

    if head._prev is saved_anchor_prev:  # noqa: SLF001
        # Already at the saved position — only the leading needs fixing.
        head.leading.pieces = list(saved_leading_pieces)
        _terminate_unless_tail(tail, doc)
        return

    # Detach [head .. tail] from its current position in the linked list.
    p = head._prev  # noqa: SLF001
    n = tail._next  # noqa: SLF001
    if p is not None:
        p._next = n  # noqa: SLF001
    else:
        doc._head = n  # noqa: SLF001
    if n is not None:
        n._prev = p  # noqa: SLF001
    else:
        doc._tail = p  # noqa: SLF001

    # Splice [head .. tail] in after saved_anchor_prev (or at doc head).
    if saved_anchor_prev is None:
        next_after = doc._head  # noqa: SLF001
        head._prev = None  # noqa: SLF001
        tail._next = next_after  # noqa: SLF001
        if next_after is not None:
            next_after._prev = tail  # noqa: SLF001
        else:
            doc._tail = tail  # noqa: SLF001
        doc._head = head  # noqa: SLF001
    else:
        next_after = saved_anchor_prev._next  # noqa: SLF001
        head._prev = saved_anchor_prev  # noqa: SLF001
        saved_anchor_prev._next = head  # noqa: SLF001
        tail._next = next_after  # noqa: SLF001
        if next_after is not None:
            next_after._prev = tail  # noqa: SLF001
        else:
            doc._tail = tail  # noqa: SLF001

    head.leading.pieces = list(saved_leading_pieces)
    _terminate_unless_tail(tail, doc)

    _resort_and_recompute_tails(parent, doc)


def _direct_child_key(
    slot: Slot, parent_path: tuple[str, ...], parent_plen: int
) -> str | None:
    """Return the direct child key of ``parent_path`` that ``slot`` binds, or None.

    Determined by the slot's full binding path: ``path`` for a
    structural header, ``(*host_path, *key_parts)`` for a KV (so
    dotted KVs like ``a.b.c = 1`` are recognised at every prefix
    depth, not just their host). Returns the first path component
    beyond ``parent_path`` if the binding path starts with
    ``parent_path`` and is strictly deeper, else None.
    """
    if isinstance(slot, StructuralHeaderSlot):
        root: tuple[str, ...] = tuple(slot.path)
    else:
        assert isinstance(slot, KVSlot), "unknown slot type"
        root = (*slot.host_path, *slot.key)
    if len(root) > parent_plen and root[:parent_plen] == parent_path:
        return root[parent_plen]
    return None


def reorder_container(c: Container, new_key_order: list[str]) -> None:
    """Reorder ``c``'s direct children to ``new_key_order``.

    ``new_key_order`` is trusted to be a permutation of
    ``dict.keys(c)``. A pure leaf or structural key moves as one block.
    A mixed key splits into leaf and structural units so every mixed
    leaf remains ahead of every section header after sorting. Positional
    separators stay within the corresponding unit kind; attached
    comments travel with their unit.

    Classification visits only subtree-owned slots; the linked-list walk
    is anchored within the subtree and stops once all owned slots are
    found rather than starting unconditionally at document head.

    Non-contiguous keys (e.g. ``[a]; [other]; [a.sub]`` where 'a'
    has two runs at root) are handled by collecting both runs and
    splicing them together. A foreign slot (one belonging to an outer
    scope) interleaved in the owned span is first hoisted to the region
    head — gathering owned blocks across it would shove it past a
    header and silently change its re-parse scope — which both keeps it
    correctly bound and lets the owned span be spliced contiguously.

    If ``c`` owns an explicit (non-synthetic) ``[c]`` header, it moves
    to the start of the reordered region so direct KVs stay bound to
    ``c`` after re-parse.

    When ``c`` is an AoT entry (``c._owner_aot_entry is not None``),
    only slots within ``c``'s own subtree participate (see
    :func:`_owned_slot_ids`): sibling entries with the same path
    are excluded so their content is not merged in, but nested
    descendants — including nested AoT children, which are owned by
    their *own* entry — do participate and move with their key.

    Only mutates the CST; dict storage is the caller's responsibility.
    """
    doc = c._layout_root  # noqa: SLF001
    assert doc is not None

    c_path = c._path  # noqa: SLF001
    c_plen = len(c_path)

    # c's explicit header is the region marker, not a sortable peer: it
    # travels at the splice head so direct KVs keep their binding.
    header_slot: StructuralHeaderSlot | None = None
    header_ref = c._header_ref  # noqa: SLF001
    if header_ref is not None:
        assert isinstance(header_ref.slot, StructuralHeaderSlot)
        # Preserve exactly the synthetic headers demotion would keep:
        # non-table/aot-entry headers, non-synthetic headers, and
        # synthetic table headers that currently bind a KV body. Skipping
        # such a header would splice its body KVs ahead of every header
        # and rebind them to the document root.
        if (
            header_ref.slot.kind != "table"
            or not header_ref.slot.synthetic
            or isinstance(
                header_ref.slot._next,  # noqa: SLF001
                KVSlot,
            )
        ):
            header_slot = header_ref.slot

    # Gather and order only c's subtree. The linked walk spans unrelated
    # slots only when they physically interleave a non-contiguous subtree,
    # where their relative scope must participate in the reorder.
    membership = _owned_slot_ids(c)
    assert c._refs, "attached sortable container must own slots"  # noqa: SLF001
    ordered_slots = _owned_slots_ordered(c._refs[0].slot, membership)  # noqa: SLF001

    key_blocks: dict[str, list[Slot]] = {k: [] for k in new_key_order}
    child_keys_in_phys_order: list[str] = []
    movable_slots: list[Slot] = []

    for cur in ordered_slots:
        is_header = cur is header_slot
        bind_key = None if is_header else _direct_child_key(cur, c_path, c_plen)
        if is_header:
            movable_slots.append(cur)
        elif bind_key is not None and bind_key in key_blocks:
            if not key_blocks[bind_key]:
                child_keys_in_phys_order.append(bind_key)
            key_blocks[bind_key].append(cur)
            movable_slots.append(cur)

    def _is_leaf_slot(slot: Slot, child_path: tuple[str, ...]) -> bool:
        if not isinstance(slot, KVSlot):
            return False
        return slot.host_path[: len(child_path)] != child_path

    if not movable_slots:
        return

    movable_ids = {id(slot) for slot in movable_slots}
    earliest_owned = movable_slots[0]
    latest_owned = movable_slots[-1]
    region_predecessor = earliest_owned._prev  # noqa: SLF001
    region_successor = latest_owned._next  # noqa: SLF001
    old_region_slots = _slots_between(doc, region_predecessor, region_successor)

    # Foreign slots interleaved in c's owned span must keep their
    # re-parse scope. Hoist foreign KVs that still belong to c's
    # containing scope to the region head; stop at a foreign header,
    # which establishes its own scope and would capture c's dotted leaves
    # if hoisted.
    front_foreign: list[Slot] = []
    seen = 1  # earliest_owned itself
    scan: Slot | None = earliest_owned._next  # noqa: SLF001
    while scan is not None and seen < len(movable_ids):
        if id(scan) in movable_ids:
            seen += 1
        elif isinstance(scan, StructuralHeaderSlot):
            break
        else:
            front_foreign.append(scan)
        scan = scan._next  # noqa: SLF001
    if front_foreign:
        head_structural, head_remainder = _split_leading_for_reorder(earliest_owned)
        earliest_owned.leading = Trivia(list(head_remainder.pieces))
        for f in front_foreign:
            unlink_slot(f, doc, strip_new_head_leading=False)
            insert_before(earliest_owned, f, doc)
        front_foreign[0].leading = Trivia(
            list(head_structural.pieces) + list(front_foreign[0].leading.pieces)
        )

    key_rank = {key: rank for rank, key in enumerate(new_key_order)}
    physical_position = {id(slot): pos for pos, slot in enumerate(ordered_slots)}
    units: list[_ReorderUnit] = []
    for key in child_keys_in_phys_order:
        child_path = (*c_path, key)
        leaves: list[Slot] = []
        structural: list[Slot] = []
        for slot in key_blocks[key]:
            (leaves if _is_leaf_slot(slot, child_path) else structural).append(slot)
        mixed = bool(leaves and structural)
        for slots, is_structural in ((leaves, False), (structural, True)):
            if not slots:
                continue
            prefix, remainder = _split_leading_for_reorder(slots[0])
            units.append(
                _ReorderUnit(
                    slots=slots,
                    key_rank=key_rank[key],
                    structural=is_structural,
                    mixed=mixed,
                    prefix=prefix,
                    remainder=remainder,
                    physical_position=physical_position[id(slots[0])],
                )
            )

    header_prefix = Trivia()
    header_remainder = Trivia()
    if header_slot is not None:
        header_prefix, header_remainder = _split_leading_for_reorder(header_slot)
        if header_slot is not earliest_owned:
            first_unit = min(units, key=lambda unit: unit.physical_position)
            header_prefix, first_unit.prefix = first_unit.prefix, header_prefix

    prefixes_by_kind: dict[tuple[bool, bool], list[Trivia]] = {}
    for unit in sorted(units, key=lambda item: item.physical_position):
        prefixes_by_kind.setdefault((unit.structural, unit.mixed), []).append(
            unit.prefix
        )
    prefix_iterators = {
        kind: iter(prefixes) for kind, prefixes in prefixes_by_kind.items()
    }
    output_units = sorted(units, key=lambda unit: (unit.structural, unit.key_rank))

    placements: list[tuple[list[Slot], Trivia]] = []
    if header_slot is not None:
        placements.append(
            (
                [header_slot],
                Trivia(list(header_prefix.pieces) + list(header_remainder.pieces)),
            )
        )
    for unit in output_units:
        prefix = next(prefix_iterators[(unit.structural, unit.mixed)])
        placements.append(
            (
                unit.slots,
                Trivia(list(prefix.pieces) + list(unit.remainder.pieces)),
            )
        )

    _splice_blocks_in_order(doc, movable_slots, placements)

    _finish_region_permutation(
        doc,
        predecessor=region_predecessor,
        successor=region_successor,
        old_slots=old_region_slots,
    )
    moved_ids = movable_ids | {id(s) for s in front_foreign}
    _invalidate_body_tail_chain(c, moved_ids)


__all__ = [
    "add_aot_entry",
    "append_direct_kv",
    "attach_empty_aot",
    "attach_section_at",
    "delete_key",
    "remove_aot_entry",
    "renormalise_aot_order",
    "reorder_container",
    "replace_aot_entry",
    "reposition_install",
]
