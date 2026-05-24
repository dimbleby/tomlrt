"""Section-side mutation primitives.

Linked-list and per-container cache updates for direct (non-dotted)
KV insert and leaf delete. Inline-table mutation lives in
``_inline_ops.py``; this module never touches inline-tables.

Design notes:

* The doc-stream linked list is the single source of physical
  ordering. Insert primitives splice exactly one slot at a time at
  an explicitly-named anchor — no list-index search, no
  doc-stream-wide rescans.

* ``c._refs`` mirrors the doc-stream subset of slots referenced by
  ``c``. For direct (non-dotted) KV inserts, the new ref's correct
  position in ``c._refs`` is **immediately after the anchor's ref**
  (or at the front if there is no anchor ref). This preserves the
  invariant that ``c._subtree_tail`` (a property = ``c._refs[-1]``)
  matches doc-stream order, and avoids drift in layouts where ``c``
  has later child-section refs sitting after its body region.

* ``c._body_tail`` is maintained incrementally: O(1) on insert, and
  O(len(c._refs)) only on the rare delete-of-current-tail.

* No ancestor walk: a non-dotted direct KV files exactly one ref,
  on its host container. Ancestors are unaffected.
"""

from __future__ import annotations

import contextlib
import copy
from typing import TYPE_CHECKING, Any, TypeVar

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
)
from tomlrt._values import make_keyparts

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence

    from tomlrt._array import AoT, Array
    from tomlrt._container import Container, Document, Table
    from tomlrt._slots import Slot
    from tomlrt._trivia import TriviaPiece
    from tomlrt._values import Value


# ---------------------------------------------------------------------------
# Pure linked-list ops
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _record_install(doc: Document) -> Iterator[list[Slot]]:
    """Capture every slot constructed during the with-block.

    Each call to ``_new_kv_slot`` / ``_new_section_header`` inside the
    block (and the one bare ``KVSlot(...)`` in
    ``_append_dotted_kv_under_implicit``) appends to the yielded list,
    in creation order — which matches doc-stream order for every
    install primitive in this module.

    Used by ``reposition_install`` to learn what ``del + set`` just
    installed without having to re-derive it from ``_index`` /
    ``_header_ref`` snapshots. Re-entrancy: nested contexts stack;
    only the innermost is active.
    """
    prev = doc._install_recorder  # noqa: SLF001
    recorder: list[Slot] = []
    doc._install_recorder = recorder  # noqa: SLF001
    try:
        yield recorder
    finally:
        doc._install_recorder = prev  # noqa: SLF001


def _record_new_slot(doc: Document, slot: Slot) -> None:
    """Append ``slot`` to ``doc``'s active install recorder, if any."""
    recorder = doc._install_recorder  # noqa: SLF001
    if recorder is not None:
        recorder.append(slot)


def reposition_install(parent: Container, key: str, value: Any) -> None:
    """Replace ``parent[key]`` while preserving its physical position.

    Snapshots the bound region's anchor, leading, and successor leading;
    deletes the binding; reinstalls via ``parent[key] = value``;
    captures the freshly-built slots through ``_record_install``;
    moves them back to the saved anchor; and restores the successor's
    leading if the moved block still sits immediately before it.

    The successor restore covers leading perturbations from both the
    delete side (``unlink_slot`` stripping a new doc-head's blank
    lines) and the install side (``_synthesise_header_then_insert_kv``
    rewriting the descendant's leading, ``append_direct_kv``'s
    head-of-doc blank-line guard).
    """
    primary_refs = list(parent._index.get(key, ()))  # noqa: SLF001
    saved_anchor_prev: Slot | None = None
    saved_leading_pieces: list[TriviaPiece] = []
    successor_slot: Slot | None = None
    successor_leading: list[TriviaPiece] | None = None
    if primary_refs:
        old_primary = primary_refs[0].slot
        saved_anchor_prev = old_primary._prev  # noqa: SLF001
        saved_leading_pieces = list(old_primary.leading.pieces)
        successor_slot = _find_binding_successor(parent, key)
        if successor_slot is not None:
            successor_leading = list(successor_slot.leading.pieces)
    del parent[key]
    doc = parent._attached_doc  # noqa: SLF001
    with _record_install(doc) as new_slots:
        parent[key] = value
    if not primary_refs or not new_slots:
        # ``not new_slots`` covers slotless bindings (e.g. an empty
        # AoT) — no physical region exists to reposition.
        return
    _move_slots_to_anchor(parent, new_slots, saved_anchor_prev, saved_leading_pieces)
    if (
        successor_slot is not None
        and successor_leading is not None
        and new_slots[-1]._next is successor_slot  # noqa: SLF001
    ):
        successor_slot.leading.pieces = list(successor_leading)


def _ancestor_chain(c: Container) -> list[Container]:
    """Ancestors from ``c._parent`` up to (and including) the document root."""
    out: list[Container] = []
    cur = c._parent  # noqa: SLF001
    while cur is not None:
        out.append(cur)
        cur = cur._parent  # noqa: SLF001
    return out


def _file_ref_at_tail(c: Container, ref: SlotRef) -> None:
    """Append ``ref`` to ``c._refs`` and (when keyed) ``c._index``."""
    c._refs.append(ref)  # noqa: SLF001
    local_key = ref.local_key
    if local_key is not None:
        c._index.setdefault(local_key, []).append(ref)  # noqa: SLF001


def record_ref(c: Container, slot: Slot) -> SlotRef:
    """Create a `SlotRef(slot, c)` and file it at the tail of ``c``'s caches.

    The ``_index`` key, when filed, is :attr:`SlotRef.local_key` —
    derived from ``(slot, container)`` geometry. Callers don't pass
    it, so they can't accidentally file the ref under a key that
    disagrees with the property's derivation.

    Used by ``_build`` (initial population from a parse) and by
    section-clone install paths. For one-off keyed refs constructed
    elsewhere, callers can use ``_file_ref_at_tail`` directly.
    """
    ref = SlotRef(slot, c)
    _file_ref_at_tail(c, ref)
    return ref


def maybe_advance_body_tail(c: Container, slot: Slot) -> None:
    """Advance ``c._body_tail`` if ``slot`` is a body-region KV of ``c``.

    Body-region predicate matches :func:`_is_body_kv`. Used by
    ``_build``'s initial population pass; mutation-time appends
    set ``_body_tail`` directly because they know they are
    appending a body slot by construction.
    """
    if _is_body_kv(c, slot):
        c._body_tail = slot  # noqa: SLF001


def _file_header_binding_chain(
    deepest: Container, header: StructuralHeaderSlot
) -> None:
    """File a binding ref to ``header`` on ``deepest`` and every ancestor.

    Mirrors the parser / ``_build`` behaviour where each strict
    ancestor of a section carries an ``_index`` entry pointing at the
    section's header. Without the ancestor refs, a later demotion of
    a sibling synthetic header (which scrubs ancestor refs to *that*
    header) can leave the ancestor chain with no binding ref into the
    subtree.
    """
    _file_ref_at_tail(deepest, SlotRef(slot=header, container=deepest))
    for anc in _ancestor_chain(deepest):
        _file_ref_at_tail(anc, SlotRef(slot=header, container=anc))


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

    Files ``c``'s own-header ref at ``header_ref_index``, builds and
    inserts the ``key = value`` KV directly after ``header_slot``,
    files the KV ref at ``header_ref_index + 1``, and updates
    ``c._header_ref`` / ``c._index[key]`` / ``c._body_tail``. Returns
    the new KV slot so callers can register it on
    ``owner.entry_slots`` and similar bookkeeping.

    Anchoring (where ``header_slot`` itself sits in the slot stream)
    and ancestor binding-ref filing remain explicit in callers; both
    are highly position-sensitive and not safe to share.
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
    header: StructuralHeaderSlot,
) -> SlotRef:
    """Initialise a freshly-built container as the owner of ``header``.

    Wires ``_layout_root`` / ``_path`` / ``_parent`` / ``_owner_aot_entry``,
    files the own-header ref onto ``c._refs``, and sets ``_header_ref``.
    Returns the filed ref so callers can keep it in scope (e.g. for
    insertion bookkeeping).
    """
    c._wire(layout_root=doc, parent=parent, path=path, owner=owner)  # noqa: SLF001
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


def insert_after(anchor: Slot, new_slot: Slot, doc: Document) -> None:
    """Splice ``new_slot`` immediately after ``anchor`` in ``doc``."""
    nxt = anchor._next  # noqa: SLF001
    new_slot._prev = anchor  # noqa: SLF001
    new_slot._next = nxt  # noqa: SLF001
    anchor._next = new_slot  # noqa: SLF001
    if nxt is not None:
        nxt._prev = new_slot  # noqa: SLF001
    else:
        doc._tail = new_slot  # noqa: SLF001


def insert_before(anchor: Slot, new_slot: Slot, doc: Document) -> None:
    """Splice ``new_slot`` immediately before ``anchor`` in ``doc``."""
    p = anchor._prev  # noqa: SLF001
    new_slot._prev = p  # noqa: SLF001
    new_slot._next = anchor  # noqa: SLF001
    anchor._prev = new_slot  # noqa: SLF001
    if p is not None:
        p._next = new_slot  # noqa: SLF001
    else:
        doc._head = new_slot  # noqa: SLF001


def insert_before_head(new_slot: Slot, doc: Document) -> None:
    """Splice ``new_slot`` at the start of ``doc``'s linked list."""
    # Preamble migration: if the doc was previously slotless and any
    # preamble lives in `_trailing` (e.g. set via `Document.preamble`
    # on an empty doc, or a comment-only source), prepend that trivia
    # to the new head's leading and clear the trailing.
    head = doc._head  # noqa: SLF001
    if head is None and doc._trailing.pieces:  # noqa: SLF001
        nl = doc._newline  # noqa: SLF001
        migrated = list(doc._trailing.pieces)  # noqa: SLF001
        # Add a blank-line separator between preamble and content.
        migrated.append(NewlineNode(nl))
        new_slot.leading.pieces = [*migrated, *new_slot.leading.pieces]
        doc._trailing.pieces = []  # noqa: SLF001
    new_slot._prev = None  # noqa: SLF001
    new_slot._next = head  # noqa: SLF001
    if head is not None:
        head._prev = new_slot  # noqa: SLF001
    else:
        doc._tail = new_slot  # noqa: SLF001
    doc._head = new_slot  # noqa: SLF001


def unlink_slot(
    slot: Slot, doc: Document, *, strip_new_head_leading: bool = True
) -> None:
    """Remove ``slot`` from ``doc``'s linked list (does not touch caches).

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


def append_direct_kv(c: Container, key: str, value: Value) -> None:
    """Append a fresh direct (non-dotted) KV to ``c``.

    Updates ``c._refs`` / ``_index`` / ``_body_tail`` and the dict
    storage.

    Routing:

    * implicit-headerless non-root container with a body anchor →
      dotted-KV synthesis under the nearest header-bearing ancestor;
    * AoT-entry sub-table body → not yet supported;
    * everything else → direct single-keypart KV with anchor =
      body_tail / header_ref / head-of-doc seam.
    """
    if c._kind is _Kind.IMPLICIT_SECTION:  # noqa: SLF001
        # Implicit / headerless non-root container. A fresh
        # ``host_path = c._path`` slot would render in whatever scope
        # the previous header (or the doc root) established, not in
        # ``c``'s logical scope — semantic mismatch. Insert via a
        # dotted KV under the nearest header-bearing ancestor instead.
        if c._body_tail is None:  # noqa: SLF001
            # Implicit ``c`` whose only contributors are descendant
            # headers (e.g. ``[a.b]\ny = 1`` then mutating
            # ``doc.table('a')['x']``). Promote ``c`` to an explicit
            # section by synthesising a ``[c._path]`` header before
            # the first descendant slot, then insert the KV directly
            # under it.
            _synthesise_header_then_insert_kv(c, key, value)
            return
        _append_dotted_kv_under_implicit(c, key, value)
        return
    if c._owner_aot_entry is not None and c._header_ref is not None:  # noqa: SLF001
        # AoT-entry root container: header-bearing, header is `[[arr]]`.
        # Body inserts work like normal header-bearing container, but
        # we also need to maintain the entry's `entry_slots` list in
        # doc-stream order.
        _append_kv_in_aot_entry(c, key, value)
        return
    if c._owner_aot_entry is not None:  # noqa: SLF001
        msg = "insert into AoT entry sub-table body is not yet supported"
        raise NotImplementedError(msg)
    doc = c._attached_doc  # noqa: SLF001
    # Capture the anchor *before* mutating any cache.
    body_tail = c._body_tail  # noqa: SLF001
    header_ref = c._header_ref  # noqa: SLF001

    new_slot = _build_kv_slot(c, key, value, doc)

    if body_tail is not None:
        insert_after(body_tail, new_slot, doc)
    elif header_ref is not None:
        # Header-only container: anchor at the header itself.
        insert_after(header_ref.slot, new_slot, doc)
    elif doc._head is not None:  # noqa: SLF001
        # Section-only doc: insert the new KV at the head of the doc
        # stream, before the first existing slot. Ensure a blank-line
        # separator on what is about to become the second slot, so the
        # new KV does not visually collide with `[s]`.
        old_head = doc._head  # noqa: SLF001
        insert_before_head(new_slot, doc)
        _ensure_leading_blank_line(old_head, doc)
    else:
        # Empty doc (slotless), possibly with preamble trivia in
        # _trailing — insert_before_head migrates that onto the new
        # slot's leading.
        insert_before_head(new_slot, doc)

    new_ref = SlotRef(slot=new_slot, container=c)
    # The new ref's correct position in ``c._refs`` is immediately
    # after the anchor ref (the previous body_tail's ref). With no
    # anchor: head-of-doc insert (3d-5) → index 0, so the ref
    # ordering matches doc-stream (existing section-header refs come
    # after); header-only / empty-doc → end (no preceding refs to
    # order against).
    if body_tail is not None:
        anchor_idx = _find_ref_index_by_slot(c, body_tail)
        c._refs.insert(anchor_idx + 1, new_ref)  # noqa: SLF001
    elif header_ref is None and doc._head is new_slot:  # noqa: SLF001
        # Head-of-doc insert: the new ref must precede any existing
        # section-header refs in c._refs to keep doc-stream order.
        c._refs.insert(0, new_ref)  # noqa: SLF001
    else:
        c._refs.append(new_ref)  # noqa: SLF001
    c._index.setdefault(key, []).append(new_ref)  # noqa: SLF001
    c._body_tail = new_slot  # noqa: SLF001


def _invalidate_body_tail_chain(
    start: Container | None,
    owned_slot_ids: set[int],
    *,
    min_depth: int = 0,
    recompute: bool,
) -> None:
    """Invalidate ``_body_tail`` on the ``start`` → root chain.

    For each container ``cc`` along the chain whose existing
    ``_body_tail`` slot is in ``owned_slot_ids``, either
    recompute the tail (eager) or clear it to ``None`` (lazy —
    next mutation will recompute).

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
            cur._body_tail = (  # noqa: SLF001
                _recompute_body_tail(cur) if recompute else None
            )
        cur = cur._parent  # noqa: SLF001


def delete_key(c: Container, key: str) -> None:
    """Delete ``key`` from ``c`` — scalar, inline, section, AoT, or dotted-subtree.

    Steps:

    1. Compute the owned-slot identity set: every physical slot whose
       refs all live in ``c._index[key]`` plus every descendant
       container's ``_refs``.
    2. Compute the containers-to-scrub set: ``c``'s ancestor chain
       (up to and including the doc root) plus every non-inline
       container in the subtree rooted at ``c[key]``.
    3. Scrub: rebuild ``_refs``/``_index`` of each scrubbed container,
       dropping refs whose slot is in the owned set. Recompute
       ``_body_tail`` on each container whose old tail was unlinked.
       Drop any unlinked slot from its owning ``AoTEntry.entry_slots``.
    4. Debug-only: assert no still-live container retains a ref to any
       owned slot.
    5. Unlink each owned slot from the doc linked list.
    6. Drop the dict entry on ``c``.

    Cascade-prune is intentionally *not* performed: ``del c[k]``
    follows Python-dict semantics, removing exactly ``k`` and leaving
    any now-emptied implicit ancestor chain reachable as nested empty
    ``Table`` views. Such slotless implicit tables have no rendering
    presence (no header_ref, no refs), so dumps stay byte-correct.

    Held views of the deleted subtree retain stale ``_layout_root`` /
    ``_path``; structural mutation through them raises
    ``NotImplementedError`` rather than corrupting the live document.
    """
    if key not in c:
        raise KeyError(key)
    val = dict.__getitem__(c, key)
    doc = c._attached_doc  # noqa: SLF001

    # 1. Owned-slot identity set + retained slot objects (for unlink).
    owned_ids: set[int] = set()
    owned_slots: list[Slot] = []

    def _add_slot(s: Slot) -> None:
        if id(s) in owned_ids:
            return
        owned_ids.add(id(s))
        owned_slots.append(s)

    for r in c._index.get(key, []):  # noqa: SLF001
        _add_slot(r.slot)

    # 2. Subtree containers + AoTs + descendant owned slots.
    subtree_containers: list[Container] = []
    subtree_aots: list[AoT] = []
    _collect_subtree(val, subtree_containers, subtree_aots, _add_slot)

    # 3. Slot-driven scrub via back-pointers, *skipping* subtree
    # containers — those are about to be orphaned to a fresh
    # Document and must keep their internal `_refs` / `_index`
    # intact. Chain containers (ancestors + ``c``) and any other
    # live container holding a ref to an owned slot are scrubbed.
    skip_ids = frozenset(id(sc) for sc in subtree_containers)
    _scrub_owned_slots_via_backptrs(owned_slots, skip_container_ids=skip_ids)

    # 4. Body-tail recompute on the ancestor chain. The
    # `min_owned_depth` bound short-circuits the walk for the
    # common leaf-KV case (chain is just ``c`` itself).
    min_owned_depth = len(c._path)  # noqa: SLF001
    for s in owned_slots:
        d = len(s.host_path) if isinstance(s, KVSlot) else 0
        if d < min_owned_depth:
            min_owned_depth = d
    _invalidate_body_tail_chain(c, owned_ids, min_depth=min_owned_depth, recompute=True)

    # 5. Unlink owned slots; transplant to an orphan Document if any
    # subtree containers / AoTs may still be user-referenced. Skip
    # the entry_slots strip for AoTEntries whose AoT is itself being
    # moved — those entries leave with their slots intact so
    # downstream clone_aot / re-install can read the full CST.
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
    for slot in owned_slots:
        owner = slot.owner_aot_entry
        if (
            owner is not None
            and id(owner) in surviving_aot_entries
            and id(owner) not in moving_aot_entry_ids
        ):
            with contextlib.suppress(ValueError):
                owner.entry_slots.remove(slot)
        unlink_slot(slot, doc)

    if subtree_containers or subtree_aots:
        from tomlrt._container import Document  # noqa: PLC0415

        orphan = Document()
        orphan._newline = doc._newline  # noqa: SLF001
        orphan._is_private = True  # noqa: SLF001
        for slot in owned_slots:
            _splice_at_end(slot, orphan)
        for sc in subtree_containers:
            sc._layout_root = orphan  # noqa: SLF001
        for ao in subtree_aots:
            ao._layout_root = orphan  # noqa: SLF001

    # Detach inline-Container / Array views inside the displaced
    # subtree. Section tables and AoTs are rehomed to a private
    # orphan above (so ``_attached`` reports False via the
    # ``_is_private`` check); inline views don't participate in
    # that orphan-rehome dance, so reset them by hand. Without
    # this, a held reference to a displaced inline table or array
    # keeps a stale ``_layout_root`` / ``_attached``, breaking
    # identity preservation on later re-assignment.
    from tomlrt._container import (  # noqa: PLC0415
        _reset_array_for_rehome,
        _reset_inline_for_rehome,
    )

    displaced_inlines: list[Container] = []
    displaced_arrays: list[Array] = []
    _collect_displaced_inline_views(val, displaced_inlines, displaced_arrays)
    for it in displaced_inlines:
        if it._layout_root is not None:  # noqa: SLF001
            _reset_inline_for_rehome(it)
    for ar in displaced_arrays:
        if ar._attached:  # noqa: SLF001
            _reset_array_for_rehome(ar)

    # 6. Drop the dict entry.
    dict.__delitem__(c, key)


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
    from tomlrt._array import AoT, Array  # noqa: PLC0415
    from tomlrt._container import Container  # noqa: PLC0415

    if isinstance(val, Container):
        if val._inline:  # noqa: SLF001
            inlines_out.append(val)
        for child in val.values():
            _collect_displaced_inline_views(child, inlines_out, arrays_out)
    elif isinstance(val, AoT):
        for entry in val:
            _collect_displaced_inline_views(entry, inlines_out, arrays_out)
    elif isinstance(val, Array):
        arrays_out.append(val)
        for item in val:
            _collect_displaced_inline_views(item, inlines_out, arrays_out)


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
        for entry in val:
            _collect_subtree(entry, containers_out, aots_out, add_slot)


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


def _last_kv(c: Container, predicate: Callable[[KVSlot], bool]) -> KVSlot | None:
    """Reverse-walk ``c._refs`` for the last KVSlot satisfying ``predicate``."""
    for ref in reversed(c._refs):  # noqa: SLF001
        s = ref.slot
        if isinstance(s, KVSlot) and predicate(s):
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


def _is_host_kv(c: Container, s: Slot) -> bool:
    """True iff ``s`` is a KV of ``c`` (any keypart length, same owner)."""
    return (
        isinstance(s, KVSlot)
        and s.host_path == c._path  # noqa: SLF001
        and s.owner_aot_entry is c._owner_aot_entry  # noqa: SLF001
    )


def _is_body_kv(c: Container, s: Slot) -> bool:
    """True iff ``s`` is a body-region KV of ``c``.

    For a header-bearing container, the body is restricted to its own
    host_path; for a header-less container (e.g. an implicit table or
    the document root) any KV with a matching owner counts.
    """
    if not isinstance(s, KVSlot) or s.owner_aot_entry is not c._owner_aot_entry:  # noqa: SLF001
        return False
    has_header = c._kind is _Kind.SECTION  # noqa: SLF001
    return not has_header or s.host_path == c._path  # noqa: SLF001


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
    return _last_kv(c, lambda s: _is_direct_kv(c, s))


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


def _leading_has_blank_line(leading: Trivia) -> bool:
    r"""Whether ``leading`` contains at least one blank physical line.

    A blank line is a line in the leading-trivia stream that contains
    no comment piece. A comment-line newline (e.g. ``# foo\n``) does
    not count as a blank — the newline belongs to the comment.
    """
    has_comment = False
    for p in leading.pieces:
        if isinstance(p, CommentNode):
            has_comment = True
        elif isinstance(p, NewlineNode):
            if not has_comment:
                return True
            has_comment = False
    return False


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
    if prev_leading is None or _leading_has_blank_line(prev_leading):
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
) -> KVSlot:
    """Synthesise a fresh KV slot, recording it on the active install recorder.

    ``key_parts`` and ``key_seps`` are derived from ``key`` in the
    canonical synthetic form (``make_keypart`` per segment, ``.`` as
    separator). Mutation-side construction is the only caller, so the
    parser's source-text-preserving spelling is not needed here.
    """
    slot = KVSlot(
        leading=leading,
        host_path=host_path,
        key_parts=make_keyparts(key),
        key_seps=["."] * (len(key) - 1),
        pre_eq=" ",
        post_eq=" ",
        value=value,
        eol=_default_eol(doc),
        owner_aot_entry=owner,
        key=key,
    )
    _record_new_slot(doc, slot)
    return slot


def _build_kv_slot(c: Container, key: str, value: Value, doc: Document) -> KVSlot:
    """Synthesise a new ``KVSlot`` carrying default trivia + style."""
    # Promote a header without final newline: e.g. user parsed `a = 1`
    # (no trailing newline) and now appends `b = 2`. The anchor's eol
    # must terminate its line before our new slot starts.
    body_tail = c._body_tail  # noqa: SLF001
    header_ref = c._header_ref  # noqa: SLF001
    anchor_slot: Slot | None = body_tail or (
        header_ref.slot if header_ref is not None else None
    )
    if anchor_slot is not None:
        _ensure_terminator(anchor_slot, doc)

    return _new_kv_slot(
        host_path=c._path,  # noqa: SLF001
        key=(key,),
        value=value,
        doc=doc,
        owner=c._owner_aot_entry,  # noqa: SLF001
        leading=_kv_separator_leading(c, doc),
    )


def _append_dotted_kv_under_implicit(c: Container, key: str, value: Value) -> None:
    """Insert into an implicit-headerless container via dotted KV.

    Routes through the nearest header-bearing ancestor (or the doc
    root). Files refs on every implicit ancestor between that host
    and ``c`` per the dotted-KV ref-propagation rule.

    Pre-conditions (checked by caller):
      * ``c._path`` is non-empty (c is not the doc root)
      * ``c._header_ref is None`` (c is implicit-headerless)
      * ``c._body_tail is not None`` (c has at least one dotted-KV
        contributor — anchors the new slot)
    """
    body_tail = c._body_tail  # noqa: SLF001
    assert body_tail is not None
    doc = c._attached_doc  # noqa: SLF001

    # Find host: nearest ancestor with a header, or the doc root.
    host: Container = c
    while host._parent is not None and host._header_ref is None:  # noqa: SLF001
        host = host._parent  # noqa: SLF001

    # Build chain [host, ..., c] via _parent walk + reverse.
    chain: list[Container] = []
    cur: Container | None = c
    while cur is not host:
        assert cur is not None
        chain.append(cur)
        cur = cur._parent  # noqa: SLF001
    chain.append(host)
    chain.reverse()

    # Per-step local keys: from host's path down to (key,).
    local_keys = [*c._path[len(host._path) :], key]  # noqa: SLF001
    assert len(local_keys) == len(chain)

    # AoT consistency: every container in the chain shares the same
    # owner. (The host is either the AoT entry root itself, or the
    # doc root; either way owners match all the way down.)
    owner = c._owner_aot_entry  # noqa: SLF001

    _ensure_terminator(body_tail, doc)

    # Build the dotted slot: keypath = host..key.
    keypath = (*c._path[len(host._path) :], key)  # noqa: SLF001
    new_slot = _new_kv_slot(
        host_path=host._path,  # noqa: SLF001
        key=keypath,
        value=value,
        doc=doc,
        owner=owner,
        leading=_kv_leading_after(_last_kv(host, lambda s: _is_host_kv(host, s)), doc),
    )

    insert_after(body_tail, new_slot, doc)

    # File refs on every chain ancestor. ``_refs`` is the doc-stream
    # subset; ``_index`` preserves "primary at index 0 + all
    # contributors". Appending to ``_index`` keeps any existing
    # structural primary (e.g. a header-owning ref on the host) at
    # index 0; a new dotted contributor is always secondary.
    for i, anc in enumerate(chain):
        new_ref = SlotRef(slot=new_slot, container=anc)
        anchor_idx = _find_ref_index_by_slot(anc, body_tail)
        anc._refs.insert(anchor_idx + 1, new_ref)  # noqa: SLF001
        # ``_index[local_key]`` must equal the doc-stream-ordered
        # subset of ``_refs`` for that key (an invariant). Rebuild it
        # for the affected key rather than blindly appending —
        # appending is only correct when the new ref is also the
        # last with its key in ``_refs``, which fails when the
        # ancestor has later structural-header refs under the same
        # name (e.g. ``a.x = 1`` then ``[a.b]`` — the ``[a.b]``
        # header sits after our new ``a.z`` slot, so the new ref
        # belongs in the middle of ``doc._index['a']``, not the end).
        _rebuild_index_for_key(anc, local_keys[i])
        if anc._body_tail is body_tail:  # noqa: SLF001
            anc._body_tail = new_slot  # noqa: SLF001

    # Maintain AoTEntry.entry_slots in doc-stream order. body_tail is by
    # invariant filed in entry_slots when owner is set.
    if owner is not None:
        anchor_idx = owner.entry_slots.index(body_tail)
        owner.entry_slots.insert(anchor_idx + 1, new_slot)


def _synthesise_header_then_insert_kv(c: Container, key: str, value: Value) -> None:
    """Promote a purely-implicit container ``c`` to an explicit section.

    Pre-conditions (checked by caller):
      * ``c._path`` is non-empty
      * ``c._header_ref is None``
      * ``c._body_tail is None``
      * ``c._refs`` is non-empty (at least one descendant binding ref)

    Synthesises a ``[c._path]`` header before the first descendant
    slot in doc-stream order, adopts that descendant's old leading
    onto the synthetic header (preserving the existing seam-from-
    above), and rewrites the descendant's leading to the document's
    current inter-section separator style (compact or blank-line).
    Then inserts a fresh single-keypart KV ``key = value`` directly
    after the synthetic header.
    """
    doc = c._attached_doc  # noqa: SLF001

    if not c._refs:  # noqa: SLF001
        # No descendants left — typically the position-preserving
        # structural-replace path: the previous binding's slots
        # were just deleted, leaving ``c`` purely implicit and
        # empty. Append at end of doc; the outer caller's
        # ``_move_slots_to_anchor`` will reposition the synthesised
        # block at the captured anchor.
        _synthesise_header_then_insert_kv_at_doc_tail(c, key, value)
        return
    anchor_slot = c._refs[0].slot  # noqa: SLF001
    owner = c._owner_aot_entry  # noqa: SLF001

    # Build the synthetic header. Adopt the descendant's existing
    # leading (so any preamble / inter-section separator that used
    # to land on the descendant lands on the synthetic header
    # instead) and give the descendant a fresh inter-section leading
    # in the doc's current style (compact / blank-separated).
    adopted_leading = anchor_slot.leading
    new_descendant_leading = _build_section_leading(doc)
    header_slot = _new_section_header(
        c._path,  # noqa: SLF001
        leading=adopted_leading,
        doc=doc,
        owner_aot_entry=owner,
    )
    insert_before(anchor_slot, header_slot, doc)
    anchor_slot.leading = new_descendant_leading

    # File the new header's refs:
    #   * own-header ref on c (local_key=None);
    #   * binding refs on every ancestor along c._path.
    # Walk ancestor chain (excluding c) top-down so we can name
    # local_keys correctly.
    ancestors = _ancestor_chain(c)
    # ancestors[0] = c._parent, ..., ancestors[-1] = doc root.
    # local_key on each ancestor is c._path[-(distance from c)] —
    # for ancestor at distance d from c, local_key = c._path[-d].
    for d, anc in enumerate(ancestors, start=1):
        local_key = c._path[-d]  # noqa: SLF001
        binding_ref = SlotRef(slot=header_slot, container=anc)
        anchor_idx_anc = _find_ref_index_by_slot(anc, anchor_slot)
        anc._refs.insert(anchor_idx_anc, binding_ref)  # noqa: SLF001
        # Rebuild _index[local_key] to preserve doc-stream order
        # (binding_ref now sits before the descendant's existing
        # binding ref, so it becomes the primary).
        _rebuild_index_for_key(anc, local_key)

    new_kv = _file_synthetic_header_and_kv(
        c,
        header_slot=header_slot,
        key=key,
        value=value,
        doc=doc,
        owner=owner,
        header_ref_index=0,
    )

    # Maintain the AoT entry's slot list when applicable. anchor_slot is by
    # invariant filed in entry_slots when owner is set.
    if owner is not None:
        anchor_idx = owner.entry_slots.index(anchor_slot)
        owner.entry_slots.insert(anchor_idx, header_slot)
        owner.entry_slots.insert(anchor_idx + 1, new_kv)


def _synthesise_header_then_insert_kv_at_doc_tail(
    c: Container, key: str, value: Value
) -> None:
    """Append ``[c._path]`` + ``key = value`` at the end of the doc.

    Used by the structural-replace path when ``c``'s previous
    contributors were just deleted, leaving ``c`` empty and implicit.
    The outer caller (typically ``_move_slots_to_anchor``) is
    responsible for repositioning the resulting block to the captured
    anchor when one exists.
    """
    doc = c._attached_doc  # noqa: SLF001
    owner = c._owner_aot_entry  # noqa: SLF001

    header_slot = _new_section_header(
        c._path,  # noqa: SLF001
        leading=_build_section_leading(doc),
        doc=doc,
        owner_aot_entry=owner,
    )
    # When ``c`` lives inside an AoT entry, the synthesised header
    # MUST sit physically inside that entry's slot region (before the
    # next sibling [[arr]] header), otherwise a re-parse would
    # attribute it to the next entry. Anchor after the entry's last
    # slot rather than ``doc._tail``.
    if owner is not None and owner.entry_slots:
        anchor = owner.entry_slots[-1]
        insert_after(anchor, header_slot, doc)
    elif doc._tail is None:  # noqa: SLF001
        doc._head = header_slot  # noqa: SLF001
        doc._tail = header_slot  # noqa: SLF001
        # Empty doc → no preceding header → drop the leading.
        header_slot.leading = Trivia()
    else:
        insert_after(doc._tail, header_slot, doc)  # noqa: SLF001

    ancestors = _ancestor_chain(c)
    # When ``c`` lives inside an AoT entry and was anchored after
    # ``owner.entry_slots[-1]`` above, the synthesised header sits
    # in the middle of the doc-stream (between this entry's last
    # slot and the next sibling [[arr]] entry). Each ancestor's
    # ``_refs`` is doc-stream-ordered, so we must INSERT the binding
    # ref at the right position rather than appending. Use the set
    # of slots already known to belong to this entry as the marker:
    # find the position just after the last ref whose slot is in
    # that set, then insert there.
    entry_slot_set: set[Slot] | None = None
    if owner is not None and owner.entry_slots:
        entry_slot_set = set(owner.entry_slots)
    for d, anc in enumerate(ancestors, start=1):
        local_key = c._path[-d]  # noqa: SLF001
        binding_ref = SlotRef(slot=header_slot, container=anc)
        if entry_slot_set is not None:
            insert_idx = len(anc._refs)  # noqa: SLF001
            for i in range(len(anc._refs) - 1, -1, -1):  # noqa: SLF001
                if anc._refs[i].slot in entry_slot_set:  # noqa: SLF001
                    insert_idx = i + 1
                    break
            anc._refs.insert(insert_idx, binding_ref)  # noqa: SLF001
            _rebuild_index_for_key(anc, local_key)
        else:
            _file_ref_at_tail(anc, binding_ref)

    new_kv = _file_synthetic_header_and_kv(
        c,
        header_slot=header_slot,
        key=key,
        value=value,
        doc=doc,
        owner=owner,
        header_ref_index=len(c._refs),  # noqa: SLF001
    )

    if owner is not None:
        owner.entry_slots.append(header_slot)
        owner.entry_slots.append(new_kv)


def _append_kv_in_aot_entry(c: Container, key: str, value: Value) -> None:
    """Append a direct KV in an AoT-entry root container's body.

    Mirrors the header-bearing path in `append_direct_kv` but also
    keeps the entry's `entry_slots` list in doc-stream order.
    """
    doc = c._attached_doc  # noqa: SLF001
    owner = c._owner_aot_entry  # noqa: SLF001
    assert owner is not None
    body_tail = c._body_tail  # noqa: SLF001
    header_ref = c._header_ref  # noqa: SLF001
    assert header_ref is not None

    new_slot = _build_kv_slot(c, key, value, doc)
    anchor: Slot = body_tail if body_tail is not None else header_ref.slot
    _ensure_terminator(anchor, doc)
    insert_after(anchor, new_slot, doc)

    new_ref = SlotRef(slot=new_slot, container=c)
    if body_tail is not None:
        anchor_idx = _find_ref_index_by_slot(c, body_tail)
        c._refs.insert(anchor_idx + 1, new_ref)  # noqa: SLF001
    else:
        c._refs.append(new_ref)  # noqa: SLF001
    c._index.setdefault(key, []).append(new_ref)  # noqa: SLF001
    c._body_tail = new_slot  # noqa: SLF001

    # Maintain entry_slots in doc-stream order. The anchor is by invariant
    # the header_ref slot (no body yet) or a body slot already filed.
    idx = owner.entry_slots.index(anchor)
    owner.entry_slots.insert(idx + 1, new_slot)


def _ensure_terminator(slot: Slot, doc: Document) -> None:
    """Give ``slot`` a trailing newline if it lacks one (no-final-newline doc)."""
    if isinstance(slot, (KVSlot, StructuralHeaderSlot)) and slot.eol.newline is None:
        slot.eol = EolTrivia(
            trailing_ws=slot.eol.trailing_ws,
            comment=slot.eol.comment,
            newline=NewlineNode(text=doc._newline),  # noqa: SLF001
        )


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
    msg = "internal: anchor slot not found in c._refs"  # pragma: no cover
    raise AssertionError(msg)  # pragma: no cover


def _recompute_body_tail(c: Container) -> Slot | None:
    """Last body-region ref's slot in ``c._refs`` (mirrors invariants rule)."""
    found = _last_kv(c, lambda s: _is_body_kv(c, s))
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
    slot = StructuralHeaderSlot(
        leading=leading,
        path=path,
        key_parts=make_keyparts(path),
        key_seps=["."] * (len(path) - 1),
        eol=_default_eol(doc),
        entry=entry,
        owner_aot_entry=owner_aot_entry,
        synthetic=True,
    )
    _record_new_slot(doc, slot)
    return slot


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


def _splice_at_end(slot: Slot, doc: Document) -> None:
    """Insert ``slot`` at the end of the doc-stream."""
    anchor = doc._tail  # noqa: SLF001
    if anchor is None:
        # Empty doc.
        insert_before_head(slot, doc)
    else:
        _ensure_terminator(anchor, doc)
        insert_after(anchor, slot, doc)


def _splice_block_after(slots: list[Slot], anchor: Slot | None, doc: Document) -> None:
    """Splice a contiguous slot block after ``anchor`` or at doc end."""
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
    # The header's physical body is the doc-stream span from the slot
    # after the header up to (but not including) the next structural
    # header or EOF.  Walk it; if any KVSlot lives there, keep the
    # header.
    s = header._next  # noqa: SLF001
    while s is not None and not isinstance(s, StructuralHeaderSlot):
        if isinstance(s, KVSlot):
            return
        s = s._next  # noqa: SLF001
    layout_root = parent._layout_root  # noqa: SLF001
    from tomlrt._container import Document  # noqa: PLC0415

    assert isinstance(layout_root, Document)
    doc = layout_root
    # Remove the header from the doc stream and from all caches.
    # Hand the demoted header's leading trivia (which carries the
    # file preamble when it sits at doc head) off to the successor
    # so comments aren't silently dropped on promotion to implicit.
    successor = header._next  # noqa: SLF001
    unlink_slot(header, doc, strip_new_head_leading=True)
    if successor is not None and header.leading.pieces:
        successor.leading.pieces = [
            *header.leading.pieces,
            *successor.leading.pieces,
        ]
    parent._header_ref = None  # noqa: SLF001
    parent._refs = [r for r in parent._refs if r is not hdr_ref]  # noqa: SLF001
    # Also clear it from any prefix container's _refs / _index.
    grand: Container | None = parent._parent  # noqa: SLF001
    while grand is not None:
        kept = [r for r in grand._refs if r.slot is not header]  # noqa: SLF001
        if len(kept) != len(grand._refs):  # noqa: SLF001
            grand._refs = kept  # noqa: SLF001
            new_index: dict[str, list[SlotRef]] = {}
            for r in kept:
                if r.local_key is not None:
                    new_index.setdefault(r.local_key, []).append(r)
            grand._index = new_index  # noqa: SLF001
        grand = grand._parent  # noqa: SLF001
    # Owner aot-entry, if any, also drops it.
    owner = header.owner_aot_entry
    if owner is not None:
        with contextlib.suppress(ValueError):
            owner.entry_slots.remove(header)


def _split_leading_structural(leading: Trivia) -> tuple[Trivia, Trivia]:
    """Split a leading-trivia stream into (structural-prefix, comment-remainder).

    The structural prefix is the run of whitespace and newline pieces
    before the first comment piece (if any). The remainder starts at
    the first comment piece and includes everything after it. If
    there is no comment piece, the whole leading is structural and
    the remainder is empty.

    Used by AoT reorder: structural separators stay positional;
    comment remainders travel with their entry.
    """
    pieces = leading.pieces
    cut = len(pieces)
    for i, p in enumerate(pieces):
        if isinstance(p, CommentNode):
            cut = i
            break
    return Trivia(list(pieces[:cut])), Trivia(list(pieces[cut:]))


def _retarget_header_separator(
    header: StructuralHeaderSlot,
    new_separator: Trivia,
) -> None:
    """Replace ``header.leading``'s structural prefix while keeping comments.

    A cloned header arrives carrying the source document's spacing
    (blank-line gap, indentation) plus any header-leading comments.
    The destination wants its own spacing convention to drive layout
    but must not lose comments that semantically belong to the
    section being copied.
    """
    _structural, remainder = _split_leading_structural(header.leading)
    header.leading = Trivia([*new_separator.pieces, *remainder.pieces])


def _build_section_leading(doc: Document) -> Trivia:
    """Trivia for a fresh section header.

    Empty doc → no leading; non-empty → mirror the most recent
    structural-header's blank-gap. The first header in the doc is
    treated as having an "implicit blank" peer (its own leading is
    the file preamble, not a separator), so subsequent headers get
    one blank line by default.
    """
    if doc._head is None:  # noqa: SLF001
        return Trivia()
    cur: Slot | None = doc._tail  # noqa: SLF001
    last_header: StructuralHeaderSlot | None = None
    while cur is not None:
        if isinstance(cur, StructuralHeaderSlot):
            last_header = cur
            break
        cur = cur._prev  # noqa: SLF001
    if last_header is None:
        return Trivia([NewlineNode(text=doc._newline)])  # noqa: SLF001
    p: Slot | None = last_header._prev  # noqa: SLF001
    while p is not None:
        if isinstance(p, StructuralHeaderSlot):
            return _peer_separator(last_header.leading, doc)
        p = p._prev  # noqa: SLF001
    # last_header is the first header in the doc; its leading is the
    # preamble, not a peer separator. Treat as no-peer.
    return _peer_separator(None, doc)


def attach_empty_aot(parent: Container, key: str, source_aot: AoT) -> AoT:
    """Bind an empty AoT under ``parent[key]``.

    No physical slots are created; subsequent ``aot.add(...)`` calls
    will materialise the first ``[[path]]`` header. The ``source_aot``
    is rehomed in place (identity preserved).
    """
    if len(source_aot) > 0:  # pragma: no cover
        msg = "non-empty AoT live-attach has its own routing"
        raise AssertionError(msg)
    # Rehome the orphan AoT into this parent's logical scope.
    source_aot._layout_root = parent._layout_root  # noqa: SLF001
    source_aot._path = (*parent._path, key)  # noqa: SLF001
    source_aot._parent = parent  # noqa: SLF001
    return source_aot


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
    entry = AoTEntry(path=path, ordinal=ordinal)
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
        body_items = list(_items_for_synth(body)) if body is not None else []
    _wire_section_container(
        entry_table,
        doc=doc,
        path=path,
        parent=parent,
        owner=entry,
        header=header,
    )

    # Splice header after the last existing AoT-owned slot if any,
    # else at end-of-doc.
    anchor = _last_aot_slot(aot)
    if anchor is None:
        _splice_at_end(header, doc)
    else:
        _ensure_terminator(anchor, doc)
        insert_after(anchor, header, doc)

    _file_header_binding_chain(parent, header)
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
        _append_kv_in_aot_entry(entry_table, k, cst)
        dict.__setitem__(entry_table, k, dec)
    return entry_table


def clone_aot_entry(
    aot: AoT,
    src: Container | AoTEntry,
    *,
    dst_path: tuple[str, ...] | None = None,
) -> Table:
    """Append a deep CST clone of ``src`` to ``aot``.

    ``src`` is either an attached entry's ``Table`` (the live-source
    case used by ``parent[k] = some_entry_table`` and AoT extension)
    or a bare ``AoTEntry`` (the AoT private-orphan rehome path where
    the source table has already been reset but the underlying
    ``AoTEntry.entry_slots`` are intact in a private orphan document).

    Preserves the source entry's per-slot leading / EOL / lexeme bytes
    so per-entry comments and trailing-comment formatting survive.
    The cloned entry's header leading is rewritten to the
    ``_aot_separator`` policy for "next entry", but any post-structural
    comment block on the source header is retained.

    Supports source entries that include nested ``[a.sub]`` headers
    and their KVSlots. ``dst_path`` (defaults to ``aot._path``) lets
    callers rebase the entry under a different key — both the
    entry header path and any nested sub-section paths are rewritten
    so e.g. ``[a.sub]`` becomes ``[b.sub]`` when cloning into ``b``.

    Returns the new ``Table`` view.
    """
    if isinstance(src, AoTEntry):
        src_entry = src
        src_layout_root = None
        src_slots: list[Slot] = list(src_entry.entry_slots)
    else:
        owner = src._owner_aot_entry  # noqa: SLF001
        if owner is None:  # pragma: no cover
            msg = "Source entry has no owning AoTEntry"
            raise RuntimeError(msg)
        src_entry = owner
        src_layout_root = src._layout_root  # noqa: SLF001
        # _gather_subtree_slots (not just entry.entry_slots) so nested
        # ``[[a.x]]`` entries living physically inside this entry's
        # body come along — entry_slots only holds the entry's *own*
        # slots, not those owned by nested AoTEntries.
        src_slots = _gather_subtree_slots(src)

    layout_root = aot._layout_root  # noqa: SLF001
    path = aot._path  # noqa: SLF001
    target_path = dst_path if dst_path is not None else path
    same_aot_clone = target_path == src_entry.path and src_layout_root is layout_root
    return _install_cloned_aot_entry(
        aot,
        src_slots,
        src_entry.path,
        target_path=target_path,
        rewrite_separator=same_aot_clone,
    )


def _install_cloned_structural_block(
    table: Table,
    *,
    parent: Container,
    doc: Document,
    target_path: tuple[str, ...],
    owner: AoTEntry | None,
    cloned_header: StructuralHeaderSlot,
    cloned_slots: list[Slot],
    anchor: Slot | None,
) -> None:
    _wire_section_container(
        table,
        doc=doc,
        path=target_path,
        parent=parent,
        owner=owner,
        header=cloned_header,
    )
    _splice_block_after(cloned_slots, anchor, doc)
    _file_header_binding_chain(parent, cloned_header)
    _populate_entry_views(
        entry_table=table,
        cloned_slots=cloned_slots[1:],
        target_prefix=target_path,
        doc=doc,
    )


def _install_cloned_aot_entry(
    aot: AoT,
    src_slots: list[Slot],
    src_prefix: tuple[str, ...],
    *,
    target_path: tuple[str, ...],
    rewrite_separator: bool,
) -> Table:
    """Common installer for appending a cloned aot-entry to ``aot``.

    Deep-clones ``src_slots`` (head becomes an aot-entry via
    ``new_entry``), wires a fresh entry container, splices the entry's
    slots after the AoT's last slot (or at doc end), files the parent
    binding ref, and populates child views.

    ``rewrite_separator``: if True, the source's structural leading
    is replaced with destination-style preamble (entry 0) or the
    AoT's existing inter-entry separator (entry > 0). If False
    (cross-doc / cross-key clone of an existing AoT entry past the
    first), keep the source's leading verbatim so the original
    inter-entry separator survives.
    """
    from tomlrt._container import Table  # noqa: PLC0415

    parent = aot._parent  # noqa: SLF001
    doc = aot._attached_doc  # noqa: SLF001
    assert parent is not None
    assert target_path

    ordinal = len(aot)
    new_entry = AoTEntry(path=target_path, ordinal=ordinal)

    cloned_slots = _clone_entry_slots(
        src_slots,
        new_entry=new_entry,
        body_owner=new_entry,
        src_prefix=src_prefix,
        target_prefix=target_path,
        dst_newline=doc._newline,  # noqa: SLF001
    )

    head = cloned_slots[0]
    assert isinstance(head, StructuralHeaderSlot)
    cloned_header: StructuralHeaderSlot = head
    if ordinal == 0:
        _retarget_header_separator(cloned_header, _build_section_leading(doc))
    elif rewrite_separator:
        _retarget_header_separator(cloned_header, _aot_separator(aot, doc))
    # else: keep source leading verbatim (cross-doc / cross-key).

    entry_table = Table()
    _install_cloned_structural_block(
        entry_table,
        parent=parent,
        doc=doc,
        target_path=target_path,
        owner=new_entry,
        cloned_header=cloned_header,
        cloned_slots=cloned_slots,
        anchor=_last_aot_slot(aot),
    )

    list.append(aot, entry_table)
    return entry_table


def _install_cloned_section(
    parent: Container,
    key: str,
    src_slots: list[Slot],
    src_prefix: tuple[str, ...],
) -> Table:
    """Common installer for ``parent[key] = <cloned section>``.

    Deep-clones ``src_slots`` (rewriting head from ``[..]`` / ``[[..]]``
    to ``[<key>]``, rebasing paths from ``src_prefix`` to
    ``parent._path + (key,)``), wires the section container, splices
    the slots in at the parent's subtree anchor, files the parent
    binding ref, and populates child views. Used by both
    ``clone_aot_entry_as_table`` and ``clone_section_as_section``.
    """
    from tomlrt._container import Table  # noqa: PLC0415

    layout_root = parent._layout_root  # noqa: SLF001
    if layout_root is None:  # pragma: no cover
        msg = "cloned-section install requires parent attached to a document"
        raise RuntimeError(msg)
    doc = layout_root
    target_path = (*parent._path, key)  # noqa: SLF001

    cloned_slots = _clone_entry_slots(
        src_slots,
        new_entry=None,
        body_owner=parent._owner_aot_entry,  # noqa: SLF001
        src_prefix=src_prefix,
        target_prefix=target_path,
        dst_newline=doc._newline,  # noqa: SLF001
    )

    head = cloned_slots[0]
    assert isinstance(head, StructuralHeaderSlot)
    cloned_header: StructuralHeaderSlot = head
    _retarget_header_separator(cloned_header, _build_section_leading(doc))

    section = Table.section()
    _install_cloned_structural_block(
        section,
        parent=parent,
        doc=doc,
        target_path=target_path,
        owner=parent._owner_aot_entry,  # noqa: SLF001
        cloned_header=cloned_header,
        cloned_slots=cloned_slots,
        anchor=_parent_subtree_tail(parent),
    )

    dict.__setitem__(parent, key, section)
    return section


def clone_aot_entry_as_table(
    parent: Container,
    key: str,
    src_entry_table: Container,
) -> Table:
    """Install an AoT entry under ``parent[key]`` as a standard ``[key]`` table.

    Used by ``parent[key] = some_aot_entry`` and ``install`` paths.
    Deep-clones the source entry's slots, rewriting the head from
    ``[[..]]`` to ``[..]``, rebasing all paths from the source's
    AoT prefix to ``parent._path + (key,)``.
    """
    src_entry = src_entry_table._owner_aot_entry  # noqa: SLF001
    if src_entry is None:  # pragma: no cover
        msg = "Source entry has no owning AoTEntry"
        raise RuntimeError(msg)
    src_slots = list(src_entry.entry_slots)
    return _install_cloned_section(parent, key, src_slots, src_entry.path)


def _gather_subtree_slots(src_table: Container) -> list[Slot]:
    """Collect a container subtree's owned slots in doc-stream order.

    Includes the container's own header, every direct/dotted KV slot,
    every nested sub-section's header + KV slots, and every nested
    ``[[a.x]]`` aot-entry's header + KV slots — i.e. the entire
    physical body of ``src_table``. Works on both standard sections
    and AoT-entry tables (both expose a ``_header_ref``).
    """
    assert src_table._header_ref is not None  # noqa: SLF001

    owned: set[int] = set()

    def _add_slot(s: Slot) -> None:
        owned.add(id(s))

    containers_out: list[Container] = []
    aots_out: list[AoT] = []
    _collect_subtree(src_table, containers_out, aots_out, _add_slot)

    head_slot: Slot = src_table._header_ref.slot  # noqa: SLF001
    out: list[Slot] = [head_slot]
    seen_slots: set[int] = {id(head_slot)}
    cur: Slot | None = head_slot._next  # noqa: SLF001
    while cur is not None and id(cur) in owned:
        if id(cur) not in seen_slots:
            out.append(cur)
            seen_slots.add(id(cur))
        cur = cur._next  # noqa: SLF001
    return out


def clone_table_as_aot_entry(
    aot: AoT,
    src_table: Container,
) -> Table:
    """Append ``src_table`` (a standard ``[k]`` section) to ``aot`` as an entry.

    Deep-clones the source section's slots, rewriting the head from
    ``[k]`` to ``[[aot._path]]``, rebasing all paths from the source's
    section path to ``aot._path``. Preserves per-slot leading / EOL
    / lexeme bytes (so per-section comments survive).
    """
    src_slots = _gather_subtree_slots(src_table)
    if not isinstance(src_slots[0], StructuralHeaderSlot):  # pragma: no cover
        msg = "Source section's first owned slot is not a header"
        raise AssertionError(msg)  # noqa: TRY004
    if src_slots[0].kind != "table":  # pragma: no cover
        msg = "clone_table_as_aot_entry: source must be a standard section"
        raise RuntimeError(msg)
    return _install_cloned_aot_entry(
        aot,
        src_slots,
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

    Used for cross-doc table assignment / same-doc clone of an
    attached standard ``[k]`` section. Preserves per-slot trivia
    (header leading, KV leading / EOL) and any nested sub-sections
    by deep-cloning every owned slot and rebasing paths from
    ``src_table._path`` to ``parent._path + (key,)``.
    """
    src_slots = _gather_subtree_slots(src_table)
    if not isinstance(src_slots[0], StructuralHeaderSlot):  # pragma: no cover
        msg = "Source section's first owned slot is not a header"
        raise AssertionError(msg)  # noqa: TRY004
    return _install_cloned_section(parent, key, src_slots, src_table._path)  # noqa: SLF001


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

    parent_index_present = key in parent._index  # noqa: SLF001
    if not parent_index_present:
        # No physical primary yet (empty AoT placeholder).
        pass
    dict.__setitem__(parent, key, new_aot)
    for src_entry_table in list(src_aot):
        clone_aot_entry(new_aot, src_entry_table, dst_path=target_path)
    return new_aot


def _clone_entry_slots(
    src_slots: list[Slot],
    *,
    new_entry: AoTEntry | None,
    body_owner: AoTEntry | None,
    src_prefix: tuple[str, ...],
    target_prefix: tuple[str, ...],
    dst_newline: str,
    has_header: bool = True,
) -> list[Slot]:
    r"""Deep-clone an entry's slot list with path/owner rebasing.

    When ``has_header`` is True (default), the first slot must be the
    entry header. Its ``entry`` is set to ``new_entry`` — so passing
    ``new_entry=None`` converts an aot-entry header to a table
    header, and passing a non-None ``new_entry`` does the inverse.
    When ``has_header`` is False, the slot list is treated as
    body-only (e.g. for in-place body replacement that keeps the
    destination's existing header).

    ``body_owner`` is written to every slot's ``owner_aot_entry`` (so
    cloning into a table that itself sits under another AoT entry
    keeps physical ownership coherent). ``new_entry`` is the
    AoTEntry the cloned slots are *logically* owned by — used for
    the head's ``entry`` back-pointer and for the ``entry_slots``
    membership list.

    Nested aot-entry headers inside the body keep their AoT shape:
    a fresh `AoTEntry` is allocated per unique source entry found
    in the body, cloned slots are repointed to it, and the
    discriminator (`StructuralHeaderSlot.entry`) is preserved so
    ``_populate_entry_views`` can rebuild the AoT view. Without
    this, cross-doc whole-section copy would downgrade nested
    ``[[a.x]]`` to a duplicated ``[a.x]`` (issue #108).

    ``dst_newline`` is the destination document's line ending; every
    cloned slot's structural-newline trivia is retargeted to it so a
    cross-document graft does not leave alien ``\r\n`` / ``\n``
    pieces behind in the destination's slot stream.
    """
    body_start = 1 if has_header else 0
    nested_entry_map: dict[int, AoTEntry] = {}
    if has_header and new_entry is not None:
        head_src = src_slots[0]
        assert isinstance(head_src, StructuralHeaderSlot)
        if head_src.entry is not None:
            nested_entry_map[id(head_src.entry)] = new_entry
    for s in src_slots[body_start:]:
        if not isinstance(s, StructuralHeaderSlot) or s.entry is None:
            continue
        if id(s.entry) in nested_entry_map:
            continue
        nested_entry_map[id(s.entry)] = AoTEntry(
            path=_rebase_path(s.entry.path, src_prefix, target_prefix),
            ordinal=s.entry.ordinal,
        )

    cloned: list[Slot] = []
    for s in src_slots:
        c: Slot = copy.deepcopy(s)
        c._prev = None  # noqa: SLF001
        c._next = None  # noqa: SLF001
        retarget_slot_newlines(c, dst_newline)
        src_owner = s.owner_aot_entry
        mapped = nested_entry_map.get(id(src_owner)) if src_owner else None
        dst_owner = mapped if mapped is not None else body_owner
        if isinstance(c, KVSlot):
            c.owner_aot_entry = dst_owner
            c.host_path = _rebase_path(c.host_path, src_prefix, target_prefix)
        elif isinstance(c, StructuralHeaderSlot):
            assert isinstance(s, StructuralHeaderSlot)
            c.owner_aot_entry = dst_owner
            c.path = _rebase_path(c.path, src_prefix, target_prefix)
            c.key_parts = make_keyparts(c.path)
            c.key_seps = ["."] * (len(c.key_parts) - 1)
            if s.entry is not None:
                c.entry = nested_entry_map.get(id(s.entry))
        cloned.append(c)
        filing_entry = mapped if mapped is not None else new_entry
        if filing_entry is not None:
            filing_entry.entry_slots.append(c)

    return cloned


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
    """Walk cloned non-header slots, building child Container views.

    Mirrors the parser's slot-builder for an entry: KV slots file
    refs into their host container; sub-section headers create child
    Containers under the entry root and file own-header refs +
    parent-binding refs. Nested ``[[a.b]]`` aot-entry headers are
    handled too: a fresh `AoT` view (and entry `Table`) is created
    or extended at the rebased path so cross-doc graft preserves
    AoT structure.

    Owner inheritance: every newly created container inherits its
    parent's ``_owner_aot_entry``, which matches how the parser
    resolves implicit containers under an entry. The caller wires
    the root ``entry_table`` with the correct owner.

    Decoded Python values are derived from each slot's (already
    deep-cloned) ``Value`` via ``_decode_value`` — never aliased
    from the source dict — so the destination view is fully
    independent of the source.
    """
    from tomlrt._array import AoT  # noqa: PLC0415
    from tomlrt._build import _decode_value  # noqa: PLC0415
    from tomlrt._container import Table  # noqa: PLC0415

    # path -> Container for every container in the entry sub-tree.
    # When a new AoT entry opens, its descendant entries are evicted
    # so re-opened sub-paths (e.g. ``[a.x.sub]`` repeated under each
    # ``[[a.x]]`` entry) resolve to a fresh container per entry.
    containers: dict[tuple[str, ...], Container] = {target_prefix: entry_table}

    def _ensure_container(path: tuple[str, ...]) -> Container:
        if path in containers:
            return containers[path]
        cur: Container = entry_table
        cur_path = target_prefix
        for comp in path[len(target_prefix) :]:
            cur_path = (*cur_path, comp)
            if cur_path in containers:
                cur = containers[cur_path]
                continue
            existing = cur.get(comp)
            if isinstance(existing, AoT) and existing:
                cur = existing[-1]
                containers[cur_path] = cur
                continue
            child = _init_implicit_table(
                doc,
                cur_path,
                cur,
                cur._owner_aot_entry,  # noqa: SLF001
            )
            containers[cur_path] = child
            dict.__setitem__(cur, comp, child)
            cur = child
        return cur

    def _evict_subtree(prefix: tuple[str, ...]) -> None:
        """Drop cached descendants of ``prefix`` (exclusive).

        Called when a new AoT entry opens — the previous entry's
        ``[a.x.sub]`` containers must not be reused by a later
        ``[a.x.sub]`` belonging to the new entry.
        """
        n = len(prefix)
        stale = [p for p in containers if len(p) > n and p[:n] == prefix]
        for p in stale:
            del containers[p]

    for s in cloned_slots:
        if isinstance(s, StructuralHeaderSlot):
            if s.kind == "aot-entry":
                assert s.entry is not None
                aot_path = s.path
                parent_view = _ensure_container(aot_path[:-1])
                name = aot_path[-1]
                aot = parent_view.get(name)
                if aot is None:
                    aot = AoT()
                    aot._layout_root = doc  # noqa: SLF001
                    aot._path = aot_path  # noqa: SLF001
                    aot._parent = parent_view  # noqa: SLF001
                    dict.__setitem__(parent_view, name, aot)
                assert isinstance(aot, AoT)
                ent_table = Table()
                ent_table._wire(  # noqa: SLF001
                    layout_root=doc,
                    parent=parent_view,
                    path=aot_path,
                    owner=s.entry,
                )
                ent_table._header_ref = record_ref(ent_table, s)  # noqa: SLF001
                ent_table._body_tail = s  # noqa: SLF001
                record_ref(parent_view, s)
                list.append(aot, ent_table)
                _evict_subtree(aot_path)
                containers[aot_path] = ent_table
                continue
            assert s.kind == "table"
            container = _ensure_container(s.path)
            container._header_ref = record_ref(container, s)  # noqa: SLF001
            if s.path == target_prefix:
                continue
            parent_view = _ensure_container(s.path[:-1])
            record_ref(parent_view, s)
            continue
        assert isinstance(s, KVSlot)
        host = _ensure_container(s.host_path)
        slot_value = s.value
        host_owner = host._owner_aot_entry  # noqa: SLF001
        if len(s.key_parts) == 1:
            key = s.key_parts[0].value
            kv_ref = SlotRef(slot=s, container=host)
            _file_ref_at_tail(host, kv_ref)
            decoded = _decode_value(
                slot_value,
                layout_root=doc,
                parent=host,
                path=(*host._path, key),  # noqa: SLF001
                owner=host_owner,
            )
            dict.__setitem__(host, key, decoded)
        else:
            cur = host
            for kp in s.key_parts[:-1]:
                comp = kp.value
                ref = SlotRef(slot=s, container=cur)
                _file_ref_at_tail(cur, ref)
                if comp not in cur:
                    sub = _init_implicit_table(
                        doc,
                        (*cur._path, comp),  # noqa: SLF001
                        cur,
                        host_owner,
                    )
                    containers[sub._path] = sub  # noqa: SLF001
                    dict.__setitem__(cur, comp, sub)
                nxt = dict.__getitem__(cur, comp)
                if not isinstance(nxt, Table):  # pragma: no cover
                    msg = "internal: dotted KV traversal hit non-Table"
                    raise AssertionError(msg)  # noqa: TRY004
                cur = nxt
            leaf_key = s.key_parts[-1].value
            kv_ref = SlotRef(slot=s, container=cur)
            _file_ref_at_tail(cur, kv_ref)
            decoded = _decode_value(
                slot_value,
                layout_root=doc,
                parent=cur,
                path=(*cur._path, leaf_key),  # noqa: SLF001
                owner=host_owner,
            )
            dict.__setitem__(cur, leaf_key, decoded)


def attach_section_at(
    parent: Container,
    sub_path: tuple[str, ...] | list[str],
    source: Mapping[str, Any] | Container | None = None,
) -> Table:
    """Synthesise ``[parent_path.sub_path]`` (multi-component) at end-of-doc.

    Intermediate components in ``sub_path[:-1]`` become implicit tables;
    the deepest component gets the explicit header. ``source`` may be
    a `Table` (rehomed) or a Mapping (snapshotted) or ``None``.
    """
    from tomlrt._container import (  # noqa: PLC0415
        Container,
        Table,
        _is_synth_inline,
        _synth_value,
    )

    sub = tuple(sub_path)
    assert sub, "attach_section_at requires a non-empty sub_path"

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

    # Build implicit chain: each intermediate is a Table view living
    # in dict storage but with no own header ref.
    chain: list[Container] = [parent]
    for j, comp in enumerate(sub[:-1]):
        cur = chain[-1]
        if comp in cur:
            nxt = dict.__getitem__(cur, comp)
            if not isinstance(nxt, Container):
                msg = f"intermediate {comp!r} is not a table"
                raise TypeError(msg)
            chain.append(nxt)
            continue
        implicit = _init_implicit_table(
            doc,
            (*parent._path, *sub[: j + 1]),  # noqa: SLF001
            cur,
            owner,
        )
        dict.__setitem__(cur, comp, implicit)
        chain.append(implicit)

    if isinstance(source, Table) and source._layout_root is None:  # noqa: SLF001
        section = source
        pending: list[tuple[str, object]] = list(source.items())
        dict.clear(section)
    else:
        section = Table()
        pending = list(_items_for_synth(source)) if source is not None else []

    _wire_section_container(
        section,
        doc=doc,
        path=full_path,
        parent=chain[-1],
        owner=None,
        header=header,
    )

    _splice_block_after([header], _parent_subtree_tail(parent), doc)
    if owner is not None:
        # Own the new header on the AoT entry so a later delete of the
        # entry takes the promoted section with it.
        owner.entry_slots.append(header)
        section._owner_aot_entry = owner  # noqa: SLF001

    # File the binding ref under the deepest implicit parent and
    # propagate ancestor-prefix bindings up to the doc root.
    deepest_parent = chain[-1]
    _file_header_binding_chain(deepest_parent, header)
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


def _items_for_synth(source: Mapping[str, Any] | Container) -> list[tuple[str, object]]:
    """Iterate items of a Mapping/dict/Container source as (key, value)."""
    return list(source.items())


def _last_aot_slot(aot: AoT) -> Slot | None:
    """Return the last doc-stream slot owned by any entry of ``aot``.

    AoT entries are stored in document order and each entry's
    `entry_slots` list is also in document order, so the answer is
    the last slot of the last entry that has any slots. Walks
    backwards to keep this O(1) in the common case.
    """
    for entry_table in reversed(aot):
        e = entry_table._owner_aot_entry  # noqa: SLF001
        if e is None or not e.entry_slots:
            continue
        return e.entry_slots[-1]
    return None


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
    if not c._inline:  # noqa: SLF001
        _pop_or_remove(c._refs, ref)  # noqa: SLF001
        local_key = ref.local_key
        if local_key is None:
            if c._header_ref is ref:  # noqa: SLF001
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


def remove_aot_entry(aot: AoT, index: int) -> Table:
    """Remove ``aot[index]``, unlink its slots, and return it detached.

    Returns the popped entry ``Table`` itself (not a fresh copy), reset
    so it behaves as an unattached, freshly-constructed container —
    mirroring `delete_key`'s orphan-transplant model. Any user-held
    reference to the entry pre-pop is therefore the same object as the
    return value, and remains usable for reads, mutations, and
    re-attachment.
    """
    n = len(aot)
    if not -n <= index < n:
        msg = f"AoT index {index} out of range (len {n})"
        raise IndexError(msg)
    if index < 0:
        index += n
    return remove_aot_entries(aot, [index])[0]


def remove_aot_entries(aot: AoT, indices: Iterable[int]) -> list[Table]:
    """Remove ``aot[i]`` for each ``i`` in ``indices`` in one batch.

    The indices must already be **non-negative, in-range, distinct,
    and ascending**; callers are responsible for normalising. Returns
    the popped entry ``Table``s themselves (reset for re-use), in the
    same order as ``indices``.

    Batching matters because the per-pop ref-scrub is O(parent
    siblings); doing it once for the union of all popped entries'
    slots makes ``AoT.clear`` and slice-delete linear instead of
    quadratic.
    """
    from tomlrt._array import AoT  # noqa: PLC0415
    from tomlrt._container import (  # noqa: PLC0415
        _is_section,
        _reset_table_for_rehome,
    )

    idx_list = list(indices)
    if not idx_list:
        return []
    doc = aot._attached_doc  # noqa: SLF001
    parent = aot._parent  # noqa: SLF001
    assert parent is not None

    # Per-entry: collect owned slots (entry + nested AoT entry slots)
    # and capture the entry table itself for return / reset.
    owned_per_entry: list[list[Slot]] = []
    popped_entries: list[Table] = []
    union_owned: set[Slot] = set()
    union_owned_ordered: list[Slot] = []  # in doc-stream order

    def _collect_nested_aot_slots(c: Container, sink: list[Slot]) -> None:
        for v in c.values():
            if isinstance(v, AoT):
                for nested_entry_table in v:
                    ne = nested_entry_table._owner_aot_entry  # noqa: SLF001
                    if ne is not None:
                        sink.extend(ne.entry_slots)
                    _collect_nested_aot_slots(nested_entry_table, sink)
            elif _is_section(v):
                _collect_nested_aot_slots(v, sink)

    for i in idx_list:
        entry_table = aot[i]
        e = entry_table._owner_aot_entry  # noqa: SLF001
        assert e is not None
        owned_ordered: list[Slot] = list(e.entry_slots)
        _collect_nested_aot_slots(entry_table, owned_ordered)
        # Dedupe while preserving order, in case nested collection
        # produces overlap with entry_slots.
        seen: set[int] = set()
        deduped: list[Slot] = []
        for s in owned_ordered:
            if id(s) in seen:
                continue
            seen.add(id(s))
            deduped.append(s)
            if s not in union_owned:
                union_owned.add(s)
                union_owned_ordered.append(s)
        owned_per_entry.append(deduped)
        popped_entries.append(entry_table)

    # Slot-driven scrub via back-pointers, in REVERSE doc-stream
    # order so each unfile_ref hits the tail-fast-path of every
    # affected `_refs` / `_index[k]` list. This is what makes the
    # batched case (clear / slice-delete) linear: a parent bucket
    # of N AoT-entry binding refs is emptied tail-pop by tail-pop
    # at C-speed O(1) each, rather than middle-of-bucket
    # O(N) C-removes.
    _scrub_owned_slots_via_backptrs(reversed(union_owned_ordered))

    # Body-tail invalidation on the parent chain. Use the same
    # min-depth bound as `delete_key`: the popped slots' min
    # bottom-depth is 0 (every popped AoT entry includes a
    # header), so this walks all the way to the doc root —
    # exactly what we want, since a binding ref to an AoT entry
    # header lives at every prefix container.
    union_owned_ids = {id(s) for s in union_owned}
    _invalidate_body_tail_chain(parent, union_owned_ids, recompute=True)

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
        _reset_table_for_rehome(entry_table, recurse=True)

    last_key = aot._path[-1]  # noqa: SLF001
    if len(aot) == 0 and not parent._index.get(last_key):  # noqa: SLF001
        parent._index.pop(last_key, None)  # noqa: SLF001

    return popped_entries


def replace_aot_entry_with_clone(
    aot: AoT,
    index: int,
    src_entry_table: Container,
) -> None:
    """Replace ``aot[index]`` with a deep clone of ``src_entry_table``.

    Preserves the *destination* entry header's leading trivia (and
    therefore any pre-header comment block above the original
    ``[[..]]`` line) while replacing the entry's body with a clone
    of the source entry's slots — preserving the source's per-KV
    leading / EOL / lexeme trivia.

    Both entries must be attached AoT-entry tables.
    """
    n = len(aot)
    if not -n <= index < n:
        msg = f"AoT index {index} out of range (len {n})"
        raise IndexError(msg)
    if index < 0:
        index += n

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
            has_header=False,
            dst_newline=doc._newline,  # noqa: SLF001
        )
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


def replace_aot_entry(aot: AoT, index: int, body: Mapping[str, Any] | None) -> None:
    """Replace ``aot[index]`` in place.

    Keeps the entry's header slot and live `Table` view; just clears
    the body and re-populates from ``body``.

    O(m) in the size of ``body``, independent of AoT length and
    document size. Header position and `_refs` ordering are preserved
    by construction (no slot splicing involved).
    """
    n = len(aot)
    if not -n <= index < n:
        msg = f"AoT index {index} out of range (len {n})"
        raise IndexError(msg)
    if index < 0:
        index += n
    entry_table = aot[index]
    if body is entry_table:
        return
    items = list(body.items()) if body is not None else []
    entry_table.clear()
    for k, v in items:
        entry_table[k] = v


def renormalise_aot_order(aot: AoT, new_logical_order: Sequence[Table]) -> None:
    """Re-order an attached AoT's entries to ``new_logical_order``.

    Implements the locked-in "normalise on reorder" policy from the
    plan: snapshot a stable splice anchor (the slot just before the
    AoT's first owned slot in doc-stream); unlink every slot owned
    by any of this AoT's entries; reinsert the entries in the new
    order, each entry as a contiguous block, immediately after the
    anchor.

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

    # Collect every entry's owned slots, in current logical order.
    per_entry_slots: list[list[Slot]] = []
    for entry_table in list(aot):
        e = entry_table._owner_aot_entry  # noqa: SLF001
        assert e is not None
        per_entry_slots.append(list(e.entry_slots))

    # Snapshot the structural-separator part of each entry header's
    # leading, by position. The structural separator (leading blank
    # lines / whitespace before any comment) belongs to the position
    # in the doc; the remainder (comment block + interior whitespace)
    # belongs to the entry and travels with it on reorder.
    structural_by_position: list[Trivia] = []
    remainder_by_entry_id: dict[int, Trivia] = {}
    for i, entry_table in enumerate(list(aot)):
        if not per_entry_slots[i]:
            structural_by_position.append(Trivia())
            remainder_by_entry_id[id(entry_table)] = Trivia()
            continue
        head_slot = per_entry_slots[i][0]
        structural, remainder = _split_leading_structural(head_slot.leading)
        structural_by_position.append(structural)
        remainder_by_entry_id[id(entry_table)] = remainder

    # Earliest owned slot in doc-stream gives us the splice anchor.
    # Walk the doc once, first hit wins. O(N_doc) but predictable
    # and faster than the pairwise back-walk for typical cases.
    owned_ids = {id(s) for slots in per_entry_slots for s in slots}
    earliest_slot: Slot | None = None
    cur = doc._head  # noqa: SLF001
    while cur is not None:
        if id(cur) in owned_ids:
            earliest_slot = cur
            break
        cur = cur._next  # noqa: SLF001
    assert earliest_slot is not None
    anchor_prev = earliest_slot._prev  # noqa: SLF001

    # Unlink every owned slot from the doc-stream linked list. We
    # don't touch refs / index / dict storage — only the linked-list
    # pointers — since the logical mapping doesn't change.
    for slots in per_entry_slots:
        for s in slots:
            unlink_slot(s, doc, strip_new_head_leading=False)

    # Build a per-entry-Table -> entry_slots map (the user-facing
    # `Table`s in new_logical_order may be re-arrangements of the
    # current ones; we need to re-attach via owner_aot_entry).
    slot_blocks: dict[int, list[Slot]] = {
        id(t): per_entry_slots[i] for i, t in enumerate(list(aot))
    }

    # Re-insert entries in new order, each as a contiguous block
    # after `anchor_prev` (or at doc head if anchor_prev is None).
    insert_after_slot = anchor_prev
    for entry_table in new_logical_order:
        block = slot_blocks[id(entry_table)]
        for slot in block:
            if insert_after_slot is None:
                insert_before_head(slot, doc)
            else:
                insert_after(insert_after_slot, slot, doc)
            insert_after_slot = slot

    # Re-apply the structural-separator portion of each new-position
    # entry's header leading from the snapshot (position-keyed),
    # stitched onto that entry's own comment-remainder (entry-keyed).
    for new_pos, entry_table in enumerate(new_logical_order):
        block = slot_blocks[id(entry_table)]
        if not block:
            continue
        head_slot = block[0]
        structural = structural_by_position[new_pos]
        remainder = remainder_by_entry_id[id(entry_table)]
        head_slot.leading = Trivia(list(structural.pieces) + list(remainder.pieces))

    # Reflect the new order in the AoT's own list view.
    list.clear(aot)
    for t in new_logical_order:
        list.append(aot, t)

    # Resort _refs lists on every container in the AoT's parent chain.
    # Each ancestor holds entry-header refs (one per entry, filed under
    # the relevant path component); after splicing, those refs are out
    # of doc-stream order. Sort by slot's new doc-stream position.
    chain: list[Container] = []
    anc: Container | None = aot._parent  # noqa: SLF001
    while anc is not None:
        chain.append(anc)
        anc = anc._parent  # noqa: SLF001
    _resort_refs_by_doc_order(chain, doc)


def _resort_refs_by_doc_order(containers: list[Container], doc: Document) -> None:
    """Resort each container's ``_refs`` and ``_index[k]`` by linked-list position."""
    position: dict[int, int] = {}
    cur = doc._head  # noqa: SLF001
    idx = 0
    while cur is not None:
        position[id(cur)] = idx
        idx += 1
        cur = cur._next  # noqa: SLF001
    for c in containers:
        c._refs.sort(key=lambda r: position.get(id(r.slot), 0))  # noqa: SLF001
        for refs in c._index.values():  # noqa: SLF001
            refs.sort(key=lambda r: position.get(id(r.slot), 0))


def _find_binding_successor(parent: Container, key: str) -> Slot | None:
    """Return the slot just after the first contiguous run bound under ``parent[key]``.

    Walks the doc-stream forward from ``parent._index[key][0].slot``,
    consuming consecutive slots whose binding root starts with
    ``(*parent._path, key)``, and returns the first non-matching slot
    (or ``None`` if the run extends to doc tail).

    Used by ``Container._structural_overwrite`` to capture the original
    successor of the binding's region — its ``leading`` is restored
    after the new binding is moved back to the saved anchor, so that
    the visual gap between the binding and what came after it survives
    the ``del + set`` round-trip.

    The "first contiguous run" choice (rather than "last of all
    matches across the whole doc") matches the semantics of
    ``_move_slots_to_anchor``: the new binding is spliced at the
    saved anchor, so its true post-move successor is the slot that
    originally followed the *first* contiguous run, not the last.
    """
    refs = parent._index.get(key)  # noqa: SLF001
    if not refs:
        return None
    path_prefix = (*parent._path, key)  # noqa: SLF001
    plen = len(path_prefix)
    cur: Slot | None = refs[0].slot
    while cur is not None:
        # The slot's "binding root" is the table-key it owns:
        # ``StructuralHeaderSlot.path`` for headers, and
        # ``(*KVSlot.host_path, key_parts[0].value)`` for KVs (the
        # first dotted-key part is what ``host_path``'s table sees as
        # bound).
        if isinstance(cur, StructuralHeaderSlot):
            root: tuple[str, ...] = tuple(cur.path)
        else:
            assert isinstance(cur, KVSlot)
            root = (*cur.host_path, cur.key_parts[0].value)
        if root[:plen] != path_prefix:
            return cur
        cur = cur._next  # noqa: SLF001
    return None


def _move_slots_to_anchor(
    parent: Container,
    slots: list[Slot],
    saved_anchor_prev: Slot | None,
    saved_leading_pieces: list[TriviaPiece],
) -> None:
    """Move ``slots`` to ``saved_anchor_prev`` in the doc-stream.

    Splices the (contiguous) slot block immediately after
    ``saved_anchor_prev`` (or to doc head if None), applies
    ``saved_leading_pieces`` to the new head, and resorts the
    affected ancestor ``_refs``. Used by the
    ``Container.__setitem__`` position-preserving structural replace
    path: capture old position + leading before the replacement is
    installed at end-of-doc, then move it back.

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

    # Resort ancestor refs by linked-list position; also recompute
    # _body_tail on each (the move may have invalidated the cached
    # tail when the moved slot block was the staging-tail of any
    # ancestor body).
    chain: list[Container] = []
    anc: Container | None = parent
    while anc is not None:
        chain.append(anc)
        anc = anc._parent  # noqa: SLF001
    _resort_refs_by_doc_order(chain, doc)
    for c in chain:
        if c._body_tail is not None:  # noqa: SLF001
            c._body_tail = _recompute_body_tail(c)  # noqa: SLF001


__all__ = [
    "add_aot_entry",
    "append_direct_kv",
    "attach_empty_aot",
    "attach_section_at",
    "delete_key",
    "insert_after",
    "insert_before",
    "insert_before_head",
    "remove_aot_entry",
    "renormalise_aot_order",
    "replace_aot_entry",
    "reposition_install",
    "unlink_slot",
]
