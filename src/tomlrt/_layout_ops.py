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
from tomlrt._values import ArrayValue, InlineTableValue, make_keyparts

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
    """Record every slot spliced into the doc-stream during the with-block.

    The three insertion primitives (:func:`insert_after`,
    :func:`insert_before`, :func:`insert_before_head`) — the sole points
    at which a slot is linked into the document — append to the yielded
    list. Recording at the splice boundary (rather than at slot
    construction) captures synthesised *and* deep-cloned slots uniformly,
    in doc-stream insertion order.

    The block is an install transaction: it only adds slots (the delete
    runs before it, the reinstall never moves existing slots), so the
    record is materialisation, not movement. ``reposition_install`` uses
    it to learn what ``del + set`` installed so it can move that block
    back to the saved anchor. Nested contexts stack; only the innermost
    is active.
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


@contextlib.contextmanager
def _record_displacements(
    doc: Document,
) -> Iterator[list[tuple[Slot, list[TriviaPiece], Slot | None]]]:
    """Capture each pre-existing slot whose ``leading`` the install rewrote.

    ``_synthesise_header_then_insert_kv`` rewrites the leading of the
    descendant it inserts a synthetic header before. When
    ``reposition_install`` relocates that synthesised block, the premise
    can break. Record ``(slot, original_leading_pieces,
    original_predecessor)`` so leading is restored only if ``slot`` ends
    up back beside its original predecessor. Re-entrancy mirrors
    ``_record_install``.
    """
    prev = doc._displaced_recorder  # noqa: SLF001
    recorder: list[tuple[Slot, list[TriviaPiece], Slot | None]] = []
    doc._displaced_recorder = recorder  # noqa: SLF001
    try:
        yield recorder
    finally:
        doc._displaced_recorder = prev  # noqa: SLF001


def reposition_install(parent: Container, key: str, value: Any) -> None:
    """Replace ``parent[key]`` while preserving its physical position.

    The binding is deleted, reinstalled via ``parent[key] = value``,
    captured with ``_record_install`` / ``_record_displacements``, then
    moved back to the saved anchor.

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
    primary_ref = parent._index[key][0]  # noqa: SLF001
    old_primary = primary_ref.slot
    saved_anchor_prev = old_primary._prev  # noqa: SLF001
    saved_leading_pieces = list(old_primary.leading.pieces)
    successor_slot = _find_binding_successor(parent, key)
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
    with _record_install(doc) as new_slots, _record_displacements(doc) as displaced:
        parent._insert_new(  # noqa: SLF001
            key,
            value,
            reinstall_as_dotted=reinstall_as_dotted,
        )
    # An install can record a slot and then unlink it again before the
    # block ends (e.g. a synthetic placeholder header demoted by
    # ``_maybe_demote_synthetic_empty_header``). Drop those orphans:
    # moving one would corrupt the linked list. ``unlink_slot`` repairs
    # the chain, so the survivors stay contiguous.
    survivors = [
        s
        for s in dict.fromkeys(new_slots)  # de-dup, preserve order
        if s is doc._head or s._prev is not None or s._next is not None  # noqa: SLF001
    ]
    if not survivors:
        return
    # The surviving recorded slots must form exactly one contiguous
    # doc-stream span (the reinstall appends them as a block). Recover
    # their true doc order by walking the linked list from the span head,
    # and assert the single-span invariant so a future installer that
    # breaks it fails loudly instead of silently moving the wrong range.
    new_slots = _ordered_recorded_span(survivors)
    # A header-less new binding (scalar / synth-inline) takes its scope
    # from physical position, so it only needs its anchor inside
    # ``parent``'s body region. A binding that brings a structural
    # header — an explicit section / AoT value, *or* a scalar that
    # promoted an implicit ``parent`` to a section — must not be
    # repositioned ahead of a KV it does not own: re-parse would
    # capture that KV under the new header. The block lands after
    # ``saved_anchor_prev``; if the slot that would follow it is a KV
    # outside the block, leave the block where ``_insert_new`` placed
    # it (end of the body) instead.
    if any(isinstance(s, StructuralHeaderSlot) for s in new_slots):
        new_ids = {id(s) for s in new_slots}
        succ = saved_anchor_prev._next if saved_anchor_prev is not None else doc._head  # noqa: SLF001
        while succ is not None and id(succ) in new_ids:
            succ = succ._next  # noqa: SLF001
        anchor_safe = not isinstance(succ, KVSlot)
    else:
        anchor_safe = in_body
    if not anchor_safe:
        return
    _move_slots_to_anchor(parent, new_slots, saved_anchor_prev, saved_leading_pieces)
    # Unified neighbour-leading restore (see docstring): restore each
    # perturbed neighbour's pre-op leading iff the move left it directly
    # after the predecessor that makes that leading correct.
    restores: list[tuple[Slot, list[TriviaPiece], Slot | None]] = list(displaced)
    if successor_slot is not None and successor_leading is not None:
        restores.append((successor_slot, successor_leading, new_slots[-1]))
    for slot, original, expected_pred in restores:
        if slot._prev is expected_pred:  # noqa: SLF001
            slot.leading.pieces = list(original)


def _ordered_recorded_span(survivors: list[Slot]) -> list[Slot]:
    """Return ``survivors`` in doc-stream order, asserting they are contiguous.

    The recorded survivors of an install transaction must form exactly
    one contiguous doc-stream span. Find the span head (the survivor with
    no predecessor in the set) and walk forward; assert that the walk
    reaches every survivor, so a non-contiguous or multi-span record
    fails loudly rather than corrupting the subsequent move.
    """
    ids = {id(s) for s in survivors}
    heads = [
        s
        for s in survivors
        if s._prev is None or id(s._prev) not in ids  # noqa: SLF001
    ]
    assert len(heads) == 1, "recorded install slots are not a single span"
    ordered: list[Slot] = []
    cur: Slot | None = heads[0]
    while cur is not None and id(cur) in ids:
        ordered.append(cur)
        cur = cur._next  # noqa: SLF001
    assert len(ordered) == len(survivors), "recorded install slots are not contiguous"
    return ordered


def _ancestor_chain(c: Container | AoT) -> list[Container]:
    """Ancestors from ``c._parent`` up to (and including) the document root."""
    out: list[Container] = []
    cur = c._parent  # noqa: SLF001
    while cur is not None:
        out.append(cur)
        cur = cur._parent  # noqa: SLF001
    return out


def _resort_and_recompute_tails(c: Container, doc: Document) -> None:
    """Resort ``[c, *ancestors]`` refs by doc order and refresh cached body tails."""
    chain: list[Container] = [c, *_ancestor_chain(c)]
    _resort_refs_by_doc_order(chain, doc)
    for cn in chain:
        if cn._body_tail is not None:  # noqa: SLF001
            cn._body_tail = _recompute_body_tail(cn)  # noqa: SLF001


def _anchor_in_parent_direct_body(parent: Container, anchor_prev: Slot | None) -> bool:
    """True iff a direct KV spliced after ``anchor_prev`` would belong to ``parent``.

    A re-parser attributes a bare ``key = value`` line to whatever
    header is open at its position — the most recent header at or
    before it. So walking the doc-stream backward from ``anchor_prev``,
    the first header encountered must be the binding's host header (or,
    at the document root, none before the stream start). A descendant
    sub-header (``[host.sub]``) or a foreign header would capture the
    KV instead, so the reposition is unsafe and the binding is left at
    its body tail.

    For an implicit (header-less, non-root) container the binding is
    emitted as a dotted key hosted by the nearest header-bearing
    ancestor, so its scope is that host's — check against the host
    header, not the implicit container itself.
    """
    host = _nearest_header_host(parent)
    host_header_ref = host._header_ref  # noqa: SLF001
    host_header = host_header_ref.slot if host_header_ref else None
    cur: Slot | None = anchor_prev
    while cur is not None:
        if isinstance(cur, StructuralHeaderSlot):
            return cur is host_header
        cur = cur._prev  # noqa: SLF001
    # Reached the stream start without a header → document-root scope.
    return host_header is None


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
        record_ref(anc, header)


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


def _file_body_ref(
    anc: Container,
    new_slot: Slot,
    *,
    anchor_slot: Slot | None,
    inserted_at_head: bool,
    local_key: str,
) -> SlotRef:
    """File a ref to ``new_slot`` on ``anc`` at its doc-stream position.

    ``anc._index[local_key]`` is the same-key ordered projection of
    ``anc._refs``; both are spliced in lockstep. The new ref goes after
    ``anchor_slot``'s ref when ``anc`` holds one, else at the head for a
    head-of-doc insert, else at the tail. Maintaining the projection
    incrementally (see :func:`_project_bucket_index`) avoids the O(len(refs))
    rescan a full rebuild would cost on every insert.
    """
    new_ref = SlotRef(slot=new_slot, container=anc)
    assert new_ref.local_key == local_key
    refs = anc._refs  # noqa: SLF001
    if anchor_slot is not None and any(r.container is anc for r in anchor_slot._refs):  # noqa: SLF001
        insert_idx = _find_ref_index_by_slot(anc, anchor_slot) + 1
    elif inserted_at_head and refs:
        insert_idx = 0
    else:
        insert_idx = len(refs)
    bucket = anc._index.setdefault(local_key, [])  # noqa: SLF001
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

    new_slot = _build_kv_slot(c, key, value, doc)

    inserted_at_head = _splice_body_slot(
        new_slot,
        anchor_body_tail=body_tail,
        anchor_header_ref=header_ref,
        doc=doc,
    )
    anchor_slot: Slot | None = body_tail or (
        header_ref.slot if header_ref is not None else None
    )
    _file_body_ref(
        c,
        new_slot,
        anchor_slot=anchor_slot,
        inserted_at_head=inserted_at_head,
        local_key=key,
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
    # Capture doc-stream order *before* the unlink loop severs ``_next``
    # links. ``owned_slots`` is in collection order (every ref bound under
    # ``key`` in ``c`` first — which front-loads nested headers — then the
    # subtree body), not doc-stream order; transplanting in that order
    # would corrupt the orphan's linked list. ``owned_slots[0]`` is the
    # binding's primary slot (``c._index[key][0]``), i.e. the owned set's
    # doc-stream-first slot, from which the forward walk recovers true
    # physical order (skipping any interleaved foreign slots).
    transplanting = bool(subtree_containers or subtree_aots)
    ordered_for_transplant = (
        _owned_slots_in_doc_order(owned_slots[0], owned_ids)
        if transplanting and owned_slots
        else owned_slots
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

    displaced_inlines: list[Container] = []
    displaced_arrays: list[Array] = []
    _collect_displaced_inline_views(val, displaced_inlines, displaced_arrays)
    if transplanting:
        from tomlrt._container import Document  # noqa: PLC0415

        orphan = Document()
        orphan._newline = doc._newline  # noqa: SLF001
        orphan._is_private = True  # noqa: SLF001
        for slot in ordered_for_transplant:
            _splice_at_end(slot, orphan)
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


def _owned_slots_in_doc_order(start: Slot, owned_ids: set[int]) -> list[Slot]:
    """Walk the doc-stream forward from ``start``, collecting owned slots.

    ``start`` must be the owned set's doc-stream-first slot. The walk
    skips interleaved foreign slots (a binding's slots need not be
    contiguous — ``[[a]] … [b] … [[a]]`` is legal) and stops after every
    owned slot has been seen. Used where physical order, not collection
    order, is required.
    """
    out: list[Slot] = []
    seen: set[int] = set()
    cur: Slot | None = start
    while cur is not None and len(seen) < len(owned_ids):
        if id(cur) in owned_ids and id(cur) not in seen:
            out.append(cur)
            seen.add(id(cur))
        cur = cur._next  # noqa: SLF001
    return out


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
) -> KVSlot:
    """Synthesise a fresh KV slot (recorded when spliced, not here).

    ``key_parts`` and ``key_seps`` are derived from ``key`` in the
    canonical synthetic form (``make_keypart`` per segment, ``.`` as
    separator). Mutation-side construction is the only caller, so the
    parser's source-text-preserving spelling is not needed here.
    """
    return KVSlot(
        leading=leading,
        host_path=host_path,
        key_parts=make_keyparts(key),
        key_seps=["."] * (len(key) - 1),
        pre_eq=" ",
        post_eq=" ",
        value=value,
        eol=_default_eol(doc),
        owner_aot_entry=owner,
    )


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


def install_dotted_kv_slot(
    host: Container,
    leaf_keypath: tuple[str, ...],
    value: Value,
    *,
    leaf_parent: Container,
    leading: Trivia | None = None,
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
    if body_tail is not None:
        _ensure_terminator(body_tail, doc)

    new_slot = _new_kv_slot(
        host_path=host._path,  # noqa: SLF001
        key=leaf_keypath,
        value=value,
        doc=doc,
        owner=owner,
        leading=leading
        if leading is not None
        else _kv_leading_after(_last_kv(host), doc),
    )

    inserted_at_head = _splice_body_slot(
        new_slot,
        anchor_body_tail=body_tail,
        anchor_header_ref=header_ref,
        doc=doc,
    )

    # File a ref on every chain ancestor, anchored at host's body_tail
    # or header (fresh implicit intermediates have empty ``_refs``).
    anchor_slot: Slot | None = body_tail or (
        header_ref.slot if header_ref is not None else None
    )
    for i, anc in enumerate(chain):
        _file_body_ref(
            anc,
            new_slot,
            anchor_slot=anchor_slot,
            inserted_at_head=inserted_at_head,
            local_key=leaf_keypath[i],
        )
        if anc._body_tail is body_tail or anc._body_tail is None:  # noqa: SLF001
            # Propagate the body_tail bump up the chain. ``is body_tail``
            # is the old-path case (every chain ancestor of an implicit
            # ``c`` tracks the same body_tail as ``c``); ``is None`` is
            # the fresh-intermediate case (``ensure_implicit_chain``
            # may have minted new implicits below host whose own
            # ``_body_tail`` is still empty).
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
    entry_slot_set: set[Slot] | None = None

    if anchor_slot is not None:
        adopted_leading = anchor_slot.leading
        original_pred = anchor_slot._prev  # noqa: SLF001
        new_descendant_leading = _build_section_leading(doc)
        header_slot = _new_owned_section_header(c, leading=adopted_leading, doc=doc)
        insert_before(anchor_slot, header_slot, doc)
        recorder = doc._displaced_recorder  # noqa: SLF001
        if recorder is not None:
            recorder.append(
                (anchor_slot, list(anchor_slot.leading.pieces), original_pred)
            )
        anchor_slot.leading = new_descendant_leading
    else:
        header_slot = _new_owned_section_header(
            c, leading=_build_section_leading(doc), doc=doc
        )
        # Keep a new sub-section physically inside its owning AoT entry;
        # otherwise append at document tail for the caller to reposition.
        entry_last = _entry_last_slot(owner) if owner is not None else None
        if entry_last is not None:
            insert_after(entry_last, header_slot, doc)
        elif doc._tail is None:  # noqa: SLF001
            insert_before_head(header_slot, doc)
            header_slot.leading = Trivia()
        else:
            insert_after(doc._tail, header_slot, doc)  # noqa: SLF001
        if isinstance(header_slot._prev, StructuralHeaderSlot):  # noqa: SLF001
            header_slot.leading = Trivia()
        if owner is not None and owner.entry_slots:
            entry_slot_set = set(owner.entry_slots)

    # File a binding ref for the new header on every ancestor along
    # c._path. The ancestor d levels above c is keyed by c._path[-d].
    for d, anc in enumerate(_ancestor_chain(c), start=1):
        local_key = c._path[-d]  # noqa: SLF001
        binding_ref = SlotRef(slot=header_slot, container=anc)
        if anchor_slot is not None:
            insert_idx = _find_ref_index_by_slot(anc, anchor_slot)
            anc._refs.insert(insert_idx, binding_ref)  # noqa: SLF001
            _rebuild_index_for_key(anc, local_key)
        elif entry_slot_set is not None:
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
        # Empty doc: this is also the first slot, so any preamble
        # parked in `_trailing` migrates onto its leading.
        insert_before_head(slot, doc)
        _promote_trailing_to_preamble(doc)
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
    if len(source_aot) > 0:  # pragma: no cover
        msg = "non-empty AoT live-attach has its own routing"
        raise AssertionError(msg)
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
    if not (isinstance(slot, KVSlot) and isinstance(slot.value, ArrayValue)):
        return None  # pragma: no cover -- key invariant: one value per key
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
        header=header,
    )

    # Splice header after the last existing AoT-owned slot if any,
    # else at end-of-doc.
    anchor = _aot_append_anchor(aot)
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
        append_direct_kv(entry_table, k, cst)
        dict.__setitem__(entry_table, k, dec)
    return entry_table


def clone_aot_entry(
    aot: AoT,
    src: Container | AoTEntry,
    *,
    dst_path: tuple[str, ...] | None = None,
    preserve_source_separator: bool = False,
) -> Table:
    """Append a deep CST clone of ``src`` to ``aot``.

    ``src`` is either an attached entry's ``Table`` or a bare
    ``AoTEntry`` from the AoT private-orphan rehome path, where the
    table has been reset but ``entry_slots`` still hold the CST.

    Preserves the source entry's per-slot leading / EOL / lexeme bytes
    so per-entry comments and trailing-comment formatting survive.
    Header leading follows the destination's ``_aot_separator`` policy
    unless ``preserve_source_separator=True`` (used by
    :func:`clone_aot` to preserve inter-entry layout).

    Supports source entries that include nested ``[a.sub]`` headers
    and KVSlots. ``dst_path`` (defaults to ``aot._path``) rebases both
    the entry header path and nested sub-section paths.
    """
    if isinstance(src, AoTEntry):
        src_entry = src
        src_slots: list[Slot] = list(src_entry.entry_slots)
    else:
        owner = src._owner_aot_entry  # noqa: SLF001
        if owner is None:  # pragma: no cover
            msg = "source entry has no owning AoTEntry"
            raise RuntimeError(msg)
        src_entry = owner
        # _gather_subtree_slots (not just entry.entry_slots) so nested
        # ``[[a.x]]`` entries living physically inside this entry's
        # body come along — entry_slots only holds the entry's *own*
        # slots, not those owned by nested AoTEntries.
        src_slots = _gather_subtree_slots(src)

    target_path = dst_path if dst_path is not None else aot._path  # noqa: SLF001
    return _install_cloned_aot_entry(
        aot,
        src_slots,
        src_entry.path,
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
    # else: keep source leading verbatim (bulk-clone past entry 0).

    entry_table = Table()
    _install_cloned_structural_block(
        entry_table,
        parent=parent,
        doc=doc,
        target_path=target_path,
        owner=new_entry,
        cloned_header=cloned_header,
        cloned_slots=cloned_slots,
        anchor=_aot_append_anchor(aot),
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
    ``parent._path + (key,)``), wires the section container, splices at
    the parent's subtree anchor, and populates child views.
    """
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
    _retarget_header_separator(head, _build_section_leading(doc))
    return _finish_cloned_section(
        parent,
        key,
        doc=doc,
        target_path=target_path,
        cloned_header=head,
        cloned_slots=cloned_slots,
    )


def _finish_cloned_section(
    parent: Container,
    key: str,
    *,
    doc: Document,
    target_path: tuple[str, ...],
    cloned_header: StructuralHeaderSlot,
    cloned_slots: list[Slot],
) -> Table:
    """Wire a cloned section block under ``parent[key]`` and store it.

    Shared tail of :func:`_install_cloned_section` (clones an existing
    header) and :func:`clone_document_as_section` (synthesises one).
    """
    from tomlrt._container import Table  # noqa: PLC0415

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

    Deep-clones the source entry's slots, rewriting the head from
    ``[[..]]`` to ``[..]`` and rebasing paths from the source AoT prefix
    to ``parent._path + (key,)``.
    """
    src_entry = src_entry_table._owner_aot_entry  # noqa: SLF001
    if src_entry is None:  # pragma: no cover
        msg = "source entry has no owning AoTEntry"
        raise RuntimeError(msg)
    src_slots = list(src_entry.entry_slots)
    return _install_cloned_section(parent, key, src_slots, src_entry.path)


def _owned_slots_from(root: Container, start: Slot) -> list[Slot]:
    """Collect ``root``'s owned slots in doc-stream order from ``start``.

    ``start`` must be the subtree's doc-stream-first owned slot (the
    header for a header-bearing section, the first dotted KV for a
    header-less one).
    """
    owned: set[int] = set()
    _collect_subtree(root, [], [], lambda s: owned.add(id(s)))
    return _owned_slots_in_doc_order(start, owned)


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
    src_slots = _gather_subtree_slots(src_table)
    head = src_slots[0]
    if not isinstance(head, StructuralHeaderSlot):  # pragma: no cover
        msg = "source section's first owned slot is not a header"
        raise AssertionError(msg)  # noqa: TRY004
    return head, src_slots


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
    if head.kind != "table":  # pragma: no cover
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

    Used for cross-doc assignment and same-doc clone of an attached
    ``[k]`` section. Preserves per-slot trivia and nested sub-sections
    while rebasing paths under ``parent._path + (key,)``.
    """
    _, src_slots = _gather_headered_subtree_slots(src_table)
    return _install_cloned_section(parent, key, src_slots, src_table._path)  # noqa: SLF001


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
    cloned_body = _clone_entry_slots(
        src_slots,
        new_entry=None,
        body_owner=parent._owner_aot_entry,  # noqa: SLF001
        src_prefix=src_doc._path,  # noqa: SLF001
        target_prefix=target_path,
        dst_newline=doc._newline,  # noqa: SLF001
        has_header=False,
    )
    return _finish_cloned_section(
        parent,
        key,
        doc=doc,
        target_path=target_path,
        cloned_header=header,
        cloned_slots=[header, *cloned_body],
    )


def adopt_private_section(
    dest_parent: Container,
    key: str,
    value: Container,
) -> Container:
    """Rehome a private-orphan section under ``dest_parent[key]`` in place.

    Moves the orphan's existing slot subtree into the destination document
    (rebasing paths / header keys / newlines) and re-points the existing
    view objects, rather than rebuilding from logical values. This is the
    inverse of :func:`delete_key`'s transplant-to-orphan, and preserves
    both the view's identity (``dest_parent[key] is value``) and its
    body trivia (comments, string / number style, inline pad).

    A header-bearing orphan that is itself an AoT entry (``[[a]]``) is
    normalised to a plain ``[key]`` section: the head's ``[[..]]``
    discriminator and the entry's slot / view ownership are cleared.

    Pre-condition (checked by the caller): ``value`` is a header-bearing
    section attached to a private orphan with intact slots.
    """
    doc = dest_parent._attached_doc  # noqa: SLF001
    old_prefix = value._path  # noqa: SLF001
    new_prefix = (*dest_parent._path, key)  # noqa: SLF001
    # Non-None iff the orphan is an AoT entry being normalised to a section.
    norm_entry = value._owner_aot_entry  # noqa: SLF001

    assert value._header_ref is not None  # noqa: SLF001
    _, slots = _gather_headered_subtree_slots(value)
    for s in slots:
        _retarget_slot_paths(s, old_prefix, new_prefix, doc._newline)  # noqa: SLF001
        if norm_entry is not None and s.owner_aot_entry is norm_entry:
            s.owner_aot_entry = None
            if isinstance(s, StructuralHeaderSlot) and s.entry is norm_entry:
                s.entry = None  # [[a]] -> [a]
    _rehome_view_tree(
        value, dest_parent, old_prefix, new_prefix, doc, clear_owner=norm_entry
    )

    header = value._header_ref.slot  # noqa: SLF001
    assert isinstance(header, StructuralHeaderSlot)
    _retarget_header_separator(header, _build_section_leading(doc))
    _splice_block_after(slots, _parent_subtree_tail(dest_parent), doc)
    _file_header_binding_chain(dest_parent, header)
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
    clear_owner: AoTEntry | None = None,
) -> None:
    """Re-point ``root``'s existing view subtree at ``doc`` with rebased paths.

    Keeps every view object (so identity survives), updating only the
    attachment fields (``_layout_root`` / ``_path`` / ``_parent``); the
    ``_refs`` / ``_index`` / ``_header_ref`` stay valid because the slots
    they reference were rebased consistently. Walks the same view spine
    (:func:`_walk_view_tree`) as the delete-side displacement walk, so
    every view that was re-pointed at the orphan is restored here.

    When ``clear_owner`` is given (an AoT entry being normalised to a
    plain section), any view owned by it has its ``_owner_aot_entry``
    cleared; nested AoT entries keep their own ownership.
    """
    from tomlrt._array import AoT, Array  # noqa: PLC0415
    from tomlrt._container import Container  # noqa: PLC0415

    def visit(node: object) -> None:
        if isinstance(node, (Container, AoT)):
            node._layout_root = doc  # noqa: SLF001
            node._path = _rebase_path(node._path, old_prefix, new_prefix)  # noqa: SLF001
            if (
                isinstance(node, Container)
                and clear_owner is not None
                and node._owner_aot_entry is clear_owner  # noqa: SLF001
            ):
                node._owner_aot_entry = None  # noqa: SLF001
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

    # An implicit table always owns at least one slot (an emptied one
    # materialises to an inline table and never reaches here).
    assert value._refs, "implicit orphan has no slots"  # noqa: SLF001
    slots = _owned_slots_from(value, value._refs[0].slot)  # noqa: SLF001

    nl = doc._newline  # noqa: SLF001
    for s in slots:
        _rebase_implicit_slot_in_place(s, old_prefix, new_prefix, host_path, nl)
    _rehome_view_tree(value, dest_parent, old_prefix, new_prefix, doc)

    _splice_block_after(slots, _parent_subtree_tail(host), doc)
    # value's own subtree refs travelled intact; re-file only the ancestor
    # binding refs the delete scrubbed: dotted KVs hosted at ``host`` and
    # nested headers propagate up the chain, but KVs under a nested
    # sub-section stay filed within value's subtree.
    chain = _dotted_chain(host, dest_parent)
    for s in slots:
        if isinstance(s, KVSlot) and s.host_path != host_path:
            continue
        for anc in chain:
            record_ref(anc, s)
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
    has_header: bool = True,
) -> list[Slot]:
    r"""Deep-clone an entry's slot list with path/owner rebasing.

    When ``has_header`` is True (default), the first slot must be the
    entry header. Its ``entry`` is set to ``new_entry`` — so passing
    ``new_entry=None`` converts an aot-entry header to a table
    header, and passing a non-None ``new_entry`` does the inverse.
    With ``has_header=False`` the list is body-only, used by in-place
    body replacement that keeps the destination header.

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
        nested_entry_map[id(s.entry)] = AoTEntry()

    cloned: list[Slot] = []
    for s in src_slots:
        c: Slot = copy.deepcopy(s)
        c._prev = None  # noqa: SLF001
        c._next = None  # noqa: SLF001
        _retarget_slot_paths(c, src_prefix, target_prefix, dst_newline)
        src_owner = s.owner_aot_entry
        mapped = nested_entry_map.get(id(src_owner)) if src_owner else None
        c.owner_aot_entry = mapped if mapped is not None else body_owner
        if isinstance(c, StructuralHeaderSlot):
            assert isinstance(s, StructuralHeaderSlot)
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
        header=header,
    )

    # Anchor past the whole subtree of the nearest header-bearing
    # ancestor: a header re-parents everything after it, so landing it
    # mid-section would capture that host's trailing KVs (e.g. a ``d = 4``
    # sibling of an implicit ``parent``) under the new header on re-parse.
    anchor = _parent_subtree_tail(_nearest_header_host(parent))
    _splice_block_after([header], anchor, doc)
    # Own the new header on the AoT entry so a later delete of the
    # entry takes the promoted section with it.
    _extend_entry_slots(owner, header)

    # File the binding ref under the deepest implicit parent and
    # propagate ancestor-prefix bindings up to the doc root.
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


def _subtree_membership(c: Container) -> set[int]:
    """Set of ``id(slot)`` for every slot owned by ``c``'s subtree.

    Membership only (order-independent): ``c``'s own slots plus those
    of every nested section and nested AoT entry.
    """
    owned: set[int] = set()
    _collect_subtree(c, [], [], lambda s: owned.add(id(s)))
    return owned


def _aot_append_anchor(aot: AoT) -> Slot | None:
    """Return the slot a newly-appended ``[[path]]`` entry splices after.

    For a non-empty AoT this is the last *physical* slot of the last
    entry's whole subtree — including slots owned by nested ``[[a.sub]]``
    AoT entries, which ``entry_slots`` deliberately excludes. Found via
    ``_parent_subtree_tail``, a forward walk bounded by the entry's own
    subtree extent — not by how much unrelated content follows it in the
    document — so repeated appends to a non-tail AoT stay cheap.

    For an empty AoT (no entries, or all entries slot-less) the anchor
    is the parent container's subtree tail, so the first materialised
    header lands inside the owning section / AoT entry rather than at
    the document tail — where a re-parse would bind it to whichever
    sibling currently sits last. ``None`` (append at doc end) is
    returned only for an empty AoT whose parent has no slots of its own.
    """
    for entry_table in reversed(aot):
        e = entry_table._owner_aot_entry  # noqa: SLF001
        if e is None or not e.entry_slots:
            continue
        return _parent_subtree_tail(entry_table)
    parent = aot._parent  # noqa: SLF001
    if parent is None:  # pragma: no cover -- attached AoT always has a parent
        return None
    # The first materialised header is a sub-section of its host — the
    # nearest header-bearing ancestor (or the document root). Anchoring
    # at the host's whole-subtree tail keeps every direct / dotted KV of
    # that header ahead of the new sub-section, where a re-parse would
    # otherwise capture them. When ``parent`` itself bears a header (a
    # section or AoT-entry table) the host is ``parent``.
    host = _nearest_header_host(parent)
    return _parent_subtree_tail(host)


def _entry_last_slot(entry: AoTEntry) -> Slot | None:
    """Return the entry's own slot with the greatest doc-stream position.

    Excludes slots owned by nested ``[[a.sub]]`` AoT entries (those
    have their own, separate ``AoTEntry.entry_slots``) — unlike
    :func:`_aot_append_anchor`, which deliberately wants the *whole*
    subtree tail. Walks forward from the entry's own header, bounded
    by ``_belongs_to_parent_extent`` (the same predicate
    :func:`_parent_subtree_tail` uses) so the search never depends on
    unrelated content elsewhere in the document — only on how much is
    physically nested inside this one entry. ``entry_slots`` need not
    be doc-stream-contiguous (nested AoT children may interleave), so
    the last in-extent slot that is also one of ``entry``'s own is
    tracked separately from the walk's stopping condition.
    """
    header = entry.entry_slots[0]
    assert isinstance(header, StructuralHeaderSlot)
    own_ids = {id(s) for s in entry.entry_slots}
    result: Slot = header
    cur: Slot = header
    while cur._next is not None:  # noqa: SLF001
        nxt = cur._next  # noqa: SLF001
        if not _belongs_to_parent_extent(nxt, header.path, entry):
            break
        if id(nxt) in own_ids:
            result = nxt
        cur = nxt
    return result


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
                placeholder = _empty_aot_placeholder_ref(v)
                if placeholder is not None:
                    sink.append(placeholder.slot)
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
        deduped = list(dict.fromkeys(owned_ordered))
        for s in deduped:
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
        _reset_table_for_rehome(entry_table, recurse=True)

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


def replace_aot_entry(aot: AoT, index: int, body: Mapping[str, Any]) -> None:
    """Replace ``aot[index]`` in place.

    Keeps the entry's header slot and live `Table` view; just clears
    the body and re-populates from ``body``.

    O(m) in the size of ``body``, independent of AoT length and
    document size. Header position and `_refs` ordering are preserved
    because no slot splicing is involved.
    """
    entry_table = aot[_norm_aot_index(aot, index)]
    if body is entry_table:
        return
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
        # Peer-block model: every entry header is a comparable block.
        # The structural separator at each physical position is
        # positional (stays at that index); the comment-remainder
        # travels with the block.
        structural_by_position: list[Trivia] = []
        remainder_by_block: list[Trivia] = []
        for block in physical_blocks:
            structural, remainder = _split_leading_for_reorder(block[0])
            structural_by_position.append(structural)
            remainder_by_block.append(remainder)

        new_order_indices = [
            phys_idx_by_id[id(t)] for t in new_logical_order if id(t) in phys_idx_by_id
        ]
        new_head_leadings = [
            Trivia(
                list(structural_by_position[new_pos].pieces)
                + list(remainder_by_block[phys_idx].pieces)
            )
            for new_pos, phys_idx in enumerate(new_order_indices)
        ]
        _splice_blocks_in_order(
            doc, physical_blocks, new_order_indices, new_head_leadings
        )

    # Reflect the new order in the AoT's own list view.
    list.clear(aot)
    for t in new_logical_order:
        list.append(aot, t)

    # Resort _refs lists on every container in the AoT's parent chain.
    # Each ancestor holds entry-header refs (one per entry, filed under
    # the relevant path component); after splicing, those refs are out
    # of doc-stream order. Sort by slot's new doc-stream position.
    _resort_refs_by_doc_order(_ancestor_chain(aot), doc)


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


def _splice_blocks_in_order(
    doc: Document,
    physical_blocks: list[list[Slot]],
    new_order_indices: list[int],
    new_head_leadings: list[Trivia],
) -> None:
    """Reorder movable layout blocks within the doc-stream.

    ``physical_blocks`` is the list of non-empty slot blocks in
    physical doc-stream order (``physical_blocks[0][0]`` is the
    earliest owned slot). ``new_order_indices`` maps each new position
    to the old physical block index. ``new_head_leadings`` is indexed by
    new position.

    The helper is purely permutational on the doc-stream linked
    list. Trivia policy (positional vs slot-attached) is the
    caller's responsibility — see ``renormalise_aot_order``
    (peer-block model) and ``reorder_container`` (region-marker
    model) for the two existing flavours.
    """
    if not physical_blocks:
        return

    anchor_prev = physical_blocks[0][0]._prev  # noqa: SLF001

    for block in physical_blocks:
        for s in block:
            unlink_slot(s, doc, strip_new_head_leading=False)

    insert_after_slot = anchor_prev
    for phys_idx in new_order_indices:
        for slot in physical_blocks[phys_idx]:
            if insert_after_slot is None:
                insert_before_head(slot, doc)
            else:
                insert_after(insert_after_slot, slot, doc)
            insert_after_slot = slot

    for new_pos, phys_idx in enumerate(new_order_indices):
        head_slot = physical_blocks[phys_idx][0]
        head_slot.leading = Trivia(list(new_head_leadings[new_pos].pieces))


def _find_binding_successor(parent: Container, key: str) -> Slot | None:
    """Return the slot just after the first contiguous run bound under ``parent[key]``.

    Walks the doc-stream forward from ``parent._index[key][0].slot``,
    consuming consecutive slots whose binding root starts with
    ``(*parent._path, key)``, and returns the first non-matching slot
    (or ``None`` if the run extends to doc tail).

    ``Container._structural_overwrite`` restores this successor's
    ``leading`` after moving the new binding back, preserving the visual
    gap across the ``del + set`` round-trip.

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
    elif isinstance(slot, KVSlot):
        root = (*slot.host_path, *slot.key)
    else:
        return None
    if len(root) > parent_plen and root[:parent_plen] == parent_path:
        return root[parent_plen]
    return None


def reorder_container(c: Container, new_key_order: list[str]) -> None:
    """Reorder ``c``'s direct children to ``new_key_order``.

    ``new_key_order`` is trusted to be a permutation of
    ``dict.keys(c)``. Each key's slots — KV slots filed under the
    key, plus child section/AoT slots gathered from anywhere in the
    doc-stream — are spliced together as one contiguous block. Only the
    head of each block receives a new *positional separator*; attached
    comments and other leadings travel with their slots.

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
    :func:`_subtree_membership`): sibling entries with the same path
    are excluded so their content is not merged in, but nested
    descendants — including nested AoT children, which are owned by
    their *own* entry — do participate and move with their key.

    Only mutates the CST; dict storage is the caller's responsibility.

    Raises:
        ValueError: the proposed order places a leaf KV after a
            structural section/AoT key (would re-bind it as nested).
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

    # Build blocks in physical doc-stream order. The c-header (if any)
    # is a one-slot block; child keys appear where first seen.
    #
    # Only AoT entries need a membership filter: they share paths with
    # sibling entries, but nested AoT children owned by their own entries
    # still move with their key. Non-AoT paths are unique, so matching
    # bind-keys are necessarily within c's subtree.
    membership = (
        _subtree_membership(c)
        if c._owner_aot_entry is not None  # noqa: SLF001
        else None
    )

    key_blocks: dict[str, list[Slot]] = {k: [] for k in new_key_order}
    physical_blocks: list[list[Slot]] = []
    phys_idx_of_header: int | None = None
    phys_idx_of_key: dict[str, int] = {}

    cur: Slot | None = doc._head  # noqa: SLF001
    while cur is not None:
        is_header = cur is header_slot
        bind_key = None if is_header else _direct_child_key(cur, c_path, c_plen)
        in_scope = membership is None or id(cur) in membership
        if is_header:
            phys_idx_of_header = len(physical_blocks)
            physical_blocks.append([cur])
        elif bind_key is not None and bind_key in key_blocks and in_scope:
            if bind_key not in phys_idx_of_key:
                phys_idx_of_key[bind_key] = len(physical_blocks)
                physical_blocks.append(key_blocks[bind_key])
            key_blocks[bind_key].append(cur)
        cur = cur._next  # noqa: SLF001

    # Reject orders that would re-bind a leaf KV under a structural
    # section header. A block with *any* leaf KV (a bare or dotted
    # ``key = value``) must not follow a structural block — including a
    # "mixed" key that owns both a leaf and a sub-section, whose leaf
    # part would be captured.
    def _has_leaf(slots: list[Slot]) -> bool:
        # Only a KV hosted directly by ``c`` (a bare or dotted leaf of
        # ``c``) can be captured by a preceding header; a KV under the
        # block's own section header (``host_path`` deeper than ``c``)
        # is correctly scoped and does not count.
        return any(isinstance(s, KVSlot) and s.host_path == c_path for s in slots)

    def _has_structural(slots: list[Slot]) -> bool:
        return any(isinstance(s, StructuralHeaderSlot) for s in slots)

    seen_structural = False
    for k in new_key_order:
        block = key_blocks[k]
        if not block:
            continue
        if _has_leaf(block) and seen_structural:
            msg = (
                f"reorder: key {k!r} has leaf content that cannot follow a "
                f"structural section/AoT key (would rebind it as nested)"
            )
            raise ValueError(msg)
        if _has_structural(block):
            seen_structural = True

    if not physical_blocks:
        return

    earliest_owned = physical_blocks[0][0]

    # Foreign slots interleaved in c's owned span must keep their
    # re-parse scope. Hoist foreign KVs that still belong to c's
    # containing scope to the region head; stop at a foreign header,
    # which establishes its own scope and would capture c's dotted leaves
    # if hoisted.
    owned_ids = {id(s) for blk in physical_blocks for s in blk}
    front_foreign: list[Slot] = []
    seen = 1  # earliest_owned itself
    scan: Slot | None = earliest_owned._next  # noqa: SLF001
    while scan is not None and seen < len(owned_ids):
        if id(scan) in owned_ids:
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

    # Region-external prefix moves to c-header's leading so preambles /
    # above-region separators survive even if c-header was not first.
    region_head_structural, region_head_remainder = _split_leading_for_reorder(
        earliest_owned
    )

    # c-header's attached comments travel with it; its structural prefix
    # is replaced by the region-external prefix above.
    if header_slot is not None:
        if header_slot is earliest_owned:
            c_header_remainder = region_head_remainder
        else:
            _, c_header_remainder = _split_leading_for_reorder(header_slot)
    else:
        c_header_remainder = Trivia()

    # Snapshot child positional prefixes. Child position 0 keeps a gap
    # only if c-header actually sat above the first child in source; else
    # that prefix is the region-external one already consumed above.
    child_keys_in_phys_order = sorted(phys_idx_of_key, key=phys_idx_of_key.__getitem__)
    child_physical_blocks = [key_blocks[k] for k in child_keys_in_phys_order]
    child_phys_pos_of_key: dict[str, int] = {
        k: i for i, k in enumerate(child_keys_in_phys_order)
    }

    header_above_first_child = (
        header_slot is not None
        and child_physical_blocks
        and earliest_owned is not child_physical_blocks[0][0]
    )

    child_structural_at_new_pos: list[Trivia] = []
    child_remainder_of: list[Trivia] = []
    for i, block in enumerate(child_physical_blocks):
        structural, remainder = _split_leading_for_reorder(block[0])
        if i == 0 and header_slot is not None and not header_above_first_child:
            structural = Trivia()
        child_structural_at_new_pos.append(structural)
        child_remainder_of.append(remainder)

    # c-header, when present, occupies new position 0; children follow.
    # Each child position gets that position's structural prefix plus the
    # moved block's attached-comment remainder.
    new_order_indices: list[int] = []
    new_head_leadings: list[Trivia] = []

    if header_slot is not None:
        assert phys_idx_of_header is not None
        new_order_indices.append(phys_idx_of_header)
        new_head_leadings.append(
            Trivia(
                list(region_head_structural.pieces) + list(c_header_remainder.pieces)
            )
        )

    new_child_pos = 0
    for k in new_key_order:
        if k not in child_phys_pos_of_key:
            continue
        phys_child_idx = child_phys_pos_of_key[k]
        new_order_indices.append(phys_idx_of_key[k])
        structural = child_structural_at_new_pos[new_child_pos]
        remainder = child_remainder_of[phys_child_idx]
        new_head_leadings.append(
            Trivia(list(structural.pieces) + list(remainder.pieces))
        )
        new_child_pos += 1

    _splice_blocks_in_order(doc, physical_blocks, new_order_indices, new_head_leadings)

    # Resort refs and recompute cached body tails invalidated by splice.
    _resort_and_recompute_tails(c, doc)


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
