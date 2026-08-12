"""Mutate section-side layout.

This module owns linked-list and per-container cache updates for
direct KV insert and leaf delete. Inline-table mutation lives in
``_inline_ops.py``.

Design notes:

* The doc-stream linked list is the single source of physical ordering;
  inserts splice exactly one slot at an explicit anchor.
* ``c._refs`` mirrors the doc-stream subset referenced by ``c``. A
  direct KV insert's ref goes immediately after the anchor's ref (or
  at the front), not blindly at the tail where child-section refs may
  already sit.
* ``c._body_tail`` is incremental: O(1) on insert, O(len(c._refs)) only
  when deleting the current tail. It also answers "what is ``c``'s
  last body KV?" (`_last_body_kv`), so no insert has to search for one.
* A non-dotted direct KV files exactly one ref on its host container;
  ancestors are unaffected.
"""

from __future__ import annotations

import bisect
import contextlib
import copy
import itertools
import operator
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
    ensure_terminator,
    retarget_slot_newlines,
    stitch_run,
)
from tomlrt._trivia import (
    EolTrivia,
    leading_has_blank_line,
    trailing_ws,
)
from tomlrt._values import (
    ArrayValue,
    InlineTableValue,
    make_keyparts,
)
from tomlrt._view import _View, is_inline_value

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence

    from tomlrt._array import AoT, Array
    from tomlrt._container import Container, Document, Table
    from tomlrt._slots import Slot
    from tomlrt._values import InlineTableEntry, KeyPart, Value


# ---------------------------------------------------------------------------
# Pure linked-list ops
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _record_install(
    doc: Document,
) -> Iterator[tuple[list[Slot], list[tuple[Slot, str, Slot | None]]]]:
    """Record slots installed and existing slots displaced by the transaction.

    The three insertion primitives (:func:`insert_after`,
    :func:`insert_before`, :func:`insert_before_head`) — the only points
    at which a slot is linked into the document — append newly linked
    slots to the first yielded list. The second captures existing
    slots whose leading trivia was rewritten by synthetic-header
    insertion. Slots are normally freshly materialised, but an install
    that adopts a subtree from the same private document relinks slots
    that already existed — and so can record the very slot the
    reposition anchor names. Nested contexts stack; only the innermost
    is active.
    """
    prev = doc._install_recorders  # noqa: SLF001
    installed: list[Slot] = []
    displaced: list[tuple[Slot, str, Slot | None]] = []
    doc._install_recorders = (installed, displaced)  # noqa: SLF001
    try:
        yield installed, displaced
    finally:
        doc._install_recorders = prev  # noqa: SLF001


@contextlib.contextmanager
def _suspend_install_recording(doc: Document) -> Iterator[None]:
    """Hide slots linked inside from any open install transaction.

    A repair made while an install is in flight is not part of the
    installed block; recording it would put it in the span
    :func:`reposition_install` moves to the saved anchor.
    """
    prev = doc._install_recorders  # noqa: SLF001
    doc._install_recorders = None  # noqa: SLF001
    try:
        yield
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
    ``StructuralHeaderSlot``: a bare KV's scope comes from whichever
    header most recently opened, not necessarily from ``anchor`` itself
    (``anchor`` may be a KV physically inside some other table's own
    body). Returns ``None`` for doc-root scope.
    """
    cur = anchor
    while cur is not None:
        if isinstance(cur, StructuralHeaderSlot):
            return cur.path
        cur = cur._prev  # noqa: SLF001
    return None


def reposition_install(parent: Container, key: str, value: Any) -> None:
    """Replace ``parent[key]`` while preserving its physical position.

    The binding is deleted, reinstalled via ``parent[key] = value``,
    captured with ``_record_install``, then moved back to the saved anchor.

    Reinstalling at the tail is what keeps `_insert_new` and the
    attach paths under it anchor-free: a ``[a]`` header claims each
    following line until the next, so a part-built block mid-stream
    owns the wrong lines while one at the tail can swallow nothing.
    Only its position is then wrong. The move is best-effort — an
    anchor the reinstall invalidated, or a destination that would
    change what the block owns, leaves it at the tail.

    A surviving neighbour keeps its pre-op leading iff, after the move,
    it sits immediately after the slot that legitimately precedes it:
    the relocated block tail for the original successor, or the original
    predecessor for a sibling temporarily displaced by synthetic header
    insertion. The expected predecessor is unique, so a slot that is
    both successor and displaced sibling is restored at most once.

    A header-less new binding (scalar / synth-inline) is left where
    ``_insert_new`` placed it when the captured anchor lies outside
    ``parent``'s body region — moving it there would silently
    re-parent it. A new binding that brings its own header carries its
    scope with it and is always safe to reposition.

    Precondition: ``key`` is currently bound under ``parent``.
    """
    primary_ref = _binding_primary_ref(parent, key)
    old_primary = primary_ref.slot
    saved_anchor_prev, successor_slot = _binding_run_neighbours(parent, key)
    saved_leading = old_primary.leading
    successor_leading = successor_slot.leading if successor_slot is not None else None
    # The header-less safety check reads the doc-stream around the
    # captured anchor, so evaluate it before ``del`` perturbs the
    # links. The header-bearing check is done after install, against
    # the actual installed slots.
    in_body = _anchor_in_parent_direct_body(parent, saved_anchor_prev)
    # A dotted-key binding (primary slot is a KVSlot, not a header) keeps
    # the dotted form when re-emitted into an emptied implicit container
    # — replacing ``a.b.c = 1`` with a scalar yields ``a.b = "str"``, not
    # a new ``[a]`` header.
    old_is_kv = isinstance(old_primary, KVSlot)
    delete_key(parent, key)
    doc = parent._attached_doc  # noqa: SLF001
    with _record_install(doc) as (new_slots, displaced):
        parent._insert_new(  # noqa: SLF001
            key,
            value,
            reinstall_as_dotted=old_is_kv,
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
    _move_slots_to_anchor(
        parent, installed, saved_anchor_prev, saved_leading, from_kv=old_is_kv
    )
    # Restore each perturbed neighbour's pre-op leading iff the move left
    # it directly after the predecessor that makes that leading correct.
    restores: list[tuple[Slot, str, Slot | None]] = list(displaced)
    if successor_slot is not None and successor_leading is not None:
        restores.append((successor_slot, successor_leading, installed[-1]))
    for slot, original, expected_pred in restores:
        if slot._prev is expected_pred:  # noqa: SLF001
            slot.leading = original


def _recorded_install_span(recorded: list[Slot], doc: Document) -> list[Slot] | None:
    """Return the linked recorded slots in order, if they form one span.

    Slots unlinked again during the transaction are ignored. Implicit
    sources can install direct KVs and structural children into separate
    regions; those records return ``None`` and are not repositioned.
    """
    survivors = list(dict.fromkeys(s for s in recorded if _slot_is_linked(s, doc)))
    span = set(survivors)
    heads = [
        s
        for s in survivors
        if s._prev is None or s._prev not in span  # noqa: SLF001
    ]
    # Also rejects an empty span, which has no head at all.
    if len(heads) != 1:
        return None
    ordered: list[Slot] = []
    cur: Slot | None = heads[0]
    while cur is not None and cur in span:
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
    """Return whether ``slots`` may be moved to sit after ``anchor``.

    False either because the anchor is inside the block itself, or
    because the move would change the block's TOML scope.
    """
    if anchor is not None and anchor in slots:
        # The reinstall took over the slot the anchor named — it can
        # happen when the new value is a sibling that physically
        # preceded the binding being replaced. "Sit after yourself" has
        # no answer, so leave the block where the reinstall put it.
        return False
    if not any(isinstance(s, StructuralHeaderSlot) for s in slots):
        return in_parent_body

    installed = set(slots)
    successor = anchor._next if anchor is not None else doc._head  # noqa: SLF001
    while successor is not None and successor in installed:
        successor = successor._next  # noqa: SLF001
    if isinstance(successor, KVSlot):
        return False

    first = slots[0]
    return not (
        isinstance(first, KVSlot)
        and _effective_header_path_before(anchor) != first.host_path
    )


def _ancestor_chain(c: Container) -> list[Container]:
    """Ancestors from ``c._parent`` up to (and including) the document root."""
    out: list[Container] = []
    cur = c._parent  # noqa: SLF001
    while cur is not None:
        out.append(cur)
        cur = cur._parent  # noqa: SLF001
    return out


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


_ref_order = operator.attrgetter("slot._order")
"""Order key of a `SlotRef`'s slot — the sort key of every ref projection."""


def _ordered_projections(c: Container, ref: SlotRef) -> tuple[list[SlotRef], ...]:
    """``c``'s doc-ordered ref lists that hold ``ref``: ``_refs`` + its bucket.

    The container's own header ref has no ``local_key`` and so lives in
    ``_refs`` alone; every other ref is also filed in its ``_index``
    bucket.
    """
    local_key = ref.local_key
    if local_key is None:
        return (c._refs,)  # noqa: SLF001
    return c._refs, c._index.setdefault(local_key, [])  # noqa: SLF001


def _ordered_index(refs: list[SlotRef], order: int) -> int:
    """Index at which order key ``order`` sits, or belongs, in ``refs``.

    Serves both "where is the ref for this slot?" and "where would a new
    one go?": a ref list holds at most one ref per slot, so the answers
    coincide, and no caller has to know which existing ref its new one
    follows.
    """
    return bisect.bisect_left(refs, order, key=_ref_order)


def _filed_predecessor(c: Container, slot: Slot) -> Slot | None:
    """Slot of ``c``'s last filed ref that precedes ``slot`` in the doc-stream."""
    refs = c._refs  # noqa: SLF001
    idx = _ordered_index(refs, slot._order)  # noqa: SLF001
    return refs[idx - 1].slot if idx else None


def record_ref(c: Container, slot: Slot) -> SlotRef:
    """Create a `SlotRef(slot, c)` and file it in doc order on ``c``.

    The ``_index`` key is :attr:`SlotRef.local_key`, derived from
    ``(slot, container)`` geometry, so callers cannot file under a
    disagreeing key. ``slot`` must already be linked into the
    doc-stream, since its order key is what places the ref.
    """
    ref = SlotRef(slot, c)
    order = slot._order  # noqa: SLF001
    for refs in _ordered_projections(c, ref):
        if not refs or refs[-1].slot._order < order:  # noqa: SLF001
            # Filing in doc order — the builder's whole-document pass,
            # every sequential body append — lands at the tail.
            refs.append(ref)
        else:
            refs.insert(_ordered_index(refs, order), ref)
    return ref


@contextlib.contextmanager
def _refile_region_refs(
    doc: Document,
    predecessor: Slot | None,
    successor: Slot | None,
) -> Iterator[None]:
    """Re-file the refs of a doc-stream region around a physical change to it.

    The region is the open interval between two slots that stay put.
    Every doc-ordered projection it appears in — a container's ``_refs``
    and each of its ``_index`` buckets — holds its refs as one
    contiguous run, so each run is put back in the order, and at the
    position, the refreshed order keys imply once the change is done.
    Cost is proportional to the region rather than to the document, and
    one mechanism serves every flavour of change: moving the region
    elsewhere, permuting it in place, or both.
    """
    runs: dict[int, tuple[list[SlotRef], list[SlotRef]]] = {}
    for slot in _slots_between(doc, predecessor, successor):
        for ref in slot._refs:  # noqa: SLF001
            for refs in _ordered_projections(ref.container, ref):
                # A projection holding this ref alone has nothing to
                # reorder and nowhere else to sit.
                if len(refs) > 1:
                    runs.setdefault(id(refs), (refs, []))[1].append(ref)
    placed = [
        (refs, run, _ordered_index(refs, _ref_order(run[0])))
        for refs, run in runs.values()
    ]
    for refs, run, start in placed:
        assert refs[start : start + len(run)] == run, "region refs must be one run"
    yield
    for refs, run, start in placed:
        _replace_ordered_run(refs, run, start)


def _replace_ordered_run(refs: list[SlotRef], run: list[SlotRef], start: int) -> None:
    """Put ``run``, the former slice of ``refs`` at ``start``, back in key order.

    It goes straight back where it was if its refs' order keys still sit
    between the same neighbours — the whole projection, or a region
    permuted in place — and is otherwise lifted out and placed afresh.
    """
    run.sort(key=_ref_order)
    end = start + len(run)
    if (start == 0 or _ref_order(refs[start - 1]) < _ref_order(run[0])) and (
        end == len(refs) or _ref_order(run[-1]) < _ref_order(refs[end])
    ):
        refs[start:end] = run
        return
    del refs[start:end]
    at = _ordered_index(refs, _ref_order(run[0]))
    refs[at:at] = run


def file_own_header(c: Container, header: StructuralHeaderSlot) -> None:
    """File ``header`` as ``c``'s own physical presence.

    Every header-filing path establishes ``_header_ref`` and
    ``_body_tail`` together — the other is
    `_file_synthetic_header_and_kv`, which lands the tail on the KV it
    inserts. Here ``c`` has no body yet, so its own header is the tail,
    which is exactly what `_recompute_body_tail` derives for it.
    """
    c._header_ref = record_ref(c, header)  # noqa: SLF001
    c._body_tail = header  # noqa: SLF001


def maybe_advance_body_tail(c: Container, slot: Slot) -> None:
    """Advance ``c._body_tail`` if ``slot`` is a body-region KV of ``c``.

    A KV's ref is filed on its host container and on each implicit
    dotted container below it. A dotted intermediate can never also be
    an explicit section (the validator rejects that), so a SECTION here
    is always the KV's own host.

    Owners really can differ, though: a KV owned by no AoT entry can be
    moved into a container owned by one, and then sits inside ``c``
    without belonging to its body.
    """
    assert isinstance(slot, KVSlot)
    assert c._kind is not _Kind.SECTION or slot.host_path == c._path  # noqa: SLF001
    if slot.owner_aot_entry is c._owner_aot_entry:  # noqa: SLF001
        c._body_tail = slot  # noqa: SLF001


def _file_header_binding_chain(
    deepest: Container,
    header: StructuralHeaderSlot,
) -> None:
    """File ``header`` in doc order on ``deepest`` and every ancestor."""
    for c in [deepest, *_ancestor_chain(deepest)]:
        record_ref(c, header)


def _extend_header_bindings_to_root(
    parent: Container,
    slots: Iterable[Slot],
) -> None:
    """Extend headers through ``parent``'s ancestors in physical order."""
    for s in slots:
        if isinstance(s, StructuralHeaderSlot):
            _file_header_binding_chain(parent, s)


def _file_synthetic_header_and_kv(
    c: Container,
    *,
    header_slot: StructuralHeaderSlot,
    key: str,
    value: Value,
    doc: Document,
    owner: AoTEntry | None,
) -> KVSlot:
    """Common tail of the two header-synthesis paths.

    Files ``c``'s own-header ref, inserts ``key = value`` directly
    after ``header_slot``, files the KV ref, and updates
    ``c._header_ref`` / ``c._index[key]`` / ``c._body_tail``.

    Anchoring and ancestor binding-ref filing stay explicit in callers;
    both are highly position-sensitive and not safe to share.
    """
    c._header_ref = record_ref(c, header_slot)  # noqa: SLF001

    new_kv = _new_kv_slot(
        host_path=c._path,  # noqa: SLF001
        key=(key,),
        value=value,
        doc=doc,
        owner=owner,
        leading="",
    )
    insert_after(header_slot, new_kv, doc)
    record_ref(c, new_kv)
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
    existing components are always ``Container`` instances.
    """
    from tomlrt._container import Container  # noqa: PLC0415

    doc = parent._attached_doc  # noqa: SLF001
    owner = parent._owner_aot_entry  # noqa: SLF001
    cur: Container = parent
    for j, comp in enumerate(sub_path):
        if comp in cur:
            nxt = dict.__getitem__(cur, comp)
            assert isinstance(nxt, Container)
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


def _default_eol(doc: Document) -> EolTrivia:
    """A bare-newline `EolTrivia` for a freshly synthesised slot."""
    return EolTrivia(trailing_ws="", comment="", newline=doc._newline)  # noqa: SLF001


def _link_run_between(
    prev: Slot | None, run: Sequence[Slot], nxt: Slot | None, doc: Document
) -> None:
    """Link ``run``, in order, between ``prev`` and ``nxt`` in ``doc``.

    `stitch_run` does the linking and the order-key stamping; this adds
    the document's own ends, which it knows nothing about. A ``prev`` or
    ``nxt`` of ``None`` names one of those ends, so the run — which may
    be empty, leaving the two ends to meet — takes it over.
    """
    stitch_run(prev, run, nxt)
    if prev is None:
        doc._head = run[0] if run else nxt  # noqa: SLF001
    if nxt is None:
        doc._tail = run[-1] if run else prev  # noqa: SLF001


def _insert_between(
    prev: Slot | None, new_slot: Slot, nxt: Slot | None, doc: Document
) -> None:
    """Splice a newly materialised ``new_slot`` between ``prev`` and ``nxt``."""
    _link_run_between(prev, (new_slot,), nxt, doc)
    _record_new_slot(doc, new_slot)


def _relink_run_after(
    anchor: Slot | None, slots: Sequence[Slot], doc: Document
) -> None:
    """Link an unlinked run of slots back into ``doc``, in order, after ``anchor``.

    The shared re-splice of the two block-permutation paths. The slots
    already belong to the document, so this is a relink rather than an
    install and nothing is recorded against an install in flight.
    """
    nxt = anchor._next if anchor is not None else doc._head  # noqa: SLF001
    _link_run_between(anchor, slots, nxt, doc)


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
    Idempotent and a no-op when the preamble is empty.
    """
    preamble = doc._preamble  # noqa: SLF001
    if not preamble:
        return
    breaks = 0
    rest = preamble
    while breaks < 2 and rest.endswith("\n"):
        rest = rest[: -2 if rest.endswith("\r\n") else -1]
        breaks += 1
    doc._preamble = preamble + doc._newline * (2 - breaks)  # noqa: SLF001


def unlink_slot(
    slot: Slot, doc: Document, *, strip_new_head_leading: bool = True
) -> None:
    """Remove ``slot`` from ``doc``'s linked list.

    When ``strip_new_head_leading`` is True (default) and the unlink
    promotes a successor to be the new doc head, blank lines on that
    successor's leading are stripped — a separator from the removed
    first slot must not show up as a stray blank at the top of the
    file. Pass False for transient unlinks (e.g. AoT renormalise that
    re-splices the same slots) where the leading must be preserved.
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
    """Drop the run of blank lines that starts ``slot.leading``.

    Comments are preserved (we don't want to silently drop user
    comments): the walk stops at the first line that is not a bare
    terminator.
    """
    leading = slot.leading
    i = 0
    while leading.startswith("\n", i) or leading.startswith("\r\n", i):
        i += 2 if leading[i] == "\r" else 1
    slot.leading = leading[i:]


# ---------------------------------------------------------------------------
# Higher-level ops
# ---------------------------------------------------------------------------


def _splice_body_slot(
    new_slot: Slot,
    *,
    anchor_body_tail: Slot | None,
    doc: Document,
) -> bool:
    """Splice ``new_slot`` into the doc-stream at the canonical body anchor.

    Anchor preference: body tail (which for a header-bearing container
    with no body yet is that header) > head-of-doc seam > empty doc.
    Returns ``True`` iff ``new_slot`` became the new doc head ahead of
    an existing head (the seam case), where ancestor refs must go at
    index 0.
    """
    if anchor_body_tail is not None:
        ensure_terminator(anchor_body_tail, doc._newline)  # noqa: SLF001
        insert_after(anchor_body_tail, new_slot, doc)
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
        # A fresh ``host_path = c._path`` slot would render in whatever
        # scope the previous header (or the doc root) established, not
        # in ``c``'s logical scope. Insert via a dotted KV under the
        # nearest header-bearing ancestor instead.
        if c._body_tail is None and not reinstall_as_dotted:  # noqa: SLF001
            # ``c`` has no dotted body to anchor a dotted KV: promote it
            # to an explicit ``[c]`` header, before its first descendant
            # header when it has one (``[a.b]`` ⇒ synthesise ``[a]``), or
            # as a fresh header when fully empty. Exception: a structural
            # overwrite replacing a dotted binding keeps the dotted form
            # (see ``reposition_install``).
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
        doc=doc,
    )
    record_ref(c, new_slot)
    c._body_tail = new_slot  # noqa: SLF001
    _extend_entry_slots(c._owner_aot_entry, new_slot)  # noqa: SLF001


def _invalidate_body_tail_chain(
    start: Container | None,
    owned_slots: set[Slot] | None,
    *,
    min_depth: int = 0,
) -> None:
    """Recompute invalidated ``_body_tail`` values on the path to root.

    For each container ``cc`` along the chain whose existing
    ``_body_tail`` slot is in ``owned_slots``, recompute the tail.
    ``owned_slots`` of ``None`` means "every cached tail is suspect" —
    used after a block move, which can hand a container a later body
    slot than the one it was caching.

    Stops once ``len(cc._path) < min_depth``: an ancestor at depth
    ``d`` cannot have its body_tail point at a slot whose minimum
    bottom-depth exceeds ``d``, so common-case leaf-KV deletes never
    walk past ``c`` itself.
    """
    cur = start
    while cur is not None and len(cur._path) >= min_depth:  # noqa: SLF001
        tail = cur._body_tail  # noqa: SLF001
        if tail is not None and (owned_slots is None or tail in owned_slots):
            cur._body_tail = _recompute_body_tail(cur)  # noqa: SLF001
        cur = cur._parent  # noqa: SLF001


def _nearest_header_host(c: Container) -> Container:
    """The closest ancestor (or ``c``) owning a header, else the subtree root.

    The walk stops at a document boundary. A popped subtree's root keeps
    pointing at the parent it was detached from, which lives in another
    document; climbing into it would host the new slot there, physically
    splicing orphan content into a document that does not own it.
    """
    host = c
    while (
        host._header_ref is None  # noqa: SLF001
        and host._parent is not None  # noqa: SLF001
        and host._parent._layout_root is host._layout_root  # noqa: SLF001
    ):
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
    new_slot: Slot,
    primary: Slot,
    doc: Document,
) -> None:
    """Splice ``new_slot`` into the doc-stream where ``primary`` sits.

    The caller is materialising a replacement for an about-to-be-deleted
    binding whose doc-stream-first slot is ``primary``. ``new_slot``
    takes ``primary``'s position — copying its leading and sharing its
    eol — and is inserted *before* it, so the later unlink of ``primary``
    leaves ``new_slot`` exactly where it was. Being in place before the
    unlink also preserves head-occupancy for free: if ``primary`` was
    the doc head, ``new_slot`` becomes the head and the unlink never
    strips the following separator.
    """
    new_slot.leading = primary.leading
    new_slot.eol = primary.eol
    if primary._prev is None:  # noqa: SLF001
        insert_before_head(new_slot, doc)
    else:
        insert_after(primary._prev, new_slot, doc)  # noqa: SLF001


def _new_owned_section_header(
    c: Container, *, leading: str, doc: Document
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


def _bind_own_section_header(c: Container, header: StructuralHeaderSlot) -> None:
    """File an already-positioned header as ``c``'s own physical presence."""
    parent = c._parent  # noqa: SLF001
    assert parent is not None
    file_own_header(c, header)
    _extend_entry_slots(c._owner_aot_entry, header)  # noqa: SLF001
    _file_header_binding_chain(parent, header)


def _materialise_empty_section_header(
    c: Container,
    primary: Slot,
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
    header = _new_owned_section_header(c, leading=_build_section_leading(doc), doc=doc)
    _replace_primary_in_place(header, primary, doc)
    _bind_own_section_header(c, header)


def _materialise_empty_inline_table(
    c: Container,
    primary: KVSlot,
    doc: Document,
) -> None:
    """Re-materialise an empty inline table for a now-empty dotted section.

    The emptied section's physical presence was a descendant dotted *KV*
    (``a`` in ``a.b.x = 1`` once ``b`` is removed). Unlike a header, an
    inline-table binding re-parents nothing, so it can take ``primary``'s
    exact position even with sibling KVs surviving around it — the
    section renders as ``a = {}`` (or the dotted ``a.b = {}``). ``c``
    flips from an implicit section to an inline-root table backed by the
    new (empty) ``InlineTableValue``.

    Must run *before* the scrub: the new binding's chain refs are filed
    immediately ahead of ``primary``'s own refs (which the scrub then
    removes), so they inherit ``primary``'s doc-stream position.
    """
    parent = c._parent  # noqa: SLF001
    assert parent is not None
    owner = c._owner_aot_entry  # noqa: SLF001

    # Take the host from the slot being replaced rather than looking for
    # the nearest header: inside a private orphan the container standing
    # in for the host carries no header of its own, so a header search
    # would climb straight past it.
    host = c
    while host._path != primary.host_path:  # noqa: SLF001
        nxt = host._parent  # noqa: SLF001
        assert nxt is not None, "KV host must be an ancestor of its binding"
        host = nxt
    key_path = c._path[len(host._path) :]  # noqa: SLF001

    val = InlineTableValue()
    kv = _new_kv_slot(
        host_path=host._path,  # noqa: SLF001
        key=key_path,
        value=val,
        doc=doc,
        owner=owner,
        leading="",
    )
    _replace_primary_in_place(kv, primary, doc)

    # File the binding chain ``[host, ..., parent]``. ``kv`` sits ahead
    # of ``primary``, so ordered filing lands each new ref at
    # ``primary``'s doc-stream position before the scrub removes
    # ``primary``'s own refs.
    chain = _dotted_chain(host, parent)
    for i, anc in enumerate(chain):
        ref = record_ref(anc, kv)
        assert ref.local_key == key_path[i]

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


def _root_orphan_subtree(
    orphan: Document, val: Container | AoT, slots: Iterable[Slot]
) -> None:
    """Give a transplanted subtree a real home inside ``orphan``.

    The subtree keeps the path it had in the document it left, because
    its slots still spell that path. Binding it at that same path under
    ``orphan`` — synthesising the implicit tables above it — makes the
    private document self-consistent: every ``_parent`` chain ends at
    its own root.

    The new ancestors need the ref projections the builder would have
    given them: headers bind at every path ancestor, while a KV binds
    at its host and then down its dotted key, never above it.
    """
    from tomlrt._array import AoT as AoTType  # noqa: PLC0415
    from tomlrt._container import Table  # noqa: PLC0415

    # ``chain`` is the run of new ancestors, outermost first; a KV hosted
    # at one of them binds from there down.
    path = val._path  # noqa: SLF001
    chain: list[Container] = [orphan]
    for depth, part in enumerate(path[:-1], start=1):
        step = Table()
        step._wire(  # noqa: SLF001
            layout_root=orphan,
            parent=chain[-1],
            path=path[:depth],
            owner=None,
        )
        dict.__setitem__(chain[-1], part, step)
        chain.append(step)
    parent = chain[-1]
    val._host = parent  # noqa: SLF001
    dict.__setitem__(parent, path[-1], val)
    if isinstance(val, AoTType):
        # An entry's host is the container holding the AoT, not the
        # AoT itself, so entries need the same re-hosting.
        for entry in val:
            entry._host = parent  # noqa: SLF001

    depth_of = {c._path: i for i, c in enumerate(chain)}  # noqa: SLF001

    for slot in slots:
        if isinstance(slot, StructuralHeaderSlot):
            for anc in chain:
                record_ref(anc, slot)
            continue
        assert isinstance(slot, KVSlot)
        host_depth = depth_of.get(slot.host_path)
        if host_depth is None:
            continue  # hosted inside the subtree; its refs travelled with it.
        # A KV binds at its host and then down its dotted key; the rest
        # of that descent is inside the subtree and already filed.
        for anc in chain[host_depth:]:
            record_ref(anc, slot)
            maybe_advance_body_tail(anc, slot)


def delete_key(c: Container, key: str, *, materialise_empty: bool = False) -> None:
    """Delete ``key`` from ``c`` — scalar, inline, section, AoT, or dotted-subtree.

    Owned slots are scrubbed from live refs/indexes via slot
    back-pointers, body tails and ``AoTEntry.entry_slots`` are repaired,
    then the slots are unlinked. Cascade-prune is intentionally *not*
    performed: ``del c[k]`` removes exactly ``k`` and leaves any
    now-emptied implicit ancestor chain reachable as nested empty
    ``Table`` views.

    ``materialise_empty`` is opt-in for the public delete API: if the
    removal leaves ``c`` itself an empty, header-less section, a
    synthetic ``[c._path]`` header is materialised so it still renders.
    Internal delete-then-reinstall callers leave it ``False``, since
    the container is repopulated immediately and a transient empty
    state must not grow a spurious header.

    Deleted structural views are transplanted to a private orphan
    document, preserving safe mutation and later reattachment without
    touching the live document.
    """
    val = dict.__getitem__(c, key)  # raises KeyError if absent
    doc = c._attached_doc  # noqa: SLF001

    # If this empties ``c`` into a live, header-less, non-inline section,
    # a ``[c._path]`` header is synthesised below (before the unlink
    # loop, while the descendant's primary slot is still in place).
    will_materialise = (
        materialise_empty
        and bool(c._path)  # noqa: SLF001
        and not c._inline  # noqa: SLF001
        and c._header_ref is None  # noqa: SLF001
        and len(c) == 1
    )
    mat_primary: Slot | None = None
    if will_materialise:
        # ``_index[key]`` is scrubbed below; grab the removed descendant's
        # doc-stream-first slot now, while it is still linked.
        mat_primary = c._index[key][0].slot  # noqa: SLF001

    owned_ids: set[Slot] = set()
    owned_slots: list[Slot] = []

    def _add_slot(s: Slot) -> None:
        if s in owned_ids:
            return
        owned_ids.add(s)
        owned_slots.append(s)

    for r in c._index.get(key, []):  # noqa: SLF001
        _add_slot(r.slot)

    subtree_containers: list[Container] = []
    subtree_aots: list[AoT] = []
    _collect_subtree(val, subtree_containers, subtree_aots, _add_slot)

    # Synthesise the now-empty section's physical presence while the
    # descendant's primary slot is still linked, so the replacement takes
    # its position in place. The descendant's *origin* picks the form: a
    # dotted-origin section (``a.b.x = 1``) re-materialises as an inline
    # table ``a = {}``; a header-origin section (``[a.b]``) as a header
    # ``[a]``.
    if will_materialise:
        assert mat_primary is not None
        if isinstance(mat_primary, KVSlot):
            _materialise_empty_inline_table(c, mat_primary, doc)
        else:
            _materialise_empty_section_header(c, mat_primary, doc)

    # Scrub via back-pointers, *skipping* subtree containers: those move
    # to a fresh Document and keep their internal caches.
    skip_ids = frozenset(id(sc) for sc in subtree_containers)
    _scrub_owned_slots_via_backptrs(owned_slots, skip_container_ids=skip_ids)

    min_owned_depth = len(c._path)  # noqa: SLF001
    for s in owned_slots:
        d = len(s.host_path) if isinstance(s, KVSlot) else 0
        if d < min_owned_depth:
            min_owned_depth = d
    _invalidate_body_tail_chain(c, owned_ids, min_depth=min_owned_depth)

    # Unlink owned slots; transplant user-referenced subtrees to an
    # orphan Document, keeping entry_slots so clone/re-install can still
    # read the full CST.
    moving_aot_entries: set[AoTEntry] = set()
    for ao in subtree_aots:
        for entry_table in list.__iter__(ao):
            owner_e = entry_table._owner_aot_entry  # noqa: SLF001
            assert owner_e is not None
            moving_aot_entries.add(owner_e)

    candidate_owners: set[AoTEntry] = set()
    for slot in owned_slots:
        owner = slot.owner_aot_entry
        if owner is not None and owner not in moving_aot_entries:
            candidate_owners.add(owner)
    surviving_aot_entries = (
        _surviving_aot_entries(doc, candidate_owners) if candidate_owners else set()
    )
    # Capture doc-stream order *before* the unlink loop severs the linked
    # list. ``owned_slots`` is in collection order (key's own refs first,
    # then the subtree body), not doc-stream order; transplanting in that
    # order would corrupt the orphan's linked list.
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
            and owner in surviving_aot_entries
            and owner not in moving_aot_entries
        ):
            with contextlib.suppress(ValueError):
                _pop_or_remove(owner.entry_slots, slot)
    # Unlink in *reverse* doc-stream order (see remove_aot_entry for the
    # same idiom): unlinking a doc-stream-first owned slot promotes its
    # successor to the new doc head, stripping that successor's leading
    # blank line. If that successor is itself about to be unlinked too,
    # the strip is wasted and the *actually* surviving new head never
    # gets stripped. Working back-to-front lands any head-promotion
    # strip on the true surviving successor.
    for slot in reversed(ordered_for_transplant):
        unlink_slot(slot, doc)

    if transplanting:
        from tomlrt._container import Document  # noqa: PLC0415

        # Keep inline descendants live against the orphan: their backing
        # CST lives inside a transplanted KV, so re-pointing (rather than
        # resetting) lets edits through a held reference flow into the
        # orphaned slot value, which a later rehome moves intact.
        displaced = _displaced_inline_views(val)
        orphan = Document()
        orphan._newline = doc._newline  # noqa: SLF001
        orphan._is_private = True  # noqa: SLF001
        _splice_block_after(ordered_for_transplant, None, orphan)
        for view in itertools.chain(subtree_containers, subtree_aots, displaced):
            view._layout_root = orphan  # noqa: SLF001
        _root_orphan_subtree(orphan, val, ordered_for_transplant)
    else:
        # No orphan (e.g. a top-level inline value): reset so a held
        # reference reports detached and can re-attach cleanly.
        reset_displaced_views(val)

    dict.__delitem__(c, key)


def _walk_view_tree(vals: Iterable[object], visit: Callable[[_View], None]) -> None:
    """Visit every view node in the given subtrees.

    Each caller supplies the per-node action. Descent is delegated to
    `_View._view_children`, so this needs no knowledge of, or deferred
    import of, the concrete view classes. Scalars are inert.

    Takes a batch of roots so a caller displacing a range — clearing an
    array, deleting a slice — makes one traversal rather than one per
    element.
    """

    def walk(node: _View) -> None:
        visit(node)
        for child in node._view_children():  # noqa: SLF001
            if isinstance(child, _View):
                walk(child)

    for val in vals:
        if isinstance(val, _View):
            walk(val)


def _displaced_inline_views(val: object) -> list[Container | Array]:
    """The inline views inside an about-to-be-displaced subtree.

    Section Containers and AoTs are handled by ``_collect_subtree``
    + the orphan-rehome step. This walker complements that by
    reaching into inline tables and inline arrays — which carry no
    doc-stream slots of their own but do hold ``_layout_root`` /
    ``_attached`` state that goes stale when their hosting KV is
    deleted.
    """
    found: list[Container | Array] = []

    def visit(node: _View) -> None:
        if is_inline_value(node):
            found.append(node)

    _walk_view_tree((val,), visit)
    return found


def _reset_view(node: _View) -> None:
    node._reset_displaced()  # noqa: SLF001


def reset_displaced_views(*vals: object) -> None:
    """Detach every inline-table / array view nested inside ``vals``.

    Used when a value's backing CST is replaced or removed: a nested
    view left pointing at the dead value would keep reporting as
    attached and resolve against it. Resetting the whole subtree — not
    just its root — lets each view re-attach live elsewhere. Takes the
    whole batch at once so displacing a range makes one traversal.
    """
    _walk_view_tree(vals, _reset_view)


def _collect_subtree(
    val: object,
    containers_out: list[Container],
    aots_out: list[AoT],
    add_slot: Callable[[Slot], None],
) -> None:
    """Walk ``val``'s container subtree, collecting containers, AoTs and owned slots.

    Only ``Container``/``AoT`` values can ever match below (an inline
    array's contents never own doc-stream slots of their own, and are
    handled separately by ``_walk_view_tree``), so non-container leaves
    are skipped without recursing into them.
    """
    from tomlrt._array import AoT  # noqa: PLC0415
    from tomlrt._container import Container  # noqa: PLC0415

    if isinstance(val, Container):
        if val._inline:  # noqa: SLF001
            return
        containers_out.append(val)
        for r in val._refs:  # noqa: SLF001
            add_slot(r.slot)
        for child in val.values():
            if isinstance(child, (Container, AoT)):
                _collect_subtree(child, containers_out, aots_out, add_slot)
    elif isinstance(val, AoT):
        aots_out.append(val)
        placeholder = _empty_aot_placeholder_ref(val)
        if placeholder is not None:
            add_slot(placeholder.slot)
        for entry in val:
            _collect_subtree(entry, containers_out, aots_out, add_slot)


def _owned_slots_ordered(start: Slot, owned: set[Slot]) -> list[Slot]:
    """Collect ``owned`` in true doc-stream order, anchored at ``start``.

    ``start`` is typically a binding's own header/primary slot, and
    usually — but not always — the owned set's doc-stream-first slot: a
    nested descendant's header or dotted KV may have been written
    physically *earlier* (legal TOML, e.g. a sub-table ``[a.b]``
    followed later by its parent's own ``[a]``). Walking forward from
    ``start`` finds every owned slot that follows it; whatever forward
    can't reach must precede it, so a walk back from ``start`` collects
    the shortfall. Interleaved foreign slots are skipped either way (a
    binding's slots need not be contiguous — ``[[a]] … [b] … [[a]]`` is
    legal).

    No backward step happens at all once the forward walk has found
    everything, and the rare shortfall walks back only as far as the
    missing slots, never the whole document.
    """
    forward: list[Slot] = []
    seen: set[Slot] = set()
    cur: Slot | None = start
    while cur is not None and len(seen) < len(owned):
        if cur in owned:
            forward.append(cur)
            seen.add(cur)
        cur = cur._next  # noqa: SLF001
    missing = len(owned) - len(seen)
    if not missing:
        return forward
    backward: list[Slot] = []
    cur = start._prev  # noqa: SLF001
    while cur is not None and len(backward) < missing:
        if cur in owned:
            backward.append(cur)
        cur = cur._prev  # noqa: SLF001
    assert len(backward) == missing, "owned slot unreachable from start"
    backward.reverse()
    return backward + forward


def _surviving_aot_entries(doc: Document, candidates: set[AoTEntry]) -> set[AoTEntry]:
    """Return entries from ``candidates`` still reachable in ``doc``.

    Bails out as soon as every candidate has been spotted.
    """
    from tomlrt._array import AoT  # noqa: PLC0415
    from tomlrt._container import Container  # noqa: PLC0415

    surviving: set[AoTEntry] = set()
    remaining = set(candidates)

    def visit(v: object) -> None:
        if isinstance(v, Container):
            owner = v._owner_aot_entry  # noqa: SLF001
            if owner is not None and owner in remaining:
                surviving.add(owner)
                remaining.discard(owner)
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


def _last_body_kv(c: Container) -> KVSlot | None:
    """``c``'s latest body-region KV, or ``None`` if its body holds none.

    A cache read, not a search: ``_body_tail`` is maintained as exactly
    this slot, and holds ``c``'s own header instead only when the body
    has no KV at all — which is what `_recompute_body_tail` derives.
    """
    tail = c._body_tail  # noqa: SLF001
    return tail if isinstance(tail, KVSlot) else None


def _aot_sibling_last_kv(c: Container) -> KVSlot | None:
    """Return the last body KV of the most recent prior AoT sibling.

    Used to inherit indent when ``c`` is an AoT entry root with no body
    KV of its own yet.
    """
    owner = c._owner_aot_entry  # noqa: SLF001
    if owner is None:
        return None
    parent = c._parent  # noqa: SLF001
    # Only the document root is parentless or pathless, and it never
    # has an owner.
    assert parent is not None
    assert c._path  # noqa: SLF001

    from tomlrt._array import AoT  # noqa: PLC0415

    # An unbound key answers this the same way a non-AoT one does: an
    # entry whose array has been emptied has no siblings to inherit
    # from either.
    aot = dict.get(parent, c._path[-1])  # noqa: SLF001
    if not isinstance(aot, AoT):
        return None
    found_self = False
    for entry_table in reversed(aot):
        if entry_table is c:
            found_self = True
            continue
        if not found_self:
            continue
        sib = _last_body_kv(entry_table)
        if sib is not None:
            return sib
    return None


def _peer_separator(prev_leading: str | None, doc: Document) -> str:
    """Mirror a peer's blank-gap when emitting a new structural sibling.

    Returns a single blank-line newline iff ``prev_leading`` itself
    contains a blank line, or when there is no peer to mirror (the
    conventional default for the first sibling of its kind). Otherwise
    returns the empty string.

    Callers supply the kind-specific peer lookup and any extra
    decoration, such as a KV's indent.
    """
    if prev_leading is None or leading_has_blank_line(prev_leading):
        return doc._newline  # noqa: SLF001
    return ""


def _kv_leading_after(
    prev: KVSlot | None, doc: Document, fallback_indent: str = ""
) -> str:
    """Build leading trivia for a new KV slot following ``prev``.

    Inherits indent from ``prev`` and mirrors its blank-gap so the
    new KV continues the user's most recent spacing convention. With
    no prior sibling, falls back to a bare ``fallback_indent``.
    """
    if prev is None:
        return fallback_indent
    return _peer_separator(prev.leading, doc) + trailing_ws(prev.leading)


def _kv_separator_leading(c: Container, doc: Document) -> str:
    """Pick leading trivia for a new direct-KV slot in container ``c``.

    The new slot lands straight after ``c``'s body tail, so that KV is
    the peer whose indent and blank-gap it continues — the same
    question `install_dotted_kv_slot` asks, and neither cares whether
    the peer is dotted. For an AoT entry with no body KV of its own
    yet, falls back to inheriting indent (only) from the previous
    sibling entry's last one.
    """
    last = _last_body_kv(c)
    if last is not None:
        return _kv_leading_after(last, doc)
    sibling = _aot_sibling_last_kv(c)
    fallback = trailing_ws(sibling.leading) if sibling is not None else ""
    return _kv_leading_after(None, doc, fallback_indent=fallback)


def _new_kv_slot(
    *,
    host_path: tuple[str, ...],
    key: tuple[str, ...],
    value: Value,
    doc: Document,
    owner: AoTEntry | None,
    leading: str,
    key_parts: Sequence[KeyPart] | None = None,
    key_seps: Sequence[str] | None = None,
) -> KVSlot:
    """Synthesise a fresh KV slot (recorded when spliced, not here).

    By default ``key_parts`` and ``key_seps`` use canonical synthetic
    spelling. Callers moving an existing value may supply source spelling.
    """
    return KVSlot(
        leading,
        owner,
        _default_eol(doc),
        host_path,
        make_keyparts(key) if key_parts is None else tuple(key_parts),
        (".",) * (len(key) - 1) if key_seps is None else tuple(key_seps),
        " ",
        " ",
        value,
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
    leading: str | None = None,
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

    chain = _dotted_chain(host, leaf_parent)
    assert len(chain) == len(leaf_keypath)

    body_tail = leaf_parent._body_tail or host._body_tail  # noqa: SLF001
    owner = host._owner_aot_entry  # noqa: SLF001

    new_slot = _new_kv_slot(
        host_path=host._path,  # noqa: SLF001
        key=leaf_keypath,
        value=value,
        doc=doc,
        owner=owner,
        leading=leading
        if leading is not None
        else _kv_leading_after(_last_body_kv(host), doc),
        key_parts=key_parts,
        key_seps=key_seps,
    )

    _splice_body_slot(
        new_slot,
        anchor_body_tail=body_tail,
        doc=doc,
    )

    # A cached tail can lie on either side of this newly-spliced slot, so
    # each ancestor advances its own only when nothing of its own is
    # filed in between.
    for i, anc in enumerate(chain):
        predecessor = _filed_predecessor(anc, new_slot)
        ref = record_ref(anc, new_slot)
        assert ref.local_key == leaf_keypath[i]
        if predecessor is anc._body_tail or anc._body_tail is None:  # noqa: SLF001
            anc._body_tail = new_slot  # noqa: SLF001

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

    if anchor_slot is not None:
        adopted_leading = anchor_slot.leading
        original_pred = anchor_slot._prev  # noqa: SLF001
        new_descendant_leading = _build_section_leading(doc)
        header_slot = _new_owned_section_header(c, leading=adopted_leading, doc=doc)
        insert_before(anchor_slot, header_slot, doc)
        recorder = doc._install_recorders  # noqa: SLF001
        if recorder is not None:
            recorder[1].append((anchor_slot, anchor_slot.leading, original_pred))
        anchor_slot.leading = new_descendant_leading
    else:
        host_tail = _nearest_header_host_tail(c)
        header_slot = _new_owned_section_header(
            c, leading=_build_section_leading(doc), doc=doc
        )
        # Keep an anchorless promoted header inside its nearest
        # header-bearing host, not under an unrelated document tail.
        _splice_block_after([header_slot], host_tail, doc)
        if isinstance(header_slot._prev, StructuralHeaderSlot):  # noqa: SLF001
            header_slot.leading = ""

    parent = c._parent  # noqa: SLF001
    assert parent is not None
    _file_header_binding_chain(parent, header_slot)

    new_kv = _file_synthetic_header_and_kv(
        c,
        header_slot=header_slot,
        key=key,
        value=value,
        doc=doc,
        owner=owner,
    )

    _extend_entry_slots(owner, header_slot, new_kv)


def _terminate_unless_tail(slot: Slot, doc: Document) -> None:
    """Ensure ``slot`` has a trailing newline, unless it is now the doc tail.

    A slot cloned or moved from a no-final-newline source (it was
    previously the very last slot there) can arrive with no trailing
    newline of its own. That's fine if it lands at this doc's own tail
    too, but anywhere else it now runs into whatever follows on the
    same line.
    """
    if slot is not doc._tail:  # noqa: SLF001
        ensure_terminator(slot, doc._newline)  # noqa: SLF001


def _ensure_leading_blank_line(slot: Slot, doc: Document) -> None:
    """Ensure ``slot.leading`` begins with a blank line.

    A leading run starts with a blank line when its first line is
    blank. If a comment comes first, prepend a fresh newline so the
    comment block stays visually detached from the slot.
    """
    leading = slot.leading
    if "\n" in leading and "#" not in leading.split("\n", 1)[0]:
        return
    slot.leading = doc._newline + slot.leading  # noqa: SLF001


def _recompute_body_tail(c: Container) -> Slot | None:
    """Last body-region ref's slot in ``c._refs`` (mirrors invariants rule).

    The one query the ``_body_tail`` cache cannot answer, and so the
    only reverse walk of ``c._refs``: it runs exactly when the cache
    has been invalidated by a delete or a move.

    No host-path filter is needed: a KV's refs propagate from its host
    container *down* its dotted path, so a KV under ``[a.b]`` is filed
    on ``a.b``, never on ``a``. A host container therefore only ever
    sees KVs hosted at its own path, and an implicit dotted container —
    the one shape that does see foreign-host KVs — wants them all.
    """
    owner = c._owner_aot_entry  # noqa: SLF001
    for ref in reversed(c._refs):  # noqa: SLF001
        s = ref.slot
        if isinstance(s, KVSlot) and s.owner_aot_entry is owner:
            return s
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
    leading: str,
    doc: Document,
    entry: AoTEntry | None = None,
    owner_aot_entry: AoTEntry | None = None,
) -> StructuralHeaderSlot:
    return StructuralHeaderSlot(
        leading,
        owner_aot_entry,
        _default_eol(doc),
        make_keyparts(path),
        (".",) * (len(path) - 1),
        "",
        "",
        entry,
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
    owner is needed to disambiguate same-level slots. A slot whose path
    strictly extends ``base_path`` is always in-extent; a slot at
    exactly ``base_path`` is in-extent only if it shares ``base_owner``
    — otherwise it is a sibling AoT entry at the same level.

    Valid only while walking a physically contiguous doc-stream region;
    must not filter an ``_index`` bucket, which can interleave
    descendants from sibling AoT entries.
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


def _nearest_header_host_tail(c: Container) -> Slot | None:
    """Return the subtree tail of ``c``'s nearest header-bearing host."""
    return _parent_subtree_tail(_nearest_header_host(c))


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
        _parent_subtree_tail(parent) or _nearest_header_host_tail(parent)
    )


def _splice_block_after(slots: list[Slot], anchor: Slot | None, doc: Document) -> None:
    """Splice a contiguous, internally terminated block after ``anchor``.

    A ``None`` anchor means "no slot in ``doc`` should precede this
    block", so it goes at the end of the stream. In an empty document
    that also makes it the head, and any preamble parked in
    ``_trailing`` migrates onto its leading.
    """
    if not slots:
        return
    tail = doc._tail if anchor is None else anchor  # noqa: SLF001
    if tail is None:
        insert_before_head(slots[0], doc)
        _promote_trailing_to_preamble(doc)
    else:
        ensure_terminator(tail, doc._newline)  # noqa: SLF001
        insert_after(tail, slots[0], doc)
    prev = slots[0]
    for s in slots[1:]:
        ensure_terminator(prev, doc._newline)  # noqa: SLF001
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
    # A placeholder that ends the document keeps its header: demotion
    # hands its leading trivia to the successor, and with none there is
    # nowhere to put it. That happens when the block whose attachment
    # prompted the demote landed ahead of the placeholder — an AoT
    # append past entry 0 anchors after its predecessor's subtree, which
    # can precede a placeholder synthesised at the document tail.
    # Every slot is a KVSlot or a StructuralHeaderSlot, so the header's
    # body — which runs to the next header or EOF — is non-empty iff the
    # very next slot is a KVSlot.
    successor = header._next  # noqa: SLF001
    if successor is None or isinstance(successor, KVSlot):
        return
    layout_root = parent._layout_root  # noqa: SLF001
    from tomlrt._container import Document  # noqa: PLC0415

    assert isinstance(layout_root, Document)
    doc = layout_root
    # Hand the demoted header's leading trivia (its separation-from-above,
    # plus any file preamble / comments it carries at doc head) off to
    # the successor so nothing is silently dropped on promotion to
    # implicit. The successor's own leading was a separator *from the
    # header* — now redundant — so strip it first, or the transfer
    # stacks a second blank line before the successor.
    unlink_slot(header, doc, strip_new_head_leading=True)
    _strip_leading_blank_lines(successor)
    successor.leading = header.leading + successor.leading
    parent._body_tail = None  # noqa: SLF001
    # Bulk-scrub via the header's back-pointer list: drops ``hdr_ref``
    # from ``parent._refs`` (clearing ``parent._header_ref`` as a side
    # effect), drops binding refs from every ancestor, and empties
    # ``header._refs`` so the orphaned slot leaves no stale back-pointers.
    _scrub_owned_slots_via_backptrs([header])
    owner = header.owner_aot_entry
    if owner is not None:
        with contextlib.suppress(ValueError):
            owner.entry_slots.remove(header)


def _split_leading_structural(leading: str) -> tuple[str, str]:
    """Split a leading-trivia stream into (above-blank prefix, slot-remainder).

    The slot-remainder is the attached comment block (immediately above
    the slot, with no blank line between) plus the slot's own column-
    offset indent. The positional prefix is everything before it.

    Used by reorder paths to decide which prefix travels with the slot
    under move and which is positional (separator) trivia at the seam.
    """
    above, attached, indent = _split_attached_block(leading)
    return above, attached + indent


def _split_leading_for_reorder(slot: Slot) -> tuple[str, str]:
    """Reorder-aware leading split: disjoint comment blocks travel with the slot.

    Per the public ownership model (``Table.header_leading_block``,
    ``Container.leading_block``), an above-blank comment block
    immediately preceding a slot is part of that slot's leading and
    must travel with it under reorder.
    """
    leading = slot.leading
    cut = leading.split("#", 1)[0].rfind("\n") + 1
    return leading[:cut], leading[cut:]


def _retarget_separator(slot: Slot, new_separator: str) -> None:
    """Replace ``slot.leading``'s positional prefix with ``new_separator``.

    See :func:`_split_leading_structural`: the slot's attached
    comments and own indent are kept; the source's positional
    prefix is dropped.
    """
    _positional, remainder = _split_leading_structural(slot.leading)
    slot.leading = new_separator + remainder


def restore_captured_leading(slot: Slot, saved: str, *, from_kv: bool) -> None:
    """Reapply the leading trivia captured from the binding ``slot`` replaces.

    Applied verbatim, except that a header replacing a KV (``from_kv``)
    that has a line above it also keeps the separator the install path
    gave it: the captured leading is a body line's, and alone would glue
    the header to whatever precedes it.

    ``slot`` must already sit at its final doc-stream position.
    """
    separates_above = (
        from_kv
        and isinstance(slot, StructuralHeaderSlot)
        and slot._prev is not None  # noqa: SLF001
        and not leading_has_blank_line(saved)
    )
    slot.leading = (slot.leading if separates_above else "") + saved


def _build_section_leading(doc: Document) -> str:
    """Trivia for a fresh section header.

    Empty doc → no leading; otherwise use the document's stable
    structural-header spacing convention learned when it was parsed
    (or the canonical blank-separated default for a fresh document).
    """
    if doc._head is None:  # noqa: SLF001
        return ""
    return doc._newline if doc._section_blank_separated else ""  # noqa: SLF001


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
    source_aot._host = parent  # noqa: SLF001
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
    parent = aot._host  # noqa: SLF001
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
    parent = aot._host  # noqa: SLF001
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
    """
    if ordinal != 0:
        return
    ref = _empty_aot_placeholder_ref(aot)
    if ref is None:
        return
    parent = aot._host  # noqa: SLF001
    assert parent is not None
    doc = aot._attached_doc  # noqa: SLF001
    slot = ref.slot
    _scrub_owned_slots_via_backptrs([slot])
    min_depth = len(slot.host_path) if isinstance(slot, KVSlot) else 0
    _invalidate_body_tail_chain(parent, {slot}, min_depth=min_depth)
    owner = slot.owner_aot_entry
    if owner is not None:
        with contextlib.suppress(ValueError):
            owner.entry_slots.remove(slot)
    unlink_slot(slot, doc)


def _aot_separator(aot: AoT, doc: Document) -> str:
    """Pick the leading-trivia for a newly-appended AoT entry header.

    Mirrors the most recent entry's blank-gap; for the first entry,
    defaults to one blank line.
    """
    if len(aot) <= 1:
        return _peer_separator(None, doc)
    last_entry = aot[-1]._owner_aot_entry  # noqa: SLF001
    assert last_entry is not None
    return _peer_separator(last_entry.header.leading, doc)


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

    parent = aot._host  # noqa: SLF001
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
    # Splice before filing: a ref is placed by its slot's order key, so
    # the slot has to be in the doc-stream first.
    _splice_block_after([header], _aot_append_anchor(aot), doc)
    file_own_header(entry_table, header)
    _file_header_binding_chain(parent, header)
    list.append(aot, entry_table)

    # An entry header under a synthetic placeholder section makes that
    # placeholder redundant — the dotted-implicit anchor lives entirely
    # in `[[tool.list]]`.
    _maybe_demote_synthetic_empty_header(parent)

    for k, v in body_items:
        if not (is_scalar(v) or _is_synth_inline(v)):
            entry_table._setitem_validated(k, v)  # noqa: SLF001
            continue
        cst, dec = _synth_value(
            v,
            layout_root=doc,
            parent=entry_table,
            name=k,
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
        # ``ensure_implicit_chain`` is a no-op for an empty sub-path,
        # so this covers both the direct-child and dotted-descendant
        # cases without a separate ``len(key_path) == 1`` branch.
        leaf_parent = ensure_implicit_chain(target, key_path[:-1])
        leaf = key_path[-1]
        decoded = _decode_value(value, doc, leaf_parent, leaf, owner)
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
    _extend_header_bindings_to_root(parent, cloned_slots)
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

    ``src_slots[0]`` must be the entry's own header: an array entry's
    content can never physically precede its own ``[[..]]``/``[..]``
    header (unlike a plain table's, which a nested descendant can
    forward-declare).

    ``rewrite_separator``: if True, the source's structural leading is
    replaced with destination-style preamble (entry 0) or the AoT's
    inter-entry separator (entry > 0). :func:`clone_aot` sets this False
    past entry 0 so source separators survive.
    """
    from tomlrt._container import Table  # noqa: PLC0415

    parent = aot._host  # noqa: SLF001
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
        _retarget_separator(cloned_header, _build_section_leading(doc))
    elif rewrite_separator:
        _retarget_separator(cloned_header, _aot_separator(aot, doc))
    # else: keep source leading verbatim (bulk-clone past entry 0).

    append_anchor = _aot_append_anchor(aot)
    entry_table = Table()
    _install_cloned_structural_block(
        entry_table,
        parent=parent,
        doc=doc,
        target_path=target_path,
        owner=new_entry,
        cloned_slots=cloned_slots,
        anchor=append_anchor,
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
    _retarget_separator(first, _build_section_leading(doc))
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
    """Wire a cloned section block under ``parent[key]`` and store it."""
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
    return _owned_slots_ordered(start, _owned_slots(root))


def _owned_slots(root: Container) -> set[Slot]:
    """Return every slot owned by ``root``'s subtree.

    ``Slot`` is identity-hashable (``eq=False``), so a plain ``set[Slot]``
    is both correct and faster to build/query than wrapping each slot
    in ``id()`` — unlike ``Container``, which is an unhashable ``dict``
    subclass and genuinely needs ``id()`` for this purpose.
    """
    owned: set[Slot] = set()
    _collect_subtree(root, [], [], owned.add)
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


def detach_aot_from_orphan(value: AoT) -> None:
    """Cut a private-orphan AoT loose from the document it lives in.

    Its entries are cloned into the destination rather than moved, so
    the orphan must stop naming them: otherwise a later adopt of the
    orphan would gather slots that now live in the destination. An
    array emptied by :meth:`AoT.pop` still renders as ``k = []`` and
    has no entry left to carry that slot away, so it goes here too.

    A genuinely detached source has nothing to cut loose from.
    """
    if value._layout_root is None:  # noqa: SLF001
        return
    if not value:
        ref = _empty_aot_placeholder_ref(value)
        assert ref is not None, "an attached empty AoT renders as `k = []`"
        _detach_from_source_doc(value, [ref.slot])
    owned: set[Slot] = set()
    _collect_subtree(value, [], [], owned.add)
    _unfile_stale_same_orphan_ancestors(value, owned)
    value._unbind_from_document()  # noqa: SLF001


def _unfile_stale_same_orphan_ancestors(
    value: Container | AoT, target_slots: Iterable[Slot]
) -> None:
    """Drop ``value``'s bindings from its old same-orphan ancestor chain.

    A detached orphan retains its internal refs. Moving a nested value
    out must scrub those refs up to, but not beyond, the orphan root.
    Slot back-pointers avoid scanning every ancestor's complete cache.

    Scrubbing can strand an ancestor's cached ``_body_tail`` on a slot
    that is no longer filed there, so the chain is revalidated after —
    the same repair the delete path makes for the same reason.
    """
    from tomlrt._array import AoT  # noqa: PLC0415
    from tomlrt._container import Container  # noqa: PLC0415

    # A private orphan is rooted at the path its slots spell, so every
    # value inside one has a parent within the same document.
    old_parent = value._host  # noqa: SLF001
    assert isinstance(old_parent, Container)
    assert old_parent._layout_root is value._layout_root  # noqa: SLF001
    # `_host` is always the immediate path parent, so the key to drop
    # is the last path component.
    assert len(value._path) == len(old_parent._path) + 1  # noqa: SLF001
    key = value._path[-1]  # noqa: SLF001
    bound = dict.get(old_parent, key)
    # `value` may be one entry of an AoT bound here — an entry's `_path`
    # names the *array*, not itself — in which case the key outlives its
    # departure and only the entry goes.
    entries: list[Table] = bound if isinstance(bound, AoT) else []
    entry_index = next((i for i, entry in enumerate(entries) if entry is value), None)
    if entry_index is None:
        dict.pop(old_parent, key, None)
    else:
        list.__delitem__(entries, entry_index)
        if not entries:
            # The last entry has gone, so the array leaves the model too
            # — and must stop being a view onto the orphan, or a caller
            # still holding it would add entries to a document that no
            # longer names it, which would then render what it denies.
            dict.__delitem__(old_parent, key)
            assert isinstance(bound, AoT)
            bound._unbind_from_document()  # noqa: SLF001

    stale_container_ids: set[int] = set()
    node: Container | None = old_parent
    while node is not None and node._layout_root is value._layout_root:  # noqa: SLF001
        stale_container_ids.add(id(node))
        node = node._parent  # noqa: SLF001
    unfiled: set[Slot] = set()
    for slot in target_slots:
        for ref in list(slot._refs):  # noqa: SLF001
            if id(ref.container) in stale_container_ids:
                unfile_ref(ref)
                unfiled.add(slot)
    _invalidate_body_tail_chain(old_parent, unfiled)


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

    _, slots = _gather_headered_subtree_slots(value)
    _detach_from_source_doc(value, slots)
    # Every header in the subtree — value's own included — retains
    # bindings to the orphan's old ancestors.
    headers = [s for s in slots if isinstance(s, StructuralHeaderSlot)]
    _unfile_stale_same_orphan_ancestors(value, headers)
    for s in slots:
        _retarget_slot_paths(s, old_prefix, new_prefix, doc._newline)  # noqa: SLF001
        _transfer_stale_owner(s, stale_owner, new_owner)
    _rehome_view_tree(
        value, dest_parent, old_prefix, new_prefix, doc, stale_owner=stale_owner
    )

    # A forward-declared descendant may physically precede value's header.
    first = slots[0]
    assert isinstance(first, StructuralHeaderSlot)
    _retarget_separator(first, _build_section_leading(doc))
    _splice_block_after(slots, _child_header_anchor(dest_parent), doc)
    _extend_header_bindings_to_root(dest_parent, slots)
    dict.__setitem__(dest_parent, key, value)
    _maybe_demote_synthetic_empty_header(dest_parent)
    return value


def _retarget_slot_paths(
    s: Slot, src_prefix: tuple[str, ...], target_prefix: tuple[str, ...], nl: str
) -> None:
    """Rebase a slot's host / header paths + header render keys, retarget newlines.

    Owner / AoT-entry handling differs by caller and stays there.
    """
    retarget_slot_newlines(s, nl)
    if isinstance(s, KVSlot):
        s.host_path = _rebase_path(s.host_path, src_prefix, target_prefix)
        return
    # `KVSlot` and `StructuralHeaderSlot` are the only concrete slots.
    assert isinstance(s, StructuralHeaderSlot)
    s.key_parts = make_keyparts(_rebase_path(s.path, src_prefix, target_prefix))
    s.key_seps = (".",) * (len(s.key_parts) - 1)


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

    def visit(node: _View) -> None:
        # Narrows for the assignments below; `_View` has no other subclass.
        assert isinstance(node, (Container, AoT, Array))
        node._layout_root = doc  # noqa: SLF001
        if isinstance(node, (Container, AoT)):
            node._path = _rebase_path(node._path, old_prefix, new_prefix)  # noqa: SLF001
            if (
                isinstance(node, Container)
                and stale_owner is not None
                and node._owner_aot_entry is stale_owner  # noqa: SLF001
            ):
                node._owner_aot_entry = new_owner  # noqa: SLF001

    root._host = dest_parent  # noqa: SLF001
    _walk_view_tree((root,), visit)


def _detach_from_source_doc(value: Container | AoT, slots: list[Slot]) -> None:
    """Unlink ``slots`` from the private orphan they currently live in.

    The adopt paths splice the block into the destination document, which
    rewrites each slot's own links but leaves the orphan's surviving
    neighbours pointing at it. That cross-links the two documents, so a
    later walk of the orphan wanders into the destination — and any
    subsequent adopt of a remaining orphan branch collects slots it can no
    longer reach. Leading trivia is preserved: the orphan is scratch space
    and the destination sets the new block's separator itself.
    """
    src_doc = value._layout_root  # noqa: SLF001
    assert src_doc is not None, "private orphan section must be attached"
    for s in reversed(slots):
        unlink_slot(s, src_doc, strip_new_head_leading=False)


def unlink_cloned_orphan_entry(entry: Container) -> None:
    """Unlink an orphan entry's own slots once it has been cloned out.

    The AoT attach path copies an entry into the destination rather than
    relinking it. For a private orphan that leaves the originals in a
    stream nothing refers to any more, so the orphan renders entries its
    model no longer claims — and a later adopt of the orphan carries
    that stray text into the destination.
    """
    _, slots = _gather_headered_subtree_slots(entry)
    _detach_from_source_doc(entry, slots)


def synthesise_header_for_emptied(parent: Container | None) -> None:
    """Give a section a header of its own if it has been emptied.

    A parent left bound in its document but backed by nothing can
    neither render nor accept a later write. A header restores both,
    and — unlike the inline table the public delete would use for a
    dotted-origin parent — leaves it able to take section children.

    Does nothing for a parent that is not in that state.
    """
    if parent is None:
        return
    doc = parent._layout_root  # noqa: SLF001
    # A detached parent has no stream to render into; the document root
    # renders whatever it holds; and a parent that still owns a slot or
    # a child was not emptied. An inline parent cannot arrive here: a
    # section value never lived inside one.
    if (
        doc is None
        or not parent._path  # noqa: SLF001
        or parent._refs  # noqa: SLF001
        or len(parent) != 0
    ):
        return
    assert not parent._inline  # noqa: SLF001
    tail = _nearest_header_host_tail(parent)
    header = _new_owned_section_header(
        parent, leading=_build_section_leading(doc), doc=doc
    )
    # The repair is not part of whatever install is in flight above it.
    with _suspend_install_recording(doc):
        _splice_block_after([header], tail, doc)
    _bind_own_section_header(parent, header)


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

    # `_attach_section` only dispatches here for an orphan that still owns
    # slots; a slotless one is synthesised instead.
    assert value._refs, "implicit orphan has no slots"  # noqa: SLF001
    slots = _owned_slots_from(value, value._refs[0].slot)  # noqa: SLF001
    _detach_from_source_doc(value, slots)
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
    # direct body rather than its descendant-inclusive subtree. A null
    # anchor would inherit the scope of an unrelated later header, so
    # the block becomes the document head instead.
    anchor = host._body_tail  # noqa: SLF001
    to_head = anchor is None and doc._head is not None  # noqa: SLF001
    # The block carries a separator sized for the orphan it came from,
    # so it is resized for the run it is joining. A head has nothing
    # before it to be separated from, so there the separator goes
    # altogether, taking a header's above-blank comment block with it;
    # a KV owns its own such block, so that stays.
    first = slots[0]
    if isinstance(first, StructuralHeaderSlot):
        _retarget_separator(first, "" if to_head else _build_section_leading(doc))
    elif to_head:
        first.leading = _split_leading_for_reorder(first)[1]

    if to_head:
        old_head = doc._head  # noqa: SLF001
        assert old_head is not None
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
    # propagate all the way to the document root, and KVs under a nested
    # sub-section stay filed within value's subtree.
    chain = _dotted_chain(host, dest_parent)
    for s in slots:
        if isinstance(s, StructuralHeaderSlot):
            _file_header_binding_chain(dest_parent, s)
            continue
        if isinstance(s, KVSlot) and s.host_path != host_path:
            continue
        for anc in chain:
            record_ref(anc, s)
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
        s.key_seps = (".",) * (len(new_key) - 1)
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
    new_aot._host = parent  # noqa: SLF001

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

    ``head``, if given, identifies ``src_slots``' own boundary header by
    identity, not position, since it need not be ``src_slots[0]`` (see
    :func:`_owned_slots_ordered`). Its clone is returned as the second
    element with ``entry`` set to ``new_entry`` — so ``new_entry=None``
    converts an aot-entry header to a table header and vice versa.
    ``head=None`` means the list is body-only and the second element is
    ``None``.

    ``body_owner`` is written to every slot's ``owner_aot_entry`` for
    physical ownership; ``new_entry`` is the AoTEntry the clone is
    *logically* owned by.

    Nested aot-entry headers inside the body keep their AoT shape: a
    fresh `AoTEntry` is allocated per unique source entry, and cloned
    slots repointed to it, so ``_populate_entry_views`` can rebuild the
    AoT view. Without this, cross-doc whole-section copy would downgrade
    a nested ``[[a.x]]`` to a duplicated ``[a.x]`` (issue #108).
    """
    nested_entry_map: dict[AoTEntry, AoTEntry] = {}
    if head is not None and new_entry is not None:
        assert isinstance(head, StructuralHeaderSlot)
        if head.entry is not None:
            nested_entry_map[head.entry] = new_entry
    for s in src_slots:
        if s is head or not isinstance(s, StructuralHeaderSlot) or s.entry is None:
            continue
        # Each AoT entry is introduced by exactly one ``[[path]]`` header.
        assert s.entry not in nested_entry_map
        nested_entry_map[s.entry] = AoTEntry()

    cloned: list[Slot] = []
    cloned_head: StructuralHeaderSlot | None = None
    for s in src_slots:
        c: Slot = copy.deepcopy(s)
        c._prev = None  # noqa: SLF001
        c._next = None  # noqa: SLF001
        _retarget_slot_paths(c, src_prefix, target_prefix, dst_newline)
        src_owner = s.owner_aot_entry
        mapped = nested_entry_map.get(src_owner) if src_owner else None
        owner_for_slot = mapped if mapped is not None else body_owner
        c.owner_aot_entry = owner_for_slot
        if isinstance(c, StructuralHeaderSlot):
            assert isinstance(s, StructuralHeaderSlot)
            if s is head:
                # head's kind always comes from new_entry, not from
                # source-entry lookup (which is None for a plain table).
                c.entry = new_entry
            elif s.entry is not None:
                c.entry = nested_entry_map.get(s.entry)
        cloned.append(c)
        if s is head:
            assert isinstance(c, StructuralHeaderSlot)
            cloned_head = c
        # Whichever AoT entry ends up owning this slot (``owner_for_slot``)
        # must also list it in its own ``entry_slots``: callers like
        # ``remove_aot_entries`` enumerate an entry's owned slots that
        # way, not by scanning for ``owner_aot_entry``.
        if owner_for_slot is not None:
            owner_for_slot.entry_slots.append(c)

    return cloned, cloned_head


def _rebase_path(
    p: tuple[str, ...],
    src_prefix: tuple[str, ...],
    target_prefix: tuple[str, ...],
) -> tuple[str, ...]:
    """Replace a leading ``src_prefix`` in ``p`` with ``target_prefix``.

    A slot host/header path or a key-hosted view ``_path`` at or below
    the root starts with ``src_prefix`` and is rebased. An array
    element's inline-table view carries an empty ``_path`` (it has no key
    of its own) and derives its host from its array, so it does not match
    and is returned unchanged.
    """
    if src_prefix == target_prefix or p[: len(src_prefix)] != src_prefix:
        return p
    return target_prefix + p[len(src_prefix) :]


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
    # Anchor past the whole subtree of the nearest header-bearing
    # ancestor: a header re-parents everything after it, so landing it
    # mid-section would capture that host's trailing KVs (e.g. a ``d = 4``
    # sibling of an implicit ``parent``) under the new header on re-parse.
    # Splice before filing: a ref is placed by its slot's order key, so
    # the slot has to be in the doc-stream first.
    _splice_block_after([header], _nearest_header_host_tail(parent), doc)
    file_own_header(section, header)
    # Own the new header on the AoT entry so a later delete of the
    # entry takes the promoted section with it.
    _extend_entry_slots(owner, header)

    # File the binding ref under the deepest implicit parent and
    # propagate ancestor-prefix bindings up to the doc root.
    _file_header_binding_chain(deepest_parent, header)
    dict.__setitem__(deepest_parent, sub[-1], section)

    _maybe_demote_synthetic_empty_header(parent)

    # Process scalars (and synth-inlines) before nested structural
    # children. TOML requires all direct KVs of a section to appear
    # before any sub-section header. It's also a defence against header
    # demotion: the recursive ``section[k] = v`` path may demote
    # ``section``'s synthetic empty header on its first sub-section
    # attach, so scalars must populate the body (making the header
    # non-empty) first.
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
            name=k,
            owner=owner,
        )
        append_direct_kv(section, k, cst)
        dict.__setitem__(section, k, dec)
    for k, v in structurals:
        section._setitem_validated(k, v)  # noqa: SLF001
    return section


def _aot_append_anchor(aot: AoT) -> Slot | None:
    """Return the anchor for a newly-appended ``[[path]]`` entry.

    Non-empty AoTs anchor after the last entry's complete subtree,
    including nested AoTs. Empty AoTs anchor in their nearest
    header-bearing host rather than at an unrelated document tail.
    """
    if aot:
        last = aot[-1]
        assert last._owner_aot_entry is not None  # noqa: SLF001
        return _parent_subtree_tail(last)
    parent = aot._host  # noqa: SLF001
    assert parent is not None, "attached AoT must have a parent"
    # A document-tail anchor could place the first entry under a later sibling.
    return _nearest_header_host_tail(parent)


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

    The indices must already be **non-empty, non-negative, in-range,
    distinct, and ascending**; callers are responsible for normalising.
    Returns the reset popped entry ``Table``s in the same order as
    ``indices``.

    Scrubbing the union in reverse document order keeps clear and
    slice-delete linear rather than quadratic in sibling count.
    """
    from tomlrt._container import _reset_table_for_rehome  # noqa: PLC0415

    idx_list = list(indices)
    assert idx_list
    doc = aot._attached_doc  # noqa: SLF001
    parent = aot._host  # noqa: SLF001
    assert parent is not None

    # Collect each entry's whole subtree in doc-stream order and capture
    # the entry table itself for return / reset. Distinct entries own
    # disjoint subtrees, so concatenating is already a union.
    owned_per_entry: list[list[Slot]] = []
    popped_entries: list[Table] = []
    union_owned_ordered: list[Slot] = []  # in doc-stream order

    for i in idx_list:
        entry_table = aot[i]
        owned_ordered = _gather_subtree_slots(entry_table)
        union_owned_ordered.extend(owned_ordered)
        owned_per_entry.append(owned_ordered)
        popped_entries.append(entry_table)
    union_owned: set[Slot] = set(union_owned_ordered)
    assert len(union_owned) == len(union_owned_ordered)

    # Reverse order lets unfile_ref use the tail fast path in each cache.
    _scrub_owned_slots_via_backptrs(reversed(union_owned_ordered))

    # Body-tail invalidation on the parent chain, walking all the way to
    # the doc root: the popped slots' min bottom-depth is 0 (every popped
    # AoT entry includes a header), and a binding ref to an AoT entry
    # header lives at every prefix container.
    _invalidate_body_tail_chain(parent, union_owned)

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
    # `_reset_table_for_rehome`'s recurse-filter knows which children
    # belong to this subtree.
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
    dst_header = dst_entry.header

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
        entry_table._setitem_validated(k, v)  # noqa: SLF001


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

    # Collect every entry's full physical block, in current logical
    # order (which equals physical doc-stream order for AoT entries),
    # and map each surviving Table identity back to its block. The
    # ``len(aot) <= 1`` early return above guarantees at least two
    # attached entries, each of which retains its ``[[path]]`` header.
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
        assert e.entry_slots
        phys_idx_by_id[id(entry_table)] = len(physical_blocks)
        physical_blocks.append(_gather_subtree_slots(entry_table))

    region_predecessor = physical_blocks[0][0]._prev  # noqa: SLF001
    region_successor = physical_blocks[-1][-1]._next  # noqa: SLF001

    new_order_indices = [
        phys_idx_by_id[id(t)] for t in new_logical_order if id(t) in phys_idx_by_id
    ]
    output_blocks = [physical_blocks[phys_idx] for phys_idx in new_order_indices]
    movable_slots = [slot for block in physical_blocks for slot in block]
    placements = _peer_placements(physical_blocks, output_blocks)
    with _refile_region_refs(doc, region_predecessor, region_successor):
        _splice_blocks_in_order(doc, movable_slots, placements)

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


@dataclass(slots=True)
class _ReorderUnit:
    """One independently sortable slot block and its leading-trivia state."""

    slots: list[Slot]
    key_rank: int
    structural: bool
    mixed: bool
    prefix: str
    remainder: str
    physical_position: int


def _peer_placements(
    physical_blocks: list[list[Slot]], output_blocks: list[list[Slot]]
) -> list[tuple[list[Slot], str]]:
    """Pair peer blocks with positional prefixes and attached remainders."""
    prefixes: list[str] = []
    remainder_by_head: dict[Slot, str] = {}
    for block in physical_blocks:
        prefix, remainder = _split_leading_for_reorder(block[0])
        prefixes.append(prefix)
        remainder_by_head[block[0]] = remainder
    return [
        (block, prefixes[position] + remainder_by_head[block[0]])
        for position, block in enumerate(output_blocks)
    ]


def _splice_blocks_in_order(
    doc: Document,
    movable_slots: list[Slot],
    placements: list[tuple[list[Slot], str]],
) -> None:
    """Reorder movable layout blocks within the doc-stream.

    ``movable_slots`` is in original physical order. ``placements`` is
    the block grouping, order, and head leading to reinsert; callers may
    split an original logical block when its binding order must change.

    Permutes the doc-stream linked list and terminates the former final
    movable slot if it moves into the middle. Trivia policy (positional
    vs slot-attached) is the caller's responsibility — see
    ``renormalise_aot_order`` and ``reorder_container`` for the two
    existing flavours.
    """
    assert movable_slots, "both callers permute a non-empty set of blocks"

    anchor_prev = movable_slots[0]._prev  # noqa: SLF001
    former_region_tail = movable_slots[-1]
    for slot in movable_slots:
        unlink_slot(slot, doc, strip_new_head_leading=False)

    ordered: list[Slot] = []
    for block, leading in placements:
        block[0].leading = leading
        ordered.extend(block)
    _relink_run_after(anchor_prev, ordered, doc)

    _terminate_unless_tail(former_region_tail, doc)


def _slot_binding_root(slot: Slot) -> tuple[str, ...]:
    """Return the direct binding path represented by ``slot``."""
    if isinstance(slot, StructuralHeaderSlot):
        return slot.path
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
    saved_leading: str,
    *,
    from_kv: bool,
) -> None:
    """Move ``slots`` to ``saved_anchor_prev`` in the doc-stream.

    Splices the contiguous block immediately after ``saved_anchor_prev``
    (or to doc head), restores ``saved_leading`` on the new head via
    :func:`restore_captured_leading`, re-files the block's refs at their
    new doc position and repairs the cached body tails the move can have
    invalidated.

    ``slots`` must be in doc-stream order and contiguous
    (``slots[i]._next is slots[i + 1]`` for all i), and non-empty:
    :func:`_recorded_install_span` returns ``None`` rather than an
    empty span, and ``reposition_install`` bails on that before calling
    in.
    """
    doc = parent._layout_root  # noqa: SLF001
    assert doc is not None
    assert slots
    head = slots[0]
    tail = slots[-1]

    if head._prev is not saved_anchor_prev:  # noqa: SLF001
        with _refile_region_refs(doc, head._prev, tail._next):  # noqa: SLF001
            for slot in slots:
                unlink_slot(slot, doc, strip_new_head_leading=False)
            _relink_run_after(saved_anchor_prev, slots, doc)
        _invalidate_body_tail_chain(parent, None)

    restore_captured_leading(head, saved_leading, from_kv=from_kv)
    _terminate_unless_tail(tail, doc)


def _direct_child_key(
    slot: Slot, parent_path: tuple[str, ...], parent_plen: int
) -> str | None:
    """Return the direct child key of ``parent_path`` that ``slot`` binds, or None.

    Determined by the slot's full binding path: ``path`` for a
    structural header, ``(*host_path, *key_parts)`` for a KV, so a
    dotted KV like ``a.b.c = 1`` is recognised at every prefix depth,
    not just its host.
    """
    if isinstance(slot, StructuralHeaderSlot):
        root: tuple[str, ...] = slot.path
    else:
        assert isinstance(slot, KVSlot), "unknown slot type"
        root = (*slot.host_path, *[p.value for p in slot.key_parts])
    if len(root) > parent_plen and root[:parent_plen] == parent_path:
        return root[parent_plen]
    return None


def reorder_container(c: Container, new_key_order: list[str]) -> None:
    """Reorder ``c``'s direct children to ``new_key_order``.

    ``new_key_order`` is trusted to be a permutation of
    ``dict.keys(c)``. A pure leaf or structural key moves as one block;
    a mixed key splits into leaf and structural units so every mixed
    leaf stays ahead of every section header after sorting. Positional
    separators stay within their unit kind; attached comments travel
    with their unit.

    Non-contiguous keys (e.g. ``[a]; [other]; [a.sub]``, where ``a`` has
    two runs at root) are handled by collecting both runs and splicing
    them together. A foreign slot interleaved in the owned span is
    first hoisted to the region head — gathering owned blocks across it
    would shove it past a header and silently change its re-parse scope.

    An explicit ``[c]`` header moves to the start of the reordered
    region so direct KVs stay bound to ``c``. For an AoT entry, only
    slots within ``c``'s own subtree participate (see
    :func:`_owned_slots`): nested descendants move with their key, but
    same-path sibling entries are excluded.

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
        # non-table/aot-entry, non-synthetic, or synthetic-but-bearing-a-
        # body. Skipping such a header would splice its body KVs ahead
        # of every header and rebind them to the document root.
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
    membership = _owned_slots(c)
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

    # `Container.sort` only calls in when the order actually changes, so
    # at least two keys are bound, each contributing a slot.
    assert movable_slots

    movable_ids = set(movable_slots)
    earliest_owned = movable_slots[0]
    latest_owned = movable_slots[-1]
    region_predecessor = earliest_owned._prev  # noqa: SLF001
    region_successor = latest_owned._next  # noqa: SLF001

    # Foreign slots interleaved in c's owned span must keep their
    # re-parse scope. Hoist foreign KVs that still belong to c's
    # containing scope to the region head; stop at a foreign header,
    # which establishes its own scope and would capture c's dotted leaves
    # if hoisted.
    front_foreign: list[Slot] = []
    seen = 1  # earliest_owned itself
    scan: Slot | None = earliest_owned._next  # noqa: SLF001
    while scan is not None and seen < len(movable_ids):
        if scan in movable_ids:
            seen += 1
        elif isinstance(scan, StructuralHeaderSlot):
            break
        else:
            front_foreign.append(scan)
        scan = scan._next  # noqa: SLF001
    if front_foreign:
        head_structural, head_remainder = _split_leading_for_reorder(earliest_owned)
        earliest_owned.leading = head_remainder
        with _refile_region_refs(doc, region_predecessor, region_successor):
            for f in front_foreign:
                unlink_slot(f, doc, strip_new_head_leading=False)
            _relink_run_after(region_predecessor, front_foreign, doc)
        front_foreign[0].leading = head_structural + front_foreign[0].leading

    key_rank = {key: rank for rank, key in enumerate(new_key_order)}
    physical_position = {slot: pos for pos, slot in enumerate(ordered_slots)}
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
                    physical_position=physical_position[slots[0]],
                )
            )

    header_prefix = ""
    header_remainder = ""
    if header_slot is not None:
        header_prefix, header_remainder = _split_leading_for_reorder(header_slot)
        if header_slot is not earliest_owned:
            first_unit = min(units, key=lambda unit: unit.physical_position)
            header_prefix, first_unit.prefix = first_unit.prefix, header_prefix

    prefixes_by_kind: dict[tuple[bool, bool], list[str]] = {}
    for unit in sorted(units, key=lambda item: item.physical_position):
        prefixes_by_kind.setdefault((unit.structural, unit.mixed), []).append(
            unit.prefix
        )
    prefix_iterators = {
        kind: iter(prefixes) for kind, prefixes in prefixes_by_kind.items()
    }
    output_units = sorted(units, key=lambda unit: (unit.structural, unit.key_rank))

    placements: list[tuple[list[Slot], str]] = []
    if header_slot is not None:
        placements.append(([header_slot], header_prefix + header_remainder))
    for unit in output_units:
        prefix = next(prefix_iterators[(unit.structural, unit.mixed)])
        placements.append((unit.slots, prefix + unit.remainder))

    with _refile_region_refs(doc, region_predecessor, region_successor):
        _splice_blocks_in_order(doc, movable_slots, placements)
    moved_ids = movable_ids | set(front_foreign)
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
