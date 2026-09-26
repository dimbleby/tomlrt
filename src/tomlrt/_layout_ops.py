"""Mutate section-side layout.

This module owns linked-list and per-container cache updates for
direct KV insert and leaf delete. Inline-table mutation lives in
``_inline_ops.py``.

Design notes:

* The doc-stream linked list is the single source of physical ordering;
  inserts splice a slot run at an explicit anchor.
* ``c._refs`` mirrors the doc-stream subset referenced by ``c``. A
  direct KV insert's ref goes immediately after the anchor's ref (or
  at the front), not blindly at the tail where child-section refs may
  already sit.
* ``c._body_tail`` is incremental: O(1) on insert, and on deleting the
  current tail a bisect plus a walk back over ``c``'s own body. It also
  answers "what is ``c``'s last body KV?" (`_last_body_kv`), so no
  insert has to search for one.
* A non-dotted direct KV files exactly one ref on its host container;
  ancestors are unaffected.
"""

from __future__ import annotations

import bisect
import contextlib
import copy
import itertools
import operator
from collections.abc import Mapping
from typing import TYPE_CHECKING

from tomlrt import _array, _container
from tomlrt._kind import _Kind
from tomlrt._list_ops import delete_runs, index_runs
from tomlrt._scalar import SCALAR_TYPES
from tomlrt._slots import (
    AoTEntry,
    KVSlot,
    StructuralHeaderSlot,
    ensure_terminator,
    retarget_slot_newlines,
    slot_local_key,
    stitch_run,
)
from tomlrt._trivia import (
    leading_has_blank_line,
    split_line,
    strip_trailing_ws,
    trailing_ws,
)
from tomlrt._typecheck import _mapping_items
from tomlrt._values import (
    ArrayValue,
    EmptyAoTValue,
    InlineTableValue,
    make_keyparts,
    respell_key_prefix,
)
from tomlrt._view import _View, is_inline_value

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

    from tomlrt._array import AoT, Array
    from tomlrt._container import Container, Document, Table, TomlInput
    from tomlrt._slots import Slot
    from tomlrt._values import InlineTableEntry, Value


# ---------------------------------------------------------------------------
# Pure linked-list ops
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _record_install(
    doc: Document,
) -> Iterator[tuple[list[Slot], list[tuple[Slot, str, Slot | None]]]]:
    """Record slots installed and existing slots displaced by the transaction.

    :func:`_insert_run_between` records newly linked slots in the first
    yielded list, whether installed individually or as a block. The
    second captures existing slots whose leading trivia was rewritten
    by synthetic-header insertion. Slots are normally freshly materialised,
    but adopting a subtree from the same private document relinks existing
    slots — and so can record the very slot the
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
            return cur.key_path
        cur = cur._prev  # noqa: SLF001
    return None


@contextlib.contextmanager
def reposition_install(parent: Container, key: str) -> Iterator[bool]:
    """Replace ``parent[key]`` while preserving its physical position.

    Delete an existing binding, then capture the caller's installation and move
    it back to the saved anchor. Yield whether the old primary slot was a KV.
    Value preparation belongs to the caller; a failed installation is not
    rolled back.

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
    old_primary = _binding_primary_slot(parent, key)
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
        yield old_is_kv
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
    """Return the recorded survivors in physical order if they form one run.

    Ignore slots unlinked again during the transaction, such as demoted
    synthetic headers. Repairing an emptied source parent can insert an
    unrecorded header between later factory children. A move must not
    carry that repair along, so a split run stays where it was installed.
    """
    span = {slot for slot in recorded if _slot_is_linked(slot, doc)}
    assert span, "a successful install must emit slots"
    ordered: list[Slot] = []
    cur: Slot | None = min(span, key=operator.attrgetter("_order"))
    while cur is not None and cur in span:
        ordered.append(cur)
        cur = cur._next  # noqa: SLF001
    return ordered if len(ordered) == len(span) else None


def _anchor_accepts_install(
    slots: list[Slot],
    anchor: Slot | None,
    *,
    in_parent_body: bool,
    doc: Document,
) -> bool:
    """Return whether ``slots`` may be moved to sit after ``anchor``.

    ``slots`` is the nonempty, ordered, contiguous run verified by
    `_recorded_install_span`.

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

    first = slots[0]
    successor = anchor._next if anchor is not None else doc._head  # noqa: SLF001
    if successor is first:
        successor = slots[-1]._next  # noqa: SLF001
    if isinstance(successor, KVSlot):
        return False

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
    host_header = host._header  # noqa: SLF001
    if anchor_prev is None:
        return host_header is None
    if isinstance(anchor_prev, StructuralHeaderSlot):
        return anchor_prev is host_header
    assert isinstance(anchor_prev, KVSlot)
    return (
        anchor_prev.host_path == host._path  # noqa: SLF001
        and anchor_prev.owner_aot_entry is host._owner_aot_entry  # noqa: SLF001
    )


_slot_order = operator.attrgetter("_order")
"""The sort key of every container's slot projection."""


def _ordered_projections(c: Container, slot: Slot) -> tuple[list[Slot], ...]:
    """``c``'s doc-ordered slot lists: ``_refs`` + the local key's bucket.

    The container's own header has no local key and so lives in
    ``_refs`` alone; every other ref is also filed in its ``_index``
    bucket.
    """
    local_key = slot_local_key(slot, c)
    if local_key is None:
        return (c._refs,)  # noqa: SLF001
    return c._refs, c._index.setdefault(local_key, [])  # noqa: SLF001


def _ordered_index(refs: list[Slot], order: int) -> int:
    """Index at which order key ``order`` sits, or belongs, in ``refs``.

    Serves both "where is the ref for this slot?" and "where would a new
    one go?": a ref list holds at most one ref per slot, so the answers
    coincide, and no caller has to know which existing ref its new one
    follows.
    """
    return bisect.bisect_left(refs, order, key=_slot_order)


def record_slot(c: Container, slot: Slot) -> None:
    """File ``slot`` in doc order on ``c`` and register the back-pointer.

    The ``_index`` key is derived by :func:`slot_local_key` from
    ``(slot, container)`` geometry, so callers cannot file under a
    disagreeing key. ``slot`` must already be linked into the
    doc-stream, since its order key is what places the ref.

    Filing is also where ``c._body_tail`` advances. The tail is the
    latest body slot among ``c``'s refs, so a ref filed past it is
    exactly what moves it; deciding that here is what stops the two
    from disagreeing.
    """
    slot._containers.append(c)  # noqa: SLF001
    order = slot._order  # noqa: SLF001
    for refs in _ordered_projections(c, slot):
        if not refs or refs[-1]._order < order:  # noqa: SLF001
            # Filing in doc order — the builder's whole-document pass,
            # every sequential body append — lands at the tail.
            refs.append(slot)
        else:
            refs.insert(_ordered_index(refs, order), slot)
    _maybe_advance_body_tail(c, slot, order)


def _maybe_advance_body_tail(c: Container, slot: Slot, order: int) -> None:
    """Advance ``c._body_tail`` to ``slot`` if it is a later body slot.

    Owners can differ from ``c``'s: a KV owned by no AoT entry can be
    moved into a container owned by one, and then sits inside ``c``
    without belonging to its body.

    Filing runs in doc order only for an append, so the order test
    carries weight: a replacement spliced in ahead of the slot it
    supersedes, as `_materialise_empty_inline_table` does, files a body
    KV that sits before a tail which has to survive.
    """
    if not isinstance(slot, KVSlot):
        return
    if slot.owner_aot_entry is not c._owner_aot_entry:  # noqa: SLF001
        return
    tail = c._body_tail  # noqa: SLF001
    if tail is None or tail._order < order:  # noqa: SLF001
        c._body_tail = slot  # noqa: SLF001


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
    with _refile_slot_refs(_slots_between(doc, predecessor, successor)):
        yield


@contextlib.contextmanager
def _refile_slot_refs(slots: Iterable[Slot]) -> Iterator[None]:
    """Keep retained projections ordered while a slot run is moved or split."""
    runs: dict[int, tuple[list[Slot], list[Slot]]] = {}
    for slot in slots:
        for c in slot._containers:  # noqa: SLF001
            for refs in _ordered_projections(c, slot):
                # A projection holding this ref alone has nothing to
                # reorder and nowhere else to sit.
                if len(refs) > 1:
                    runs.setdefault(id(refs), (refs, []))[1].append(slot)
    placed = [
        (refs, run, _ordered_index(refs, _slot_order(run[0])))
        for refs, run in runs.values()
    ]
    for refs, run, start in placed:
        assert refs[start : start + len(run)] == run, "region refs must be one run"
    yield
    for refs, run, start in placed:
        _replace_ordered_run(refs, run, start)


def _replace_ordered_run(refs: list[Slot], run: list[Slot], start: int) -> None:
    """Put ``run``, the former slice of ``refs`` at ``start``, back in key order.

    It goes straight back where it was if its refs' order keys still sit
    between the same neighbours — the whole projection, or a region
    permuted in place — and is otherwise lifted out and placed afresh.
    """
    run.sort(key=_slot_order)
    end = start + len(run)
    if (start == 0 or _slot_order(refs[start - 1]) < _slot_order(run[0])) and (
        end == len(refs) or _slot_order(run[-1]) < _slot_order(refs[end])
    ):
        refs[start:end] = run
        return
    del refs[start:end]
    at = _ordered_index(refs, _slot_order(run[0]))
    refs[at:at] = run


def file_own_header(c: Container, header: StructuralHeaderSlot) -> None:
    """File ``header`` as ``c``'s own physical presence.

    A header cannot advance the body tail through `record_slot`, which
    only advances for a body KV, so this is one of the paths that
    establishes ``_header`` and ``_body_tail`` together. Here ``c``
    has no body yet, so its own header is the tail, which is exactly
    what `_recompute_body_tail` derives for it.
    """
    record_slot(c, header)
    c._header = header  # noqa: SLF001
    c._body_tail = header  # noqa: SLF001


def _file_header_binding_chain(
    deepest: Container,
    header: StructuralHeaderSlot,
) -> None:
    """File ``header`` in doc order on ``deepest`` and every ancestor."""
    for c in [deepest, *_ancestor_chain(deepest)]:
        record_slot(c, header)


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
    ``c._header`` / ``c._index[key]`` / ``c._body_tail``.

    Anchoring and ancestor binding-ref filing stay explicit in callers;
    both are highly position-sensitive and not safe to share.
    """
    record_slot(c, header_slot)
    c._header = header_slot  # noqa: SLF001

    new_kv = _new_kv_slot(
        host_path=c._path,  # noqa: SLF001
        key=(key,),
        value=value,
        doc=doc,
        owner=owner,
        leading="",
    )
    insert_after(header_slot, new_kv, doc)
    record_slot(c, new_kv)
    return new_kv


def ensure_implicit_chain(
    parent: Container,
    sub_path: tuple[str, ...],
) -> Container:
    """Navigate or create implicit Tables along ``sub_path`` under ``parent``.

    Returns the deepest container. Missing components become new
    implicit (header-less) tables wired into ``parent._attached_doc``;
    existing components are always ``Container`` instances.
    """
    doc = parent._attached_doc  # noqa: SLF001
    owner = parent._owner_aot_entry  # noqa: SLF001
    cur: Container = parent
    for j, comp in enumerate(sub_path):
        if comp in cur:
            nxt = dict.__getitem__(cur, comp)
            assert isinstance(nxt, _container.Container)
            cur = nxt
            continue
        implicit = _container.Table()
        implicit._wire(  # noqa: SLF001
            layout_root=doc,
            parent=cur,
            path=(*parent._path, *sub_path[: j + 1]),  # noqa: SLF001
            owner=owner,
        )
        dict.__setitem__(cur, comp, implicit)
        cur = implicit
    return cur


def _default_eol(doc: Document) -> str:
    """A bare-newline EOL run for a freshly synthesised slot."""
    return doc._newline  # noqa: SLF001


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


def _insert_run_between(
    prev: Slot | None, slots: Sequence[Slot], nxt: Slot | None, doc: Document
) -> None:
    """Splice newly installed ``slots`` and record the whole run at once."""
    _link_run_between(prev, slots, nxt, doc)
    recorder = doc._install_recorders  # noqa: SLF001
    if recorder is not None:
        recorder[0].extend(slots)


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
    _insert_run_between(anchor, (new_slot,), anchor._next, doc)  # noqa: SLF001


def insert_before(anchor: Slot, new_slot: Slot, doc: Document) -> None:
    """Splice ``new_slot`` immediately before ``anchor`` in ``doc``."""
    _insert_run_between(anchor._prev, (new_slot,), anchor, doc)  # noqa: SLF001


def insert_before_head(new_slot: Slot, doc: Document) -> None:
    """Splice ``new_slot`` at the start of ``doc``'s linked list.

    Purely mechanical; does not touch ``doc._trailing``. Callers
    inserting the very first slot into an empty doc that may carry
    preamble trivia in ``_trailing`` (e.g. set via
    :attr:`Document.preamble` or parsed from a comment-only source)
    should follow up with :func:`_promote_trailing_to_preamble`.
    """
    _insert_run_between(None, (new_slot,), doc._head, doc)  # noqa: SLF001


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
) -> None:
    """Splice ``new_slot`` into the doc-stream at the canonical body anchor.

    Anchor preference: body tail (which for a header-bearing container
    with no body yet is that header) > head-of-doc seam > empty doc.
    """
    if anchor_body_tail is not None:
        ensure_terminator(anchor_body_tail, doc._newline)  # noqa: SLF001
        insert_after(anchor_body_tail, new_slot, doc)
        return
    if doc._head is not None:  # noqa: SLF001
        # Section-only doc: splice before the first slot, separating it.
        old_head = doc._head  # noqa: SLF001
        insert_before_head(new_slot, doc)
        _ensure_leading_blank_line(old_head, doc)
        return
    # Empty doc: splice in as head, hoisting any preamble trivia.
    insert_before_head(new_slot, doc)
    _promote_trailing_to_preamble(doc)


def append_direct_kv(
    c: Container,
    key: str,
    value: Value,
    *,
    reinstall_as_dotted: bool = False,
    key_parts: tuple[str, ...] | None = None,
    key_seps: tuple[str, ...] | None = None,
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
    record_slot(c, new_slot)


def append_synth_kv(
    c: Container,
    key: str,
    v: TomlInput,
) -> None:
    """Append ``key = v`` to ``c`` as a freshly synthesised KV line."""
    cst, dec = c._synth_local_value(key, v)  # noqa: SLF001
    append_direct_kv(c, key, cst)
    dict.__setitem__(c, key, dec)


def _invalidate_body_tail_chain(
    start: Container | None,
    owned_slots: set[Slot] | None,
    *,
    min_depth: int = 0,
    departing: bool = False,
) -> None:
    """Recompute invalidated ``_body_tail`` values on the path to root.

    For each container ``cc`` along the chain whose existing
    ``_body_tail`` slot is in ``owned_slots``, recompute the tail.
    ``owned_slots`` of ``None`` means "every cached tail is suspect" —
    used after a block move, which can hand a container a later body
    slot than the one it was caching.

    ``departing`` says the named slots are leaving, so a tail among
    them can only be replaced by an earlier body slot and each
    recompute can bound its search there. A reorder keeps its slots and
    can promote a later one, so it leaves this alone.

    Stops once ``len(cc._path) < min_depth``: an ancestor at depth
    ``d`` cannot have its body_tail point at a slot whose minimum
    bottom-depth exceeds ``d``, so common-case leaf-KV deletes never
    walk past ``c`` itself.
    """
    cur = start
    while cur is not None and len(cur._path) >= min_depth:  # noqa: SLF001
        tail = cur._body_tail  # noqa: SLF001
        if tail is not None and (owned_slots is None or tail in owned_slots):
            cur._body_tail = _recompute_body_tail(  # noqa: SLF001
                cur, below=tail if departing else None
            )
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
        host._header is None  # noqa: SLF001
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
    insert_before(primary, new_slot, doc)


def _new_owned_section_header(
    c: Container, *, leading: str, doc: Document
) -> StructuralHeaderSlot:
    return _new_section_header(
        c._path,  # noqa: SLF001
        leading=leading,
        doc=doc,
        owner_aot_entry=c._owner_aot_entry,  # noqa: SLF001
    )


def _transfer_stale_owner(
    slot: Slot, stale_owner: AoTEntry | None, new_owner: AoTEntry | None
) -> None:
    if slot.owner_aot_entry is not stale_owner:
        return
    slot.owner_aot_entry = new_owner
    if isinstance(slot, StructuralHeaderSlot) and slot.entry is stale_owner:
        slot.entry = None


def _bind_own_section_header(c: Container, header: StructuralHeaderSlot) -> None:
    """File an already-positioned header as ``c``'s own physical presence."""
    parent = c._parent  # noqa: SLF001
    assert parent is not None
    file_own_header(c, header)
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
        record_slot(anc, kv)
        assert slot_local_key(kv, anc) == key_path[i]

    # ``c`` becomes an inline-root table, which keeps no ``_refs`` /
    # ``_index`` of its own (the binding lives on the parent chain, and
    # entries live in ``val.items``). Unfile its remaining
    # descendant-binding refs first — this also unregisters their slot
    # back-pointers, so the later scrub no longer reaches ``c``.
    for slot in list(c._refs):  # noqa: SLF001
        unfile_slot(c, slot)
    c._inline = True  # noqa: SLF001
    c._value = val  # noqa: SLF001
    c._body_tail = None  # noqa: SLF001


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
    # ``chain`` is the run of new ancestors, outermost first; a KV hosted
    # at one of them binds from there down.
    path = val._path  # noqa: SLF001
    chain: list[Container] = [orphan]
    for depth, part in enumerate(path[:-1], start=1):
        step = _container.Table()
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
    if isinstance(val, _array.AoT):
        for entry in val:
            entry._host = val  # noqa: SLF001

    depth_of = {c._path: i for i, c in enumerate(chain)}  # noqa: SLF001

    for slot in slots:
        if isinstance(slot, StructuralHeaderSlot):
            for anc in chain:
                record_slot(anc, slot)
            continue
        assert isinstance(slot, KVSlot)
        host_depth = depth_of.get(slot.host_path)
        if host_depth is None:
            continue  # hosted inside the subtree; its refs travelled with it.
        # A KV binds at its host and then down its dotted key; the rest
        # of that descent is inside the subtree and already filed.
        for anc in chain[host_depth:]:
            record_slot(anc, slot)


def delete_key(c: Container, key: str, *, materialise_empty: bool = False) -> None:
    """Delete ``key`` from ``c`` — scalar, inline, section, AoT, or dotted-subtree.

    Owned slots are scrubbed from live refs/indexes via slot
    back-pointers, body tails are repaired, then the slots are unlinked.
    Cascade-prune is intentionally *not*
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
        and c._header is None  # noqa: SLF001
        and len(c) == 1
    )
    mat_primary: Slot | None = None
    if will_materialise:
        # ``_index[key]`` is scrubbed below; grab the removed descendant's
        # doc-stream-first slot now, while it is still linked.
        mat_primary = c._index[key][0]  # noqa: SLF001

    owned = set(c._index.get(key, ()))  # noqa: SLF001
    if _container._is_section(val) or isinstance(val, _array.AoT):  # noqa: SLF001
        owned.update(owned_slots(val))
        views = list(_walk_views((val,)))
    else:
        views = []
    slots = sorted(owned, key=operator.attrgetter("_order"))
    del owned  # The removal step builds its own membership set.

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

    _detach_departing_slots(c, slots, views)

    if views:
        assert isinstance(val, (_container.Container, _array.AoT))
        _transplant_to_orphan(
            val,
            slots,
            doc._newline,  # noqa: SLF001
            views,
        )
    else:
        # No orphan (e.g. a top-level inline value): reset so a held
        # reference reports detached and can re-attach cleanly.
        reset_displaced_views(val)

    dict.__delitem__(c, key)


def _detach_departing_slots(
    start: Container, slots: list[Slot], views: list[_View]
) -> None:
    """Remove doc-ordered slots, keeping retained views' internal refs.

    ``views`` includes every retained descendant, including inline values.
    """
    doc = start._attached_doc  # noqa: SLF001
    owned = set(slots)
    assert len(owned) == len(slots), "departing slots must be distinct"
    skip_ids = frozenset(
        id(view)
        for view in views
        if not view._inline and isinstance(view, _container.Container)  # noqa: SLF001
    )
    _scrub_owned_slots_via_backptrs(slots, skip_container_ids=skip_ids)
    min_depth = len(start._path)  # noqa: SLF001
    for slot in slots:
        if not min_depth:
            break
        depth = len(slot.host_path) if isinstance(slot, KVSlot) else 0
        if depth < min_depth:
            min_depth = depth
    _invalidate_body_tail_chain(start, owned, min_depth=min_depth, departing=True)
    # Unlinking the document head strips leading blanks from its successor.
    # Work backwards so only a surviving slot can be promoted and changed.
    for slot in reversed(slots):
        unlink_slot(slot, doc)


def _transplant_to_orphan(
    val: Container | AoT,
    slots: list[Slot],
    nl: str,
    views: Iterable[_View],
) -> None:
    """Give ``val``'s unlinked ``slots`` a private document to live on.

    What is removed from a document keeps its own layout: the slots move
    to a document of their own rather than being dropped, so a held view
    goes on working and reinstalling it elsewhere writes the lines the
    source had rather than re-deriving them from its data.

    Inline descendants are re-pointed rather than reset: their backing
    CST lives inside a transplanted KV, so an edit through a held
    reference flows into the orphaned slot value, which a later rehome
    moves intact.

    ``slots`` must be in doc-stream order, and ``views`` must include
    every retained view, including inline descendants.
    """
    orphan = _container.Document()
    orphan._newline = nl  # noqa: SLF001
    orphan._is_private = True  # noqa: SLF001
    _splice_block_after(slots, None, orphan)
    for view in views:
        assert isinstance(view, (_container.Container, _array.AoT, _array.Array))
        view._layout_root = orphan  # noqa: SLF001
    _root_orphan_subtree(orphan, val, slots)


def _walk_views(vals: Iterable[_View]) -> Iterator[_View]:
    """Yield subtree views in preorder without recursive calls.

    Resume each parent's child iterator after its child's whole subtree.
    Scalars are skipped, and the pending stack grows only with depth.
    """
    pending: list[Iterator[object]] = [iter(vals)]
    while pending:
        for node in pending[-1]:
            if isinstance(node, _View):
                yield node
                pending.append(node._view_children())  # noqa: SLF001
                break
        else:
            pending.pop()


def _detach_displaced_inline(root: Container | Array) -> None:
    """Free CST components, keeping the bindings inside each component intact."""
    pending: list[Container | Array] = [root]
    while pending:
        current = pending.pop()
        current._host = None  # noqa: SLF001
        if isinstance(current, _array.Array) or current._value is not None:  # noqa: SLF001
            for node in _walk_views((current,)):
                node._reset_displaced()  # noqa: SLF001
        else:
            current._reset_displaced()  # noqa: SLF001
            pending.extend(
                child for child in current.values() if is_inline_value(child)
            )


def reset_displaced_views(*vals: object) -> None:
    """Detach inline roots removed from their current materialised owner.

    A CST-owning root becomes adoptable while keeping its descendants
    bound to their existing occurrences. A CST-less dotted navigator
    becomes a factory and frees each child CST component independently.
    """
    for val in vals:
        if is_inline_value(val):
            _detach_displaced_inline(val)


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
    aot = c._host  # noqa: SLF001
    if not isinstance(aot, _array.AoT):
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
    key_parts: tuple[str, ...] | None = None,
    key_seps: tuple[str, ...] | None = None,
) -> KVSlot:
    """Synthesise a fresh KV slot (recorded when spliced, not here).

    Keys use canonical spelling unless supplied, as inline promotion does.
    """
    return KVSlot(
        leading,
        owner,
        _default_eol(doc),
        host_path,
        make_keyparts(key) if key_parts is None else key_parts,
        (".",) * (len(key) - 1) if key_seps is None else key_seps,
        key,
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
    key_parts: tuple[str, ...] | None = None,
    key_seps: tuple[str, ...] | None = None,
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
    key_parts: tuple[str, ...] | None = None,
    key_seps: tuple[str, ...] | None = None,
) -> None:
    """Insert a single dotted-KV slot hosted by ``host``.

    Files refs on ``host`` and every implicit intermediate in
    ``[host, ..., leaf_parent]`` and updates ``_body_tail`` along the
    chain. The caller owns dict storage at ``leaf_parent``.

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
        leading=_kv_leading_after(_last_body_kv(host), doc),
        key_parts=key_parts,
        key_seps=key_seps,
    )

    _splice_body_slot(
        new_slot,
        anchor_body_tail=body_tail,
        doc=doc,
    )

    for i, anc in enumerate(chain):
        record_slot(anc, new_slot)
        assert slot_local_key(new_slot, anc) == leaf_keypath[i]


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
    anchor_slot = c._refs[0] if c._refs else None  # noqa: SLF001

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

    _file_synthetic_header_and_kv(
        c,
        header_slot=header_slot,
        key=key,
        value=value,
        doc=doc,
        owner=owner,
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


def _recompute_body_tail(c: Container, *, below: Slot | None = None) -> Slot | None:
    """Last body-region ref's slot in ``c._refs`` (mirrors invariants rule).

    The one query the ``_body_tail`` cache cannot answer, and so the
    only reverse walk of ``c._refs``: it runs exactly when the cache
    has been invalidated by a delete or a move.

    ``below`` is the departing tail. It held the latest body slot, so
    nothing past it in ``_refs`` is one and the walk starts from its
    place instead of the end — bisected, because the refs past it are
    child headers, of which there can be any number.

    No host-path filter is needed: a KV's refs propagate from its host
    container *down* its dotted path, so a KV under ``[a.b]`` is filed
    on ``a.b``, never on ``a``. A host container therefore only ever
    sees KVs hosted at its own path, and an implicit dotted container —
    the one shape that does see foreign-host KVs — wants them all.
    """
    refs = c._refs  # noqa: SLF001
    i = len(refs) if below is None else _ordered_index(refs, below._order)  # noqa: SLF001
    owner = c._owner_aot_entry  # noqa: SLF001
    while i:
        i -= 1
        s = refs[i]
        if isinstance(s, KVSlot) and s.owner_aot_entry is owner:
            return s
    return c._header  # noqa: SLF001


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
    header = StructuralHeaderSlot(
        leading,
        owner_aot_entry,
        _default_eol(doc),
        make_keyparts(path),
        (".",) * (len(path) - 1),
        path,
        "",
        "",
        entry,
        synthetic=True,
    )
    if entry is not None:
        entry.bind_header(header)
    return header


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
        path = slot.key_path
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
    cur = refs[-1]
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
    """Splice a block after ``anchor``, allocating its order keys as one run.

    A ``None`` anchor defaults to the document tail. Every slot that
    gains a successor needs a terminator; a block entering an empty
    document also needs separation from any preamble.
    """
    if not slots:
        return
    tail = doc._tail if anchor is None else anchor  # noqa: SLF001
    if tail is not None:
        ensure_terminator(tail, doc._newline)  # noqa: SLF001
    nxt = tail._next if tail is not None else doc._head  # noqa: SLF001
    _insert_run_between(tail, slots, nxt, doc)
    for slot in slots:
        _terminate_unless_tail(slot, doc)
    if tail is None:
        _promote_trailing_to_preamble(doc)


def _maybe_demote_synthetic_empty_header(parent: Container) -> None:
    """Drop ``parent``'s header if it is synthetic and has no direct KV body.

    Used after attaching a child header under ``parent``: if ``parent``
    was synthesised as an empty placeholder (e.g.
    ``doc["tool"] = Table.section({})``) and the new child gives it a
    dotted-implicit anchor (``[tool.poetry]``), the placeholder header
    is redundant and is removed.
    """
    header = parent._header  # noqa: SLF001
    if header is None:
        return
    if not header.synthetic or header.entry is not None:
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

    assert isinstance(layout_root, _container.Document)
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
    # Bulk-scrub via the header's back-pointer list: drops ``header``
    # from ``parent._refs`` (clearing ``parent._header`` as a side
    # effect), drops binding refs from every ancestor, and empties
    # ``header._containers`` so the orphaned slot leaves no stale back-pointers.
    _scrub_owned_slots_via_backptrs([header])


def _split_leading_trivia(slot: Slot) -> tuple[str, str]:
    """Split positional blank space from the slot's full comment block.

    Comment groups and their intervening blank lines belong to the slot
    when it is copied, moved or reordered.
    """
    leading = slot.leading
    cut = leading.split("#", 1)[0].rfind("\n") + 1
    return leading[:cut], leading[cut:]


def _retarget_separator(slot: Slot, new_separator: str) -> None:
    """Replace ``slot.leading``'s positional prefix with ``new_separator``.

    Keep all comment groups and their indentation; only the blank
    separator before them is replaced.
    """
    remainder = _split_leading_trivia(slot)[1] if slot.leading else ""
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
    _bind_aot(parent, key, source_aot)
    _materialise_empty_aot(source_aot)
    return source_aot


def _bind_aot(parent: Container, key: str, aot: AoT) -> None:
    """Wire an AoT to its host without creating or changing its slots."""
    aot._layout_root = parent._layout_root  # noqa: SLF001
    aot._path = (*parent._path, key)  # noqa: SLF001
    aot._host = parent  # noqa: SLF001


def _materialise_empty_aot(aot: AoT) -> None:
    """Splice a ``key = []`` placeholder for a now-empty attached AoT.

    The placeholder is a normal direct KV (an ``EmptyAoTValue``) under
    the AoT's parent, so it lands in the parent's body region rather
    than at a header position a re-parse would misattribute. Dict
    storage at ``parent[key]`` is left as the AoT — only the physical
    slot is created.
    """
    parent = aot._host  # noqa: SLF001
    assert parent is not None
    assert len(aot) == 0
    key = aot._path[-1]  # noqa: SLF001
    append_direct_kv(parent, key, EmptyAoTValue())


def _empty_aot_placeholder_slot(aot: AoT) -> KVSlot | None:
    """Return the ``key = []`` placeholder backing an empty AoT, if any.

    Derived from the parent's ``_index[key]``: an empty AoT's only
    physical presence is one ``KVSlot`` whose value is an empty
    ``EmptyAoTValue``. Returns ``None`` before a fresh AoT has received
    its first entry or placeholder.
    """
    assert not aot
    parent = aot._host  # noqa: SLF001
    assert parent is not None
    key = aot._path[-1]  # noqa: SLF001
    bucket = parent._index.get(key)  # noqa: SLF001
    if not bucket:
        return None
    slot = bucket[0]
    assert isinstance(slot, KVSlot), "empty AoT placeholder must be a KV slot"
    assert isinstance(slot.value, EmptyAoTValue), (
        "empty AoT key must be bound to an array placeholder"
    )
    return slot


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
    slot = _empty_aot_placeholder_slot(aot)
    if slot is None:
        return
    parent = aot._host  # noqa: SLF001
    assert parent is not None
    doc = aot._attached_doc  # noqa: SLF001
    _scrub_owned_slots_via_backptrs([slot])
    min_depth = len(slot.host_path)
    _invalidate_body_tail_chain(parent, {slot}, min_depth=min_depth, departing=True)
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
    aot: AoT,
    body: Mapping[str, TomlInput] | None,
    *,
    rehome: Table | None = None,
    preserve_source_separator: bool = False,
) -> Table:
    """Capture and append an entry, preserving a structural source's layout.

    ``rehome`` selects an unattached factory entry as the destination
    view and supplies its body. Bulk cloning can retain source separators
    after the first entry instead of deriving destination-style gaps.
    """
    if rehome is not None:
        assert rehome._layout_root is None  # noqa: SLF001
        body = rehome
    table = _container.Table() if rehome is None else rehome
    prepared = _prepare_entry(aot, table, {} if body is None else body, (aot,), {})
    return _install_entry(
        aot, prepared, preserve_source_separator=preserve_source_separator
    )


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


def _install_section_layout(
    parent: Container,
    slots: list[Slot],
    *,
    doc: Document,
    target_path: tuple[str, ...],
    existing: Container | None = None,
) -> Container:
    """Install a destination-ready, header-bearing slot run.

    An adopted ``existing`` view is already rehomed and retains its refs.
    Copies acquire fresh views after linking, when slots have order keys.
    The section's own header may follow forward-declared descendants.
    """
    section = existing
    if section is None:
        section = _container.Table.section()
        section._wire(  # noqa: SLF001
            layout_root=doc,
            path=target_path,
            parent=parent,
            owner=parent._owner_aot_entry,  # noqa: SLF001
        )
    _splice_block_after(slots, _child_header_anchor(parent), doc)
    if existing is None:
        _populate_entry_views(
            entry_table=section,
            cloned_slots=slots,
            target_prefix=target_path,
            doc=doc,
        )
    _extend_header_bindings_to_root(parent, slots)
    return section


def clone_graft_slots(
    view: Container | AoT,
    *,
    target_path: tuple[str, ...],
    host_path: tuple[str, ...],
    owner: AoTEntry | None,
    nl: str,
) -> list[Slot]:
    """Deep-clone ``view``'s block, rebased to ``target_path``.

    The clone comes back unlinked, for a caller writing a slot run of
    its own; the source is left exactly as it was. Every ``[[..]]``
    header in it gets a fresh `AoTEntry`, so an array-of-tables keeps
    one entry per header and a section that was itself an entry becomes
    a plain ``[table]``. ``host_path`` hosts the dotted keys of a
    header-less section — see :func:`_clone_entry_slots`.
    """
    own_header = view._header if isinstance(view, _container.Container) else None  # noqa: SLF001
    cloned, _head = _clone_entry_slots(
        owned_slots(view),
        new_entry=None,
        body_owner=owner,
        src_prefix=view._path,  # noqa: SLF001
        target_prefix=target_path,
        dst_newline=nl,
        head=own_header,
        host_path=host_path,
    )
    return cloned


def owned_slots(view: Container | AoT) -> list[Slot]:
    """Every slot ``view``'s block spans, in doc-stream order.

    A container's ordered refs name its own KVs and descendant headers.
    Only descendant headers need their following body runs expanded;
    the own header's body is already filed. A changed KV host also
    ends a run because a private orphan can omit an enclosing header.
    """
    if isinstance(view, _array.AoT):
        return [s for entry in view for s in owned_slots(entry)]
    if isinstance(view, _container.Document):
        slots: list[Slot] = []
        cur = view._head  # noqa: SLF001
        while cur is not None:
            slots.append(cur)
            cur = cur._next  # noqa: SLF001
        return slots
    owned: list[Slot] = []
    own_header = view._header  # noqa: SLF001
    for slot in view._refs:  # noqa: SLF001
        owned.append(slot)
        if isinstance(slot, StructuralHeaderSlot) and slot is not own_header:
            host_path = slot.key_path
            body = slot._next  # noqa: SLF001
            while isinstance(body, KVSlot) and body.host_path == host_path:
                owned.append(body)
                body = body._next  # noqa: SLF001
    return owned


def _gather_headered_subtree_slots(
    src_table: Container,
) -> tuple[StructuralHeaderSlot, list[Slot]]:
    """Collect ``src_table``'s subtree slots plus its own header, by identity.

    ``src_table._header`` — not ``src_slots[0]`` — is the
    container's own header: doc-stream order may put a forward-declared
    nested descendant's header earlier in the returned list.
    """
    src_slots = owned_slots(src_table)
    head = src_table._header  # noqa: SLF001
    assert head is not None
    return head, src_slots


def _hoist_root_level_kvs(run: list[Slot], doc: Document) -> list[Slot]:
    """Move the document root's own keys ahead of any header in ``run``.

    A re-rooted key is only in scope before the first header, and a
    forward-declared descendant (``[a.b]`` written above its own ``[a]``)
    leaves one after one. The seam that opens was never a boundary in
    the source, so it takes the document's section spacing.
    """
    body, blocks = split_subtree_slots(run, 1)
    if body == run[: len(body)]:
        return run
    _retarget_separator(blocks[0], _build_section_leading(doc))
    return body + blocks


def _promoted_header_comments(head: StructuralHeaderSlot, nl: str) -> str:
    """Render a dropped header's own comments as free-standing lines.

    Extraction discards the table's header, so the comments that would
    travel with it under reorder — its above-block and its EOL comment —
    become the extracted document's opening block instead. The trailing
    blank keeps that block from attaching itself to the first construct.
    """
    _positional, above = _split_leading_trivia(head)
    # Any trailing indent belonged to the header's own line, which is gone.
    above = strip_trailing_ws(above)
    if "#" in head.eol:
        above += f"{split_line(head.eol)[1]}{nl}"
    if not above:
        return ""
    return above if above.endswith(nl * 2) else above + nl


def extract_subtree_slots(src_table: Container) -> tuple[list[Slot], str]:
    """Clone ``src_table``'s subtree as a stand-alone document's slot run.

    Returns the cloned run — linked, rebased to a document root, and
    ordered so a re-parse sees the same shape — plus the comment text
    promoted off the table's own header, which re-rooting drops. The
    source document is left untouched.
    """
    doc = src_table._layout_root  # noqa: SLF001
    assert doc is not None, "subtree extraction requires an attached container"
    nl = doc._newline  # noqa: SLF001
    if src_table._header is not None:  # noqa: SLF001
        head, src_slots = _gather_headered_subtree_slots(src_table)
    else:
        # A header-less section is bound by its descendants' slots, and
        # an attached one always has at least one.
        assert src_table._refs, "implicit section has no slots"  # noqa: SLF001
        head = None
        src_slots = owned_slots(src_table)

    cloned, cloned_head = _clone_entry_slots(
        src_slots,
        new_entry=None,
        body_owner=None,
        src_prefix=src_table._path,  # noqa: SLF001
        target_prefix=(),
        dst_newline=nl,
        head=head,
    )
    promoted = ""
    if cloned_head is not None:
        promoted = _promoted_header_comments(cloned_head, nl)
        cloned = [s for s in cloned if s is not cloned_head]
    cloned = _hoist_root_level_kvs(cloned, doc)
    if cloned:
        # The run starts a document of its own: it keeps the comment
        # block it owns but not the separator that positioned it, and
        # the source document's final line may lack a terminator.
        _retarget_separator(cloned[0], "")
        for s in cloned[:-1]:
            ensure_terminator(s, nl)
    stitch_run(None, cloned, None)
    return cloned, promoted


def clone_section(
    parent: Container,
    key: str,
    source: Container,
) -> Container:
    """Clone a header-bearing table or document under ``parent[key]``.

    A table keeps its physical order and spelling, with an AoT entry's
    own header normalised to a section. A document supplies its body,
    without its file envelope, beneath a fresh destination header.
    """
    doc = parent._attached_doc  # noqa: SLF001
    target_path = (*parent._path, key)  # noqa: SLF001
    # Snapshot membership before destination-key spelling can run user code.
    src_slots = owned_slots(source)
    source_header = source._header  # noqa: SLF001
    header = (
        _new_section_header(
            target_path,
            leading=_build_section_leading(doc),
            doc=doc,
            owner_aot_entry=parent._owner_aot_entry,  # noqa: SLF001
        )
        if isinstance(source, _container.Document)
        else None
    )
    cloned_slots, cloned_head = _clone_entry_slots(
        src_slots,
        new_entry=None,
        body_owner=parent._owner_aot_entry,  # noqa: SLF001
        src_prefix=source._path,  # noqa: SLF001
        target_prefix=target_path,
        dst_newline=doc._newline,  # noqa: SLF001
        head=source_header,
    )
    if header is not None:
        cloned_slots.insert(0, header)
    else:
        assert cloned_head is not None
        first = cloned_slots[0]
        assert isinstance(first, StructuralHeaderSlot)
        _retarget_separator(first, _build_section_leading(doc))
    section = _install_section_layout(
        parent,
        cloned_slots,
        doc=doc,
        target_path=target_path,
    )
    _maybe_demote_synthetic_empty_header(parent)
    dict.__setitem__(parent, key, section)
    return section


def detach_aot_from_orphan(value: AoT) -> None:
    """Cut a private-orphan AoT loose from the document it lives in.

    The orphan must stop naming entries that move to the destination:
    otherwise a later adopt of the orphan would gather their slots. An
    array emptied by :meth:`AoT.pop` still renders as ``k = []`` and
    has no entry left to carry that slot away, so it goes here too.

    Free factories are untouched. Materialized entries release their
    hosts after bulk scrubbing, retaining source roots and slots until
    their individual adoption.
    """
    if value._layout_root is None:  # noqa: SLF001
        return
    if not value:
        slot = _empty_aot_placeholder_slot(value)
        assert slot is not None, "an attached empty AoT renders as `k = []`"
        slots: list[Slot] = [slot]
        _detach_from_source_doc(value, slots)
    else:
        slots = owned_slots(value)
    _unfile_stale_same_orphan_ancestors(value, slots)
    for entry in value:
        assert entry._host is value  # noqa: SLF001
        entry._host = None  # noqa: SLF001
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
    host = value._host  # noqa: SLF001
    if host is None:
        # Whole-AoT detachment already removed these ancestor bindings.
        return
    old_parent = host._host if isinstance(host, _array.AoT) else host  # noqa: SLF001
    assert isinstance(old_parent, _container.Container)
    assert old_parent._layout_root is value._layout_root  # noqa: SLF001
    assert len(value._path) == len(old_parent._path) + 1  # noqa: SLF001
    key = value._path[-1]  # noqa: SLF001
    if isinstance(host, _array.AoT):
        entry_index = next(i for i, entry in enumerate(host) if entry is value)
        list.__delitem__(host, entry_index)
        if not host:
            # The last entry has gone, so the array leaves the model too
            # — and must stop being a view onto the orphan, or a caller
            # still holding it would add entries to a document that no
            # longer names it, which would then render what it denies.
            dict.__delitem__(old_parent, key)
            host._unbind_from_document()  # noqa: SLF001
    else:
        dict.pop(old_parent, key, None)

    stale_container_ids: set[int] = set()
    node: Container | None = old_parent
    while node is not None and node._layout_root is value._layout_root:  # noqa: SLF001
        stale_container_ids.add(id(node))
        node = node._parent  # noqa: SLF001
    unfiled: set[Slot] = set()
    for slot in target_slots:
        for c in list(slot._containers):  # noqa: SLF001
            if id(c) in stale_container_ids:
                unfile_slot(c, slot)
                unfiled.add(slot)
    _invalidate_body_tail_chain(old_parent, unfiled, departing=True)


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
    new_prefix = (*dest_parent._path, key)  # noqa: SLF001
    slots = _move_private_subtree(
        value,
        dest_parent,
        new_prefix,
        owner=dest_parent._owner_aot_entry,  # noqa: SLF001
    )

    # A forward-declared descendant may physically precede value's header.
    first = slots[0]
    assert isinstance(first, StructuralHeaderSlot)
    _retarget_separator(first, _build_section_leading(doc))
    _install_section_layout(
        dest_parent,
        slots,
        doc=doc,
        target_path=new_prefix,
        existing=value,
    )
    dict.__setitem__(dest_parent, key, value)
    _maybe_demote_synthetic_empty_header(dest_parent)
    return value


def _move_private_subtree(
    value: Container,
    dest_host: Container | AoT,
    new_prefix: tuple[str, ...],
    *,
    owner: AoTEntry | None,
    host_path: tuple[str, ...] | None = None,
) -> list[Slot]:
    """Detach and rebase private layout, leaving publication to the caller."""
    doc = dest_host._attached_doc  # noqa: SLF001
    old_prefix = value._path  # noqa: SLF001
    stale_owner = value._owner_aot_entry  # noqa: SLF001
    slots = owned_slots(value)
    _detach_from_source_doc(value, slots)
    targets = (
        slots
        if value._header is None  # noqa: SLF001
        else [s for s in slots if isinstance(s, StructuralHeaderSlot)]
    )
    _unfile_stale_same_orphan_ancestors(value, targets)
    if host_path is None:
        host_path = new_prefix
    for slot in slots:
        _rebase_slot(
            slot,
            old_prefix,
            new_prefix,
            host_path,
            doc._newline,  # noqa: SLF001
        )
        _transfer_stale_owner(slot, stale_owner, owner)
    _rehome_view_tree(
        value,
        dest_host,
        new_prefix,
        doc,
        stale_owner=stale_owner,
        new_owner=owner,
    )
    return slots


def adopt_private_entry(
    aot: AoT, value: Table, *, preserve_source_separator: bool = False
) -> None:
    """Move a private table into its AoT, retaining layout and all live views.

    The caller retains the source parent and repairs it after adoption.
    """
    doc = aot._attached_doc  # noqa: SLF001
    parent = aot._host  # noqa: SLF001
    assert parent is not None
    path = aot._path  # noqa: SLF001
    owner = AoTEntry()
    original_header = value._header  # noqa: SLF001
    slots = _move_private_subtree(value, aot, path, owner=owner)
    if original_header is None:
        header = _new_section_header(
            path, leading="", doc=doc, entry=owner, owner_aot_entry=owner
        )
        slots.insert(0, header)
    else:
        header = original_header
        header.entry = owner
        owner.bind_header(header)

    _append_entry_run(
        aot, header, slots, preserve_source_separator=preserve_source_separator
    )
    # Rebasing has placed every KV at this entry's path or below it.
    body, blocks = split_subtree_slots(
        (slot for slot in slots if slot is not header), len(path) + 1
    )
    ordered = [header, *body, *blocks]
    del body, blocks  # Release scratch lists before refiling the ordered run.
    if ordered == slots:
        ordered = slots
    else:
        predecessor, successor = slots[0]._prev, slots[-1]._next  # noqa: SLF001
        with _refile_region_refs(doc, predecessor, successor):
            _link_run_between(predecessor, ordered, successor, doc)
    if original_header is None:
        file_own_header(value, header)
        value._body_tail = _recompute_body_tail(value)  # noqa: SLF001
    _extend_header_bindings_to_root(parent, ordered)
    list.append(aot, value)
    _maybe_demote_synthetic_empty_header(parent)


def _rehome_view_tree(
    root: Container,
    dest_host: Container | AoT,
    new_prefix: tuple[str, ...],
    doc: Document,
    *,
    stale_owner: AoTEntry | None,
    new_owner: AoTEntry | None,
) -> None:
    """Re-point ``root``'s subtree at ``doc``, deriving paths from its owners.

    Slot-backed caches remain valid because their slots are rebased in
    parallel. Views owned by ``stale_owner`` transfer to the destination
    entry; nested AoT entries retain their own owners. Array elements
    start relative paths, even when their keys resemble the moved section.
    """
    root._host = dest_host  # noqa: SLF001
    root._path = new_prefix  # noqa: SLF001
    for node in _walk_views((root,)):
        # Narrows for the assignments below; `_View` has no other subclass.
        assert isinstance(node, (_container.Container, _array.AoT, _array.Array))
        node._layout_root = doc  # noqa: SLF001
        if isinstance(node, (_container.Container, _array.AoT)):
            if node is not root:
                host = node._host  # noqa: SLF001
                if isinstance(host, _array.AoT):
                    node._path = host._path  # noqa: SLF001
                elif isinstance(host, _container.Container):
                    node._path = (*host._path, node._path[-1])  # noqa: SLF001
                else:
                    assert isinstance(host, _array.Array), "rehomed view has no owner"
                    node._path = ()  # noqa: SLF001
            if (
                isinstance(node, _container.Container)
                and node._owner_aot_entry is stale_owner  # noqa: SLF001
            ):
                node._owner_aot_entry = new_owner  # noqa: SLF001


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
    new_prefix = (*dest_parent._path, key)  # noqa: SLF001
    host = _nearest_header_host(dest_parent)
    host_path = host._path  # noqa: SLF001

    # `_attach_section` only dispatches here for an orphan that still owns
    # slots; a slotless one is synthesised instead.
    assert value._refs, "implicit orphan has no slots"  # noqa: SLF001
    slots = _move_private_subtree(
        value,
        dest_parent,
        new_prefix,
        owner=dest_parent._owner_aot_entry,  # noqa: SLF001
        host_path=host_path,
    )
    with _refile_slot_refs(slots):
        slots = _splice_implicit_run(dest_parent, host, slots)
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
            record_slot(anc, s)
    dict.__setitem__(dest_parent, key, value)
    return value


def clone_implicit_section(
    dest_parent: Container, key: str, source: Container
) -> Container:
    """Clone an implicit section's physical run, preserving its dotted shape."""
    doc = dest_parent._attached_doc  # noqa: SLF001
    host = _nearest_header_host(dest_parent)
    slots = clone_graft_slots(
        source,
        target_path=(*dest_parent._path, key),  # noqa: SLF001
        host_path=host._path,  # noqa: SLF001
        owner=dest_parent._owner_aot_entry,  # noqa: SLF001
        nl=doc._newline,  # noqa: SLF001
    )
    slots = _splice_implicit_run(dest_parent, host, slots)
    _populate_entry_views(
        entry_table=host,
        cloned_slots=slots,
        target_prefix=host._path,  # noqa: SLF001
        doc=doc,
    )
    # The builder files through its root; only ancestors above it remain.
    if host._parent is not None:  # noqa: SLF001
        _extend_header_bindings_to_root(host._parent, slots)  # noqa: SLF001
    result = dict.__getitem__(dest_parent, key)
    assert isinstance(result, _container.Container)
    return result


def _spells_own_key(slot: Slot, depth: int) -> bool:
    """Whether a subtree slot is a dotted KV hosted above its root."""
    return isinstance(slot, KVSlot) and len(slot.host_path) < depth


def implicit_body_slots(container: Container) -> list[Slot]:
    """Outer-hosted dotted KVs owned by an implicit section, in source order."""
    depth = len(container._path)  # noqa: SLF001
    owner = container._owner_aot_entry  # noqa: SLF001
    return [
        slot
        for slot in container._refs  # noqa: SLF001
        if _spells_own_key(slot, depth) and slot.owner_aot_entry is owner
    ]


def split_subtree_slots(
    slots: Iterable[Slot], depth: int
) -> tuple[list[Slot], list[Slot]]:
    """Separate a subtree's outer-hosted body from its structural blocks."""
    body: list[Slot] = []
    blocks: list[Slot] = []
    for slot in slots:
        (body if _spells_own_key(slot, depth) else blocks).append(slot)
    return body, blocks


def _splice_implicit_run(
    dest_parent: Container, host: Container, slots: list[Slot]
) -> list[Slot]:
    """Place an implicit subtree's body and blocks in their respective regions."""
    doc = host._attached_doc  # noqa: SLF001
    body, blocks = split_subtree_slots(slots, len(dest_parent._path) + 1)  # noqa: SLF001
    if body:
        anchor = host._body_tail  # noqa: SLF001
        if anchor is None and doc._head is not None:  # noqa: SLF001
            old_head = doc._head  # noqa: SLF001
            _retarget_separator(body[0], "")
            insert_before_head(body[0], doc)
            for prev, slot in itertools.pairwise(body):
                insert_after(prev, slot, doc)
            _ensure_leading_blank_line(old_head, doc)
            _terminate_unless_tail(body[-1], doc)
        else:
            _splice_block_after(body, anchor, doc)
    if blocks:
        if slots[0] is blocks[0]:
            _retarget_separator(blocks[0], _build_section_leading(doc))
        anchor = (
            _safe_header_anchor(body[-1]) if body else _child_header_anchor(dest_parent)
        )
        _splice_block_after(blocks, anchor, doc)
    return body + blocks


def _rebase_slot(
    s: Slot,
    old_prefix: tuple[str, ...],
    new_prefix: tuple[str, ...],
    host_path: tuple[str, ...],
    nl: str,
) -> None:
    """Rebase a slot and its trivia to the destination subtree.

    In-subtree KVs replace their host prefix; headers replace their spelled prefix.
    Dotted KVs hosted above the source subtree instead move to ``host_path``,
    keeping the spelling of key components within the subtree.
    """
    retarget_slot_newlines(s, nl)
    if isinstance(s, KVSlot):
        if s.host_path[: len(old_prefix)] == old_prefix:
            if old_prefix != new_prefix:
                s.host_path = new_prefix + s.host_path[len(old_prefix) :]
        else:
            within = (*s.host_path, *s.key_path)[len(old_prefix) :]
            new_key = (*new_prefix, *within)[len(host_path) :]
            head_n = len(new_key) - len(within)
            s.host_path = host_path
            s.key_parts, s.key_seps, s.key_path = respell_key_prefix(
                s.key_parts,
                s.key_seps,
                s.key_path,
                len(s.key_path) - len(within),
                new_key[:head_n],
            )
        return
    assert isinstance(s, StructuralHeaderSlot)
    s.key_parts, s.key_seps, s.key_path = respell_key_prefix(
        s.key_parts, s.key_seps, s.key_path, len(old_prefix), new_prefix
    )


def clone_aot(
    parent: Container,
    key: str,
    src_aot: AoT,
) -> AoT:
    """Install ``src_aot`` (an attached AoT) under ``parent[key]``.

    Each entry is deep-cloned with path-rebasing so any nested
    sub-sections stay logically inside the new key.
    """
    new_aot = _array.AoT()
    _bind_aot(parent, key, new_aot)
    dict.__setitem__(parent, key, new_aot)
    for src_entry_table in list(src_aot):
        add_aot_entry(
            new_aot,
            src_entry_table,
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
    dst_newline: str | None,
    head: StructuralHeaderSlot | None = None,
    host_path: tuple[str, ...] | None = None,
) -> tuple[list[Slot], StructuralHeaderSlot | None]:
    r"""Deep-clone an entry's slot list with path/owner rebasing.

    ``head``, if given, identifies ``src_slots``' own boundary header by
    identity, not position, since it need not be ``src_slots[0]`` (see
    :func:`owned_slots`). Its clone is returned as the second
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

    A KV hosted *above* ``src_prefix`` — a header-less section's own
    dotted key — cannot be rebased by path, and is re-hosted at
    ``host_path`` instead, which defaults to ``target_prefix``. A clone
    that lands under a header of its own wants that default; one that
    stays header-less, spelled by dotted keys, wants the enclosing
    section that will host them.

    ``dst_newline=None`` copies a whole document without retargeting its
    paths or its potentially mixed line endings.
    """
    if host_path is None:
        host_path = target_prefix
    nested_entry_map: dict[AoTEntry, AoTEntry] = {}
    cloned: list[Slot] = []
    cloned_head: StructuralHeaderSlot | None = None
    memo: dict[int, object] = {}
    for s in src_slots:
        c: Slot = copy.deepcopy(s, memo)
        if dst_newline is not None:
            _rebase_slot(c, src_prefix, target_prefix, host_path, dst_newline)
        if isinstance(c, StructuralHeaderSlot):
            assert isinstance(s, StructuralHeaderSlot)
            if s is head:
                # head's kind always comes from new_entry, not from
                # source-entry lookup (which is None for a plain table).
                c.entry = new_entry
                cloned_head = c
            elif s.entry is not None:
                c.entry = AoTEntry()
            if c.entry is not None:
                if s.entry is not None:
                    assert s.entry not in nested_entry_map
                    nested_entry_map[s.entry] = c.entry
                c.entry.bind_header(c)
        # An AoT header introduces its own owner before any of its body slots.
        src_owner = s.owner_aot_entry
        c.owner_aot_entry = (
            nested_entry_map.get(src_owner, body_owner)
            if src_owner is not None
            else body_owner
        )
        cloned.append(c)

    return cloned, cloned_head


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


def materialise_section(source: Table, *, preserve_header: bool) -> None:
    """Give a validated section factory private layout until its first attachment."""
    holder = _container.Document()
    holder._is_private = True  # noqa: SLF001
    attach_section_at(holder, ("",), source, preserve_header=preserve_header)


def attach_section_at(
    parent: Container,
    sub_path: tuple[str, ...] | list[str],
    source: Table,
    *,
    preserve_header: bool = False,
) -> Table:
    """Synthesise ``[parent_path.sub_path]`` (multi-component) at end-of-doc.

    Intermediate components in ``sub_path[:-1]`` become implicit tables;
    the deepest component gets the explicit header. ``source`` is a validated,
    unattached `Table`, rehomed in place.
    """
    from tomlrt._container import _is_inline_input  # noqa: PLC0415

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
    if preserve_header:
        header.synthetic = False

    # Build implicit chain: intermediates become header-less Tables
    # living in dict storage; the deepest is where the new explicit
    # header is filed.
    deepest_parent = ensure_implicit_chain(parent, sub[:-1])

    section = source
    pending: list[tuple[str, TomlInput]] = list(_mapping_items(source))
    dict.clear(section)

    section._wire(  # noqa: SLF001
        layout_root=doc,
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
    scalars: list[tuple[str, TomlInput]] = []
    structurals: list[tuple[str, TomlInput]] = []
    for k, v in pending:
        if _is_inline_input(v):
            scalars.append((k, v))
        else:
            structurals.append((k, v))
    for k, v in scalars:
        append_synth_kv(section, k, v)
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


def _unfile_ordered(refs: list[Slot], slot: Slot) -> None:
    """Remove one slot, trying the ends before bisecting its order key."""
    if refs[-1] is slot:
        refs.pop()
        return
    i = 0 if refs[0] is slot else _ordered_index(refs, slot._order)  # noqa: SLF001
    assert refs[i] is slot, "slot must be filed in doc order"
    refs.pop(i)


def _unfile_ordered_many(refs: list[Slot], removed: Sequence[Slot]) -> None:
    """Drop ordered refs without repeatedly shifting the projection's survivors."""
    if len(refs) == len(removed):
        refs.clear()
        return
    positions: list[int] = []
    i = 0
    for slot in removed:
        if refs[i] is not slot:
            i = _ordered_index(refs, slot._order)  # noqa: SLF001
        assert refs[i] is slot, "slot must be filed in doc order"
        positions.append(i)
        i += 1
    delete_runs(refs, index_runs(positions))


def _unfile_container(slot: Slot, c: Container) -> None:
    """Remove an existing back-pointer by identity, never by dict equality."""
    containers = slot._containers  # noqa: SLF001
    i = 0
    while containers[i] is not c:
        i += 1
    del containers[i]


def unfile_slot(c: Container, slot: Slot) -> None:
    """Remove ``slot`` from ``c``'s projections and unregister its back-pointer.

    Also clears ``c._header`` if this is the container's own header.
    """
    assert not c._inline, "inline containers do not file refs"  # noqa: SLF001
    _unfile_ordered(c._refs, slot)  # noqa: SLF001
    local_key = slot_local_key(slot, c)
    if local_key is None:
        assert c._header is slot  # noqa: SLF001
        c._header = None  # noqa: SLF001
    else:
        bucket = c._index[local_key]  # noqa: SLF001
        _unfile_ordered(bucket, slot)
        if not bucket:
            del c._index[local_key]  # noqa: SLF001
    _unfile_container(slot, c)


def _scrub_owned_slots_via_backptrs(
    owned: Sequence[Slot],
    *,
    skip_container_ids: frozenset[int] = frozenset(),
) -> None:
    """Remove every live ref to each slot in ``owned`` via slot back-pointers.

    Walks ``slot._containers`` directly (length ≤ path depth, bounded
    independent of doc size) instead of scanning ancestor containers'
    ``_index``/``_refs`` lists. Refs are grouped by container so each
    affected projection can be spliced in one operation.

    ``skip_container_ids`` names containers whose internal refs to
    owned slots should be left in place — the typical caller is
    `delete_key`, which transplants the deleted subtree to a fresh
    orphan doc and needs the subtree containers' internal
    structure intact. AoT removal likewise retains the departing
    entries' internal refs. Both multi-slot callers visit binding refs
    in document order. Each affected ancestor therefore loses an ordered
    subset of one child binding, never its own header or several different keys.
    """
    # One slot has at most one ref per container: there is nothing to batch.
    if len(owned) == 1:
        slot = owned[0]
        for c in list(slot._containers):  # noqa: SLF001
            if id(c) not in skip_container_ids:
                unfile_slot(c, slot)
        return
    by_container: dict[int, tuple[Container, list[Slot]]] = {}
    for s in owned:
        for c in s._containers:  # noqa: SLF001
            container_id = id(c)
            if container_id in skip_container_ids:
                continue
            by_container.setdefault(container_id, (c, []))[1].append(s)
    for c, removed in by_container.values():
        key = slot_local_key(removed[0], c)
        assert key is not None, "bulk removal retains subtree containers' own refs"
        bucket = c._index[key]  # noqa: SLF001
        _unfile_ordered_many(c._refs, removed)  # noqa: SLF001
        _unfile_ordered_many(bucket, removed)
        if not bucket:
            del c._index[key]  # noqa: SLF001
        for slot in removed:
            _unfile_container(slot, c)


def _norm_aot_index(aot: AoT, index: int) -> int:
    """Normalise ``index`` to non-negative; raise IndexError if out of range."""
    n = len(aot)
    if not -n <= index < n:
        msg = f"AoT index {index} out of range (len {n})"
        raise IndexError(msg)
    return index + n if index < 0 else index


def remove_aot_entry(aot: AoT, index: int) -> Table:
    """Remove ``aot[index]``, unlink its slots, and return it detached.

    Returns the popped entry itself, with its layout preserved in a
    private document, mirroring ``delete_key``. Held references remain
    usable without affecting the source document.
    """
    return remove_aot_entries(aot, [_norm_aot_index(aot, index)])[0]


def remove_aot_entries(aot: AoT, indices: Iterable[int]) -> list[Table]:
    """Remove ``aot[i]`` for each ``i`` in ``indices`` in one batch.

    The indices must already be **non-empty, non-negative, in-range,
    distinct, and ascending**; callers are responsible for normalising.
    Returns the orphaned entry ``Table``s in the same order as
    ``indices``.

    Scrubbing the union together removes each projection in bulk,
    rather than shifting surviving refs after every entry.
    """
    idx_list = list(indices)
    assert idx_list
    doc = aot._attached_doc  # noqa: SLF001
    parent = aot._host  # noqa: SLF001
    assert parent is not None

    # Collect each entry's whole subtree in doc-stream order and capture
    # the entry table itself for return. Distinct entries own
    # disjoint subtrees, so concatenating is already a union.
    popped_entries: list[Table] = []
    union_owned_ordered: list[Slot] = []  # in doc-stream order

    for i in idx_list:
        entry_table = aot[i]
        union_owned_ordered.extend(owned_slots(entry_table))
        popped_entries.append(entry_table)

    # The entries themselves keep their internal caches: they are moving
    # to a document of their own, not being taken apart.
    views = list(_walk_views(popped_entries))

    _detach_departing_slots(parent, union_owned_ordered, views)

    delete_runs(aot, index_runs(idx_list))

    # The popped entries move to a private document, keeping their own
    # lines, exactly as a deleted section does. An array-of-tables is
    # what their ``[[path]]`` headers spell, so that is what holds them
    # there, at the path they came from.
    holder = _array.AoT()
    holder._path = aot._path  # noqa: SLF001
    list.extend(holder, popped_entries)
    views.append(holder)
    _transplant_to_orphan(
        holder,
        union_owned_ordered,
        doc._newline,  # noqa: SLF001
        views,
    )

    last_key = aot._path[-1]  # noqa: SLF001
    if len(aot) == 0 and not parent._index.get(last_key):  # noqa: SLF001
        parent._index.pop(last_key, None)  # noqa: SLF001
        # An empty AoT still lives in dict storage; a ``key = []``
        # placeholder gives it a physical presence so the document keeps
        # the same semantic shape as the dict view.
        _materialise_empty_aot(aot)

    return popped_entries


def _view_route(view: Container | AoT) -> list[tuple[str, int | None]]:
    """The ``(key, entry ordinal)`` steps from a document down to ``view``.

    A path alone cannot say which entry of an array-of-tables a view
    sits in, so each step records the ordinal too, and only for the
    steps that pass through one.
    """
    route: list[tuple[str, int | None]] = []
    cur: Container | AoT = view
    while cur._path:  # noqa: SLF001
        host = cur._host  # noqa: SLF001
        assert host is not None, "an attached view below the root has a host"
        key = cur._path[-1]  # noqa: SLF001
        ordinal = None
        if isinstance(host, _array.AoT):
            ordinal = next(i for i, entry in enumerate(host) if entry is cur)
            host = host._host  # noqa: SLF001
        assert isinstance(host, _container.Container)
        route.append((key, ordinal))
        cur = host
    route.reverse()
    return route


def stable_snapshot(view: Container | AoT) -> Container | AoT:
    """``view`` as it is now, safe to read across a write to its document.

    For a source an install would otherwise damage or grow while
    reading it.
    """
    return _snapshot_in_copy(view, {})


def _follow_view_route(
    root: Container | AoT, route: Sequence[tuple[str, int | None]]
) -> Container | AoT:
    """Resolve a structural route, including ordinals through nested AoTs."""
    cur = root
    for key, ordinal in route:
        assert isinstance(cur, _container.Container)
        cur = dict.__getitem__(cur, key)
        if ordinal is not None:
            assert isinstance(cur, _array.AoT)
            cur = cur[ordinal]
    return cur


def _snapshot_in_copy(
    view: Container | AoT, snapshots: dict[int, Document]
) -> Container | AoT:
    """``view`` as it is now, in a copy of the document it lives in.

    One copy serves every source taken from the same document, and it
    is a byte-exact one, so installing from it preserves the source's
    own trivia exactly as installing from the original would.
    """
    root = view._layout_root  # noqa: SLF001
    assert root is not None, "only an attached view has a document to copy"
    snapshot = snapshots.get(id(root))
    if snapshot is None:
        snapshot = copy.copy(root)
        snapshots[id(root)] = snapshot
    return _follow_view_route(snapshot, _view_route(view))


def _capture_items(
    items: Iterable[tuple[str, TomlInput]],
    sites: Sequence[Container | AoT],
    snapshots: dict[int, Document],
) -> dict[str, TomlInput]:
    """Snapshot mapping storage, recursing only into compound values."""
    return {
        key: value
        if isinstance(value, SCALAR_TYPES)
        else _capture_input(value, sites, snapshots)
        for key, value in items
    }


def _capture_input(
    value: TomlInput,
    sites: Sequence[Container | AoT],
    snapshots: dict[int, Document],
) -> TomlInput:
    """Capture sources containing a write site, recursively through wrappers.

    Detached factories retain their identity; plain mappings and lists
    are rebuilt. Other values keep ordinary installation semantics,
    including adoption of descendants orphaned by the write. Callers
    handle scalar leaves without entering this recursive path.
    """
    if isinstance(value, (_container.Container, _array.AoT)):
        # A detached view's storage is all it has, so that is where its
        # own sources are captured; an attached one is copied from its
        # slots at install time, which reads nothing this write touches.
        if value._layout_root is None:  # noqa: SLF001
            _capture_into_factory(value, sites, snapshots)
        elif any(hosts_site(value, site) for site in sites):
            return _snapshot_in_copy(value, snapshots)
        return value
    if is_inline_value(value):
        return value
    if isinstance(value, Mapping):
        return _capture_items(_mapping_items(value), sites, snapshots)
    assert isinstance(value, list)
    return [
        v if isinstance(v, SCALAR_TYPES) else _capture_input(v, sites, snapshots)
        for v in value
    ]


def _capture_into_factory(
    factory: Container | AoT,
    sites: Sequence[Container | AoT],
    snapshots: dict[int, Document],
) -> None:
    """Capture in the factory's storage so its first occurrence still attaches."""
    if isinstance(factory, _array.AoT):
        for entry in factory:
            _capture_into_factory(entry, sites, snapshots)
        return
    dict.update(factory, _capture_items(_mapping_items(factory), sites, snapshots))


def hosts_site(view: Container | AoT, site: Container | AoT) -> bool:
    """True iff writing ``site`` writes ``view`` too.

    That is ``site`` itself, one of its ancestors, or an
    array-of-tables reached along the way — the views a source has to
    be read out of before the write, because the write is inside them.
    A descendant is not one of them: it leaves with its body intact in
    a private orphan, which an install adopts or copies from.

    The question is asked of the views themselves. A path cannot answer
    it: an array-of-tables gives all its entries one path, so a sibling
    entry prefixes the site without ever containing it.
    """
    cur: Container | AoT | Array | None = site
    while cur is not None:
        if cur is view:
            return True
        cur = cur._host  # noqa: SLF001
    return False


class _PreparedEntry:
    """An existing or unpublished entry and its destination-ready body."""

    __slots__ = ("body", "header", "table")

    def __init__(
        self,
        table: Table,
        header: StructuralHeaderSlot,
        body: list[Slot] | dict[str, TomlInput],
    ) -> None:
        self.table = table
        self.header = header
        self.body = body


def clone_aot_entry_layout(
    source: Container,
    *,
    path: tuple[str, ...],
    owner: AoTEntry,
    nl: str,
    keep_header: bool = True,
) -> tuple[StructuralHeaderSlot | None, list[Slot]]:
    """Clone a structural source as an AoT header and body ready for publication.

    An existing destination entry keeps its own header. A new one may
    take the source's header, normalised to ``[[path]]``. Its own body
    must precede forward-declared descendants because an AoT entry
    cannot be reopened with a later plain section header.
    """
    slots = owned_slots(source)
    head = source._header  # noqa: SLF001
    if not keep_header:
        slots = [slot for slot in slots if slot is not head]
        head = None
    cloned, cloned_head = _clone_entry_slots(
        slots,
        new_entry=owner,
        body_owner=owner,
        src_prefix=source._path,  # noqa: SLF001
        target_prefix=path,
        dst_newline=nl,
        head=head,
    )
    # Rebasing has placed every KV at this entry's path or below it.
    body, blocks = split_subtree_slots(
        (slot for slot in cloned if slot is not cloned_head), len(path) + 1
    )
    return cloned_head, body + blocks


def _prepare_entry(
    aot: AoT,
    table: Table,
    body: Mapping[str, TomlInput],
    sites: Sequence[Container | AoT],
    snapshots: dict[int, Document],
) -> _PreparedEntry:
    """Capture a body without clearing, wiring or publishing its destination.

    A whole structural source contributes slots, independently of any
    later adoption of its children. Nested values in mapping bodies
    retain ordinary attachment semantics.
    """
    doc = aot._attached_doc  # noqa: SLF001
    path = aot._path  # noqa: SLF001
    owner: AoTEntry | None
    if table._layout_root is None:  # noqa: SLF001
        owner = AoTEntry()
        header = None
    else:
        owner = table._owner_aot_entry  # noqa: SLF001
        assert owner is not None
        header = owner.header
    payload: list[Slot] | dict[str, TomlInput]
    if (
        isinstance(body, _container.Container)
        and body._layout_root is not None  # noqa: SLF001
        and not body._inline  # noqa: SLF001
    ):
        cloned_head, payload = clone_aot_entry_layout(
            body,
            path=path,
            owner=owner,
            nl=doc._newline,  # noqa: SLF001
            keep_header=header is None,
        )
        if cloned_head is not None:
            header = cloned_head
    else:
        payload = _capture_items(_mapping_items(body), sites, snapshots)
    if header is None:
        header = _new_section_header(
            path, leading="", doc=doc, entry=owner, owner_aot_entry=owner
        )
    return _PreparedEntry(table, header, payload)


def _append_entry_run(
    aot: AoT,
    header: StructuralHeaderSlot,
    run: list[Slot],
    *,
    preserve_source_separator: bool,
) -> None:
    """Splice a new entry's slots at the AoT's tail, separated to fit there.

    A bulk clone may keep the separators its source had, but the first
    entry always takes the destination's own section spacing: it is the
    one being positioned, not spaced from a predecessor.
    """
    doc = aot._attached_doc  # noqa: SLF001
    ordinal = len(aot)
    _consume_first_entry_placeholder(aot, ordinal)
    if ordinal == 0:
        _retarget_separator(header, _build_section_leading(doc))
    elif not preserve_source_separator:
        _retarget_separator(header, _aot_separator(aot, doc))
    _splice_block_after(run, _aot_append_anchor(aot), doc)


def _install_entry(
    aot: AoT, prepared: _PreparedEntry, *, preserve_source_separator: bool = False
) -> Table:
    """Publish a fresh header or retain an existing one, then consume its body."""
    table, header = prepared.table, prepared.header
    owner = header.entry
    assert owner is not None
    parent = aot._host  # noqa: SLF001
    assert parent is not None
    doc = aot._attached_doc  # noqa: SLF001
    path = aot._path  # noqa: SLF001
    if table._layout_root is None:  # noqa: SLF001
        dict.clear(table)
        table._wire(  # noqa: SLF001
            layout_root=doc, parent=aot, path=path, owner=owner
        )
        _append_entry_run(
            aot, header, [header], preserve_source_separator=preserve_source_separator
        )
        file_own_header(table, header)
        _file_header_binding_chain(parent, header)
        list.append(aot, table)
        _maybe_demote_synthetic_empty_header(parent)
    else:
        table.clear()
    if isinstance(prepared.body, dict):
        for key, value in prepared.body.items():
            table._setitem_validated(key, value)  # noqa: SLF001
    else:
        _splice_block_after(prepared.body, header, doc)
        _populate_entry_views(
            entry_table=table,
            cloned_slots=prepared.body,
            target_prefix=path,
            doc=doc,
        )
        _extend_header_bindings_to_root(parent, prepared.body)
    return table


def assign_aot_entries(
    aot: AoT, index: int | slice, bodies: Sequence[Mapping[str, TomlInput]]
) -> None:
    """Capture all sources, then replace in place or resize the array."""
    if isinstance(index, slice):
        start, stop, step = index.indices(len(aot))
        indices = range(start, stop, step)
    else:
        start = _norm_aot_index(aot, index)
        indices = range(start, start + 1)
    resizing = len(indices) != len(bodies)
    tables = (
        [_container.Table() for _ in bodies] if resizing else [aot[i] for i in indices]
    )
    targets = [
        (table, body)
        for table, body in zip(tables, bodies, strict=True)
        if table is not body
    ]
    sites = [aot] if resizing else [table for table, _ in targets]
    snapshots: dict[int, Document] = {}
    prepared = [
        _prepare_entry(aot, table, body, sites, snapshots) for table, body in targets
    ]
    if resizing and indices:
        remove_aot_entries(aot, indices)
    for entry in prepared:
        _install_entry(aot, entry)
    # New entries were appended; moving them is unnecessary for a tail splice.
    if resizing and prepared and start != len(aot) - len(prepared):
        order = list(aot)[: -len(prepared)]
        order[start:start] = [entry.table for entry in prepared]
        renormalise_aot_order(aot, order)


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
    # header + KV slots: a nested ``[[a.x]]`` has its own AoTEntry and
    # must travel with its parent entry, so that reordering doesn't
    # strand it and re-parent it onto whichever entry lands at its old
    # position.
    physical_blocks: list[list[Slot]] = []
    phys_idx_by_id: dict[int, int] = {}
    for entry_table in aot:
        e = entry_table._owner_aot_entry  # noqa: SLF001
        assert e is not None
        phys_idx_by_id[id(entry_table)] = len(physical_blocks)
        physical_blocks.append(owned_slots(entry_table))

    region_predecessor = physical_blocks[0][0]._prev  # noqa: SLF001
    region_successor = physical_blocks[-1][-1]._next  # noqa: SLF001

    new_order_indices = [
        phys_idx_by_id[id(t)] for t in new_logical_order if id(t) in phys_idx_by_id
    ]
    output_blocks = [physical_blocks[phys_idx] for phys_idx in new_order_indices]
    movable_slots = [slot for block in physical_blocks for slot in block]
    placements = _peer_placements(physical_blocks, output_blocks)
    with _refile_region_refs(doc, region_predecessor, region_successor):
        _splice_blocks_in_order(
            doc, movable_slots, placements, anchor_prev=region_predecessor
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


class _ReorderUnit:
    """One independently sortable slot block and its leading-trivia state."""

    __slots__ = (
        "key_rank",
        "mixed",
        "prefix",
        "remainder",
        "slots",
        "structural",
    )

    def __init__(
        self,
        slots: list[Slot],
        key_rank: int,
        structural: bool,  # noqa: FBT001
        mixed: bool,  # noqa: FBT001
        prefix: str,
        remainder: str,
    ) -> None:
        self.slots = slots
        self.key_rank = key_rank
        self.structural = structural
        self.mixed = mixed
        self.prefix = prefix
        self.remainder = remainder


def _peer_placements(
    physical_blocks: list[list[Slot]], output_blocks: list[list[Slot]]
) -> list[tuple[list[Slot], str]]:
    """Pair peer blocks with positional prefixes and attached remainders."""
    prefixes: list[str] = []
    remainder_by_head: dict[Slot, str] = {}
    for block in physical_blocks:
        prefix, remainder = _split_leading_trivia(block[0])
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
    *,
    anchor_prev: Slot | None,
) -> None:
    """Reorder movable layout blocks within the doc-stream.

    ``movable_slots`` is in original physical order. ``placements`` is
    the block grouping, order, and head leading to reinsert; callers may
    split an original logical block when its binding order must change.

    ``anchor_prev`` stays linked and names a position in the output's
    containing scope. Terminates the anchor and former final movable
    slot if they gain successors. Trivia policy (positional
    vs slot-attached) is the caller's responsibility — see
    ``renormalise_aot_order`` and ``reorder_container`` for the two
    existing flavours.
    """
    assert movable_slots, "both callers permute a non-empty set of blocks"

    former_region_tail = movable_slots[-1]
    for slot in movable_slots:
        unlink_slot(slot, doc, strip_new_head_leading=False)

    ordered: list[Slot] = []
    for block, leading in placements:
        block[0].leading = leading
        ordered.extend(block)
    if anchor_prev is not None:
        ensure_terminator(anchor_prev, doc._newline)  # noqa: SLF001
    _relink_run_after(anchor_prev, ordered, doc)

    _terminate_unless_tail(former_region_tail, doc)


def _slot_binding_root(slot: Slot) -> tuple[str, ...]:
    """Return the direct binding path represented by ``slot``."""
    if isinstance(slot, StructuralHeaderSlot):
        return slot.key_path
    assert isinstance(slot, KVSlot)
    return (*slot.host_path, slot.key_path[0])


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
    primary = _binding_primary_slot(parent, key)

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


def _binding_primary_slot(parent: Container, key: str) -> Slot:
    """Return a binding's direct slot, or its first descendant when implicit."""
    refs = parent._index.get(key)  # noqa: SLF001
    assert refs, "bound key must have refs"
    path = (*parent._path, key)  # noqa: SLF001
    return next(
        (slot for slot in refs if _slot_binding_root(slot) == path),
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

    :func:`_recorded_install_span` supplies nonempty slots in contiguous
    doc-stream order; its caller leaves scattered installations in place.
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


def _owned_child_key(slot: Slot, depth: int) -> str | None:
    """Classify a slot already known to belong to the container at ``depth``.

    ``owned_slots`` establishes membership, including AoT-entry ownership.
    The container's own header names no child; KVs use host/key geometry.
    """
    if isinstance(slot, KVSlot):
        host = slot.host_path
        host_depth = len(host)
        if host_depth > depth:
            return host[depth]
        return slot.key_path[depth - host_depth]
    assert isinstance(slot, StructuralHeaderSlot)
    path = slot.key_path
    return path[depth] if len(path) > depth else None


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
    region so direct KVs stay bound to ``c``. An implicit table's dotted
    body stays after its containing header, even when a forward-declared
    child starts the subtree before that header. For an AoT entry, only
    slots within ``c``'s own subtree participate (see
    :func:`_owned_slots`): nested descendants move with their key, but
    same-path sibling entries are excluded.

    Only mutates the CST; dict storage is the caller's responsibility.
    """
    doc = c._layout_root  # noqa: SLF001
    assert doc is not None

    c_plen = len(c._path)  # noqa: SLF001

    # c's explicit header is the region marker, not a sortable peer: it
    # travels at the splice head so direct KVs keep their binding.
    header_slot: StructuralHeaderSlot | None = None
    header = c._header  # noqa: SLF001
    # Keep headers that demotion would preserve, so their body KVs cannot
    # move ahead of all headers and rebind to the document root.
    if header is not None and (
        header.entry is not None
        or not header.synthetic
        or isinstance(header._next, KVSlot)  # noqa: SLF001
    ):
        header_slot = header

    ordered_slots = owned_slots(c)

    key_blocks: dict[str, list[Slot]] = {k: [] for k in new_key_order}
    child_keys_in_phys_order: list[str] = []
    movable_slots: list[Slot] = []

    for cur in ordered_slots:
        is_header = cur is header_slot
        bind_key = None if is_header else _owned_child_key(cur, c_plen)
        if is_header:
            movable_slots.append(cur)
        elif bind_key is not None and bind_key in key_blocks:
            if not key_blocks[bind_key]:
                child_keys_in_phys_order.append(bind_key)
            key_blocks[bind_key].append(cur)
            movable_slots.append(cur)

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
        head_structural, head_remainder = _split_leading_trivia(earliest_owned)
        earliest_owned.leading = head_remainder
        with _refile_region_refs(doc, region_predecessor, region_successor):
            for f in front_foreign:
                unlink_slot(f, doc, strip_new_head_leading=False)
            _relink_run_after(region_predecessor, front_foreign, doc)
        front_foreign[0].leading = head_structural + front_foreign[0].leading

    key_rank = {key: rank for rank, key in enumerate(new_key_order)}
    units: list[_ReorderUnit] = []
    for key in child_keys_in_phys_order:
        leaves, structural = split_subtree_slots(key_blocks[key], c_plen + 1)
        mixed = bool(leaves and structural)
        for slots, is_structural in ((leaves, False), (structural, True)):
            if not slots:
                continue
            prefix, remainder = _split_leading_trivia(slots[0])
            units.append(
                _ReorderUnit(
                    slots,
                    key_rank[key],
                    is_structural,
                    mixed,
                    prefix,
                    remainder,
                )
            )

    header_prefix = ""
    header_remainder = ""
    if header_slot is not None:
        header_prefix, header_remainder = _split_leading_trivia(header_slot)
        if header_slot is not earliest_owned:
            first_unit = min(units, key=lambda unit: unit.slots[0]._order)  # noqa: SLF001
            header_prefix, first_unit.prefix = first_unit.prefix, header_prefix

    prefixes_by_kind: dict[tuple[bool, bool], list[str]] = {}
    for unit in sorted(units, key=lambda item: item.slots[0]._order):  # noqa: SLF001
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

    original_anchor = earliest_owned._prev  # noqa: SLF001
    anchor_prev = original_anchor
    min_depth = 0
    first_slot = placements[0][0][0]
    if isinstance(first_slot, KVSlot):
        host = _nearest_header_host(c)
        host_header = host._header  # noqa: SLF001
        if (
            host_header is not None and host_header._order > earliest_owned._order  # noqa: SLF001
        ):
            # A forward-declared child precedes the header hosting c's
            # dotted body. Keep that header and its other body keys ahead
            # of the sorted run rather than moving leaves out of scope.
            min_depth = len(host._path)  # noqa: SLF001
            anchor_prev = host._body_tail  # noqa: SLF001
            assert anchor_prev is not None
            while anchor_prev in movable_ids:
                anchor_prev = anchor_prev._prev  # noqa: SLF001
                assert anchor_prev is not None
            if anchor_prev._order > latest_owned._order:  # noqa: SLF001
                region_successor = anchor_prev._next  # noqa: SLF001

    with _refile_region_refs(doc, region_predecessor, region_successor):
        _splice_blocks_in_order(doc, movable_slots, placements, anchor_prev=anchor_prev)
    moved_ids = (
        movable_ids | set(front_foreign) if anchor_prev is original_anchor else None
    )
    _invalidate_body_tail_chain(c, moved_ids, min_depth=min_depth)


__all__ = [
    "add_aot_entry",
    "append_direct_kv",
    "append_synth_kv",
    "assign_aot_entries",
    "attach_empty_aot",
    "attach_section_at",
    "delete_key",
    "hosts_site",
    "remove_aot_entry",
    "renormalise_aot_order",
    "reorder_container",
    "reposition_install",
    "stable_snapshot",
]
