"""Physical slot layer.

A document is an intrusive doubly linked list of physical slots:
``KVSlot`` for ``key = value`` lines and ``StructuralHeaderSlot`` for
``[a.b]`` / ``[[a.b]]`` headers. Containers index slots directly;
each slot keeps back-pointers to the containers that index it.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tomlrt._container import Container
    from tomlrt._values import Value

import sys

if sys.version_info >= (3, 12):
    from typing import override
else:  # pragma: no cover -- backport for Python < 3.12
    from typing_extensions import override

from tomlrt._trivia import retarget_newlines
from tomlrt._values import ScalarValue, render_dotted, retarget_value_newlines

# ---------------------------------------------------------------------------
# AoT entry token (physical ownership marker)
# ---------------------------------------------------------------------------


class AoTEntry:
    """Identifies one entry of an array-of-tables.

    Carried by every physical slot in that entry. The linked slot stream
    owns membership and order; this marker retains only the entry's own
    ``[[a]]`` header.
    """

    __slots__ = ("_header",)

    def __init__(self) -> None:
        self._header: StructuralHeaderSlot | None = None

    def bind_header(self, header: StructuralHeaderSlot) -> None:
        """Record the unique ``[[a]]`` header that introduces this entry."""
        assert self._header is None, "AoT entry already has a header"
        self._header = header

    @property
    def header(self) -> StructuralHeaderSlot:
        """The entry's ``[[a]]`` header."""
        header = self._header
        assert header is not None, "AoT entry header has not been bound"
        return header


# ---------------------------------------------------------------------------
# Slot base + kinds
# ---------------------------------------------------------------------------


class Slot:
    """Base for physical slots, subclassed by `KVSlot` and `StructuralHeaderSlot`.

    Every slot spells one physical line as ``leading`` + body + ``eol``;
    those trivia fields, plus ``owner_aot_entry``, live here so every kind
    exposes them uniformly and only the body differs.

    Deliberately not an `abc.ABC`: an `ABCMeta` metaclass would slow every
    `isinstance(slot, KVSlot)` / `isinstance(slot, StructuralHeaderSlot)`
    check on the parse/mutation hot path. `render` raises rather than
    being left unimplemented so a missing override fails loudly.

    Constructor fields are positional and required — a slot is built once
    per line and keyword binding roughly doubles that cost. ``_prev`` /
    ``_next`` / ``_order`` / ``_containers`` are runtime wiring, initialized
    independently for every new slot. Concrete constructors initialize
    inherited fields directly to avoid a base-initializer call per line.
    """

    __slots__ = (
        "_containers",
        "_next",
        "_order",
        "_prev",
        "eol",
        "leading",
        "owner_aot_entry",
    )

    leading: str
    """Trivia before the slot's own text: blank lines, comment lines, indent."""

    owner_aot_entry: AoTEntry | None
    """The AoT entry that physically contains this slot, if any."""

    eol: str
    """Trivia after the slot's own text: gap, comment, line terminator.

    Verbatim source text, like ``leading``. Empty of a terminator only
    for the last line of a file that ends without one.
    """

    _prev: Slot | None
    _next: Slot | None
    _order: int
    """Doc-stream order key: strictly increasing along ``_next``.

    Lets "which slot comes first?" — and so where a ref belongs in a
    doc-ordered ``Container._refs`` — be a comparison rather than a walk.
    Maintained by `stitch_run`; meaningless for an unlinked slot, which
    is stamped afresh when spliced back in.
    """
    _containers: list[Container]
    """Containers that index this slot, compared by identity.

    Bounded length (≤ path depth + 1). AoT removal uses this to scrub
    memberships in O(depth) per slot instead of O(siblings) per container.
    """

    def __deepcopy__(self, memo: dict[int, object]) -> Slot:
        """Copy the slot's own state, sharing everything immutable.

        Clones start with empty refs and no doc-stream links; callers
        file new refs and splice them in themselves. Following
        ``_prev``/``_next`` would drag the whole source document in.

        Values follow their own sharing policy: even a scalar can hold
        a mutable Python payload. Trivia, paths and key parts can be
        shared. Constructors supply independent links and ref lists.
        The clone registers itself in ``memo``; slot-run copying calls
        this method directly.
        """
        new: Slot
        if type(self) is KVSlot:
            value = self.value
            if not isinstance(value, ScalarValue) or not value.is_shareable:
                value = copy.deepcopy(value, memo)
            new = KVSlot(
                self.leading,
                None,
                self.eol,
                self.host_path,
                self.key_parts,
                self.key_seps,
                self.key_path,
                self.pre_eq,
                self.post_eq,
                value,
            )
        else:
            assert type(self) is StructuralHeaderSlot
            new = StructuralHeaderSlot(
                self.leading,
                None,
                self.eol,
                self.key_parts,
                self.key_seps,
                self.key_path,
                self.inner_pre,
                self.inner_post,
                None,
                self.synthetic,
            )
        memo[id(self)] = new
        return new

    def render(self) -> str:
        raise NotImplementedError


class KVSlot(Slot):
    """A single ``key = value`` line.

    ``key_parts`` holds verbatim spellings aligned with ``key_path``'s
    decoded names; ``key_seps`` has one separator between each pair.
    All three tuples are replaced together when rebasing a key.
    """

    __slots__ = (
        "host_path",
        "key_parts",
        "key_path",
        "key_seps",
        "post_eq",
        "pre_eq",
        "value",
    )

    host_path: tuple[str, ...]
    """Full path of the table body this KV physically belongs to."""

    key_parts: tuple[str, ...]
    key_seps: tuple[str, ...]
    key_path: tuple[str, ...]

    pre_eq: str
    post_eq: str
    value: Value

    def __init__(
        self,
        leading: str,
        owner_aot_entry: AoTEntry | None,
        eol: str,
        host_path: tuple[str, ...],
        key_parts: tuple[str, ...],
        key_seps: tuple[str, ...],
        key_path: tuple[str, ...],
        pre_eq: str,
        post_eq: str,
        value: Value,
    ) -> None:
        self.leading = leading
        self.owner_aot_entry = owner_aot_entry
        self.eol = eol
        self._prev = None
        self._next = None
        self._order = 0
        self._containers = []
        self.host_path = host_path
        self.key_parts = key_parts
        self.key_seps = key_seps
        self.key_path = key_path
        self.pre_eq = pre_eq
        self.post_eq = post_eq
        self.value = value

    @override
    def render(self) -> str:
        return (
            f"{self.leading}{render_dotted(self.key_parts, self.key_seps)}"
            f"{self.pre_eq}={self.post_eq}{self.value.render()}{self.eol}"
        )


class StructuralHeaderSlot(Slot):
    """One ``[a.b]`` or ``[[a.b]]`` header line.

    ``entry`` is the discriminator: AoT-entry headers carry an
    :class:`AoTEntry`, plain table headers carry ``None``.
    """

    __slots__ = (
        "entry",
        "inner_post",
        "inner_pre",
        "key_parts",
        "key_path",
        "key_seps",
        "synthetic",
    )

    key_parts: tuple[str, ...]
    key_seps: tuple[str, ...]
    key_path: tuple[str, ...]
    inner_pre: str
    inner_post: str

    entry: AoTEntry | None
    """The AoT entry this header opens; ``None`` for a plain table."""

    synthetic: bool
    """Whether mutation may omit this header when its section becomes implicit."""

    def __init__(
        self,
        leading: str,
        owner_aot_entry: AoTEntry | None,
        eol: str,
        key_parts: tuple[str, ...],
        key_seps: tuple[str, ...],
        key_path: tuple[str, ...],
        inner_pre: str,
        inner_post: str,
        entry: AoTEntry | None,
        synthetic: bool,  # noqa: FBT001
    ) -> None:
        self.leading = leading
        self.owner_aot_entry = owner_aot_entry
        self.eol = eol
        self._prev = None
        self._next = None
        self._order = 0
        self._containers = []
        self.key_parts = key_parts
        self.key_seps = key_seps
        self.key_path = key_path
        self.inner_pre = inner_pre
        self.inner_post = inner_post
        self.entry = entry
        self.synthetic = synthetic

    @override
    def render(self) -> str:
        open_br, close_br = ("[[", "]]") if self.entry is not None else ("[", "]")
        return (
            f"{self.leading}{open_br}{self.inner_pre}"
            f"{render_dotted(self.key_parts, self.key_seps)}"
            f"{self.inner_post}{close_br}{self.eol}"
        )


# ---------------------------------------------------------------------------
# Doc-stream order keys
#
# Handing every item of a list an integer label that stays monotone in
# list order under insertion is the *list-labeling* problem, and the
# shape here — lay keys out with gaps, and on collision redistribute a
# neighbourhood — is its standard solution, not something invented for
# tomlrt.
#
# `_respread` is a folklore variant of that solution rather than one of
# the published algorithms, though. Those (Itai, Konheim and Rodeh, ICALP
# 1981; Bender et al., ESA 2002) redistribute an interval of the *label
# range*, aligned to an implicit binary tree over it, and pick which
# interval by a density threshold that tightens the larger the interval
# gets. The tightening is what keeps labels inside a bounded range, which
# matters when they have to fit a fixed universe. `_respread` instead
# redistributes a window of neighbouring *slots*, unaligned, taking the
# first window whose keys leave room for the run.
#
# So there is no published amortised bound here. What there is: a repair
# stays next to the edit that caused it, and these labels are Python ints
# with no universe to keep them inside, so a tightening threshold buys
# nothing — measured, it only makes each respread reach further for room
# it does not need.
# ---------------------------------------------------------------------------

_ORDER_GAP = 1 << 16
"""Nominal spacing between the order keys of adjacent slots.

Fresh keys are laid out this far apart so that a slot spliced into a
seam can take the midpoint; only a seam that has absorbed
``log2(_ORDER_GAP)`` inserts runs out of room and needs `_respread`.
"""

_ORDER_MIN_STEP = 8
"""Spacing `_respread` settles for when it has to compress a window.

Small enough to keep the respread window local — insisting on the
nominal spacing makes it reach further for the room — but not so small
that the window is exhausted again immediately.
"""


def stitch_run(prev: Slot | None, run: Sequence[Slot], nxt: Slot | None) -> None:
    """Link ``run``, in order, between ``prev`` and ``nxt``, stamping order keys.

    The sole point where slots join a doc-stream, so the sole point
    responsible for keeping :attr:`Slot._order` monotone. The whole
    run's keys are allocated up front, from the stream state before any
    of it is linked. The caller is responsible for the document's own
    head and tail.
    """
    key, step = _order_run_between(prev, nxt, len(run))
    for slot in run:
        slot._order = key  # noqa: SLF001
        key += step
        slot._prev = prev  # noqa: SLF001
        slot._next = nxt  # noqa: SLF001
        if prev is not None:
            prev._next = slot  # noqa: SLF001
        prev = slot
    if nxt is not None:
        nxt._prev = prev  # noqa: SLF001


def _order_run_between(
    prev: Slot | None, nxt: Slot | None, count: int
) -> tuple[int, int]:
    """First key and step for ``count`` keys between two adjacent slots.

    The keys ``first + i * step`` for ``i`` in ``range(count)`` all lie
    strictly between the two slots' own keys. ``None`` means "no slot
    on that side" — the head or tail seam, with unlimited room at
    nominal spacing. Respreads a neighbourhood of the stream if the
    seam can't hold the whole run; allocating one run in a single pass
    is what keeps a bulk splice (reorder, block move) linear, rather
    than respreading per slot.
    """
    if prev is None:
        stop = nxt._order if nxt is not None else count * _ORDER_GAP  # noqa: SLF001
        return stop - count * _ORDER_GAP, _ORDER_GAP
    if nxt is None:
        return prev._order + _ORDER_GAP, _ORDER_GAP  # noqa: SLF001
    if nxt._order - prev._order <= count:  # noqa: SLF001
        _respread(prev, nxt, count)
    step = (nxt._order - prev._order) // (count + 1)  # noqa: SLF001
    return prev._order + step, step  # noqa: SLF001


def _respread(left: Slot, right: Slot, count: int) -> None:
    """Re-lay the order keys of a window around an exhausted seam.

    Grows a window of slots outward from ``left``/``right`` until the
    enclosing key range can hold ``count`` slots at ``_ORDER_MIN_STEP``
    spacing — or wider, if the slots about to be inserted need more room
    than that — then redistributes them evenly. Growth always terminates
    because reaching an end of the stream gives the window room to expand
    into rather than compress.

    The window doubles at each probe rather than creeping outward one
    slot at a time, which settles within a factor of two of the smallest
    window that would serve; the room that overshoot carries is what
    pushes the next exhaustion of the same seam further out.
    """
    need = max(_ORDER_MIN_STEP, count + 1)
    lo, hi, window = left, right, 2
    below, above = lo._prev, hi._next  # noqa: SLF001
    while below is not None and above is not None:
        room = above._order - below._order  # noqa: SLF001
        if room >= (window + 1) * need:
            break
        target = window * 2
        while window < target and below is not None and above is not None:
            lo, hi, window = below, above, window + 2
            below, above = lo._prev, hi._next  # noqa: SLF001

    # An absent bound is an end of the stream, where the window can have
    # all the room it wants; keys are plain ints, so a head-end window
    # simply extends below the current head key, negative if need be.
    span = (window + 1) * max(_ORDER_GAP, need)
    low = below._order if below is not None else lo._order - span  # noqa: SLF001
    high = above._order if above is not None else low + span  # noqa: SLF001

    step = (high - low) // (window + 1)
    cur: Slot | None = lo
    for i in range(1, window + 1):
        assert cur is not None
        cur._order = low + i * step  # noqa: SLF001
        cur = cur._next  # noqa: SLF001


# ---------------------------------------------------------------------------
# Per-container slot geometry
# ---------------------------------------------------------------------------


def slot_local_key(slot: Slot, container: Container) -> str | None:
    """Derive a slot's key in ``container._index`` from their paths.

    The container's own header has no local key; it lives in ``_refs``
    and ``_header``, not ``_index``.
    """
    depth = len(container._path)  # noqa: SLF001
    if isinstance(slot, KVSlot):
        return slot.key_path[depth - len(slot.host_path)]
    assert isinstance(slot, StructuralHeaderSlot)
    path = slot.key_path
    return path[depth] if len(path) > depth else None


def ensure_terminator(slot: Slot, nl: str) -> None:
    """Give ``slot`` a trailing ``nl`` if it lacks one.

    A slot parsed as the file's final line carries no terminator. Once a
    mutation moves it off the tail it needs one, or it would run into
    whatever now follows it.
    """
    if not slot.eol.endswith("\n"):
        slot.eol += nl


def retarget_slot_newlines(slot: Slot, target: str) -> None:
    """Rewrite every line terminator reachable from ``slot`` to ``target``.

    Used when splicing slots across documents, so they adopt the
    destination's line ending, including in nested inline values.
    """
    slot.leading = retarget_newlines(slot.leading, target)
    slot.eol = retarget_newlines(slot.eol, target)
    if isinstance(slot, KVSlot):
        retarget_value_newlines(slot.value, target)


__all__ = [
    "AoTEntry",
    "KVSlot",
    "Slot",
    "StructuralHeaderSlot",
    "ensure_terminator",
    "retarget_slot_newlines",
    "slot_local_key",
    "stitch_run",
]
