"""Physical slot layer.

A document is an intrusive doubly linked list of physical slots:
``KVSlot`` for ``key = value`` lines and ``StructuralHeaderSlot`` for
``[a.b]`` / ``[[a.b]]`` headers. `SlotRef` records a slot's occurrence
in one container.
"""

from __future__ import annotations

import copy
from dataclasses import MISSING, dataclass, field, fields
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tomlrt._container import Container
    from tomlrt._trivia import EolTrivia
    from tomlrt._values import KeyPart, Value

import sys

if sys.version_info >= (3, 12):
    from typing import override
else:  # pragma: no cover -- backport for Python < 3.12
    from typing_extensions import override

from tomlrt._trivia import retarget_eol_newline, retarget_newlines
from tomlrt._values import render_dotted, retarget_value_newlines

# ---------------------------------------------------------------------------
# AoT entry token (physical ownership marker)
# ---------------------------------------------------------------------------


@dataclass(slots=True, eq=False)
class AoTEntry:
    """Identifies one entry of an array-of-tables.

    Carried by every physical slot in that entry. ``entry_slots`` is a
    membership list populated by the slot-builder, with the entry's
    ``[[a]]`` header kept first; :attr:`path` is derived from it.
    """

    entry_slots: list[Slot] = field(default_factory=list)

    @property
    def header(self) -> StructuralHeaderSlot:
        """The entry's ``[[a]]`` header, kept first in ``entry_slots``."""
        header = self.entry_slots[0]
        assert isinstance(header, StructuralHeaderSlot)
        return header

    @property
    def path(self) -> tuple[str, ...]:
        """Decoded path of the entry, taken from its header slot."""
        return self.header.path


# ---------------------------------------------------------------------------
# Slot base + kinds
# ---------------------------------------------------------------------------


@dataclass(slots=True, eq=False)
class Slot:
    """Base for physical slots.

    Subclassed by `KVSlot` and `StructuralHeaderSlot`. Every slot kind
    spells one physical line as ``leading`` + body + ``eol``, so both
    trivia fields — and ``owner_aot_entry`` — live here and all kinds
    expose them uniformly; only the body differs.

    Deliberately not an `abc.ABC`: giving it an `ABCMeta` metaclass
    would make every `isinstance(slot, KVSlot)` /
    `isinstance(slot, StructuralHeaderSlot)` check on the parse /
    mutation hot path go through the much slower ABC instance-check
    machinery instead of the plain-type fast path. `render` still
    raises rather than being left unimplemented so a missing override
    fails loudly instead of silently.

    Constructor fields are positional and required: a slot is built once
    per physical line, and keyword binding roughly doubles that cost.
    ``_prev`` / ``_next`` / ``_order`` / ``_refs`` are runtime wiring,
    declared ``init=False`` so they take no constructor argument and
    don't force subclass fields keyword-only.
    """

    leading: str
    """Trivia before the slot's own text: blank lines, comment lines, indent."""

    owner_aot_entry: AoTEntry | None
    """The AoT entry that physically contains this slot, if any."""

    eol: EolTrivia
    """Trivia after the slot's own text: comment + line terminator."""

    _prev: Slot | None = field(default=None, init=False, repr=False, compare=False)
    _next: Slot | None = field(default=None, init=False, repr=False, compare=False)
    _order: int = field(default=0, init=False, repr=False, compare=False)
    """Doc-stream order key: strictly increasing along ``_next``.

    Lets "which of these two slots comes first?" — and hence the
    position of a slot's ref in a doc-ordered ``Container._refs`` — be
    answered by comparison instead of by walking the stream. Maintained
    by `stitch_run`; meaningless for an unlinked slot, which is stamped
    afresh when it is spliced back in.
    """
    _refs: list[SlotRef] = field(
        default_factory=list, init=False, repr=False, compare=False
    )
    """Back-pointers from this slot to every `SlotRef` that references it.

    Bounded length (≤ path depth + 1). AoT removal uses this to scrub
    refs in O(depth) per slot instead of O(siblings) per container.
    """

    def __deepcopy__(self, memo: dict[int, object]) -> Slot:
        """Deep-copy without following ``_refs``/``_prev``/``_next``.

        Clones start with empty refs and no doc-stream links; callers
        file new refs and splice them in. Following ``_prev``/``_next``
        would drag the whole source document into the deepcopy.
        """
        new = type(self).__new__(type(self))
        memo[id(self)] = new
        for f in fields(self):
            if f.init:
                value: object = copy.deepcopy(getattr(self, f.name), memo)
            elif f.default_factory is not MISSING:
                value = f.default_factory()
            else:
                value = f.default
            setattr(new, f.name, value)
        return new

    def render(self) -> str:
        raise NotImplementedError


@dataclass(slots=True, eq=False)
class KVSlot(Slot):
    """A single ``key = value`` line."""

    host_path: tuple[str, ...]
    """Full path of the table body this KV physically belongs to."""

    key_parts: list[KeyPart]
    """The dotted-key parts as written. ``len >= 1``."""

    key_seps: list[str]
    """Whitespace + ``.`` between parts. Length ``len(key_parts) - 1``."""

    pre_eq: str
    post_eq: str
    value: Value

    @property
    def key(self) -> tuple[str, ...]:
        """Decoded dotted-key path, derived from ``key_parts``."""
        # A list comprehension, not a generator: `tuple` can size the
        # result up front, and these are read on the build hot path.
        return tuple([p.value for p in self.key_parts])

    def render_key(self) -> str:
        return render_dotted(self.key_parts, self.key_seps)

    @override
    def render(self) -> str:
        return (
            f"{self.leading}{self.render_key()}"
            f"{self.pre_eq}={self.post_eq}"
            f"{self.value.render()}{self.eol.render()}"
        )


@dataclass(slots=True, eq=False)
class StructuralHeaderSlot(Slot):
    """One ``[a.b]`` or ``[[a.b]]`` header line.

    ``entry`` is the discriminator: AoT-entry headers carry an
    :class:`AoTEntry`, plain table headers carry ``None``. :attr:`kind`
    is derived from ``entry`` so the two cannot drift.
    """

    key_parts: list[KeyPart]
    key_seps: list[str]
    inner_pre: str
    inner_post: str

    entry: AoTEntry | None
    """The AoT entry this header opens; ``None`` for a plain table."""

    synthetic: bool
    """True iff this header was introduced by mutation."""

    @property
    def path(self) -> tuple[str, ...]:
        """Full decoded path of the section / AoT entry, from ``key_parts``."""
        return tuple([p.value for p in self.key_parts])

    @property
    def kind(self) -> Literal["table", "aot-entry"]:
        """``"aot-entry"`` iff ``entry is not None``, else ``"table"``."""
        return "aot-entry" if self.entry is not None else "table"

    def render_key(self) -> str:
        return render_dotted(self.key_parts, self.key_seps)

    @override
    def render(self) -> str:
        if self.entry is not None:
            open_br, close_br = "[[", "]]"
        else:
            open_br, close_br = "[", "]"
        return (
            f"{self.leading}{open_br}{self.inner_pre}"
            f"{self.render_key()}{self.inner_post}{close_br}{self.eol.render()}"
        )


# ---------------------------------------------------------------------------
# Doc-stream order keys
# ---------------------------------------------------------------------------

_ORDER_GAP = 1 << 16
"""Nominal spacing between the order keys of adjacent slots.

Fresh keys are laid out this far apart so that a slot spliced into a
seam can take the midpoint; only a seam that has absorbed
``log2(_ORDER_GAP)`` inserts runs out of room and needs `_respread`.
"""

_ORDER_MIN_STEP = 8
"""Spacing `_respread` settles for when it has to compress a window.

Small enough that a respread window stays local — the more the nominal
spacing is insisted on, the further a respread has to reach for the
room to restore it — but not so small that the window is exhausted
again immediately.
"""


def stitch_run(prev: Slot | None, run: Sequence[Slot], nxt: Slot | None) -> None:
    """Link ``run``, in order, between ``prev`` and ``nxt``, stamping order keys.

    The sole point at which slots join a doc-stream, and so the sole
    point that has to keep :attr:`Slot._order` — the key every
    doc-ordered projection of the stream is sorted by — monotone. The
    whole run's keys are allocated up front, from the state of the
    stream before any of it is linked. A caller holding the enclosing
    document is responsible for its head and tail.
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
    strictly between the two slots' own keys. ``None`` means "no slot on
    that side", i.e. the head or tail seam of the document, where there
    is unlimited room and the nominal spacing is used. Respreads a
    neighbourhood of the stream when the seam itself cannot hold the
    whole run. Allocating a run in one go is what keeps a bulk splice —
    a reorder, a block move — linear, rather than repeatedly halving one
    seam until it is exhausted and then respreading per slot.
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

    Grows the window outward from the ``left``/``right`` pair until the
    key range enclosing it can hold that many slots at ``_ORDER_MIN_STEP``
    spacing — or wider, if the ``count`` slots about to be inserted
    between the pair need more room than that — then redistributes them
    evenly across it. Growth always terminates because reaching an end
    of the stream gives the window room to expand into rather than
    compress.
    """
    need = max(_ORDER_MIN_STEP, count + 1)
    lo, hi, window = left, right, 2
    below, above = lo._prev, hi._next  # noqa: SLF001
    while below is not None and above is not None:
        room = above._order - below._order  # noqa: SLF001
        if room >= (window + 1) * need:
            break
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
# SlotRef (per-container occurrence)
# ---------------------------------------------------------------------------


class SlotRef:
    """A per-container occurrence of a slot.

    `local_key` derives the key under which the ref is filed in
    `container._index` from `(slot, container)` geometry.
    """

    __slots__ = ("container", "slot")

    def __init__(self, slot: Slot, container: Container) -> None:
        self.slot = slot
        self.container = container
        # Back-pointers let AoT removal scrub doomed slots without
        # scanning ancestor containers.
        slot._refs.append(self)  # noqa: SLF001

    @property
    def local_key(self) -> str | None:
        """Key under which this ref is filed in ``container._index``.

        ``None`` for the container's own header ref (which lives in
        ``_refs`` + ``_header_ref``, not ``_index``); otherwise a single
        path component derived from the slot path and container depth.
        """
        slot = self.slot
        c_path = self.container._path  # noqa: SLF001
        if isinstance(slot, KVSlot):
            return slot.key_parts[len(c_path) - len(slot.host_path)].value
        assert isinstance(slot, StructuralHeaderSlot)
        parts = slot.key_parts
        depth = len(c_path)
        return parts[depth].value if len(parts) > depth else None


def ensure_terminator(slot: Slot, nl: str) -> None:
    """Give ``slot`` a trailing ``nl`` if it lacks one.

    A slot parsed as the file's final line carries no terminator. Once a
    mutation moves it off the tail it needs one, or it would run into
    whatever now follows it.
    """
    if not slot.eol.newline:
        slot.eol.newline = nl


def retarget_slot_newlines(slot: Slot, target: str) -> None:
    """Rewrite every line terminator reachable from ``slot`` to ``target``.

    Used by graft paths so cross-document spliced slots adopt the
    destination document's line ending, including nested inline values.
    """
    slot.leading = retarget_newlines(slot.leading, target)
    retarget_eol_newline(slot.eol, target)
    if isinstance(slot, KVSlot):
        retarget_value_newlines(slot.value, target)


__all__ = [
    "AoTEntry",
    "KVSlot",
    "Slot",
    "SlotRef",
    "StructuralHeaderSlot",
    "ensure_terminator",
    "retarget_slot_newlines",
    "stitch_run",
]
