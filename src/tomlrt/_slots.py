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
    def path(self) -> tuple[str, ...]:
        """Decoded path of the entry, taken from its header slot."""
        header = self.entry_slots[0]
        assert isinstance(header, StructuralHeaderSlot)
        return header.path


# ---------------------------------------------------------------------------
# Slot base + kinds
# ---------------------------------------------------------------------------


@dataclass(slots=True, eq=False)
class Slot:
    """Base for physical slots.

    Subclassed by `KVSlot` and `StructuralHeaderSlot`.

    Deliberately not an `abc.ABC`: giving it an `ABCMeta` metaclass
    would make every `isinstance(slot, KVSlot)` /
    `isinstance(slot, StructuralHeaderSlot)` check on the parse /
    mutation hot path go through the much slower ABC instance-check
    machinery instead of the plain-type fast path. `render` still
    raises rather than being left unimplemented so a missing override
    fails loudly instead of silently.

    Constructor fields are positional and required: a slot is built once
    per physical line, and keyword binding roughly doubles that cost.
    ``_prev`` / ``_next`` / ``_refs`` are runtime wiring, declared
    ``init=False`` so they take no constructor argument and don't force
    subclass fields keyword-only.
    """

    leading: str
    owner_aot_entry: AoTEntry | None
    """The AoT entry that physically contains this slot, if any.

    Lives on the base so all slot kinds expose the field uniformly.
    """

    _prev: Slot | None = field(default=None, init=False, repr=False, compare=False)
    _next: Slot | None = field(default=None, init=False, repr=False, compare=False)
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
    eol: EolTrivia

    @property
    def key(self) -> tuple[str, ...]:
        """Decoded dotted-key path, derived from ``key_parts``."""
        return tuple(p.value for p in self.key_parts)

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

    path: tuple[str, ...]
    """Full decoded path of the section / AoT entry header."""

    key_parts: list[KeyPart]
    key_seps: list[str]
    inner_pre: str
    inner_post: str
    eol: EolTrivia

    entry: AoTEntry | None
    """The AoT entry this header opens; ``None`` for a plain table."""

    synthetic: bool
    """True iff this header was introduced by mutation."""

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
        if slot.path == c_path:
            return None
        return slot.path[len(c_path)]


def ensure_terminator(slot: Slot, nl: str) -> None:
    """Give ``slot`` a trailing ``nl`` if it lacks one.

    A slot parsed as the file's final line carries no terminator. Once a
    mutation moves it off the tail it needs one, or it would run into
    whatever now follows it.
    """
    if isinstance(slot, (KVSlot, StructuralHeaderSlot)) and not slot.eol.newline:
        slot.eol.newline = nl


def retarget_slot_newlines(slot: Slot, target: str) -> None:
    """Rewrite every line terminator reachable from ``slot`` to ``target``.

    Used by graft paths so cross-document spliced slots adopt the
    destination document's line ending, including nested inline values.
    """
    slot.leading = retarget_newlines(slot.leading, target)
    if isinstance(slot, KVSlot):
        retarget_eol_newline(slot.eol, target)
        retarget_value_newlines(slot.value, target)
    elif isinstance(slot, StructuralHeaderSlot):
        retarget_eol_newline(slot.eol, target)


__all__ = [
    "AoTEntry",
    "KVSlot",
    "Slot",
    "SlotRef",
    "StructuralHeaderSlot",
    "ensure_terminator",
    "retarget_slot_newlines",
]
