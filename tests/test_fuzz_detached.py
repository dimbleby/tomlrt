"""Fuzzer for detached subtrees and long-lived view references.

``test_fuzz_mutation.py`` only ever mutates a document in place, and its
oracle only looks at what the document renders. Neither reaches the
shapes this module covers, and a whole family of faults lived in that
gap:

* **Detached subtrees.** ``doc.pop(k)`` hands back a live view whose
  slots move to a private orphan document. Adopting part of that orphan
  elsewhere, writing to what remains, and adopting the rest exercises
  the move-in-place machinery from both ends. Faults here cross-linked
  the two documents' slot streams, or leaked orphan content into the
  destination.

* **Held references.** Almost every fault found by hand took the form
  "take a view, displace it, keep using it". The in-place fuzzer holds
  no references across steps, so a view left resolving against a CST the
  document had discarded went unnoticed.

Two oracles run after every step. The first is the model check the
in-place fuzzer uses -- the render must re-parse to ``doc.to_dict()``.
The second, :func:`check_slot_chain`, inspects the CST itself.
That matters because a corrupt document can still render perfectly: a
cross-linked slot stream produced byte-correct output and only failed
much later, on an unrelated operation. Structural checks catch it at the
step that caused it.

Marked slow, like the other fuzzers. Each run draws fresh seeds and
reports the failing one for reproduction.
"""

from __future__ import annotations

import itertools
import random
import secrets
from typing import TYPE_CHECKING, Any

import pytest
import tomli

import tomlrt
from tomlrt import AoT, Array
from tomlrt._container import Container

if TYPE_CHECKING:
    from collections.abc import Callable

    from tomlrt._container import Document
    from tomlrt._slots import Slot

pytestmark = pytest.mark.slow

_PROGRAMS = 60  # random programs per starting shape, per run
_MAX_STEPS = 6

# Starting shapes chosen for their detach behaviour: dotted keys that
# leave an implicit orphan, headers that leave a header-bearing one,
# AoTs (whose entries detach separately from the AoT itself), a
# forward-declared descendant, and nested inline values that ride along
# inside a moved KV.
_SHAPES = (
    "[root.a]\nb.c = 1\n\n[root.a.e]\nq = 1\n\n[dest]\nz = 0\n",
    "[root.a]\nx = 1\n\n[dest]\nz = 0\n",
    "[[root.t]]\nx = 1\n\n[root.a]\nb.c = 3\n",
    "root.a.x = 1\nroot.a.y = 2\n\n[dest]\nz = 0\n",
    "[root.a.e]\nq = 1\n\n[root.a]\nb.c = 1\n\n[dest]\nz = 0\n",
    "[root.a]\np = 1\n\n[root.b]\nq = 2\n\n[dest]\nz = 0\n",
    "[[root.t]]\nx = 1\n\n[[root.t]]\nx = 2\n\n[dest]\nz = 0\n",
    "root.b.x = 1\n\n# comment\nroot.c.y = 2\n",
    "[[root.t]]\nx = 1\n\n[[root.t.n]]\ny = 2\n\n[dest]\nz = 0\n",
    "[root.a]\nv = { p = 1, q = [1, 2] }\n\n[dest]\nz = 0\n",
    "[root.a]\narr = [ { m = 1 }, 2 ]\n\n[dest]\nz = 0\n",
    # Adopting out of these empties a purely dotted ancestor, which then
    # has to grow a header of its own to stay renderable.
    "root.c.y.x = 1\n",
    "root.a.b.c = 1\nroot.a.b.d = 2\n\n[dest]\nz = 0\n",
    "root.a.b.c = 1\n\n# comment\nroot.e.f = 2\n\n[dest]\nz = 0\n",
)


def _chain(doc: Document) -> list[Slot]:
    """The document's slots, walked forward from its head."""
    out: list[Slot] = []
    seen: set[int] = set()
    cur = doc._head  # noqa: SLF001
    while cur is not None:
        assert id(cur) not in seen, "cycle in slot chain"
        seen.add(id(cur))
        out.append(cur)
        cur = cur._next  # noqa: SLF001
    return out


def check_slot_chain(doc: Document, ctx: str) -> None:
    """Assert the document's physical slot stream is well formed.

    Cheap enough to run after every fuzz step. These are the checks
    whose violation has been seen to render correctly and only fail
    later, on an unrelated operation:

    * the chain is acyclic and its links are symmetric;
    * ``_head`` / ``_tail`` really are its ends.
    """
    slots = _chain(doc)

    if slots:
        assert doc._tail is slots[-1], f"{ctx}: _tail is not the chain end"  # noqa: SLF001
        assert slots[0]._prev is None, f"{ctx}: head has a predecessor"  # noqa: SLF001
    else:
        assert doc._tail is None, f"{ctx}: empty chain with a _tail"  # noqa: SLF001

    for a, b in itertools.pairwise(slots):
        assert b._prev is a, f"{ctx}: broken back-link"  # noqa: SLF001


def foreign_refs(doc: Document) -> list[tuple[tuple[str, ...], Slot]]:
    """Refs filed on ``doc``'s containers naming slots outside its chain.

    A container's caches should only ever name slots of its own
    document; anything else means two documents share bookkeeping.
    """
    in_chain = {id(s) for s in _chain(doc)}
    found: list[tuple[tuple[str, ...], Slot]] = []

    def visit(c: Container) -> None:
        found.extend(
            (c._path, ref.slot)  # noqa: SLF001
            for ref in c._refs  # noqa: SLF001
            if id(ref.slot) not in in_chain
        )
        for child in c.values():
            if isinstance(child, Container) and not child._inline:  # noqa: SLF001
                visit(child)
            elif isinstance(child, AoT):
                for entry in child:
                    visit(entry)

    visit(doc)
    return found


def _view_paths(node: Container, prefix: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
    """Paths of the section-ish children reachable from ``node``."""
    out: list[tuple[str, ...]] = []
    for key in list(node.keys()):
        value = dict.__getitem__(node, key)
        if isinstance(value, Container) and not value._inline:  # noqa: SLF001
            out.append((*prefix, key))
            out.extend(_view_paths(value, (*prefix, key)))
        elif isinstance(value, (AoT, Array)):
            out.append((*prefix, key))
    return out


def _resolve(node: Any, path: tuple[str, ...]) -> Any:
    for part in path:
        node = dict.__getitem__(node, part)
    return node


def _run_program(src: str, seed: int) -> None:
    """Run one random detach / adopt / write program against ``src``."""
    rng = random.Random(seed)  # noqa: S311
    doc = tomlrt.loads(src)
    orphan = doc.pop("root")
    held: list[Any] = []

    for step in range(rng.randint(1, _MAX_STEPS)):
        choices = [
            "adopt",
            "adopt",
            "write_orphan",
            "write_doc",
            "hold",
            "write_held",
            "delete_orphan",
            "overwrite_orphan",
            "sort_orphan",
        ]
        op = rng.choice(choices)
        paths = _view_paths(orphan)
        if op == "adopt":
            target = _resolve(orphan, rng.choice([*paths, ()]))
            doc[f"m{step}"] = target
        elif op == "write_orphan":
            target = _resolve(orphan, rng.choice([*paths, ()]))
            if isinstance(target, Container):
                target[f"n{step}"] = step
        elif op == "hold":
            held.append(_resolve(orphan, rng.choice([*paths, ()])))
        elif op == "write_held" and held:
            target = rng.choice(held)
            if isinstance(target, Container):
                target[f"h{step}"] = step
            elif isinstance(target, Array):
                target.append(step)
        elif op == "overwrite_orphan":
            # Re-using an existing key is what reaches the reposition
            # path; every other write here invents a fresh one.
            target = _resolve(orphan, rng.choice([*paths, ()]))
            if isinstance(target, Container) and list(target.keys()):
                key = rng.choice(list(target.keys()))
                # Replacing a key with an existing sibling view reaches
                # the same-document adopt; a fresh value never does.
                cands: list[Any] = [step, {"r": step}]
                if paths:
                    cands.append(_resolve(orphan, rng.choice(paths)))
                value = rng.choice(cands)
                if value is target or value is dict.__getitem__(target, key):
                    value = step
                target[key] = value
        elif op == "sort_orphan":
            target = _resolve(orphan, rng.choice([*paths, ()]))
            if isinstance(target, Container):
                target.sort(reverse=bool(rng.getrandbits(1)))
        elif op == "delete_orphan":
            target = _resolve(orphan, rng.choice([*paths, ()]))
            if isinstance(target, Container) and list(target.keys()):
                del target[rng.choice(list(target.keys()))]
        else:
            doc[f"k{step}"] = step

        ctx = f"seed={seed} step={step} op={op} src={src!r}"
        out = tomlrt.dumps(doc)
        assert tomli.loads(out) == doc.to_dict(), f"{ctx}: render disagrees with model"
        check_slot_chain(doc, ctx)
        assert foreign_refs(doc) == [], f"{ctx}: source document holds a foreign ref"
        # The orphan is a document in its own right, so it answers the
        # same questions. Once it has been adopted away it is no longer
        # the orphan's, and the destination has already been checked.
        private = orphan._layout_root  # noqa: SLF001
        if private is not None and private is not doc:
            check_slot_chain(private, f"{ctx} [orphan]")
            assert foreign_refs(private) == [], f"{ctx}: orphan holds a foreign ref"


@pytest.mark.parametrize("src", _SHAPES)
def test_detached_subtree_programs_keep_model_consistent(src: str) -> None:
    """Random detach / adopt / held-write programs stay self-consistent."""
    for _ in range(_PROGRAMS):
        _run_program(src, secrets.randbits(64))


def test_displaced_views_never_resolve_against_a_dead_value() -> None:
    """A view displaced from a document must not still claim to be in it.

    Pins the live-attach rule directly rather than through rendered
    output: after a displacement the held view is detached, and
    re-assigning it makes it the live view at its new home.
    """
    displacements: tuple[tuple[str, str, str, Callable[[Document], object]], ...] = (
        (
            "inline overwrite",
            "x = { n.a = 1, z = 9 }\n",
            "x.n",
            lambda d: d["x"].__setitem__("n", 5),
        ),
        (
            "inline delete",
            "x = { n.a = 1, z = 9 }\n",
            "x.n",
            lambda d: d["x"].__delitem__("n"),
        ),
        ("inline pop", "x = { n.a = 1, z = 9 }\n", "x.n", lambda d: d["x"].pop("n")),
        (
            "array overwrite",
            "x = [ { c = 1 }, 2 ]\n",
            "x.0",
            lambda d: d["x"].__setitem__(0, 9),
        ),
        (
            "array delete",
            "x = [ { c = 1 }, 2 ]\n",
            "x.0",
            lambda d: d["x"].__delitem__(0),
        ),
        ("array pop", "x = [ { c = 1 }, 2 ]\n", "x.0", lambda d: d["x"].pop(0)),
        ("array clear", "x = [ { c = 1 }, 2 ]\n", "x.0", lambda d: d["x"].clear()),
        ("key overwrite", "x = { c = 1 }\n", "x", lambda d: d.__setitem__("x", 5)),
        ("key delete", "x = { c = 1 }\n", "x", lambda d: d.__delitem__("x")),
    )
    for label, src, hold, displace in displacements:
        doc = tomlrt.loads(src)
        held = doc["x"]
        if hold.endswith(".n"):
            held = held["n"]
        elif hold.endswith(".0"):
            held = held[0]
        displace(doc)

        assert held._layout_root is None, f"{label}: displaced view still attached"  # noqa: SLF001
        doc["y"] = held
        assert doc["y"] is held, f"{label}: reassignment cloned instead of attaching"
        out = tomlrt.dumps(doc)
        assert tomli.loads(out) == doc.to_dict(), f"{label}: render disagrees"
        check_slot_chain(doc, label)


def test_writing_to_a_popped_subtree_leaves_no_ref_behind() -> None:
    """A popped subtree's writes must not touch the document it left.

    A private orphan is rooted at the path its slots spell, so filing a
    binding walks an ancestor chain that ends at the orphan's own root.
    While the subtree's ancestry still ran back into the source
    document, that walk left the source naming a slot it did not
    contain, under a key it no longer had.
    """
    doc = tomlrt.loads("[root.a]\nb.c = 1\n\n[dest]\nz = 0\n")
    orphan = doc.pop("root")
    orphan["n"] = 1

    assert foreign_refs(doc) == []
    assert "root" not in doc._index  # noqa: SLF001


def test_adopting_a_slotless_orphan_root_unbinds_it() -> None:
    """Adopting an emptied orphan leaves nothing behind in it.

    The orphan is emptied by adopting its last slot-owning child away,
    so the root itself is synthesised at its new home rather than
    moved. That path still has to unbind it from the document it came
    from, exactly as the move paths do.
    """
    doc = tomlrt.loads("[root.a]\nb.c = 1\n\n[root.a.e]\nq = 1\n\n[dest]\nz = 0\n")
    orphan = doc.pop("root")
    private = orphan._layout_root  # noqa: SLF001
    assert private is not None

    doc["m0"] = orphan["a"]
    doc["m1"] = orphan

    assert dict.keys(private) == {}.keys()
    assert foreign_refs(private) == []
    check_slot_chain(private, "emptied orphan")
    out = tomlrt.dumps(doc)
    assert tomli.loads(out) == doc.to_dict()
