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
reports the failing one, which ``TOMLRT_FUZZ_SEED=<seed>`` replays.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

import pytest
import tomli

import tomlrt
from _helpers import (
    _chain,
    _containers,
    check_slot_chain,
    check_view_caches,
    fuzz_context,
    fuzz_seeds,
    td,
)
from tomlrt import AoT, Array, Table
from tomlrt._container import Container, _is_section

if TYPE_CHECKING:
    from collections.abc import Callable

    from tomlrt._container import Document
    from tomlrt._slots import Slot

pytestmark = pytest.mark.slow

_PROGRAMS = 200  # random programs per starting shape, per run
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
    td("""
        root.a.x=0x02 # body
        root.a.nested.value = [ 1,2 ]
        root.a.last = 3

        [root.a.deep]
        z = 0x01 # child

        [[root.a.rows]]
        id = 4

        [dest]
        z = 0
        """),
    td("""
        [root.a.deep]
        z = 0x01 # forward

        [root]
        a.x=0x02 # body
        a.nested.value = [ 1,2 ]
        a.last = 3

        [[root.a.rows]]
        id = 4

        [dest]
        z = 0
        """),
)

# One random operation per step, drawn uniformly; ``adopt`` appears
# twice because every other op is most interesting on a document an
# adopt has already reshaped.
_OPS = (
    "adopt",
    "adopt",
    "write_orphan",
    "write_doc",
    "hold",
    "write_held",
    "delete_orphan",
    "overwrite_orphan",
    "sort_orphan",
    "adopt_aot_entry",
    "adopt_into_section",
    "delete_doc",
    "mutate_array",
    "mutate_aot",
    "attach_commented",
    "attach_empty",
)


def foreign_refs(doc: Document) -> list[tuple[tuple[str, ...], Slot]]:
    """Refs filed on ``doc``'s containers naming slots outside its chain.

    A container's caches should only ever name slots of its own
    document; anything else means two documents share bookkeeping.
    """
    in_chain = {id(s) for s in _chain(doc)}
    return [
        (c._path, slot)  # noqa: SLF001
        for c in _containers(doc)
        for slot in c._refs  # noqa: SLF001
        if id(slot) not in in_chain
    ]


def _view_paths(node: Container, prefix: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
    """Paths of the section-ish children reachable from ``node``."""
    out: list[tuple[str, ...]] = []
    for key in list(node.keys()):
        value = dict.__getitem__(node, key)
        if _is_section(value):
            out.append((*prefix, key))
            out.extend(_view_paths(value, (*prefix, key)))
        elif isinstance(value, (AoT, Array)):
            out.append((*prefix, key))
    return out


def _resolve(node: Any, path: tuple[str, ...]) -> Any:
    for part in path:
        node = dict.__getitem__(node, part)
    return node


def _typed_paths(node: Container, want: type) -> list[tuple[str, ...]]:
    """Paths under ``node`` whose value is an instance of ``want``."""
    return [p for p in _view_paths(node) if isinstance(_resolve(node, p), want)]


def _run_program(src: str, seed: int) -> None:
    """Run one random detach / adopt / write program against ``src``."""
    rng = random.Random(seed)  # noqa: S311
    doc = tomlrt.loads(src)
    orphan = doc.pop("root")
    held: list[Any] = []

    for step in range(rng.randint(1, _MAX_STEPS)):
        op = rng.choice(_OPS)
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
            elif isinstance(target, AoT):
                # A held array may have been emptied by its last entry
                # being moved away, which detaches it; writing through
                # it must not reach the document that dropped it.
                target.add({f"h{step}": step})
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
        elif op == "adopt_aot_entry":
            # An entry detaches separately from the AoT that holds it,
            # and its `_path` names the array rather than the entry.
            aots = _typed_paths(orphan, AoT)
            if aots:
                aot = _resolve(orphan, rng.choice(aots))
                if len(aot):
                    doc[f"m{step}"] = aot[rng.randrange(len(aot))]
        elif op == "adopt_into_section":
            # Adopting below the destination root exercises a different
            # host / anchor than adopting at top level.
            dests = [
                p for p in _view_paths(doc) if isinstance(_resolve(doc, p), Container)
            ]
            if dests:
                into = _resolve(doc, rng.choice(dests))
                into[f"m{step}"] = _resolve(orphan, rng.choice([*paths, ()]))
        elif op == "delete_doc":
            keys = [k for k in doc if k != "root"]
            if keys:
                del doc[rng.choice(keys)]
        elif op == "mutate_array":
            arrays = _typed_paths(orphan, Array)
            if arrays:
                arr = _resolve(orphan, rng.choice(arrays))
                if arr and rng.getrandbits(1):
                    arr.pop(rng.randrange(len(arr)))
                else:
                    arr.append(step)
        elif op == "mutate_aot":
            aots = _typed_paths(orphan, AoT)
            if aots:
                aot = _resolve(orphan, rng.choice(aots))
                # Popping the last entry too: an array emptied that way
                # keeps its key and renders `k = []`, which is a shape
                # nothing else here produces.
                if len(aot) and rng.getrandbits(1):
                    aot.pop(rng.randrange(len(aot)))
                else:
                    aot.append({f"e{step}": step})
        elif op == "attach_commented":
            target = _resolve(orphan, rng.choice([*paths, ()]))
            inline = isinstance(target, Array) or (
                isinstance(target, tomlrt.Table) and target.is_inline
            )
            factory = (
                tomlrt.Table.inline({"value": step})
                if inline
                else tomlrt.Table.section({"value": step})
            )
            factory.comments["value"] = f"created {step}"
            if isinstance(target, Container):
                target[f"n{step}"] = factory
            else:
                target.append(factory)
        elif op == "attach_empty":
            # A section (or AoT entry) attached with no body of its own
            # is header-only: the one shape whose body-region cache has
            # nothing but the container's own header to name.
            aots = _typed_paths(orphan, AoT)
            if aots and rng.getrandbits(1):
                _resolve(orphan, rng.choice(aots)).append({})
            else:
                target = _resolve(orphan, rng.choice([*paths, ()]))
                # The orphan root itself may have become inline, which
                # cannot hold a section.
                if _is_section(target):
                    target.ensure_table(f"s{step}")
        else:
            doc[f"k{step}"] = step

        ctx = f"step={step} op={op}"
        out = tomlrt.dumps(doc)
        assert tomli.loads(out) == doc.to_dict(), f"{ctx}: render disagrees with model"
        check_slot_chain(doc, ctx)
        check_view_caches(doc, ctx)
        assert foreign_refs(doc) == [], f"{ctx}: source document holds a foreign ref"
        # The orphan is a document in its own right, so it answers the
        # same questions. Once it has been adopted away it is no longer
        # the orphan's, and the destination has already been checked.
        private = orphan._layout_root  # noqa: SLF001
        if private is not None and private is not doc:
            check_slot_chain(private, f"{ctx} [orphan]")
            check_view_caches(private, f"{ctx} [orphan]")
            assert foreign_refs(private) == [], f"{ctx}: orphan holds a foreign ref"
            left = tomlrt.dumps(private)
            assert tomli.loads(left) == private.to_dict(), (
                f"{ctx}: orphan render disagrees with its model"
            )


@pytest.mark.parametrize("src", _SHAPES)
def test_detached_subtree_programs_keep_model_consistent(src: str) -> None:
    """Random detach / adopt / held-write programs stay self-consistent."""
    for seed in fuzz_seeds(_PROGRAMS):
        with fuzz_context(f"seed={seed} src={src!r}"):
            _run_program(src, seed)


_CROWD_SECTIONS = 40
_CROWD_FILL = 2000
_CROWD_BODY = 1000


def test_bulk_move_into_a_seam_with_no_room_left() -> None:
    """A block moved into an exhausted seam leaves the chain ordered.

    The order keys that place refs in doc order are laid out with gaps,
    and a seam runs out of them only after thousands of writes into the
    one place -- far more than the programs above generate. A *bulk*
    move landing in such a seam is the case that needs the most room:
    getting it wrong hands several slots the same key, which reorders
    every cache built over those keys while leaving the render
    untouched, so only :func:`check_slot_chain` sees it.
    """
    src = "".join(f"[s{i:04d}]\nx = {i}\n\n" for i in range(_CROWD_SECTIONS))
    doc = tomlrt.loads(src)
    mid = _CROWD_SECTIONS // 2
    crowded = doc.table(f"s{mid:04d}")
    for i in range(_CROWD_FILL):
        crowded[f"k{i}"] = i

    def check(ctx: str) -> None:
        check_slot_chain(doc, ctx)
        check_view_caches(doc, ctx)
        assert foreign_refs(doc) == [], f"{ctx}: document holds a foreign ref"
        out = tomlrt.dumps(doc)
        assert tomli.loads(out) == doc.to_dict(), f"{ctx}: render disagrees with model"

    check("crowded")
    # The block replacing the section right after the crowded one lands
    # in the seam those writes compressed, and wants several times the
    # room left there.
    doc[f"s{mid + 1:04d}"] = Table.section({f"b{i}": i for i in range(_CROWD_BODY)})
    check("bulk move into the crowded seam")
    crowded["last"] = 1
    check("write into the crowded section")
    doc.sort()
    check("permute the compressed region")


_INLINE = "x = { n.a = 1, z = 9 }\n"
_ARRAY = "x = [ { c = 1 }, 2 ]\n"
_TABLE = "x = { c = 1 }\n"


@pytest.mark.parametrize(
    ("src", "hold", "displace"),
    [
        (_INLINE, lambda d: d["x"]["n"], lambda d: d["x"].__setitem__("n", 5)),
        (_INLINE, lambda d: d["x"]["n"], lambda d: d["x"].__delitem__("n")),
        (_INLINE, lambda d: d["x"]["n"], lambda d: d["x"].pop("n")),
        (_ARRAY, lambda d: d["x"][0], lambda d: d["x"].__setitem__(0, 9)),
        (_ARRAY, lambda d: d["x"][0], lambda d: d["x"].__delitem__(0)),
        (_ARRAY, lambda d: d["x"][0], lambda d: d["x"].pop(0)),
        (_ARRAY, lambda d: d["x"][0], lambda d: d["x"].clear()),
        (_TABLE, lambda d: d["x"], lambda d: d.__setitem__("x", 5)),
        (_TABLE, lambda d: d["x"], lambda d: d.__delitem__("x")),
    ],
    ids=[
        "inline-overwrite",
        "inline-delete",
        "inline-pop",
        "array-overwrite",
        "array-delete",
        "array-pop",
        "array-clear",
        "key-overwrite",
        "key-delete",
    ],
)
def test_displaced_views_never_resolve_against_a_dead_value(
    src: str,
    hold: Callable[[Document], Any],
    displace: Callable[[Document], object],
) -> None:
    """A view displaced from a document must not still claim to be in it.

    Pins the live-attach rule directly rather than through rendered
    output: after a displacement the held view is detached, and
    re-assigning it makes it the live view at its new home.
    """
    doc = tomlrt.loads(src)
    held = hold(doc)
    displace(doc)

    assert held._layout_root is None, "displaced view still attached"  # noqa: SLF001
    doc["y"] = held
    assert doc["y"] is held, "reassignment cloned instead of attaching"
    out = tomlrt.dumps(doc)
    assert tomli.loads(out) == doc.to_dict()
    check_slot_chain(doc, "reattached")


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
