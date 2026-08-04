"""Mutation fuzzer over the ``toml-test`` valid corpus, and from empty.

The deterministic regression tests pin specific mutation outcomes; the
property tests in ``test_fuzz_roundtrip.py`` exercise round-trip and
synthesis invariants, not broad mutation-API fuzzing. This module fills
that gap: it runs the same random edit-program fuzzer against every
corpus document (rich structural shapes: nested AoTs, dotted keys,
sub-sections, out-of-order headers) *and* against a single starting
empty document (reaching the from-scratch section/AoT creation path a
parsed starting point doesn't exercise as its first step). Programs
draw from one shared operation vocabulary -- set / delete / overwrite
/ sort / clone-or-graft across containers, plus append / insert / pop
/ sort / reverse across arrays and arrays-of-tables -- and assert the
model stayed self-consistent:

* the rendered output is valid TOML (``tomli`` accepts it);
* dump -> load -> dump is a fixed point (byte-exact idempotence);
* the rendered output reads back as the in-memory logical model
  (``tomli.loads(dumps(doc))`` matches ``doc.to_dict()``).

The last oracle is the important one: it is sensitive to a mutation
that places a slot where a re-parse attributes it to a different owner
than the logical view says — the silent, self-consistent corruption an
idempotence check alone cannot see.

Each run draws **fresh random seeds**, so the fuzzer explores new
programs every time rather than re-checking a frozen grid — it keeps
finding regressions instead of going stale. Every failure — an oracle
that disagrees or an exception raised from inside the library — reports
the 64-bit seed of the program that produced it, and re-running the same
test id with ``TOMLRT_FUZZ_SEED=<seed>`` set replays that one program::

    TOMLRT_FUZZ_SEED=<seed> uv run pytest -m slow \\
        "tests/test_fuzz_mutation.py::test_mutation_keeps_model_consistent[<id>]"

Raise ``_PROGRAMS`` to fuzz harder per run.

Per file the search runs many short uniform-random programs over the
parsed document; that is what reaches the precise multi-step states
(e.g. a non-contiguous dotted key with a foreign sibling, then a sort)
these bugs need — the structurally interesting corpus files get plenty
of attempts cheaply, while the trivial ones cost almost nothing.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pytest
import tomli

import tomlrt
from _helpers import deep_equal, fuzz_context, fuzz_seeds
from tomlrt import AoT, Array
from tomlrt._container import Container, Table

pytestmark = pytest.mark.slow

_REPO_ROOT = Path(__file__).resolve().parent.parent
_VALID_ROOT = _REPO_ROOT / "vendor" / "toml-test" / "tests" / "valid"

if not _VALID_ROOT.is_dir():
    pytest.skip(
        "toml-test corpus not vendored; run "
        "`git clone --depth 1 https://github.com/toml-lang/toml-test "
        "vendor/toml-test`",
        allow_module_level=True,
    )

_CORPUS = sorted(_VALID_ROOT.rglob("*.toml"))
_PROGRAMS = 100  # random mutation programs per corpus file, per run


def _rand_value(rng: random.Random) -> Any:
    return rng.choice(
        [1, 3.5, "str", True, -7, "x y", "", [1, 2], {"a": 1}, [{"q": 1}]]
    )


def _targets(node: object, out: list[tuple[str, Any]], depth: int = 0) -> None:
    if depth > 7:
        return
    if isinstance(node, AoT):
        out.append(("aot", node))
        for entry in node:
            _targets(entry, out, depth + 1)
    elif isinstance(node, Array):
        out.append(("array", node))
        for item in node:
            _targets(item, out, depth + 1)
    elif isinstance(node, Container):
        out.append(("container", node))
        for key in list(node.keys()):
            _targets(node[key], out, depth + 1)


def _mutate(
    doc: tomlrt.Document, rng: random.Random, foreign_pool: list[tuple[str, Any]]
) -> None:
    """Apply one random mutation to ``doc``.

    ``foreign_pool`` is a second document's targets, combined with
    ``doc``'s own for clone/graft ops so they draw from either the same
    or a different document uniformly through one code path.
    """
    targets: list[tuple[str, Any]] = []
    _targets(doc, targets)
    if not targets:
        return
    kind, node = rng.choice(targets)
    pool = targets + foreign_pool
    try:
        if kind == "container":
            _mutate_container(node, rng, pool)
        elif kind == "array":
            _mutate_array(node, rng, pool)
        else:
            _mutate_aot(node, rng, pool)
    except (KeyError, IndexError, TypeError, ValueError, tomlrt.TOMLError):
        # The fuzzer drives the API into unusual shapes; user-facing
        # errors are expected and tolerated. The invariant is that the
        # model stays consistent, not that every op succeeds.
        pass


def _mutate_container(
    node: Container, rng: random.Random, pool: list[tuple[str, Any]]
) -> None:
    keys = list(node.keys())
    # "clone_table" / "clone_aot" assign an already-attached Table / AoT
    # (same- or foreign-document) rather than a fresh dict, exercising
    # `_attach_section` / `_attach_aot`'s clone paths. "set_new_structural"
    # builds a header-bearing ``[key]`` / ``[[key]]`` value from scratch
    # (not the plain-dict/-list inline forms `_rand_value` produces),
    # exercising the same structural-creation path whether ``node``
    # started out empty or already had content.
    tables = [t for kind, t in pool if kind == "container" and isinstance(t, Table)]
    aots = [t for kind, t in pool if kind == "aot"]
    ops = ["set_new", "del", "overwrite", "sort", "set_new_structural"]
    if tables:
        ops.append("clone_table")
    if aots:
        ops.append("clone_aot")
    op = rng.choice(ops)
    new_key = f"k{rng.randint(0, 99)}"
    clone_key = rng.choice(keys) if keys and rng.random() < 0.5 else new_key
    if op == "set_new":
        node[new_key] = _rand_value(rng)
    elif op == "del" and keys:
        del node[rng.choice(keys)]
    elif op == "overwrite" and keys:
        node[rng.choice(keys)] = _rand_value(rng)
    elif op == "clone_table":
        node[clone_key] = rng.choice(tables)
    elif op == "clone_aot":
        node[clone_key] = rng.choice(aots)
    elif op == "set_new_structural":
        node[new_key] = (
            Table.section({"v": _rand_value(rng)})
            if rng.random() < 0.5
            else AoT([{"v": _rand_value(rng)} for _ in range(rng.randint(0, 3))])
        )
    elif op == "sort":
        node.sort()


def _mutate_array(node: Array, rng: random.Random, pool: list[tuple[str, Any]]) -> None:
    # "clone_value" appends an already-attached inline Table or Array
    # (same- or foreign-document) rather than a fresh scalar/dict/list.
    clonables = [
        t
        for kind, t in pool
        if kind == "array"
        or (kind == "container" and isinstance(t, Table) and t.is_inline)
    ]
    ops = ["append", "insert", "pop", "setitem", "sort", "reverse"]
    if clonables:
        ops.append("clone_value")
    op = rng.choice(ops)
    if op == "append":
        node.append(_rand_value(rng))
    elif op == "insert":
        node.insert(rng.randint(0, len(node)), _rand_value(rng))
    elif op == "clone_value":
        node.append(rng.choice(clonables))
    elif op == "pop" and node:
        node.pop(rng.randrange(len(node)))
    elif op == "setitem" and node:
        node[rng.randrange(len(node))] = _rand_value(rng)
    elif op == "sort":
        node.sort(key=repr)
    elif op == "reverse":
        node.reverse()


def _mutate_aot(node: AoT, rng: random.Random, pool: list[tuple[str, Any]]) -> None:
    # "clone_table" appends an already-attached Table (same- or
    # foreign-document) rather than a fresh dict, exercising the AoT
    # clone path.
    tables = [t for kind, t in pool if kind == "container" and isinstance(t, Table)]
    ops = ["append", "insert", "pop", "reverse", "sort"]
    if tables:
        ops.append("clone_table")
    op = rng.choice(ops)
    if op == "append":
        node.append({"new": _rand_value(rng)})
    elif op == "insert":
        node.insert(rng.randint(0, len(node)), {"new": _rand_value(rng)})
    elif op == "clone_table":
        node.append(rng.choice(tables))
    elif op == "pop" and node:
        node.pop(rng.randrange(len(node)))
    elif op == "reverse":
        node.reverse()
    elif op == "sort":
        node.sort(key=lambda t: repr(dict(t)))


def _run_fuzz_programs(
    src: str, foreign_pool: list[tuple[str, Any]], ctx_label: str
) -> None:
    """Run one random mutation program per seed from :func:`fuzz_seeds`
    against ``src``, asserting the model stays self-consistent throughout
    (see module docstring).
    """
    for seed in fuzz_seeds(_PROGRAMS):
        rng = random.Random(seed)  # noqa: S311
        with fuzz_context(f"{ctx_label} seed={seed}"):
            doc = tomlrt.loads(src)
            for _ in range(rng.randint(1, 15)):
                _mutate(doc, rng, foreign_pool)
            out = tomlrt.dumps(doc)
            # Valid TOML and a fixed point of dump -> load -> dump.
            tomli.loads(out)
            assert tomlrt.dumps(tomlrt.loads(out)) == out, f"non-idempotent\n{out!r}"
            # The rendered output reflects the in-memory logical model.
            assert deep_equal(tomli.loads(out), doc.to_dict()), (
                f"render/model mismatch\n{out!r}\n"
                f"logical={doc.to_dict()!r}\nreparsed={tomli.loads(out)!r}"
            )


@pytest.mark.parametrize(
    "path", _CORPUS, ids=lambda p: p.relative_to(_VALID_ROOT).as_posix()
)
def test_mutation_keeps_model_consistent(path: Path) -> None:
    src = path.read_text(encoding="utf-8")
    try:
        tomli.loads(src)
    except tomli.TOMLDecodeError:
        pytest.skip("corpus entry not valid under tomli")

    # A fixed foreign document (the next corpus file, wrapping around)
    # gives clone/graft ops a cross-document source alongside
    # same-document ones, through the same code path. Read-only, so
    # parsed once and reused across every program below.
    foreign_path = _CORPUS[(_CORPUS.index(path) + 1) % len(_CORPUS)]
    try:
        foreign_doc = tomlrt.loads(foreign_path.read_text(encoding="utf-8"))
    except tomlrt.TOMLError:
        foreign_doc = tomlrt.loads("")
    foreign_pool: list[tuple[str, Any]] = []
    _targets(foreign_doc, foreign_pool)

    _run_fuzz_programs(src, foreign_pool, path.relative_to(_VALID_ROOT).as_posix())


def test_mutation_keeps_model_consistent_from_empty_document() -> None:
    """Same fuzzer and oracle, seeded from an empty document.

    Growing every structure from nothing -- rather than mutating what
    a parser already produced -- reaches the from-scratch section/AoT
    creation path (``Container.__setitem__`` assigning a fresh
    ``Table.section(...)`` / ``AoT(...)``, via `_mutate_container`'s
    "set_new_structural" op) as its *starting* state, not just as one
    mutation among many applied to an already-rich corpus document.
    Uses the first corpus file as a foreign pool so clone/graft ops
    still have something to draw from.
    """
    foreign_doc = tomlrt.loads(_CORPUS[0].read_text(encoding="utf-8"))
    foreign_pool: list[tuple[str, Any]] = []
    _targets(foreign_doc, foreign_pool)
    _run_fuzz_programs("", foreign_pool, "<empty>")
