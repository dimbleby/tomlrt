"""Mutation fuzzer over the ``toml-test`` valid corpus.

The deterministic regression tests pin specific mutation outcomes; the
property tests in ``test_hypothesis.py`` exercise the editing API but
mostly from an empty document. Neither broadly mutates *parsed*
documents with the structural shapes (nested AoTs, dotted keys,
sub-sections, out-of-order headers) that the corpus provides.

This module fills that gap. For each corpus document it runs a grid of
seeded random edit programs (set / delete / overwrite / sort across
containers, plus append / insert / pop / sort / reverse across arrays
and arrays-of-tables) and asserts the model stayed self-consistent:

* the rendered output is valid TOML (``tomllib`` accepts it);
* dump -> load -> dump is a fixed point (byte-exact idempotence);
* the rendered output reads back as the in-memory logical model
  (``tomllib.loads(dumps(doc))`` matches ``doc.to_dict()``).

The last oracle is the important one: it is sensitive to a mutation
that places a slot where a re-parse attributes it to a different owner
than the logical view says — the silent, self-consistent corruption an
idempotence check alone cannot see. Empty arrays-of-tables have no TOML
representation, so examples that produce one are excluded there.

Each run draws **fresh random seeds**, so the fuzzer explores new
programs every time rather than re-checking a frozen grid — it keeps
finding regressions instead of going stale. A failing example reports
its 64-bit seed, so it can be reproduced exactly with
``random.Random(<seed>)``. Raise ``_PROGRAMS`` to fuzz harder per run.

Per file the search runs many short uniform-random programs over the
parsed document; that is what reaches the precise multi-step states
(e.g. a non-contiguous dotted key with a foreign sibling, then a sort)
these bugs need — the structurally interesting corpus files get plenty
of attempts cheaply, while the trivial ones cost almost nothing.
"""

from __future__ import annotations

import math
import random
import secrets
import sys
from pathlib import Path
from typing import Any

import pytest

import tomlrt
from tomlrt import AoT, Array
from tomlrt._container import Container

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover -- 3.10 backport
    import tomli as tomllib

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
_PROGRAMS = 150  # random mutation programs per corpus file, per run


def _deep_eq(a: object, b: object) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        return (math.isnan(a) and math.isnan(b)) or a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_deep_eq(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(
            _deep_eq(x, y) for x, y in zip(a, b, strict=True)
        )
    return type(a) is type(b) and a == b


def _flatten(
    value: object, path: tuple[Any, ...] = ()
) -> dict[tuple[Any, ...], object]:
    """Map every non-empty scalar / array leaf to its path.

    Empty dicts contribute nothing, so the comparison is blind to which
    empty tables are representable in TOML (an inherent ambiguity that
    ``to_dict`` cannot see) and focuses on real value placement.
    """
    out: dict[tuple[Any, ...], object] = {}
    if isinstance(value, dict):
        for k, v in value.items():
            out.update(_flatten(v, (*path, k)))
    elif isinstance(value, list) and value and all(isinstance(x, dict) for x in value):
        for i, v in enumerate(value):
            out.update(_flatten(v, (*path, i)))
    else:
        out[path] = value
    return out


def _leaves_match(rendered: object, logical: object) -> bool:
    fa, fb = _flatten(rendered), _flatten(logical)
    return fa.keys() == fb.keys() and all(_deep_eq(fa[k], fb[k]) for k in fa)


def _has_empty_aot(value: object) -> bool:
    if isinstance(value, AoT):
        return len(value) == 0 or any(_has_empty_aot(e) for e in value)
    if isinstance(value, Array):
        return any(_has_empty_aot(e) for e in value)
    if isinstance(value, Container):
        return any(_has_empty_aot(value[k]) for k in value)
    return False


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


def _mutate(doc: tomlrt.Document, rng: random.Random) -> None:
    targets: list[tuple[str, Any]] = []
    _targets(doc, targets)
    if not targets:
        return
    kind, node = rng.choice(targets)
    try:
        if kind == "container":
            _mutate_container(node, rng)
        elif kind == "array":
            _mutate_array(node, rng)
        else:
            _mutate_aot(node, rng)
    except (KeyError, IndexError, TypeError, ValueError, tomlrt.TOMLError):
        # The fuzzer drives the API into unusual shapes; user-facing
        # errors are expected and tolerated. The invariant is that the
        # model stays consistent, not that every op succeeds.
        pass


def _mutate_container(node: Container, rng: random.Random) -> None:
    keys = list(node.keys())
    op = rng.choice(["set_new", "del", "overwrite", "sort"])
    if op == "set_new":
        node[f"k{rng.randint(0, 99)}"] = _rand_value(rng)
    elif op == "del" and keys:
        del node[rng.choice(keys)]
    elif op == "overwrite" and keys:
        node[rng.choice(keys)] = _rand_value(rng)
    elif op == "sort":
        node.sort()


def _mutate_array(node: Array, rng: random.Random) -> None:
    op = rng.choice(["append", "insert", "pop", "setitem", "sort", "reverse"])
    if op == "append":
        node.append(_rand_value(rng))
    elif op == "insert":
        node.insert(rng.randint(0, len(node)), _rand_value(rng))
    elif op == "pop" and node:
        node.pop(rng.randrange(len(node)))
    elif op == "setitem" and node:
        node[rng.randrange(len(node))] = _rand_value(rng)
    elif op == "sort":
        node.sort(key=repr)
    elif op == "reverse":
        node.reverse()


def _mutate_aot(node: AoT, rng: random.Random) -> None:
    op = rng.choice(["append", "insert", "pop", "reverse", "sort"])
    if op == "append":
        node.append({"new": _rand_value(rng)})
    elif op == "insert":
        node.insert(rng.randint(0, len(node)), {"new": _rand_value(rng)})
    elif op == "pop" and node:
        node.pop(rng.randrange(len(node)))
    elif op == "reverse":
        node.reverse()
    elif op == "sort":
        node.sort(key=lambda t: repr(dict(t)))


@pytest.mark.parametrize(
    "path", _CORPUS, ids=lambda p: p.relative_to(_VALID_ROOT).as_posix()
)
def test_mutation_keeps_model_consistent(path: Path) -> None:
    src = path.read_text(encoding="utf-8")
    try:
        tomllib.loads(src)
    except tomllib.TOMLDecodeError:
        pytest.skip("corpus entry not valid under stdlib tomllib")

    for _ in range(_PROGRAMS):
        # A fresh random seed every run, so the fuzzer explores new
        # programs instead of re-checking a frozen grid. The seed is
        # captured and reported on failure so any bug it finds is
        # reproducible (``random.Random(<seed>)``).
        seed = secrets.randbits(64)
        rng = random.Random(seed)  # noqa: S311
        doc = tomlrt.loads(src)
        for _ in range(rng.randint(1, 15)):
            _mutate(doc, rng)
        out = tomlrt.dumps(doc)
        ctx = f"{path.relative_to(_VALID_ROOT).as_posix()} seed={seed}"
        # Valid TOML and a fixed point of dump -> load -> dump.
        tomllib.loads(out)
        assert tomlrt.dumps(tomlrt.loads(out)) == out, f"non-idempotent: {ctx}\n{out!r}"
        # The rendered output reflects the in-memory logical model. Skip
        # when the model holds a non-representable empty array-of-tables.
        if not _has_empty_aot(doc):
            assert _leaves_match(tomllib.loads(out), doc.to_dict()), (
                f"render/model mismatch: {ctx}\n{out!r}\n"
                f"logical={doc.to_dict()!r}\nreparsed={tomllib.loads(out)!r}"
            )
