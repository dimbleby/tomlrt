#!/usr/bin/env python3

"""Manipulation-throughput benchmark.

Times common edit workflows over a parsed document — append AoT
entry, deep set, scalar update, structural replace — followed by
re-render. The aim is to surface regressions in the mutation path,
which the parse-only benchmarks do not exercise.

Usage:

    uv run python benchmarks/bench_mutate.py
"""

from __future__ import annotations

import gc
import statistics
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

import tomlrt

if TYPE_CHECKING:
    from collections.abc import Callable

    from tomlrt import Document

_T = TypeVar("_T")

REPO_ROOT = Path(__file__).resolve().parent.parent


def _build_aot_doc(rows: int) -> str:
    return "".join(
        f'[[items]]\nname = "item-{i}"\nvalue = {i}\n\n' for i in range(rows)
    )


def _build_aot_with_trailing(rows: int, trailing_kvs: int) -> str:
    """AoT followed by an unrelated section with ``trailing_kvs`` keys.

    Mutating a non-tail AoT should cost independently of how much
    unrelated content follows it in the document.
    """
    return (
        _build_aot_doc(rows)
        + "[other]\n"
        + "".join(f"k{i} = {i}\n" for i in range(trailing_kvs))
    )


def _build_section_doc_with_trailing(trailing_kvs: int) -> str:
    """A section followed by another with ``trailing_kvs`` keys.

    Creating new sections should cost independently of how much
    unrelated content follows the insertion point.
    """
    return "[a]\nx = 1\n\n[other]\n" + "".join(
        f"k{i} = {i}\n" for i in range(trailing_kvs)
    )


def _build_section_doc(sections: int, kvs: int) -> str:
    parts: list[str] = []
    for s in range(sections):
        parts.append(f"[s{s}]\n")
        parts.extend(f"k{k} = {k}\n" for k in range(kvs))
        parts.append("\n")
    return "".join(parts)


def _bench(name: str, work: Callable[[], None], *, repeats: int) -> None:
    """Run ``work`` ``repeats`` times and print best/median wall time."""
    timings: list[float] = []
    for _ in range(repeats):
        gc.collect()
        gc.disable()
        try:
            t0 = time.perf_counter()
            work()
            t1 = time.perf_counter()
        finally:
            gc.enable()
        timings.append(t1 - t0)
    best = min(timings)
    median = statistics.median(timings)
    print(f"  {name:42s} best {best * 1e6:8.1f} us   median {median * 1e6:8.1f} us")


def _bench_with_setup(
    name: str,
    setup: Callable[[], _T],
    work: Callable[[_T], None],
    *,
    repeats: int,
) -> None:
    """Like :func:`_bench`, but ``setup()`` runs *outside* the timer.

    Used where the closure under test needs a freshly parsed document
    each iteration (mutation is not idempotent) but a large trailing
    document body would otherwise let one-time parse cost swamp the
    mutation cost being measured.
    """
    timings: list[float] = []
    for _ in range(repeats):
        state = setup()
        gc.collect()
        gc.disable()
        try:
            t0 = time.perf_counter()
            work(state)
            t1 = time.perf_counter()
        finally:
            gc.enable()
        timings.append(t1 - t0)
    best = min(timings)
    median = statistics.median(timings)
    print(f"  {name:42s} best {best * 1e6:8.1f} us   median {median * 1e6:8.1f} us")


def main() -> None:
    """Run the manipulation benchmark suite."""
    print(f"tomlrt : {tomlrt.__file__}")
    print(f"python : {sys.version.split()[0]}")
    print()

    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    aot_src = _build_aot_doc(500)
    section_src = _build_section_doc(50, 20)
    aot_trailing_src = _build_aot_with_trailing(50, 20_000)
    section_trailing_src = _build_section_doc_with_trailing(20_000)

    def parse_and_render_pyproject() -> None:
        doc = tomlrt.loads(pyproject)
        tomlrt.dumps(doc)

    def update_scalar() -> None:
        doc = tomlrt.loads(pyproject)
        doc["project"]["version"] = "9.9.9"
        tomlrt.dumps(doc)

    def append_aot_entry() -> None:
        doc = tomlrt.loads(aot_src)
        aot = doc.aot("items")
        for i in range(50):
            aot.append({"name": f"new-{i}", "value": 1000 + i})
        tomlrt.dumps(doc)

    def append_non_tail_aot_entry(doc: Document) -> None:
        aot = doc.aot("items")
        for i in range(20):
            aot.append({"name": f"new-{i}", "value": 1000 + i})

    def install_non_tail_new_sections(doc: Document) -> None:
        target = doc.table("a")
        for i in range(20):
            target.install((f"s{i}",), tomlrt.Table.section({"x": i}))

    def deep_set_new_section() -> None:
        doc = tomlrt.loads(section_src)
        for i in range(20):
            doc.install(("new", f"s{i}"), tomlrt.Table.section({"x": i}))
        tomlrt.dumps(doc)

    def bulk_kv_update() -> None:
        doc = tomlrt.loads(section_src)
        for s in range(50):
            sec = doc.table(f"s{s}")
            for k in range(20):
                sec[f"k{k}"] = k * 10
        tomlrt.dumps(doc)

    def build_section_new_keys() -> None:
        doc = tomlrt.loads("[t]\n")
        sec = doc["t"]
        for i in range(1000):
            sec[f"k{i}"] = i
        tomlrt.dumps(doc)

    def delete_kvs() -> None:
        doc = tomlrt.loads(section_src)
        for s in range(50):
            sec = doc.table(f"s{s}")
            del sec["k0"]
            del sec["k1"]
        tomlrt.dumps(doc)

    def render_only() -> None:
        tomlrt.dumps(doc_pyproject)

    doc_pyproject = tomlrt.loads(pyproject)

    _bench("parse + render: pyproject.toml", parse_and_render_pyproject, repeats=500)
    _bench("render only: pyproject.toml", render_only, repeats=2000)
    _bench("update scalar in pyproject", update_scalar, repeats=500)
    _bench("append 50 AoT entries (base 500)", append_aot_entry, repeats=200)
    _bench_with_setup(
        "append 20 entries, non-tail AoT (20k trailing)",
        lambda: tomlrt.loads(aot_trailing_src),
        append_non_tail_aot_entry,
        repeats=50,
    )
    _bench_with_setup(
        "install 20 new sections, non-tail (20k trailing)",
        lambda: tomlrt.loads(section_trailing_src),
        install_non_tail_new_sections,
        repeats=50,
    )
    _bench("install 20 new sections", deep_set_new_section, repeats=200)
    _bench("bulk update 1000 KVs", bulk_kv_update, repeats=100)
    _bench("build section with 1000 new keys", build_section_new_keys, repeats=100)
    _bench("delete 100 KVs", delete_kvs, repeats=200)


if __name__ == "__main__":
    main()
