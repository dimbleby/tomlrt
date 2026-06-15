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
from typing import TYPE_CHECKING

import tomlrt

if TYPE_CHECKING:
    from collections.abc import Callable

REPO_ROOT = Path(__file__).resolve().parent.parent


def _build_aot_doc(rows: int) -> str:
    return "".join(
        f'[[items]]\nname = "item-{i}"\nvalue = {i}\n\n' for i in range(rows)
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


def main() -> None:
    """Run the manipulation benchmark suite."""
    print(f"tomlrt : {tomlrt.__file__}")
    print(f"python : {sys.version.split()[0]}")
    print()

    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    aot_src = _build_aot_doc(500)
    section_src = _build_section_doc(50, 20)

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
    _bench("install 20 new sections", deep_set_new_section, repeats=200)
    _bench("bulk update 1000 KVs", bulk_kv_update, repeats=100)
    _bench("build section with 1000 new keys", build_section_new_keys, repeats=100)
    _bench("delete 100 KVs", delete_kvs, repeats=200)


if __name__ == "__main__":
    main()
