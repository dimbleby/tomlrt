#!/usr/bin/env python3

"""Manipulation-throughput benchmark.

Times common edits over freshly parsed documents, with setup outside the
timed region. Parse/render workflows are labelled explicitly. The aim is to
surface regressions in the mutation path, which the parse-only benchmarks do
not exercise.

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


def _build_large_aot_entry(kvs: int) -> str:
    return "[[a]]\n" + "".join(f"k{i} = {i}\n" for i in range(kvs))


def _build_large_aot_entry_for_overwrite(kvs: int) -> str:
    return _build_large_aot_entry(kvs) + "\n[a.b.c]\nx = 1\n"


def _build_section_doc(sections: int, kvs: int) -> str:
    parts: list[str] = []
    for s in range(sections):
        parts.append(f"[s{s}]\n")
        parts.extend(f"k{k} = {k}\n" for k in range(kvs))
        parts.append("\n")
    return "".join(parts)


def _build_nested_section_doc(sections: int) -> str:
    return "".join(
        f"[s{section}]\n"
        "leaf = 1\n"
        f"[s{section}.z]\n"
        "value = 1\n"
        f"[s{section}.a]\n"
        "value = 2\n"
        for section in range(sections)
    )


def _build_forward_declared_sort_doc(trailing_sections: int) -> str:
    return "[target.z]\nvalue = 1\n[target]\na = 2\n" + "".join(
        f"[trailing_{i}]\nvalue = {i}\n" for i in range(trailing_sections)
    )


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

    Used where the closure under test needs fresh state each iteration
    because mutation is not idempotent.
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
    sortable_section_src = _build_section_doc(800, 20)
    wide_section_src = _build_section_doc(1, 8_000)
    nested_section_src = _build_nested_section_doc(400)
    forward_declared_sort_src = _build_forward_declared_sort_doc(20_000)
    aot_trailing_src = _build_aot_with_trailing(50, 20_000)
    section_trailing_src = _build_section_doc_with_trailing(20_000)
    large_aot_entry_src = _build_large_aot_entry(20_000)
    large_aot_overwrite_src = _build_large_aot_entry_for_overwrite(20_000)
    large_inline_array_src = (
        "items = [{" + ", ".join(f"k{i} = {i}" for i in range(1_000)) + "}] # tail\n"
    )

    def parse_and_render_pyproject() -> None:
        doc = tomlrt.loads(pyproject)
        tomlrt.dumps(doc)

    def update_scalar() -> None:
        doc = tomlrt.loads(pyproject)
        doc["project"]["version"] = "9.9.9"
        tomlrt.dumps(doc)

    def append_aot_entry(doc: Document) -> None:
        aot = doc.aot("items")
        for i in range(50):
            aot.append({"name": f"new-{i}", "value": 1000 + i})

    def append_non_tail_aot_entry(doc: Document) -> None:
        aot = doc.aot("items")
        for i in range(20):
            aot.append({"name": f"new-{i}", "value": 1000 + i})

    def append_after_large_aot_entry(doc: Document) -> None:
        doc.aot("a").append({"new": 1})

    def install_non_tail_new_sections(doc: Document) -> None:
        target = doc.table("a")
        for i in range(20):
            target.install((f"s{i}",), tomlrt.Table.section({"x": i}))

    def promote_large_inline_array(doc: Document) -> None:
        doc.promote_array("items")

    def overwrite_section_inside_aot_entry(doc: Document) -> None:
        doc.aot("a")[0].table("b")["c"] = 5

    def deep_set_new_section(doc: Document) -> None:
        for i in range(20):
            doc.install(("new", f"s{i}"), tomlrt.Table.section({"x": i}))

    def bulk_kv_update(doc: Document) -> None:
        for s in range(50):
            sec = doc.table(f"s{s}")
            for k in range(20):
                sec[f"k{k}"] = k * 10

    def build_section_new_keys(doc: Document) -> None:
        sec = doc["t"]
        for i in range(1000):
            sec[f"k{i}"] = i

    def delete_kvs(doc: Document) -> None:
        for s in range(50):
            sec = doc.table(f"s{s}")
            del sec["k0"]
            del sec["k1"]

    def sort_all_sections(doc: Document) -> None:
        for section in doc.values():
            assert isinstance(section, tomlrt.Table)
            section.sort()

    def reverse_sort_all_sections(doc: Document) -> None:
        for section in doc.values():
            assert isinstance(section, tomlrt.Table)
            section.sort(reverse=True)

    def sort_forward_declared_table(doc: Document) -> None:
        doc.table("target").sort()

    def reverse_sort_wide_section(doc: Document) -> None:
        doc.table("s0").sort(reverse=True)

    def render_only() -> None:
        tomlrt.dumps(doc_pyproject)

    doc_pyproject = tomlrt.loads(pyproject)

    _bench("parse + render: pyproject.toml", parse_and_render_pyproject, repeats=500)
    _bench("render only: pyproject.toml", render_only, repeats=2000)
    _bench("parse + update + render: pyproject", update_scalar, repeats=500)
    _bench_with_setup(
        "append 50 AoT entries (base 500)",
        lambda: tomlrt.loads(aot_src),
        append_aot_entry,
        repeats=200,
    )
    _bench_with_setup(
        "append 20 entries, non-tail AoT (20k trailing)",
        lambda: tomlrt.loads(aot_trailing_src),
        append_non_tail_aot_entry,
        repeats=50,
    )
    _bench_with_setup(
        "append after AoT entry with 20k KVs",
        lambda: tomlrt.loads(large_aot_entry_src),
        append_after_large_aot_entry,
        repeats=50,
    )
    _bench_with_setup(
        "install 20 new sections, non-tail (20k trailing)",
        lambda: tomlrt.loads(section_trailing_src),
        install_non_tail_new_sections,
        repeats=50,
    )
    _bench_with_setup(
        "promote array: one table with 1k fields",
        lambda: tomlrt.loads(large_inline_array_src),
        promote_large_inline_array,
        repeats=100,
    )
    _bench_with_setup(
        "overwrite section in AoT entry (20k KVs)",
        lambda: tomlrt.loads(large_aot_overwrite_src),
        overwrite_section_inside_aot_entry,
        repeats=50,
    )
    _bench_with_setup(
        "install 20 new sections (base 1k KVs)",
        lambda: tomlrt.loads(section_src),
        deep_set_new_section,
        repeats=200,
    )
    _bench_with_setup(
        "bulk update 1000 KVs",
        lambda: tomlrt.loads(section_src),
        bulk_kv_update,
        repeats=100,
    )
    _bench_with_setup(
        "build section with 1000 new keys",
        lambda: tomlrt.loads("[t]\n"),
        build_section_new_keys,
        repeats=100,
    )
    _bench_with_setup(
        "delete 100 KVs (base 1k KVs)",
        lambda: tomlrt.loads(section_src),
        delete_kvs,
        repeats=200,
    )
    _bench_with_setup(
        "sort 800 sections (20 keys each)",
        lambda: tomlrt.loads(sortable_section_src),
        reverse_sort_all_sections,
        repeats=20,
    )
    _bench_with_setup(
        "sort 400 sections (2 nested sections)",
        lambda: tomlrt.loads(nested_section_src),
        sort_all_sections,
        repeats=20,
    )
    _bench_with_setup(
        "sort forward-declared table (20k trailing)",
        lambda: tomlrt.loads(forward_declared_sort_src),
        sort_forward_declared_table,
        repeats=50,
    )
    _bench_with_setup(
        "sort one section (8k keys)",
        lambda: tomlrt.loads(wide_section_src),
        reverse_sort_wide_section,
        repeats=20,
    )


if __name__ == "__main__":
    main()
