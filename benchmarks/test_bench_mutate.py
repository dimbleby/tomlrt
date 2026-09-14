"""Manipulation-throughput benchmarks.

Times common edits over freshly parsed documents. Parse/render workflows
are labelled explicitly. The aim is to surface regressions in the
mutation path, which the parse-only benchmarks do not exercise.

Most cases are not idempotent -- appending grows the document, deleting
twice raises -- so each round needs its own parse, and `setup` is
pedantic-only. Cases that only read, or whose edit is a plain overwrite,
use the plain fixture and get calibrated iteration counts.

Usage:

    make bench
"""

from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING

import pytest

import tomlrt

if TYPE_CHECKING:
    from collections.abc import Callable

    from pytest_benchmark.fixture import BenchmarkFixture

    from tomlrt import Document


def _aot_doc(rows: int) -> str:
    return "".join(
        f'[[items]]\nname = "item-{i}"\nvalue = {i}\n\n' for i in range(rows)
    )


def _aot_with_trailing(rows: int, trailing_kvs: int) -> str:
    """AoT followed by an unrelated section with ``trailing_kvs`` keys.

    Mutating a non-tail AoT should cost independently of how much
    unrelated content follows it in the document.
    """
    return (
        _aot_doc(rows)
        + "[other]\n"
        + "".join(f"k{i} = {i}\n" for i in range(trailing_kvs))
    )


def _section_doc(sections: int, kvs: int) -> str:
    parts: list[str] = []
    for s in range(sections):
        parts.append(f"[s{s}]\n")
        parts.extend(f"k{k} = {k}\n" for k in range(kvs))
        parts.append("\n")
    return "".join(parts)


def _large_aot_entry(kvs: int) -> str:
    return "[[a]]\n" + "".join(f"k{i} = {i}\n" for i in range(kvs))


def _parsed(src: str) -> Callable[[], tuple[tuple[Document], dict[str, object]]]:
    """A ``pedantic`` setup that hands each round its own parse of ``src``."""

    def setup() -> tuple[tuple[Document], dict[str, object]]:
        return (tomlrt.loads(src),), {}

    return setup


# --- parse / render workflows (no per-round state) -------------------------


def test_render_only_pyproject(benchmark: BenchmarkFixture, pyproject_src: str) -> None:
    doc = tomlrt.loads(pyproject_src)
    benchmark(tomlrt.dumps, doc)


@pytest.mark.parametrize("operation", ["copy", "export"])
def test_copy_or_export_formatted_table(
    benchmark: BenchmarkFixture, operation: str
) -> None:
    values = ",".join(str(i) for i in range(1000))
    table = tomlrt.loads(f"t = {{\n  values = [ {values} ] # values\n}}\n").table("t")
    if operation == "copy":
        benchmark(copy, table)
    else:
        benchmark(table.to_dict)


def test_iterate_root_comments_over_nested_sections(
    benchmark: BenchmarkFixture,
) -> None:
    src = "".join(
        f"[packages.package_{i:04d}]\n"
        f'z_name = "package-{i}"\n'
        "enabled = true\n"
        f"versions = [{i + 2}, {i}, {i + 1}]\n"
        f'a_description = "Synthetic package number {i}"\n'
        for i in reversed(range(1_000))
    )
    comments = tomlrt.loads(src).comments

    def work() -> None:
        for key in comments:
            raise AssertionError(key)

    benchmark(work)


def test_parse_update_render_pyproject(
    benchmark: BenchmarkFixture, pyproject_src: str
) -> None:
    def work() -> None:
        doc = tomlrt.loads(pyproject_src)
        doc["project"]["version"] = "9.9.9"
        tomlrt.dumps(doc)

    benchmark(work)


# --- array-of-tables -------------------------------------------------------


def test_append_non_tail_aot_entries(benchmark: BenchmarkFixture) -> None:
    def work(doc: Document) -> None:
        aot = doc.aot("items")
        for i in range(20):
            aot.append({"name": f"new-{i}", "value": 1000 + i})

    benchmark.pedantic(work, setup=_parsed(_aot_with_trailing(50, 10_000)), rounds=50)


def test_clone_large_aot_entry(benchmark: BenchmarkFixture) -> None:
    def work(doc: Document) -> None:
        aot = doc.aot("a")
        aot.append(aot[0])

    benchmark.pedantic(work, setup=_parsed(_large_aot_entry(500)), rounds=50)


@pytest.mark.parametrize("trailing_kvs", [0, 10_000])
def test_clone_forward_declared_table(
    benchmark: BenchmarkFixture, trailing_kvs: int
) -> None:
    source = tomlrt.loads(
        "[target.child]\nx = 1\n[target]\ny = 2\n[other]\n"
        + "".join(f"k{i} = {i}\n" for i in range(trailing_kvs))
    ).table("target")

    def work(doc: Document) -> None:
        doc["copy"] = source

    benchmark.pedantic(work, setup=_parsed(""), rounds=200)


@pytest.mark.parametrize("size", [1_000, 16_000])
def test_clear_aot_before_sections(benchmark: BenchmarkFixture, size: int) -> None:
    def work(doc: Document) -> None:
        doc.aot("items").clear()

    src = _aot_doc(size) + _section_doc(size, 1)
    benchmark.pedantic(work, setup=_parsed(src), rounds=10)


def test_replace_aot_slice(benchmark: BenchmarkFixture) -> None:
    def work(doc: Document) -> None:
        doc.aot("items")[150:350] = (
            {"name": f"new-{i}", "value": 1000 + i} for i in range(200)
        )

    benchmark.pedantic(work, setup=_parsed(_aot_doc(500)), rounds=100)


def test_replace_aot_extended_slice(benchmark: BenchmarkFixture) -> None:
    def work(doc: Document) -> None:
        items = doc.aot("items")
        items[98::-2] = list(items[::2])

    benchmark.pedantic(work, setup=_parsed(_aot_doc(100)), rounds=100)


def test_replace_aot_entry(benchmark: BenchmarkFixture) -> None:
    def work(doc: Document) -> None:
        doc.aot("items")[0] = {"x": 1}

    benchmark.pedantic(work, setup=_parsed(_aot_doc(50)), rounds=200)


@pytest.mark.parametrize(
    ("entries", "depth"),
    [(500, 0), (1, 25)],
)
def test_attach_aot_factory(
    benchmark: BenchmarkFixture, entries: int, depth: int
) -> None:
    def setup() -> tuple[tuple[Document, tomlrt.AoT], dict[str, object]]:
        factory = tomlrt.AoT([{"values": list(range(5))} for _ in range(entries)])
        for _ in range(depth):
            factory = tomlrt.AoT([{"child": factory}])
        return (tomlrt.Document(), factory), {}

    def work(doc: Document, factory: tomlrt.AoT) -> None:
        doc["items"] = factory

    benchmark.pedantic(work, setup=setup, rounds=100)


def test_clone_implicit_with_empty_aot_entries(benchmark: BenchmarkFixture) -> None:
    source = tomlrt.loads(
        "source.marker = 0\n" + _aot_doc(1000).replace("[[items]]", "[[source.rows]]")
    )
    for entry in source.aot("source.rows"):
        entry["pending"] = tomlrt.AoT()
    table = source.table("source")

    def work(doc: Document) -> None:
        doc["copy"] = table

    benchmark.pedantic(work, setup=_parsed(""), rounds=20)


def test_replace_aot_from_ancestor(benchmark: BenchmarkFixture) -> None:
    src = "[parent]\nroot = 1\n\n" + _aot_doc(1000).replace(
        "[[items]]", "[[parent.items]]"
    )

    def work(doc: Document) -> None:
        parent = doc.table("parent")
        items = parent.aot("items")
        items[0] = {"nested": parent}

    benchmark.pedantic(work, setup=_parsed(src), rounds=100)


# --- sections --------------------------------------------------------------


def test_install_new_sections(benchmark: BenchmarkFixture) -> None:
    def work(doc: Document) -> None:
        for i in range(20):
            doc.install(("new", f"s{i}"), tomlrt.Table.section({"x": i}))

    benchmark.pedantic(work, setup=_parsed(_section_doc(50, 20)), rounds=200)


def test_promote_large_inline_array(benchmark: BenchmarkFixture) -> None:
    def work(doc: Document) -> None:
        doc.promote_array("items")

    src = "items = [{" + ", ".join(f"k{i} = {i}" for i in range(1_000)) + "}] # tail\n"
    benchmark.pedantic(work, setup=_parsed(src), rounds=100)


# --- inline arrays ---------------------------------------------------------


def _inline_array(rows: int) -> str:
    """A multi-line inline array carrying a comment on every row.

    Editing one of these is what drives the boundary machinery in
    `_comma_ops`: every insert or delete captures, composes and restores
    the trivia seam spanning the items either side of it. The comments
    matter -- a seam with a comment in it takes the whole `Boundary`
    path rather than the blank-only shortcut.
    """
    return (
        "items = [\n" + "".join(f"    {i},  # item {i}\n" for i in range(rows)) + "]\n"
    )


def test_clone_section_with_inline_value(benchmark: BenchmarkFixture) -> None:
    src = (
        "items = {\n"
        + "".join(f"    k{i} = {i}, # item {i}\n" for i in range(1_000))
        + "}\n"
    )
    source = tomlrt.loads("[source]\n" + src).table("source")

    def work(doc: Document) -> None:
        doc["copy"] = source

    benchmark.pedantic(work, setup=_parsed(""), rounds=100)


def test_insert_into_inline_array(benchmark: BenchmarkFixture) -> None:
    def work(doc: Document) -> None:
        arr = doc.array("items")
        for i in range(100):
            arr.insert(0, i)

    benchmark.pedantic(work, setup=_parsed(_inline_array(500)), rounds=100)


def test_append_nested_list_to_inline_array(benchmark: BenchmarkFixture) -> None:
    value = list(range(1000))

    def work(doc: Document) -> None:
        doc.array("items").append(value)

    benchmark.pedantic(work, setup=_parsed("items = []\n"), rounds=2_000)


def test_delete_from_inline_array(benchmark: BenchmarkFixture) -> None:
    def work(doc: Document) -> None:
        arr = doc.array("items")
        for _ in range(150):
            del arr[0]

    benchmark.pedantic(work, setup=_parsed(_inline_array(500)), rounds=100)


@pytest.mark.parametrize(("layout", "step"), [("single", 1), ("multiline", 2)])
def test_delete_inline_array_slice(
    benchmark: BenchmarkFixture, layout: str, step: int
) -> None:
    size = 16_000

    def work(doc: Document) -> None:
        del doc.array("items")[: size // 2 : step]

    src = (
        _inline_array(size)
        if layout == "multiline"
        else "items = [" + ", ".join(str(i) for i in range(size)) + "]\n"
    )
    benchmark.pedantic(work, setup=_parsed(src), rounds=20)


def test_insert_inline_array_slice(benchmark: BenchmarkFixture) -> None:
    size = 16_000
    values = list(range(size // 2))

    def work(doc: Document) -> None:
        at = size // 4
        doc.array("items")[at:at] = values

    benchmark.pedantic(work, setup=_parsed(_inline_array(size)), rounds=20)


# --- key-level edits -------------------------------------------------------


def test_bulk_update_1000_kvs(benchmark: BenchmarkFixture) -> None:
    """Overwriting existing keys is idempotent, so one document will do."""
    doc = tomlrt.loads(_section_doc(50, 20))

    def work() -> None:
        for s in range(50):
            section = doc.table(f"s{s}")
            for k in range(20):
                section[f"k{k}"] = k * 10

    benchmark(work)


def test_build_section_1000_new_keys(benchmark: BenchmarkFixture) -> None:
    def work(doc: Document) -> None:
        section = doc["t"]
        for i in range(1_000):
            section[f"k{i}"] = i

    benchmark.pedantic(work, setup=_parsed("[t]\n"), rounds=100)


def test_delete_root_kvs_tail_first(benchmark: BenchmarkFixture) -> None:
    """Delete the root body backwards, under a pile of section headers.

    Only a delete of the *current* body tail invalidates the cache, so
    going backwards invalidates on every key. The document root also
    holds a ref per section header, and those sit past the body.
    """

    def work(doc: Document) -> None:
        for i in reversed(range(1_000)):
            del doc[f"r{i}"]

    src = "".join(f"r{i} = {i}\n" for i in range(1_000)) + _section_doc(1_000, 1)
    benchmark.pedantic(work, setup=_parsed(src), rounds=20)


# --- sorting ---------------------------------------------------------------


def test_sort_nested_sections(benchmark: BenchmarkFixture) -> None:
    def work(doc: Document) -> None:
        for section in doc.values():
            assert isinstance(section, tomlrt.Table)
            section.sort()

    src = "".join(
        f"[s{s}]\nleaf = 1\n[s{s}.z]\nvalue = 1\n[s{s}.a]\nvalue = 2\n"
        for s in range(400)
    )
    benchmark.pedantic(work, setup=_parsed(src), rounds=20)


def test_sort_forward_declared_table(benchmark: BenchmarkFixture) -> None:
    def work(doc: Document) -> None:
        doc.table("target.inner").sort()

    src = (
        "root_value = 0\n[target.inner.z]\nvalue = 1\n[target]\ninner.a = 2\n"
        + "".join(f"[trailing_{i}]\nvalue = {i}\n" for i in range(1_000))
    )
    benchmark.pedantic(work, setup=_parsed(src), rounds=50)


def test_sort_wide_section(benchmark: BenchmarkFixture) -> None:
    def work(doc: Document) -> None:
        doc.table("s0").sort(reverse=True)

    benchmark.pedantic(work, setup=_parsed(_section_doc(1, 8_000)), rounds=20)
