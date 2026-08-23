"""Build-throughput benchmarks for ``Document(mapping)``.

Times construction of a document from plain Python data -- the path that
turns nested mappings into ``[section]`` blocks, lists of mappings into
``[[aot]]`` blocks and everything else into key-value lines. It shares
almost nothing with parsing, so the parse benchmarks say nothing about
it.

The cases are the document *shapes* the build path treats differently,
sized so each round is worth timing. ``Document`` snapshot-copies plain
data, so a round leaves its input untouched and the plain fixture
applies.

Usage:

    make bench
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import tomlrt

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture


def _row(keys: int) -> dict[str, Any]:
    return {f"k{i}": i for i in range(keys)}


@pytest.fixture(scope="session")
def pyproject_data(pyproject_src: str) -> dict[str, Any]:
    """The repository's own ``pyproject.toml``, as plain data."""
    return tomlrt.loads(pyproject_src).to_dict()


def test_build_flat(benchmark: BenchmarkFixture) -> None:
    benchmark(tomlrt.Document, _row(2_000))


def test_build_sections(benchmark: BenchmarkFixture) -> None:
    data = {f"s{i}": _row(5) for i in range(500)}
    benchmark(tomlrt.Document, data)


def test_build_nested_sections(benchmark: BenchmarkFixture) -> None:
    data = {f"s{i}": {"mid": {"leaf": _row(5)}} for i in range(250)}
    benchmark(tomlrt.Document, data)


def test_build_aot(benchmark: BenchmarkFixture) -> None:
    data = {"items": [_row(5) for _ in range(500)]}
    benchmark(tomlrt.Document, data)


def test_build_many_aots(benchmark: BenchmarkFixture) -> None:
    """One single-entry AoT per key.

    Each attach installs an empty AoT, whose ``a = []`` placeholder the
    first entry then consumes -- invalidating the document's cached
    ``_body_tail`` once per key. This used to cost a walk of every ref
    filed so far.
    """
    data = {f"a{i}": [{"k": 1}] for i in range(1_000)}
    benchmark(tomlrt.Document, data)


def test_build_inline_arrays(benchmark: BenchmarkFixture) -> None:
    data = {f"k{i}": [1, 2, 3, 4, 5] for i in range(1_000)}
    benchmark(tomlrt.Document, data)


def test_build_pyproject(
    benchmark: BenchmarkFixture, pyproject_data: dict[str, Any]
) -> None:
    benchmark(tomlrt.Document, pyproject_data)
