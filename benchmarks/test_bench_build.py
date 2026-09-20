"""Construction and serialization benchmarks for plain Python mappings.

Times the path that turns nested mappings into ``[section]`` blocks,
lists of mappings into ``[[aot]]`` blocks and everything else into
key-value lines. ``Document(mapping)`` also constructs logical views;
``dumps(mapping)`` serializes without them. Neither path parses input.

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


def test_build_nested_sections(benchmark: BenchmarkFixture) -> None:
    data = {f"s{i}": {"mid": {"leaf": _row(5)}} for i in range(250)}
    benchmark(tomlrt.Document, data)


def test_build_aot(benchmark: BenchmarkFixture) -> None:
    data = {"items": [_row(5) for _ in range(500)]}
    benchmark(tomlrt.Document, data)


def test_build_inline_arrays(benchmark: BenchmarkFixture) -> None:
    data = {f"k{i}": [1, 2, 3, 4, 5] for i in range(1_000)}
    benchmark(tomlrt.Document, data)


def test_build_pyproject(
    benchmark: BenchmarkFixture, pyproject_data: dict[str, Any]
) -> None:
    benchmark(tomlrt.Document, pyproject_data)


def test_dump_pyproject_mapping(
    benchmark: BenchmarkFixture, pyproject_data: dict[str, Any]
) -> None:
    benchmark(tomlrt.dumps, pyproject_data)
