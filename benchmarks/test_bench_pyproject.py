"""Parse-throughput benchmark for a pyproject.toml-shaped input.

Times `tomlrt.loads` over the repository's own `pyproject.toml`, which is
broadly representative of the workload `tomlrt` is expected to handle in
real projects.

Usage:

    make bench
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import tomlrt

if TYPE_CHECKING:
    from collections.abc import Callable

    from pytest_benchmark.fixture import BenchmarkFixture

    Throughput = Callable[[BenchmarkFixture, int], None]


def test_parse_pyproject(
    benchmark: BenchmarkFixture, pyproject_src: str, record_throughput: Throughput
) -> None:
    benchmark(tomlrt.loads, pyproject_src)
    record_throughput(benchmark, len(pyproject_src.encode()))
