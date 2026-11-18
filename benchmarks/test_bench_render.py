"""Render parsed values without timing their construction."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import tomlrt

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture


def test_render_only_pyproject(benchmark: BenchmarkFixture, pyproject_src: str) -> None:
    doc = tomlrt.loads(pyproject_src)
    benchmark(tomlrt.dumps, doc)


@pytest.mark.parametrize("depth", [2, 20])
def test_render_nested_payload(benchmark: BenchmarkFixture, depth: int) -> None:
    payload = '"' + "x" * 16_384 + '"'
    src = "value = " + "[" * depth + payload + "]" * depth + "\n"
    benchmark(tomlrt.dumps, tomlrt.loads(src))


def test_render_wide_inline_table(benchmark: BenchmarkFixture) -> None:
    src = "value = {" + ", ".join(f"v{i} = {i}" for i in range(4000)) + "}\n"
    benchmark(tomlrt.dumps, tomlrt.loads(src))
