"""Scoped formatting over fresh layouts, including scattered ownership."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import tomlrt

if TYPE_CHECKING:
    from collections.abc import Callable

    from pytest_benchmark.fixture import BenchmarkFixture

    from tomlrt import Table


def _prepared(
    source: str,
) -> Callable[[], tuple[tuple[Table], dict[str, object]]]:
    def setup() -> tuple[tuple[Table], dict[str, object]]:
        return (tomlrt.loads(source).table("a"),), {}

    return setup


@pytest.mark.parametrize("depth", [1, 12, 24])
def test_format_deep_implicit(benchmark: BenchmarkFixture, depth: int) -> None:
    path = ".".join(["a"] * depth)
    values = ",".join(str(i) for i in range(80))
    source = f"{path}.values = [ {values} ]\n"
    benchmark.pedantic(tomlrt.Table.format, setup=_prepared(source), rounds=100)


def test_format_interleaved_sections(benchmark: BenchmarkFixture) -> None:
    source = "".join(
        f"[a.part{i}]\nvalue=1\n[foreign{i}]\nvalue=2\n" for i in range(100)
    )
    benchmark.pedantic(tomlrt.Table.format, setup=_prepared(source), rounds=100)


def test_format_headered_scalar_body(benchmark: BenchmarkFixture) -> None:
    source = "[a]\n" + "".join(f"k{i}  ={i}\n" for i in range(2000))
    benchmark.pedantic(tomlrt.Table.format, setup=_prepared(source), rounds=100)


def test_format_headered_child_sections(benchmark: BenchmarkFixture) -> None:
    source = "[a]\n" + "".join(f"[a.part{i}]\nvalue=1\n" for i in range(250))
    benchmark.pedantic(tomlrt.Table.format, setup=_prepared(source), rounds=100)


def test_format_headered_dotted_array(benchmark: BenchmarkFixture) -> None:
    path = ".".join(["nested"] * 24)
    values = ",".join(str(i) for i in range(80))
    source = f"[a]\n{path}.values = [ {values} ]\n"
    benchmark.pedantic(tomlrt.Table.format, setup=_prepared(source), rounds=100)


def test_format_headered_dotted_keys(benchmark: BenchmarkFixture) -> None:
    path = ".".join(["nested"] * 24)
    source = "[a]\n" + "".join(f"{path}.k{i}  ={i}\n" for i in range(200))
    benchmark.pedantic(tomlrt.Table.format, setup=_prepared(source), rounds=100)
