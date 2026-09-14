"""Scoped formatting over fresh layouts, including scattered ownership."""

from __future__ import annotations

from typing import TYPE_CHECKING

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


def test_format_deep_implicit(benchmark: BenchmarkFixture) -> None:
    path = ".".join(["a"] * 24)
    values = ",".join(str(i) for i in range(80))
    source = f"{path}.values = [ {values} ]\n"
    benchmark.pedantic(tomlrt.Table.format, setup=_prepared(source), rounds=100)


def test_format_headered_dotted_keys(benchmark: BenchmarkFixture) -> None:
    path = ".".join(["nested"] * 24)
    source = "[a]\n" + "".join(f"{path}.k{i}  ={i}\n" for i in range(200))
    benchmark.pedantic(tomlrt.Table.format, setup=_prepared(source), rounds=100)


def test_format_multiline_array_without_comments(benchmark: BenchmarkFixture) -> None:
    """The shape `_canon_multiline_shape` can canonicalise without text passes.

    A comment-free multi-line array has no above-block to harvest and
    nothing for `_finalise_inline_trivia` to retarget, so both passes
    are elided. The comment-bearing arrays in `test_bench_mutate` take
    the full `Boundary` path instead.
    """
    source = "[a]\nvalues = [\n" + "".join(f"  {i},\n" for i in range(2_000)) + "]\n"
    benchmark.pedantic(tomlrt.Table.format, setup=_prepared(source), rounds=50)
