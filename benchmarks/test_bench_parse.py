"""Parse-throughput benchmarks for `tomlrt.loads`.

Times `tomlrt.loads` over (a) the vendored `toml-test/valid` corpus
treated as one combined input pool and (b) three synthetic stress inputs
covering deep nesting, a large array-of-tables and a wide inline table.

Usage:

    make bench
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import tomlrt

if TYPE_CHECKING:
    from collections.abc import Callable

    from pytest_benchmark.fixture import BenchmarkFixture

    Throughput = Callable[[BenchmarkFixture, int], None]

CORPUS_DIR = (
    Path(__file__).resolve().parent.parent / "vendor" / "toml-test" / "tests" / "valid"
)


@pytest.fixture(scope="session")
def corpus() -> list[str]:
    """Every UTF-8 `.toml` file under the vendored toml-test corpus.

    toml-test ships a couple of intentionally non-UTF-8 fixtures; they
    are excluded because `tomlrt.loads` operates on `str`.
    """

    def readable(path: Path) -> str | None:
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return None

    texts = [
        text
        for text in (readable(p) for p in sorted(CORPUS_DIR.rglob("*.toml")))
        if text is not None
    ]
    if not texts:
        pytest.skip("toml-test corpus is not vendored")
    return texts


@pytest.fixture(scope="session")
def deep_array_src() -> str:
    """Stress nested-array parsing: ``x = [[[…1…]]]`` to 80 brackets."""
    return "x = " + ("[" * 80) + "1" + ("]" * 80) + "\n"


@pytest.fixture(scope="session")
def big_aot_src() -> str:
    """Stress array-of-tables parsing: 2000 ``[[items]]`` entries."""
    lines: list[str] = []
    for i in range(2_000):
        lines.extend(
            (
                "[[items]]\n",
                f'name = "item-{i}"\n',
                f"value = {i}\n",
                f"flag = {'true' if i % 2 == 0 else 'false'}\n",
                f'tags = ["a", "b", "c-{i % 17}"]\n',
                "\n",
            )
        )
    return "".join(lines)


@pytest.fixture(scope="session")
def big_inline_src() -> str:
    """Stress inline-table parsing: one ``{ k0=…, k1=…, … }`` with 1000 entries."""
    parts = [f'k{i} = "value-{i}"' for i in range(1_000)]
    return "config = { " + ", ".join(parts) + " }\n"


def test_corpus(
    benchmark: BenchmarkFixture,
    corpus: list[str],
    record_throughput: Throughput,
) -> None:
    """One round parses every file in the corpus."""

    def parse_all() -> None:
        for text in corpus:
            tomlrt.loads(text)

    benchmark(parse_all)
    record_throughput(benchmark, sum(len(t.encode()) for t in corpus))


def test_deep_array(
    benchmark: BenchmarkFixture, deep_array_src: str, record_throughput: Throughput
) -> None:
    benchmark(tomlrt.loads, deep_array_src)
    record_throughput(benchmark, len(deep_array_src.encode()))


def test_big_aot(
    benchmark: BenchmarkFixture, big_aot_src: str, record_throughput: Throughput
) -> None:
    benchmark(tomlrt.loads, big_aot_src)
    record_throughput(benchmark, len(big_aot_src.encode()))


def test_big_inline_table(
    benchmark: BenchmarkFixture, big_inline_src: str, record_throughput: Throughput
) -> None:
    benchmark(tomlrt.loads, big_inline_src)
    record_throughput(benchmark, len(big_inline_src.encode()))
