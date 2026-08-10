"""Shared fixtures for the benchmark suite.

The benchmarks live outside ``testpaths``, so a plain ``pytest`` run does
not collect them; ``make bench`` runs them explicitly.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

    from pytest_benchmark.fixture import BenchmarkFixture

REPO_ROOT = Path(__file__).resolve().parent.parent


def pytest_configure(config: pytest.Config) -> None:
    """Settings the suite is wrong without.

    A collection landing inside a timed round dominates every other
    source of noise here: it costs ``test_big_aot`` a 96ms IQR against an
    83ms median, against 6ms with the collector off.

    Reporting splits by mode. Against a saved baseline the plugin is at
    its best: grouping by name puts the baseline and this run in one
    table, and its ratio column is then exactly the regression. With
    nothing to compare against, the plain run prints its own table --
    see `pytest_terminal_summary`.
    """
    config.option.benchmark_disable_gc = True
    if config.option.benchmark_compare:
        config.option.benchmark_group_by = "name"
        config.option.benchmark_columns = ["median", "iqr", "rounds"]
    else:
        config.option.benchmark_quiet = True


def _format_time(seconds: float) -> str:
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f} us"
    if seconds < 1:
        return f"{seconds * 1e3:.1f} ms"
    return f"{seconds:.2f} s"


def _format_noise(stats: Any) -> str:
    """Twice the standard error of the median, relative to it.

    ``0.929 = sqrt(pi/2) / 1.349`` converts an IQR to the median's
    standard error: 1.349 is the IQR of a standard normal, so IQR/1.349
    estimates sigma, and the median's error is sqrt(pi/2) * sigma /
    sqrt(n) rather than the mean's sigma / sqrt(n). Both constants are
    normal-theory, but the input is the IQR, so a stray outlier perturbs
    a scale factor instead of wrecking the estimate.

    This is sampling error within the run, and so a floor rather than a
    threshold.
    """
    error = 0.929 * stats.iqr / math.sqrt(stats.rounds)
    return f"{2 * error / stats.median * 100:.1f}%"


def pytest_terminal_summary(terminalreporter: Any) -> None:
    """Print each case's median and its within-run sampling noise.

    The plugin's own table is not wrong, it answers a different question:
    it reports the IQR and the round count without saying what they imply
    between them, and spends ten columns doing it. These two fit every
    case on one screen, and say whether a case measures itself precisely
    enough to be worth reading at all.
    """
    config = terminalreporter.config
    # Set unconditionally by the plugin's own pytest_configure, so its
    # absence is a broken install and should raise rather than be
    # swallowed. An empty list is the real case: -k matched nothing.
    session = config._benchmarksession  # noqa: SLF001
    if not session.benchmarks or config.option.benchmark_compare:
        return

    write = terminalreporter.write_line
    write("")
    write(f"{'median':>10}  {'+/-':>6}  case")
    for bench in sorted(session.benchmarks, key=lambda b: str(b.name)):
        stats = bench.stats
        median, noise = _format_time(stats.median), _format_noise(stats)
        write(f"{median:>10}  {noise:>6}  {bench.name}")
    write("")
    write("+/- is sampling noise within this run; compare against a saved")
    write("baseline with --benchmark-autosave/--benchmark-compare.")


@pytest.fixture(scope="session")
def pyproject_src() -> str:
    """The repository's own ``pyproject.toml``."""
    return (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")


@pytest.fixture
def record_throughput() -> Callable[[BenchmarkFixture, int], None]:
    """Record a byte count and the MiB/s it implies in the JSON report.

    Not a column: bytes per case are fixed, so MiB/s only rescales the
    median. It is here for absolute comparison against other parsers.
    """

    def record(benchmark: BenchmarkFixture, nbytes: int) -> None:
        stats = benchmark.stats
        assert stats is not None, "record_throughput must run after the benchmark"
        benchmark.extra_info["bytes"] = nbytes
        benchmark.extra_info["mib_per_s"] = nbytes / stats.stats.median / 1024**2

    return record
