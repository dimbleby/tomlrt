"""Shared fixtures for the benchmark suite.

The benchmarks live outside ``testpaths``, so a plain ``pytest`` run does
not collect them; ``make bench`` runs them explicitly.
"""

from __future__ import annotations

import math
import textwrap
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

    Both plain runs and comparisons use the width-bounded summary below.
    The plugin still loads and saves results and checks regression limits.
    """
    config.option.benchmark_disable_gc = True
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


def _write_wrapped(reporter: Any, text: str, *, indent: int = 0) -> None:
    """Keep labels and complete case names inside the terminal width."""
    width = max(1, reporter._tw.fullwidth)  # noqa: SLF001
    lines = textwrap.wrap(
        text, width=width, subsequent_indent=" " * min(indent, width // 2)
    )
    for line in lines or [""]:
        reporter.write_line(line)


def _write_comparison(
    reporter: Any, benchmarks: list[Any], baseline: dict[str, Any]
) -> None:
    """Show median changes, largest slowdowns first, without hiding new cases."""
    rows: list[tuple[Any, float | None, float | None]] = []
    for bench in benchmarks:
        previous = baseline.get(bench.fullname)
        before = float(previous["stats"]["median"]) if previous is not None else None
        change = (
            100 * (bench.stats.median / before - 1)
            if before is not None and before > 0
            else None
        )
        rows.append((bench, before, change))
    rows.sort(
        key=lambda row: (
            row[2] is None,
            -row[2] if row[2] is not None else 0,
            str(row[0].name),
        )
    )
    _write_wrapped(reporter, f"{'base':>10}  {'now':>10}  {'change':>8}  case")
    for bench, before, change in rows:
        old = _format_time(before) if before is not None else "-"
        delta = (
            f"{change:+.1f}%"
            if change is not None
            else ("new" if before is None else "n/a")
        )
        prefix = f"{old:>10}  {_format_time(bench.stats.median):>10}  {delta:>8}  "
        _write_wrapped(reporter, prefix + bench.name, indent=len(prefix))
    removed = len(baseline.keys() - {bench.fullname for bench in benchmarks})
    if removed:
        _write_wrapped(
            reporter, f"{removed} baseline cases not run (filtered or removed)."
        )


def pytest_terminal_summary(terminalreporter: Any) -> None:
    """Print compact plain results or one comparison table per saved baseline."""
    config = terminalreporter.config
    # Set unconditionally by the plugin's own pytest_configure, so its
    # absence is a broken install and should raise rather than be
    # swallowed. An empty list is the real case: -k matched nothing.
    session = config._benchmarksession  # noqa: SLF001
    benchmarks = [bench for bench in session.benchmarks if bench]
    if not benchmarks:
        return

    if config.option.benchmark_compare and session.compared_mapping:
        for path, baseline in session.compared_mapping.items():
            _write_wrapped(terminalreporter, "")
            _write_wrapped(terminalreporter, f"Baseline: {path}")
            _write_comparison(terminalreporter, benchmarks, baseline)
        _write_wrapped(terminalreporter, "")
        _write_wrapped(
            terminalreporter,
            "Positive change is slower; negative is faster. "
            "Small changes may be noise.",
        )
        return

    def write(text: str) -> None:
        _write_wrapped(terminalreporter, text)

    write("")
    write(f"{'median':>10}  {'+/-':>6}  case")
    for bench in sorted(benchmarks, key=lambda b: str(b.name)):
        stats = bench.stats
        median, noise = _format_time(stats.median), _format_noise(stats)
        prefix = f"{median:>10}  {noise:>6}  "
        _write_wrapped(terminalreporter, prefix + bench.name, indent=len(prefix))
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
