from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "benchmark_reporting",
    Path(__file__).resolve().parents[1] / "benchmarks/conftest.py",
)
assert _SPEC is not None
assert _SPEC.loader is not None
_REPORTING = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_REPORTING)


def _benchmark(name: str, median: float) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        fullname=f"benchmarks/test_case.py::{name}",
        stats=SimpleNamespace(median=median, iqr=median / 10, rounds=20),
    )


def _render(
    benches: list[SimpleNamespace],
    baselines: dict[str, Any],
    width: int,
    *,
    compare: bool = True,
) -> list[str]:
    lines: list[str] = []
    reporter = SimpleNamespace(
        _tw=SimpleNamespace(fullwidth=width),
        write_line=lines.append,
        config=SimpleNamespace(
            option=SimpleNamespace(benchmark_compare=compare),
            _benchmarksession=SimpleNamespace(
                benchmarks=benches, compared_mapping=baselines
            ),
        ),
    )
    _REPORTING.pytest_terminal_summary(reporter)
    return lines


@pytest.mark.parametrize("width", [40, 60, 80, 120])
def test_comparison_wraps_names_and_baselines_without_losing_information(
    width: int,
) -> None:
    name = "test_delete_inline_array_slice[2-multiline-16000]"
    bench = _benchmark(name, 0.012)
    path = "Linux-CPython-3.14/0001_" + "a" * 40 + "_20260913_210000.json"
    lines = _render(
        [bench], {path: {bench.fullname: {"stats": {"median": 0.01}}}}, width
    )
    assert all(len(line) <= width for line in lines)
    compact = "".join("".join(lines).split())
    assert path in compact
    assert name in compact
    assert "+20.0%" in compact


def test_comparison_orders_slowdowns_and_identifies_unmatched_cases() -> None:
    fast, slow, new, zero = (
        _benchmark("fast", 0.005),
        _benchmark("slow", 0.02),
        _benchmark("new", 0.01),
        _benchmark("zero", 0.01),
    )
    baseline = {bench.fullname: {"stats": {"median": 0.01}} for bench in (fast, slow)}
    baseline[zero.fullname] = {"stats": {"median": 0.0}}
    baseline["not_selected"] = {"stats": {"median": 0.01}}
    assert _render([fast, new, slow, zero], {"0001": baseline}, 80) == [
        "",
        "Baseline: 0001",
        "      base         now    change  case",
        "   10.0 ms     20.0 ms   +100.0%  slow",
        "   10.0 ms      5.0 ms    -50.0%  fast",
        "         -     10.0 ms       new  new",
        "    0.0 us     10.0 ms       n/a  zero",
        "1 baseline cases not run (filtered or removed).",
        "",
        "Positive change is slower; negative is faster. Small changes may be noise.",
    ]


def test_multiple_baselines_have_separate_tables() -> None:
    bench = _benchmark("case", 0.01)
    lines = _render(
        [bench],
        {
            "0001": {bench.fullname: {"stats": {"median": 0.02}}},
            "0002": {bench.fullname: {"stats": {"median": 0.005}}},
        },
        80,
    )
    assert lines.count("      base         now    change  case") == 2
    assert "   20.0 ms     10.0 ms    -50.0%  case" in lines
    assert "    5.0 ms     10.0 ms   +100.0%  case" in lines


@pytest.mark.parametrize("compare", [False, True])
def test_no_baseline_keeps_plain_summary(*, compare: bool) -> None:
    lines = _render([_benchmark("case", 0.01)], {}, 80, compare=compare)
    assert "    median     +/-  case" in lines
    assert "   10.0 ms    4.2%  case" in lines
    assert _render([], {}, 80, compare=compare) == []
