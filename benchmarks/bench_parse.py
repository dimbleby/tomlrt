#!/usr/bin/env python3

"""Parse-throughput benchmark for `tomlrt.loads`.

Times `tomlrt.loads` over (a) the vendored `toml-test/valid` corpus
treated as one combined input pool and (b) two synthetic stress
inputs covering deep nesting and a large array-of-tables. Numbers
are wall-clock per-byte and per-document; both are reported.

Usage:

    uv run python benchmarks/bench_parse.py
"""

from __future__ import annotations

import gc
import statistics
import sys
import time
from pathlib import Path

import tomlrt

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = REPO_ROOT / "vendor" / "toml-test" / "tests" / "valid"


def _load_corpus() -> list[tuple[str, str]]:
    """Read every UTF-8 `.toml` file under the vendored toml-test corpus."""
    files: list[tuple[str, str]] = []
    for path in sorted(CORPUS_DIR.rglob("*.toml")):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # toml-test ships a couple of intentionally non-UTF-8
            # fixtures; we exclude them — `tomlrt.loads` operates on
            # `str` so they are out of scope for this microbench.
            continue
        files.append((str(path.relative_to(CORPUS_DIR)), text))
    return files


def _synth_deep_array(depth: int) -> str:
    """Stress nested-array parsing: `x = [[[…1…]]]` to `depth` brackets."""
    return "x = " + ("[" * depth) + "1" + ("]" * depth) + "\n"


def _synth_big_aot(rows: int) -> str:
    """Stress array-of-tables parsing: `rows` `[[items]]` entries."""
    lines: list[str] = []
    for i in range(rows):
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


def _synth_big_inline_table(entries: int) -> str:
    """Stress inline-table parsing: one `{ k0=…, k1=…, … }` with N entries."""
    parts = [f'k{i} = "value-{i}"' for i in range(entries)]
    return "config = { " + ", ".join(parts) + " }\n"


def _time_one(text: str, *, repeats: int) -> tuple[float, float]:
    """Return (best_seconds, median_seconds) over `repeats` runs."""
    timings: list[float] = []
    for _ in range(repeats):
        gc.collect()
        gc.disable()
        try:
            t0 = time.perf_counter()
            tomlrt.loads(text)
            t1 = time.perf_counter()
        finally:
            gc.enable()
        timings.append(t1 - t0)
    return min(timings), statistics.median(timings)


def _bench_corpus(corpus: list[tuple[str, str]], *, repeats: int) -> None:
    """Time the whole corpus as a batch (one repeat = parse every file)."""
    total_bytes = sum(len(t.encode("utf-8")) for _, t in corpus)
    timings: list[float] = []
    for _ in range(repeats):
        gc.collect()
        gc.disable()
        try:
            t0 = time.perf_counter()
            for _, text in corpus:
                tomlrt.loads(text)
            t1 = time.perf_counter()
        finally:
            gc.enable()
        timings.append(t1 - t0)
    best = min(timings)
    median = statistics.median(timings)
    mb_per_s = (total_bytes / best) / (1024 * 1024)
    print(
        f"  toml-test corpus      "
        f"{len(corpus):4d} files / "
        f"{total_bytes / 1024:7.1f} KiB    "
        f"best {best * 1000:7.2f} ms   "
        f"median {median * 1000:7.2f} ms   "
        f"{mb_per_s:6.2f} MiB/s",
    )


def _bench_one(name: str, text: str, *, repeats: int) -> None:
    """Time `repeats` parses of one input and print best/median/throughput."""
    nbytes = len(text.encode("utf-8"))
    best, median = _time_one(text, repeats=repeats)
    mb_per_s = (nbytes / best) / (1024 * 1024)
    print(
        f"  {name:22s}"
        f"            {nbytes / 1024:7.1f} KiB    "
        f"best {best * 1000:7.2f} ms   "
        f"median {median * 1000:7.2f} ms   "
        f"{mb_per_s:6.2f} MiB/s",
    )


def main() -> None:
    """Run the corpus benchmark followed by the three synthetic stress inputs."""
    corpus = _load_corpus()
    deep = _synth_deep_array(80)
    big_aot = _synth_big_aot(2000)
    big_inline = _synth_big_inline_table(1000)

    print(f"tomlrt           : {tomlrt.__file__}")
    print(f"python           : {sys.version.split()[0]}")
    print()

    _bench_corpus(corpus, repeats=5)
    _bench_one("synth: deep array (80)", deep, repeats=2000)
    _bench_one("synth: big AoT (2000)", big_aot, repeats=20)
    _bench_one("synth: big inline tbl", big_inline, repeats=200)


if __name__ == "__main__":
    main()
