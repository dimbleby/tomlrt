"""Splice ordered runs without repeatedly shifting the same list tail."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Sequence

_T = TypeVar("_T")


def index_runs(indices: Sequence[int]) -> list[tuple[int, int]]:
    """Group non-empty, sorted, distinct indices into half-open runs."""
    runs: list[tuple[int, int]] = []
    start = prev = indices[0]
    for index in indices[1:]:
        if index != prev + 1:
            runs.append((start, prev + 1))
            start = index
        prev = index
    runs.append((start, prev + 1))
    return runs


def delete_runs(items: list[_T], runs: Sequence[tuple[int, int]]) -> None:
    """Remove sorted disjoint runs directly from the underlying list storage."""
    start, stop = runs[0]
    retained: list[_T] = []
    for next_start, next_stop in runs[1:]:
        retained.extend(items[stop:next_start])
        stop = next_stop
    list.__setitem__(items, slice(start, stop), retained)
