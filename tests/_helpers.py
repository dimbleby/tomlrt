"""Shared test helpers.

* :func:`td` — write TOML fixtures as indented triple-quoted literals
  so tests don't degenerate into walls of ``\\n``-escaped strings.
* :func:`reparses` — re-parse a rendered document with ``tomli`` to
  sanity-check that the output is still valid TOML carrying the
  expected logical values. ``tomli`` is preferred over the stdlib
  ``tomllib`` because as of writing (Python 3.14) ``tomllib`` is TOML
  1.0 only, whereas ``tomli`` 2.4+ accepts TOML 1.1 syntax (multi-line
  inline tables, etc.).
* :func:`deep_equal` — structural equality (dict/list-recursive,
  NaN-as-equal-to-itself, type-exact at the leaves) used by the
  property/fuzz suites to compare a decoded `Document` against a
  `tomli`-parsed oracle or a Python-dict shadow model.
* :func:`fuzz_context` — attach the seed to whatever a fuzz program
  raises, so a CI failure is reproducible.
* :func:`fuzz_seeds` — the fuzzers' seed source, honouring
  ``TOMLRT_FUZZ_SEED`` so a reported seed can be replayed.
"""

from __future__ import annotations

import math
import os
import secrets
from contextlib import contextmanager
from textwrap import dedent
from typing import TYPE_CHECKING, Any

import tomli

if TYPE_CHECKING:
    from collections.abc import Iterator


def td(src: str) -> str:
    """Dedent ``src`` and strip a single leading newline.

    Lets tests embed TOML fixtures as indented triple-quoted literals::

        src = td('''
            [a]
            x = 1
            [a.sub]
            y = 2
        ''')

    A single leading newline (the one immediately after the opening
    ``\"\"\"``) is stripped so the first content line starts at column 0,
    then :func:`textwrap.dedent` removes the common leading whitespace.
    The result is byte-identical to ``"[a]\\nx = 1\\n[a.sub]\\ny = 2\\n"``,
    which matters because everything in this project is round-tripped
    byte-for-byte.
    """
    return dedent(src).removeprefix("\n")


def reparses(src: str) -> dict[str, Any]:
    """Parse ``src`` with ``tomli`` and return the result."""
    return tomli.loads(src)


def deep_equal(a: object, b: object) -> bool:
    """Structural equality treating NaN as equal to itself.

    Recurses through ``dict`` and ``list`` values; leaves compare with
    ``==`` gated on exact type match (so e.g. ``1`` and ``True`` are
    not conflated). Used across the property/fuzz suites to compare a
    decoded `Document` against a `tomli`-parsed oracle or a plain
    Python-dict shadow model.
    """
    if isinstance(a, float) and isinstance(b, float):
        return (math.isnan(a) and math.isnan(b)) or a == b
    if isinstance(a, dict) and isinstance(b, dict):
        am: dict[Any, Any] = a
        bm: dict[Any, Any] = b
        return am.keys() == bm.keys() and all(
            deep_equal(v, bm[k]) for k, v in am.items()
        )
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(
            deep_equal(x, y) for x, y in zip(a, b, strict=True)
        )
    return type(a) is type(b) and a == b


@contextmanager
def fuzz_context(ctx: str) -> Iterator[None]:
    """Re-raise whatever the block raises with ``ctx`` prefixed to it.

    A seed-driven fuzzer only earns its keep if a failure says which
    seed reproduces it. The oracle assertions could carry that in their
    own messages, but the usual way a fuzzer finds a bug is a raise from
    inside the library itself, which would otherwise report a traceback
    with the seed nowhere in it. Wrapping the whole program covers both.
    """
    try:
        yield
    except Exception as exc:
        msg = f"{ctx}: {type(exc).__name__}: {exc}"
        raise AssertionError(msg) from exc


def fuzz_seeds(count: int) -> Iterator[int]:
    """Yield ``count`` fresh 64-bit seeds, or the one pinned by the environment.

    Fresh seeds every run keep the fuzzers exploring new programs rather
    than re-checking a frozen grid. Setting ``TOMLRT_FUZZ_SEED=<n>``
    replays exactly one program instead, which is how a seed reported by
    :func:`fuzz_context` is reproduced::

        TOMLRT_FUZZ_SEED=<n> uv run pytest -m slow <the failing test id>
    """
    override = os.environ.get("TOMLRT_FUZZ_SEED")
    if override is not None:
        yield int(override)
        return
    for _ in range(count):
        yield secrets.randbits(64)
