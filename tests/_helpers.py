"""Shared test helpers.

* :func:`td` — write TOML fixtures as indented triple-quoted literals
  so tests don't degenerate into walls of ``\\n``-escaped strings.
* :func:`reparses` — re-parse a rendered document with stdlib ``tomllib``
  to sanity-check that the output is still valid TOML carrying the
  expected logical values.
"""

from __future__ import annotations

import sys
from textwrap import dedent
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover -- backport for Python < 3.11
    import tomli as tomllib


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
    """Parse ``src`` with stdlib ``tomllib`` and return the result."""
    return tomllib.loads(src)
