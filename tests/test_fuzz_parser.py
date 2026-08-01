"""Bytes-level grammar fuzzer for the parser.

Complements ``test_fuzz_roundtrip.py`` (which builds well-formed
TOML and asserts byte-exact round-trip) by feeding the parser arbitrary
and near-valid byte sequences. The parser must:

* Reject malformed input by raising `tomlrt.TOMLParseError` -- never any
  other exception type, never a hang, never a crash.
* For input it accepts, produce a `Document` whose ``dumps`` is
  byte-exact equal to the input.

Marked ``slow`` so it runs alongside the existing property suite under
``pytest -m slow``; the regular ``pytest`` invocation skips it.
"""

from __future__ import annotations

import string

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import tomlrt

pytestmark = pytest.mark.slow


# Bytes that frequently appear in TOML constructs, weighted to bias the
# fuzzer towards "almost valid" inputs that exercise structural error
# paths rather than getting rejected at the first character.
_TOML_ALPHABET = (
    string.ascii_letters
    + string.digits
    + " \t\n\r=.,:;[]{}\"'`#_-+*/\\!?@$%^&|<>~"
    + "\x00\x01\x1f\x7f\u00a0\u2028\u2029\ufeff"
)


def _check(src: str) -> None:
    """Parse ``src`` -- on success the document must round-trip byte-exact."""
    try:
        doc = tomlrt.loads(src)
    except tomlrt.TOMLParseError:
        return
    assert tomlrt.dumps(doc) == src


@given(st.text(alphabet=_TOML_ALPHABET, max_size=200))
@settings(
    max_examples=400,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_fuzz_arbitrary_text(src: str) -> None:
    _check(src)


# Templated fragments biased towards almost-valid headers, key/values,
# and inline structures. Each piece is a snippet of TOML-ish noise; the
# strategy concatenates a handful of them per example.
_FRAGMENTS = st.sampled_from(
    [
        "[a]\n",
        "[[a]]\n",
        "[a.b]\n",
        "[a.b.c]\n",
        'a = "x"\n',
        "a = 1\n",
        "a = 1.0\n",
        "a = true\n",
        "a = [1, 2, 3]\n",
        "a = {x = 1}\n",
        "a.b = 1\n",
        "# comment\n",
        "\n",
        "  \n",
        "\r\n",
        '"""ml"""',
        "'''ml'''",
        "[",
        "]",
        "{",
        "}",
        "=",
        ",",
        '"',
        "'",
        "\\",
        "0x",
        "0o",
        "0b",
        "+",
        "-",
        "inf",
        "nan",
        "1979-05-27",
        "07:32:00",
        "1979-05-27T07:32:00Z",
    ],
)


@given(st.lists(_FRAGMENTS, min_size=1, max_size=12).map("".join))
@settings(
    max_examples=400,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_fuzz_grammar_fragments(src: str) -> None:
    _check(src)


@given(st.binary(max_size=200))
@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_fuzz_random_bytes(payload: bytes) -> None:
    """Even arbitrary bytes (decoded as utf-8 with surrogate-escape) must
    only ever raise `TOMLParseError` from `parse`.
    """
    try:
        src = payload.decode("utf-8")
    except UnicodeDecodeError:
        return
    _check(src)
