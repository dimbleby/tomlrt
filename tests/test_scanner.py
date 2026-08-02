"""Tests for the cursor primitives on `tomlrt._scanner._Scanner`.

The higher-level scan_* methods (strings, keys, numbers, trivia)
are exercised indirectly through the parser-level round-trip tests.
This file pins only the cursor + diagnostics contract that those
methods build on.
"""

from __future__ import annotations

import pytest

from tomlrt._errors import TOMLParseError
from tomlrt._scanner import _Scanner


def test_initial_state() -> None:
    s = _Scanner("abc")
    assert s.pos == 0
    assert s.end == 3
    assert s.src == "abc"
    assert not s.eof()


def test_peek_within_and_past_end() -> None:
    s = _Scanner("ab")
    assert s.peek() == "a"
    assert s.peek(1) == "b"
    assert s.peek(2) == ""
    assert s.peek(99) == ""


def test_starts_with() -> None:
    s = _Scanner("hello world")
    assert s.starts_with("hello")
    assert not s.starts_with("world")
    s.pos = 6
    assert s.starts_with("world")


def test_pos_supports_backtracking() -> None:
    s = _Scanner("abcdef")
    s.pos += 4
    save = s.pos
    s.pos += 1
    assert s.peek() == "f"
    s.pos = save
    assert s.peek() == "e"


def test_eof_transitions() -> None:
    s = _Scanner("xy")
    assert not s.eof()
    s.pos += 2
    assert s.eof()
    assert s.peek() == ""


def test_eof_on_empty_source() -> None:
    s = _Scanner("")
    assert s.eof()
    assert s.peek() == ""
    assert not s.starts_with("anything")
    assert s.starts_with("")


def test_line_col_first_line() -> None:
    s = _Scanner("hello")
    assert s.line_col(0) == (1, 1)
    assert s.line_col(3) == (1, 4)


def test_line_col_after_newlines() -> None:
    src = "a\nb\nccc"
    s = _Scanner(src)
    # offsets: a=0, \n=1, b=2, \n=3, c=4, c=5, c=6
    assert s.line_col(0) == (1, 1)
    assert s.line_col(2) == (2, 1)
    assert s.line_col(3) == (2, 2)
    assert s.line_col(4) == (3, 1)
    assert s.line_col(6) == (3, 3)


def test_line_col_handles_crlf() -> None:
    # Both halves of a CRLF count as still-on-the-old-line until
    # *after* the LF, mirroring the existing parser's _line_col.
    src = "a\r\nb"
    s = _Scanner(src)
    assert s.line_col(0) == (1, 1)
    assert s.line_col(1) == (1, 2)  # the CR
    assert s.line_col(2) == (1, 3)  # the LF (not yet a new line)
    assert s.line_col(3) == (2, 1)


def test_error_uses_cursor_by_default() -> None:
    s = _Scanner("abc\ndef")
    s.pos += 5  # at 'e'
    err = s.error("nope")
    assert isinstance(err, TOMLParseError)
    assert err.offset == 5
    assert err.line == 2
    assert err.col == 2
    assert "line 2" in str(err)
    assert "column 2" in str(err)


def test_error_accepts_explicit_offset() -> None:
    s = _Scanner("abc\ndef")
    s.pos += 6
    err = s.error("rewound", at=2)
    assert err.offset == 2
    assert err.line == 1
    assert err.col == 3


def test_error_can_be_raised() -> None:
    s = _Scanner("xyz")
    err = s.error("boom")
    with pytest.raises(TOMLParseError) as info:
        raise err
    assert info.value.line == 1
    assert info.value.col == 1
