"""Exception types raised by tomlrt."""

from __future__ import annotations


class TOMLError(Exception):
    """Base class for all tomlrt errors."""


class TOMLParseError(TOMLError, ValueError):
    """Raised when a TOML document cannot be parsed.

    The human-readable problem description is available via
    ``str(exc)`` (or ``exc.args[0]``) and has the form
    ``"{message} (line L, column C)"``.

    Attributes:
        line: 1-based line number where the error was detected.
        col:  1-based column number where the error was detected.
        offset: 0-based character offset into the source string.
    """

    __slots__ = ("col", "line", "offset")

    def __init__(self, message: str, *, line: int, col: int, offset: int) -> None:
        super().__init__(f"{message} (line {line}, column {col})")
        self.line: int = line
        self.col: int = col
        self.offset: int = offset
