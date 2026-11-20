"""Cursor + scanner used by `tomlrt._parser`.

The scanner owns `(src, end, pos)`, so it is the parser's cursor
authority. It returns CST nodes for source-preserving constructs and
small tuples/strings for tokens the parser still dispatches.

String scanning is semantic: escapes are decoded, surrogates rejected,
the leading newline after an opening triple quote trimmed, and the
multi-line trailing-quote allowance enforced. `scan_string` fills the
raw lexeme while preserving the decoded value.
"""

from __future__ import annotations

import re
from datetime import date, datetime, time
from typing import TYPE_CHECKING, Final

from tomlrt._errors import TOMLParseError
from tomlrt._values import (
    BoolValue,
    DateTimeValue,
    FloatValue,
    IntegerValue,
    StringValue,
)

if TYPE_CHECKING:
    from tomlrt._values import Value

# Comment body: anything except newline + control chars (tab is OK).
_RE_COMMENT_BODY: Final = re.compile(r"[^\r\n\x00-\x08\x0b-\x1f\x7f]*")

# Body fragments stop at newlines; string scanners handle or reject them.
# Basic string: not quote, backslash, newline, or control
# char (U+0000-U+001F / U+007F, except tab).
_RE_BASIC_BODY: Final = re.compile(r'[^"\\\n\r\x00-\x08\x0b-\x1f\x7f]+')
# Body of a literal string: no quote, newline, or control char.
_RE_LITERAL_BODY: Final = re.compile(r"[^'\n\r\x00-\x08\x0b-\x1f\x7f]+")
# Bare key (ASCII letters, digits, underscore, dash — the same set in
# TOML 1.0 and 1.1), plus the inline whitespace that follows it: one
# scan of adjacent source rather than two.
_RE_BARE_KEY: Final = re.compile(r"([A-Za-z0-9_\-]+)([ \t]*)")

# Per-flavour multi-line string body pattern and name for diagnostics.
_ML_FLAVOURS: Final = {
    '"': (_RE_BASIC_BODY, "basic"),
    "'": (_RE_LITERAL_BODY, "literal"),
}

_HEX_DIGITS: Final[frozenset[str]] = frozenset("0123456789abcdefABCDEF")
_DECIMAL: Final = r"(?:0|[1-9][0-9]*(?:_[0-9]+)*)"
_DIGITS: Final = r"[0-9]+(?:_[0-9]+)*"
_EXPONENT: Final = rf"[eE][+-]?{_DIGITS}"
_DATE: Final = r"[0-9]{4}-[0-9]{2}-[0-9]{2}"
_HOUR_MINUTE: Final = r"(?:[01][0-9]|2[0-3]):[0-5][0-9]"
_TIME: Final = rf"{_HOUR_MINUTE}(?::[0-5][0-9](?:\.[0-9]+)?)?"
# fromisoformat normalizes offset minutes >= 60 and, on Python 3.15, hour 24.
_OFFSET: Final = rf"(?:[Zz]|[+-]{_HOUR_MINUTE})"
_RE_TIME_CONTINUATION: Final = re.compile(r" [0-9][\s\S]:")
_VALUE_DELIMITER: Final = r" \t\n\r,\]}#"
_RE_ATOM: Final = re.compile(
    rf"""
    (?:
        [+-]?{_DECIMAL}(?P<float>\.{_DIGITS}(?:{_EXPONENT})?|{_EXPONENT})?
      | 0x[0-9a-fA-F]+(?:_[0-9a-fA-F]+)*
      | 0o[0-7]+(?:_[0-7]+)*
      | 0b[01]+(?:_[01]+)*
      | (?P<boolean>true|false)
      | (?P<infinity>[+-]?inf)
      | (?P<nan>[+-]?nan)
      | (?P<datetime>{_DATE}(?:[Tt ]{_TIME}{_OFFSET}?)?|{_TIME})
    )
    (?=[{_VALUE_DELIMITER}]|\Z)
    """,
    re.VERBOSE,
)


def _is_ascii_digits(s: str) -> bool:
    """Return True iff ``s`` is non-empty and contains only ASCII ``0-9``.

    ``str.isdigit`` and ``int(s)`` accept non-ASCII decimal digits, but
    TOML integer / float literals are restricted to ASCII.
    """
    return bool(s) and s.isascii() and s.isdecimal()


# A bare-value token: everything up to the first character that can end
# one. Matches empty, which the caller reports as a missing value.
_RE_VALUE_TOKEN: Final = re.compile(rf"[^{_VALUE_DELIMITER}]*")

# Shared simple backslash-escape map.
_SIMPLE_ESCAPES: Final[dict[str, str]] = {
    "b": "\b",
    "t": "\t",
    "n": "\n",
    "f": "\f",
    "r": "\r",
    "e": "\x1b",  # TOML 1.1: ESC
    '"': '"',
    "\\": "\\",
}


class _Scanner:
    __slots__ = ("_seen_crlf", "_seen_lf", "_string_newline", "end", "pos", "src")

    def __init__(self, src: str) -> None:
        self.src = src
        self.end = len(src)
        self.pos = 0
        # Structural/trivia newlines determine the document's mutation policy.
        self._seen_lf = False
        self._seen_crlf = False
        self._string_newline: str | None = None

    def detected_newline(self) -> str:
        r"""Return the document-wide newline kind seen during scanning.

        Structural and trivia newlines take precedence over multiline-string
        content. If there are none, string content is the fallback.
        """
        if self._seen_lf:
            return "\n"
        if self._seen_crlf:
            return "\r\n"
        return self._string_newline or "\n"

    def line_col(self, pos: int) -> tuple[int, int]:
        r"""Return the 1-based (line, column) for source offset `pos`.

        Columns count code points, and a `\r` counts as one of them, so
        both halves of a CRLF belong to the line they terminate.
        """
        src = self.src
        # rfind gives -1 when `pos` is on the first line, which is
        # exactly the "column base" that makes offset 0 column 1.
        return src.count("\n", 0, pos) + 1, pos - src.rfind("\n", 0, pos)

    def error(self, message: str, *, at: int | None = None) -> TOMLParseError:
        """Build a `TOMLParseError` pointing at `at` (default: cursor)."""
        offset = self.pos if at is None else at
        line, col = self.line_col(offset)
        return TOMLParseError(message, line=line, col=col, offset=offset)

    def scan_comment(self) -> None:
        """Consume a comment from `#` to (but not including) the newline.

        The cursor must be on `#`. Raises if the comment body contains
        a control character other than tab.
        """
        src = self.src
        m = _RE_COMMENT_BODY.match(src, self.pos + 1)
        assert m is not None  # pattern is unbounded above (*).
        end_pos = m.end()
        if end_pos < self.end:
            ch = src[end_pos]
            if ch not in {"\n", "\r"}:
                self.pos = end_pos
                cp = ord(ch)
                msg = f"invalid control character U+{cp:04X} in comment"
                raise self.error(msg)
        self.pos = end_pos

    def scan_doc_trivia(self) -> str:
        """Consume whitespace, blank lines and comments up to the next token."""
        start = self.pos
        src = self.src
        end = self.end
        pos = self.pos
        while pos < end:
            ch = src[pos]
            if ch in {" ", "\t"}:
                pos += 1
            elif ch == "#":
                self.pos = pos
                self.scan_comment()
                pos = self.pos
            elif ch == "\n":
                pos += 1
                self._seen_lf = True
            elif ch == "\r":
                if pos + 1 >= end or src[pos + 1] != "\n":
                    self.pos = pos
                    msg = "stray carriage return"
                    raise self.error(msg)
                pos += 2
                self._seen_crlf = True
            else:
                break
        self.pos = pos
        return src[start:pos]

    def skip_inline_ws(self) -> int:
        """Consume a run of spaces/tabs; return the position it started at.

        Unlike `scan_array_trivia`, newlines and comments don't count here.
        """
        src = self.src
        end = self.end
        pos = start = self.pos
        while pos < end:
            c = src[pos]
            if c not in {" ", "\t"}:
                break
            pos += 1
        self.pos = pos
        return start

    def scan_inline_ws_text(self) -> str:
        """Consume a run of spaces/tabs; return raw text (or "")."""
        start = self.skip_inline_ws()
        return self.src[start : self.pos]

    def scan_array_trivia(self) -> str:
        """Consume trivia inside an array (or TOML 1.1 inline table).

        Unlike `scan_inline_ws_text`, newlines and comments are permitted.
        """
        start = self.pos
        src = self.src
        end = self.end
        pos = self.pos
        while pos < end:
            ch = src[pos]
            if ch in {" ", "\t"}:
                pos += 1
            elif ch == "\n":
                pos += 1
                self._seen_lf = True
            elif ch == "\r" and pos + 1 < end and src[pos + 1] == "\n":
                pos += 2
                self._seen_crlf = True
            elif ch == "#":
                self.pos = pos
                self.scan_comment()
                pos = self.pos
            else:
                break
        self.pos = pos
        return src[start:pos]

    def scan_eol(self) -> str:
        """Consume optional trailing-ws + comment + newline (or EOF).

        Returns the whole run as one slice of the source, so a line's
        tail costs no object the cycle collector has to account for.

        Raises if a non-newline, non-comment, non-EOF character is
        found after the optional whitespace.
        """
        src = self.src
        end = self.end
        start = self.skip_inline_ws()
        pos = self.pos
        ch = src[pos] if pos < end else ""
        if ch == "#":
            self.scan_comment()
            pos = self.pos
            ch = src[pos] if pos < end else ""
        if ch == "\n":
            self.pos = pos + 1
            self._seen_lf = True
        elif ch == "\r" and pos + 1 < end and src[pos + 1] == "\n":
            self.pos = pos + 2
            self._seen_crlf = True
        elif pos < end:
            msg = f"expected newline or end of file, got {ch!r}"
            raise self.error(msg)
        return src[start : self.pos]

    def scan_string(self, *, allow_multiline: bool = True) -> StringValue:
        """Scan a string starting at the cursor and return its `StringValue`.

        Dispatches on the opening quote character. Key parsers pass
        ``allow_multiline=False`` to reject multi-line strings.

        Precondition: cursor is at `"` or `'`; callers check the
        character first, so this only asserts it.
        """
        src = self.src
        start = self.pos
        ch = src[start]
        assert ch in ('"', "'"), f"scan_string called at {ch!r}"
        if ch == '"':
            ml = allow_multiline and src.startswith('"""', start)
            node = self._scan_ml_string(ch) if ml else self._scan_basic_string()
        else:
            ml = allow_multiline and src.startswith("'''", start)
            node = self._scan_ml_string(ch) if ml else self._scan_literal_string()
        node.lexeme = src[start : self.pos]
        return node

    def _scan_basic_string(self) -> StringValue:
        """Scan a single-line basic string. Precondition: cursor is at `"`."""
        src = self.src
        end = self.end
        # Fast path: simple basic string with no escapes.
        m = _RE_BASIC_BODY.match(src, self.pos + 1)
        body_start = self.pos + 1
        if m is not None:
            body_end = m.end()
            if body_end < end and src[body_end] == '"':
                self.pos = body_end + 1
                return StringValue("", src[body_start:body_end])
            self.pos = body_end
        else:
            self.pos = body_start
        out: list[str] = []
        if m is not None:
            out.append(m.group(0))
        while True:
            if self.pos >= end:
                msg = "unterminated basic string"
                raise self.error(msg)
            ch = src[self.pos]
            if ch == '"':
                self.pos += 1
                return StringValue("", "".join(out))
            if ch == "\\":
                out.append(self._scan_escape())
            elif ch in {"\n", "\r"}:
                msg = "newline in basic string"
                raise self.error(msg)
            else:
                msg = f"invalid control character U+{ord(ch):04X} in string"
                raise self.error(msg)
            m = _RE_BASIC_BODY.match(src, self.pos)
            if m is not None:
                out.append(m.group(0))
                self.pos = m.end()

    def _scan_ml_string(self, quote: str) -> StringValue:
        """Scan a ``\"\"\"`` / ``\'\'\'``-delimited string.

        Precondition: cursor is at the opening delimiter. ``body``
        matches the flavour's run of ordinary characters, so the tail
        below only ever sees the delimiter quote, a line ending, a
        backslash -- which reaches it in a basic string but is ordinary
        body in a literal one -- or an invalid control character.
        """
        body, kind = _ML_FLAVOURS[quote]
        src = self.src
        end = self.end
        delim = quote * 3
        pos = self.pos + 3
        # A newline immediately after the opening delimiter is trimmed.
        if src.startswith("\n", pos):
            pos += 1
            self._string_newline = "\n"
        elif src.startswith("\r\n", pos):
            pos += 2
            self._string_newline = self._string_newline or "\r\n"
        out: list[str] = []
        while True:
            m = body.match(src, pos)
            if m is not None:
                out.append(m.group(0))
                pos = m.end()
            self.pos = pos
            if pos >= end:
                msg = f"unterminated multi-line {kind} string"
                raise self.error(msg)
            if src.startswith(delim, pos):
                # Up to two extra trailing quotes are allowed inside.
                pos += 3
                extras = 0
                while extras < 2 and pos < end and src[pos] == quote:
                    out.append(quote)
                    pos += 1
                    extras += 1
                self.pos = pos
                return StringValue("", "".join(out))
            ch = src[pos]
            if ch == quote:
                out.append(quote)
                pos += 1
            elif ch == "\\":
                pos = self._scan_ml_escape(out, pos)
            elif ch == "\n":
                out.append("\n")
                pos += 1
                self._string_newline = "\n"
            elif ch == "\r":
                if not src.startswith("\r\n", pos):
                    msg = "stray carriage return in string"
                    raise self.error(msg)
                out.append("\r\n")
                pos += 2
                self._string_newline = self._string_newline or "\r\n"
            else:
                msg = f"invalid control character U+{ord(ch):04X} in string"
                raise self.error(msg)

    def _scan_ml_escape(self, out: list[str], pos: int) -> int:
        """Consume one backslash inside a multi-line basic string.

        A backslash that ends a line swallows the trailing whitespace,
        the line break, and the indent of every following blank line;
        anything else is an ordinary escape. Returns the new cursor.
        """
        src = self.src
        end = self.end
        if src[pos + 1 : pos + 2] in ("\n", " ", "\t", "\r"):
            scan = pos + 1
            while scan < end and src[scan] in " \t":
                scan += 1
            if src.startswith("\n", scan) or src.startswith("\r\n", scan):
                while True:
                    if src.startswith("\n", scan):
                        scan += 1
                        self._string_newline = "\n"
                    elif src.startswith("\r\n", scan):
                        scan += 2
                        self._string_newline = self._string_newline or "\r\n"
                    else:
                        return scan
                    while scan < end and src[scan] in " \t":
                        scan += 1
        self.pos = pos
        out.append(self._scan_escape())
        return self.pos

    def _scan_literal_string(self) -> StringValue:
        """Scan a single-line literal string. Precondition: cursor is at `'`."""
        self.pos += 1
        src = self.src
        end = self.end
        start = self.pos
        m = _RE_LITERAL_BODY.match(src, start)
        if m is not None:
            self.pos = m.end()
        if self.pos >= end:
            msg = "unterminated literal string"
            raise self.error(msg)
        ch = src[self.pos]
        if ch == "'":
            value = src[start : self.pos]
            self.pos += 1
            return StringValue("", value)
        if ch in {"\n", "\r"}:
            msg = "newline in literal string"
            raise self.error(msg)
        msg = f"invalid control character U+{ord(ch):04X} in string"
        raise self.error(msg)

    def _scan_escape(self) -> str:
        r"""Scan one ``\\``-escape. Precondition: cursor is at ``\\``."""
        pos = self.pos + 1
        ch = self.src[pos : pos + 1]
        self.pos = pos + 1
        escaped = _SIMPLE_ESCAPES.get(ch)
        if escaped is not None:
            return escaped
        if ch == "x":  # TOML 1.1: 2-digit hex escape (U+0000..U+00FF)
            return self._scan_unicode_escape(2)
        if ch == "u":
            return self._scan_unicode_escape(4)
        if ch == "U":
            return self._scan_unicode_escape(8)
        msg = f"invalid escape sequence: \\{ch}"
        raise self.error(msg, at=self.pos - 2)

    def _scan_unicode_escape(self, n: int) -> str:
        if self.pos + n > self.end:
            msg = f"truncated unicode escape; expected {n} hex digits"
            raise self.error(msg)
        hex_str = self.src[self.pos : self.pos + n]
        for c in hex_str:
            if c not in _HEX_DIGITS:
                msg = f"invalid hex digit {c!r} in unicode escape"
                raise self.error(msg)
        self.pos += n
        cp = int(hex_str, 16)
        if cp > 0x10FFFF or 0xD800 <= cp <= 0xDFFF:
            msg = f"invalid unicode scalar U+{cp:04X}"
            raise self.error(msg)
        return chr(cp)

    def scan_key(
        self,
    ) -> tuple[tuple[str, ...], tuple[str, ...], str, tuple[str, ...]]:
        """Return raw parts, separators, trailing whitespace and decoded path.

        Each part is bare, basic-quoted or literal-quoted; each
        separator is the literal ``ws "." ws`` between two parts. The
        whitespace after the last part is consumed too, and can be used
        directly as ``pre_eq`` / ``inner_post``.
        """
        src = self.src
        end = self.end
        parts: list[str] = []
        seps: list[str] = []
        path: list[str] = []
        while True:
            start = self.pos
            ch = src[start] if start < end else ""
            if ch in {'"', "'"}:
                quoted = self.scan_string(allow_multiline=False)
                raw = quoted.lexeme
                path.append(quoted.value)
                ws = self.scan_inline_ws_text()
            else:
                m = _RE_BARE_KEY.match(src, start)
                if m is None:
                    msg = f"expected key, got {ch!r}"
                    raise self.error(msg)
                raw = m[1]
                ws = m[2]
                path.append(raw)
                self.pos = m.end()
            parts.append(raw)
            pos = self.pos
            if pos >= end or src[pos] != ".":
                key_path = tuple(path)
                key_parts = key_path if parts == path else tuple(parts)
                return key_parts, tuple(seps), ws, key_path
            # The separator starts where the already-consumed whitespace began.
            sep_start = pos - len(ws)
            self.pos = pos + 1
            self.scan_inline_ws_text()
            seps.append(src[sep_start : self.pos])

    # Bare value tokens (bool, special float, number, date/time); the parser
    # dispatches strings, arrays and inline tables itself.

    def scan_value_atom(self) -> Value:
        """Recognize a complete bare value directly in the source."""
        start = self.pos
        match = _RE_ATOM.match(self.src, start)
        if match is not None:
            self.pos = match.end()
            token = match[0]
            kind = match.lastgroup
            if kind is None:
                return self._convert_integer(token, at=start)
            if kind == "boolean":
                return BoolValue(token, token == "true")  # noqa: S105
            if kind == "datetime":
                # The match may have backed off to a date before a
                # malformed space-separated time. Report the whole token.
                if len(token) == 10 and _RE_TIME_CONTINUATION.match(self.src, self.pos):
                    raise self._invalid_atom(token, at=start)
                return self._convert_datetime(token, at=start)
            return FloatValue(token, float("nan" if kind == "nan" else token))

        raise self._invalid_atom(self._scan_value_token(), at=start)

    def _invalid_atom(self, token: str, *, at: int) -> TOMLParseError:
        if not token:
            msg = f"expected value, got {self.src[at : at + 1]!r}"
            return self.error(msg, at=at)

        if self._looks_like_datetime(token):
            category = "time" if token[2] == ":" else "date/datetime"
            if len(token) == 10 and _RE_TIME_CONTINUATION.match(self.src, self.pos):
                self.pos += 1
                token += " " + self._scan_value_token()
        else:
            category = "float" if self._looks_like_float(token) else "integer"
        msg = f"invalid {category} {token!r}"
        return self.error(msg, at=at)

    def _scan_value_token(self) -> str:
        """Consume the bare-value token at the cursor and return its text."""
        m = _RE_VALUE_TOKEN.match(self.src, self.pos)
        assert m is not None  # the pattern matches empty
        self.pos = m.end()
        return m[0]

    @staticmethod
    def _looks_like_datetime(token: str) -> bool:
        # Date uses ``YYYY-MM-DD``; local time uses ``HH:MM:SS``. The
        # grammar requires ASCII digits, so don't use ``str.isdigit``.
        if len(token) >= 5 and token[4] == "-" and _is_ascii_digits(token[:4]):
            return True
        return bool(len(token) >= 3 and token[2] == ":" and _is_ascii_digits(token[:2]))

    @staticmethod
    def _looks_like_float(token: str) -> bool:
        # Decimal floats contain ``.``, ``e`` or ``E``; hex/oct/bin
        # integers never do.
        body = token[1:] if token[0] in "+-" else token
        if body.startswith(("0x", "0o", "0b")):
            return False
        return "." in body or "e" in body or "E" in body

    def _convert_integer(self, token: str, *, at: int) -> IntegerValue:
        try:
            value = int(token, 0)
        except ValueError as exc:
            msg = f"invalid integer {token!r}: {exc}"
            raise self.error(msg, at=at) from exc
        return IntegerValue(token, value)

    def _convert_datetime(self, text: str, *, at: int) -> DateTimeValue:
        local_time = text[2] == ":"
        try:
            value: date | time
            if local_time:
                value = time.fromisoformat(text)
            elif len(text) == 10:
                value = date.fromisoformat(text)
            else:
                value = datetime.fromisoformat(text.replace("t", "T").replace("z", "Z"))
        except ValueError as exc:
            category = "time" if local_time else "date/datetime"
            msg = f"invalid {category} {text!r}: {exc}"
            raise self.error(msg, at=at) from exc
        return DateTimeValue(text, value)


__all__ = ["_Scanner"]
