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
from datetime import date, datetime, time, timedelta, timezone
from typing import TYPE_CHECKING, Final

from tomlrt._errors import TOMLParseError
from tomlrt._trivia import EolTrivia
from tomlrt._values import (
    BoolValue,
    DateTimeValue,
    FloatValue,
    IntegerValue,
    KeyPart,
    StringValue,
)

if TYPE_CHECKING:
    from tomlrt._values import Value

# Comment body: anything except newline + control chars (tab is OK).
_RE_COMMENT_BODY: Final = re.compile(r"[^\r\n\x00-\x08\x0b-\x1f\x7f]*")

# Body of a basic string: not quote, backslash, newline, or control
# char (U+0000-U+001F / U+007F, except tab).
_RE_BASIC_STR_BODY: Final = re.compile(r'[^"\\\n\r\x00-\x08\x0b-\x1f\x7f]+')
# Body of a literal string: no quote, newline, or control char.
_RE_LITERAL_STR_BODY: Final = re.compile(r"[^'\n\r\x00-\x08\x0b-\x1f\x7f]+")
# Body of a multi-line basic string fragment. Newlines are valid there,
# so the caller handles them; stop at \r to verify CRLF before emitting it.
_RE_ML_BASIC_BODY: Final = re.compile(r'[^"\\\r\n\x00-\x08\x0b-\x1f\x7f]+')
_RE_ML_LITERAL_BODY: Final = re.compile(r"[^'\r\n\x00-\x08\x0b-\x1f\x7f]+")
# Bare key: ASCII alphanum + underscore + dash. (TOML 1.1 broadens
# this; if/when tomlrt opts in, widen the pattern here.)
_RE_BARE_KEY: Final = re.compile(r"[A-Za-z0-9_\-]+")

# Per-flavour multi-line string body pattern and name for diagnostics.
_ML_FLAVOURS: Final = {
    '"': (_RE_ML_BASIC_BODY, "basic"),
    "'": (_RE_ML_LITERAL_BODY, "literal"),
}

_HEX_DIGITS: Final[frozenset[str]] = frozenset("0123456789abcdefABCDEF")
_OCT_DIGITS: Final[frozenset[str]] = frozenset("01234567")
_BIN_DIGITS: Final[frozenset[str]] = frozenset("01")
_DEC_DIGITS: Final[frozenset[str]] = frozenset("0123456789")


def _is_ascii_digits(s: str) -> bool:
    """Return True iff ``s`` is non-empty and contains only ASCII ``0-9``.

    ``str.isdigit`` and ``int(s)`` accept non-ASCII decimal digits, but
    TOML integer / float literals are restricted to ASCII.
    """
    return bool(s) and s.isascii() and s.isdecimal()


# First character that ends a bare-value token.
_RE_VALUE_END: Final = re.compile(r"[ \t\n\r,\]}#]")

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
    __slots__ = ("_seen_crlf", "_seen_lf", "end", "pos", "src")

    def __init__(self, src: str) -> None:
        self.src = src
        self.end = len(src)
        self.pos = 0
        # Track newline kinds during scanning so Document needn't walk
        # the CST. Report CRLF only when every emitted newline was CRLF.
        self._seen_lf = False
        self._seen_crlf = False

    def detected_newline(self) -> str:
        r"""Return the document-wide newline kind seen during scanning.

        ``"\r\n"`` if every emitted newline was CRLF; ``"\n"``
        otherwise (LF-only, mixed, or no newlines at all).
        """
        if self._seen_crlf and not self._seen_lf:
            return "\r\n"
        return "\n"

    def line_col(self, pos: int) -> tuple[int, int]:
        """Return the 1-based (line, column) for source offset `pos`."""
        line = 1
        last_nl = -1
        for i in range(pos):
            if self.src[i] == "\n":
                line += 1
                last_nl = i
        col = pos - last_nl
        return line, col

    def error(self, message: str, *, at: int | None = None) -> TOMLParseError:
        """Build a `TOMLParseError` pointing at `at` (default: cursor)."""
        offset = self.pos if at is None else at
        line, col = self.line_col(offset)
        return TOMLParseError(message, line=line, col=col, offset=offset)

    def scan_comment(self) -> str:
        """Consume a comment from `#` to (but not including) the newline.

        The cursor must be on `#`. Raises if the comment body contains
        a control character other than tab.
        """
        src = self.src
        start = self.pos
        m = _RE_COMMENT_BODY.match(src, start + 1)
        assert m is not None  # pattern is unbounded above (*).
        end_pos = m.end()
        if end_pos < self.end:
            ch = src[end_pos]
            if ch != "\n" and ch != "\r":
                self.pos = end_pos
                cp = ord(ch)
                msg = f"invalid control character U+{cp:04X} in comment"
                raise self.error(msg)
        self.pos = end_pos
        return src[start:end_pos]

    def scan_doc_trivia(self) -> str:
        """Consume a document-scope trivia block.

        Whitespace, blank lines and full-line comments are consumed.
        Stops before the next structural token (or EOF).
        """
        start = self.pos
        src = self.src
        end = self.end
        pos = self.pos
        while pos < end:
            ch = src[pos]
            if ch == " " or ch == "\t":
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

    def scan_inline_ws_text(self) -> str:
        """Consume one run of inline whitespace; return raw text (or "").

        Newlines and comments are not whitespace here; the cursor stops
        at the first character that is neither space nor tab.
        """
        src = self.src
        end = self.end
        pos = self.pos
        if pos >= end:
            return ""
        ch = src[pos]
        if ch != " " and ch != "\t":
            return ""
        start = pos
        pos += 1
        while pos < end:
            c = src[pos]
            if c != " " and c != "\t":
                break
            pos += 1
        self.pos = pos
        return src[start:pos]

    def scan_array_trivia(self) -> str:
        """Consume trivia inside an array (or TOML 1.1 inline table).

        Whitespace, newlines and comments are all permitted. Stops
        before the next structural character.
        """
        start = self.pos
        src = self.src
        end = self.end
        pos = self.pos
        while pos < end:
            ch = src[pos]
            if ch == " " or ch == "\t":
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

    def scan_eol(self) -> EolTrivia:
        """Consume optional trailing-ws + comment + newline (or EOF).

        Raises if a non-newline, non-comment, non-EOF character is
        found after the optional whitespace.
        """
        trailing = self.scan_inline_ws_text()
        comment = ""
        src = self.src
        end = self.end
        pos = self.pos
        ch = src[pos] if pos < end else ""
        if ch == "#":
            comment = self.scan_comment()
            pos = self.pos
            ch = src[pos] if pos < end else ""
        newline = ""
        if ch == "\n":
            self.pos = pos + 1
            newline = "\n"
            self._seen_lf = True
        elif ch == "\r" and pos + 1 < end and src[pos + 1] == "\n":
            self.pos = pos + 2
            newline = "\r\n"
            self._seen_crlf = True
        elif pos < end:
            msg = f"expected newline or end of file, got {ch!r}"
            raise self.error(msg)
        return EolTrivia(trailing, comment, newline)

    def scan_string(self, *, allow_multiline: bool = True) -> StringValue:
        """Scan a string starting at the cursor; populate `raw`.

        Dispatches on the opening quote character and returns a
        `StringValue` with verbatim source (for round-tripping), decoded
        value, and style. Key parsers pass ``allow_multiline=False`` to
        reject multi-line strings.

        Precondition: cursor is at `"` or `'`. Callers look at the
        character first; this is asserted, not validated.
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
        m = _RE_BASIC_STR_BODY.match(src, self.pos + 1)
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
            elif ch == "\n" or ch == "\r":
                msg = "newline in basic string"
                raise self.error(msg)
            else:
                msg = f"invalid control character U+{ord(ch):04X} in string"
                raise self.error(msg)
            m = _RE_BASIC_STR_BODY.match(src, self.pos)
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
        elif src.startswith("\r\n", pos):
            pos += 2
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
            elif ch == "\r":
                if not src.startswith("\r\n", pos):
                    msg = "stray carriage return in string"
                    raise self.error(msg)
                out.append("\r\n")
                pos += 2
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
                    elif src.startswith("\r\n", scan):
                        scan += 2
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
        m = _RE_LITERAL_STR_BODY.match(src, start)
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
        if ch == "\n" or ch == "\r":
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

    def scan_key(self) -> tuple[list[KeyPart], list[str], str]:
        """Scan a dotted key; return its parts, separators, trailing ws.

        Each part is bare, basic-quoted or literal-quoted; each
        separator is the literal ``ws "." ws`` between two parts. The
        whitespace after the last part is consumed too, and can be used
        directly as ``pre_eq`` / ``inner_post``.
        """
        src = self.src
        parts: list[KeyPart] = []
        seps: list[str] = []
        while True:
            start = self.pos
            ch = src[start] if start < self.end else ""
            if ch == '"' or ch == "'":
                quoted = self.scan_string(allow_multiline=False)
                parts.append(KeyPart(quoted.lexeme, quoted.value))
            else:
                m = _RE_BARE_KEY.match(src, start)
                if m is None:
                    msg = f"expected key, got {ch!r}"
                    raise self.error(msg)
                self.pos = m.end()
                raw = src[start : self.pos]
                parts.append(KeyPart(raw, raw))
            sep_start = self.pos
            ws = self.scan_inline_ws_text()
            if self.pos >= self.end or src[self.pos] != ".":
                return parts, seps, ws
            self.pos += 1
            self.scan_inline_ws_text()
            seps.append(src[sep_start : self.pos])

    # Bare value tokens: bool, special floats, numbers, and date/time values.
    # The parser dispatches strings, arrays, and inline tables itself.

    def scan_value_atom(self) -> Value:
        """Scan a non-container, non-string value at the cursor.

        Bool and special-float keywords are whole-token matches, so
        ``trueish`` / ``infinity`` error rather than parsing as
        ``true`` / ``inf`` followed by garbage.
        """
        start = self.pos
        end = self._scan_value_end(start)
        token = self.src[start:end]
        if not token:
            msg = f"expected value, got {self.src[start : start + 1]!r}"
            raise self.error(msg)
        self.pos = end

        # Whole-token keyword classification.
        if token in ("true", "false"):
            return BoolValue(token, token == "true")  # noqa: S105
        if token in ("inf", "+inf"):
            return FloatValue(token, float("inf"))
        if token == "-inf":  # noqa: S105
            return FloatValue(token, float("-inf"))
        if token in ("nan", "+nan", "-nan"):
            return FloatValue(token, float("nan"))

        # Date/time literals always carry a fixed punctuation char in
        # a known position. Try them before numbers so e.g. ``1979-…``
        # is not mistaken for an integer.
        if self._looks_like_datetime(token):
            return self._parse_datetime_token(token, at=start)

        if self._looks_like_float(token):
            return self._parse_float_token(token, at=start)

        return self._parse_integer_token(token, at=start)

    def _scan_value_end(self, start: int) -> int:
        """Return the offset of the first char that ends a bare value."""
        m = _RE_VALUE_END.search(self.src, start)
        return m.start() if m is not None else len(self.src)

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
        body = token[1:] if token[:1] in "+-" else token
        if body.startswith(("0x", "0o", "0b")):
            return False
        return "." in body or "e" in body or "E" in body

    def _parse_integer_token(self, token: str, *, at: int) -> IntegerValue:
        body = token
        if body.startswith(("0x", "0o", "0b")):
            prefix = body[:2]
            digits = body[2:]
            if not digits or digits.startswith("_") or digits.endswith("_"):
                msg = f"invalid integer {token!r}"
                raise self.error(msg, at=at)
            allowed = {"0x": _HEX_DIGITS, "0o": _OCT_DIGITS, "0b": _BIN_DIGITS}[prefix]
            for c in digits:
                if c == "_":
                    continue
                if c not in allowed:
                    msg = f"invalid digit {c!r} in {token!r}"
                    raise self.error(msg, at=at)
            if "__" in digits:
                msg = f"consecutive underscores in {token!r}"
                raise self.error(msg, at=at)
            base = {"0x": 16, "0o": 8, "0b": 2}[prefix]
            value = int(digits.replace("_", ""), base)
            return IntegerValue(token, value)

        sign = ""
        if body and body[0] in "+-":
            sign = body[0]
            body = body[1:]
        if not body:
            msg = f"invalid integer {token!r}"
            raise self.error(msg, at=at)
        if body.startswith("_") or body.endswith("_"):
            msg = f"invalid integer {token!r}"
            raise self.error(msg, at=at)
        if "__" in body:
            msg = f"consecutive underscores in {token!r}"
            raise self.error(msg, at=at)
        digits_only = body.replace("_", "")
        if not _is_ascii_digits(digits_only):
            msg = f"invalid integer {token!r}"
            raise self.error(msg, at=at)
        if len(digits_only) > 1 and digits_only.startswith("0"):
            msg = f"leading zeros are not allowed in {token!r}"
            raise self.error(msg, at=at)
        try:
            value = int(sign + digits_only)
        except ValueError as exc:
            msg = f"invalid integer {token!r}: {exc}"
            raise self.error(msg, at=at) from exc
        return IntegerValue(token, value)

    def _parse_float_token(self, token: str, *, at: int) -> FloatValue:
        body = token
        sign = ""
        if body and body[0] in "+-":
            sign = body[0]
            body = body[1:]
        if "__" in body:
            msg = f"consecutive underscores in {token!r}"
            raise self.error(msg, at=at)
        for i, c in enumerate(body):
            if c == "_" and not (
                0 < i < len(body) - 1
                and body[i - 1] in _DEC_DIGITS
                and body[i + 1] in _DEC_DIGITS
            ):
                msg = f"misplaced underscore in {token!r}"
                raise self.error(msg, at=at)

        # Validate structure manually; ``float`` accepts forms TOML doesn't.
        norm = body.replace("_", "")
        exp_pos = -1
        for i, c in enumerate(norm):
            if c in ("e", "E"):
                exp_pos = i
                break
        if exp_pos != -1:
            mantissa = norm[:exp_pos]
            exponent = norm[exp_pos + 1 :]
            if not exponent or (exponent[0] in "+-" and len(exponent) == 1):
                msg = f"invalid float exponent in {token!r}"
                raise self.error(msg, at=at)
            if exponent[0] in "+-":
                exponent = exponent[1:]
            if not _is_ascii_digits(exponent):
                msg = f"invalid float exponent in {token!r}"
                raise self.error(msg, at=at)
        else:
            mantissa = norm

        if "." in mantissa:
            int_part, _, frac_part = mantissa.partition(".")
            if not int_part or not frac_part:
                msg = f"invalid float {token!r}"
                raise self.error(msg, at=at)
            if not _is_ascii_digits(int_part) or not _is_ascii_digits(frac_part):
                msg = f"invalid float {token!r}"
                raise self.error(msg, at=at)
            if len(int_part) > 1 and int_part.startswith("0"):
                msg = f"leading zeros not allowed in float {token!r}"
                raise self.error(msg, at=at)
        else:
            if not _is_ascii_digits(mantissa):
                msg = f"invalid float {token!r}"
                raise self.error(msg, at=at)
            if len(mantissa) > 1 and mantissa.startswith("0"):
                msg = f"leading zeros not allowed in float {token!r}"
                raise self.error(msg, at=at)

        value = float(sign + norm)
        return FloatValue(token, value)

    def _parse_datetime_token(self, token: str, *, at: int) -> DateTimeValue:
        # Fold a TOML date followed by ``" HH:..."`` into one
        # local-datetime token.
        src = self.src
        pos = self.pos
        if (
            len(token) == 10
            and pos < len(src)
            and src[pos] == " "
            and pos + 3 < len(src)
            and src[pos + 1] in _DEC_DIGITS
            and src[pos + 3] == ":"
        ):
            pos += 1
            extra_end = self._scan_value_end(pos)
            extra = src[pos:extra_end]
            self.pos = extra_end
            return self._parse_datetime_text(token + " " + extra, at=at)
        return self._parse_datetime_text(token, at=at)

    def _parse_datetime_text(self, text: str, *, at: int) -> DateTimeValue:
        if len(text) >= 3 and text[2] == ":":
            try:
                value = self._parse_time_text(text)
            except ValueError as exc:
                msg = f"invalid time {text!r}: {exc}"
                raise self.error(msg, at=at) from exc
            return DateTimeValue(text, value)

        if len(text) < 10 or text[4] != "-" or text[7] != "-":
            msg = f"invalid date/datetime {text!r}"
            raise self.error(msg, at=at)
        date_part = text[:10]
        year_s, month_s, day_s = date_part[:4], date_part[5:7], date_part[8:10]
        if not (
            _is_ascii_digits(year_s)
            and _is_ascii_digits(month_s)
            and _is_ascii_digits(day_s)
        ):
            msg = f"invalid date {date_part!r}"
            raise self.error(msg, at=at)
        try:
            d = date(int(year_s), int(month_s), int(day_s))
        except ValueError as exc:
            msg = f"invalid date {date_part!r}: {exc}"
            raise self.error(msg, at=at) from exc

        rest = text[10:]
        if not rest:
            return DateTimeValue(text, d)
        if rest[0] not in ("T", "t", " "):
            msg = f"expected date/time separator, got {rest[0]!r}"
            raise self.error(msg, at=at)
        time_part = rest[1:]
        offset_pos = -1
        for i, c in enumerate(time_part):
            if c in ("Z", "z", "+", "-") and i >= 1:
                offset_pos = i
                break
        if offset_pos == -1:
            try:
                t = self._parse_time_text(time_part)
            except ValueError as exc:
                msg = f"invalid time {time_part!r}: {exc}"
                raise self.error(msg, at=at) from exc
            return DateTimeValue(text, datetime.combine(d, t))
        try:
            t = self._parse_time_text(time_part[:offset_pos])
            tz = self._parse_offset(time_part[offset_pos:])
        except ValueError as exc:
            msg = f"invalid datetime {text!r}: {exc}"
            raise self.error(msg, at=at) from exc
        dt = datetime.combine(d, t).replace(tzinfo=tz)
        return DateTimeValue(text, dt)

    @staticmethod
    def _parse_time_text(text: str) -> time:
        # TOML 1.1: seconds are optional; ``HH:MM`` defaults to ``:00``.
        if len(text) < 5 or text[2] != ":":
            msg = f"bad time format: {text!r}"
            raise ValueError(msg)
        hh_s, mm_s = text[:2], text[3:5]
        if not (_is_ascii_digits(hh_s) and _is_ascii_digits(mm_s)):
            msg = f"bad time format: {text!r}"
            raise ValueError(msg)
        hh = int(hh_s)
        mm = int(mm_s)
        rest = text[5:]
        if not rest:
            return time(hh, mm, 0, 0)
        if rest[0] != ":":
            msg = f"bad time format: {text!r}"
            raise ValueError(msg)
        if len(rest) < 3:
            msg = f"bad seconds in {text!r}"
            raise ValueError(msg)
        ss_s = rest[1:3]
        if not _is_ascii_digits(ss_s):
            msg = f"bad seconds in {text!r}"
            raise ValueError(msg)
        ss = int(ss_s)
        rest = rest[3:]
        usec = 0
        if rest:
            if rest[0] != ".":
                msg = f"bad fractional seconds in {text!r}"
                raise ValueError(msg)
            frac = rest[1:]
            if not frac or not _is_ascii_digits(frac):
                msg = f"bad fractional seconds in {text!r}"
                raise ValueError(msg)
            digits = (frac + "000000")[:6]
            usec = int(digits)
        return time(hh, mm, ss, usec)

    @staticmethod
    def _parse_offset(text: str) -> timezone:
        if text in ("Z", "z"):
            return timezone.utc
        if len(text) != 6 or text[0] not in "+-" or text[3] != ":":
            msg = f"bad timezone offset: {text!r}"
            raise ValueError(msg)
        hh_s, mm_s = text[1:3], text[4:6]
        if not (_is_ascii_digits(hh_s) and _is_ascii_digits(mm_s)):
            msg = f"bad timezone offset: {text!r}"
            raise ValueError(msg)
        sign = 1 if text[0] == "+" else -1
        hh = int(hh_s)
        mm = int(mm_s)
        if hh > 23 or mm > 59:
            msg = f"timezone offset out of range: {text!r}"
            raise ValueError(msg)
        delta = timedelta(hours=hh, minutes=mm) * sign
        return timezone(delta)


__all__ = ["_Scanner"]
