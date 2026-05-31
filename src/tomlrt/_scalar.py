"""Scalar predicates and Python-to-TOML scalar coercion.

These helpers are pure: they depend on `_values` for the wire-format
`Value` types and on `datetime` for type discrimination, but they do
not touch the container layer. Lifted out of `_container` so the same
tests cover them in isolation and the container module shrinks.
"""

from __future__ import annotations

import math
from datetime import date, datetime, time

from tomlrt._values import (
    BoolValue,
    DateTimeValue,
    FloatValue,
    IntegerValue,
    StringValue,
)


def is_scalar(v: object) -> bool:
    """True iff ``v`` is a TOML scalar (and not an array / table)."""
    # `bool` is an `int` subclass — explicit allow keeps the semantics
    # in this gate clear.
    if isinstance(v, bool):
        return True
    if isinstance(v, (int, float, str)):
        return True
    return isinstance(v, (datetime, date, time))


def coerce_scalar(
    v: object,
) -> StringValue | IntegerValue | FloatValue | BoolValue | DateTimeValue:
    """Coerce a Python scalar to a fresh `Value` with a default lexeme."""
    if isinstance(v, bool):
        return BoolValue(lexeme="true" if v else "false", value=v)
    if isinstance(v, int):
        return IntegerValue(lexeme=str(v), value=v)
    if isinstance(v, float):
        return FloatValue(lexeme=float_lexeme(v), value=v)
    if isinstance(v, str):
        return StringValue(lexeme=basic_string_lexeme(v), value=v)
    if isinstance(v, datetime):
        return DateTimeValue(lexeme=v.isoformat(), value=v)
    if isinstance(v, date):
        return DateTimeValue(lexeme=v.isoformat(), value=v)
    if isinstance(v, time):
        return DateTimeValue(lexeme=v.isoformat(), value=v)
    msg = f"cannot coerce {type(v).__name__} to a TOML scalar"
    raise TypeError(msg)


def float_lexeme(v: float) -> str:
    if math.isnan(v):
        return "nan"
    if math.isinf(v):
        return "-inf" if v < 0 else "inf"
    s = repr(v)
    # Python may emit "1e10" — TOML requires a fractional component or an
    # exponent; keep the repr() output as is (TOML accepts both).
    if "." not in s and "e" not in s and "E" not in s and "n" not in s:
        s += ".0"
    return s


def basic_string_lexeme(v: str) -> str:
    out = ['"']
    for ch in v:
        c = ord(ch)
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch == "\b":
            out.append("\\b")
        elif ch == "\t":
            out.append("\\t")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\f":
            out.append("\\f")
        elif ch == "\r":
            out.append("\\r")
        elif c < 0x20 or c == 0x7F:
            out.append(f"\\u{c:04X}")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)
