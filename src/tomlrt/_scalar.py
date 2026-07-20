"""Scalar predicates and Python-to-TOML scalar coercion.

Pure helpers for wire-format `Value` types, kept out of the container
layer.
"""

from __future__ import annotations

import math
from datetime import date, datetime, time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import sys

    if sys.version_info >= (3, 13):
        from typing import TypeIs
    else:  # pragma: no cover -- backport for Python < 3.13
        from typing_extensions import TypeIs

from tomlrt._values import (
    BoolValue,
    DateTimeValue,
    FloatValue,
    IntegerValue,
    StringValue,
)

Scalar = bool | int | float | str | datetime | date | time


def is_scalar(v: object) -> TypeIs[Scalar]:
    """True iff ``v`` is a TOML scalar (and not an array / table)."""
    # `bool` is an `int` subclass; keep the gate explicit.
    if isinstance(v, bool):
        return True
    if isinstance(v, (int, float, str)):
        return True
    return isinstance(v, (datetime, date, time))


def coerce_scalar(
    v: Scalar,
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
    return DateTimeValue(lexeme=v.isoformat(), value=v)


def float_lexeme(v: float) -> str:
    if math.isnan(v):
        return "nan"
    if math.isinf(v):
        return "-inf" if v < 0 else "inf"
    # repr() of finite floats includes a TOML-accepted fraction or exponent.
    return repr(v)


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
