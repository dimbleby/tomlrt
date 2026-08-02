"""Scalar predicates and Python-to-TOML scalar coercion.

Pure helpers for wire-format `Value` types, kept out of the container
layer.
"""

from __future__ import annotations

import math
from datetime import date, datetime, time, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import sys

    if sys.version_info >= (3, 13):
        from typing import TypeIs
    else:  # pragma: no cover -- backport for Python < 3.13
        from typing_extensions import TypeIs

from tomlrt._values import (
    _KEY_ESCAPES,
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
        return BoolValue("true" if v else "false", v)
    if isinstance(v, int):
        return IntegerValue(str(v), v)
    if isinstance(v, float):
        return FloatValue(float_lexeme(v), v)
    if isinstance(v, str):
        return StringValue(basic_string_lexeme(v), v)
    validate_scalar(v)
    return DateTimeValue(v.isoformat(), v)


def validate_scalar(v: Scalar) -> None:
    """Raise if a scalar cannot be represented in TOML."""
    if isinstance(v, datetime):
        _check_toml_offset(v)
    elif isinstance(v, time) and v.tzinfo is not None:
        msg = f"cannot represent {v!r} in TOML: local time cannot carry a timezone"
        raise ValueError(msg)


def _check_toml_offset(v: datetime) -> None:
    """Raise if ``v.tzinfo`` isn't representable as TOML's whole-minute offset."""
    offset = v.utcoffset()
    if offset is not None and offset % timedelta(minutes=1):
        msg = (
            f"cannot represent {v!r} in TOML: timezone offset {offset} is not "
            "a whole number of minutes"
        )
        raise ValueError(msg)


def float_lexeme(v: float) -> str:
    if math.isnan(v):
        return "nan"
    if math.isinf(v):
        return "-inf" if v < 0 else "inf"
    # repr() of finite floats includes a TOML-accepted fraction or exponent.
    return repr(v)


_STRING_ESCAPES: dict[int, str] = {
    **_KEY_ESCAPES,
    0x08: "\\b",
    0x09: "\\t",
    0x0A: "\\n",
    0x0C: "\\f",
    0x0D: "\\r",
}


def basic_string_lexeme(v: str) -> str:
    return f'"{v.translate(_STRING_ESCAPES)}"'
