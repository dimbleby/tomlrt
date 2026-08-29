"""Runtime type-checks for user-supplied keys and mappings.

These pure helpers have no container / array dependencies, so public
API boundaries can import them without joining that circular-import
graph.

Unlike :mod:`tomlrt._validator`, this module handles runtime API input
validation, not parse-time TOML semantics.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

_MappingT = TypeVar("_MappingT", bound=Mapping[str, object])


def _validate_key(key: object) -> str:
    """Reject a non-``str`` TOML key with a consistent ``TypeError``.

    Returns the validated key (narrowed to ``str``) so call sites that
    care about the type can do ``k = _validate_key(k)``.
    """
    if not isinstance(key, str):
        msg = f"TOML keys must be str, got {type(key).__name__}"
        raise TypeError(msg)
    return key


def _validate_mapping(value: _MappingT, *, label: str) -> _MappingT:
    """Require a mapping and shallowly validate its item keys."""
    _require_mapping(value, label=label)
    for item in value.items():
        _validate_key(item[0])
    return value


def _require_mapping(value: object, *, label: str) -> Mapping[Any, object]:
    """Return ``value`` as a mapping, or raise a consistent ``TypeError``."""
    if not isinstance(value, Mapping):
        msg = f"{label} must be a Mapping, got {type(value).__name__}"
        raise TypeError(msg)
    return value
