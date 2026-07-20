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

_MappingT = TypeVar("_MappingT", bound=Mapping[Any, Any])


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
    """Reject a non-Mapping ``value`` or any Mapping with non-string keys.

    Returns the same mapping so downstream paths that branch on
    concrete type (e.g. ``Table`` vs ``dict``) keep working. Centralise
    this so every factory / mutator boundary reports the same errors
    instead of leaking layout-pipeline ``AttributeError``s.
    """
    _require_mapping(value, label=label)
    for k in value:
        _validate_key(k)
    return value


def _require_mapping(value: object, *, label: str) -> None:
    """Reject a non-mapping passed despite the public type signature."""
    if not isinstance(value, Mapping):
        msg = f"{label} must be a Mapping, got {type(value).__name__}"
        raise TypeError(msg)
