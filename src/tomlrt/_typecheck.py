"""Runtime type-checks for user-supplied keys and mappings.

These pure helpers have no container / array dependencies, so public
API boundaries can import them without joining that circular-import
graph.

Unlike :mod:`tomlrt._validator`, this module handles runtime API input
validation, not parse-time TOML semantics.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import sys

    if sys.version_info >= (3, 13):
        from typing import TypeIs
    else:  # pragma: no cover -- backport for Python < 3.13
        from typing_extensions import TypeIs


def _validate_key(key: object) -> str:
    """Reject a non-``str`` TOML key with a consistent ``TypeError``.

    Returns the validated key (narrowed to ``str``) so call sites that
    care about the type can do ``k = _validate_key(k)``.
    """
    if not isinstance(key, str):
        msg = f"TOML keys must be str, got {type(key).__name__}"
        raise TypeError(msg)
    return key


def _validate_mapping(value: object, *, label: str) -> Mapping[str, Any]:
    """Reject a non-Mapping ``value`` or any Mapping with non-string keys.

    Returns the mapping unchanged so downstream paths that branch on
    concrete type (e.g. ``Table`` vs ``dict``) keep working. Centralise
    this so every factory / mutator boundary reports the same errors
    instead of leaking layout-pipeline ``AttributeError``s.
    """
    if _check_str_mapping(value, label=label):
        return value
    raise AssertionError  # pragma: no cover -- _check_str_mapping never returns False


def _check_str_mapping(value: object, *, label: str) -> TypeIs[Mapping[str, Any]]:
    """Validate-or-raise ``TypeIs`` predicate for ``Mapping[str, Any]``.

    Raises a labelled ``TypeError`` on failure. The ``TypeIs`` return
    lets callers narrow ``value`` in the ``True`` branch.

    Never returns ``False``; every rejection raises. Don't replace this
    with ``assert``, which ``python -O`` would strip.
    """
    if not isinstance(value, Mapping):
        msg = f"{label} must be a Mapping, got {type(value).__name__}"
        raise TypeError(msg)
    for k in value:
        _validate_key(k)
    return True
