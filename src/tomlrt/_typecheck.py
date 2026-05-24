"""Runtime type-checks for user-supplied keys and mappings.

Pure functions with no dependencies on the container / array layers,
so any module that accepts user-supplied data at a public boundary
(``_container``, ``_array``, …) can import them without participating
in the circular-import graph that those layers form among themselves.

Distinct from :mod:`tomlrt._validator`, which is the stateful TOML
*semantic* validator that the parser drives to enforce cross-section
grammar rules (a key bound as a value cannot later open a table, an
explicit ``[a]`` cannot redefine an already-opened table, etc.).
This module is API-boundary plumbing, not parse-time machinery.
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

    Returns the validated mapping unchanged — identity is preserved so
    downstream paths that branch on the concrete type (e.g. ``Table``
    vs plain ``dict``) keep working. Centralising this means every
    factory / mutator boundary produces the same wording
    (``"<label> must be a Mapping"`` / ``"TOML keys must be str"``)
    instead of leaking
    ``AttributeError: 'list' object has no attribute 'items'`` from
    inside the layout pipeline.
    """
    if _check_str_mapping(value, label=label):
        return value
    raise AssertionError  # pragma: no cover -- _check_str_mapping never returns False


def _check_str_mapping(value: object, *, label: str) -> TypeIs[Mapping[str, Any]]:
    """Validate-or-raise ``TypeIs`` predicate for ``Mapping[str, Any]``.

    Performs the runtime check in a single pass and raises a labelled
    ``TypeError`` on failure. The ``TypeIs`` return type lets the type
    checker narrow ``value`` to ``Mapping[str, Any]`` at the call site
    in the ``True`` branch.

    Returns ``True`` on success and never returns ``False`` (every
    rejection raises), so callers can write ``if _check_str_mapping(v):
    return v`` to drive the narrowing. Don't use ``assert`` — it would
    be stripped under ``python -O``, skipping validation entirely.
    """
    if not isinstance(value, Mapping):
        msg = f"{label} must be a Mapping, got {type(value).__name__}"
        raise TypeError(msg)
    for k in value:
        _validate_key(k)
    return True
