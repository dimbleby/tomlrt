"""Key-path argument parsing and validation helpers."""

from __future__ import annotations

from collections.abc import Sequence


def validate_path(path: object) -> list[str]:
    """Validate a key-path argument and return its components.

    Raises ``TypeError`` for the wrong outer type, and ``ValueError``
    for empty paths or paths with empty segments.
    """
    if isinstance(path, str):
        if path == "":
            msg = "key path must not be empty"
            raise ValueError(msg)
        parts = path.split(".")
        for p in parts:
            if p == "":
                msg = f"key path {path!r} contains an empty segment"
                raise ValueError(msg)
        return parts
    if isinstance(path, Sequence):
        if len(path) == 0:
            msg = "key path must not be empty"
            raise ValueError(msg)
        out: list[str] = []
        for seg in path:
            if not isinstance(seg, str):
                msg = f"key path segment must be str, got {type(seg).__name__}"
                raise TypeError(msg)
            if seg == "":
                msg = "key path contains an empty segment"
                raise ValueError(msg)
            out.append(seg)
        return out
    msg = f"key path must be str or sequence of str, got {type(path).__name__}"
    raise TypeError(msg)
