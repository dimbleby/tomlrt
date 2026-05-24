"""Key-path argument parsing and validation helpers."""

from __future__ import annotations

from collections.abc import Sequence

from tomlrt._errors import TOMLError


def split_path(path: str | Sequence[str]) -> list[str]:
    """Split a path argument into a list of component names.

    A ``str`` is interpreted as a dotted path (no quoting support; for
    keys containing dots, pass a sequence). A non-string ``Sequence``
    is taken verbatim.
    """
    if isinstance(path, str):
        return path.split(".") if path else []
    return list(path)


def validate_path(path: object) -> list[str]:
    """Validate a key-path argument and return its components.

    Raises ``TypeError`` for the wrong outer type, and ``TOMLError``
    for empty paths or paths with empty segments.
    """
    if isinstance(path, str):
        if path == "":
            msg = "key path must not be empty"
            raise TOMLError(msg)
        parts = path.split(".")
        for p in parts:
            if p == "":
                msg = f"key path {path!r} contains an empty segment"
                raise TOMLError(msg)
        return parts
    if isinstance(path, Sequence):
        if len(path) == 0:
            msg = "key path must not be empty"
            raise TOMLError(msg)
        out: list[str] = []
        for seg in path:
            if not isinstance(seg, str):
                msg = f"key path segment must be str, got {type(seg).__name__}"
                raise TypeError(msg)
            if seg == "":
                msg = "key path contains an empty segment"
                raise TOMLError(msg)
            out.append(seg)
        return out
    msg = f"key path must be str or sequence of str, got {type(path).__name__}"
    raise TypeError(msg)
