"""Semantic checks and diagnostics over the namespaces being parsed."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tomlrt._array import AoT
from tomlrt._container import Table

if TYPE_CHECKING:
    from collections.abc import Container
    from typing import Protocol

    from tomlrt._errors import TOMLParseError
    from tomlrt._slots import AoTEntry

    class ErrorBuilder(Protocol):
        """Build a parse error at a known source offset."""

        def __call__(self, message: str, *, at: int) -> TOMLParseError: ...


def check_header(
    current: object,
    path: tuple[str, ...],
    entry: AoTEntry | None,
    error: ErrorBuilder,
    *,
    at: int,
) -> None:
    if isinstance(current, AoT):
        msg = f"cannot redefine array-of-tables {'.'.join(path)!r} as a normal table"
    elif not isinstance(current, Table) or current._inline:  # noqa: SLF001
        msg = f"cannot define {'.'.join(path)!r} as a table: already defined as a value"
    elif current._header is not None:  # noqa: SLF001
        joined = ".".join(path)
        msg = (
            f"redefinition of table {joined!r}"
            if entry is None
            else f"cannot redefine table {joined!r} as an array-of-tables"
        )
    # During parsing a headerless table can acquire body slots only
    # through a dotted declaration, which seals it against its own header.
    elif current._body_tail is not None:  # noqa: SLF001
        joined = ".".join(path)
        msg = f"cannot define {joined!r} as a table: already created via dotted keys"
    elif entry is not None:
        msg = (
            f"cannot define {'.'.join(path)!r} as an array-of-tables: "
            "already used as an implicit table"
        )
    else:
        return
    raise error(msg, at=at)


def _namespace(value: object) -> Table | None:
    if isinstance(value, AoT):
        return value[-1]
    if isinstance(value, Table) and not value._inline:  # noqa: SLF001
        return value
    return None


def key_conflict(
    current: object, path: tuple[str, ...], error: ErrorBuilder, *, at: int
) -> TOMLParseError:
    joined = ".".join(path)
    if _namespace(current) is None:
        msg = f"duplicate key {joined!r}"
    else:
        msg = f"key {joined!r} already defined as a table"
    return error(msg, at=at)


def dotted_conflict(
    child: Table | AoT,
    host_path: tuple[str, ...],
    key_path: tuple[str, ...],
    depth: int,
    error: ErrorBuilder,
    *,
    at: int,
) -> TOMLParseError:
    """A terminal conflict outranks the first forbidden dotted prefix."""
    current = _namespace(child)
    for name in key_path[depth:-1]:
        assert current is not None
        current = _namespace(current.get(name))
        if current is None:
            break
    if current is not None and key_path[-1] in current:
        return key_conflict(current[key_path[-1]], host_path + key_path, error, at=at)
    what = "array-of-tables" if isinstance(child, AoT) else "explicitly-defined table"
    joined = ".".join(host_path + key_path[:depth])
    return error(f"cannot extend {what} {joined!r} via dotted keys", at=at)


def check_inline_key_conflict(
    path: tuple[str, ...],
    seen_values: Container[tuple[str, ...]],
    seen_prefixes: set[tuple[str, ...]],
    error: ErrorBuilder,
    *,
    at: int,
) -> None:
    if path in seen_values:
        msg = f"duplicate key {'.'.join(path)!r} in inline table"
        raise error(msg, at=at)
    if path in seen_prefixes:
        msg = (
            f"key {'.'.join(path)!r} in inline table conflicts with "
            "an existing dotted-key prefix"
        )
        raise error(msg, at=at)
    for i in range(1, len(path)):
        sub = path[:i]
        if sub in seen_values:
            msg = f"inline-table key {'.'.join(sub)!r} already defined as a value"
            raise error(msg, at=at)
        seen_prefixes.add(sub)


__all__ = [
    "check_header",
    "check_inline_key_conflict",
    "dotted_conflict",
    "key_conflict",
]
