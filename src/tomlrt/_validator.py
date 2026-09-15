"""Semantic validator for parsed TOML.

The parser calls this for headers, key/value lines, and inline-table
local duplicate / dotted-prefix checks.

It also tracks the active ``AoTEntry`` per AoT path so the parser can
attach the correct owner to each physical slot.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from tomlrt._slots import AoTEntry

if TYPE_CHECKING:
    from typing import Protocol

    from tomlrt._errors import TOMLParseError

    class ErrorBuilder(Protocol):
        """Build a parse error at a known source offset."""

        def __call__(self, message: str, *, at: int) -> TOMLParseError: ...


_HeaderKind = Literal["table", "aot-entry"]
_ScopeKind = Literal["dotted", "explicit", "implicit"] | AoTEntry


class _Scope:
    """A table namespace; an AoT token selects its current entry."""

    __slots__ = ("children", "kind")

    def __init__(self, kind: _ScopeKind) -> None:
        self.children: dict[str, _Scope | Literal["value"]] = {}
        self.kind = kind


def _lookup(scope: _Scope, path: tuple[str, ...]) -> _Scope | Literal["value"] | None:
    """Look up a nonempty path without enforcing declaration kinds."""
    for name in path[:-1]:
        child = scope.children.get(name)
        if not isinstance(child, _Scope):
            return None
        scope = child
    return scope.children.get(path[-1])


class _Validator:
    __slots__ = (
        "_current",
        "_error",
        "_root",
        "current_owner_aot_entry",
        "current_section",
    )

    def __init__(self, error_builder: ErrorBuilder) -> None:
        self._error = error_builder
        self._root = _Scope("implicit")
        self._current = self._root
        self.current_section: tuple[str, ...] = ()
        self.current_owner_aot_entry: AoTEntry | None = None

    def enter_header(
        self, path: tuple[str, ...], kind: _HeaderKind, *, at: int
    ) -> AoTEntry | None:
        """Validate a ``[H]`` / ``[[H]]`` header.

        Returns the opened ``AoTEntry`` for ``[[H]]``, otherwise ``None``.
        """
        parent = self._root
        owner: AoTEntry | None = None
        for i, name in enumerate(path[:-1], start=1):
            child = parent.children.get(name)
            if child == "value":
                joined = ".".join(path[:i])
                msg = f"cannot use {joined!r} as a table: already defined as a value"
                raise self._error(msg, at=at)
            if child is None:
                child = _Scope("implicit")
                parent.children[name] = child
            parent = child
            if isinstance(parent.kind, AoTEntry):
                owner = parent.kind
        name = path[-1]
        current = parent.children.get(name)
        if current == "value":
            joined = ".".join(path)
            msg = f"cannot define {joined!r} as a table: already defined as a value"
            raise self._error(msg, at=at)
        current_kind = current.kind if current is not None else None
        if current_kind == "dotted":
            joined = ".".join(path)
            msg = (
                f"cannot define {joined!r} as a table: already created via dotted keys"
            )
            raise self._error(msg, at=at)

        new_entry: AoTEntry | None = None
        next_kind: _ScopeKind
        if kind == "table":
            if current_kind == "explicit":
                msg = f"redefinition of table {'.'.join(path)!r}"
                raise self._error(msg, at=at)
            if isinstance(current_kind, AoTEntry):
                joined = ".".join(path)
                msg = f"cannot redefine array-of-tables {joined!r} as a normal table"
                raise self._error(msg, at=at)
            next_kind = "explicit"
        else:  # aot-entry
            if current_kind == "explicit":
                msg = f"cannot redefine table {'.'.join(path)!r} as an array-of-tables"
                raise self._error(msg, at=at)
            if current_kind == "implicit":
                msg = (
                    f"cannot define {'.'.join(path)!r} as an array-of-tables: "
                    "already used as an implicit table"
                )
                raise self._error(msg, at=at)
            new_entry = AoTEntry()
            next_kind = owner = new_entry

        if current is None:
            current = _Scope(next_kind)
            parent.children[name] = current
        else:
            current.kind = next_kind
            if new_entry is not None:
                current.children.clear()
        self._current = current
        self.current_section = path
        self.current_owner_aot_entry = owner
        return new_entry

    def record_keyvalue(self, key_path: tuple[str, ...], *, at: int) -> None:
        """Bind a value, sealing its path against subsequent extension.

        Inline-table contents are checked in their own local scope;
        recording the complete value here closes all its descendants.
        """
        section = self.current_section
        # Terminal conflicts take precedence over invalid dotted prefixes.
        existing = _lookup(self._current, key_path)
        if existing == "value":
            msg = f"duplicate key {'.'.join(section + key_path)!r}"
            raise self._error(msg, at=at)
        if existing is not None:
            msg = f"key {'.'.join(section + key_path)!r} already defined as a table"
            raise self._error(msg, at=at)
        scope = self._current
        if len(key_path) > 1:
            for i, name in enumerate(key_path[:-1], start=1):
                child = scope.children.get(name)
                if child == "value":
                    msg = (
                        f"key {'.'.join(section + key_path[:i])!r} "
                        "already defined as a value"
                    )
                    raise self._error(msg, at=at)
                if child is None:
                    child = _Scope("dotted")
                    scope.children[name] = child
                elif child.kind == "explicit":
                    msg = (
                        "cannot extend explicitly-defined table "
                        f"{'.'.join(section + key_path[:i])!r} via dotted keys"
                    )
                    raise self._error(msg, at=at)
                elif isinstance(child.kind, AoTEntry):
                    msg = (
                        "cannot extend array-of-tables "
                        f"{'.'.join(section + key_path[:i])!r} via dotted keys"
                    )
                    raise self._error(msg, at=at)
                else:
                    child.kind = "dotted"
                scope = child
        scope.children[key_path[-1]] = "value"

    def check_inline_key_conflict(
        self,
        path: tuple[str, ...],
        seen_values: set[tuple[str, ...]],
        seen_prefixes: set[tuple[str, ...]],
        *,
        at: int,
    ) -> None:
        if path in seen_values:
            msg = f"duplicate key {'.'.join(path)!r} in inline table"
            raise self._error(msg, at=at)
        if path in seen_prefixes:
            msg = (
                f"key {'.'.join(path)!r} in inline table conflicts with "
                "an existing dotted-key prefix"
            )
            raise self._error(msg, at=at)
        for i in range(1, len(path)):
            sub = path[:i]
            if sub in seen_values:
                msg = f"inline-table key {'.'.join(sub)!r} already defined as a value"
                raise self._error(msg, at=at)
            seen_prefixes.add(sub)


__all__ = ["_Validator"]
