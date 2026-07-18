"""Semantic validator for parsed TOML.

The parser calls this for headers, key/value lines, and inline-table
local duplicate / dotted-prefix checks.

It also tracks the active ``AoTEntry`` per AoT path so the parser can
attach the correct owner to each physical slot.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from tomlrt._slots import AoTEntry
from tomlrt._values import ArrayValue, InlineTableValue

if TYPE_CHECKING:
    from collections.abc import Callable

    from tomlrt._errors import TOMLParseError
    from tomlrt._values import Value

    ErrorBuilder = Callable[..., TOMLParseError]


_HeaderKind = Literal["table", "aot-entry"]
_PathKind = Literal["aot", "dotted", "explicit", "implicit", "value"]


class _Validator:
    __slots__ = (
        "_active_aot_entries",
        "_aot_subpaths",
        "_current_owner_aot_entry",
        "_current_section",
        "_error",
        "_path_kinds",
    )

    def __init__(self, error_builder: ErrorBuilder) -> None:
        self._error = error_builder
        self._path_kinds: dict[tuple[str, ...], _PathKind] = {}
        # Index from each active AoT path to all sub-paths registered under it.
        self._aot_subpaths: dict[tuple[str, ...], list[tuple[str, ...]]] = {}
        self._current_section: tuple[str, ...] = ()

        # Active AoT paths map to their most recently opened entry.
        self._active_aot_entries: dict[tuple[str, ...], AoTEntry] = {}
        self._current_owner_aot_entry: AoTEntry | None = None

    def current_section(self) -> tuple[str, ...]:
        return self._current_section

    def current_owner_aot_entry(self) -> AoTEntry | None:
        return self._current_owner_aot_entry

    def enter_header(
        self, path: tuple[str, ...], kind: _HeaderKind, *, at: int
    ) -> AoTEntry | None:
        """Validate a ``[H]`` / ``[[H]]`` header.

        Returns the opened ``AoTEntry`` for ``[[H]]``, otherwise ``None``.
        """
        path_kinds = self._path_kinds
        # Prefix overlaps with a bound value would mean overwriting a scalar
        # (or an inline-table value) with a table — always invalid.
        for i in range(1, len(path)):
            prefix = path[:i]
            if path_kinds.get(prefix) == "value":
                joined = ".".join(prefix)
                msg = f"cannot use {joined!r} as a table: already defined as a value"
                raise self._error(msg, at=at)
        current_kind = path_kinds.get(path)
        if current_kind == "value":
            joined = ".".join(path)
            msg = f"cannot define {joined!r} as a table: already defined as a value"
            raise self._error(msg, at=at)
        if current_kind == "dotted":
            joined = ".".join(path)
            msg = (
                f"cannot define {joined!r} as a table: already created via dotted keys"
            )
            raise self._error(msg, at=at)

        new_entry: AoTEntry | None = None
        if kind == "table":
            if current_kind == "explicit":
                msg = f"redefinition of table {'.'.join(path)!r}"
                raise self._error(msg, at=at)
            if current_kind == "aot":
                joined = ".".join(path)
                msg = f"cannot redefine array-of-tables {joined!r} as a normal table"
                raise self._error(msg, at=at)
            self._record_path(path, "explicit")
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
            # A new AoT entry invalidates per-entry tracking under it.
            self._reset_scope_under(path)
            self._record_path(path, "aot")
            new_entry = AoTEntry()
            self._active_aot_entries[path] = new_entry

        # Intermediate prefixes become implicit tables.
        for i in range(1, len(path)):
            sub = path[:i]
            if sub not in path_kinds:
                self._record_path(sub, "implicit")

        self._current_section = path
        self._current_owner_aot_entry = self._compute_owner_aot_entry(path)
        return new_entry

    def _compute_owner_aot_entry(
        self, section_path: tuple[str, ...]
    ) -> AoTEntry | None:
        """Return the deepest active AoTEntry whose path is a prefix of the section.

        ``section_path`` is included as a prefix of itself: the entry
        opened by ``[[a]]`` has owner_aot_entry = itself.
        """
        if not self._active_aot_entries:
            return None
        # Walk from longest to shortest prefix.
        for i in range(len(section_path), 0, -1):
            prefix = section_path[:i]
            entry = self._active_aot_entries.get(prefix)
            if entry is not None:
                return entry
        return None

    def record_keyvalue(
        self, key_path: tuple[str, ...], value: Value, *, at: int
    ) -> None:
        section = self._current_section
        full = section + key_path if section else key_path
        path_kinds = self._path_kinds
        current_kind = path_kinds.get(full)
        if current_kind == "value":
            msg = f"duplicate key {'.'.join(full)!r}"
            raise self._error(msg, at=at)
        if current_kind is not None:
            msg = f"key {'.'.join(full)!r} already defined as a table"
            raise self._error(msg, at=at)
        # Intermediate-prefix conflicts.
        slen = len(section)
        flen = len(full)
        if flen > slen + 1:
            for i in range(slen + 1, flen):
                sub = full[:i]
                sub_kind = path_kinds.get(sub)
                if sub_kind == "value":
                    msg = f"key {'.'.join(sub)!r} already defined as a value"
                    raise self._error(msg, at=at)
                if sub_kind == "explicit":
                    joined = ".".join(sub)
                    msg = (
                        f"cannot extend explicitly-defined table {joined!r} "
                        "via dotted keys"
                    )
                    raise self._error(msg, at=at)
                if sub_kind == "aot":
                    msg = (
                        f"cannot extend array-of-tables {'.'.join(sub)!r} "
                        "via dotted keys"
                    )
                    raise self._error(msg, at=at)
                self._record_path(sub, "dotted")
        self._record_path(full, "value")
        if isinstance(value, InlineTableValue):
            self._register_inline_table(value, abs_prefix=full)
        elif isinstance(value, ArrayValue):
            for item in value.items:
                if isinstance(item.value, InlineTableValue):
                    self._register_inline_table(item.value, abs_prefix=None)

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

    def _register_inline_table(
        self,
        table: InlineTableValue,
        *,
        abs_prefix: tuple[str, ...] | None,
    ) -> None:
        for entry in table.items:
            path = entry.key_path
            if abs_prefix is not None:
                full = abs_prefix + path
                self._record_path(full, "value")
                for i in range(1, len(path)):
                    sub = abs_prefix + path[:i]
                    self._record_path(sub, "dotted")
            sub_abs: tuple[str, ...] | None
            if isinstance(entry.value, InlineTableValue):
                sub_abs = (abs_prefix + path) if abs_prefix is not None else None
                self._register_inline_table(entry.value, abs_prefix=sub_abs)
            elif isinstance(entry.value, ArrayValue):
                for item in entry.value.items:
                    if isinstance(item.value, InlineTableValue):
                        self._register_inline_table(item.value, abs_prefix=None)

    def _record_path(self, path: tuple[str, ...], kind: _PathKind) -> None:
        previous = self._path_kinds.get(path)
        assert (
            previous is None
            or previous == kind
            or (previous == "implicit" and kind in ("dotted", "explicit"))
        )
        if previous is None:
            self._track(path)
        self._path_kinds[path] = kind

    def _track(self, path: tuple[str, ...]) -> None:
        if not self._active_aot_entries:
            return
        for i in range(len(path) - 1, 0, -1):
            prefix = path[:i]
            if prefix in self._active_aot_entries:
                self._aot_subpaths.setdefault(prefix, []).append(path)
                return

    def _reset_scope_under(self, path: tuple[str, ...]) -> None:
        subs = self._aot_subpaths.pop(path, None)
        if not subs:
            return
        nested_aots: list[tuple[str, ...]] = []
        for sub in subs:
            if self._path_kinds.pop(sub) == "aot":
                nested_aots.append(sub)
                self._active_aot_entries.pop(sub)
        for nested in nested_aots:
            self._reset_scope_under(nested)


__all__ = ["_Validator"]
