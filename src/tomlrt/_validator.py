"""Semantic validator for parsed TOML.

The parser calls this for headers, key/value lines, and inline-table
local duplicate / dotted-prefix checks.

It also tracks the active ``AoTEntry`` per AoT path so the parser can
attach the correct owner to each physical slot.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from tomlrt._slots import AoTEntry
from tomlrt._values import InlineTableValue

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
        "_error",
        "_path_kinds",
        "current_owner_aot_entry",
        "current_owner_path",
        "current_section",
    )

    def __init__(self, error_builder: ErrorBuilder) -> None:
        self._error = error_builder
        self._path_kinds: dict[tuple[str, ...], _PathKind] = {}
        # Index from each active AoT path to all sub-paths registered under it.
        self._aot_subpaths: dict[tuple[str, ...], list[tuple[str, ...]]] = {}
        self.current_section: tuple[str, ...] = ()

        # Active AoT paths map to their most recently opened entry.
        self._active_aot_entries: dict[tuple[str, ...], AoTEntry] = {}
        self.current_owner_aot_entry: AoTEntry | None = None
        self.current_owner_path: tuple[str, ...] | None = None

    def enter_header(
        self, path: tuple[str, ...], kind: _HeaderKind, *, at: int
    ) -> AoTEntry | None:
        """Validate a ``[H]`` / ``[[H]]`` header.

        Returns the opened ``AoTEntry`` for ``[[H]]``, otherwise ``None``.
        """
        path_kinds = self._path_kinds
        active_aot_entries = self._active_aot_entries
        # ``owner`` becomes the deepest active AoT path among the
        # ancestor prefixes (visited shortest-first): once a prefix is
        # itself active, every longer prefix is owned by it, so one
        # forward pass finds each prefix's owner directly.
        owner: tuple[str, ...] | None = None
        for i in range(1, len(path)):
            prefix = path[:i]
            prefix_kind = path_kinds.get(prefix)
            if prefix_kind == "value":
                joined = ".".join(prefix)
                msg = f"cannot use {joined!r} as a table: already defined as a value"
                raise self._error(msg, at=at)
            if prefix_kind is None:
                self._record_path(prefix, "implicit", owner)
            if prefix in active_aot_entries:
                owner = prefix
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
            self._record_path(path, "explicit", owner)
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
            self._record_path(path, "aot", owner)
            new_entry = AoTEntry()
            active_aot_entries[path] = new_entry
            owner = path  # a fresh AoT entry owns itself and its subtree

        self.current_section = path
        self.current_owner_path = owner
        self.current_owner_aot_entry = (
            active_aot_entries[owner] if owner is not None else None
        )
        return new_entry

    def record_keyvalue(
        self, key_path: tuple[str, ...], value: Value, *, at: int
    ) -> None:
        section = self.current_section
        full = section + key_path if section else key_path
        path_kinds = self._path_kinds
        current_kind = path_kinds.get(full)
        if current_kind == "value":
            msg = f"duplicate key {'.'.join(full)!r}"
            raise self._error(msg, at=at)
        if current_kind is not None:
            msg = f"key {'.'.join(full)!r} already defined as a table"
            raise self._error(msg, at=at)
        # A value/dotted-key path is always within the current section's
        # own tree, so its owner (if any) is just the current section's
        # owner, already resolved.
        owner = self.current_owner_path
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
                self._record_path(sub, "dotted", owner)
        self._record_path(full, "value", owner)
        if isinstance(value, InlineTableValue):
            self._register_inline_table(value, abs_prefix=full, owner=owner)

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
        abs_prefix: tuple[str, ...],
        owner: tuple[str, ...] | None,
    ) -> None:
        """Record the absolute paths an inline table's entries bind.

        Only reachable for a table that is itself addressable: an inline
        table nested in an *array* is an element, not a key, so nothing
        below it has an absolute path and there is nothing to record.
        """
        for entry in table.items:
            path = entry.key_path
            full = abs_prefix + path
            self._record_path(full, "value", owner)
            for i in range(1, len(path)):
                self._record_path(abs_prefix + path[:i], "dotted", owner)
            if isinstance(entry.value, InlineTableValue):
                self._register_inline_table(entry.value, abs_prefix=full, owner=owner)

    def _record_path(
        self,
        path: tuple[str, ...],
        kind: _PathKind,
        owner: tuple[str, ...] | None,
    ) -> None:
        """Record ``path``'s kind, filing it under ``owner`` if it's new.

        ``owner`` is the active AoT path (if any) that ``path`` falls
        under, supplied by the caller -- see :meth:`enter_header` and
        :meth:`record_keyvalue` for how each finds it in O(1).
        """
        previous = self._path_kinds.get(path)
        assert (
            previous is None
            or previous == kind
            or (previous == "implicit" and kind in ("dotted", "explicit"))
        )
        if previous is None and owner is not None:
            self._aot_subpaths.setdefault(owner, []).append(path)
        self._path_kinds[path] = kind

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
