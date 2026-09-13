"""Comment side-channel views for inline tables, keyed by direct leaf key.

All read / write plumbing lives in `_comma_comments`; this module supplies
only leaf-key resolution and the multi-line promotion policy.

A leaf is exposed only when it names exactly one physical entry. A
dotted-prefix leaf (``a`` in ``{a.b = 1, a.c = 2}``) names no single
entry and is absent here — descend with ``t["a"].comments["b"]``,
mirroring how ``doc.comments`` skips ``a`` for a top-level ``a.b = 1``.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if sys.version_info >= (3, 12):
    from typing import override
else:  # pragma: no cover -- backport for Python < 3.12
    from typing_extensions import override

from tomlrt._comma_comments import (
    CommaCommentAdapter,
)
from tomlrt._inline_ops import (
    _entry_key_path,
    _find_entry,
    _outermost_inline,
    ensure_inline_multiline,
)
from tomlrt._values import InlineTableEntry

if TYPE_CHECKING:
    from collections.abc import Iterator

    from tomlrt._container import Container
    from tomlrt._values import InlineTableValue


class _InlineAdapter(CommaCommentAdapter[str, InlineTableEntry]):
    __slots__ = ("_c", "_indices")

    def __init__(self, container: Container) -> None:
        self._c = container
        self._indices: dict[str, int] = {}

    @override
    def value(self) -> InlineTableValue:
        iv = _outermost_inline(self._c)._value  # noqa: SLF001
        assert iv is not None
        return iv

    @override
    def resolve(self, key: object) -> int | None:
        if not isinstance(key, str) or self._c._needs_layout:  # noqa: SLF001
            return None
        iv = self.value()
        key_path = _entry_key_path(self._c, key)
        cached = self._indices.get(key)
        if (
            cached is not None
            and cached < len(iv.items)
            and iv.items[cached].key_path == key_path
        ):
            return cached
        found = _find_entry(iv, key_path)
        if found is None:
            self._indices.pop(key, None)
            return None
        idx, _entry = found
        self._indices[key] = idx
        return idx

    @override
    def ensure_value(self, key: str, *, materialize: bool) -> bool:
        return self._c._prepare_comment_write(key, materialize=materialize)  # noqa: SLF001

    @override
    def promote(self) -> None:
        ensure_inline_multiline(self._c)

    @override
    def newline(self) -> str:
        return self._c._doc_newline  # noqa: SLF001

    @override
    def indexed_candidates(self) -> Iterator[tuple[str, int]]:
        if self._c._needs_layout:  # noqa: SLF001
            return
        iv = self.value()
        root = _outermost_inline(self._c)
        prefix = self._c._path[len(root._path) :]  # noqa: SLF001
        plen = len(prefix)
        self._indices.clear()
        seen: set[str] = set()
        for i, e in enumerate(iv.items):
            kp = e.key_path
            if len(kp) != plen + 1 or kp[:plen] != prefix:
                continue
            leaf = kp[plen]
            assert leaf not in seen, "inline table cannot contain duplicate leaves"
            seen.add(leaf)
            self._indices[leaf] = i
            yield leaf, i


__all__ = ["_InlineAdapter"]
