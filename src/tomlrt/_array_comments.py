"""Comment side-channel views for `Array`, keyed by item index.

All read / write plumbing lives in `_comma_comments`; this module supplies
only the integer keying and the multi-line promotion policy.
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
from tomlrt._values import ArrayItem

if TYPE_CHECKING:
    from collections.abc import Iterator

    from tomlrt._array import Array
    from tomlrt._values import ArrayValue


class _ArrayAdapter(CommaCommentAdapter[int, ArrayItem]):
    __slots__ = ("_arr",)

    def __init__(self, arr: Array) -> None:
        self._arr = arr

    @override
    def value(self) -> ArrayValue:
        return self._arr._value  # noqa: SLF001

    @override
    def resolve(self, key: object) -> int | None:
        if not isinstance(key, int) or isinstance(key, bool):
            msg = f"Array comment indices must be int, not {type(key).__name__}"
            raise TypeError(msg)
        items = self._arr._value.items  # noqa: SLF001
        n = len(items)
        idx = key if key >= 0 else key + n
        return idx if 0 <= idx < n else None

    @override
    def promote(self) -> None:
        if not self._arr.multiline:
            self._arr.set_multiline(multiline=True)

    @override
    def newline(self) -> str:
        return self._arr._doc_newline  # noqa: SLF001

    @override
    def indexed_candidates(self) -> Iterator[tuple[int, int]]:
        return ((i, i) for i in range(len(self._arr._value.items)))  # noqa: SLF001


__all__ = ["_ArrayAdapter"]
