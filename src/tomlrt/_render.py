"""Render — slot stream to source string.

Pure linear walk of the document's intrusive slot list, plus the
trailing trivia. Byte-exact for any unmodified parse.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tomlrt._container import Document


def render(doc: Document) -> str:
    out: list[str] = []
    prelude = doc._prelude  # noqa: SLF001
    if prelude:
        out.append(prelude)
    out.append(doc._preamble)  # noqa: SLF001
    slot = doc._head  # noqa: SLF001
    while slot is not None:
        out.append(slot.render())
        slot = slot._next  # noqa: SLF001
    out.append(doc._trailing)  # noqa: SLF001
    return "".join(out)


__all__ = ["render"]
