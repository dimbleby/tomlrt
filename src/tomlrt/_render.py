"""Render — slot stream to source string.

Pure linear walk of the document's intrusive slot list, plus the
trailing trivia. Byte-exact for any unmodified parse.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tomlrt._container import Document
    from tomlrt._slots import Slot


def render(doc: Document) -> str:
    """Render ``doc``: its envelope around its slot run."""
    return render_run(
        doc._prelude,  # noqa: SLF001
        doc._preamble,  # noqa: SLF001
        doc._head,  # noqa: SLF001
        doc._trailing,  # noqa: SLF001
    )


def render_run(prelude: str, preamble: str, head: Slot | None, trailing: str) -> str:
    """Render a slot run inside the four pieces of trivia around it.

    All a rendered document is: a `Document` names these, and a run
    synthesised for its text alone can be handed them directly rather
    than have one built to hold them.
    """
    out: list[str] = []
    if prelude:
        out.append(prelude)
    out.append(preamble)
    slot = head
    while slot is not None:
        out.append(slot.render())
        slot = slot._next  # noqa: SLF001
    out.append(trailing)
    return "".join(out)


__all__ = ["render", "render_run"]
