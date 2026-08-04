"""The base shared by the three live views over a document.

`Container`, `AoT` and `Array` all present a Python view onto part of a
parsed document. Traversals in :mod:`tomlrt._layout_ops` need to
recognise one and descend it; depending on this module rather than on
those three classes keeps such a walk free of deferred imports, because
this module imports nothing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from tomlrt._container import Document


class _View:
    """Base for a live view; the subclasses own all the storage.

    ``_layout_root`` — the document a view reads and writes through, or
    ``None`` when detached — is declared here so a walk can ask a view
    about its document without first narrowing it to a concrete class.
    The slot itself stays on each subclass: a slotted base would clash
    with the ``dict`` / ``list`` they also inherit from.
    """

    __slots__ = ()

    _layout_root: Document | None

    def _view_children(self) -> Iterable[object]:
        """The values directly held by this view.

        Mapping views yield their values, sequence views their items;
        either way a walk descends into whichever of those are
        themselves views.
        """
        raise NotImplementedError

    def _reset_displaced(self) -> None:
        """Detach this view because its backing CST is going away.

        A no-op for views whose attachment is rebuilt by the caller
        (an `AoT`'s entries are re-rooted wholesale), overridden by
        those that must forget the value they resolve against.
        """
