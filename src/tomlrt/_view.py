"""The base shared by the three live views over a document.

`Container`, `AoT` and `Array` all present a Python view onto part of a
parsed document. Traversals need to recognise one and descend it, but
they live in :mod:`tomlrt._layout_ops`, which those three modules import
in turn — so naming the classes directly would mean resolving a deferred
import on every walk. Depending on this module instead breaks the cycle:
it imports nothing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable


class _View:
    """Marker base for a live view; carries no state of its own."""

    __slots__ = ()

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
