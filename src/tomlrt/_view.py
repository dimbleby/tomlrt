"""The base shared by the three live views over a document.

`Container`, `AoT` and `Array` all present a Python view onto part of a
parsed document. Traversals in :mod:`tomlrt._layout_ops` need to
recognise one and descend it; depending on this module rather than on
those three classes keeps such a walk free of deferred imports, because
this module imports nothing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeGuard

if TYPE_CHECKING:
    from collections.abc import Iterable

    from tomlrt._array import Array
    from tomlrt._container import Container, Document


class _View:
    """Base for a live view; the subclasses own all the storage.

    ``_layout_root`` — the document a view reads and writes through, or
    ``None`` when detached — and ``_inline`` are declared here so a walk
    can ask a view about its document and its shape without first
    narrowing it to a concrete class, and the accessors derived from
    them are shared. The storage itself stays on each subclass: a
    slotted base would clash with the ``dict`` / ``list`` they also
    inherit from.
    """

    __slots__ = ()

    _layout_root: Document | None

    _inline: bool
    """Whether this view is an inline *value* rather than section-backed.

    Only `Container` varies per instance, so only `Container` spends a
    slot on it; `Array` and `AoT` answer with a class attribute, which
    their ``__slots__`` also make read-only.
    """

    @property
    def _attached(self) -> bool:
        """True iff this view is wired to a user-visible document.

        Attached means the layout root is a real document: not ``None``
        (factory mode) and not a private orphan root.
        """
        lr = self._layout_root
        return lr is not None and not lr._is_private  # noqa: SLF001

    @property
    def _doc_newline(self) -> str:
        r"""The active newline of the owning document, or ``"\n"`` if detached."""
        lr = self._layout_root
        return lr._newline if lr is not None else "\n"  # noqa: SLF001

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


def is_inline_value(v: object) -> TypeGuard[Container | Array]:
    """True iff ``v`` is an inline value view — an `Array` or inline `Table`."""
    return isinstance(v, _View) and v._inline  # noqa: SLF001
