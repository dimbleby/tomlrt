"""The shape a `Container` (Document or Table) is in.

This leaf module lets `_container.py` and `_inline_ops.py` share the
enum without worsening their existing circular import.
"""

from __future__ import annotations

from enum import Enum, auto


class _Kind(Enum):
    """The state a `Container` is in.

    `Container` (base of `Table` and `Document`) covers six distinct
    shapes, each picked out by a combination of `_inline`, `_value`,
    `_layout_root`, `_header_ref`, and whether the instance is the
    `Document` itself. The combinations are:

    ============================  ========  ==========  ============  ==============
    Kind                          _inline   _value      _layout_root  _header_ref
    ============================  ========  ==========  ============  ==============
    `DOCUMENT`                    False     None        self          None
    `SECTION` (``[a.b]``)         False     None        doc           SlotRef
    `IMPLICIT_SECTION`            False     None        doc           None
    `INLINE_ROOT` (``{x = 1}``)   True      InlineVal   doc           None
    `INLINE_FACTORY`              True      None        None          None
    `INLINE_DOTTED_INNER`         True      None        doc           None
    ============================  ========  ==========  ============  ==============

    `INLINE_FACTORY` and `INLINE_DOTTED_INNER` share their flag
    pattern except for `_layout_root`; this enum names the
    distinction so callers can dispatch on intent rather than
    re-deriving the discriminator.
    """

    DOCUMENT = auto()
    SECTION = auto()
    IMPLICIT_SECTION = auto()
    INLINE_ROOT = auto()
    INLINE_FACTORY = auto()
    INLINE_DOTTED_INNER = auto()


__all__ = ["_Kind"]
