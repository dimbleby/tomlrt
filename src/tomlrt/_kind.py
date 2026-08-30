"""The shape a `Container` (Document or Table) is in.

This leaf module lets `_container.py` and `_inline_ops.py` share the
enum without worsening their existing circular import.
"""

from __future__ import annotations

from enum import Enum, auto


class _Kind(Enum):
    """The state a `Container` is in.

    `Container` (base of `Table` and `Document`) covers six distinct
    shapes. For inline tables, `_value` names CST ownership and `_host`
    distinguishes an unmaterialised factory from a dotted navigator:

    ============================  ========  ==========  ============  ==============
    Kind                          _inline   _value      _layout_root  _host
    ============================  ========  ==========  ============  ==============
    `DOCUMENT`                    False     None        self          None
    `SECTION` (``[a.b]``)         False     None        doc           Container
    `IMPLICIT_SECTION`            False     None        doc / None    Container / None
    `INLINE_ROOT` (``{x = 1}``)   True      InlineVal   doc / None    view / None
    `INLINE_FACTORY`              True      None        None          None
    `INLINE_DOTTED_INNER`         True      None        doc / None    Container
    ============================  ========  ==========  ============  ==============

    Attachment is deliberately not part of an inline table's kind: a
    materialised root or dotted navigator can live inside a standalone
    `Array` without belonging to a document.
    """

    DOCUMENT = auto()
    SECTION = auto()
    IMPLICIT_SECTION = auto()
    INLINE_ROOT = auto()
    INLINE_FACTORY = auto()
    INLINE_DOTTED_INNER = auto()


__all__ = ["_Kind"]
