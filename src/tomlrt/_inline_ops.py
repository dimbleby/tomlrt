"""Mutate inline-table entries without touching doc-stream slots.

A top-level inline table is a single ``KVSlot`` whose value is an
``InlineTableValue``; mutations edit its ``items`` plus the logical dict
view. Dotted-key entries like ``{a.b = 1}`` live in the outermost inline
root with multi-component ``key_parts``, so lookups climb ``_parent``
before splicing.

Structural layout fixes — bracket pads, comma state, EOL comments, and
trailing-comma policy — are delegated to :mod:`tomlrt._comma_ops`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tomlrt._comma_ops import (
    detect_style,
    reorder_owned,
    splice_in,
    splice_out,
)
from tomlrt._format import set_comma_value_multiline
from tomlrt._kind import _Kind
from tomlrt._trivia import (
    Trivia,
)
from tomlrt._values import (
    InlineTableEntry,
    make_keyparts,
)

if TYPE_CHECKING:
    from tomlrt._container import Container
    from tomlrt._values import InlineTableValue, Value


def _outermost_inline(t: Container) -> Container:
    """Walk up `_parent` until reaching the inline table that owns `_value`."""
    cur = t
    while cur._kind is _Kind.INLINE_DOTTED_INNER:  # noqa: SLF001
        parent = cur._parent  # noqa: SLF001
        assert parent is not None, "internal: inline-dotted-inner without _parent"
        cur = parent
    assert cur._kind is _Kind.INLINE_ROOT, (  # noqa: SLF001
        f"internal: inline-table chain reached {cur._kind.name}; "  # noqa: SLF001
        "expected INLINE_ROOT"
    )
    return cur


def _entry_key_path(t: Container, leaf: str) -> tuple[str, ...]:
    """Return the dotted path used as ``key_parts`` in the outermost value."""
    root = _outermost_inline(t)
    suffix = t._path[len(root._path) :]  # noqa: SLF001
    return (*suffix, leaf)


def _find_entry(
    iv: InlineTableValue, key_path: tuple[str, ...]
) -> tuple[int, InlineTableEntry] | None:
    for i, e in enumerate(iv.items):
        if e.key_path == key_path:
            return i, e
    return None


def _find_prefix_entries(iv: InlineTableValue, key_path: tuple[str, ...]) -> list[int]:
    """Return indices of entries whose ``key_parts`` start with ``key_path``.

    Used when deleting a synthetic dotted-prefix container, e.g.
    ``del obj["a"]`` for ``{a.b = 1, a.c = 2}``.
    """
    n = len(key_path)
    out: list[int] = []
    for i, e in enumerate(iv.items):
        if len(e.key_path) > n and e.key_path[:n] == key_path:
            out.append(i)
    return out


# ---------------------------------------------------------------------------
# Public ops
# ---------------------------------------------------------------------------


def replace_entry_value(t: Container, key: str, new_value: Value) -> None:
    """Replace the value of an existing entry in place.

    The logical inline view must already contain ``key``. No trivia is altered.
    """
    root = _outermost_inline(t)
    iv = root._value  # noqa: SLF001
    assert iv is not None
    found = _find_entry(iv, _entry_key_path(t, key))
    assert found is not None, "inline view key must have a backing entry"
    _, entry = found
    entry.value = new_value


def append_entry(t: Container, key: str, new_value: Value) -> None:
    """Append a fresh entry for `key` to the outermost inline table."""
    root = _outermost_inline(t)
    iv = root._value  # noqa: SLF001
    assert iv is not None
    key_path = _entry_key_path(t, key)

    # Sample `=` padding from any existing entry; default to ` = `.
    if iv.items:
        sample = iv.items[0]
        eq_pre = sample.pre_eq
        eq_post = sample.post_eq
    else:
        eq_pre = " "
        eq_post = " "
    new_entry = InlineTableEntry(
        leading=Trivia(),
        value=new_value,
        trailing=Trivia(),
        has_comma=False,
        post_comma_trivia=Trivia(),
        key_parts=make_keyparts(key_path),
        key_seps=["."] * (len(key_path) - 1),
        pre_eq=eq_pre,
        post_eq=eq_post,
        key_path=key_path,
    )
    style = detect_style(iv, nl=root._doc_newline)  # noqa: SLF001
    splice_in(iv, new_entry, style, root._doc_newline)  # noqa: SLF001


def overwrite_entry(t: Container, key: str, new_value: Value) -> None:
    """Replace a dotted-prefix sub-table with a single fresh entry.

    Drops every ``key.*`` entry, adds ``key``, and preserves the authored
    single-line bracket pad while the outer table is transiently empty.
    Without that, overwriting a sole-content prefix would canonicalise
    tight ``{...}`` to padded ``{ ... }``. Multi-line pads are recomputed
    by ``splice_in``.
    """
    iv = _outermost_inline(t)._value  # noqa: SLF001
    assert iv is not None
    keep_pad = (
        None if iv.is_multiline() else (iv.header_trivia.copy(), iv.final_trivia.copy())
    )
    delete_entry(t, key)
    append_entry(t, key, new_value)
    if keep_pad is not None:
        iv.header_trivia, iv.final_trivia = keep_pad


def delete_entry(t: Container, key: str) -> None:
    """Remove the entry (or all dotted-prefix entries) matching `key`.

    The logical inline view must already contain ``key``. When ``key`` names a
    synthetic dotted-prefix container (e.g. ``a`` in
    ``{a.b = 1, a.c = 2}``), every entry whose ``key_parts`` start with
    that prefix is removed.
    """
    root = _outermost_inline(t)
    iv = root._value  # noqa: SLF001
    assert iv is not None
    full_path = _entry_key_path(t, key)

    found = _find_entry(iv, full_path)
    if found is not None:
        indices: list[int] = [found[0]]
    else:
        indices = _find_prefix_entries(iv, full_path)
        assert indices, "inline view key must have backing prefix entries"
    splice_out(
        iv,
        indices,
        root._doc_newline,  # noqa: SLF001
        is_multiline=iv.is_multiline(),
    )


def reorder_inline(c: Container, new_key_order: list[str]) -> None:
    """Reorder direct children of an inline-table container.

    Direct-child-key grouping keeps dotted-prefix entries adjacent under
    their shared prefix. Foreign entries (only possible for a dotted-inner
    navigator) keep their absolute positions; only owned positions are
    reordered. ``new_key_order`` is trusted to be a permutation of
    ``dict.keys(c)``; dict storage is caller-owned.
    """
    root = _outermost_inline(c)
    iv = root._value  # noqa: SLF001
    assert iv is not None

    prefix = c._path[len(root._path) :]  # noqa: SLF001
    plen = len(prefix)

    blocks: dict[str, list[InlineTableEntry]] = {k: [] for k in new_key_order}
    owned_positions: list[int] = []
    for i, e in enumerate(iv.items):
        kp = e.key_path
        if len(kp) > plen and kp[:plen] == prefix and kp[plen] in blocks:
            blocks[kp[plen]].append(e)
            owned_positions.append(i)

    new_owned = [e for k in new_key_order for e in blocks[k]]
    reorder_owned(
        iv,
        owned_positions,
        new_owned,
        root._doc_newline,  # noqa: SLF001
        is_multiline=iv.is_multiline(),
    )


def set_inline_multiline(root: Container, *, multiline: bool, indent: str) -> None:
    """Switch an inline-table root between single-line and multi-line form.

    ``root`` must be an `INLINE_ROOT` (it owns ``_value``). Collapsing
    raises `TOMLError` if a comment would be orphaned.
    """
    iv = root._value  # noqa: SLF001
    assert iv is not None
    set_comma_value_multiline(
        iv,
        multiline=multiline,
        nl=root._doc_newline,  # noqa: SLF001
        indent=indent,
    )


def ensure_inline_multiline(c: Container) -> None:
    """Promote the inline table owning ``c`` to multi-line if it is not.

    Resolves the outermost inline root, so a dotted-inner navigator
    promotes the whole physical table (the only place a row comment can
    live). No-op when already multi-line.
    """
    root = _outermost_inline(c)
    iv = root._value  # noqa: SLF001
    assert iv is not None
    if not iv.is_multiline():
        set_inline_multiline(root, multiline=True, indent="    ")


__all__ = [
    "append_entry",
    "delete_entry",
    "ensure_inline_multiline",
    "reorder_inline",
    "replace_entry_value",
    "set_inline_multiline",
]
