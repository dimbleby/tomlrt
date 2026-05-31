"""Inline-table mutation primitives.

Inline tables are decoupled from the doc-stream linked list: a top-
level inline table is wrapped by a single `KVSlot` whose `value` is
an `InlineTableValue`. Mutation of the inline-table contents is a
local operation on the `InlineTableValue.items` list, plus a
matching `dict.__setitem__` / `__delitem__` on the logical view.
This module owns the trivia fixups required to keep the result a
valid, nicely-spaced inline table:

* `append_entry` — splice a new entry at the end, transferring
  the prior closing space to the new entry's trailing and giving
  the previous entry a comma + a single space after it.
* `replace_entry_value` — overwrite the `value` field of the entry
  matching the given logical key (no spacing changes).
* `delete_entry` — remove the entry, then if the deleted entry was
  last, fold the prior entry's post-comma trivia into its trailing
  and clear its comma so we don't render a trailing comma (illegal
  in TOML 1.0; allowed in 1.1 but not what we want by default for a
  delete).

All entry lookups walk up the inline-table chain (via `_parent`) to
the outermost inline-table that owns the backing
`InlineTableValue` — entries for dotted keys like ``{a.b = 1}`` are
filed there, with multi-component `key_parts`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tomlrt._format import (
    _put_eol,
    _take_eol,
    append_to_comma_value,
    detect_style,
    remove_head_from_comma_value,
    remove_tail_from_comma_value,
)
from tomlrt._kind import _Kind
from tomlrt._trivia import (
    Trivia,
    join_above_block,
    split_above_block,
    split_item_above,
    strip_trailing_indent,
)
from tomlrt._values import (
    InlineTableEntry,
    make_keyparts,
    value_is_multiline,
)

if TYPE_CHECKING:
    from tomlrt._container import Container
    from tomlrt._values import InlineTableValue, Value


def _outermost_inline(t: Container) -> Container:
    """Walk up `_parent` until reaching the inline table that owns `_value`."""
    cur = t
    while cur._kind is _Kind.INLINE_DOTTED_INNER:  # noqa: SLF001
        parent = cur._parent  # noqa: SLF001
        if parent is None:  # pragma: no cover -- invariant
            msg = "internal: inline-dotted-inner without _parent"
            raise AssertionError(msg)
        cur = parent
    if cur._kind is not _Kind.INLINE_ROOT:  # noqa: SLF001  # pragma: no cover -- invariant
        msg = (
            f"internal: inline-table chain reached {cur._kind.name}; "  # noqa: SLF001
            "expected INLINE_ROOT"
        )
        raise AssertionError(msg)
    return cur


def _entry_key_path(t: Container, leaf: str) -> tuple[str, ...]:
    """Full dotted path used as `key_parts` in the outermost inline value."""
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
    """Indices of entries whose `key_parts` start with `key_path`.

    Used when deleting a synthetic dotted-prefix container — e.g.
    ``del obj["a"]`` for ``{a.b = 1, a.c = 2}`` removes both entries.
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


def replace_entry_value(t: Container, key: str, new_value: Value) -> bool:
    """Replace the value of an existing entry in place.

    Returns True iff an entry was found and replaced. No trivia is
    altered.
    """
    root = _outermost_inline(t)
    iv = root._value  # noqa: SLF001
    assert iv is not None
    found = _find_entry(iv, _entry_key_path(t, key))
    if found is None:
        return False
    _, entry = found
    entry.value = new_value
    return True


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
    style = detect_style(iv, multiline_flag=False)
    append_to_comma_value(iv, new_entry, style, root._doc_newline)  # noqa: SLF001


def delete_entry(t: Container, key: str) -> bool:
    """Remove the entry (or all dotted-prefix entries) matching `key`.

    Returns True iff at least one entry was removed. When ``key`` names
    a synthetic dotted-prefix container (e.g. ``a`` in
    ``{a.b = 1, a.c = 2}``), every entry whose ``key_parts`` start with
    that prefix is removed.
    """
    root = _outermost_inline(t)
    iv = root._value  # noqa: SLF001
    assert iv is not None
    full_path = _entry_key_path(t, key)

    # Single exact match (the common, leaf-key case).
    found = _find_entry(iv, full_path)
    if found is not None:
        idx, removed = found
        new_first_above = _capture_new_first_above(iv, removed_indices={idx})
        iv.items.pop(idx)
        _fix_tail_after_delete(iv, idx, removed, root._doc_newline)  # noqa: SLF001
        if idx == 0 and iv.items:
            remove_head_from_comma_value(iv, new_first_above)
        if not iv.items:
            strip_trailing_indent(iv.header_trivia, iv.final_trivia)
        return True

    # Prefix delete: dotted-prefix container.
    indices = _find_prefix_entries(iv, full_path)
    if not indices:
        return False
    original_len = len(iv.items)
    last_removed_idx = indices[-1]
    last_removed_entry = iv.items[last_removed_idx]
    first_removed_was_head = indices[0] == 0
    new_first_above = (
        _capture_new_first_above(iv, removed_indices=set(indices))
        if first_removed_was_head
        else Trivia()
    )
    for i in reversed(indices):
        iv.items.pop(i)
    # Tail fixup: only if the original tail was actually removed.
    if last_removed_idx == original_len - 1:
        _fix_tail_after_delete(
            iv,
            len(iv.items),
            last_removed_entry,
            root._doc_newline,  # noqa: SLF001
        )
    if first_removed_was_head and iv.items:
        remove_head_from_comma_value(iv, new_first_above)
    if not iv.items:
        strip_trailing_indent(iv.header_trivia, iv.final_trivia)
    return True


def _capture_new_first_above(
    iv: InlineTableValue, *, removed_indices: set[int]
) -> Trivia:
    """Snapshot the above-block of the entry that will become entries[0].

    Called *before* removing ``removed_indices`` from ``iv.items`` when
    index 0 is among them. Returns the above-item block of the smallest
    surviving index > 0; empty when no such survivor exists.
    """
    if 0 not in removed_indices:
        return Trivia()
    for k in range(1, len(iv.items)):
        if k not in removed_indices:
            _head, above, _tail = split_item_above(iv.items[k].leading)
            return above
    return Trivia()


def _fix_tail_after_delete(
    iv: InlineTableValue,
    removed_idx: int,
    removed: InlineTableEntry,
    nl: str,
) -> None:
    """Promote a new tail after deleting the trailing entry.

    Thin wrapper around :func:`remove_tail_from_comma_value` that
    bails out when the deletion was not of the original tail (in
    which case the existing tail is unchanged and no fix-up is
    needed). All comma / EOL / row-break logic lives in the shared
    helper, alongside the matching :func:`append_to_comma_value`.
    """
    if not iv.items or removed_idx != len(iv.items):
        return
    remove_tail_from_comma_value(
        iv,
        nl,
        removed_had_comma=removed.has_comma,
        is_multiline=value_is_multiline(iv),
    )


def reorder_inline(c: Container, new_key_order: list[str]) -> None:
    """Reorder direct children of an inline-table container.

    Permutes ``InlineTableValue.items`` so that the c-direct-child
    keys appear in ``new_key_order``. The above-block and the EOL
    section (the row-attached ``# comment`` line) both travel with the
    entry. Purely positional state — the inter-entry pad, ``has_comma``,
    and any structural ws in ``trailing`` / ``post_comma_trivia`` —
    stays at its position, so e.g. the entry that ends up at the last
    position correctly drops its trailing comma.

    Foreign entries (whose key path doesn't start with c's prefix —
    only possible when c is a dotted-inner navigator) keep their
    absolute positions; only owned positions are reordered.

    ``new_key_order`` is trusted to be a permutation of
    ``dict.keys(c)``. Only mutates the CST; dict storage is the
    caller's responsibility.
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

    if len(owned_positions) <= 1:
        return

    # Snapshot positional state (stays at position) and entry-attached
    # state (travels with the entry) for each owned position.
    # entries[0]'s above-block lives in iv.header_trivia under the
    # canonical model.
    pos_state: dict[int, tuple[Trivia, bool, Trivia, Trivia]] = {}
    above_by_entry: dict[int, Trivia] = {}
    eol_by_entry: dict[int, Trivia] = {}
    for i in owned_positions:
        e = iv.items[i]
        src = iv.header_trivia if i == 0 else e.leading
        pad, above = split_above_block(src)
        eol_by_entry[id(e)] = _take_eol(e)
        pos_state[i] = (pad, e.has_comma, e.post_comma_trivia, e.trailing)
        above_by_entry[id(e)] = above

    new_owned = [e for k in new_key_order for e in blocks[k]]
    new_entries = list(iv.items)
    for pos, e in zip(owned_positions, new_owned, strict=True):
        new_entries[pos] = e
    iv.items = new_entries

    # Restore positional state and re-stitch each entry's travelling
    # pieces. The EOL section is routed to whichever channel matches
    # the new position's ``has_comma``.
    for pos in owned_positions:
        e = iv.items[pos]
        pad, has_comma, post_rest, trail_rest = pos_state[pos]
        e.has_comma = has_comma
        e.post_comma_trivia = post_rest
        e.trailing = trail_rest
        _put_eol(e, eol_by_entry[id(e)])
        above = above_by_entry[id(e)]
        if pos == 0:
            iv.header_trivia = join_above_block(pad, above)
            e.leading = Trivia()
        else:
            e.leading = join_above_block(pad, above)


__all__ = ["append_entry", "delete_entry", "reorder_inline", "replace_entry_value"]
