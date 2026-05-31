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
    _normalise_row_breaks,
    _put_eol,
    _take_eol,
)
from tomlrt._kind import _Kind
from tomlrt._trivia import (
    Trivia,
    join_above_block,
    restamp_bracket_pad_for_first,
    split_above_block,
    split_eol_section,
    strip_trailing_indent,
)
from tomlrt._values import (
    InlineTableEntry,
    inter_item_separator,
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
        key_parts=make_keyparts(key_path),
        key_seps=["."] * (len(key_path) - 1),
        pre_eq=eq_pre,
        post_eq=eq_post,
        value=new_value,
        trailing=Trivia(),
        has_comma=False,
        post_comma_trivia=Trivia(),
        key_path=key_path,
    )

    if not iv.items:
        # Empty {}: reframe the bracket pad. For a single-line empty
        # (``{}`` or ``{ }``) `final_trivia` is empty or one WS; the
        # helper mirrors it on both sides. For a multi-line empty
        # (``{\n}``) the helper splits at the trailing newline so the
        # new entry gets an indented row of its own. In the multi-line
        # case the entry also needs a trailing comma to match the
        # canonical ``{\n    k = v,\n}`` shape.
        is_multiline = value_is_multiline(iv)
        iv.header_trivia, iv.final_trivia = restamp_bracket_pad_for_first(
            iv.final_trivia
        )
        if is_multiline:
            new_entry.has_comma = True
        iv.items.append(new_entry)
        return

    # Inter-entry separator: structural pad portion of entries[1].leading
    # (mirrors :func:`tomlrt._array._detect_style`). Cloning the full
    # leading would replicate any above-entry comment block onto the
    # new entry.
    inter_sep = inter_item_separator(iv.items)
    is_multiline = value_is_multiline(iv)

    last = iv.items[-1]
    keep_trailing_comma = last.has_comma
    if not last.has_comma:
        # Mirror the inline-array policy: transfer any row-attached
        # EOL section from `trailing` (where it sits while the entry
        # is terminal-without-comma) to `post_comma_trivia` (where it
        # belongs once the entry becomes internal-with-comma). This
        # keeps the comma immediately after the value and the
        # comment after the comma — the canonical multi-line layout.
        eol = _take_eol(last)
        last.has_comma = True
        _put_eol(last, eol)
    new_entry.leading = inter_sep
    if keep_trailing_comma:
        new_entry.has_comma = True
        new_entry.post_comma_trivia = Trivia()
    iv.items.append(new_entry)
    _normalise_row_breaks(
        iv.items,
        iv,
        root._doc_newline,  # noqa: SLF001
        multiline=is_multiline,
    )


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
        iv.items.pop(idx)
        _fix_tail_after_delete(iv, idx, removed, root._doc_newline)  # noqa: SLF001
        _fix_head_after_delete(iv, idx)
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
    if first_removed_was_head:
        _fix_head_after_delete(iv, 0)
    if not iv.items:
        strip_trailing_indent(iv.header_trivia, iv.final_trivia)
    return True


def _fix_tail_after_delete(
    iv: InlineTableValue,
    removed_idx: int,
    removed: InlineTableEntry,
    nl: str,
) -> None:
    """Promote a new tail after deleting the trailing entry.

    The structural pad and comma-style come from the removed entry's
    position; the EOL section already on the new tail is entry-attached
    and must be preserved across any ``has_comma`` flip.
    """
    if not iv.items or removed_idx != len(iv.items):
        return
    new_last = iv.items[-1]
    new_last_eol = _take_eol(new_last)
    is_multiline = value_is_multiline(iv)
    new_last.has_comma = removed.has_comma
    new_last.post_comma_trivia = Trivia()
    if not removed.has_comma and not new_last.trailing.pieces:
        _, removed_trail_rest = split_eol_section(removed.trailing)
        new_last.trailing = removed_trail_rest
    _put_eol(new_last, new_last_eol)
    _normalise_row_breaks(
        iv.items,
        iv,
        nl,
        multiline=is_multiline,
    )


def _fix_head_after_delete(iv: InlineTableValue, removed_idx: int) -> None:
    """Restore canonical entries[0].leading == Trivia() after head delete.

    Under the canonical model, the bracket pad before entries[0] lives
    in ``header_trivia``; ``entries[0].leading`` is always empty. After
    deleting the head, the new head's ``leading`` (which used to be the
    inter-entry separator) becomes redundant — drop it.
    """
    if not iv.items or removed_idx != 0:
        return
    iv.items[0].leading = Trivia()


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
