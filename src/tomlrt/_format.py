"""Container / Array formatter — canonicalise layout without losing comments.

Pure functions over slots / trivia / values. Idempotent and
shape-preserving for inline values (single-line stays single-line;
multi-line stays multi-line). Used by the public
``Container.format`` / ``Array.format`` methods.

Canonical layout enforced here:

KV slot
    pre_eq=" ", post_eq=" ", key_seps=".", strip column-indent WS,
    EOL trailing_ws = " " before any comment.

Section / AoT-entry header
    inner_pre="", inner_post="", strip column-indent WS, EOL as for KV.

Sibling spacing (within the subtree of the container being formatted)
    Between body KVs of the same container: 0 blank lines.
    Between sibling sections / AoT entries / between body and next
    section: exactly 1 blank line.

Inline arrays / inline tables
    Single-line: ``[a, b, c]`` / ``{ a = 1, b = 2 }``.
    Multi-line: each item on its own line, last item carries a
    trailing comma. Indent inherited from the first item if present
    else two spaces.

Orphan comment blocks (``# …`` runs separated from the slot/header
by a blank line) and EOL / leading-attached comments are preserved
in place — comment text is rewritten to ``# body`` form when
``comments=True``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

from tomlrt._slots import KVSlot, StructuralHeaderSlot
from tomlrt._trivia import (
    CommentNode,
    NewlineNode,
    Trivia,
    WhitespaceNode,
    indent_from_final_trivia,
    join_above_block,
    line_has_comment,
    line_has_newline,
    restamp_bracket_pad_for_first,
    retarget_eol_newline,
    retarget_trivia_newlines,
    split_above_block,
    split_eol_section,
    split_item_above,
    split_lines,
    strip_trailing_indent,
)
from tomlrt._values import (
    ArrayValue,
    InlineTableEntry,
    InlineTableValue,
    inter_item_separator,
    value_is_multiline,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tomlrt._slots import AoTEntry, Slot
    from tomlrt._trivia import (
        EolTrivia,
        TriviaPiece,
    )
    from tomlrt._values import (
        CommaItem,
        CommaValue,
        Value,
    )


# ---------------------------------------------------------------------------
# Comment text + line cleanup
# ---------------------------------------------------------------------------


def _canon_comment_text(text: str) -> str:
    """Rewrite a comment lexeme to canonical ``# body`` / ``#`` form.

    Input always starts with ``#``.  Normalises any leading whitespace
    inside the body to exactly one space; strips trailing whitespace.
    An empty-body comment collapses to a bare ``#``.
    """
    assert text.startswith("#"), text
    body = text[1:].rstrip().lstrip(" \t")
    return "# " + body if body else "#"


def _canon_trivia_text(
    t: Trivia,
    *,
    comments: bool,
    strip_pre_comment_ws: bool = True,
    comment_indent: str = "",
    first_line_is_eol: bool = False,
) -> None:
    r"""Normalise trivia content in place.

    Always strips trailing whitespace on blank lines (``  \n`` →
    ``\n``).  When ``strip_pre_comment_ws`` is true (the default),
    also strips any whitespace that precedes a CommentNode within
    a trivia line. If ``comment_indent`` is non-empty, restamps the
    pre-comment whitespace to that indent string instead of stripping
    to column 0 — used inside multi-line inline arrays / tables where
    full-line comments sit at the array's element indent. Pass
    ``strip_pre_comment_ws=False`` for EOL-style trivia where the
    space between the previous token and the ``#`` is the structural
    separator, set by the caller.

    When ``first_line_is_eol`` is true, the pre-first-NL run is
    treated as an EOL context (``strip_pre_comment_ws=False``, no
    ``comment_indent``), and only subsequent lines use the full-line
    rules. Used for ``header_trivia`` of multi-line inline values,
    where the opening bracket may carry a row-attached EOL comment.

    When ``comments`` is true also rewrites each CommentNode's text
    via :func:`_canon_comment_text`.

    Newline text itself is untouched here; callers run
    :func:`retarget_trivia_newlines` separately.
    """
    pieces = t.pieces
    new: list[TriviaPiece] = []
    line: list[TriviaPiece] = []
    in_eol = first_line_is_eol
    for p in pieces:
        if isinstance(p, NewlineNode):
            while line and isinstance(line[-1], WhitespaceNode):
                line.pop()
            new.extend(line)
            new.append(p)
            line = []
            in_eol = False
            continue
        if isinstance(p, CommentNode):
            if strip_pre_comment_ws and not in_eol:
                while line and isinstance(line[-1], WhitespaceNode):
                    line.pop()
                if comment_indent:
                    line.append(WhitespaceNode(comment_indent))
            if comments:
                p.text = _canon_comment_text(p.text)
            new.extend(line)
            new.append(p)
            line = []
            continue
        line.append(p)
    new.extend(line)
    t.pieces = new


# ---------------------------------------------------------------------------
# Leading trivia of slots
# ---------------------------------------------------------------------------


def _canon_leading(
    slot: Slot,
    *,
    nl: str,
    target_blanks: int | None,
    comments: bool,
) -> None:
    """Rewrite ``slot.leading`` to canonical form.

    The leading is decomposed into three regions:

    * ``head_blanks`` — leading whitespace-only lines (structural
      blank gap from the previous slot)
    * ``middle`` — everything else *except* the slot's own column
      indent (orphan blank-separated groups + the attached comment
      run)
    * ``indent`` — the slot's column-indent line (trailing
      whitespace-only piece with no terminating newline)

    Canonical form: ``target_blanks`` blank lines at the head,
    middle preserved verbatim (with newline / comment text
    normalised), no indent at the tail.

    When ``target_blanks`` is ``None`` the head-blank region is
    preserved verbatim (used for the first slot in a document /
    subtree, where the preceding gap is owned by something outside
    the subtree).
    """
    lines = split_lines(list(slot.leading.pieces))

    # Trailing column-indent line (no newline, no comment).
    if lines and not line_has_newline(lines[-1]) and not line_has_comment(lines[-1]):
        # Drop it entirely — canonical layout is unindented.
        lines.pop()

    # Split off the head-blank region (run of newline-only lines at the start).
    head_count = 0
    while head_count < len(lines) and not line_has_comment(lines[head_count]):
        if not line_has_newline(lines[head_count]):
            break
        head_count += 1
    middle = lines[head_count:]

    # Normalise newline / comment / blank-line WS in the middle.
    middle_t = Trivia([p for line in middle for p in line])
    retarget_trivia_newlines(middle_t, nl)
    _canon_trivia_text(middle_t, comments=comments)

    # Pick the head-blank count:
    #
    # * ``target_blanks is None`` — preamble / subtree boundary,
    #   preserve the user's count verbatim.
    # * ``middle`` non-empty — comment block carries the user's
    #   separation intent; clamp the head blank to 0 or 1 matching
    #   whether the source had a blank above the comment block.
    # * otherwise — apply the structural ``target_blanks`` exactly.
    if target_blanks is None:
        n_blanks = head_count
    elif middle:
        n_blanks = min(head_count, 1)
    else:
        n_blanks = target_blanks
    head_t = Trivia([NewlineNode(nl)] * n_blanks)

    slot.leading.pieces = [*head_t.pieces, *middle_t.pieces]


# ---------------------------------------------------------------------------
# EOL
# ---------------------------------------------------------------------------


def _canon_eol(eol: EolTrivia, *, nl: str, comments: bool) -> None:
    """Normalise an :class:`EolTrivia`.

    * Newline retargeted to ``nl`` (preserved as ``None`` for the
      last line of a no-final-newline file).
    * ``trailing_ws`` → ``None`` when no comment; ``" "`` (one
      space) before a comment.
    * Comment text rewritten when ``comments`` is true.
    """
    retarget_eol_newline(eol, nl)
    if eol.comment is None:
        eol.trailing_ws = None
        return
    if comments:
        eol.comment.text = _canon_comment_text(eol.comment.text)
    eol.trailing_ws = WhitespaceNode(" ")


# ---------------------------------------------------------------------------
# Key parts
# ---------------------------------------------------------------------------


def _canon_key_equals(node: KVSlot | InlineTableEntry) -> None:
    """Canonicalise the key / ``=`` body of a KV slot or inline-table entry."""
    node.pre_eq = " "
    node.post_eq = " "
    node.key_seps = ["."] * (len(node.key_parts) - 1)


# ---------------------------------------------------------------------------
# Slots — KV / Header
# ---------------------------------------------------------------------------


def _canon_kv_slot(slot: KVSlot, *, nl: str, comments: bool) -> None:
    """Normalise a KV slot's body (key/eq/value/eol).

    Leading is handled separately by :func:`_canon_leading` so that
    the subtree-aware blank-line policy can be applied at the walk
    level.
    """
    _canon_key_equals(slot)
    _canon_value(slot.value, nl=nl, comments=comments)
    _canon_eol(slot.eol, nl=nl, comments=comments)


def _canon_header_slot(slot: StructuralHeaderSlot, *, nl: str, comments: bool) -> None:
    slot.inner_pre = ""
    slot.inner_post = ""
    slot.key_seps = ["."] * (len(slot.key_parts) - 1)
    _canon_eol(slot.eol, nl=nl, comments=comments)


# ---------------------------------------------------------------------------
# Inline values
# ---------------------------------------------------------------------------


def _canon_inline_value(
    v: ArrayValue | InlineTableValue,
    *,
    nl: str,
    comments: bool,
    parent_indent: str = "",
) -> None:
    """Canonicalise the layout of an inline array or inline table.

    Recurses into nested inline values, then dispatches to the
    single-line or multi-line shape, then (for multi-line) runs a
    final text pass over the value's trivia tree.
    """
    items = v.items
    multi = value_is_multiline(v)
    item_indent = parent_indent + "  " if multi else parent_indent

    for it in items:
        if isinstance(it, InlineTableEntry):
            _canon_key_equals(it)
        _canon_value(it.value, nl=nl, comments=comments, parent_indent=item_indent)

    if not multi:
        _canon_single_line_inline(v)
        return

    _canon_multiline_shape(
        v, nl=nl, comments=comments, item_indent=item_indent, outer_indent=parent_indent
    )


def _canon_multiline_shape(
    v: ArrayValue | InlineTableValue,
    *,
    nl: str,
    comments: bool,
    item_indent: str,
    outer_indent: str,
) -> None:
    """Apply multi-line canonical shape to ``v``.

    Shared backbone of the ``format()`` walk (``_canon_inline_value``)
    and ``Array.set_multiline(multiline=True, ...)``. Canonicalises
    per-item trivia, restamps the bracket-pad, then runs the final
    text pass that retargets newlines and rewrites comment text.

    The single-line path bypasses this entirely — its shaping
    produces only empty / single-space trivia (no newlines, no
    comments), so the finalise pass would be a no-op there.
    """
    items = v.items
    last_line_open = _canon_multi_line_items(items, nl=nl, indent=item_indent)
    if items:
        head_eol, _ = split_eol_section(v.header_trivia)
        _, head_above = split_above_block(v.header_trivia)
        v.header_trivia = _compose_pad(
            head_eol=head_eol,
            above=head_above,
            nl=nl,
            trailing_indent=item_indent,
        )
        _, final_above = split_above_block(v.final_trivia)
        v.final_trivia = _compose_pad(
            head_eol=Trivia(),
            above=final_above,
            nl=nl,
            trailing_indent=outer_indent,
            line_already_open=last_line_open,
        )
        final_eol_first = False
    else:
        # An empty multi-line value carries all of its trivia
        # (bracket-EOL + above-block + closing pad) in final_trivia;
        # header_trivia is empty by construction.
        final_eol, _ = split_eol_section(v.final_trivia)
        _, final_above = split_above_block(v.final_trivia)
        v.header_trivia = Trivia()
        v.final_trivia = _compose_pad(
            head_eol=final_eol,
            above=final_above,
            nl=nl,
            trailing_indent=outer_indent,
        )
        final_eol_first = bool(final_eol.pieces)
    _finalise_inline_trivia(
        v,
        nl=nl,
        comments=comments,
        item_indent=item_indent,
        final_first_line_is_eol=final_eol_first,
    )


def _compose_pad(
    *,
    head_eol: Trivia,
    above: Trivia,
    nl: str,
    trailing_indent: str,
    line_already_open: bool = False,
) -> Trivia:
    r"""Compose a bracket-pad from (row-attached EOL, above-block, indent).

    Layout: ``head_eol`` (already terminated by ``\n`` when
    non-empty), an optional structural ``\n`` to start the fresh
    line, the above-block, then a trailing indent (column of the
    first item, or of the closing bracket).

    Skip the structural ``\n`` when ``head_eol`` supplies one, or
    when ``line_already_open`` says the upstream context (the
    previous item's ``post_comma_trivia``) already terminated the
    line.
    """
    pieces: list[TriviaPiece] = list(head_eol.pieces)
    if not head_eol.pieces and not line_already_open:
        pieces.append(NewlineNode(nl))
    pieces.extend(above.pieces)
    if trailing_indent:
        pieces.append(WhitespaceNode(trailing_indent))
    return Trivia(pieces)


def _item_has_eol(item: CommaItem) -> bool:
    """True if the item carries an inline EOL comment.

    When the item has a comma, the EOL section lives in
    ``post_comma_trivia``; otherwise it lives in ``trailing``.
    """
    target = item.post_comma_trivia if item.has_comma else item.trailing
    eol, _rest = split_eol_section(target)
    return bool(eol.pieces)


def _normalise_row_breaks(
    items: Sequence[CommaItem],
    value: CommaValue[_CV_ItemT],
    nl: str,
    *,
    multiline: bool,
) -> None:
    """Enforce the one-row-break-per-row invariant.

    In a multi-line inline value each item occupies one physical row;
    each row needs exactly one terminating newline. That newline lives
    either in the row's EOL section (``items[i].post_comma_trivia`` or
    ``items[i].trailing``) — or in the next row's leading region
    (``items[i+1].leading`` for inter-item gaps, ``value.final_trivia``
    for the gap before the closing bracket).

    This helper restores the invariant after a mutation has touched
    item count, comma state, or EOL state — call it once at the end.
    Idempotent.
    """
    if not multiline:
        return
    # Inter-item: items[i].leading must start with NL iff items[i-1]
    # carries no EOL.
    for i in range(1, len(items)):
        pred = items[i - 1]
        pieces = items[i].leading.pieces
        has_nl = bool(pieces) and isinstance(pieces[0], NewlineNode)
        if _item_has_eol(pred) and has_nl:
            items[i].leading = Trivia(pieces[1:])
        elif not _item_has_eol(pred) and not has_nl:
            items[i].leading = Trivia([NewlineNode(text=nl), *pieces])
    # Closing-bracket gap: final_trivia must start with NL iff the
    # last item carries no EOL.
    if not items:
        return
    ft = value.final_trivia
    has_nl = bool(ft.pieces) and isinstance(ft.pieces[0], NewlineNode)
    last_has_eol = _item_has_eol(items[-1])
    if last_has_eol and has_nl:
        ft.pieces = list(ft.pieces[1:])
    elif not last_has_eol and not has_nl:
        ft.pieces = [NewlineNode(text=nl), *ft.pieces]


def _take_eol(item: CommaItem) -> Trivia:
    """Split out and return the item's row-attached EOL section.

    The EOL section lives in ``post_comma_trivia`` when the item has
    a comma, and in ``trailing`` otherwise. On return the item holds
    only the structural rest in that channel.
    """
    if item.has_comma:
        eol, rest = split_eol_section(item.post_comma_trivia)
        item.post_comma_trivia = rest
    else:
        eol, rest = split_eol_section(item.trailing)
        item.trailing = rest
    return eol


def _put_eol(item: CommaItem, eol: Trivia) -> None:
    """Append a previously-taken EOL section onto the item.

    Routes to ``post_comma_trivia`` or ``trailing`` according to the
    item's *current* ``has_comma``, which may differ from the value
    at extraction time.
    """
    if not eol.pieces:
        return
    if item.has_comma:
        item.post_comma_trivia = Trivia([*item.post_comma_trivia.pieces, *eol.pieces])
    else:
        item.trailing = Trivia([*item.trailing.pieces, *eol.pieces])


def flip_to_internal(item: CommaItem) -> None:
    """Make ``item`` look like an internal (non-last) item.

    Under the canonical model the inter-item separator lives in the
    NEXT item's leading; this function only ensures the comma is set
    and carries any EOL comment across the channel flip (trailing →
    post_comma_trivia) so the comma stays immediately after the
    value and the comment after the comma. No-op if the item already
    has a comma.

    Shared between the inline-array append path and the inline-table
    append path, both of which transition a previously-terminal item
    into an internal item when a new item is being appended.
    """
    if item.has_comma:
        return
    eol = _take_eol(item)
    item.has_comma = True
    _put_eol(item, eol)


def _inner_space(v: ArrayValue | InlineTableValue) -> Trivia:
    """Bracket-inner padding for a single-line inline value.

    Inline tables wear ``{ a = 1 }`` with one space of inner
    padding; inline arrays wear ``[a, b]`` with none.  Each call
    produces a fresh ``Trivia`` so callers can assign it to
    ``header_trivia`` and ``final_trivia`` independently.
    """
    if isinstance(v, InlineTableValue) and v.items:
        return Trivia([WhitespaceNode(" ")])
    return Trivia()


def _canon_single_line_inline(v: ArrayValue | InlineTableValue) -> None:
    v.header_trivia = _inner_space(v)
    v.final_trivia = _inner_space(v)
    items = v.items
    n = len(items)
    for k, it in enumerate(items):
        it.leading = Trivia() if k == 0 else Trivia([WhitespaceNode(" ")])
        it.trailing = Trivia()
        it.post_comma_trivia = Trivia()
        it.has_comma = k < n - 1


def _finalise_inline_trivia(
    v: ArrayValue | InlineTableValue,
    *,
    nl: str,
    comments: bool,
    item_indent: str = "",
    final_first_line_is_eol: bool = False,
) -> None:
    """Retarget newlines + canonicalise comment / blank-WS text across ``v``.

    Run after shape canonicalisation; touches the bracket-pad trivia
    and every per-item trivia channel. ``item_indent`` is the column
    at which full-line comments inside ``v`` should sit; passed by
    multi-line callers so above-item comment blocks are indented to
    align with the items, not stripped to column 0.

    ``final_first_line_is_eol`` flags the empty-value case where the
    row-attached EOL on the opening bracket is stored in
    ``final_trivia`` (because there is no item to delimit it from
    the closing bracket), so its first line must be canonicalised as
    an EOL context, not a full-line one.
    """
    retarget_trivia_newlines(v.header_trivia, nl)
    retarget_trivia_newlines(v.final_trivia, nl)
    _canon_trivia_text(
        v.header_trivia,
        comments=comments,
        comment_indent=item_indent,
        first_line_is_eol=True,
    )
    _canon_trivia_text(
        v.final_trivia,
        comments=comments,
        comment_indent=item_indent,
        first_line_is_eol=final_first_line_is_eol,
    )
    for it in v.items:
        retarget_trivia_newlines(it.leading, nl)
        retarget_trivia_newlines(it.trailing, nl)
        retarget_trivia_newlines(it.post_comma_trivia, nl)
        _canon_trivia_text(it.leading, comments=comments, comment_indent=item_indent)
        # ``trailing`` and ``post_comma_trivia`` are EOL contexts:
        # the space before ``#`` is the structural separator the
        # callers have already canonicalised to one space.
        _canon_trivia_text(it.trailing, comments=comments, strip_pre_comment_ws=False)
        _canon_trivia_text(
            it.post_comma_trivia, comments=comments, strip_pre_comment_ws=False
        )


def _canon_multi_line_items(
    items: Sequence[CommaItem],
    *,
    nl: str,
    indent: str,
) -> bool:
    r"""Shape-preserving multi-line canonicalisation of per-item trivia.

    Returns whether the last item's ``post_comma_trivia`` closed its
    line (a row-attached EOL comment was preserved): the caller uses
    this to decide whether ``final_trivia`` should start with a fresh
    newline.

    For each item k:
      - k == 0: ``leading`` stays empty (item 0's structural pad
        lives in ``header_trivia`` per the canonical model).
      - k >= 1: split into (head_pad, above, tail_pad) via
        :func:`split_item_above`; replace head/tail with canonical
        ``\n`` + indent; preserve ``above`` (an above-item comment
        block, possibly empty).  The leading ``\n`` is suppressed
        when item ``k-1``'s ``post_comma_trivia`` already closed
        the line (typically because it carries an EOL comment).

    All items get ``has_comma=True`` (trailing-comma idiom). When an
    item had no comma in source, its EOL comment may live in
    ``trailing``; that section is migrated into ``post_comma_trivia``
    so the synthesised comma sits between value and comment. After
    migration ``trailing`` is cleared and ``post_comma_trivia``
    shrinks to just an EOL section (`" # comment\n"`) when a comment
    is attached, otherwise empty.
    """
    prev_line_open = False
    for k, it in enumerate(items):
        if k == 0:
            it.leading = Trivia()
        else:
            _, above, _ = split_item_above(it.leading)
            it.leading = _compose_pad(
                head_eol=Trivia(),
                above=above,
                nl=nl,
                trailing_indent=indent,
                line_already_open=prev_line_open,
            )
        # Synthesising a comma may shift the EOL row from ``trailing``
        # to ``post_comma_trivia``; the take/put pair preserves the
        # comment across any has_comma flip.
        eol = _take_eol(it)
        it.has_comma = True
        it.trailing = Trivia()
        _put_eol(it, eol)
        prev_line_open = _canon_post_comma_trivia(it, nl=nl)
    return prev_line_open


def _canon_post_comma_trivia(item: CommaItem, *, nl: str) -> bool:
    r"""Canonicalise ``post_comma_trivia``; return True if the line was closed.

    If the first non-whitespace piece is a comment it is preserved as
    an EOL row (``" # comment\\n"``) and we report the line as
    closed; otherwise the channel is cleared and the line stays open
    for the next item's leading pad to terminate.
    """
    first = next(
        (p for p in item.post_comma_trivia.pieces if not isinstance(p, WhitespaceNode)),
        None,
    )
    if isinstance(first, CommentNode):
        item.post_comma_trivia = Trivia([WhitespaceNode(" "), first, NewlineNode(nl)])
        return True
    item.post_comma_trivia = Trivia()
    return False


# ---------------------------------------------------------------------------
# Value dispatch
# ---------------------------------------------------------------------------


def _canon_value(v: Value, *, nl: str, comments: bool, parent_indent: str = "") -> None:
    if isinstance(v, (ArrayValue, InlineTableValue)):
        _canon_inline_value(v, nl=nl, comments=comments, parent_indent=parent_indent)
    # Other value kinds carry no formattable trivia.


# ---------------------------------------------------------------------------
# Subtree walk and orchestration
# ---------------------------------------------------------------------------


def _in_subtree(
    slot: Slot,
    path: tuple[str, ...],
    owner: AoTEntry | None,
) -> bool:
    """True iff ``slot`` belongs to the subtree rooted at ``path`` / ``owner``.

    When ``owner`` is set the caller is an AoT-entry view (or a section
    nested inside one): the path-prefix check is not enough because
    sibling entries of the same AoT share the caller's path. Restrict
    the walk to slots whose nearest AoT-entry ancestor is ``owner``
    itself or a descendant entry (strictly longer path prefix-matching
    ``owner.path``).
    """
    if isinstance(slot, KVSlot):
        if slot.host_path[: len(path)] != path:
            return False
    else:
        assert isinstance(slot, StructuralHeaderSlot)
        if slot.path[: len(path)] != path:
            return False
    if owner is None:
        return True
    slot_owner = slot.owner_aot_entry
    if slot_owner is owner:
        return True
    if slot_owner is None:
        return False
    op = owner.path
    return len(slot_owner.path) > len(op) and slot_owner.path[: len(op)] == op


def format_subtree(
    *,
    start: Slot | None,
    path: tuple[str, ...],
    owner: AoTEntry | None,
    nl: str,
    comments: bool,
) -> None:
    """Canonicalise every slot in the subtree rooted at ``path``.

    Walks the doc-stream linked list starting at ``start`` forward,
    stopping at the first slot outside the subtree.

    ``owner`` is the AoT entry that anchors the subtree, when the
    caller is an AoT-entry view or a non-AoT section nested inside
    one. It disambiguates sibling AoT entries that share ``path``.

    The first slot's leading head-blanks are preserved verbatim
    (they belong to the document preamble or to the parent of the
    subtree).  Subsequent slots' leading head-blanks are rewritten
    to the canonical count: 1 blank line before any structural
    header, 0 otherwise.
    """
    prev: Slot | None = None
    slot = start
    while slot is not None and _in_subtree(slot, path, owner):
        # A slot parsed as the file's final line carries
        # ``eol.newline=None``.  If a later mutation (sort, splice)
        # moved it off the tail, restore the terminator so the
        # canonical inter-slot blank line materialises.  The walk's
        # genuinely-final slot is never visited as ``prev``, so its
        # no-final-newline state survives.
        if prev is not None:
            assert isinstance(prev, (KVSlot, StructuralHeaderSlot))
            if prev.eol.newline is None:
                prev.eol.newline = NewlineNode(nl)
        if isinstance(slot, KVSlot):
            _canon_kv_slot(slot, nl=nl, comments=comments)
        elif isinstance(slot, StructuralHeaderSlot):
            _canon_header_slot(slot, nl=nl, comments=comments)
        if prev is None:
            target: int | None = None
        else:
            target = 1 if isinstance(slot, StructuralHeaderSlot) else 0
        _canon_leading(slot, nl=nl, target_blanks=target, comments=comments)
        prev = slot
        slot = slot._next  # noqa: SLF001


def format_document_trailing(
    trailing: Trivia,
    *,
    nl: str,
    comments: bool,
) -> None:
    """Canonicalise the trailing trivia of a :class:`Document`.

    Retargets newlines to ``nl``, strips trailing whitespace on
    blank lines, strips column-indent whitespace before orphan
    comments, and (when ``comments`` is true) rewrites comment
    text to canonical ``# body`` form.

    Structural shape — line count, blank-line placement — is
    preserved, so the user-visible split between preamble and
    epilogue is unaffected.
    """
    retarget_trivia_newlines(trailing, nl)
    _canon_trivia_text(trailing, comments=comments)


# ---------------------------------------------------------------------------
# Comma-separated value style detection + append orchestration
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class CommaStyle:
    """Inferred layout policy for an inline array or inline table.

    Used by both the inline-array and inline-table append paths to
    decide where the new item goes, how the previous-last is flipped
    to internal, and whether the new last carries a trailing comma.
    """

    is_multiline: bool
    inter_separator: Trivia
    trailing_comma: bool
    trailing_post: Trivia


_CV_ItemT = TypeVar("_CV_ItemT", bound="CommaItem")


def detect_style(
    value: ArrayValue | InlineTableValue | None, *, multiline_flag: bool
) -> CommaStyle:
    """Infer a :class:`CommaStyle` for ``value`` (or for a fresh value).

    Sample-bounded under the canonical model: multi-line-ness shows up
    in ``header_trivia`` / ``final_trivia`` / ``items[k>=1].leading``,
    so we don't need to walk every item. For a single-item multi-line
    value we synthesise the inter-item separator from the bracket-pad
    indent (no peer to sample from).
    """
    if value is None:
        return CommaStyle(
            is_multiline=multiline_flag,
            inter_separator=Trivia([WhitespaceNode(text=" ")]),
            trailing_comma=multiline_flag,
            trailing_post=Trivia(),
        )
    items = value.items
    is_multiline = multiline_flag or value_is_multiline(value)
    if is_multiline and len(items) < 2:
        nl_text = "\n"
        for p in value.header_trivia.pieces:
            if isinstance(p, NewlineNode):
                nl_text = p.text
                break
        else:
            for p in value.final_trivia.pieces:
                if isinstance(p, NewlineNode):
                    nl_text = p.text
                    break
        indent = _first_indent_after_newline(value.header_trivia)
        if not indent:
            indent = indent_from_final_trivia(value.final_trivia) or "    "
        inter_sep = Trivia([NewlineNode(text=nl_text), WhitespaceNode(text=indent)])
    else:
        inter_sep = inter_item_separator(items)
    trailing_comma = items[-1].has_comma if items else is_multiline
    pad_ft, _above_ft = split_above_block(value.final_trivia)
    trailing_post = pad_ft if pad_ft.pieces else value.final_trivia.copy()
    if not items and is_multiline and not trailing_post.pieces:
        nl_text = "\n"
        trailing_post = Trivia([NewlineNode(text=nl_text)])
    return CommaStyle(
        is_multiline=is_multiline,
        inter_separator=inter_sep,
        trailing_comma=trailing_comma,
        trailing_post=trailing_post,
    )


def _first_indent_after_newline(trivia: Trivia) -> str:
    pieces = trivia.pieces
    for i, p in enumerate(pieces):
        if (
            isinstance(p, NewlineNode)
            and i + 1 < len(pieces)
            and isinstance(pieces[i + 1], WhitespaceNode)
        ):
            return str(pieces[i + 1].text)
    return ""


def migrate_bracket_above(bracket: Trivia, separator: Trivia) -> tuple[Trivia, Trivia]:
    """Migrate any above-bracket comment block onto a new item's leading.

    An above-block in ``header_trivia`` / ``final_trivia`` (the
    structural row(s) between the bracket and the next/last item)
    conceptually belongs to the item below it. When a new boundary
    item is being inserted or appended, the block migrates from the
    bracket pad onto that item's leading.

    Returns ``(new_bracket, new_leading)``.
    """
    pad, above = split_above_block(bracket)
    return pad, join_above_block(separator, above)


def flip_to_terminal(item: CommaItem, style: CommaStyle) -> None:
    """Make ``item`` look like the terminal (last) item per style."""
    if style.trailing_comma:
        if not item.has_comma:
            eol = _take_eol(item)
            item.has_comma = True
            _put_eol(item, eol)
        # When has_comma==True, post_comma_trivia carries any EOL the
        # parser/mutation already filed there; keep it intact.
        return
    # No trailing comma policy: drop the comma; carry any EOL back
    # to trailing.
    if item.has_comma:
        eol = _take_eol(item)
        item.has_comma = False
        item.post_comma_trivia = Trivia()
        _put_eol(item, eol)


def append_to_comma_value(
    cv: CommaValue[_CV_ItemT],
    new_item: _CV_ItemT,
    style: CommaStyle,
    nl: str,
) -> None:
    """Append ``new_item`` to ``cv`` honouring ``style``.

    Shared orchestration for inline arrays and inline tables. The
    caller is responsible for constructing ``new_item`` of the
    appropriate concrete type (``ArrayItem`` for an array,
    ``InlineTableEntry`` for an inline table) and for computing the
    appropriate ``style`` (typically via :func:`detect_style`).

    Empty case: reframes the bracket pad so the new item sits on its
    own canonical row, then flips it to terminal-per-style.

    Non-empty case: migrates any above-bracket comment block onto the
    new item's leading, flips the previous-last to internal, appends,
    flips the new item to terminal-per-style, and renormalises the
    row-break invariant.
    """
    items = cv.items
    if not items:
        cv.header_trivia, cv.final_trivia = restamp_bracket_pad_for_first(
            cv.final_trivia
        )
        items.append(new_item)
        flip_to_terminal(new_item, style)
        return
    cv.final_trivia, new_item.leading = migrate_bracket_above(
        cv.final_trivia, style.inter_separator
    )
    flip_to_internal(items[-1])
    items.append(new_item)
    flip_to_terminal(new_item, style)
    _normalise_row_breaks(items, cv, nl, multiline=style.is_multiline)


def remove_tail_from_comma_value(
    cv: CommaValue[_CV_ItemT],
    nl: str,
    *,
    removed_had_comma: bool,
    is_multiline: bool,
) -> None:
    """Re-terminalise the new last item after a tail removal.

    Shared between :class:`tomlrt._array.Array`'s tail-delete and
    :mod:`tomlrt._inline_ops`'s ``_fix_tail_after_delete``. Both face
    the same transition: the previous tail has been spliced out, and
    the item that was internal is now the new tail.

    Steps (mirroring :func:`append_to_comma_value` in reverse):
      * take the entry-attached EOL section off the new tail (it
        currently lives in ``post_comma_trivia`` because the entry
        was internal);
      * adopt the removed entry's trailing-comma policy on the new
        tail and clear its now-stale ``post_comma_trivia``;
      * put the EOL section back, routed to ``trailing`` or
        ``post_comma_trivia`` per the new ``has_comma`` state;
      * renormalise the row-break invariant.

    No-ops when ``cv.items`` is empty (the caller's ``strip_trailing_indent``
    or equivalent handles the no-item canonical form).
    """
    items = cv.items
    if not items:
        return
    new_last = items[-1]
    eol = _take_eol(new_last)
    new_last.has_comma = removed_had_comma
    new_last.post_comma_trivia = Trivia()
    _put_eol(new_last, eol)
    _normalise_row_breaks(items, cv, nl, multiline=is_multiline)


def remove_head_from_comma_value(
    cv: CommaValue[_CV_ItemT],
    new_first_above: Trivia,
) -> None:
    """Migrate the above-block of the new first item into ``header_trivia``.

    After deleting the original head item(s), the item that is now
    ``items[0]`` was previously internal; its ``leading`` still
    carries the inter-item separator plus any above-item comment
    block that conceptually belonged above it. Under the canonical
    model ``items[0].leading`` is always empty and the above-block
    above the first item lives in ``header_trivia``. This helper
    completes the migration: the caller has already extracted the
    above-block (before deletion, via :func:`split_item_above` on
    the soon-to-be-new-first item's leading); we splice it into
    ``header_trivia`` and reset ``items[0].leading``.

    Shared between :class:`tomlrt._array.Array`'s head-delete and
    :mod:`tomlrt._inline_ops`'s delete machinery.
    """
    if not cv.items:
        return
    head_pad, _drop = split_above_block(cv.header_trivia)
    cv.header_trivia = join_above_block(head_pad, new_first_above)
    cv.items[0].leading = Trivia()


def remove_owned_from_comma_value(
    cv: CommaValue[_CV_ItemT],
    removed_indices: Sequence[int],
    nl: str,
    *,
    is_multiline: bool,
    strip_when_empty: bool = True,
) -> None:
    """Splice out ``removed_indices`` and run the boundary fixups.

    Shared orchestration for inline-array tail/head/internal removal
    (``Array.__delitem__``) and inline-table key / dotted-prefix
    removal (``_inline_ops.delete_entry``). The caller is responsible
    for the kind-specific logic that *finds* the indices to remove
    (slice-vs-int normalisation for arrays, key / prefix lookup for
    inline tables) and for any associated logical-view bookkeeping
    (decoded ``list`` / ``dict`` removal).

    Steps:
      * snapshot the above-block of the soon-to-be-new-first item
        when position 0 is among ``removed_indices`` (so the
        canonical "above-block of item 0 lives in header_trivia"
        invariant can be restored after the splice);
      * capture the about-to-be-removed tail's ``has_comma`` policy
        (so the new tail can adopt it);
      * pop the removed indices from ``cv.items`` in reverse order;
      * if no items remain, call :func:`strip_trailing_indent` to
        canonicalise the empty bracket pad (unless
        ``strip_when_empty`` is False — used by callers that own a
        different empty-canonicalisation policy);
      * else dispatch to :func:`remove_head_from_comma_value` /
        :func:`remove_tail_from_comma_value` /
        :func:`_normalise_row_breaks` as appropriate.

    ``removed_indices`` must be a list of valid distinct indices into
    ``cv.items`` (not necessarily sorted on input; sorted internally).
    """
    if not removed_indices:
        return
    items = cv.items
    if not items:
        return
    sorted_removed = sorted(removed_indices)
    removed_set = set(sorted_removed)
    last_idx = len(items) - 1
    zero_removed = 0 in removed_set
    tail_removed = last_idx in removed_set

    new_first_above: Trivia = Trivia()
    if zero_removed:
        for k in range(1, len(items)):
            if k not in removed_set:
                _h, new_first_above, _t = split_item_above(items[k].leading)
                break
    new_terminal_has_comma = items[last_idx].has_comma if tail_removed else False

    for i in reversed(sorted_removed):
        items.pop(i)

    if not items:
        if strip_when_empty:
            strip_trailing_indent(cv.header_trivia, cv.final_trivia)
        return
    if zero_removed:
        remove_head_from_comma_value(cv, new_first_above)
    if tail_removed:
        remove_tail_from_comma_value(
            cv,
            nl,
            removed_had_comma=new_terminal_has_comma,
            is_multiline=is_multiline,
        )
    else:
        _normalise_row_breaks(items, cv, nl, multiline=is_multiline)


def reorder_comma_value_owned(
    cv: CommaValue[_CV_ItemT],
    owned_positions: Sequence[int],
    new_owned: Sequence[_CV_ItemT],
    nl: str,
    *,
    is_multiline: bool,
) -> None:
    """Reorder ``owned_positions`` within ``cv.items`` in place.

    Positional-preserve strategy shared by inline arrays (where every
    position is owned) and inline tables (where the caller may be a
    dotted-inner navigator with foreign entries interspersed). The
    split between positional state (stays at the position) and
    entry-attached state (travels with the entry) is:

      * **Positional**: structural pad (the part of ``leading`` —
        or ``header_trivia`` for position 0 — before any above-item
        comment block), ``has_comma``, ``post_comma_trivia``,
        ``trailing``.
      * **Entry-attached**: above-item comment block, row-attached
        EOL section.

    ``owned_positions`` is a strictly ascending list of indices into
    ``cv.items`` that are subject to reordering; non-owned positions
    are left untouched. ``new_owned`` is the permutation: the entry
    to place at each owned position, in the same order. Both lists
    must have equal length, and ``new_owned`` must be a permutation
    of ``cv.items[i]`` for ``i in owned_positions``.

    Bracket trivia (``header_trivia`` / ``final_trivia``) is
    untouched, so above-bracket comment blocks survive the reorder
    at their structural position. No-op for fewer than two owned
    positions.
    """
    if len(owned_positions) <= 1:
        return
    assert len(owned_positions) == len(new_owned)
    items = cv.items
    pos_state: dict[int, tuple[Trivia, bool, Trivia, Trivia]] = {}
    above_by_entry: dict[int, Trivia] = {}
    eol_by_entry: dict[int, Trivia] = {}
    for i in owned_positions:
        e = items[i]
        src = cv.header_trivia if i == 0 else e.leading
        pad, above = split_above_block(src)
        eol_by_entry[id(e)] = _take_eol(e)
        pos_state[i] = (pad, e.has_comma, e.post_comma_trivia, e.trailing)
        above_by_entry[id(e)] = above

    new_items = list(items)
    for pos, e in zip(owned_positions, new_owned, strict=True):
        new_items[pos] = e
    items[:] = new_items

    for pos in owned_positions:
        e = items[pos]
        pad, has_comma, post_rest, trail_rest = pos_state[pos]
        e.has_comma = has_comma
        e.post_comma_trivia = post_rest
        e.trailing = trail_rest
        _put_eol(e, eol_by_entry[id(e)])
        above = above_by_entry[id(e)]
        if pos == 0:
            cv.header_trivia = join_above_block(pad, above)
            e.leading = Trivia()
        else:
            e.leading = join_above_block(pad, above)
    # Restore the one-row-break-per-row invariant: an EOL section
    # ends in its own ``\n``, which would otherwise double up with
    # the structural ``\n`` on the next position's pad.
    _normalise_row_breaks(items, cv, nl, multiline=is_multiline)


__all__ = [
    "CommaStyle",
    "_canon_eol",
    "_canon_header_slot",
    "_canon_inline_value",
    "_canon_kv_slot",
    "_canon_leading",
    "_canon_value",
    "append_to_comma_value",
    "detect_style",
    "flip_to_internal",
    "flip_to_terminal",
    "format_document_trailing",
    "format_subtree",
    "migrate_bracket_above",
    "remove_head_from_comma_value",
    "remove_owned_from_comma_value",
    "remove_tail_from_comma_value",
    "reorder_comma_value_owned",
]
