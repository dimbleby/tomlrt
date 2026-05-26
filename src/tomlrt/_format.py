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

from typing import TYPE_CHECKING

from tomlrt._slots import KVSlot, StructuralHeaderSlot
from tomlrt._trivia import (
    CommentNode,
    NewlineNode,
    Trivia,
    WhitespaceNode,
    line_has_comment,
    line_has_newline,
    retarget_eol_newline,
    retarget_trivia_newlines,
    split_above_block,
    split_eol_section,
    split_item_above,
    split_lines,
    trivia_has_newline,
)
from tomlrt._values import (
    ArrayValue,
    InlineTableEntry,
    InlineTableValue,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tomlrt._slots import AoTEntry, Slot
    from tomlrt._trivia import (
        EolTrivia,
        TriviaPiece,
    )
    from tomlrt._values import (
        ArrayItem,
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


def _entries_of(
    v: ArrayValue | InlineTableValue,
) -> Sequence[ArrayItem | InlineTableEntry]:
    """Common accessor for the items list of an inline array / table."""
    return v.items if isinstance(v, ArrayValue) else v.entries


def _value_is_multiline(av: ArrayValue | InlineTableValue) -> bool:
    """Shape detection: any structural NewlineNode inside means multi-line."""
    if trivia_has_newline(av.header_trivia) or trivia_has_newline(av.final_trivia):
        return True
    for it in _entries_of(av):
        if (
            trivia_has_newline(it.leading)
            or trivia_has_newline(it.post_comma_trivia)
            or trivia_has_newline(it.trailing)
        ):
            return True
    return False


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
    items = _entries_of(v)
    multi = _value_is_multiline(v)
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
    items = _entries_of(v)
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


def _item_has_eol(item: ArrayItem | InlineTableEntry) -> bool:
    """True if the item carries an inline EOL comment.

    When the item has a comma, the EOL section lives in
    ``post_comma_trivia``; otherwise it lives in ``trailing``.
    """
    target = item.post_comma_trivia if item.has_comma else item.trailing
    eol, _rest = split_eol_section(target)
    return bool(eol.pieces)


def _normalise_row_breaks(
    items: list[ArrayItem] | list[InlineTableEntry],
    value: ArrayValue | InlineTableValue,
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


def _take_eol(item: ArrayItem | InlineTableEntry) -> Trivia:
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


def _put_eol(item: ArrayItem | InlineTableEntry, eol: Trivia) -> None:
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


def _inner_space(v: ArrayValue | InlineTableValue) -> Trivia:
    """Bracket-inner padding for a single-line inline value.

    Inline tables wear ``{ a = 1 }`` with one space of inner
    padding; inline arrays wear ``[a, b]`` with none.  Each call
    produces a fresh ``Trivia`` so callers can assign it to
    ``header_trivia`` and ``final_trivia`` independently.
    """
    if isinstance(v, InlineTableValue) and v.entries:
        return Trivia([WhitespaceNode(" ")])
    return Trivia()


def _canon_single_line_inline(v: ArrayValue | InlineTableValue) -> None:
    v.header_trivia = _inner_space(v)
    v.final_trivia = _inner_space(v)
    items = _entries_of(v)
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
    for it in _entries_of(v):
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
    items: Sequence[ArrayItem | InlineTableEntry],
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


def _canon_post_comma_trivia(item: ArrayItem | InlineTableEntry, *, nl: str) -> bool:
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


__all__ = [
    "_canon_eol",
    "_canon_header_slot",
    "_canon_inline_value",
    "_canon_kv_slot",
    "_canon_leading",
    "_canon_value",
    "format_document_trailing",
    "format_subtree",
]
