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
    join_above_block,
    retarget_eol_newline,
    retarget_trivia_newlines,
    split_above_block,
    split_item_above,
    trivia_has_newline,
)
from tomlrt._values import (
    ArrayValue,
    InlineTableEntry,
    InlineTableValue,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tomlrt._slots import Slot
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
    t: Trivia, *, comments: bool, strip_pre_comment_ws: bool = True
) -> None:
    r"""Normalise trivia content in place.

    Always strips trailing whitespace on blank lines (``  \n`` →
    ``\n``).  When ``strip_pre_comment_ws`` is true (the default),
    also strips any whitespace that precedes a CommentNode within
    a trivia line (column indent of orphan / attached comments —
    canonical column 0).  Pass ``strip_pre_comment_ws=False`` for
    EOL-style trivia where the space between the previous token and
    the ``#`` is the structural separator, set by the caller.

    When ``comments`` is true also rewrites each CommentNode's text
    via :func:`_canon_comment_text`.

    Newline text itself is untouched here; callers run
    :func:`retarget_trivia_newlines` separately.
    """
    pieces = t.pieces
    new: list[TriviaPiece] = []
    line: list[TriviaPiece] = []
    for p in pieces:
        if isinstance(p, NewlineNode):
            while line and isinstance(line[-1], WhitespaceNode):
                line.pop()
            new.extend(line)
            new.append(p)
            line = []
            continue
        if isinstance(p, CommentNode):
            if strip_pre_comment_ws:
                while line and isinstance(line[-1], WhitespaceNode):
                    line.pop()
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


def _split_lines(pieces: list[TriviaPiece]) -> list[list[TriviaPiece]]:
    """Group pieces into lines terminated by NewlineNode (or the tail)."""
    out: list[list[TriviaPiece]] = []
    cur: list[TriviaPiece] = []
    for p in pieces:
        cur.append(p)
        if isinstance(p, NewlineNode):
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def _line_has_comment(line: list[TriviaPiece]) -> bool:
    return any(isinstance(p, CommentNode) for p in line)


def _line_has_newline(line: list[TriviaPiece]) -> bool:
    return any(isinstance(p, NewlineNode) for p in line)


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
    lines = _split_lines(list(slot.leading.pieces))

    # Trailing column-indent line (no newline, no comment).
    if lines and not _line_has_newline(lines[-1]) and not _line_has_comment(lines[-1]):
        # Drop it entirely — canonical layout is unindented.
        lines.pop()

    # Split off the head-blank region (run of newline-only lines at the start).
    head_count = 0
    while head_count < len(lines) and not _line_has_comment(lines[head_count]):
        if not _line_has_newline(lines[head_count]):
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

    _canon_multi_line_items(items, nl=nl, indent=item_indent)
    head_pad = (
        Trivia([NewlineNode(nl), WhitespaceNode(item_indent)]) if items else Trivia()
    )
    final_pad = Trivia(
        [NewlineNode(nl), WhitespaceNode(parent_indent)]
        if parent_indent
        else [NewlineNode(nl)]
    )
    v.header_trivia = _replace_pad(v.header_trivia, head_pad)
    v.final_trivia = _replace_pad(v.final_trivia, final_pad)
    # Single-line shaping produces only empty / single-space trivia
    # (no newlines, no comments), so the finalise pass would be a
    # no-op there.
    _finalise_inline_trivia(v, nl=nl, comments=comments)


def _replace_pad(t: Trivia, new_pad: Trivia) -> Trivia:
    """Substitute the bracket-pad of ``t`` while preserving its above-block."""
    _, above = split_above_block(t)
    return join_above_block(new_pad, above)


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
    v: ArrayValue | InlineTableValue, *, nl: str, comments: bool
) -> None:
    """Retarget newlines + canonicalise comment / blank-WS text across ``v``.

    Run after shape canonicalisation; touches the bracket-pad trivia
    and every per-item trivia channel.
    """
    retarget_trivia_newlines(v.header_trivia, nl)
    retarget_trivia_newlines(v.final_trivia, nl)
    _canon_trivia_text(v.header_trivia, comments=comments)
    _canon_trivia_text(v.final_trivia, comments=comments)
    for it in _entries_of(v):
        retarget_trivia_newlines(it.leading, nl)
        retarget_trivia_newlines(it.trailing, nl)
        retarget_trivia_newlines(it.post_comma_trivia, nl)
        _canon_trivia_text(it.leading, comments=comments)
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
) -> None:
    r"""Shape-preserving multi-line canonicalisation of per-item trivia.

    For each item k:
      - k == 0: ``leading`` stays empty (item 0's structural pad
        lives in ``header_trivia`` per the canonical model).
      - k >= 1: split into (head_pad, above, tail_pad) via
        :func:`split_item_above`; replace head/tail with canonical
        ``\n`` + indent; preserve ``above`` (an above-item comment
        block, possibly empty).

    All items get ``has_comma=True`` (trailing-comma idiom).
    ``post_comma_trivia`` shrinks to just an EOL section (`" #
    comment\n"`) if a comment was attached, otherwise empty.
    """
    for k, it in enumerate(items):
        if k == 0:
            it.leading = Trivia()
        else:
            _, above, _ = split_item_above(it.leading)
            canon_pad = Trivia([NewlineNode(nl), WhitespaceNode(indent)])
            it.leading = join_above_block(canon_pad, above)
        it.trailing = Trivia()
        it.has_comma = True
        _canon_post_comma_trivia(it, nl=nl)


def _canon_post_comma_trivia(item: ArrayItem | InlineTableEntry, *, nl: str) -> None:
    # Look at the first non-WS piece: if it's a comment, keep it as an EOL.
    first = next(
        (p for p in item.post_comma_trivia.pieces if not isinstance(p, WhitespaceNode)),
        None,
    )
    if isinstance(first, CommentNode):
        item.post_comma_trivia = Trivia([WhitespaceNode(" "), first, NewlineNode(nl)])
    else:
        item.post_comma_trivia = Trivia()


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


def _in_subtree(slot: Slot, path: tuple[str, ...]) -> bool:
    """True iff ``slot`` belongs to the subtree rooted at ``path``."""
    if isinstance(slot, KVSlot):
        return slot.host_path[: len(path)] == path
    assert isinstance(slot, StructuralHeaderSlot)
    return slot.path[: len(path)] == path


def format_subtree(
    *,
    start: Slot | None,
    path: tuple[str, ...],
    nl: str,
    comments: bool,
) -> None:
    """Canonicalise every slot in the subtree rooted at ``path``.

    Walks the doc-stream linked list starting at ``start`` forward,
    stopping at the first slot outside the subtree.

    The first slot's leading head-blanks are preserved verbatim
    (they belong to the document preamble or to the parent of the
    subtree).  Subsequent slots' leading head-blanks are rewritten
    to the canonical count: 1 blank line before any structural
    header, 0 otherwise.
    """
    first = True
    slot = start
    while slot is not None and _in_subtree(slot, path):
        if isinstance(slot, KVSlot):
            _canon_kv_slot(slot, nl=nl, comments=comments)
        elif isinstance(slot, StructuralHeaderSlot):
            _canon_header_slot(slot, nl=nl, comments=comments)
        target: int | None = (
            None if first else (1 if isinstance(slot, StructuralHeaderSlot) else 0)
        )
        _canon_leading(slot, nl=nl, target_blanks=target, comments=comments)
        first = False
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
