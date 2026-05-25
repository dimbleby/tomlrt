"""Trivia primitives.

A `Trivia` is an ordered run of `TriviaPiece`s — whitespace,
newlines, and comments. Trivia hangs off slots and value nodes;
together with the `lexeme`/`raw` of value-bearing nodes the trivia
captures every byte of the source so the document round-trips
exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True, eq=False)
class WhitespaceNode:
    """Run of spaces and/or tabs (no newlines)."""

    text: str

    def render(self) -> str:
        return self.text


@dataclass(slots=True, eq=False)
class NewlineNode:
    r"""A single line terminator (``\n`` or ``\r\n``)."""

    text: str

    def render(self) -> str:
        return self.text


@dataclass(slots=True, eq=False)
class CommentNode:
    """A ``# ...`` comment, *not* including the trailing newline."""

    text: str  # includes the leading '#'

    def render(self) -> str:
        return self.text


TriviaPiece = WhitespaceNode | NewlineNode | CommentNode


@dataclass(slots=True, eq=False)
class Trivia:
    """An ordered run of trivia pieces."""

    pieces: list[TriviaPiece] = field(default_factory=list)

    def render(self) -> str:
        pieces = self.pieces
        if not pieces:
            return ""
        if len(pieces) == 1:
            return pieces[0].render()
        return "".join([p.render() for p in pieces])


def trivia_has_comment(t: Trivia) -> bool:
    """True iff ``t`` contains any ``CommentNode`` piece."""
    return any(isinstance(p, CommentNode) for p in t.pieces)


def trivia_has_newline(t: Trivia) -> bool:
    """True iff ``t`` contains any ``NewlineNode`` piece."""
    return any(isinstance(p, NewlineNode) for p in t.pieces)


def retarget_trivia_newlines(t: Trivia, target: str) -> None:
    """Rewrite every ``NewlineNode.text`` in ``t`` to ``target``.

    Used by graft paths (cross-document clone / inline-value
    deepcopy) so freshly-spliced slots adopt the destination
    document's line ending instead of preserving the source's. A
    no-op when every newline already matches.
    """
    for p in t.pieces:
        if isinstance(p, NewlineNode) and p.text != target:
            p.text = target


def retarget_eol_newline(eol: EolTrivia, target: str) -> None:
    """Rewrite ``eol.newline.text`` to ``target`` (if present)."""
    if eol.newline is not None and eol.newline.text != target:
        eol.newline.text = target


def clone_trivia(t: Trivia) -> Trivia:
    """Return a shallow copy of ``t`` (same pieces, fresh list)."""
    return Trivia(list(t.pieces))


def split_above_block(t: Trivia) -> tuple[Trivia, Trivia]:
    """Split ``t`` into ``(pad, above)``.

    ``above`` is the item-attached comment block and ``pad`` is the
    structural padding that surrounds it (opening newline + value
    indent).

    Used by inline-array / inline-table mutators to relocate an
    above-item comment block from one item's leading slot to another's
    when the boundary item changes (``insert(0)``, ``del[0]``, sort,
    reverse, slice assignment at index 0, append onto an array whose
    ``final_trivia`` carries an above-`]` comment block).

    Two parts are returned, but they are *not* a simple concatenation:
    the comment block lives between the opening newline (a single
    ``NewlineNode`` from ``pad``) and the trailing value indent (the
    rest of ``pad``). To reconstruct, splice ``above`` between
    ``pad[0]`` and ``pad[1:]`` (use :func:`join_above_block`).

    By construction ``above`` is the run of pieces strictly between
    the first ``NewlineNode`` and the trailing whitespace immediately
    preceding nothing (or the end of trivia). ``above`` is empty iff
    no ``CommentNode`` appears in that run.
    """
    pieces = t.pieces
    first_nl = -1
    for i, p in enumerate(pieces):
        if isinstance(p, NewlineNode):
            first_nl = i
            break
    if first_nl < 0:
        return Trivia(list(pieces)), Trivia()
    # Trailing-WS run (zero or one piece).
    tail_start = len(pieces)
    if tail_start > first_nl + 1 and isinstance(pieces[tail_start - 1], WhitespaceNode):
        tail_start -= 1
    middle = pieces[first_nl + 1 : tail_start]
    # ``above`` is non-empty only if a CommentNode appears in middle.
    if not any(isinstance(p, CommentNode) for p in middle):
        return Trivia(list(pieces)), Trivia()
    pad = Trivia(list(pieces[: first_nl + 1]) + list(pieces[tail_start:]))
    above = Trivia(list(middle))
    return pad, above


def join_above_block(pad: Trivia, above: Trivia) -> Trivia:
    """Splice ``above`` back into ``pad``.

    ``above`` is inserted between ``pad[0]`` (opening NL) and the rest
    of ``pad`` (value indent). Inverse of :func:`split_above_block`.
    """
    pieces = list(pad.pieces)
    if not pieces:
        return Trivia(list(above.pieces))
    return Trivia([pieces[0], *above.pieces, *pieces[1:]])


def strip_trailing_indent(t: Trivia) -> None:
    r"""Drop the bracket-pad indent that anchored the now-removed first item.

    Used after deleting the last item / entry of a multi-line inline
    array or inline table from ``header_trivia``: the per-item indent
    has no item left to anchor and would otherwise render as
    ``[<indent>\\n]`` / ``{<indent>\\n}``.

    * If the trivia contains a ``CommentNode`` (a bracket-EOL comment
      glued to the opening bracket) only the trailing
      ``WhitespaceNode`` run is dropped — the comment and its
      terminator newline stay so the comment doesn't swallow the
      closing bracket.
    * Otherwise the entire trailing ``WhitespaceNode`` / ``NewlineNode``
      run is dropped, restoring the canonical empty form (``[\\n]``
      with ``header_trivia`` empty and ``final_trivia`` holding the
      single newline).
    """
    has_comment = any(isinstance(p, CommentNode) for p in t.pieces)
    if has_comment:
        while t.pieces and isinstance(t.pieces[-1], WhitespaceNode):
            t.pieces.pop()
    else:
        while t.pieces and isinstance(t.pieces[-1], (WhitespaceNode, NewlineNode)):
            t.pieces.pop()


def split_item_above(t: Trivia) -> tuple[Trivia, Trivia, Trivia]:
    """Split an item-leading region into ``(head_pad, above, tail_pad)``.

    Distinct from :func:`split_above_block` — which is for the bracket
    pad (``header_trivia`` / ``final_trivia``) and treats a pre-NL
    comment as an EOL glued to the bracket. This splitter is for
    ``items[i].leading`` (i >= 1) in an inline array or inline table,
    where there is no bracket and the leading NL may be absent
    because item ``i-1``'s EOL hoisted it onto its
    ``post_comma_trivia``.

    ``head_pad`` is the leading NL (or empty); ``tail_pad`` is the
    trailing value-indent WS (or empty); ``above`` is the comment
    block between them (typically ``[WS, Comment, NL] * n``).
    """
    rest = list(t.pieces)
    head: list[TriviaPiece] = []
    if rest and isinstance(rest[0], NewlineNode):
        head = [rest.pop(0)]
    tail: list[TriviaPiece] = []
    if rest and isinstance(rest[-1], WhitespaceNode):
        tail = [rest.pop()]
    return Trivia(head), Trivia(rest), Trivia(tail)


def split_eol_section(t: Trivia) -> tuple[Trivia, Trivia]:
    """Split ``t`` into the inline EOL section and the structural rest.

    Used to canonicalise post-comma trivia in inline arrays and inline
    tables.  The "EOL section" is the row-attached part: any inline
    whitespace, an EOL comment, and the terminating newline of that
    comment row.  Anything beyond — additional newlines, indent,
    above-item comment blocks — is structural and belongs to the
    *next* item's leading.

    If no EOL comment is present on the comma's row, the whole input
    is structural and the EOL half is empty.

    Note: when no comment is present, ``t`` itself is returned as the
    structural half — callers must treat it as owned by the result and
    must not retain or mutate the original.
    """
    pieces = t.pieces
    n = len(pieces)
    j = 0
    while j < n and isinstance(pieces[j], WhitespaceNode):
        j += 1
    if j >= n or not isinstance(pieces[j], CommentNode):
        return Trivia(), t
    end = j + 1
    if end < n and isinstance(pieces[end], NewlineNode):
        end += 1
    return Trivia(pieces[:end]), Trivia(pieces[end:])


@dataclass(slots=True, eq=False)
class EolTrivia:
    """End-of-line tail of a single physical line.

    Used by `KVSlot` and `StructuralHeaderSlot` to capture the
    optional inline comment plus the line terminator. ``newline``
    may be ``None`` only for the last line of a file with no
    final newline.
    """

    trailing_ws: WhitespaceNode | None  # whitespace before any comment / newline
    comment: CommentNode | None
    newline: NewlineNode | None

    def render(self) -> str:
        out: list[str] = []
        if self.trailing_ws is not None:
            out.append(self.trailing_ws.text)
        if self.comment is not None:
            out.append(self.comment.text)
        if self.newline is not None:
            out.append(self.newline.text)
        return "".join(out)


__all__ = [
    "CommentNode",
    "EolTrivia",
    "NewlineNode",
    "Trivia",
    "TriviaPiece",
    "WhitespaceNode",
    "clone_trivia",
    "join_above_block",
    "retarget_eol_newline",
    "retarget_trivia_newlines",
    "split_above_block",
    "split_eol_section",
    "split_item_above",
    "trivia_has_comment",
    "trivia_has_newline",
]
