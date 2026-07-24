"""Represent source-preserving whitespace, newlines, and comments.

Trivia hangs off slots and value nodes; together with value lexemes it
captures every byte needed for exact round-trips.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


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

    def copy(self) -> Trivia:
        """Return a shallow copy (same pieces, fresh list)."""
        return Trivia(list(self.pieces))


def has_comment(pieces: Iterable[TriviaPiece]) -> bool:
    """True iff ``pieces`` contains any ``CommentNode``."""
    # A plain loop avoids generator overhead; this runs on every
    # comma-value boundary during reorder and is hot enough to matter.
    for piece in pieces:  # noqa: SIM110
        if isinstance(piece, CommentNode):
            return True
    return False


def leading_has_blank_line(leading: Trivia) -> bool:
    r"""Whether ``leading`` contains at least one blank physical line.

    A blank line is a line in the leading-trivia stream that contains
    no comment piece. A comment-line newline (e.g. ``# foo\n``) does
    not count as a blank — the newline belongs to the comment.
    """
    has_comment_piece = False
    for piece in leading.pieces:
        if isinstance(piece, CommentNode):
            has_comment_piece = True
        elif isinstance(piece, NewlineNode):
            if not has_comment_piece:
                return True
            has_comment_piece = False
    return False


def has_newline(pieces: Iterable[TriviaPiece]) -> bool:
    """True iff ``pieces`` contains any ``NewlineNode``."""
    # See has_comment: a plain loop measurably beats any()+generator here.
    for piece in pieces:  # noqa: SIM110
        if isinstance(piece, NewlineNode):
            return True
    return False


def leading_break_index(pieces: Sequence[TriviaPiece]) -> int | None:
    r"""Index of a row-terminating newline at the start of ``pieces``.

    An optional leading whitespace run is skipped first, so trailing
    whitespace from the previous row (``1,  \n``) does not mask the
    break. Returns the ``NewlineNode``'s index, or ``None`` if the first
    non-whitespace piece is not a newline.
    """
    k = 0
    while k < len(pieces) and isinstance(pieces[k], WhitespaceNode):
        k += 1
    if k < len(pieces) and isinstance(pieces[k], NewlineNode):
        return k
    return None


def split_lines(pieces: Sequence[TriviaPiece]) -> list[list[TriviaPiece]]:
    """Group ``pieces`` into logical lines terminated by ``NewlineNode``.

    A trailing run without a newline becomes the final inner list.
    """
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


def retarget_trivia_newlines(t: Trivia, target: str) -> None:
    """Rewrite every ``NewlineNode.text`` in ``t`` to ``target``.

    Grafted slots adopt the destination document's line ending rather
    than preserving the source's.
    """
    for p in t.pieces:
        if isinstance(p, NewlineNode) and p.text != target:
            p.text = target


def retarget_eol_newline(eol: EolTrivia, target: str) -> None:
    """Rewrite ``eol.newline.text`` to ``target`` (if present)."""
    if eol.newline is not None and eol.newline.text != target:
        eol.newline.text = target


def _pad_above_from(
    pieces: Sequence[TriviaPiece], body_start: int
) -> tuple[Trivia, Trivia]:
    """Split ``pieces`` into ``(pad, above)`` at ``body_start``.

    ``above`` is the comment block from ``body_start`` up to any trailing
    value-indent whitespace; ``pad`` is everything around it. Absent a
    comment in that span there is nothing to detach, so ``above`` is
    empty and ``pad`` is the whole input.
    """
    tail_start = len(pieces)
    if tail_start > body_start and isinstance(pieces[tail_start - 1], WhitespaceNode):
        tail_start -= 1
    middle = pieces[body_start:tail_start]
    if not any(isinstance(p, CommentNode) for p in middle):
        return Trivia(list(pieces)), Trivia()
    pad = Trivia(list(pieces[:body_start]) + list(pieces[tail_start:]))
    return pad, Trivia(list(middle))


def split_above_block(t: Trivia) -> tuple[Trivia, Trivia]:
    """Split ``t`` into ``(pad, above)``.

    ``above`` is the item-attached comment block; ``pad`` is the
    opening newline plus value indent that surrounds it. Boundary
    mutators use this to move an above-item block between bracket pads
    and item-leading trivia.

    The parts are *not* a simple concatenation: reconstruct with
    :func:`join_above_block`, which splices ``above`` between
    ``pad[0]`` and ``pad[1:]``. ``above`` is empty iff that middle
    region contains no ``CommentNode``.
    """
    pieces = t.pieces
    first_nl = next(
        (i for i, p in enumerate(pieces) if isinstance(p, NewlineNode)), None
    )
    if first_nl is None:
        return Trivia(list(pieces)), Trivia()
    return _pad_above_from(pieces, first_nl + 1)


def join_above_block(pad: Trivia, above: Trivia) -> Trivia:
    """Splice ``above`` back into ``pad``.

    ``above`` is inserted between ``pad[0]`` (opening NL) and the rest
    of ``pad`` (value indent). Inverse of :func:`split_above_block`.
    """
    pieces = list(pad.pieces)
    if not pieces:
        return Trivia(list(above.pieces))
    return Trivia([pieces[0], *above.pieces, *pieces[1:]])


def indent_from_final_trivia(ft: Trivia) -> str:
    """Extract a logical indent from a bracket-pad's pieces.

    Prefers the indent of the last comment line (so a varied-indent
    or blank-line-prefixed comment block aligns the new item with
    the *most recent* commented line). Falls back to the indent of
    the last whitespace-after-newline block, then to "".
    """
    pieces = ft.pieces
    last_comment_indent: str | None = None
    last_ws_after_nl: str | None = None
    j = 0
    while j < len(pieces):
        if isinstance(pieces[j], NewlineNode) and j + 1 < len(pieces):
            ws = ""
            if isinstance(pieces[j + 1], WhitespaceNode):
                ws = str(pieces[j + 1].text)
                last_ws_after_nl = ws
                k = j + 2
            else:
                k = j + 1
            if k < len(pieces) and isinstance(pieces[k], CommentNode):
                last_comment_indent = ws
        j += 1
    if last_comment_indent is not None:
        return last_comment_indent
    if last_ws_after_nl is not None:
        return last_ws_after_nl
    return ""


def restamp_bracket_pad_for_first(ft: Trivia) -> tuple[Trivia, Trivia]:
    r"""Reframe an empty bracket pad ahead of inserting the first item.

    For an empty value, ``final_trivia`` owns everything between the
    brackets. Return the ``(header_trivia, final_trivia)`` pair for an
    about-to-be-inserted first item:

    * Empty pad: two empty trivia.
    * Single-line pad: mirror the pad on both bracket faces.
    * Multi-line pad: split at the last newline; keep a value-indent
      on ``header_trivia`` and the row break on ``final_trivia``.

    Shared between :class:`Array` and inline-table append paths.
    """
    if not ft.pieces:
        return Trivia(), Trivia()
    if not has_newline(ft.pieces):
        return ft.copy(), ft
    pieces = list(ft.pieces)
    last_nl = max(i for i, p in enumerate(pieces) if isinstance(p, NewlineNode))
    head_pieces = pieces[: last_nl + 1]
    tail_pieces = pieces[last_nl + 1 :]
    if tail_pieces and isinstance(tail_pieces[0], WhitespaceNode):
        value_indent = str(tail_pieces[0].text)
    else:
        value_indent = indent_from_final_trivia(ft) or "    "
    new_header = Trivia([*head_pieces, WhitespaceNode(text=value_indent)])
    new_final = Trivia([NewlineNode(text=pieces[last_nl].text)])
    return new_header, new_final


def strip_trailing_indent(header_trivia: Trivia, final_trivia: Trivia) -> None:
    r"""Normalise an emptied bracket pad to canonical empty form.

    After deleting the last item, ``header_trivia`` may still hold the
    removed item's indent or a bracket-EOL comment. Without comments,
    drop the trailing whitespace/newline run so ``final_trivia`` owns
    the canonical empty ``[\\n]`` / ``{\\n}`` newline.

    With a bracket-EOL comment, migrate the surviving block into
    ``final_trivia``. That matches the parse-empty ownership for
    ``[ # tail\\n]`` / ``{ # tail\\n}``, so the next append can
    re-stamp via :func:`restamp_bracket_pad_for_first`.
    """
    if not has_comment(header_trivia.pieces):
        while header_trivia.pieces and isinstance(
            header_trivia.pieces[-1], (WhitespaceNode, NewlineNode)
        ):
            header_trivia.pieces.pop()
        return
    while header_trivia.pieces and isinstance(header_trivia.pieces[-1], WhitespaceNode):
        header_trivia.pieces.pop()
    # Drop final_trivia's leading newline if the comment's terminator
    # newline already produces a line break before `]` / `}`.
    if (
        header_trivia.pieces
        and isinstance(header_trivia.pieces[-1], NewlineNode)
        and final_trivia.pieces
        and isinstance(final_trivia.pieces[0], NewlineNode)
    ):
        final_trivia.pieces.pop(0)
    final_trivia.pieces[:0] = header_trivia.pieces
    header_trivia.pieces.clear()


def split_item_above(t: Trivia) -> tuple[Trivia, Trivia, Trivia]:
    """Split an item-leading region into ``(head_pad, above, tail_pad)``.

    Unlike :func:`split_above_block`, this is for ``items[i].leading``
    (i >= 1) where there is no bracket and the leading newline may
    have been hoisted onto item ``i-1``'s EOL section.

    ``head_pad`` is the leading NL (or empty); ``tail_pad`` is the
    trailing value-indent WS (or empty); ``above`` is the comment block
    between them.
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

    The EOL section is row-attached: inline whitespace, an EOL comment,
    and that row's terminating newline. Anything beyond — additional
    newlines, indent, above-item blocks — is structural and belongs to
    the *next* item's leading.

    If no EOL comment is present on the comma's row, the whole input
    is structural and the EOL half is empty. In that case ``t`` itself
    is returned as the structural half; callers must treat it as owned
    by the result.
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
        # Fast path: the overwhelming majority of lines carry no
        # trailing whitespace or comment, just a bare newline (or EOF).
        if self.trailing_ws is None and self.comment is None:
            return self.newline.text if self.newline is not None else ""
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
    "has_comment",
    "has_newline",
    "indent_from_final_trivia",
    "join_above_block",
    "leading_break_index",
    "leading_has_blank_line",
    "restamp_bracket_pad_for_first",
    "retarget_eol_newline",
    "retarget_trivia_newlines",
    "split_above_block",
    "split_eol_section",
    "split_item_above",
    "split_lines",
    "strip_trailing_indent",
]
