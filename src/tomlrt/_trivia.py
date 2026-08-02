"""Source-preserving whitespace, newlines, and comments.

Trivia is verbatim source text: a run of inline whitespace, line
terminators and ``#`` comments, hanging off slots and value nodes.
Together with value lexemes it captures every byte needed for exact
round-trips, so it is stored as a plain ``str`` and the helpers here
are ordinary string operations.

The text decomposes into *pieces* -- a whitespace run, a line
terminator, or a comment -- and several helpers below talk in those
terms even though nothing materialises them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

# One trivia piece: a whitespace run, a line terminator, or a comment.
_RE_PIECE: Final = re.compile(r"[ \t]+|\r?\n|#[^\r\n]*")

# The row-attached end-of-line section: inline whitespace, a comment,
# and that row's terminator.
_RE_EOL_SECTION: Final = re.compile(r"[ \t]*#[^\r\n]*(?:\r?\n)?")

# A row break: optional inline whitespace then a line terminator.
_RE_BREAK: Final = re.compile(r"[ \t]*\r?\n")

# A line terminator.
_RE_NEWLINE: Final = re.compile(r"\r?\n")

_WS: Final = " \t"


def split_line(line: str) -> tuple[str, str, str]:
    """Split one trivia line into ``(pre, comment, terminator)``.

    ``comment`` is the ``#``-to-end-of-line lexeme, empty when the line
    carries none; ``pre`` is whatever precedes it.
    """
    term = line[len(line.rstrip("\r\n")) :]
    content = line[: len(line) - len(term)]
    start = content.find("#")
    if start == -1:
        return content, "", term
    return content[:start], content[start:], term


def leading_has_blank_line(leading: str) -> bool:
    r"""Whether ``leading`` contains at least one blank physical line.

    A blank line is a line in the leading-trivia stream that carries no
    comment. A comment-line terminator (e.g. ``# foo\n``) does not count
    as a blank -- the newline belongs to the comment.
    """
    return any("#" not in line for line in leading.split("\n")[:-1])


def leading_break(t: str) -> int:
    r"""Length of the row break at the start of ``t``, or 0 if there is none.

    An optional inline-whitespace run is skipped first, so trailing
    whitespace from the previous row (``1,  \n``) does not mask the
    break.
    """
    m = _RE_BREAK.match(t)
    return m.end() if m is not None else 0


def split_lines(t: str) -> list[str]:
    r"""Split ``t`` into lines, each keeping its terminator.

    A trailing run without a terminator becomes the final element.
    Trivia only ever breaks on ``\n``, so this is deliberately not
    ``str.splitlines``, which also splits on other Unicode separators.
    """
    lines = [line + "\n" for line in t.split("\n")]
    last = lines.pop()[:-1]
    if last:
        lines.append(last)
    return lines


def retarget_newlines(t: str, target: str) -> str:
    """Rewrite every line terminator in ``t`` to ``target``.

    Grafted slots adopt the destination document's line ending rather
    than preserving the source's.
    """
    flat = t.replace("\r\n", "\n")
    return flat if target == "\n" else flat.replace("\n", target)


def trailing_ws(t: str) -> str:
    """The trailing inline-whitespace run of ``t`` (possibly empty)."""
    return t[len(t.rstrip(_WS)) :]


def leading_ws(t: str) -> str:
    """The leading inline-whitespace run of ``t`` (possibly empty)."""
    return t[: len(t) - len(t.lstrip(_WS))]


def _pad_above_from(t: str, body_start: int) -> tuple[str, str]:
    """Split ``t`` into ``(pad, above)`` at ``body_start``.

    ``above`` is the comment block from ``body_start`` up to any trailing
    value-indent whitespace; ``pad`` is everything around it. Absent a
    comment in that span there is nothing to detach, so ``above`` is
    empty and ``pad`` is the whole input.
    """
    tail_start = len(t) - len(trailing_ws(t[body_start:]))
    middle = t[body_start:tail_start]
    if "#" not in middle:
        return t, ""
    return t[:body_start] + t[tail_start:], middle


def split_above_block(t: str) -> tuple[str, str]:
    """Split ``t`` into ``(pad, above)``.

    ``above`` is the item-attached comment block; ``pad`` is the opening
    newline plus value indent that surrounds it. Boundary mutators use
    this to move an above-item block between bracket pads and
    item-leading trivia.

    The parts are *not* a simple concatenation: reconstruct with
    :func:`join_above_block`, which splices ``above`` after ``pad``'s
    first piece. ``above`` is empty iff that middle region carries no
    comment.
    """
    first_nl = t.find("\n")
    if first_nl == -1:
        return t, ""
    return _pad_above_from(t, first_nl + 1)


def join_above_block(pad: str, above: str) -> str:
    """Splice ``above`` back into ``pad``.

    ``above`` is inserted after ``pad``'s first piece (the opening
    newline) and before the rest (the value indent). Inverse of
    :func:`split_above_block`.
    """
    if not pad:
        return above
    m = _RE_PIECE.match(pad)
    cut = m.end() if m is not None else 0
    return pad[:cut] + above + pad[cut:]


def indent_from_trivia(t: str) -> str:
    """Extract a logical indent from a bracket pad.

    Prefers the indent of the last comment line (so a varied-indent or
    blank-line-prefixed comment block aligns the new item with the *most
    recent* commented line). Falls back to the indent of the last
    non-empty whitespace-after-newline run, then to "".
    """
    last_comment_indent: str | None = None
    last_ws: str | None = None
    for line in t.split("\n")[1:]:
        ws = leading_ws(line)
        if ws:
            last_ws = ws
        if line[len(ws) :].startswith("#"):
            last_comment_indent = ws
    if last_comment_indent is not None:
        return last_comment_indent
    return last_ws if last_ws is not None else ""


def restamp_bracket_pad_for_first(ft: str) -> tuple[str, str]:
    r"""Reframe an empty bracket pad ahead of inserting the first item.

    For an empty value, ``final_trivia`` owns everything between the
    brackets. Return the ``(header_trivia, final_trivia)`` pair for an
    about-to-be-inserted first item:

    * Empty pad: two empty trivia.
    * Single-line pad: mirror the pad on both bracket faces.
    * Multi-line pad: split at the last newline; keep a value-indent on
      ``header_trivia`` and the row break on ``final_trivia``.

    Shared between :class:`Array` and inline-table append paths.
    """
    if not ft:
        return "", ""
    last_nl = ft.rfind("\n")
    if last_nl == -1:
        return ft, ft
    head, tail = ft[: last_nl + 1], ft[last_nl + 1 :]
    value_indent = leading_ws(tail) or indent_from_trivia(ft) or "    "
    newline = "\r\n" if ft[last_nl - 1 : last_nl] == "\r" else "\n"
    return head + value_indent, newline


def strip_trailing_indent(header_trivia: str, final_trivia: str) -> tuple[str, str]:
    r"""Normalise an emptied bracket pad to canonical empty form.

    After deleting the last item, ``header_trivia`` may still hold the
    removed item's indent or a bracket-EOL comment. Without comments,
    drop the trailing whitespace/newline run so ``final_trivia`` owns
    the canonical empty ``[\n]`` / ``{\n}`` newline.

    With a bracket-EOL comment, migrate the surviving block into
    ``final_trivia``. That matches the parse-empty ownership for
    ``[ # tail\n]`` / ``{ # tail\n}``, so the next append can re-stamp
    via :func:`restamp_bracket_pad_for_first`.
    """
    if "#" not in header_trivia:
        return header_trivia.rstrip(" \t\r\n"), final_trivia
    header_trivia = header_trivia.rstrip(_WS)
    # Drop final_trivia's leading newline if the comment's terminator
    # newline already produces a line break before `]` / `}`.
    if header_trivia.endswith("\n") and final_trivia.startswith(("\n", "\r\n")):
        final_trivia = final_trivia[2:] if final_trivia[0] == "\r" else final_trivia[1:]
    return "", header_trivia + final_trivia


def split_item_above(t: str) -> tuple[str, str, str]:
    """Split an item-leading region into ``(head_pad, above, tail_pad)``.

    Unlike :func:`split_above_block`, this is for ``items[i].leading``
    (i >= 1) where there is no bracket and the leading newline may have
    been hoisted onto item ``i-1``'s EOL section.

    ``head_pad`` is the leading newline (or empty); ``tail_pad`` is the
    trailing value-indent whitespace (or empty); ``above`` is the
    comment block between them.
    """
    m = _RE_NEWLINE.match(t)
    head = m.group() if m is not None else ""
    t = t[len(head) :]
    tail = trailing_ws(t)
    return head, t[: len(t) - len(tail)], tail


def split_eol_section(t: str) -> tuple[str, str]:
    """Split ``t`` into the inline EOL section and the structural rest.

    The EOL section is row-attached: inline whitespace, an EOL comment,
    and that row's terminating newline. Anything beyond -- additional
    newlines, indent, above-item blocks -- is structural and belongs to
    the *next* item's leading.

    If no EOL comment is present on the comma's row, the whole input is
    structural and the EOL half is empty.
    """
    m = _RE_EOL_SECTION.match(t)
    if m is None:
        return "", t
    return t[: m.end()], t[m.end() :]


@dataclass(slots=True, eq=False)
class EolTrivia:
    """End-of-line tail of a single physical line.

    Used by `KVSlot` and `StructuralHeaderSlot` to capture the
    optional inline comment plus the line terminator. Each part is
    verbatim source text, empty when absent; ``newline`` is empty only
    for the last line of a file with no final newline.
    """

    trailing_ws: str  # whitespace before any comment / newline
    comment: str  # includes the leading '#'
    newline: str

    def render(self) -> str:
        return f"{self.trailing_ws}{self.comment}{self.newline}"


def retarget_eol_newline(eol: EolTrivia, target: str) -> None:
    """Rewrite ``eol.newline`` to ``target`` (if present)."""
    if eol.newline:
        eol.newline = target


__all__ = [
    "EolTrivia",
    "indent_from_trivia",
    "join_above_block",
    "leading_break",
    "leading_has_blank_line",
    "leading_ws",
    "restamp_bracket_pad_for_first",
    "retarget_eol_newline",
    "retarget_newlines",
    "split_above_block",
    "split_eol_section",
    "split_item_above",
    "split_line",
    "split_lines",
    "strip_trailing_indent",
    "trailing_ws",
]
