"""Source-preserving whitespace, newlines, and comments.

Trivia is verbatim source text -- inline whitespace, line terminators
and ``#`` comments -- hanging off slots and value nodes. With value
lexemes it captures every byte needed for exact round-trips, so it is
stored as a plain ``str`` and the helpers here are ordinary string
operations.
"""

from __future__ import annotations

import re
from typing import Final

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

    A comment line does not count as blank: its terminating newline
    belongs to the comment.
    """
    return any("#" not in line for line in leading.split("\n")[:-1])


def leading_break(t: str) -> int:
    r"""Length of the row break at the start of ``t``, or 0 if there is none.

    Whitespace is skipped first, so a trailing run from the previous
    row (``1,  \n``) does not mask the break.
    """
    m = _RE_BREAK.match(t)
    return m.end() if m is not None else 0


def split_lines(t: str) -> list[str]:
    r"""Split ``t`` into lines, each keeping its terminator.

    Deliberately not ``str.splitlines``, which also breaks on other
    Unicode separators; trivia only ever breaks on ``\n``.
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


def newline_at(t: str, nl: int) -> str:
    r"""The flavour of the line terminator whose ``"\n"`` is ``t[nl]``.

    Sliced, not indexed: a newline at position zero must not wrap round
    and sample the last character of ``t``.
    """
    return "\r\n" if t[nl - 1 : nl] == "\r" else "\n"


def strip_trailing_ws(t: str) -> str:
    """``t`` without its trailing inline-whitespace run."""
    return t.rstrip(_WS)


def trailing_ws(t: str) -> str:
    """The trailing inline-whitespace run of ``t`` (possibly empty)."""
    return t[len(strip_trailing_ws(t)) :]


def leading_ws(t: str) -> str:
    """The leading inline-whitespace run of ``t`` (possibly empty)."""
    return t[: len(t) - len(t.lstrip(_WS))]


def _pad_above_from(t: str, body_start: int) -> tuple[str, str]:
    """Split ``t`` into ``(pad, above)`` at ``body_start``.

    ``above`` is the comment block from ``body_start`` up to any
    trailing value-indent whitespace; ``pad`` is everything around it.
    """
    tail_start = len(t) - len(trailing_ws(t[body_start:]))
    middle = t[body_start:tail_start]
    if "#" not in middle:
        return t, ""
    return t[:body_start] + t[tail_start:], middle


def split_above_block(t: str) -> tuple[str, str]:
    """Split a bracket pad ``t`` into ``(pad, above)``.

    ``above`` is the item-attached comment block below the pad's opening
    newline; ``pad`` is the framing around it -- that newline plus the
    trailing value indent -- so the two are not a simple concatenation.
    """
    first_nl = t.find("\n")
    if first_nl == -1:
        return t, ""
    return _pad_above_from(t, first_nl + 1)


def indent_from_trivia(t: str) -> str:
    """Extract a logical indent from a bracket pad.

    Prefers the last comment line's indent, so that a block with varied
    indents aligns the new item with the most recent commented line.
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

    For an empty value ``final_trivia`` owns everything between the
    brackets; return the ``(header_trivia, final_trivia)`` pair that
    ownership should become once an item sits between them.
    """
    if not ft:
        return "", ""
    last_nl = ft.rfind("\n")
    if last_nl == -1:
        return ft, ft
    head, tail = ft[: last_nl + 1], ft[last_nl + 1 :]
    value_indent = leading_ws(tail) or indent_from_trivia(ft) or "    "
    newline = newline_at(ft, last_nl)
    return head + value_indent, newline


def strip_trailing_indent(header_trivia: str, final_trivia: str) -> tuple[str, str]:
    r"""Normalise an emptied bracket pad to canonical empty form.

    After deleting the last item, ``header_trivia`` may still hold the
    removed item's indent or a bracket-EOL comment. Without comments,
    drop the trailing whitespace run so ``final_trivia`` owns the
    canonical empty ``[\n]`` / ``{\n}`` newline. With one, migrate the
    surviving block into ``final_trivia``, matching how ``[ # tail\n]``
    parses when empty so the next append can re-stamp it.
    """
    if "#" not in header_trivia:
        return header_trivia.rstrip(" \t\r\n"), final_trivia
    header_trivia = strip_trailing_ws(header_trivia)
    # Drop final_trivia's leading newline if the comment's terminator
    # newline already produces a line break before `]` / `}`.
    if header_trivia.endswith("\n") and final_trivia.startswith(("\n", "\r\n")):
        final_trivia = final_trivia[2:] if final_trivia[0] == "\r" else final_trivia[1:]
    return "", header_trivia + final_trivia


def split_item_above(t: str) -> tuple[str, str, str]:
    """Split an item-leading region into ``(head_pad, above, tail_pad)``.

    ``head_pad`` is the leading newline, ``tail_pad`` the trailing
    value-indent, ``above`` the comment block between them.

    Unlike :func:`split_above_block` this is for ``items[i].leading``
    (i >= 1), where there is no bracket and the leading newline may have
    been hoisted onto item ``i-1``'s EOL section.
    """
    m = _RE_NEWLINE.match(t)
    head = m.group() if m is not None else ""
    t = t[len(head) :]
    tail = trailing_ws(t)
    return head, t[: len(t) - len(tail)], tail


def split_eol_section(t: str) -> tuple[str, str]:
    """Split ``t`` into the inline EOL section and the structural rest.

    The EOL section is row-attached: inline whitespace, an EOL comment,
    and that row's terminating newline. Anything beyond -- further
    newlines, indent, above-item blocks -- is structural and belongs to
    the *next* item's leading.
    """
    # The pattern cannot match without a `#`, and most trivia has none;
    # asking the string directly is much cheaper than starting the engine.
    if "#" not in t:
        return "", t
    m = _RE_EOL_SECTION.match(t)
    if m is None:
        return "", t
    return t[: m.end()], t[m.end() :]


__all__ = [
    "indent_from_trivia",
    "leading_break",
    "leading_has_blank_line",
    "leading_ws",
    "newline_at",
    "restamp_bracket_pad_for_first",
    "retarget_newlines",
    "split_above_block",
    "split_eol_section",
    "split_item_above",
    "split_line",
    "split_lines",
    "strip_trailing_indent",
    "strip_trailing_ws",
    "trailing_ws",
]
