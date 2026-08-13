"""The comment text format: encode, decode, validate, and split.

A comment is stored verbatim in trivia as ``# ...``; these helpers are
the one place that knows how to get between that raw form and the
decoded strings the public API exposes, and how to carve a leading
trivia run into its comment blocks.

The two comment-view flavours sit on different substrates -- the slot
stream (`tomlrt._comments`) and comma values (`tomlrt._comma_comments`)
-- but agree on the text format itself, so both import it from here as
peers rather than one reaching into the other.
"""

from __future__ import annotations

from collections.abc import Iterable

from tomlrt._trivia import split_line, split_lines


def _validate_comment_controls(text: str) -> None:
    """Reject ASCII control characters (other than TAB) and DEL."""
    for ch in text:
        cp = ord(ch)
        if cp == 0x09:
            continue
        if cp < 0x20 or cp == 0x7F:
            msg = f"comment may not contain control character U+{cp:04X}"
            raise ValueError(msg)


def _validate_comment_content(text: str, newline_msg: str) -> None:
    """Reject a comment value that would not round-trip via the parser.

    ``newline_msg`` lets callers phrase the line-terminator error for
    their context (a lone scalar vs. one entry of a sequence).
    """
    if "\n" in text or "\r" in text:
        raise ValueError(newline_msg)
    _validate_comment_controls(text)


def _validate_comment_str(value: object, name: str) -> str:
    """Type-check ``value`` is a str and validate its content; return it."""
    if not isinstance(value, str):
        msg = f"{name} must be str, got {type(value).__name__}"
        raise TypeError(msg)
    _validate_comment_content(value, "comment must be single-line")
    return value


def _validate_comment_entries(
    value: object, name: str, *, allow_none: bool
) -> tuple[str | None, ...]:
    """Type-check a comment iterable, optionally allowing ``None`` entries."""
    kind = "comment strings or None" if allow_none else "comment strings"
    if isinstance(value, str) or not isinstance(value, Iterable):
        msg = f"{name} must be an iterable of {kind}"
        raise TypeError(msg)
    out: list[str | None] = []
    for c in value:
        if c is None and allow_none:
            out.append(None)
            continue
        if not isinstance(c, str):
            msg = f"{name} entries must be strings{' or None' if allow_none else ''}"
            raise TypeError(msg)
        _validate_comment_content(
            c, f"{name} entries must not contain a line terminator"
        )
        out.append(c)
    return tuple(out)


def _validate_comment_seq(value: object, name: str) -> tuple[str, ...]:
    return tuple(
        c
        for c in _validate_comment_entries(value, name, allow_none=False)
        if c is not None
    )


def _decode_comment(raw: str) -> str:
    """Strip the leading ``#`` and one optional space from a raw comment."""
    return raw.removeprefix("#").removeprefix(" ")


def _encode_comment(text: str) -> str:
    """Encode a logical comment into a raw ``# ...`` form."""
    if text == "":
        return "#"
    return f"# {text}"


def _line_to_comment(line: str) -> str | None:
    """Decoded comment text for a line, or ``None`` if it has no comment."""
    comment = split_line(line)[1]
    return _decode_comment(comment) if comment else None


def _lines_to_comments(t: str) -> tuple[str, ...]:
    """Extract one decoded comment per line of ``t`` that carries one."""
    return tuple(
        c for line in split_lines(t) if (c := _line_to_comment(line)) is not None
    )


def _render_comment_lines(
    block: tuple[str | None, ...], nl: str, indent: str = ""
) -> str:
    """Render logical comment lines, using ``None`` for blanks."""
    return "".join(
        nl if entry is None else f"{indent}{_encode_comment(entry)}{nl}"
        for entry in block
    )


def _split_preamble(leading: str) -> tuple[str, str]:
    """Split the head slot's leading at the first blank line into (preamble, rest).

    Dual of :func:`_split_attached_block`, which cuts at the *last* blank.
    There is no preamble if the opening comment run attaches straight to
    the first construct, or the leading starts with a blank line.
    """
    lines = split_lines(leading)
    i = 0
    while i < len(lines) and "#" in lines[i]:
        i += 1
    # A blank *separator* line is a newline with no comment, distinct from
    # the slot's trailing indent (whitespace with no newline).
    if i == 0 or i >= len(lines) or "\n" not in lines[i]:
        return "", leading
    return "".join(lines[: i + 1]), "".join(lines[i + 1 :])


def _split_attached_block(leading: str) -> tuple[str, str, str]:
    """Split the leading into (above_blank, attached_comment_lines, slot_indent).

    The attached group is the contiguous comment run immediately before
    the slot. Earlier lines are preamble/archived blocks. ``slot_indent``
    is the trailing whitespace-only, newline-less column offset that
    rebuilders must reapply.
    """
    lines = split_lines(leading)
    indent = ""
    if lines and "\n" not in lines[-1] and "#" not in lines[-1]:
        indent = lines.pop()
    i = len(lines)
    while i > 0 and "#" in lines[i - 1]:
        i -= 1
    return "".join(lines[:i]), "".join(lines[i:]), indent


def _extract_leading_comments(leading: str) -> tuple[str, ...]:
    """Return only the *attached* run of comment-bearing lines.

    Comments separated by a blank line are preamble or archived blocks
    and are excluded.
    """
    _above, attached, _indent = _split_attached_block(leading)
    return _lines_to_comments(attached)


def _set_attached_block(leading: str, comments: tuple[str, ...], nl: str) -> str:
    """Replace the attached comment block on ``leading`` with ``comments``.

    Preserve preamble/archived blocks and reapply the slot's indent to
    each new comment line and the slot itself.
    """
    above, _attached, indent = _split_attached_block(leading)
    return above + _render_comment_lines(comments, nl, indent) + indent


__all__ = [
    "_decode_comment",
    "_encode_comment",
    "_extract_leading_comments",
    "_line_to_comment",
    "_lines_to_comments",
    "_render_comment_lines",
    "_set_attached_block",
    "_split_attached_block",
    "_split_preamble",
    "_validate_comment_content",
    "_validate_comment_controls",
    "_validate_comment_entries",
    "_validate_comment_seq",
    "_validate_comment_str",
]
