"""Property-based tests: round-trip fidelity and API invariants over
generated, well-formed TOML (as opposed to `test_fuzz_parser.py`, which
throws adversarial/malformed input at the parser, and
`test_fuzz_mutation.py`, which fuzzes the mutation API over real
corpus documents and, from empty, via the builder API). Built with
Hypothesis, but organised by what each property covers: round-trip +
semantic equivalence to `tomli` over generated documents, comment-API
round-trip, `format()` idempotence, CRLF preservation, and synthesis
from a plain Python tree.
"""

from __future__ import annotations

from typing import Any

import pytest
import tomli
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import tomlrt
from _helpers import deep_equal
from tomlrt import Document, FormatOptions

pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Strategies for safe TOML fragments (we generate values whose canonical
# rendering by the parser is byte-stable, so we can assert exact round-trip).
# ---------------------------------------------------------------------------

_BARE_KEY = st.from_regex(r"\A[A-Za-z][A-Za-z0-9_-]{0,15}\Z")

_BASIC_STR_CHARS = st.text(
    alphabet=st.characters(
        min_codepoint=0x20,
        max_codepoint=0x7E,
        blacklist_characters='"\\',
    ),
    max_size=20,
)


def _quoted(s: str) -> str:
    return '"' + s + '"'


_STRINGS = _BASIC_STR_CHARS.map(_quoted)
_INTS = st.integers(min_value=-(2**31), max_value=2**31 - 1).map(str)
_BOOLS = st.sampled_from(["true", "false"])
_FLOATS = st.sampled_from(["0.0", "1.5", "-3.25", "1e10", "-2.5e-3", "inf", "-inf"])

# TOML 1.1 scalar literals that `tomli` (>=2.4) accepts but the stdlib
# `tomllib` does not yet: the `\xHH` / `\e` basic-string escapes and the
# seconds-optional date-time / time forms. Including them here threads
# 1.1 syntax through every `_document()`-based round-trip + oracle test.
_TOML11_STRINGS = st.sampled_from(
    [r'"\xe9"', r'"tail \xE9"', r'"esc\e end"', r'"\x00\x7f"']
)
_TOML11_DATETIMES = st.sampled_from(
    ["07:32", "1979-05-27T07:32", "1979-05-27 07:32Z", "1979-05-27 07:32-07:00"]
)
_TOML11_SCALARS = st.one_of(_TOML11_STRINGS, _TOML11_DATETIMES)

_SCALARS = st.one_of(_STRINGS, _INTS, _BOOLS, _FLOATS, _TOML11_SCALARS)

_COMMENT_TEXT = st.text(
    alphabet=st.characters(
        blacklist_categories=["Cs"],
        blacklist_characters="\n\r"
        + "".join(chr(c) for c in range(0x20) if c != 0x09)
        + "\x7f",
    ),
    max_size=30,
)


@st.composite
def _kv_lines(draw: st.DrawFn, keys: list[str] | None = None) -> str:
    if keys is None:
        keys = draw(st.lists(_BARE_KEY, max_size=4, unique=True))
    out: list[str] = []
    for k in keys:
        v = draw(_SCALARS)
        out.append(f"{k} = {v}")
    return "\n".join(out) + ("\n" if out else "")


@st.composite
def _array_value(draw: st.DrawFn) -> str:
    elems = draw(st.lists(_SCALARS, max_size=5))
    if not elems:
        return "[]"
    return "[ " + ", ".join(elems) + " ]"


@st.composite
def _section(draw: st.DrawFn) -> str:
    parts = draw(
        st.lists(_BARE_KEY, min_size=1, max_size=3, unique=True),
    )
    header = "[" + ".".join(parts) + "]\n"
    body = draw(_kv_lines())
    return header + body


@st.composite
def _document(draw: st.DrawFn) -> str:
    # Reserve a pool of names; partition them between pre-section keys
    # and section first-name components so a key like ``a = 1`` never
    # collides with a later ``[a]`` (which is invalid TOML).
    pool = draw(st.lists(_BARE_KEY, min_size=0, max_size=8, unique=True))
    cut = draw(st.integers(min_value=0, max_value=len(pool)))
    pre_keys = pool[:cut]
    section_roots = pool[cut:]

    pre = draw(_kv_lines(keys=pre_keys))
    # Build unique section paths from the available roots; each root
    # gets a unique single- or two-part path.
    sec_paths: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    for root in section_roots:
        depth = draw(st.integers(min_value=1, max_value=2))
        if depth == 1:
            path: tuple[str, ...] = (root,)
        else:
            sub = draw(_BARE_KEY)
            path = (root, sub)
        if path in seen:
            continue
        seen.add(path)
        sec_paths.append(path)

    parts: list[str] = []
    if pre:
        parts.append(pre)
    for path in sec_paths:
        header = "[" + ".".join(path) + "]\n"
        # KVs inside a section must not collide with the section's own
        # first-part name reserved at root level.
        body_keys = draw(
            st.lists(
                _BARE_KEY.filter(lambda k: k not in section_roots),
                max_size=4,
                unique=True,
            ),
        )
        body = draw(_kv_lines(keys=body_keys))
        parts.append(header + body)
    return "".join(parts) or "\n"


def _python_scalar(literal: str) -> Any:
    """Decode a TOML scalar literal (as our generator emits it) to Python."""
    return tomli.loads(f"_x = {literal}")["_x"]


@st.composite
def _document_with_overrides(draw: st.DrawFn) -> tuple[str, dict[str, Any]]:
    src = draw(_document())
    parsed = tomli.loads(src)
    top_keys = [k for k, v in parsed.items() if not isinstance(v, (dict, list))]
    overrides: dict[str, Any] = {}
    for k in top_keys:
        if draw(st.booleans()):
            overrides[k] = _python_scalar(draw(_SCALARS))
    return src, overrides


@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(case=_document_with_overrides())
def test_document_invariants(case: tuple[str, dict[str, Any]]) -> None:
    """All `_document()`-strategy invariants in one parser-pass per example.

    For each generated source, asserts:
    * byte-exact round-trip (parse + dumps == src);
    * semantic equivalence to `tomli`;
    * mutating top-level scalar slots reflects in to_dict() and
      survives a dump/parse cycle.
    """
    src, overrides = case
    doc = tomlrt.loads(src)
    assert tomlrt.dumps(doc) == src
    expected = tomli.loads(src)
    assert deep_equal(doc.to_dict(), expected)
    for k, v in overrides.items():
        doc[k] = v
        expected[k] = v
    assert deep_equal(doc.to_dict(), expected)
    assert deep_equal(tomlrt.loads(tomlrt.dumps(doc)).to_dict(), expected)


_FORMAT_OPTIONS = st.builds(
    FormatOptions,
    normalize_comments=st.booleans(),
    indent=st.integers(min_value=0, max_value=4),
    eol_comment_spaces=st.integers(min_value=0, max_value=3),
    multiline_trailing_comma=st.booleans(),
)


def _lay_out(draw: st.DrawFn, entries: list[str], open_b: str, close_b: str) -> str:
    """Render ``entries`` single- or multi-line, with a random trailing
    comma and EOL comment when multi-line -- the source-level choices
    `format()`'s options canonicalise.
    """
    if not entries:
        return open_b + close_b
    if not draw(st.booleans()):
        return open_b + " " + ", ".join(entries) + " " + close_b
    lines = [f"  {e}," for e in entries]
    if not draw(st.booleans()):
        lines[-1] = lines[-1].removesuffix(",")
    if comment := draw(_COMMENT_TEXT):
        lines[-1] += f" #{comment}"
    return open_b + "\n" + "\n".join(lines) + "\n" + close_b


@st.composite
def _format_value(draw: st.DrawFn, depth: int) -> str:
    """A scalar, or (if ``depth`` allows) a nested array / inline table of
    these, randomly laid out -- see `_lay_out`.
    """
    if depth <= 0 or draw(st.booleans()):
        return draw(_SCALARS)
    n = draw(st.integers(min_value=0, max_value=3))
    if draw(st.booleans()):
        keys = draw(st.lists(_BARE_KEY, min_size=n, max_size=n, unique=True))
        entries = [f"{k} = {draw(_format_value(depth - 1))}" for k in keys]
        return _lay_out(draw, entries, "{", "}")
    entries = [draw(_format_value(depth - 1)) for _ in range(n)]
    return _lay_out(draw, entries, "[", "]")


@given(
    options=_FORMAT_OPTIONS,
    root=_SCALARS,
    comment=_COMMENT_TEXT,
    outer=_format_value(depth=3),
)
@settings(max_examples=100, database=None)
def test_format_options_preserve_data_and_are_idempotent(
    options: FormatOptions, root: str, comment: str, outer: str
) -> None:
    src = f"root = {root} #{comment}\nouter = {outer}\n"
    expected = tomli.loads(src)
    doc = tomlrt.loads(src)
    doc.format(options=options)
    once = tomlrt.dumps(doc)
    assert deep_equal(doc.to_dict(), expected)
    assert deep_equal(tomli.loads(once), expected)
    doc.format(options=options)
    assert tomlrt.dumps(doc) == once
    assert tomlrt.dumps(tomlrt.loads(once)) == once


# ---------------------------------------------------------------------------
# Comment-view round-trip: writing back what we read must be a no-op, and
# the rendered comment must read back as the value we wrote. These caught
# the "user already supplied #" branch, the empty-string-as-delete shortcut,
# and the rstrip in the marker-stripper.
# ---------------------------------------------------------------------------

_COMMENT_LINES = st.lists(_COMMENT_TEXT, min_size=1, max_size=4).map(tuple)


def _set_eol(doc: Document, v: Any) -> None:
    doc.comments["a"] = v


def _get_eol(doc: Document) -> Any:
    return doc.comments["a"]


def _set_leading(doc: Document, v: Any) -> None:
    doc.leading_comments["a"] = v


def _get_leading(doc: Document) -> Any:
    return doc.leading_comments["a"]


def _set_arr_eol(doc: Document, v: Any) -> None:
    doc.array("a").comments[0] = v


def _get_arr_eol(doc: Document) -> Any:
    return doc.array("a").comments[0]


def _set_arr_leading(doc: Document, v: Any) -> None:
    doc.array("a").leading_comments[1] = v


def _get_arr_leading(doc: Document) -> Any:
    return doc.array("a").leading_comments[1]


def _set_header(doc: Document, v: Any) -> None:
    doc.table("s").header_comment = v


def _get_header(doc: Document) -> Any:
    return doc.table("s").header_comment


def _set_header_leading(doc: Document, v: Any) -> None:
    doc.table("s").header_leading_comments = v


def _get_header_leading(doc: Document) -> Any:
    return doc.table("s").header_leading_comments


def _set_preamble(doc: Document, v: Any) -> None:
    doc.preamble = v


def _get_preamble(doc: Document) -> Any:
    return doc.preamble


def _set_epilogue(doc: Document, v: Any) -> None:
    doc.epilogue = v


def _get_epilogue(doc: Document) -> Any:
    return doc.epilogue


_KV_FIXTURE = "a = 1\n"
_ARR_FIXTURE = "a = [1, 2]\n"
_SECT_FIXTURE = "[s]\nx = 1\n"


@pytest.mark.parametrize(
    ("fixture", "setter", "getter", "values"),
    [
        (_KV_FIXTURE, _set_eol, _get_eol, _COMMENT_TEXT),
        (_KV_FIXTURE, _set_leading, _get_leading, _COMMENT_LINES),
        (_ARR_FIXTURE, _set_arr_eol, _get_arr_eol, _COMMENT_TEXT),
        (_ARR_FIXTURE, _set_arr_leading, _get_arr_leading, _COMMENT_LINES),
        (_SECT_FIXTURE, _set_header, _get_header, _COMMENT_TEXT),
        (_SECT_FIXTURE, _set_header_leading, _get_header_leading, _COMMENT_LINES),
        (_KV_FIXTURE, _set_preamble, _get_preamble, _COMMENT_LINES),
        (_KV_FIXTURE, _set_epilogue, _get_epilogue, _COMMENT_LINES),
    ],
)
def test_comment_roundtrip(
    fixture: str,
    setter: Any,
    getter: Any,
    values: st.SearchStrategy[Any],
) -> None:
    @given(value=values)
    @settings(max_examples=50, database=None)
    def check(value: Any) -> None:
        doc = tomlrt.loads(fixture)
        setter(doc, value)
        out = tomlrt.dumps(doc)
        assert getter(tomlrt.loads(out)) == value

    check()


@given(text=_COMMENT_TEXT.filter(bool))
@settings(max_examples=200, database=None)
def test_eol_comment_set_then_clear(text: str) -> None:
    """Setting then deleting an EOL comment must restore a comment-free dump."""
    base = "a = 1\n"
    doc = tomlrt.loads(base)
    doc.comments["a"] = text
    del doc.comments["a"]
    assert tomlrt.dumps(doc) == base


# ---------------------------------------------------------------------------
# CRLF preservation: the round-trip invariant explicitly covers line endings.
# Take a generated source and randomly map each '\n' to '\n' or '\r\n'.
# ---------------------------------------------------------------------------


@st.composite
def _crlf_variant(draw: st.DrawFn) -> str:
    src = draw(_document())
    n = src.count("\n")
    flips = draw(st.lists(st.booleans(), min_size=n, max_size=n))
    out: list[str] = []
    i = 0
    for ch in src:
        if ch == "\n":
            out.append("\r\n" if flips[i] else "\n")
            i += 1
        else:
            out.append(ch)
    return "".join(out)


@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(src=_crlf_variant())
def test_crlf_roundtrip_exact(src: str) -> None:
    assert tomlrt.dumps(tomlrt.loads(src)) == src


# ---------------------------------------------------------------------------
# Synthesis round-trip: build a Document from a plain Python tree, dump it,
# parse it back, and compare the recovered data to the original.
# ---------------------------------------------------------------------------

_PY_SCALARS: st.SearchStrategy[Any] = st.one_of(
    _BASIC_STR_CHARS,
    st.integers(min_value=-(2**31), max_value=2**31 - 1),
    st.booleans(),
    st.floats(allow_nan=False, allow_infinity=False, width=64),
)


def _py_dict(max_depth: int) -> st.SearchStrategy[dict[str, Any]]:
    if max_depth <= 0:
        values: st.SearchStrategy[Any] = _PY_SCALARS
    else:
        values = st.one_of(
            _PY_SCALARS,
            st.lists(_PY_SCALARS, max_size=4),
            _py_dict(max_depth - 1),
        )
    return st.dictionaries(_BARE_KEY, values, max_size=4)


@settings(max_examples=100, deadline=None)
@given(data=_py_dict(max_depth=2))
def test_synthesise_roundtrip(data: dict[str, Any]) -> None:
    doc = Document(data)
    out = tomlrt.dumps(doc)
    recovered = tomlrt.loads(out).to_dict()
    assert deep_equal(recovered, data)
