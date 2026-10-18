"""Public load/dump wrappers, binary I/O, encoding, and newline handling."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import pytest

import tomlrt
from _helpers import td
from tomlrt import Table

if TYPE_CHECKING:
    from pathlib import Path


def test_loads_is_alias_for_parse() -> None:
    src = "x = 1\ny = 'hi'\n"
    a = tomlrt.loads(src)
    b = tomlrt.loads(src)
    assert tomlrt.dumps(a) == tomlrt.dumps(b) == src


def test_load_from_binary_stream() -> None:
    fp = io.BytesIO(b"name = 'ada'\n")
    doc = tomlrt.load(fp)
    assert doc["name"] == "ada"


def test_load_from_real_file_path(tmp_path: Path) -> None:
    p = tmp_path / "doc.toml"
    p.write_text("k = 42\n", encoding="utf-8")
    with p.open("rb") as fp:
        doc = tomlrt.load(fp)
    assert doc["k"] == 42


def test_load_rejects_text_stream() -> None:
    fp = io.StringIO("port = 8080\n")
    with pytest.raises(TypeError, match="binary"):
        tomlrt.load(fp)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]


def test_load_reports_invalid_utf8_as_parse_error() -> None:
    fp = io.BytesIO("é = 1\nx = ".encode() + b"\xff")
    with pytest.raises(
        tomlrt.TOMLParseError,
        match=r"^invalid UTF-8 \(line 2, column 5\)$",
    ) as exc_info:
        tomlrt.load(fp)
    assert exc_info.value.line == 2
    assert exc_info.value.col == 5
    assert exc_info.value.offset == 10


def test_load_preserves_crlf_line_endings(tmp_path: Path) -> None:
    p = tmp_path / "win.toml"
    p.write_bytes(b"a = 1\r\nb = 2\r\n")
    with p.open("rb") as fp:
        doc = tomlrt.load(fp)
    out = io.BytesIO()
    tomlrt.dump(doc, out)
    assert out.getvalue() == b"a = 1\r\nb = 2\r\n"


def test_crlf_document_keeps_crlf_after_mutation() -> None:
    doc = tomlrt.loads("a = 1\r\nb = 2\r\n")
    doc["c"] = 3
    assert tomlrt.dumps(doc) == "a = 1\r\nb = 2\r\nc = 3\r\n"


@pytest.mark.parametrize(
    "src",
    [
        td(
            '''
            value = """
            text"""''',
        ).replace("\n", "\r\n"),
        td(
            """
            value = '''text
            more'''""",
        ).replace("\n", "\r\n"),
        td(
            '''
            value = """text\\
              more"""''',
        ).replace("\n", "\r\n"),
    ],
    ids=["trimmed-opening", "literal-body", "line-continuation"],
)
def test_crlf_multiline_string_is_document_newline_fallback(src: str) -> None:
    doc = tomlrt.loads(src)
    assert tomlrt.dumps(doc) == src
    doc["added"] = 1
    out = tomlrt.dumps(doc)
    assert out == src + "\r\nadded = 1\r\n"
    assert tomlrt.loads(out).to_dict() == doc.to_dict()


def test_crlf_structure_overrides_lf_multiline_string() -> None:
    src = (
        td(
            '''
            value = """foo
            bar"""''',
        )
        + "\r\nother = 1\r\n"
    )
    doc = tomlrt.loads(src)
    assert tomlrt.dumps(doc) == src
    doc["added"] = 2
    out = tomlrt.dumps(doc)
    assert out == src + "added = 2\r\n"
    assert tomlrt.loads(out).to_dict() == doc.to_dict()


def test_lf_structure_overrides_crlf_multiline_string() -> None:
    src = td(
        '''
        value = """foo
        bar"""
        other = 1
        ''',
    ).replace("foo\nbar", "foo\r\nbar")
    doc = tomlrt.loads(src)
    assert tomlrt.dumps(doc) == src
    doc["added"] = 2
    out = tomlrt.dumps(doc)
    assert out == src + "added = 2\n"
    assert tomlrt.loads(out).to_dict() == doc.to_dict()


def test_escaped_newline_does_not_affect_document_newline() -> None:
    src = td(
        '''
        value = """foo\\nbar"""
        other = 1
        ''',
    ).replace("\n", "\r\n")
    doc = tomlrt.loads(src)
    assert tomlrt.dumps(doc) == src
    doc["added"] = 2
    out = tomlrt.dumps(doc)
    assert out == src + "added = 2\r\n"
    assert tomlrt.loads(out).to_dict() == doc.to_dict()


def test_dump_writes_to_binary_stream() -> None:
    doc = tomlrt.loads("x = 1\n")
    out = io.BytesIO()
    tomlrt.dump(doc, out)
    assert out.getvalue() == b"x = 1\n"


def test_dump_emits_utf8_for_non_ascii() -> None:
    doc = tomlrt.loads("name = 'café'\n")
    out = io.BytesIO()
    tomlrt.dump(doc, out)
    assert out.getvalue() == "name = 'café'\n".encode()


def test_dumps_accepts_plain_dict() -> None:
    out = tomlrt.dumps({"a": 1, "s": {"b": 2}})
    assert out == td(
        """
        a = 1

        [s]
        b = 2
        """
    )


def test_dumps_document_is_byte_exact() -> None:
    src = td(
        """
        # preamble
        a = 1 # eol
        """
    )
    doc = tomlrt.loads(src)
    assert tomlrt.dumps(doc) == src


def test_dumps_accepts_table() -> None:
    out = tomlrt.dumps(Table.section({"k": "v"}))
    assert out == 'k = "v"\n'


def test_dump_accepts_plain_dict() -> None:
    out = io.BytesIO()
    tomlrt.dump({"a": 1}, out)
    assert out.getvalue() == b"a = 1\n"


def test_dumps_rejects_non_mapping() -> None:
    with pytest.raises(TypeError):
        tomlrt.dumps([1, 2, 3])  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
