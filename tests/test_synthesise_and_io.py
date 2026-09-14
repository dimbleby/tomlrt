"""Tests for value synthesis (``value_to_node``) and the public file I/O.

These cover the corners of ``_synthesise.py`` and ``_public.py`` that
the rest of the suite skirts past: every escape branch in basic
strings, every scalar flavour accepted by ``value_to_node``, and the
``loads`` / ``load`` / ``dump`` wrappers.
"""

from __future__ import annotations

import io
import math
import sys
from copy import copy, deepcopy
from datetime import date, datetime, time, timedelta, timezone, tzinfo
from typing import TYPE_CHECKING, Any

import pytest

import tomlrt
from _helpers import reparses, td
from tomlrt import AoT, Array, Document, Table, TOMLError
from tomlrt._values import (
    ArrayItem,
    ArrayValue,
    InlineTableEntry,
    InlineTableValue,
    IntegerValue,
    KeyPart,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from typing_extensions import Self

if sys.version_info >= (3, 12):
    from typing import override
else:  # pragma: no cover -- backport for Python < 3.12
    from typing_extensions import override


# ---------------------------------------------------------------------------
# Public I/O wrappers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# String escaping (every branch in _escape_basic_string)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("py_value", "expected_quoted"),
    [
        ("plain", '"plain"'),
        ("back\\slash", '"back\\\\slash"'),
        ('with"quote', '"with\\"quote"'),
        ("line\nbreak", '"line\\nbreak"'),
        ("carriage\rreturn", '"carriage\\rreturn"'),
        ("tab\there", '"tab\\there"'),
        ("bell\bback", '"bell\\bback"'),
        ("form\ffeed", '"form\\ffeed"'),
        ("ctrl\x01char", '"ctrl\\u0001char"'),
        ("del\x7fchar", '"del\\u007Fchar"'),
    ],
)
def test_string_escape_emits_canonical_form(
    py_value: str, expected_quoted: str
) -> None:
    doc = tomlrt.loads("x = 0\n")
    doc["x"] = py_value
    out = tomlrt.dumps(doc)
    assert out == f"x = {expected_quoted}\n"
    # And it round-trips back to the same Python value.
    assert tomlrt.loads(out)["x"] == py_value


# ---------------------------------------------------------------------------
# value_to_node: every accepted Python type
# ---------------------------------------------------------------------------


def test_assign_bool_renders_as_toml_bool() -> None:
    doc = tomlrt.loads("x = 0\n")
    doc["x"] = True
    doc["y"] = False
    out = tomlrt.dumps(doc)
    assert out == "x = true\ny = false\n"
    re = tomlrt.loads(out)
    assert re["x"] is True
    assert re["y"] is False


def test_assign_int_renders_decimal() -> None:
    doc = tomlrt.loads("x = 0\n")
    doc["x"] = -123
    assert tomlrt.dumps(doc) == "x = -123\n"


def test_assign_float_basic_gets_dot_zero_when_missing() -> None:
    doc = tomlrt.loads("x = 0\n")
    doc["x"] = 3.0
    out = tomlrt.dumps(doc)
    # repr(3.0) is "3.0" already, but values like 1e10 round-trip via repr
    # which emits no dot; the helper appends one.
    assert out == "x = 3.0\n"
    assert tomlrt.loads(out)["x"] == 3.0


def test_assign_float_scientific_no_dot_added() -> None:
    doc = tomlrt.loads("x = 0\n")
    doc["x"] = 1e20
    out = tomlrt.dumps(doc)
    assert out == "x = 1e+20\n"
    re = tomlrt.loads(out)
    assert re["x"] == 1e20


def test_assign_float_inf_and_nan() -> None:
    doc = tomlrt.loads("x = 0\n")
    doc["x"] = math.inf
    doc["y"] = -math.inf
    doc["z"] = math.nan
    out = tomlrt.dumps(doc)
    assert out == "x = inf\ny = -inf\nz = nan\n"
    re = tomlrt.loads(out)
    assert re["x"] == math.inf
    assert re["y"] == -math.inf
    assert math.isnan(re["z"])


def test_assign_local_date() -> None:
    doc = tomlrt.loads("x = 0\n")
    doc["x"] = date(2024, 7, 4)
    out = tomlrt.dumps(doc)
    assert out == "x = 2024-07-04\n"
    assert tomlrt.loads(out)["x"] == date(2024, 7, 4)


def test_assign_local_time() -> None:
    doc = tomlrt.loads("x = 0\n")
    doc["x"] = time(13, 30, 45)
    out = tomlrt.dumps(doc)
    assert out == "x = 13:30:45\n"
    assert tomlrt.loads(out)["x"] == time(13, 30, 45)


def test_assign_local_datetime() -> None:
    doc = tomlrt.loads("x = 0\n")
    doc["x"] = datetime(2024, 7, 4, 12, 0, 0)  # noqa: DTZ001
    out = tomlrt.dumps(doc)
    assert out == "x = 2024-07-04T12:00:00\n"
    assert tomlrt.loads(out)["x"] == datetime(2024, 7, 4, 12, 0, 0)  # noqa: DTZ001


def test_assign_offset_datetime() -> None:
    doc = tomlrt.loads("x = 0\n")
    tz = timezone(timedelta(hours=2))
    doc["x"] = datetime(2024, 7, 4, 12, 0, 0, tzinfo=tz)
    out = tomlrt.dumps(doc)
    assert out == "x = 2024-07-04T12:00:00+02:00\n"
    re_value = tomlrt.loads(out)["x"]
    assert isinstance(re_value, datetime)
    assert re_value == datetime(2024, 7, 4, 12, 0, 0, tzinfo=tz)


def test_assign_datetime_utc_offset() -> None:
    doc = tomlrt.loads("x = 0\n")
    doc["x"] = datetime(2024, 7, 4, 12, 0, 0, tzinfo=timezone.utc)
    out = tomlrt.dumps(doc)
    assert out == "x = 2024-07-04T12:00:00+00:00\n"
    re_value = tomlrt.loads(out)["x"]
    assert isinstance(re_value, datetime)
    assert re_value == datetime(2024, 7, 4, 12, 0, 0, tzinfo=timezone.utc)


def test_assign_datetime_seconds_offset_rejected() -> None:
    doc = tomlrt.loads("x = 0\n")
    tz = timezone(timedelta(hours=1, minutes=2, seconds=3))
    dt = datetime(2020, 1, 1, 10, 0, 0, tzinfo=tz)
    with pytest.raises(ValueError, match="whole number of minutes"):
        doc["x"] = dt


def test_assign_local_time_with_tzinfo_rejected() -> None:
    doc = tomlrt.loads("x = 0\n")
    t = time(10, 0, 0, tzinfo=timezone(timedelta(hours=2)))
    with pytest.raises(ValueError, match="local time cannot carry a timezone"):
        doc["x"] = t


def test_assign_plain_list_becomes_inline_array() -> None:
    doc = tomlrt.loads("x = 0\n")
    doc["x"] = [1, 2, 3]
    out = tomlrt.dumps(doc)
    assert out == "x = [1, 2, 3]\n"
    re = tomlrt.loads(out)
    assert list(re.array("x")) == [1, 2, 3]


def test_assign_plain_dict_becomes_inline_table() -> None:
    doc = tomlrt.loads("x = 0\n")
    doc["x"] = {"a": 1, "b": "two"}
    out = tomlrt.dumps(doc)
    assert out == 'x = { a = 1, b = "two" }\n'
    re = tomlrt.loads(out)
    tbl = re.table("x")
    assert tbl["a"] == 1
    assert tbl["b"] == "two"


def test_assign_tuple_rejected() -> None:
    doc = tomlrt.loads("x = 0\n")
    with pytest.raises(TypeError, match="tuple"):
        doc["x"] = (1, 2, 3)


def test_assign_mappingproxy_becomes_inline_table() -> None:
    from types import MappingProxyType  # noqa: PLC0415

    doc = tomlrt.loads("x = 0\n")
    doc["x"] = MappingProxyType({"a": 1, "b": 2})
    out = tomlrt.dumps(doc)
    assert out == "x = { a = 1, b = 2 }\n"
    re = tomlrt.loads(out)
    tbl = re.table("x")
    assert tbl["a"] == 1
    assert tbl["b"] == 2


def test_assign_bytes_rejected() -> None:
    doc = tomlrt.loads("x = 0\n")
    with pytest.raises(TypeError, match="bytes"):
        doc["x"] = b"hi"


def test_assign_nested_dict_in_list() -> None:
    doc = tomlrt.loads("x = 0\n")
    doc["x"] = [{"a": 1}, {"a": 2}]
    out = tomlrt.dumps(doc)
    assert out == "x = [{ a = 1 }, { a = 2 }]\n"
    re = tomlrt.loads(out)
    arr = re.array("x")
    assert arr.table(0)["a"] == 1
    assert arr.table(1)["a"] == 2


def test_assign_existing_array_deep_copies() -> None:
    src = tomlrt.loads("source = [1, 2, 3]\n")
    dest = tomlrt.loads("dest = []\n")
    dest["dest"] = src.array("source")
    src.array("source")[0] = 99
    # The mutation on `source` must not leak into `dest`.
    assert list(dest.array("dest")) == [1, 2, 3]


def test_assign_existing_inline_table_deep_copies() -> None:
    src = tomlrt.loads("source = {a = 1}\n")
    dest = tomlrt.loads("dest = {}\n")
    dest["dest"] = src.table("source")
    src.table("source")["a"] = 99
    assert dest.table("dest")["a"] == 1


def test_assign_unsupported_type_raises() -> None:
    doc = tomlrt.loads("x = 0\n")
    with pytest.raises(TypeError, match="cannot convert"):
        doc["x"] = object()


def test_assign_inline_table_with_unsupported_value_raises() -> None:
    """Synthesis recurses into mappings; the inner value gets the same check."""
    doc = tomlrt.loads("x = 0\n")
    with pytest.raises(TypeError, match="cannot convert"):
        doc["x"] = {"a": object()}


def test_assign_inline_array_with_unsupported_value_raises() -> None:
    doc = tomlrt.loads("x = 0\n")
    with pytest.raises(TypeError, match="cannot convert"):
        doc["x"] = [object()]


def test_assign_inline_table_with_non_str_key_raises() -> None:
    doc = tomlrt.loads("x = 0\n")
    with pytest.raises(TypeError, match="must be str"):
        doc["x"] = {1: "v"}


def test_assign_inline_table_containing_aot_value_rejected() -> None:
    """An ``AoT`` cannot live as the value of an inline-table entry."""
    src = tomlrt.loads(
        td("""
            [[products]]
            name = 'a'
            """),
    )
    dest = tomlrt.loads("dest = 0\n")
    aot = src.aot("products")
    with pytest.raises(TOMLError, match="array-of-tables"):
        dest["dest"] = {"items": aot}


def test_assign_inline_table_containing_section_container_rejected() -> None:
    """A section ``Table`` cannot live as the value of an inline-table entry."""
    src = tomlrt.loads(
        td("""
            [sub]
            x = 1
            """),
    )
    dest = tomlrt.loads("dest = 0\n")
    section = src.table("sub")
    with pytest.raises(TOMLError, match="section-style table"):
        dest["dest"] = {"nested": section}


def test_detached_inline_rejects_section_value_eagerly() -> None:
    """A detached ``Table.inline()`` used to silently accept a
    section-typed value, with the error deferred to attach time. The
    check is now eager: it fires at the actual point of mistake.
    """
    sub = Table.section({"y": 1})
    parent = Table.inline()
    with pytest.raises(TOMLError, match="section-style table"):
        parent["x"] = sub


def test_detached_inline_rejects_aot_value_eagerly() -> None:
    """Sibling of the section case: detached inline rejects AoT eagerly."""
    aot = tomlrt.AoT([{"name": "a"}])
    parent = Table.inline()
    with pytest.raises(TOMLError, match="array-of-tables"):
        parent["x"] = aot


def test_assign_aot_over_scalar() -> None:
    src = tomlrt.loads(
        td("""
            [[products]]
            name = 'a'
            [[products]]
            name = 'b'
            """),
    )
    dest = tomlrt.loads("dest = 0\n")
    dest["dest"] = src.aot("products")
    out = tomlrt.dumps(dest)
    assert out == td("""
        [[dest]]
        name = 'a'
        [[dest]]
        name = 'b'
        """)
    assert tomlrt.loads(out) == {
        "dest": [{"name": "a"}, {"name": "b"}],
    }


def test_document_factory_returns_empty_document() -> None:
    doc = Document()
    assert isinstance(doc, tomlrt.Document)
    assert len(doc) == 0
    assert tomlrt.dumps(doc) == ""


def test_document_factory_is_independent_of_other_calls() -> None:
    a = Document()
    b = Document()
    a["x"] = 1
    assert "x" not in b
    assert tomlrt.dumps(b) == ""


def test_document_factory_supports_full_build_and_dump() -> None:
    doc = Document()
    doc["title"] = "demo"
    doc["server"] = Table.section({"port": 8080})
    out = tomlrt.dumps(doc)
    assert out == td("""
        title = "demo"

        [server]
        port = 8080
        """)
    parsed = tomlrt.loads(out)
    assert parsed["title"] == "demo"
    server = parsed.table("server")
    assert server["port"] == 8080


def test_document_factory_with_data_uses_sections_for_nested_mappings() -> None:
    doc = Document({"server": {"port": 8080, "host": "localhost"}})
    out = tomlrt.dumps(doc)
    assert out == td("""
        [server]
        port = 8080
        host = "localhost"
        """)
    assert tomlrt.loads(out) == {"server": {"port": 8080, "host": "localhost"}}


def test_document_factory_with_data_uses_aot_for_list_of_mappings() -> None:
    doc = Document(
        {"package": [{"name": "foo"}, {"name": "bar"}]},
    )
    out = tomlrt.dumps(doc)
    assert out == td("""
        [[package]]
        name = "foo"

        [[package]]
        name = "bar"
        """)
    assert tomlrt.loads(out) == {"package": [{"name": "foo"}, {"name": "bar"}]}


def test_document_factory_with_explicit_array_keeps_inline_array() -> None:
    doc = Document({"xs": tomlrt.Array([{"a": 1}])})
    out = tomlrt.dumps(doc)
    assert out == "xs = [{ a = 1 }]\n"
    assert tomlrt.loads(out) == {"xs": [{"a": 1}]}


def test_document_factory_with_data_keeps_leaf_arrays_inline() -> None:
    doc = Document({"xs": [1, 2, 3]})
    out = tomlrt.dumps(doc)
    assert out == "xs = [1, 2, 3]\n"
    assert tomlrt.loads(out) == {"xs": [1, 2, 3]}


def test_document_factory_with_data_keeps_top_level_scalars_at_top() -> None:
    doc = Document({"title": "demo", "server": {"port": 8080}})
    out = tomlrt.dumps(doc)
    # Top-level scalar must precede the [server] section header.
    assert out == td("""
        title = "demo"

        [server]
        port = 8080
        """)


def test_document_factory_with_data_recurses_deeply() -> None:
    data = {
        "tool": {
            "poetry": {
                "name": "demo",
                "dependencies": {"requests": "^2.0"},
            },
        },
    }
    doc = Document(data)
    out = tomlrt.dumps(doc)
    assert out == td("""
        [tool.poetry]
        name = "demo"

        [tool.poetry.dependencies]
        requests = "^2.0"
        """)
    assert tomlrt.loads(out) == data


def test_document_factory_with_data_aot_with_nested_table() -> None:
    data = {
        "package": [
            {"name": "foo", "version": "1.0", "dep": {"x": 1}},
            {"name": "bar", "version": "2.0"},
        ],
    }
    doc = Document(data)
    out = tomlrt.dumps(doc)
    assert out == td("""
        [[package]]
        name = "foo"
        version = "1.0"

        [package.dep]
        x = 1

        [[package]]
        name = "bar"
        version = "2.0"
        """)
    assert tomlrt.loads(out) == data


def test_document_factory_with_empty_list_stays_inline_empty_array() -> None:
    doc = Document({"xs": []})
    out = tomlrt.dumps(doc)
    assert out == "xs = []\n"
    assert tomlrt.loads(out) == {"xs": []}


def test_document_factory_with_data_does_not_share_mutable_state() -> None:
    data: dict[str, object] = {"server": {"port": 8080}}
    doc = Document(data)
    server_dict = data["server"]
    assert isinstance(server_dict, dict)
    server_dict["port"] = 9999  # mutate the source after construction
    server = doc.table("server")
    assert server["port"] == 8080


def test_document_factory_with_data_passes_aot_through() -> None:
    """An existing ``AoT`` value passes straight through the init coercion."""
    src = tomlrt.loads(
        td("""
            [[products]]
            name = 'a'
            [[products]]
            name = 'b'
            """),
    )
    out = Document({"products": src.aot("products")})
    assert tomlrt.dumps(out) == td("""
        [[products]]
        name = 'a'
        [[products]]
        name = 'b'
        """)


def test_document_factory_with_data_passes_container_through() -> None:
    """An existing section ``Table`` value passes straight through init."""
    src = tomlrt.loads(
        td("""
            [sub]
            x = 1
            """),
    )
    out = Document({"sub": src.table("sub")})
    assert tomlrt.dumps(out) == td("""
        [sub]
        x = 1
        """)


# ``Document(...)`` copies: a view handed to it contributes its contents and
# its shape, and the document does not adopt the object. Assigning one still
# attaches it live. These pin that down.


def test_document_factory_detached_array_is_copied() -> None:
    arr = Array([1, 2, 3])
    doc = Document({"xs": arr})
    assert doc["xs"] is not arr
    arr.append(4)  # constructing copied it, so this is invisible
    out = tomlrt.dumps(doc)
    assert out == "xs = [1, 2, 3]\n"
    assert reparses(out) == {"xs": [1, 2, 3]}


def test_document_factory_detached_inline_table_is_copied() -> None:
    obj = Table.inline({"x": 1})
    doc = Document({"obj": obj})
    assert doc["obj"] is not obj
    obj["y"] = 2
    out = tomlrt.dumps(doc)
    assert out == "obj = { x = 1 }\n"
    assert reparses(out) == {"obj": {"x": 1}}


def test_document_factory_detached_section_is_copied() -> None:
    sec = Table.section({"x": 1})
    doc = Document({"sec": sec})
    assert doc["sec"] is not sec
    sec["y"] = 2
    out = tomlrt.dumps(doc)
    assert out == td("""
        [sec]
        x = 1
        """)
    assert reparses(out) == {"sec": {"x": 1}}


def test_document_factory_detached_aot_is_copied() -> None:
    aot = AoT([{"x": 1}])
    doc = Document({"srv": aot})
    assert doc["srv"] is not aot
    aot.append({"y": 2})
    out = tomlrt.dumps(doc)
    assert out == td("""
        [[srv]]
        x = 1
        """)
    assert reparses(out) == {"srv": [{"x": 1}]}


def test_document_factory_view_from_another_document_is_deep_cloned() -> None:
    src = tomlrt.loads("v = [1, 2, 3]\n")
    view = src.array("v")
    doc = Document({"xs": view})
    assert doc["xs"] is not view
    view.append(99)  # source view is detached from doc, so this is invisible
    out = tomlrt.dumps(doc)
    assert out == "xs = [1, 2, 3]\n"
    assert reparses(out) == {"xs": [1, 2, 3]}


def test_deepcopy_preserves_document_structure() -> None:

    src = td("""
        [a]
        x = 1

        [[b]]
        y = 2
        [[b]]
        y = 3
        """)
    doc1 = tomlrt.loads(src)
    doc2 = deepcopy(doc1)
    assert tomlrt.dumps(doc2) == src


@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
def test_cloned_comma_values_preserve_nested_layout(newline: str) -> None:
    src = td("""
        [source]
        values = [
            # first item
            { "odd key" . 'child' = [0x2A, 'literal'], empty = { }, }, # keep
            [true, 1979-05-27T07:32:00Z],
        ]
        """).replace("\n", newline)
    source = tomlrt.loads(src)
    target = tomlrt.loads(f"[source]{newline}")
    target["source"] = source.table("source")
    assert tomlrt.dumps(target) == src

    values = target.table("source").array("values")
    values.table(0).table("odd key").array("child")[0] = 7
    values.array(1)[0] = False
    expected = td("""
        [source]
        values = [
            # first item
            { "odd key" . 'child' = [7, 'literal'], empty = { }, }, # keep
            [false, 1979-05-27T07:32:00Z],
        ]
        """).replace("\n", newline)
    assert tomlrt.dumps(target) == expected
    assert tomlrt.dumps(source) == src
    assert reparses(expected) == target.to_dict()


@pytest.mark.parametrize(
    ("literal", "replacement", "rendered"),
    [
        ("'literal'", "changed", '"changed"'),
        ('"old\\tvalue"', "changed", '"changed"'),
        ('"""first\nsecond"""', "changed", '"changed"'),
        ("0x2A", 7, "7"),
        ("1_000.0", 3.25, "3.25"),
        ("true", False, "false"),
        (
            "1979-05-27T07:32:00Z",
            datetime(2000, 1, 2, tzinfo=timezone.utc),
            "2000-01-02T00:00:00+00:00",
        ),
        ("1979-05-27", date(2000, 1, 2), "2000-01-02"),
        ("07:32:00", time(1, 2, 3), "01:02:03"),
    ],
)
def test_cloned_scalars_remain_independent_under_mutation(
    literal: str, replacement: tomlrt.TomlInput, rendered: str
) -> None:
    src = td("""
        [source]
        direct = SCALAR # keep
        array = [SCALAR]
        inline = { value = SCALAR }
        """).replace("SCALAR", literal)
    source = tomlrt.loads(src)
    target = Document()
    target["copy"] = source.table("source")
    copied = target.table("copy")
    copied["direct"] = replacement
    copied.array("array")[0] = replacement
    copied.table("inline")["value"] = replacement
    assert tomlrt.dumps(source) == src
    assert tomlrt.dumps(target) == td("""
        [copy]
        direct = SCALAR # keep
        array = [SCALAR]
        inline = { value = SCALAR }
        """).replace("SCALAR", rendered)
    assert reparses(tomlrt.dumps(target)) == target.to_dict()


class _MutableOffset(tzinfo):
    def __init__(self) -> None:
        self.hours = 0

    @override
    def utcoffset(self, _dt: datetime | None) -> timedelta:
        return timedelta(hours=self.hours)

    @override
    def dst(self, _dt: datetime | None) -> timedelta:
        return timedelta(0)

    @override
    def tzname(self, _dt: datetime | None) -> str:
        return f"offset-{self.hours}"


def test_cloned_datetime_payloads_are_independent() -> None:
    zone = _MutableOffset()
    value = datetime(2020, 1, 1, tzinfo=zone)
    source = Document()
    source["source"] = Table.section(
        {
            "direct": value,
            "array": [value, value],
            "inline": Table.inline({"value": value}),
        }
    )
    target = Document()
    target["copy"] = source.table("source")
    zone.hours = 2
    copied = target.table("copy")["direct"]
    assert isinstance(copied, datetime)
    assert copied.tzname() == "offset-0"
    array = target.table("copy").array("array")
    assert array[0] is array[1]
    expected = td("""
        [copy]
        direct = 2020-01-01T00:00:00+00:00
        array = [2020-01-01T00:00:00+00:00, 2020-01-01T00:00:00+00:00]
        inline = { value = 2020-01-01T00:00:00+00:00 }
        """)
    assert tomlrt.dumps(target) == expected
    assert reparses(expected) == target.to_dict()


class _MutableInt(int):
    def __init__(self, _value: int) -> None:
        self.labels = ["original"]


class _MutableFloat(float):
    def __init__(self, _value: float) -> None:
        self.labels = ["original"]


class _MutableStr(str):
    __slots__ = ("labels",)

    def __init__(self, _value: str) -> None:
        self.labels = ["original"]


class _MutableDelta(timedelta):
    __slots__ = ("labels",)

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        self.labels = ["original"]


def test_cloned_timezone_payload_subclasses_are_independent() -> None:
    """An exact ``timezone`` is only as immutable as what it was built from.

    It retains the offset and name objects it was handed, so either one
    being a subclass with state of its own reaches mutable data through
    an otherwise shareable instance.
    """
    offset = _MutableDelta(hours=2)
    name = _MutableStr("plus-two")
    source = Document()
    source["source"] = Table.section(
        {
            "offset": datetime(2020, 1, 1, tzinfo=timezone(offset)),
            "named": datetime(2020, 1, 1, tzinfo=timezone(timedelta(hours=2), name)),
        }
    )
    target = Document()
    target["copy"] = source.table("source")
    offset.labels.append("changed")
    name.labels.append("changed")
    copied_offset = target.table("copy")["offset"]
    copied_named = target.table("copy")["named"]
    assert isinstance(copied_offset, datetime)
    assert isinstance(copied_named, datetime)
    carried_offset = copied_offset.utcoffset()
    carried_name = copied_named.tzname()
    assert isinstance(carried_offset, _MutableDelta)
    assert isinstance(carried_name, _MutableStr)
    assert carried_offset.labels == ["original"]
    assert carried_name.labels == ["original"]
    expected = td("""
        [copy]
        offset = 2020-01-01T00:00:00+02:00
        named = 2020-01-01T00:00:00+02:00
        """)
    assert tomlrt.dumps(target) == expected
    assert reparses(expected) == target.to_dict()


class _CopyAwareInt(_MutableInt):
    def __init__(self, value: int) -> None:
        super().__init__(value)
        self.owners: list[ArrayValue] = []

    def __deepcopy__(self, memo: dict[int, object]) -> Self:
        new = type(self)(self)
        memo[id(self)] = new
        new.labels = ["custom copy"]
        new.owners = deepcopy(self.owners, memo)
        return new


def test_comma_node_deepcopy_preserves_memo_aliases_and_payload_hooks() -> None:
    payload = _CopyAwareInt(42)
    scalar = IntegerValue("0x2A", payload)
    item = ArrayItem("", scalar, "", has_comma=True, post_comma_trivia=" ")
    node = ArrayValue([item, item])
    sibling = ArrayValue(node.items)
    payload.owners = [node, sibling]
    memo: dict[int, object] = {}

    cloned = deepcopy(node, memo)
    cloned_sibling = deepcopy(sibling, memo)
    assert cloned.items is cloned_sibling.items
    assert cloned.items is not node.items
    assert cloned.items[0] is cloned.items[1]
    assert cloned.items[0] is not item
    cloned_scalar = cloned.items[0].value
    assert isinstance(cloned_scalar, IntegerValue)
    cloned_payload = cloned_scalar.value
    assert isinstance(cloned_payload, _CopyAwareInt)
    assert cloned_payload is not payload
    assert cloned_payload.labels == ["custom copy"]
    assert cloned_payload.owners[0] is cloned
    assert cloned_payload.owners[1] is cloned_sibling
    assert deepcopy(node, memo) is cloned
    assert cloned.render() == "[0x2A, 0x2A, ]"
    assert cloned_sibling.render() == node.render() == cloned.render()


def test_comma_node_deepcopy_honors_preset_scalar_memo() -> None:
    scalar = IntegerValue("0x2A", 42)
    item = ArrayItem("", scalar, "", has_comma=False, post_comma_trivia="")
    node = ArrayValue([item])
    replacement = IntegerValue("7", 7)
    cloned = deepcopy(node, {id(scalar): replacement})

    assert cloned.items[0].value is replacement
    assert cloned.render() == "[7]"
    assert node.render() == "[0x2A]"


def test_comma_node_deepcopy_copies_mutable_trivia_and_key_fields() -> None:
    padding = _MutableStr(" ")
    key = KeyPart(_MutableStr("'key'"), _MutableStr("key"))
    entry = InlineTableEntry(
        "",
        ArrayValue(),
        "",
        has_comma=False,
        post_comma_trivia="",
        key_parts=(key,),
        key_seps=(),
        pre_eq=padding,
        post_eq=padding,
        key_path=(key.value,),
    )
    node = InlineTableValue([entry], padding, padding)
    assert not node.is_multiline()
    cloned = deepcopy(node)
    assert not cloned.is_multiline()
    cloned_entry = cloned.items[0]
    assert cloned_entry is not entry
    assert cloned_entry.key_parts[0] is not key
    assert cloned_entry.key_parts[0].value is cloned_entry.key_path[0]
    assert isinstance(cloned_entry.key_path[0], _MutableStr)
    assert cloned_entry.key_path[0] is not key.value
    assert cloned.header_trivia is cloned.final_trivia
    assert cloned.header_trivia is cloned_entry.pre_eq is cloned_entry.post_eq
    assert isinstance(cloned.header_trivia, _MutableStr)
    assert cloned.header_trivia is not padding
    padding.labels.append("changed")
    key.raw = "'changed'"
    assert cloned.header_trivia.labels == ["original"]
    assert cloned.render() == "{ 'key' = [] }"
    assert node.render() == "{ 'changed' = [] }"


class _IntLikeType(type):
    @override
    def __eq__(cls, other: object) -> bool:
        return other is int or super().__eq__(other)

    __hash__ = type.__hash__


class _IntLikeInt(_MutableInt, metaclass=_IntLikeType):
    pass


@pytest.mark.parametrize(
    ("factory", "rendered"),
    [
        (lambda: _MutableInt(1), "1"),
        (lambda: _IntLikeInt(1), "1"),
        (lambda: _MutableFloat(1.25), "1.25"),
        (lambda: _MutableStr("value"), '"value"'),
    ],
    ids=["int", "int-like-type", "float", "str"],
)
def test_cloned_scalar_subclass_payloads_are_independent(
    factory: Callable[[], _MutableInt | _MutableFloat | _MutableStr], rendered: str
) -> None:
    value = factory()
    source = Document()
    source["source"] = Table.section(
        {
            "direct": value,
            "array": [value],
            "inline": Table.inline({"value": value}),
        }
    )
    target = Document()
    target["copy"] = source.table("source")
    value.labels.append("changed")
    copied = target.table("copy")
    for item in (
        copied["direct"],
        copied.array("array")[0],
        copied.table("inline")["value"],
    ):
        assert isinstance(item, (_MutableInt, _MutableFloat, _MutableStr))
        assert item.labels == ["original"]
    expected = td("""
        [copy]
        direct = SCALAR
        array = [SCALAR]
        inline = { value = SCALAR }
        """).replace("SCALAR", rendered)
    assert tomlrt.dumps(target) == expected
    assert reparses(expected) == target.to_dict()


class _MutableLexeme(str):
    __slots__ = ("rendered",)

    rendered: str

    def __new__(cls, value: str) -> Self:
        lexeme = super().__new__(cls, value)
        lexeme.rendered = value
        return lexeme

    @override
    def __format__(self, _format_spec: str) -> str:
        return self.rendered


class _RenderedInt(int):
    def __init__(self, value: int) -> None:
        self.lexeme = _MutableLexeme(str(value))

    @override
    def __str__(self) -> str:
        return self.lexeme


def test_cloned_scalar_lexemes_are_independent() -> None:
    value = _RenderedInt(1)
    source = Document()
    source["source"] = Table.section(
        {
            "direct": value,
            "array": [value],
            "inline": Table.inline({"value": value}),
        }
    )
    target = Document()
    target["copy"] = source.table("source")
    value.lexeme.rendered = "9"
    assert tomlrt.dumps(source) == td("""
        [source]
        direct = 9
        array = [9]
        inline = { value = 9 }
        """)
    expected = td("""
        [copy]
        direct = 1
        array = [1]
        inline = { value = 1 }
        """)
    assert tomlrt.dumps(target) == expected
    assert reparses(expected) == target.to_dict()
    copied = target.table("copy")["direct"]
    assert isinstance(copied, _RenderedInt)
    assert copied.lexeme is not value.lexeme
    assert copied.lexeme.rendered == "1"


def test_plain_export_copies_raw_nested_containers_and_views() -> None:
    array = tomlrt.Array([2])
    raw: dict[str, Any] = {"items": [1, {"nested": array}], "pending": None}
    table = tomlrt.Table.section({"raw": raw})
    exported = table.to_dict()
    exported["raw"]["items"].append(3)
    exported["raw"]["items"][1]["nested"].append(4)
    raw["pending"] = "ready"
    assert exported == {"raw": {"items": [1, {"nested": [2, 4]}, 3], "pending": None}}
    doc = tomlrt.Document()
    doc["original"] = table
    assert tomlrt.dumps(doc) == td("""
        [original]
        raw = { items = [1, { nested = [2] }], pending = "ready" }
        """)


def test_deepcopy_yields_independent_document() -> None:

    src = "[a]\nx = 1\n"
    doc1 = tomlrt.loads(src)
    doc2 = deepcopy(doc1)
    doc2["a"]["x"] = 99
    assert doc1["a"]["x"] == 1
    assert doc2["a"]["x"] == 99
    # And the unmutated half stays format-preserved.
    assert tomlrt.dumps(doc1) == src


def test_copy_yields_independent_document() -> None:

    src = "[a]\nx = 1\n"
    doc1 = tomlrt.loads(src)
    doc2 = copy(doc1)
    doc2["a"]["x"] = 99
    assert doc1["a"]["x"] == 1


def test_deepcopy_table_subview_is_independent_and_round_trips() -> None:

    src = td("""
        [t]
        x = 1
        y = [ 1,2 ] # array
        """).replace("\n", "\r\n")
    doc = tomlrt.loads(src)
    t = doc.table("t")
    t2 = deepcopy(t)
    assert tomlrt.dumps(t2) == td("""
        x = 1
        y = [ 1,2 ] # array
        """).replace("\n", "\r\n")
    t2["x"] = 99
    t2["y"].append(3)
    fresh = tomlrt.loads("prefix = 0\r\n")
    fresh["copy"] = t2
    assert tomlrt.dumps(fresh) == td("""
        prefix = 0

        [copy]
        x = 99
        y = [ 1,2,3 ] # array
        """).replace("\n", "\r\n")
    assert tomlrt.dumps(doc) == src


def test_deepcopy_inline_table_preserves_inline_shape() -> None:
    src = "x = { a = 1, b = 2 }\n"
    doc = tomlrt.loads(src)
    t = doc.table("x")
    t2 = deepcopy(t)
    fresh = tomlrt.loads("")
    fresh["y"] = t2
    assert tomlrt.dumps(fresh) == "y = { a = 1, b = 2 }\n"


def test_deepcopy_array_subview_does_not_double_cst() -> None:

    src = "xs = [1, 2, 3]\n"
    doc = tomlrt.loads(src)
    arr = doc.array("xs")
    arr2 = deepcopy(arr)
    assert list(arr2) == [1, 2, 3]
    # The CST must not have doubled items: appending one and rendering
    # the array node in isolation should reflect exactly four entries.
    arr2.append(4)
    assert list(arr2) == [1, 2, 3, 4]
    # Re-attach the detached copy to a fresh document and render through
    # the public API: any doubled CST items would surface here.
    fresh = tomlrt.loads("")
    fresh["ys"] = arr2
    assert tomlrt.dumps(fresh) == "ys = [1, 2, 3, 4]\n"
    # Original is untouched.
    assert tomlrt.dumps(doc) == src


def test_deepcopy_aot_subview_preserves_length() -> None:

    src = td("""
        [[t]]
        x = 1 # first

        # second
        [[t]]
        x = 2
        """)
    doc = tomlrt.loads(src)
    aot = doc.aot("t")
    aot2 = deepcopy(aot)
    assert len(aot2) == 2
    assert [dict(e) for e in aot2] == [{"x": 1}, {"x": 2}]
    # Mutations on the copy do not leak.
    aot2[0]["x"] = 99
    assert aot[0]["x"] == 1
    fresh = tomlrt.Document()
    fresh["copy"] = aot2
    assert tomlrt.dumps(fresh) == td("""
        [[copy]]
        x = 99 # first

        # second
        [[copy]]
        x = 2
        """)
    assert tomlrt.dumps(doc) == src


def test_copy_array_preserves_layout_and_independent_nested_values() -> None:
    src = td("""
        xs = [
          0x1, # keep
          [ 2,3 ],
        ]
        """)
    doc = tomlrt.loads(src)
    arr = doc.array("xs")
    arr2 = copy(arr)
    arr2.array(1).append(4)
    fresh = tomlrt.loads("")
    fresh["ys"] = arr2
    assert tomlrt.dumps(fresh) == td("""
        ys = [
          0x1, # keep
          [ 2,3,4 ],
        ]
        """)
    assert tomlrt.dumps(doc) == src


def test_copy_aot_subview_preserves_length() -> None:

    src = td("""
        [[t]]
        x = 1
        [[t]]
        x = 2
        """)
    doc = tomlrt.loads(src)
    aot = doc.aot("t")
    aot2 = copy(aot)
    assert len(aot2) == 2


def test_copy_table_subview_is_independent() -> None:
    src = td("""
        [t]
        x = 1
        """)
    doc = tomlrt.loads(src)
    t = doc.table("t")
    t2 = copy(t)
    t2["x"] = 99
    assert t["x"] == 1
    assert t2["x"] == 99
    assert tomlrt.dumps(doc) == src


def test_copied_section_keeps_empty_aot_distinct_from_array() -> None:
    source = tomlrt.loads("[section]\narray = [] # ordinary\n")
    source.table("section")["pending"] = AoT()
    doc = Document()
    doc["copy"] = source.table("section")
    assert doc.table("copy").array("array") == []
    doc.table("copy").aot("pending").add({"id": 1})
    expected = td("""
        [copy]
        array = [] # ordinary

        [[copy.pending]]
        id = 1
        """)
    assert tomlrt.dumps(doc) == expected
    assert reparses(expected) == doc.to_dict()
    assert tomlrt.dumps(source) == td("""
        [section]
        array = [] # ordinary
        pending = []
        """)


def test_copied_aot_keeps_nested_empty_aots_in_their_entries() -> None:
    source = tomlrt.loads(
        td("""
        [[rows]]
        id = 1

        [[rows]]
        id = 2
        """)
    )
    for entry in source.aot("rows"):
        entry["pending"] = AoT()
    doc = Document()
    doc["copied"] = deepcopy(source.aot("rows"))
    rows = doc.aot("copied")
    assert rows[0].aot("pending") == []
    rows[1].aot("pending").add({"v": 3})
    expected = td("""
        [[copied]]
        id = 1
        pending = []

        [[copied]]
        id = 2

        [[copied.pending]]
        v = 3
        """)
    assert tomlrt.dumps(doc) == expected
    assert reparses(expected) == doc.to_dict()
    assert tomlrt.dumps(source) == td("""
        [[rows]]
        id = 1
        pending = []

        [[rows]]
        id = 2
        pending = []
        """)


def test_document_construction_keeps_empty_aot_kind() -> None:
    doc = Document({"pending": AoT(), "plain": []})
    assert doc.array("plain") == []
    assert tomlrt.dumps(doc) == "pending = []\nplain = []\n"
    doc.aot("pending").add({"id": 1})
    expected = td("""
        plain = []

        [[pending]]
        id = 1
        """)
    assert tomlrt.dumps(doc) == expected
    assert reparses(expected) == doc.to_dict()


def test_whole_document_copy_keeps_bytes_and_empty_aot_kind() -> None:
    text = "\ufeff" + td("""
        # title

        pending = []
        root = 0x2A # value

        [section]
        plain = []

        # ending
        """).replace("\n", "\r\n", 2).rstrip("\n")
    source = tomlrt.loads(text)
    source["pending"] = AoT()
    assert tomlrt.dumps(source) == text
    copied = copy(source)
    constructed = Document(source)
    assert copied.aot("pending") == []
    assert constructed.aot("pending") == []
    assert copied.table("section").array("plain") == []
    assert tomlrt.dumps(copied) == text
    assert tomlrt.dumps(constructed) == text
    assert tomlrt.dumps(source) == text


def test_document_copy_preserves_root_mapping_order() -> None:
    source = Document({"sub": {"y": 2}, "a": 1})
    constructed = Document(source)
    copied = copy(source)
    assert list(source) == ["sub", "a"]
    assert list(constructed) == ["sub", "a"]
    assert list(copied) == ["sub", "a"]
    expected = td("""
        a = 1

        [sub]
        y = 2
        """)
    assert tomlrt.dumps(source) == expected
    assert tomlrt.dumps(constructed) == expected
    assert tomlrt.dumps(copied) == expected


def test_factory_deepcopy_isolates_mutable_scalar_payloads() -> None:
    zone = _MutableOffset()
    value = datetime(2020, 1, 1, tzinfo=zone)
    source = Table.section({"values": [value, value], "count": _MutableInt(2)})
    cloned = deepcopy(source)
    zone.hours = 3
    count = source["count"]
    assert isinstance(count, _MutableInt)
    count.labels.append("changed")
    copied_count = cloned["count"]
    assert isinstance(copied_count, _MutableInt)
    assert copied_count.labels == ["original"]
    doc = Document()
    doc["copy"] = cloned
    values = doc.table("copy").array("values")
    assert values[0] is values[1]
    expected = td("""
        [copy]
        values = [2020-01-01T00:00:00+00:00, 2020-01-01T00:00:00+00:00]
        count = 2
        """)
    assert tomlrt.dumps(doc) == expected
    assert reparses(expected) == doc.to_dict()


def test_document_construction_isolates_mutable_scalar_payloads() -> None:
    zone = _MutableOffset()
    value = datetime(2020, 1, 1, tzinfo=zone)
    count = _MutableInt(2)
    doc = Document({"values": [value, value], "count": count})
    zone.hours = 3
    count.labels.append("changed")
    copied_count = doc["count"]
    assert isinstance(copied_count, _MutableInt)
    assert copied_count.labels == ["original"]
    values = doc.array("values")
    assert values[0] is values[1]
    expected = td("""
        values = [2020-01-01T00:00:00+00:00, 2020-01-01T00:00:00+00:00]
        count = 2
        """)
    assert tomlrt.dumps(doc) == expected
    assert reparses(expected) == doc.to_dict()


def test_document_construction_copies_externally_shared_scalar_text() -> None:
    lexeme = _MutableLexeme("2")

    class SharedTextInt(int):
        @override
        def __str__(self) -> str:
            return lexeme

    value = SharedTextInt(2)
    doc = Document({"value": value})
    lexeme.rendered = "9"
    assert tomlrt.dumps(doc) == "value = 2\n"
    assert tomlrt.dumps({"value": value}) == "value = 9\n"


def test_constructed_scalar_keeps_its_own_text_binding() -> None:
    original = _RenderedInt(2)
    doc = Document({"value": original})
    stored = doc["value"]
    assert isinstance(stored, _RenderedInt)
    stored.lexeme.rendered = "0x2"
    assert tomlrt.dumps(doc) == "value = 0x2\n"
    assert reparses(tomlrt.dumps(doc)) == doc.to_dict()
    assert tomlrt.dumps({"value": original}) == "value = 2\n"


def test_whole_document_copy_keeps_independent_scalar_subclasses() -> None:
    value = _MutableInt(2)
    source = Document()
    source["value"] = value
    copied = copy(source)
    constructed = Document(source)
    value.labels.append("changed")
    copied_value = copied["value"]
    constructed_value = constructed["value"]
    assert isinstance(copied_value, _MutableInt)
    assert isinstance(constructed_value, _MutableInt)
    assert copied_value.labels == constructed_value.labels == ["original"]
    assert copied_value is not constructed_value
    assert tomlrt.dumps(source) == "value = 2\n"
    assert tomlrt.dumps(copied) == "value = 2\n"
    assert tomlrt.dumps(constructed) == "value = 2\n"


def test_exports_isolate_mutable_scalar_payloads() -> None:
    zone = _MutableOffset()
    value = datetime(2020, 1, 1, tzinfo=zone)
    count = _MutableInt(2)
    doc = Document()
    doc["values"] = Array([value, value])
    doc["count"] = count
    exported = doc.to_dict()
    exported_list = doc.array("values").to_list()
    zone.hours = 3
    count.labels.append("changed")
    assert exported["values"][0] is exported["values"][1]
    assert exported_list[0] is exported_list[1]
    assert exported["values"][0].utcoffset() == timedelta(0)
    assert exported_list[0].utcoffset() == timedelta(0)
    assert exported["count"].labels == ["original"]
    expected = td("""
        values = [2020-01-01T00:00:00+00:00, 2020-01-01T00:00:00+00:00]
        count = 2
        """)
    assert tomlrt.dumps(doc) == expected
    assert tomlrt.dumps(exported) == expected
    assert tomlrt.dumps({"values": exported_list}) == td("""
        values = [2020-01-01T00:00:00+00:00, 2020-01-01T00:00:00+00:00]
        """)


def test_serializing_mapping_does_not_copy_scalar_payloads() -> None:
    class NonCopyableInt(int):
        def __deepcopy__(self, _memo: dict[int, object]) -> Self:
            msg = "serialization must not copy"
            raise AssertionError(msg)

    assert tomlrt.dumps({"value": NonCopyableInt(2)}) == "value = 2\n"


def test_deepcopy_table_subview_supports_nested_mutation() -> None:

    src = td("""
        [t]
        [t.inner]
        x = 1
        """)
    doc = tomlrt.loads(src)
    t = doc.table("t")
    t2 = deepcopy(t)
    t2.table("inner")["x"] = 42
    assert t.table("inner")["x"] == 1
    assert tomlrt.dumps(doc) == src


def test_deepcopy_table_subview_recurses_into_aot_child() -> None:
    """A section table containing an AoT clones the AoT as typed entries."""
    src = td("""
        [t]
        [[t.items]]
        x = 1
        [[t.items]]
        x = 2
        """)
    doc = tomlrt.loads(src)
    t = doc.table("t")
    t2 = deepcopy(t)
    # The clone still exposes the AoT as a typed list of tables.
    items = t2.aot("items")
    assert [dict(e) for e in items] == [{"x": 1}, {"x": 2}]
    # Mutations on the clone do not leak back to the original.
    items[0]["x"] = 99
    assert t.aot("items")[0]["x"] == 1
    assert tomlrt.dumps(doc) == src


def test_copy_dotted_inline_view_preserves_quoted_keys_and_values() -> None:
    src = "cfg = { scope.'g'.\"x\"=0x0f, other=9, scope.'g'.inner . y = 1_000 }\n"
    doc = tomlrt.loads(src)
    cloned = copy(doc.table(("cfg", "scope", "g")))
    fresh = tomlrt.Document()
    fresh["picked"] = cloned
    out = tomlrt.dumps(fresh)
    assert out == 'picked = { "x"=0x0f, inner . y = 1_000 }\n'
    assert reparses(out) == fresh.to_dict()
    assert tomlrt.dumps(doc) == src


def test_deepcopy_dotted_inline_view_keeps_owned_comments_not_framing() -> None:
    src = td("""
        cfg = { # outer
          # first
          group.x = 0x0f, # x
          other = 0, # foreign
          # older

          # y
          group.'y' = [ 1,2 ], # y-tail
          # closing
        }
        """).replace("\n", "\r\n")
    doc = tomlrt.loads(src)
    cloned = deepcopy(doc.table("cfg.group"))
    fresh = tomlrt.loads("prefix = 0\r\n")
    fresh["picked"] = cloned
    cloned["x"] = 2
    out = tomlrt.dumps(fresh)
    assert out == td("""
        prefix = 0
        picked = {
          # first
          x = 2, # x
          # older

          # y
          'y' = [ 1,2 ], # y-tail
        }
        """).replace("\n", "\r\n")
    assert reparses(out) == fresh.to_dict()
    assert tomlrt.dumps(doc) == src


def test_copy_dotted_inline_view_keeps_blocks_before_leading_commas() -> None:
    src = td("""
        cfg = {
          other = 0
          # keep
          , group.x = 0x0f # x
          , other2 = 2
        }
        """)
    doc = tomlrt.loads(src)
    fresh = tomlrt.Document()
    fresh["picked"] = copy(doc.table("cfg.group"))
    out = tomlrt.dumps(fresh)
    assert out == td("""
        picked = {
          # keep
          x = 0x0f   # x
        }
        """)
    assert reparses(out) == fresh.to_dict()
    assert tomlrt.dumps(doc) == src


def test_copy_dotted_inline_view_preserves_multiline_shape() -> None:
    src = "cfg = { group.x = 1,\n other = 2 }\n"
    doc = tomlrt.loads(src)
    fresh = tomlrt.Document()
    fresh["picked"] = copy(doc.table("cfg.group"))
    out = tomlrt.dumps(fresh)
    assert out == "picked = {\n x = 1,\n}\n"
    assert reparses(out) == fresh.to_dict()
    assert tomlrt.dumps(doc) == src


@pytest.mark.parametrize("operation", ["copy", "assign", "construct"])
def test_dotted_inline_extraction_keeps_nested_values_independent(
    operation: str,
) -> None:
    text = "cfg = { group.a=[ 1 ], foreign=[ 2 ], group.b={ x=0x03 } }\n"
    source = tomlrt.loads(text)
    group = source.table("cfg.group")
    if operation == "construct":
        doc = tomlrt.Document({"picked": group})
    else:
        doc = tomlrt.Document()
        doc["picked"] = copy(group) if operation == "copy" else group
    doc.table("picked").array("a").append(4)
    doc.table("picked.b")["x"] = 5
    expected = "picked = { a=[ 1, 4 ], b={ x=5 } }\n"
    assert tomlrt.dumps(doc) == expected
    assert reparses(expected) == doc.to_dict()
    assert tomlrt.dumps(source) == text


def test_copy_empty_held_inline_navigator() -> None:
    doc = tomlrt.loads("cfg = { group.x = 1 }\n")
    group = doc.table("cfg.group")
    del group["x"]
    before = tomlrt.dumps(doc)
    fresh = tomlrt.Document()
    fresh["picked"] = copy(group)
    assert tomlrt.dumps(fresh) == "picked = {}\n"
    assert tomlrt.dumps(doc) == before


def test_view_copies_independently_copy_unmaterialized_payloads() -> None:
    source = tomlrt.Table.section({"raw": {"values": [1]}})
    shallow = copy(source)
    deep = deepcopy(source)
    shallow["raw"]["values"].append(2)
    deep["raw"]["values"].append(3)
    doc = tomlrt.Document()
    doc["original"] = source
    doc["shallow"] = shallow
    doc["deep"] = deep
    out = tomlrt.dumps(doc)
    assert out == td("""
        [original]
        raw = { values = [1] }

        [shallow]
        raw = { values = [1, 2] }

        [deep]
        raw = { values = [1, 3] }
        """)
    assert reparses(out) == doc.to_dict()


@pytest.mark.parametrize("aot", [False, True])
def test_view_copy_validates_factory_before_adopting_children(*, aot: bool) -> None:
    child = tomlrt.Table.inline({"x": 1})
    bad: Any = object()
    body = {"child": child, "bad": bad}
    value = tomlrt.AoT([body]) if aot else tomlrt.Table.section(body)
    with pytest.raises(TypeError, match="cannot convert object"):
        copy(value)
    with pytest.raises(TypeError, match="cannot convert object"):
        deepcopy(value)
    fresh = tomlrt.Document()
    fresh["child"] = child
    child["x"] = 2
    assert tomlrt.dumps(fresh) == "child = { x = 2 }\n"


def test_typed_container_assign_now_clones_from_other_doc() -> None:
    # Assigning a section table from one document into another
    # deep-clones it; the two views are independent thereafter.
    src = tomlrt.loads("[a]\nx = 1\n")
    dst = tomlrt.loads("")
    dst["a"] = src["a"]
    assert tomlrt.dumps(dst) == "[a]\nx = 1\n"
    assert src["a"] is not dst["a"]
    src["a"]["x"] = 99
    assert dst["a"]["x"] == 1


def test_dumps_a_popped_subtree_leaves_it_intact() -> None:
    """Rendering a mapping must not consume it.

    A popped subtree's slots live in a private document, and installing
    from one of those is a move — right when a caller assigns a value
    somewhere, wrong when the value is only being read to build a new
    document from. Nested trivia survives, exactly as it does when the
    same subtree is still attached.
    """
    doc = tomlrt.loads(
        td("""
        [root]
        x = 1

        [root.sub]
        # lead
        y = 2  # eol
        """)
    )
    orphan = doc.pop("root")

    rendered = tomlrt.dumps(orphan)
    assert rendered == td("""
        x = 1

        [sub]
        # lead
        y = 2  # eol
        """)
    assert orphan.to_dict() == {"x": 1, "sub": {"y": 2}}
    # Still renderable, and still the same: the first dump consumed nothing.
    assert tomlrt.dumps(orphan) == rendered


def test_dumps_a_popped_subtree_holding_an_array_of_tables() -> None:
    """The same, for the shape that used to raise instead.

    Installing an AoT unbinds it from the document it came from, which
    while iterating that document's own items ended the iteration.
    """
    doc = tomlrt.loads(
        td("""
        [[root.t]]
        x = 1

        [[root.t]]
        x = 2
        """)
    )
    orphan = doc.pop("root")

    assert tomlrt.dumps(orphan) == td("""
        [[t]]
        x = 1

        [[t]]
        x = 2
        """)
    assert orphan.to_dict() == {"t": [{"x": 1}, {"x": 2}]}


def test_update_from_a_popped_subtree_leaves_it_intact() -> None:
    """``update`` reads its argument; it does not take it apart."""
    doc = tomlrt.loads(
        td("""
        [root]
        x = 1

        [root.sub]
        y = 2
        """)
    )
    orphan = doc.pop("root")

    built = tomlrt.Document()
    built.update(orphan)

    assert tomlrt.dumps(built) == td("""
        x = 1

        [sub]
        y = 2
        """)
    assert orphan.to_dict() == {"x": 1, "sub": {"y": 2}}


def test_copy_callback_can_adopt_a_sibling_into_another_document() -> None:
    other = Document()

    class MovingInt(int):
        def __deepcopy__(self, _memo: dict[int, object]) -> int:
            other["moved"] = sibling
            return int(self)

    source = tomlrt.loads(
        td("""
        [parent.a]
        value = 1
        [parent.b]
        x = 2
        """)
    )
    source.table("parent.a")["value"] = MovingInt(1)
    held = source.table("parent")
    del source["parent"]
    sibling = held.table("b")
    copied = copy(held.table("a"))
    assert other.table("moved") is sibling
    sibling["x"] = 3
    held.table("a")["value"] = 2
    target = Document()
    target["copy"] = copied
    source["remaining"] = held
    assert tomlrt.dumps(other) == td("""
        [moved]
        x = 3
        """)
    assert tomlrt.dumps(target) == td("""
        [copy]
        value = 1
        """)
    assert tomlrt.dumps(source) == td("""
        [remaining.a]
        value = 2
        """)


def test_failed_update_restores_normal_adoption_after_partial_progress() -> None:
    source = tomlrt.loads("[private]\nx = 1\n")
    private = source.table("private")
    del source["private"]
    target = tomlrt.loads("keep = 0\n")
    with pytest.raises(TypeError, match="cannot convert NoneType"):
        target.update({"copied": private, "bad": None})
    assert target.table("copied") is not private
    target["moved"] = private
    assert target.table("moved") is private
    private["x"] = 2
    expected = td("""
        keep = 0

        [copied]
        x = 1

        [moved]
        x = 2
        """)
    assert tomlrt.dumps(target) == expected
    assert reparses(expected) == target.to_dict()
    assert tomlrt.dumps(source) == ""


def test_nested_update_protection_is_scoped_to_its_destination() -> None:
    outer_doc = tomlrt.loads(
        td("""
        [root.source]
        value = 1
        [root.sibling]
        x = 2
        """)
    )
    outer = outer_doc.table("root")
    del outer_doc["root"]
    incoming = outer.table("source")
    sibling = outer.table("sibling")
    inner_doc = tomlrt.loads("[source]\ny = 3\n")
    inner = inner_doc.table("source")
    del inner_doc["source"]
    target = Document()

    class InnerInt(int):
        def __deepcopy__(self, _memo: dict[int, object]) -> int:
            target["outer_sibling"] = sibling
            return int(self)

    class OuterInt(int):
        def __deepcopy__(self, _memo: dict[int, object]) -> int:
            with pytest.raises(TypeError, match="cannot convert NoneType"):
                target.update({"inner": inner, "bad": None})
            target["outer_after"] = sibling
            target["inner_released"] = inner
            return int(self)

    inner["y"] = InnerInt(3)
    incoming["value"] = OuterInt(1)
    target.update({"copied": incoming})
    assert target.table("outer_sibling") is not sibling
    assert target.table("outer_after") is not sibling
    assert target.table("inner") is not inner
    assert target.table("inner_released") is inner
    sibling["x"] = 20
    incoming["value"] = 5
    inner["y"] = 4
    expected = td("""
        [outer_sibling]
        x = 2

        [inner]
        y = 3

        [outer_after]
        x = 2

        [inner_released]
        y = 4

        [copied]
        value = 1
        """)
    assert tomlrt.dumps(target) == expected
    assert reparses(expected) == target.to_dict()
    outer_doc["remaining"] = outer
    assert tomlrt.dumps(outer_doc) == td("""
        [remaining.source]
        value = 5
        [remaining.sibling]
        x = 20
        """)
    assert tomlrt.dumps(inner_doc) == ""


def test_update_adopts_roots_orphaned_by_the_current_write() -> None:
    source = tomlrt.loads("[template]\nvalue = 9\n")
    private = source.table("template")
    del source["template"]
    doc = tomlrt.loads("[outer]\n[outer.child]\nx = 1\n")
    child = doc.table("outer.child")
    doc.update({"private": private, "outer": child})
    assert doc.table("outer") is child
    assert doc.table("private") is not private
    expected = td("""
        [outer]
        x = 1
        [private]
        value = 9
        """)
    assert tomlrt.dumps(doc) == expected
    assert reparses(expected) == doc.to_dict()
    assert tomlrt.dumps(private) == "value = 9\n"
    assert tomlrt.dumps(source) == ""


def test_factory_update_keeps_references_for_later_attachment() -> None:
    source = tomlrt.loads("[template]\nvalue = 9\n")
    private = source.table("template")
    del source["template"]
    factory = Table.section()
    factory.update({"child": private})
    target = Document()
    target["parent"] = factory
    assert target.table("parent") is factory
    assert target.table("parent.child") is private
    private["value"] = 10
    assert tomlrt.dumps(target) == td("""
        [parent.child]
        value = 10
        """)
    assert tomlrt.dumps(source) == ""


def test_update_keeps_fresh_aot_entries_live_but_copies_existing_layout() -> None:
    source = tomlrt.loads("[template]\nx = 1 # keep\n")
    factory = AoT([source.table("template"), {"x": 2}])
    layout_entry, fresh_entry = factory
    target = Document()
    target.update({"rows": factory})
    assert target.aot("rows") is factory
    assert factory[0] is not layout_entry
    assert factory[1] is fresh_entry
    layout_entry["x"] = 8
    fresh_entry["x"] = 3
    expected = td("""
        [[rows]]
        x = 1 # keep

        [[rows]]
        x = 3
        """)
    assert tomlrt.dumps(target) == expected
    assert reparses(expected) == target.to_dict()
    assert tomlrt.dumps(layout_entry) == "x = 8 # keep\n"
    assert tomlrt.dumps(source) == "[template]\nx = 1 # keep\n"


def test_update_copies_a_private_aot_without_preventing_later_adoption() -> None:
    source = tomlrt.loads("[[items]]\nx = 1\n")
    held = source.aot("items")
    del source["items"]
    target = Document()
    target.update({"copied": held})
    assert target.aot("copied") is not held
    target["moved"] = held
    assert target.aot("moved") is held
    held.add({"x": 2})
    expected = td("""
        [[copied]]
        x = 1

        [[moved]]
        x = 1

        [[moved]]
        x = 2
        """)
    assert tomlrt.dumps(target) == expected
    assert reparses(expected) == target.to_dict()
    assert tomlrt.dumps(source) == ""


def test_dumps_a_popped_subtree_wrapped_in_plain_mappings() -> None:
    """A source one level down is still the caller's.

    The plain mappings around it are rebuilt on the way in, but the view
    inside them is installed as it is — so it has to be recognised
    through them, not just at the top level.
    """
    doc = tomlrt.loads(
        td("""
        [root]
        x = 1

        [root.sub]
        # lead
        y = 2  # eol
        """)
    )
    orphan = doc.pop("root")

    assert tomlrt.dumps({"a": {"b": orphan["sub"]}}) == td("""
        [a.b]
        # lead
        y = 2  # eol
        """)
    assert orphan.to_dict() == {"x": 1, "sub": {"y": 2}}
