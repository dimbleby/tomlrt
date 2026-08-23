"""Tests for value synthesis (``value_to_node``) and the public file I/O.

These cover the corners of ``_synthesise.py`` and ``_public.py`` that
the rest of the suite skirts past: every escape branch in basic
strings, every scalar flavour accepted by ``value_to_node``, and the
``loads`` / ``load`` / ``dump`` wrappers.
"""

from __future__ import annotations

import io
import math
from copy import copy, deepcopy
from datetime import date, datetime, time, timedelta, timezone
from typing import TYPE_CHECKING

import pytest

import tomlrt
from _helpers import reparses, td
from tomlrt import AoT, Array, Document, Table, TOMLError

if TYPE_CHECKING:
    from pathlib import Path


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
        y = 2
        """)
    doc = tomlrt.loads(src)
    t = doc.table("t")
    t2 = deepcopy(t)
    assert dict(t2) == {"x": 1, "y": 2}
    t2["x"] = 99
    assert t["x"] == 1
    assert t2["x"] == 99
    # The original document's bytes are unaffected.
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
        x = 1
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
    assert tomlrt.dumps(doc) == src


def test_copy_array_subview_does_not_double_cst() -> None:

    src = "xs = [1, 2, 3]\n"
    doc = tomlrt.loads(src)
    arr = doc.array("xs")
    arr2 = copy(arr)
    arr2.append(4)
    fresh = tomlrt.loads("")
    fresh["ys"] = arr2
    assert tomlrt.dumps(fresh) == "ys = [1, 2, 3, 4]\n"
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
