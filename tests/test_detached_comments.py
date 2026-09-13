"""Comments on table factories, before their final attachment."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import tomlrt
from _helpers import reparses, td
from tomlrt import AoT, Array, Table

if TYPE_CHECKING:
    from collections.abc import Callable


def test_section_factory_comments_survive_attachment_and_edits() -> None:
    table = Table.section({"enabled": True, "count": 1})
    table.comments["enabled"] = "switch"
    table.leading_comments["enabled"] = ("about",)
    table.leading_block["count"] = ("older", None, "count")
    table.header_comment = "feature"
    table.header_leading_block = ("heading", None, "old title")
    table.header_leading_comments = ("section",)
    table.comments["count"] = "temporary"
    del table.comments["count"]
    table["count"] = 2

    doc = tomlrt.loads("prefix = 1\r\n")
    doc["feature"] = table
    assert doc.table("feature") is table
    table["count"] = 3
    out = tomlrt.dumps(doc)
    assert out == td("""
        prefix = 1

        # heading

        # section
        [feature] # feature
        # about
        enabled = true # switch
        # older

        # count
        count = 3
        """).replace("\n", "\r\n")
    assert reparses(out) == doc.to_dict()


def test_inline_factory_comments_survive_attachment_and_edits() -> None:
    table = Table.inline({"a": 1, "b": 2})
    comments = table.comments
    assert dict(comments) == {}
    table.leading_block["b"] = ("older", None, "old title")
    table.leading_comments["b"] = ("latest",)
    table.leading_comments["a"] = ("first",)
    comments["a"] = "note"

    doc = tomlrt.loads("prefix = 1\r\n")
    doc["item"] = table
    assert doc.table("item") is table
    table["a"] = 3
    out = tomlrt.dumps(doc)
    assert out == td("""
        prefix = 1
        item = {
            # first
            a = 3, # note
            # older

            # latest
            b = 2,
        }
        """).replace("\n", "\r\n")
    assert reparses(out) == doc.to_dict()


def test_inline_leading_comments_can_initialize_layout() -> None:
    table = Table.inline({"x": 1})
    table.leading_comments["x"] = ("first",)
    doc = tomlrt.Document()
    doc["feature"] = table
    out = tomlrt.dumps(doc)
    assert out == td("""
        feature = {
            # first
            x = 1,
        }
        """)
    assert reparses(out) == doc.to_dict()


@pytest.mark.parametrize("factory", [Table.section, Table.inline])
def test_factory_reads_noops_and_errors_leave_children_free(
    factory: Callable[[dict[str, tomlrt.TomlInput]], Table],
) -> None:
    child = Table.inline({"x": 1})
    table = factory({"child": child, "value": 1})
    assert dict(table.comments) == {}
    assert dict(table.leading_comments) == {}
    assert dict(table.leading_block) == {}
    assert table.comments.get("value") is None
    assert "value" not in table.comments
    with pytest.raises(KeyError):
        del table.comments["value"]
    with pytest.raises(KeyError):
        del table.leading_comments["value"]
    with pytest.raises(KeyError):
        del table.leading_block["value"]
    table.leading_comments["value"] = ()
    table.leading_block["value"] = ()
    with pytest.raises(KeyError):
        table.comments["missing"] = "note"
    with pytest.raises(KeyError):
        table.leading_comments["missing"] = ()
    bad_key: Any = 0
    with pytest.raises(KeyError):
        table.comments[bad_key] = "note"
    bad: Any = 42
    with pytest.raises(TypeError):
        table.comments["value"] = bad
    with pytest.raises(TypeError):
        table.leading_comments["value"] = bad
    with pytest.raises(TypeError):
        table.leading_block["value"] = bad
    if table.is_inline:
        with pytest.raises(tomlrt.TOMLError, match="header comment API"):
            table.header_comment = "not a section"
    else:
        assert table.header_comment is None
        assert table.header_leading_comments == ()
        assert table.header_leading_block == ()
        table.header_comment = None
        table.header_leading_comments = ()
        table.header_leading_block = ()

    doc = tomlrt.Document()
    doc["child"] = child
    child["x"] = 2
    assert tomlrt.dumps(doc) == "child = { x = 2 }\n"


@pytest.mark.parametrize("factory", [Table.section, Table.inline])
def test_invalid_factory_payload_does_not_adopt_an_earlier_child(
    factory: Callable[[dict[str, tomlrt.TomlInput]], Table],
) -> None:
    child = Table.inline({"x": 1})
    bad: Any = object()
    table = factory({"child": child, "bad": bad})
    table.leading_comments["child"] = ()
    table.leading_block["child"] = ()
    with pytest.raises(TypeError, match="cannot convert object"):
        table.comments["child"] = "note"
    doc = tomlrt.Document()
    doc["child"] = child
    child["x"] = 2
    assert tomlrt.dumps(doc) == "child = { x = 2 }\n"


def test_section_child_is_not_a_direct_comment_target() -> None:
    child = Table.section({"x": 1})
    parent = Table.section({"child": child, "rows": AoT([{"x": 1}])})
    with pytest.raises(KeyError):
        parent.comments["child"] = "note"
    with pytest.raises(KeyError):
        parent.leading_comments["rows"] = ("note",)
    doc = tomlrt.Document()
    doc["child"] = child
    assert parent.table("child") is child
    assert tomlrt.dumps(doc) == "[child]\nx = 1\n"


def test_empty_aot_placeholder_accepts_a_factory_comment() -> None:
    table = Table.section({"rows": AoT()})
    table.comments["rows"] = "none yet"
    doc = tomlrt.Document()
    doc["feature"] = table
    out = tomlrt.dumps(doc)
    assert out == "[feature]\nrows = [] # none yet\n"
    assert reparses(out) == doc.to_dict()


@pytest.mark.parametrize("children_first", [False, True])
def test_factory_header_comments_survive_children_and_sort(
    *, children_first: bool
) -> None:
    table = Table.section()
    if children_first:
        table["child"] = Table.section({"x": 1})
    else:
        table["temporary"] = 1
        table.comments["temporary"] = "removed"
        del table["temporary"]
    table.header_comment = "keep"
    if not children_first:
        table["child"] = Table.section({"x": 1})
    table.sort()
    doc = tomlrt.Document()
    doc["feature"] = table
    out = tomlrt.dumps(doc)
    assert out == td("""
        [feature] # keep

        [feature.child]
        x = 1
        """)
    assert reparses(out) == doc.to_dict()


def test_headerless_factory_keeps_child_header_blocks_when_attached() -> None:
    child = Table.section({"x": 1})
    table = Table.section({"temporary": 1})
    table.comments["temporary"] = "removed"
    del table["temporary"]
    table["child"] = child
    child.header_leading_block = ("older", None, "child")
    doc = tomlrt.Document()
    doc["feature"] = table
    out = tomlrt.dumps(doc)
    assert out == td("""
        # older

        # child
        [feature.child]
        x = 1
        """)
    assert reparses(out) == doc.to_dict()


@pytest.mark.parametrize("child_first", [False, True])
def test_nested_section_factories_keep_live_views(*, child_first: bool) -> None:
    child = Table.section({"x": 1})
    parent = Table.section({"enabled": True, "child": child})
    if child_first:
        child.comments["x"] = "child"
    parent.comments["enabled"] = "parent"
    if not child_first:
        child.comments["x"] = "child"
    doc = tomlrt.Document()
    doc["feature"] = parent
    assert doc.table("feature.child") is child
    child["x"] = 2
    out = tomlrt.dumps(doc)
    assert out == td("""
        [feature]
        enabled = true # parent

        [feature.child]
        x = 2 # child
        """)
    assert reparses(out) == doc.to_dict()


def test_nested_inline_factories_keep_live_views() -> None:
    child = Table.inline({"x": 1})
    child.comments["x"] = "child"
    parent = Table.inline({"child": child})
    parent.comments["child"] = "parent"
    doc = tomlrt.Document()
    doc["feature"] = parent
    assert doc.table("feature.child") is child
    child["x"] = 2
    out = tomlrt.dumps(doc)
    assert out == td("""
        feature = {
            child = {
            x = 2, # child
        }, # parent
        }
        """)
    assert reparses(out) == doc.to_dict()


def test_materializing_factory_clones_an_already_owned_inline_child() -> None:
    array = Array([{"x": 1}])
    source = array.table(0)
    table = Table.inline({"child": source})
    table.comments["child"] = "copy"
    assert table.table("child") is not source
    doc = tomlrt.Document()
    doc["original"] = array
    doc["copy"] = table
    table.table("child")["x"] = 2
    out = tomlrt.dumps(doc)
    assert out == td("""
        original = [{ x = 1 }]
        copy = {
            child = { x = 2 }, # copy
        }
        """)
    assert reparses(out) == doc.to_dict()


def test_annotated_section_is_copied_into_a_standalone_aot() -> None:
    source = Table.section({"enabled": True})
    source.comments["enabled"] = "original"
    source.header_comment = "entry"
    values = AoT([source])
    entry = values[0]
    assert entry is not source
    doc = tomlrt.Document()
    doc["features"] = values
    assert values[0] is entry
    source.comments["enabled"] = "changed"
    out = tomlrt.dumps(doc)
    assert out == "[[features]] # entry\nenabled = true # original\n"
    assert reparses(out) == doc.to_dict()
    assert tomlrt.dumps(source) == td("""
        # entry

        enabled = true # changed
        """)


def test_annotated_aot_entry_keeps_nested_views_and_ownership() -> None:
    values = AoT(
        [
            {
                "enabled": True,
                "child": Table.section({"x": 1}),
                "nested": AoT([{"id": "old"}]),
            }
        ]
    )
    entry = values[0]
    child = entry.table("child")
    nested = entry.aot("nested")[0]
    nested.comments["id"] = "nested"
    entry.comments["enabled"] = "entry"
    entry.header_comment = "feature"
    child.comments["x"] = "child"
    doc = tomlrt.Document()
    doc["features"] = values
    assert doc.pop("features") is values
    assert tomlrt.dumps(doc) == ""
    doc = tomlrt.Document()
    doc["features"] = values
    assert values[0] is entry
    assert entry.table("child") is child
    assert entry.aot("nested")[0] is nested
    entry["enabled"] = False
    child["x"] = 3
    nested["id"] = "new"
    out = tomlrt.dumps(doc)
    assert out == td("""
        [[features]] # feature
        enabled = false # entry

        [features.child]
        x = 3 # child

        [[features.nested]]
        id = "new" # nested
        """)
    assert reparses(out) == doc.to_dict()
    values.clear()
    assert tomlrt.dumps(doc) == "features = []\n"


def test_annotated_section_can_be_inserted_beneath_an_aot_entry() -> None:
    table = Table.section({"enabled": True})
    table.comments["enabled"] = "note"
    doc = tomlrt.loads("[[items]]\nx = 1\n")
    entry = doc.aot("items")[0]
    entry["sub"] = table
    assert entry.table("sub") is table
    table["more"] = 2
    entry["after"] = 3
    out = tomlrt.dumps(doc)
    assert out == td("""
        [[items]]
        x = 1
        after = 3

        [items.sub]
        enabled = true # note
        more = 2
        """)
    assert reparses(out) == doc.to_dict()
    doc.aot("items").clear()
    assert tomlrt.dumps(doc) == "items = []\n"


def test_materialized_factory_entry_already_used_elsewhere_is_copied() -> None:
    values = AoT([{"x": 1}])
    source = values[0]
    source.comments["x"] = "note"
    original = tomlrt.loads("[[outer]]\nid = 1\n")
    original.aot("outer")[0]["sub"] = source
    doc = tomlrt.Document()
    doc["features"] = values
    assert values[0] is not source
    values[0]["x"] = 2
    out = tomlrt.dumps(doc)
    assert out == "[[features]]\nx = 2 # note\n"
    assert reparses(out) == doc.to_dict()
    source["x"] = 3
    original_out = tomlrt.dumps(original)
    assert original_out == td("""
        [[outer]]
        id = 1

        [outer.sub]
        x = 3 # note
        """)
    assert reparses(original_out) == original.to_dict()
    assert tomlrt.dumps(doc) == out


def test_private_aot_reattachment_preserves_entry_separators() -> None:
    original = tomlrt.loads(
        td("""
        [[rows]]
        x = 1


        [[rows]]
        x = 2 # second
        """)
    )
    values = original.aot("rows")
    first, second = values
    assert original.pop("rows") is values
    doc = tomlrt.loads("prefix = 1\r\n")
    doc["rows"] = values
    assert values[0] is first
    assert values[1] is second
    first["x"] = 3
    out = tomlrt.dumps(doc)
    assert out == td("""
        prefix = 1

        [[rows]]
        x = 3


        [[rows]]
        x = 2 # second
        """).replace("\n", "\r\n")
    assert reparses(out) == doc.to_dict()
    assert tomlrt.dumps(original) == ""


@pytest.mark.parametrize(
    ("attached", "annotated"), [(False, False), (False, True), (True, True)]
)
def test_aot_repetition_keeps_layout_and_independent_values(
    *, attached: bool, annotated: bool
) -> None:
    values = AoT([{"x": 1, "nested": {"items": [1]}}])
    original = values[0]
    if annotated:
        original.comments["x"] = "keep"
    doc = tomlrt.Document()
    if attached:
        doc["rows"] = values
    values *= 3
    values[1]["nested"]["items"].append(2)
    if not attached:
        doc["rows"] = values
    assert values[0] is original
    eol = " # keep" if annotated else ""
    out = tomlrt.dumps(doc)
    assert out == td(f"""
        [[rows]]
        x = 1{eol}
        nested = {{ items = [1] }}

        [[rows]]
        x = 1{eol}
        nested = {{ items = [1, 2] }}

        [[rows]]
        x = 1{eol}
        nested = {{ items = [1] }}
        """)
    assert reparses(out) == doc.to_dict()


def test_empty_standalone_aot_repetition_is_noop() -> None:
    values = AoT()
    values *= 3
    doc = tomlrt.Document()
    doc["rows"] = values
    assert tomlrt.dumps(doc) == "rows = []\n"


def test_invalid_repeated_aot_does_not_adopt_an_earlier_child() -> None:
    child = Table.inline({"x": 1})
    bad: Any = object()
    values = AoT([{"child": child}, {"bad": bad}])
    with pytest.raises(TypeError):
        values *= 2
    doc = tomlrt.Document()
    doc["child"] = child
    child["x"] = 2
    assert tomlrt.dumps(doc) == "child = { x = 2 }\n"


def test_aot_repetition_preserves_layout_inside_unmaterialized_wrappers() -> None:
    source = tomlrt.loads(
        td("""
        inline = { x=0x10 }
        array = [ 1,2, ]
        [[existing]]
        id = 1 # row
        [section]
        x = 0x20 # section
        """)
    )
    inline, array = source.table("inline"), source.array("array")
    del source["inline"]
    del source["array"]
    values = AoT(
        [
            {
                "wrapped": {
                    "items": [inline, array],
                    "fresh": Table.inline({"v": [1]}),
                },
                "section": source.table("section"),
                "existing": source.aot("existing"),
                "new": AoT([{"x": 1}]),
            }
        ]
    )
    values *= 3
    changed = values[1]
    changed["wrapped"]["items"][0]["x"] = 1
    changed["wrapped"]["items"][1].append(3)
    changed["wrapped"]["fresh"]["v"].append(2)
    changed.table("section")["x"] = 2
    changed.aot("existing")[0]["id"] = 2
    changed.aot("new")[0]["x"] = 2
    doc = tomlrt.Document()
    doc["rows"] = values
    out = tomlrt.dumps(doc)
    assert out == td("""
        [[rows]]
        wrapped = { items = [{ x=0x10 }, [ 1,2, ]], fresh = { v = [1] } }

        [rows.section]
        x = 0x20 # section

        [[rows.existing]]
        id = 1 # row

        [[rows.new]]
        x = 1

        [[rows]]
        wrapped = { items = [{ x=1 }, [ 1,2,3, ]], fresh = { v = [1, 2] } }

        [rows.section]
        x = 2 # section

        [[rows.existing]]
        id = 2 # row

        [[rows.new]]
        x = 2

        [[rows]]
        wrapped = { items = [{ x=0x10 }, [ 1,2, ]], fresh = { v = [1] } }

        [rows.section]
        x = 0x20 # section

        [[rows.existing]]
        id = 1 # row

        [[rows.new]]
        x = 1
        """)
    assert reparses(out) == doc.to_dict()
    assert tomlrt.dumps(source) == td("""
        [[existing]]
        id = 1 # row
        [section]
        x = 0x20 # section
        """)


def test_headerless_materialized_entry_can_be_attached() -> None:
    values = AoT([{"temporary": 1}])
    entry = values[0]
    entry.comments["temporary"] = "removed"
    del entry["temporary"]
    child = Table.section({"x": 1})
    entry["child"] = child
    child.comments["x"] = "kept"
    doc = tomlrt.Document()
    doc["features"] = values
    entry["new"] = 2
    child["x"] = 3
    assert values[0] is entry
    out = tomlrt.dumps(doc)
    assert out == td("""
        [[features]]
        new = 2
        [features.child]
        x = 3 # kept
        """)
    assert reparses(out) == doc.to_dict()


def test_aot_factory_preserves_forward_declared_source_and_held_views() -> None:
    source = tomlrt.loads(
        td("""
        [a.child]
        x = 1 # child
        [a]
        z = 2 # root
        """)
    )
    values = AoT([source.table("a")])
    entry = values[0]
    child = entry.table("child")
    doc = tomlrt.Document()
    doc["features"] = values
    assert values[0] is entry
    assert entry.table("child") is child
    entry["z"] = 4
    entry["new"] = 5
    child["x"] = 3
    out = tomlrt.dumps(doc)
    assert out == td("""
        [[features]]
        z = 4 # root
        new = 5
        [features.child]
        x = 3 # child
        """)
    assert reparses(out) == doc.to_dict()
    assert tomlrt.dumps(source) == td("""
        [a.child]
        x = 1 # child
        [a]
        z = 2 # root
        """)
