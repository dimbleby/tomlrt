"""Copy/deepcopy semantics, layout preservation, and independent exported data."""

from __future__ import annotations

import sys
from copy import copy, deepcopy
from datetime import UTC, date, datetime, time, timedelta, timezone, tzinfo
from typing import TYPE_CHECKING, Any

import pytest

import tomlrt
from _helpers import reparses, td
from tomlrt import AoT, Array, Document, Table
from tomlrt._values import (
    ArrayItem,
    ArrayValue,
    InlineTableEntry,
    InlineTableValue,
    IntegerValue,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Self

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override


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
            datetime(2000, 1, 2, tzinfo=UTC),
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
    item = ArrayItem(scalar, has_comma=True, after=" ")
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
    item = ArrayItem(scalar)
    node = ArrayValue([item])
    replacement = IntegerValue("7", 7)
    cloned = deepcopy(node, {id(scalar): replacement})

    assert cloned.items[0].value is replacement
    assert cloned.render() == "[7]"
    assert node.render() == "[0x2A]"


def test_comma_node_deepcopy_copies_mutable_trivia_and_key_fields() -> None:
    padding = _MutableStr(" ")
    key_parts = (_MutableStr("'key'"),)
    key_path = (_MutableStr("key"),)
    entry = InlineTableEntry(
        ArrayValue(),
        key_parts=key_parts,
        key_seps=(),
        key_path=key_path,
        pre_eq=padding,
        post_eq=padding,
        following=padding,
    )
    node = InlineTableValue([entry], padding)
    assert not node.is_multiline()
    cloned = deepcopy(node)
    assert not cloned.is_multiline()
    cloned_entry = cloned.items[0]
    assert cloned_entry is not entry
    assert cloned_entry.key_parts is not key_parts
    assert isinstance(cloned_entry.key_path[0], _MutableStr)
    assert cloned_entry.key_path[0] is not key_path[0]
    assert cloned.opening is cloned_entry.following
    assert cloned.opening is cloned_entry.pre_eq is cloned_entry.post_eq
    assert isinstance(cloned.opening, _MutableStr)
    assert cloned.opening is not padding
    padding.labels.append("changed")
    entry.key_parts = ("'changed'",)
    assert cloned.opening.labels == ["original"]
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


@pytest.mark.parametrize("copier", [copy, deepcopy], ids=["copy", "deepcopy"])
def test_copy_yields_independent_document(
    copier: Callable[[Document], Document],
) -> None:
    src = "[a]\nx = 1\n"
    doc1 = tomlrt.loads(src)
    doc2 = copier(doc1)
    doc2["a"]["x"] = 99
    assert doc1["a"]["x"] == 1
    assert doc2["a"]["x"] == 99
    # And the unmutated half stays format-preserved.
    assert tomlrt.dumps(doc1) == src


@pytest.mark.parametrize("copier", [copy, deepcopy], ids=["copy", "deepcopy"])
def test_copy_table_subview_is_independent_and_round_trips(
    copier: Callable[[Table], Table],
) -> None:
    src = td("""
        [t]
        x = 1
        y = [ 1,2 ] # array
        """).replace("\n", "\r\n")
    doc = tomlrt.loads(src)
    t = doc.table("t")
    t2 = copier(t)
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


@pytest.mark.parametrize("copier", [copy, deepcopy], ids=["copy", "deepcopy"])
def test_copy_aot_subview_preserves_layout_and_independence(
    copier: Callable[[AoT], AoT],
) -> None:
    src = td("""
        [[t]]
        x = 1 # first

        # second
        [[t]]
        x = 2
        """)
    doc = tomlrt.loads(src)
    aot = doc.aot("t")
    aot2 = copier(aot)
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


def test_aot_factory_copies_a_deep_private_section() -> None:
    path = ("child",) * 1500
    body = f"{'.'.join(path)}.value = 1\n"
    doc = tomlrt.loads(f"root.{body}")
    source = doc.pop("root")
    assert isinstance(source, Table)
    rows = AoT([source])
    assert rows[0] is not source
    copied = Document(rows[0])
    copied.table(path)["value"] = 2
    assert tomlrt.dumps(copied) == body.replace(" = 1", " = 2")
    assert tomlrt.dumps(Document(source)) == body
    assert tomlrt.dumps(doc) == ""


def test_whole_document_copy_keeps_bytes_and_empty_aot_kind() -> None:
    text = (
        td("""
        # title

        pending = []
        root = 0x2A # value

        [section]
        plain = []

        # ending
        """)
        .replace("\n", "\r\n", 2)
        .rstrip("\n")
    )
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


def test_copy_dotted_inline_view_leaves_discarded_tail_comments_untouched() -> None:
    src = td("""
        cfg = {
          group.items = [ 1 ],
          # unrelated tail
          other = [{ x = [2] }],
        }
        """)
    source = tomlrt.loads(src)
    target = tomlrt.Document()
    target["picked"] = copy(source.table("cfg.group"))
    target.table("picked").array("items").append(3)
    expected = td("""
        picked = {
          items = [ 1, 3 ],
        }
        """)
    assert tomlrt.dumps(target) == expected
    assert reparses(expected) == target.to_dict()
    assert tomlrt.dumps(source) == src


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
