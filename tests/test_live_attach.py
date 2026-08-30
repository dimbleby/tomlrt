"""Live-attach semantics for ``Table.inline`` (and later ``Array``, ``AoT``).

A typed container assigned to a document attaches *live* when it has no
current materialised owner: the user's own reference becomes the view at
the assignment site, and subsequent mutations through that reference are
visible in the document. An already-owned typed container is cloned on
assignment, so a single object never lives in two CST locations. Plain
``dict`` / ``list`` continue to be snapshot-synthesised and are unchanged.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

import pytest

import tomlrt
from _helpers import reparses as _reparses
from _helpers import td
from tomlrt import AoT, Array, Table

if sys.version_info >= (3, 12):
    from typing import override
else:  # pragma: no cover -- backport for Python < 3.12
    from typing_extensions import override

if TYPE_CHECKING:
    from collections.abc import Iterator

# ---------------------------------------------------------------------------
# Table.inline factory
# ---------------------------------------------------------------------------


def test_inline_factory_returns_inline_table_view() -> None:
    t = Table.inline({"a": 1, "b": 2})
    assert isinstance(t, Table)
    assert dict(t) == {"a": 1, "b": 2}


def test_inline_factory_empty() -> None:
    t = Table.inline()
    assert dict(t) == {}


def test_inline_factory_can_be_populated_before_assignment() -> None:
    t = Table.inline()
    t["x"] = 1
    t["y"] = "hello"
    assert dict(t) == {"x": 1, "y": "hello"}


def test_detached_section_del_removes_key() -> None:
    t = Table.section({"a": 1, "b": 2})
    del t["a"]
    assert dict(t) == {"b": 2}


def test_detached_section_pop_and_clear() -> None:
    t = Table.section({"a": 1, "b": 2})
    assert t.pop("a") == 1
    assert dict(t) == {"b": 2}
    t.clear()
    assert dict(t) == {}


def test_detached_section_del_missing_key_raises_keyerror() -> None:
    t = Table.section({"a": 1})
    with pytest.raises(KeyError):
        del t["nope"]


def test_detached_inline_del_removes_key() -> None:
    t = Table.inline({"a": 1, "b": 2})
    del t["a"]
    assert dict(t) == {"b": 2}


def test_detached_inline_nested_del_removes_key() -> None:
    t = Table.inline({"a": {"b": 1, "c": 2}})
    del t["a"]["b"]
    assert dict(t["a"]) == {"c": 2}


def test_detached_section_del_then_attach_round_trips() -> None:
    t = Table.section({"a": 1, "b": 2, "c": 3})
    del t["b"]
    doc = tomlrt.loads("")
    doc["x"] = t
    assert tomlrt.dumps(doc) == td(
        """
        [x]
        a = 1
        c = 3
        """,
    )


def test_detached_inline_del_then_attach_round_trips() -> None:
    t = Table.inline({"p": 1, "q": 2})
    del t["p"]
    doc = tomlrt.loads("")
    doc["y"] = t
    assert tomlrt.dumps(doc) == "y = { q = 2 }\n"


def test_inline_factory_renders_with_spaced_braces() -> None:
    # Synthesised inline tables use the same spaced ({ k = v }) style
    # as plain dicts assigned through value_to_node. Empty stays {}.
    doc = tomlrt.loads("")
    doc["a"] = Table.inline({"key": "v"})
    doc["b"] = {"key": "v"}
    doc["c"] = Table.inline()
    t = Table.inline()
    t["x"] = 1
    doc["d"] = t
    doc["e"] = Table.inline({"k1": 1, "k2": 2})
    assert tomlrt.dumps(doc) == (
        'a = { key = "v" }\n'
        'b = { key = "v" }\n'
        "c = {}\n"
        "d = { x = 1 }\n"
        "e = { k1 = 1, k2 = 2 }\n"
    )


# ---------------------------------------------------------------------------
# Live attach on assignment
# ---------------------------------------------------------------------------


def test_mutation_after_assignment_is_visible_in_document() -> None:
    doc = tomlrt.loads("")
    t = Table.inline({"a": 1})
    doc["foo"] = t
    t["b"] = 2
    rendered = tomlrt.dumps(doc)
    assert rendered == "foo = { a = 1, b = 2 }\n"
    assert _reparses(rendered) == {"foo": {"a": 1, "b": 2}}


def test_assigned_inline_is_user_reference() -> None:
    doc = tomlrt.loads("")
    t = Table.inline({"a": 1})
    doc["foo"] = t
    assert doc["foo"] is t


def test_incremental_population_then_assign_then_more_mutations() -> None:
    doc = tomlrt.loads("")
    t = Table.inline()
    t["x"] = 1
    t["y"] = 2
    doc["bar"] = t
    t["z"] = 3
    out = tomlrt.dumps(doc)
    assert out == "bar = { x = 1, y = 2, z = 3 }\n"
    assert _reparses(out) == {"bar": {"x": 1, "y": 2, "z": 3}}


def test_mutation_through_doc_visible_on_user_reference() -> None:
    doc = tomlrt.loads("")
    t = Table.inline({"a": 1})
    doc["foo"] = t
    doc["foo"]["c"] = 3
    assert dict(t) == {"a": 1, "c": 3}


def test_del_through_doc_visible_on_user_reference() -> None:
    doc = tomlrt.loads("")
    t = Table.inline({"a": 1, "b": 2})
    doc["foo"] = t
    del doc["foo"]["a"]
    assert dict(t) == {"b": 2}


# ---------------------------------------------------------------------------
# Already-attached source clones on assignment
# ---------------------------------------------------------------------------


def test_double_assign_clones_second_slot() -> None:
    doc = tomlrt.loads("")
    t = Table.inline({"k": "v"})
    doc["p"] = t
    doc["q"] = t
    assert doc["p"] is t
    assert doc["q"] is not t
    # First slot is live, second is independent.
    t["k"] = "changed"
    rendered = tomlrt.dumps(doc)
    assert rendered == 'p = { k = "changed" }\nq = { k = "v" }\n'
    parsed = _reparses(rendered)
    assert parsed == {"p": {"k": "changed"}, "q": {"k": "v"}}


def test_cross_document_assignment_clones() -> None:
    d1 = tomlrt.loads("")
    d2 = tomlrt.loads("")
    t = Table.inline({"k": 1})
    d1["a"] = t
    d2["a"] = d1["a"]
    assert d2["a"] is not d1["a"]
    d1["a"]["k"] = 99
    out1 = tomlrt.dumps(d1)
    out2 = tomlrt.dumps(d2)
    assert out1 == "a = { k = 99 }\n"
    assert out2 == "a = { k = 1 }\n"
    assert _reparses(out1)["a"] == {"k": 99}
    assert _reparses(out2)["a"] == {"k": 1}


def test_intra_document_assignment_clones() -> None:
    doc = tomlrt.loads("")
    doc["a"] = Table.inline({"k": 1})
    doc["b"] = doc["a"]
    assert doc["b"] is not doc["a"]
    doc["a"]["k"] = 99
    out = tomlrt.dumps(doc)
    assert out == "a = { k = 99 }\nb = { k = 1 }\n"
    parsed = _reparses(out)
    assert parsed == {"a": {"k": 99}, "b": {"k": 1}}


# ---------------------------------------------------------------------------
# Plain dict assignment is still snapshot
# ---------------------------------------------------------------------------


def test_plain_dict_assignment_is_snapshot() -> None:
    doc = tomlrt.loads("")
    plain = {"a": 1}
    doc["foo"] = plain
    plain["b"] = 2  # plain dict mutation must not reach doc
    out = tomlrt.dumps(doc)
    assert out == "foo = { a = 1 }\n"
    assert "b" not in _reparses(out)["foo"]


def test_plain_dict_assignment_returns_a_view_not_user_reference() -> None:
    doc = tomlrt.loads("")
    plain = {"a": 1}
    doc["foo"] = plain
    assert doc["foo"] is not plain


# ---------------------------------------------------------------------------
# Round-trip preservation for unrelated documents
# ---------------------------------------------------------------------------


def test_parse_dump_byte_exact_unchanged() -> None:
    src = "# header\nfoo = { a = 1, b = 2 }\n[section]\nx = 1\n"
    doc = tomlrt.loads(src)
    assert tomlrt.dumps(doc) == src


# ---------------------------------------------------------------------------
# Detached-after-overwrite still works (mutations write to the orphan node)
# ---------------------------------------------------------------------------


def test_detached_inline_still_writable() -> None:
    doc = tomlrt.loads("")
    t = Table.inline({"a": 1})
    doc["foo"] = t
    # Overwrite ``foo`` -- ``t`` is now detached.
    doc["foo"] = Table.inline({"new": True})
    # Mutations on the orphan still work locally.
    t["b"] = 2
    assert dict(t) == {"a": 1, "b": 2}
    # But they don't leak into the document.
    out = tomlrt.dumps(doc)
    assert out == "foo = { new = true }\n"
    assert "b" not in _reparses(out)["foo"]


def test_reassign_after_detach_attaches_again() -> None:
    doc = tomlrt.loads("")
    t = Table.inline({"a": 1})
    doc["foo"] = t
    doc["foo"] = Table.inline({"placeholder": True})
    # ``t`` is detached; it should be re-installable as live.
    doc["bar"] = t
    assert doc["bar"] is t
    t["c"] = 3
    out = tomlrt.dumps(doc)
    assert out == "foo = { placeholder = true }\nbar = { a = 1, c = 3 }\n"
    assert _reparses(out)["bar"] == {"a": 1, "c": 3}


# ---------------------------------------------------------------------------
# Sanity: the type errors we don't want to silently swallow
# ---------------------------------------------------------------------------


def test_inline_factory_rejects_non_string_keys() -> None:
    with pytest.raises(TypeError):
        Table.inline({1: "no"})  # type: ignore[dict-item]  # ty: ignore[invalid-argument-type]


def test_section_factory_rejects_non_string_keys() -> None:
    # Regression: ``Table.section({1: "x"})`` used to accept the bad
    # key silently and crash later with an opaque ``regex`` error.
    with pytest.raises(TypeError, match="must be str"):
        Table.section({1: "no"})  # type: ignore[dict-item]  # ty: ignore[invalid-argument-type]


def test_aot_factory_rejects_non_string_keys() -> None:
    # Regression: ``AoT([{1: "x"}])`` likewise accepted and crashed
    # later with the same opaque error.
    with pytest.raises(TypeError, match="must be str"):
        AoT([{1: "no"}])  # type: ignore[dict-item]  # ty: ignore[invalid-argument-type]


def test_attached_aot_append_rejects_non_string_keys() -> None:
    # Regression: ``AoT.append`` / ``.add`` / ``.insert`` on an
    # attached AoT only typechecked that the value was a Mapping, not
    # that its keys were strings. A non-string key sailed past the
    # entry-typecheck and then crashed deep inside the layout pipeline
    # with an opaque ``expected string or bytes-like object``.
    doc = tomlrt.loads("")
    doc["pkg"] = AoT()
    with pytest.raises(TypeError, match="must be str"):
        doc["pkg"].append({1: "no"})
    with pytest.raises(TypeError, match="must be str"):
        doc["pkg"].add({1: "no"})
    with pytest.raises(TypeError, match="must be str"):
        doc["pkg"].insert(0, {1: "no"})


def test_inline_synth_from_plain_dict_rejects_non_string_keys() -> None:
    # Regression: assigning a plain ``dict`` containing a non-string
    # key into a section table routed through ``_populate_inline_table``
    # which raised with the historical ``inline-table key must be str``
    # wording. Unified to ``TOML keys must be str`` so every entry
    # point reports the same error.
    doc = tomlrt.loads("")
    doc["t"] = Table.section()
    with pytest.raises(TypeError, match="TOML keys must be str"):
        doc["t"]["sub"] = {1: "no"}


def test_factories_reject_non_mapping_input() -> None:
    # Regression: ``Table.section`` / ``Table.inline`` / ``Document(data)``
    # / ``AoT`` accepted any object and crashed in ``.items()`` with a
    # raw ``AttributeError``. Each factory now raises a clean
    # ``TypeError`` up front.
    with pytest.raises(TypeError, match="must be a Mapping"):
        Table.section([("a", 1)])  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match="must be a Mapping"):
        Table.inline([("a", 1)])  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match="must be a Mapping"):
        tomlrt.Document([("a", 1)])  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match="must be a Mapping"):
        AoT(["not a mapping"])  # type: ignore[list-item]  # ty: ignore[invalid-argument-type]


def test_factories_validate_the_keys_they_copy() -> None:
    class Inconsistent(dict[Any, Any]):
        @override
        def __iter__(self) -> Iterator[Any]:
            return iter(k for k in dict.keys(self) if isinstance(k, str))

    data = Inconsistent()
    dict.__setitem__(data, 1, "bad")

    with pytest.raises(TypeError, match="TOML keys must be str"):
        Table.section(data)
    with pytest.raises(TypeError, match="TOML keys must be str"):
        Table.inline(data)
    with pytest.raises(TypeError, match="TOML keys must be str"):
        AoT([data])


# ---------------------------------------------------------------------------
# Array live attach
# ---------------------------------------------------------------------------


def test_array_factory_returns_array_view() -> None:
    arr = Array([1, 2, 3])
    assert isinstance(arr, list)
    assert list(arr) == [1, 2, 3]


def test_array_mutation_after_assignment_visible_in_document() -> None:
    doc = tomlrt.loads("")
    arr = Array([1, 2, 3])
    doc["xs"] = arr
    arr.append(4)
    arr[0] = 99
    out = tomlrt.dumps(doc)
    assert out == "xs = [99, 2, 3, 4]\n"
    assert _reparses(out) == {"xs": [99, 2, 3, 4]}


def test_assigned_array_is_user_reference() -> None:
    doc = tomlrt.loads("")
    arr = Array([1, 2, 3])
    doc["xs"] = arr
    assert doc["xs"] is arr


def test_incremental_array_population_then_assign_then_more() -> None:
    doc = tomlrt.loads("")
    arr = Array()
    arr.append(1)
    arr.append(2)
    doc["xs"] = arr
    arr.extend([3, 4])
    out = tomlrt.dumps(doc)
    assert out == "xs = [1, 2, 3, 4]\n"
    assert _reparses(out) == {"xs": [1, 2, 3, 4]}


def test_mutation_through_doc_visible_on_user_reference_array() -> None:
    doc = tomlrt.loads("")
    arr = Array([1, 2])
    doc["xs"] = arr
    doc["xs"].append(3)
    assert list(arr) == [1, 2, 3]


def test_array_double_assign_clones_second_slot() -> None:
    doc = tomlrt.loads("")
    arr = Array([1, 2])
    doc["p"] = arr
    doc["q"] = arr
    assert doc["p"] is arr
    assert doc["q"] is not arr
    arr.append(99)
    out = tomlrt.dumps(doc)
    assert out == "p = [1, 2, 99]\nq = [1, 2]\n"
    parsed = _reparses(out)
    assert parsed == {"p": [1, 2, 99], "q": [1, 2]}


def test_array_cross_document_assignment_clones() -> None:
    d1 = tomlrt.loads("")
    d2 = tomlrt.loads("")
    arr = Array([1, 2, 3])
    d1["xs"] = arr
    d2["xs"] = d1["xs"]
    assert d2["xs"] is not d1["xs"]
    d1["xs"].append(99)
    out1 = tomlrt.dumps(d1)
    out2 = tomlrt.dumps(d2)
    assert out1 == "xs = [1, 2, 3, 99]\n"
    assert out2 == "xs = [1, 2, 3]\n"
    assert _reparses(out1) == {"xs": [1, 2, 3, 99]}
    assert _reparses(out2) == {"xs": [1, 2, 3]}


def test_table_inside_standalone_array_mutates_its_cst() -> None:
    arr = Array([{"x": 1, "remove": 2}])
    item = arr.table(0)
    item["x"] = 9
    del item["remove"]

    doc = tomlrt.Document()
    doc["items"] = arr
    assert doc["items"] is arr
    assert doc.array("items").table(0) is item
    out = tomlrt.dumps(doc)
    assert out == "items = [{ x = 9 }]\n"
    assert _reparses(out) == doc.to_dict()


@pytest.mark.parametrize(
    "operation", ["constructor", "append", "extend", "insert", "slice"]
)
def test_repeated_table_in_standalone_array_clones_new_bindings(
    operation: str,
) -> None:
    table = Table.inline({"x": 1})
    if operation == "constructor":
        arr = Array([table, table])
    else:
        arr = Array()
        if operation == "append":
            arr.append(table)
            arr.append(table)
        elif operation == "extend":
            arr.extend([table, table])
        elif operation == "insert":
            arr.append(table)
            arr.insert(1, table)
        else:
            arr[0:0] = [table, table]

    first = arr.table(0)
    second = arr.table(1)
    assert first is table
    assert second is not table
    first["first"] = 10
    second["second"] = 20

    doc = tomlrt.Document()
    doc["items"] = arr
    out = tomlrt.dumps(doc)
    assert out == ("items = [{ x = 1, first = 10 }, { x = 1, second = 20 }]\n")
    assert _reparses(out) == doc.to_dict()


def test_bound_standalone_array_item_clones_until_removed() -> None:
    source = Array([{"x": 1}])
    item = source.table(0)

    copied = tomlrt.Document()
    copied["item"] = item
    assert copied["item"] is not item
    item["x"] = 2
    copied.table("item")["y"] = 3

    source_doc = tomlrt.Document()
    source_doc["source"] = source
    copied_out = tomlrt.dumps(copied)
    source_out = tomlrt.dumps(source_doc)
    assert copied_out == "item = { x = 1, y = 3 }\n"
    assert source_out == "source = [{ x = 2 }]\n"
    assert _reparses(copied_out) == copied.to_dict()
    assert _reparses(source_out) == source_doc.to_dict()


def test_removed_standalone_array_item_attaches_live() -> None:
    source = Array([{"x": 1}])
    item = source.pop(0)

    doc = tomlrt.Document()
    doc["item"] = item
    assert doc["item"] is item
    item["x"] = 2
    out = tomlrt.dumps(doc)
    assert out == "item = { x = 2 }\n"
    assert _reparses(out) == doc.to_dict()


def test_removed_materialised_table_attaches_live_inside_array() -> None:
    source = Array([{"x": 1}])
    item = source.pop(0)
    target = Array([item])
    assert target.table(0) is item

    item["x"] = 2
    doc = tomlrt.Document()
    doc["target"] = target
    out = tomlrt.dumps(doc)
    assert out == "target = [{ x = 2 }]\n"
    assert _reparses(out) == doc.to_dict()


def test_array_multiline_layout_preserved_through_live_attach() -> None:
    doc = tomlrt.loads("")
    arr = Array([1, 2, 3], multiline=True)
    doc["xs"] = arr
    assert doc["xs"] is arr
    out = tomlrt.dumps(doc)
    assert out == td("""
        xs = [
            1,
            2,
            3,
        ]
        """)


def test_array_multiline_live_attach_into_crlf_document() -> None:
    doc = tomlrt.loads('name = "x"\r\n')
    doc["xs"] = Array(["foo", "bar"], multiline=True)
    out = tomlrt.dumps(doc)
    assert out == 'name = "x"\r\nxs = [\r\n    "foo",\r\n    "bar",\r\n]\r\n'


def test_array_detached_from_crlf_reattached_into_lf_document() -> None:
    doc1 = tomlrt.loads("xs = [\r\n    1,\r\n    2,\r\n]\r\n")
    arr = doc1["xs"]
    doc1["xs"] = []  # detach arr
    doc2 = tomlrt.loads("")
    doc2["ys"] = arr
    out = tomlrt.dumps(doc2)
    assert out == "ys = [\n    1,\n    2,\n]\n"


def test_plain_list_assignment_is_snapshot() -> None:
    doc = tomlrt.loads("")
    plain = [1, 2, 3]
    doc["xs"] = plain
    plain.append(99)
    out = tomlrt.dumps(doc)
    assert out == "xs = [1, 2, 3]\n"
    assert _reparses(out)["xs"] == [1, 2, 3]


def test_detached_array_still_writable() -> None:
    doc = tomlrt.loads("")
    arr = Array([1, 2])
    doc["xs"] = arr
    doc["xs"] = Array([10, 20])  # arr is now detached
    arr.append(3)
    assert list(arr) == [1, 2, 3]
    out = tomlrt.dumps(doc)
    assert out == "xs = [10, 20]\n"
    assert _reparses(out)["xs"] == [10, 20]


def test_reassign_array_after_detach_attaches_again() -> None:
    doc = tomlrt.loads("")
    arr = Array([1, 2])
    doc["xs"] = arr
    doc["xs"] = Array([99])
    doc["ys"] = arr  # arr is detached, so re-attaches live here
    assert doc["ys"] is arr
    arr.append(3)
    out = tomlrt.dumps(doc)
    assert out == "xs = [99]\nys = [1, 2, 3]\n"
    assert _reparses(out)["ys"] == [1, 2, 3]


def test_deleted_array_with_inline_table_item_reattaches_live() -> None:
    # Regression: deleting an Array containing an inline-table item and
    # reassigning the same (now-detached) Array back into a document used
    # to leave the nested inline-table item's `_value` null (an internal
    # invariant violation) because the array-reuse reattach path never
    # restored it — mutating the nested item then crashed. Covers both
    # reassignment at the same key and at a different one.
    doc = tomlrt.loads(
        td("""
        mixed = [{ q = 1 }]
    """)
    )
    arr = doc["mixed"]
    del doc["mixed"]
    doc["mixed"] = arr
    item = doc["mixed"].table(0)
    item["q"] = 99
    item["r"] = "added"
    out = tomlrt.dumps(doc)
    assert out == td("""
        mixed = [{ q = 99, r = "added" }]
    """)
    assert _reparses(out) == {"mixed": [{"q": 99, "r": "added"}]}


def test_deleted_array_with_nested_array_of_inline_tables_reattaches_live() -> None:
    doc = tomlrt.loads(
        td("""
        mixed = [[{ q = 1 }]]
    """)
    )
    arr = doc["mixed"]
    del doc["mixed"]
    doc["other"] = arr
    item = doc["other"].array(0).table(0)
    item["q"] = 99
    out = tomlrt.dumps(doc)
    assert out == td("""
        other = [[{ q = 99 }]]
    """)
    assert _reparses(out) == {"other": [[{"q": 99}]]}


# ---------------------------------------------------------------------------
# Mixed: Array inside an inline table, both live-attached
# ---------------------------------------------------------------------------


def test_array_inside_inline_table_both_live() -> None:
    doc = tomlrt.loads("")
    arr = Array([1, 2])
    inline = Table.inline({"xs": arr})
    doc["t"] = inline
    inline["k"] = "added"
    arr.append(3)
    out = tomlrt.dumps(doc)
    assert out == 't = { xs = [1, 2, 3], k = "added" }\n'
    parsed = _reparses(out)
    assert parsed == {"t": {"xs": [1, 2, 3], "k": "added"}}


# ---------------------------------------------------------------------------
# AoT live attach
# ---------------------------------------------------------------------------


def test_aot_factory_returns_unattached() -> None:
    aot = tomlrt.AoT([{"name": "a"}, {"name": "b"}])
    assert isinstance(aot, list)
    assert [dict(t) for t in aot] == [{"name": "a"}, {"name": "b"}]


def test_aot_assignment_is_user_reference() -> None:
    doc = tomlrt.loads("")
    aot = tomlrt.AoT([{"name": "a"}])
    doc["servers"] = aot
    assert doc["servers"] is aot


def test_aot_mutation_after_assignment_visible_in_document() -> None:
    doc = tomlrt.loads("")
    aot = tomlrt.AoT([{"name": "a"}])
    doc["servers"] = aot
    aot.append({"name": "b"})
    out = tomlrt.dumps(doc)
    assert out == td("""
        [[servers]]
        name = "a"

        [[servers]]
        name = "b"
        """)
    parsed = _reparses(out)
    assert parsed == {"servers": [{"name": "a"}, {"name": "b"}]}


def test_aot_entry_mutation_after_assignment_visible_in_document() -> None:
    doc = tomlrt.loads("")
    aot = tomlrt.AoT([{"name": "a"}])
    doc["servers"] = aot
    aot[0]["extra"] = 42
    out = tomlrt.dumps(doc)
    assert out == td("""
        [[servers]]
        name = "a"
        extra = 42
        """)
    parsed = _reparses(out)
    assert parsed == {"servers": [{"name": "a", "extra": 42}]}


def test_empty_aot_attaches_then_appends_via_user_reference() -> None:
    doc = tomlrt.loads("")
    aot = tomlrt.AoT()
    doc["servers"] = aot
    aot.append({"name": "a"})
    aot.append({"name": "b"})
    out = tomlrt.dumps(doc)
    assert out == td("""
        [[servers]]
        name = "a"

        [[servers]]
        name = "b"
        """)
    parsed = _reparses(out)
    assert parsed == {"servers": [{"name": "a"}, {"name": "b"}]}


def test_aot_double_assign_clones_second_slot() -> None:
    doc = tomlrt.loads("")
    aot = tomlrt.AoT([{"name": "a"}])
    doc["p"] = aot
    doc["q"] = aot
    assert doc["p"] is aot
    assert doc["q"] is not aot
    aot.append({"name": "b"})
    out = tomlrt.dumps(doc)
    assert out == td("""
        [[p]]
        name = "a"

        [[p]]
        name = "b"

        [[q]]
        name = "a"
        """)
    parsed = _reparses(out)
    assert parsed == {"p": [{"name": "a"}, {"name": "b"}], "q": [{"name": "a"}]}


def test_aot_cross_document_assignment_clones() -> None:
    d1 = tomlrt.loads("")
    aot = tomlrt.AoT([{"name": "a"}])
    d1["servers"] = aot
    d2 = tomlrt.loads("")
    d2["servers"] = d1["servers"]
    assert d1["servers"] is aot
    assert d2["servers"] is not d1["servers"]
    aot.append({"name": "b"})
    out1 = tomlrt.dumps(d1)
    out2 = tomlrt.dumps(d2)
    assert out1 == td("""
        [[servers]]
        name = "a"

        [[servers]]
        name = "b"
        """)
    assert out2 == td("""
        [[servers]]
        name = "a"
        """)
    assert _reparses(out1) == {
        "servers": [{"name": "a"}, {"name": "b"}],
    }
    assert _reparses(out2) == {"servers": [{"name": "a"}]}


def test_aot_intra_document_assignment_clones() -> None:
    doc = tomlrt.loads("")
    aot = tomlrt.AoT([{"name": "a"}])
    doc["p"] = aot
    doc["q"] = doc["p"]
    assert doc["p"] is aot
    assert doc["q"] is not doc["p"]
    aot.append({"name": "b"})
    out = tomlrt.dumps(doc)
    assert out == td("""
        [[p]]
        name = "a"

        [[p]]
        name = "b"

        [[q]]
        name = "a"
        """)
    parsed = _reparses(out)
    assert parsed == {"p": [{"name": "a"}, {"name": "b"}], "q": [{"name": "a"}]}


def test_detached_aot_reattaches_live() -> None:
    doc = tomlrt.loads("")
    aot = tomlrt.AoT([{"name": "a"}])
    doc["servers"] = aot
    doc["servers"] = tomlrt.AoT([{"name": "z"}])  # aot now detached
    doc["others"] = aot  # re-attaches live here
    assert doc["others"] is aot
    aot.append({"name": "b"})
    out = tomlrt.dumps(doc)
    assert out == td("""
        [[servers]]
        name = "z"

        [[others]]
        name = "a"

        [[others]]
        name = "b"
        """)
    parsed = _reparses(out)
    assert parsed == {
        "servers": [{"name": "z"}],
        "others": [{"name": "a"}, {"name": "b"}],
    }


def test_detached_aot_reattach_with_kv_before_nested_section() -> None:
    """``entry_slots`` is membership order, not doc-stream order.

    Adding a direct KV to an AoT entry that already has a nested
    sub-section appends the new slot to the *tail* of ``entry_slots``
    even though it is spliced physically *before* the nested header.
    Deleting and re-attaching the same (now private-orphan) ``AoT``
    must still walk the entry in true doc-stream order, not
    ``entry_slots`` order.
    """
    src = td("""
        [[arr]]
        [arr.subtab]
        val = 1

        [[arr]]
        [arr.subtab]
        val = 2
        """)
    doc = tomlrt.loads(src)
    doc["arr"][0]["newkey"] = 1
    aot = doc["arr"]
    del doc["arr"]
    doc["arr"] = aot
    out = tomlrt.dumps(doc)
    assert out == td("""
        [[arr]]
        newkey = 1
        [arr.subtab]
        val = 1

        [[arr]]
        [arr.subtab]
        val = 2
        """)
    parsed = _reparses(out)
    assert parsed == {
        "arr": [
            {"newkey": 1, "subtab": {"val": 1}},
            {"subtab": {"val": 2}},
        ],
    }


def test_detached_table_writes_survive_reattach() -> None:
    """Writes to a detached ``_StdTable`` view must persist when the view
    is later re-assigned into a document.
    """
    doc = tomlrt.loads("[t]\na = 1\n")
    t = doc.table("t")
    del doc["t"]  # t is now detached against an orphan doc_node
    t["b"] = 2
    del t["a"]
    doc["t"] = t  # re-attach via deep-clone of the orphan
    out = tomlrt.dumps(doc)
    assert out == td("""
        [t]
        b = 2
        """)
    parsed = _reparses(out)
    assert parsed == {"t": {"b": 2}}


def test_aot_entry_view_identity_preserved_through_attach() -> None:
    aot = tomlrt.AoT([{"name": "a"}])
    entry = aot[0]
    doc = tomlrt.loads("")
    doc["servers"] = aot
    # The same Table view the user grabbed before assignment is still
    # the live entry post-attach.
    assert aot[0] is entry
    entry["extra"] = 1
    out = tomlrt.dumps(doc)
    assert out == td("""
        [[servers]]
        name = "a"
        extra = 1
        """)
    parsed = _reparses(out)
    assert parsed == {"servers": [{"name": "a", "extra": 1}]}


def test_aot_entry_as_table_with_kv_before_nested_section() -> None:
    """``clone_aot_entry_as_table`` must clone entry slots in doc-stream
    order, not ``entry_slots`` membership order (same class of bug as
    :func:`clone_aot_entry`'s private-orphan branch).
    """
    src = td("""
        [[arr]]
        [arr.subtab]
        val = 1

        [[arr]]
        [arr.subtab]
        val = 2
        """)
    doc = tomlrt.loads(src)
    entry = doc["arr"][0]
    entry["newkey"] = 1
    doc["standalone"] = entry  # live entry reinstalled as a plain table
    out = tomlrt.dumps(doc)
    assert out == td("""
        [[arr]]
        newkey = 1
        [arr.subtab]
        val = 1

        [[arr]]
        [arr.subtab]
        val = 2
        [standalone]
        newkey = 1
        [standalone.subtab]
        val = 1
        """)
    parsed = _reparses(out)
    assert parsed["standalone"] == {"newkey": 1, "subtab": {"val": 1}}


def test_aot_held_nested_section_under_entry_survives_attach() -> None:
    """A nested live container assigned into an AoT entry *before* the AoT
    itself is installed must remain wired to the destination document.
    """
    nested = Table.section({"x": 1})
    aot = AoT([{"name": "first"}])
    aot[0]["cfg"] = nested
    doc = tomlrt.loads("")
    doc["pkgs"] = aot
    nested["y"] = 2
    out = tomlrt.dumps(doc)
    assert out == td("""
        [[pkgs]]
        name = "first"

        [pkgs.cfg]
        x = 1
        y = 2
        """)
    parsed = _reparses(out)
    assert parsed == {"pkgs": [{"name": "first", "cfg": {"x": 1, "y": 2}}]}
    assert doc["pkgs"][0]["cfg"] is nested


# ---------------------------------------------------------------------------
# Recursive attach: typed containers inside plain dict/list values
# ---------------------------------------------------------------------------


def test_inline_inside_plain_dict_attaches_live() -> None:
    doc = tomlrt.loads("")
    inner = Table.inline({"z": 1})
    doc["x"] = {"y": inner}
    inner["extra"] = 99
    out = tomlrt.dumps(doc)
    assert out == "x = { y = { z = 1, extra = 99 } }\n"
    assert _reparses(out) == {"x": {"y": {"z": 1, "extra": 99}}}


def test_array_inside_plain_dict_attaches_live() -> None:
    doc = tomlrt.loads("")
    arr = Array([1, 2])
    doc["x"] = {"xs": arr}
    arr.append(3)
    out = tomlrt.dumps(doc)
    assert out == "x = { xs = [1, 2, 3] }\n"
    assert _reparses(out) == {"x": {"xs": [1, 2, 3]}}


def test_inline_inside_plain_list_attaches_live() -> None:
    doc = tomlrt.loads("")
    inner = Table.inline({"z": 1})
    doc["xs"] = [inner, {"q": 2}]
    inner["extra"] = 99
    out = tomlrt.dumps(doc)
    assert out == "xs = [{ z = 1, extra = 99 }, { q = 2 }]\n"
    assert _reparses(out) == {"xs": [{"z": 1, "extra": 99}, {"q": 2}]}


def test_array_inside_array_attaches_live() -> None:
    doc = tomlrt.loads("")
    inner = Array([1, 2])
    doc["xs"] = Array([inner, [3, 4]])
    inner.append(99)
    out = tomlrt.dumps(doc)
    assert out == "xs = [[1, 2, 99], [3, 4]]\n"
    assert _reparses(out) == {"xs": [[1, 2, 99], [3, 4]]}


def test_inline_table_inside_unattached_array_attaches_live() -> None:
    doc = tomlrt.loads("")
    inner = Table.inline({"a": 1})
    doc["xs"] = Array([inner])
    inner["b"] = 2
    out = tomlrt.dumps(doc)
    assert out == "xs = [{ a = 1, b = 2 }]\n"
    assert _reparses(out) == {"xs": [{"a": 1, "b": 2}]}


def test_outer_plain_dict_remains_snapshot() -> None:
    # The plain-dict outer is still a snapshot: mutating it after
    # assignment does *not* show up in the document, even though a
    # nested typed container inside it attaches live.
    doc = tomlrt.loads("")
    plain: dict[str, object] = {"y": Table.inline({"z": 1})}
    doc["x"] = plain
    plain["new"] = 42  # outer is snapshot — not visible in doc
    out = tomlrt.dumps(doc)
    assert out == "x = { y = { z = 1 } }\n"
    assert _reparses(out) == {"x": {"y": {"z": 1}}}


# ---------------------------------------------------------------------------
# Table.section live-attach semantics.
#
# Symmetric with Table.inline / Array / AoT: an unattached section table
# (the return value of ``Table.section()``) attaches live on assignment.
# ``doc[k] is t`` afterwards, post-assignment mutations through ``t`` are
# visible in the document, and a second assignment deep-clones.
# ---------------------------------------------------------------------------


def test_section_factory_returns_a_table() -> None:
    t = Table.section({"x": 1})
    assert isinstance(t, Table)
    assert t["x"] == 1


def test_section_assigned_is_user_reference() -> None:
    doc = tomlrt.loads("")
    t = Table.section({"x": 1})
    doc["a"] = t
    assert doc["a"] is t


def test_section_post_assign_scalar_mutation_visible_in_dump() -> None:
    doc = tomlrt.loads("")
    t = Table.section()
    doc["a"] = t
    t["x"] = 1
    out = tomlrt.dumps(doc)
    assert out == td("""
        [a]
        x = 1
        """)
    assert _reparses(out) == {"a": {"x": 1}}


def test_section_pre_assign_population_carries_through() -> None:
    doc = tomlrt.loads("")
    t = Table.section()
    t["x"] = 1
    t["y"] = 2
    doc["a"] = t
    t["z"] = 3
    out = tomlrt.dumps(doc)
    assert out == td("""
        [a]
        x = 1
        y = 2
        z = 3
        """)
    assert _reparses(out) == {"a": {"x": 1, "y": 2, "z": 3}}


def test_section_double_assign_clones_second_slot() -> None:
    doc = tomlrt.loads("")
    t = Table.section({"x": 1})
    doc["a"] = t
    doc["b"] = t
    t["x"] = 99
    out = tomlrt.dumps(doc)
    assert out == td("""
        [a]
        x = 99

        [b]
        x = 1
        """)
    parsed = _reparses(out)
    assert parsed == {"a": {"x": 99}, "b": {"x": 1}}
    assert doc["a"] is t
    assert doc["b"] is not t


def test_section_held_nested_section_survives_parent_attach() -> None:
    doc = tomlrt.loads("")
    parent = Table.section({"name": "p"})
    child = Table.section({"k": "v"})
    parent["child"] = child
    doc["a"] = parent
    child["new"] = 42
    out = tomlrt.dumps(doc)
    assert out == td("""
        [a]
        name = "p"

        [a.child]
        k = "v"
        new = 42
        """)
    parsed = _reparses(out)
    assert parsed == {"a": {"name": "p", "child": {"k": "v", "new": 42}}}
    assert doc["a"]["child"] is child


def test_section_held_nested_aot_survives_parent_attach() -> None:
    doc = tomlrt.loads("")
    parent = Table.section()
    pkgs = AoT([{"name": "a"}])
    parent["pkgs"] = pkgs
    doc["tool"] = parent
    pkgs.add({"name": "b"})
    out = tomlrt.dumps(doc)
    assert out == td("""
        [[tool.pkgs]]
        name = "a"

        [[tool.pkgs]]
        name = "b"
        """)
    assert _reparses(out) == {"tool": {"pkgs": [{"name": "a"}, {"name": "b"}]}}
    assert doc["tool"]["pkgs"] is pkgs


def test_section_into_aot_entry_is_scoped_to_that_entry() -> None:
    doc = tomlrt.loads("")
    doc["pkg"] = AoT([{"name": "a"}, {"name": "b"}])
    src = Table.section({"url": "u1"})
    doc["pkg"][0]["source"] = src
    src["hash"] = "h"
    out = tomlrt.dumps(doc)
    assert out == td("""
        [[pkg]]
        name = "a"

        [pkg.source]
        url = "u1"
        hash = "h"

        [[pkg]]
        name = "b"
        """)
    parsed = _reparses(out)
    assert parsed == {
        "pkg": [
            {"name": "a", "source": {"url": "u1", "hash": "h"}},
            {"name": "b"},
        ],
    }


def test_section_structural_mutation_through_held_child() -> None:
    doc = tomlrt.loads("")
    parent = Table.section()
    child = Table.section()
    parent["child"] = child
    doc["a"] = parent
    # Structural insert (a fresh Table.section under child) after the
    # parent's attach proves that ``child`` was rehomed to the live
    # document, not just reading the same section nodes by accident.
    child["deep"] = Table.section({"z": 1})
    out = tomlrt.dumps(doc)
    assert out == td("""
        [a.child.deep]
        z = 1
        """)
    parsed = _reparses(out)
    assert parsed == {"a": {"child": {"deep": {"z": 1}}}}


def test_section_install_multi_segment_path() -> None:
    doc = tomlrt.loads("")
    t = Table.section({"x": 1})
    doc.install(("a", "b"), t)
    assert doc["a"]["b"] is t
    t["y"] = 2
    out = tomlrt.dumps(doc)
    assert out == td("""
        [a.b]
        x = 1
        y = 2
        """)
    assert _reparses(out) == {"a": {"b": {"x": 1, "y": 2}}}


def test_section_inside_inline_is_rejected() -> None:
    doc = tomlrt.loads("")
    doc["inline"] = Table.inline({"a": 1})
    with pytest.raises(tomlrt.TOMLError):
        doc["inline"]["nested"] = Table.section({"x": 1})


def test_section_placeholder_does_not_leak_into_dump() -> None:
    doc = tomlrt.loads("")
    parent = Table.section({"name": "p"})
    parent["sub"] = Table.section({"k": "v"})
    doc["root"] = parent
    out = tomlrt.dumps(doc)
    assert out == td("""
        [root]
        name = "p"

        [root.sub]
        k = "v"
        """)


def test_section_replacement_preserves_prior_leading() -> None:
    src = """\
# comment above

[a]
x = 1
"""
    doc = tomlrt.loads(src)
    doc["a"] = Table.section({"x": 2})
    out = tomlrt.dumps(doc)
    assert out == td("""
        # comment above

        [a]
        x = 2
        """)


def test_section_parse_dump_byte_exact_unchanged() -> None:
    src = """\
[a]
x = 1

[a.b]
y = 2

[[c]]
v = 1
"""
    doc = tomlrt.loads(src)
    assert tomlrt.dumps(doc) == src


def test_held_view_after_delete_does_not_corrupt_doc() -> None:
    # Held views survive delete via a private orphan root with live
    # mutation. Mutating the orphan must not affect the live document.
    doc = tomlrt.loads("[a]\nx = 1\n[b]\ny = 2\n")
    held = doc.table("a")
    del doc["a"]
    assert "a" not in doc
    assert tomlrt.dumps(doc) == "[b]\ny = 2\n"
    assert held["x"] == 1
    held["new"] = 99
    assert tomlrt.dumps(doc) == "[b]\ny = 2\n"
    assert held["new"] == 99


# ---------------------------------------------------------------------------
# Per-key clone of dotted-form sub-tables preserves dotted form.
# ---------------------------------------------------------------------------


def test_per_key_clone_of_dotted_preserves_dotted_form() -> None:
    src = tomlrt.loads(
        td("""
        [x]
        v.w = "hi"
        """)
    )
    dst = tomlrt.loads(
        td("""
        [x]
        """)
    )
    dst["x"]["v"] = src["x"]["v"]
    assert tomlrt.dumps(dst) == td("""
        [x]
        v.w = "hi"
        """)


def test_cross_document_clone_of_dotted_preserves_dotted_form() -> None:
    src = tomlrt.loads(
        td("""
        [x]
        v.w = 1
        """)
    )
    dst = tomlrt.loads(
        td("""
        [x]
        """)
    )
    dst["x"]["v"] = src["x"]["v"]
    # Source unchanged after clone.
    assert tomlrt.dumps(src) == td("""
        [x]
        v.w = 1
        """)
    assert tomlrt.dumps(dst) == td("""
        [x]
        v.w = 1
        """)


def test_per_key_clone_of_implicit_super_table_still_synthesises_header() -> None:
    src = tomlrt.loads(
        td("""
        [x.v.w]
        a = 1
        """)
    )
    dst = tomlrt.loads(
        td("""
        [x]
        """)
    )
    dst["x"]["v"] = src["x"]["v"]
    assert tomlrt.dumps(dst) == td("""
        [x]

        [x.v.w]
        a = 1
        """)


def test_per_key_clone_of_mixed_emits_dotted_kvs_and_subsection() -> None:
    src = tomlrt.loads(
        td("""
        [x]
        v.w = 1

        [x.v.q]
        a = 2
        """)
    )
    dst = tomlrt.loads(
        td("""
        [x]
        """)
    )
    dst["x"]["v"] = src["x"]["v"]
    assert tomlrt.dumps(dst) == td("""
        [x]
        v.w = 1

        [x.v.q]
        a = 2
        """)


def test_top_level_clone_of_dotted_preserves_dotted_form() -> None:
    src = tomlrt.loads("v.w = 1\n")
    dst = tomlrt.loads("")
    dst["v"] = src["v"]
    assert tomlrt.dumps(dst) == "v.w = 1\n"


def test_clone_of_nested_dotted_preserves_full_dotted_path() -> None:
    src = tomlrt.loads(
        td("""
        [x]
        v.w.z = 1
        """)
    )
    dst = tomlrt.loads(
        td("""
        [x]
        """)
    )
    dst["x"]["v"] = src["x"]["v"]
    assert tomlrt.dumps(dst) == td("""
        [x]
        v.w.z = 1
        """)


def test_clone_of_dotted_with_inline_value_preserves_dotted_form() -> None:
    src = tomlrt.loads(
        td("""
        [x]
        v.w = { a = 1 }
        """)
    )
    dst = tomlrt.loads(
        td("""
        [x]
        """)
    )
    dst["x"]["v"] = src["x"]["v"]
    assert tomlrt.dumps(dst) == td("""
        [x]
        v.w = { a = 1 }
        """)


def test_clone_dotted_to_top_level_before_existing_section() -> None:
    src = tomlrt.loads("v.w = 1\n")
    dst = tomlrt.loads(
        td("""
        [a]
        y = 2
        """)
    )
    dst["v"] = src["v"]
    assert tomlrt.dumps(dst) == td("""
        v.w = 1

        [a]
        y = 2
        """)


def test_append_aot_entry_from_overwritten_section_keeps_body() -> None:
    """Regression for #166: an AoT entry read from a just-overwritten
    section must keep its body when appended elsewhere.

    Overwriting a section that held an AoT (preceded by a scalar key)
    transplanted the detached old subtree to its orphan document in the
    wrong order — the nested ``[[demo.items]]`` header landed ahead of
    the section's ``name`` KV instead of after it. That left the entry
    header's ``_next`` pointing at ``name`` rather than the entry's own
    body, so cloning the entry gathered only the header and silently
    dropped the body.
    """
    doc = tomlrt.loads(
        td("""
        [demo]
        name = "widget"
        [[demo.items]]
        id = "KEEP-ME"
        """)
    )
    old = doc["demo"]
    doc["demo"] = "gadget"  # overwrite detaches the old section
    doc["dst"] = tomlrt.AoT()
    doc["dst"].append(old["items"][0])
    out = tomlrt.dumps(doc)
    assert out == td("""
        demo = "gadget"
        [[dst]]
        id = "KEEP-ME"
        """)
    assert _reparses(out) == {"demo": "gadget", "dst": [{"id": "KEEP-ME"}]}


def test_append_aot_entry_from_overwritten_noncontiguous_section() -> None:
    """Regression for #166 (non-contiguous variant): a binding's slots
    need not be contiguous in the doc-stream — ``[demo] … [other] …
    [[demo.items]]`` is legal TOML. Transplanting the overwritten
    section must still gather every owned slot across the foreign
    ``[other]`` gap, so the post-gap entry body survives an append.
    """
    doc = tomlrt.loads(
        td("""
        [demo]
        name = "widget"
        [other]
        y = 1
        [[demo.items]]
        id = "KEEP-ME"
        """)
    )
    old = doc["demo"]
    doc["demo"] = "gadget"
    doc["dst"] = tomlrt.AoT()
    doc["dst"].append(old["items"][0])
    out = tomlrt.dumps(doc)
    assert out == td("""
        demo = "gadget"
        [other]
        y = 1
        [[dst]]
        id = "KEEP-ME"
        """)
    assert _reparses(out) == {
        "demo": "gadget",
        "other": {"y": 1},
        "dst": [{"id": "KEEP-ME"}],
    }


def test_inline_overwrite_detaches_nested_dotted_view() -> None:
    """A dotted child remains live inside its displaced materialised root."""
    doc = tomlrt.loads("x = { a.c = 1, a.b = 2 }\n")
    inner = doc["x"]["a"]
    doc["x"] = 5
    assert tomlrt.dumps(doc) == "x = 5\n"

    inner.sort()
    inner["d"] = 3
    assert list(inner) == ["b", "c", "d"]

    doc["y"] = inner
    out = tomlrt.dumps(doc)
    assert out == td("""
        x = 5
        y = { b = 2, c = 1, d = 3 }
        """)
    assert _reparses(out) == doc.to_dict()


def test_inline_overwrite_keeps_nested_array_bound_to_displaced_root() -> None:
    """A nested array clones until its displaced materialised root moves."""
    doc = tomlrt.loads("x = { arr = [1, 2] }\n")
    outer = doc.table("x")
    arr = outer.array("arr")
    doc["x"] = 5
    assert not arr._attached  # noqa: SLF001

    arr.append(3)
    doc["y"] = arr
    assert doc["y"] is not arr
    arr.append(4)
    doc["z"] = outer
    assert doc["z"] is outer
    assert doc.table("z").array("arr") is arr
    out = tomlrt.dumps(doc)
    assert out == td("""
        x = 5
        y = [1, 2, 3]
        z = { arr = [1, 2, 3, 4] }
        """)
    assert _reparses(out) == doc.to_dict()


def test_displaced_nested_inline_tables_reattach_with_identity() -> None:
    doc = tomlrt.loads("x = { inner = { value = 1 } }\n")
    outer = doc.table("x")
    inner = outer.table("inner")
    del doc["x"]

    doc["y"] = outer
    assert doc["y"] is outer
    assert doc.table("y").table("inner") is inner
    inner["value"] = 2
    out = tomlrt.dumps(doc)
    assert out == "y = { inner = { value = 2 } }\n"
    assert _reparses(out) == doc.to_dict()


def test_inline_overwrite_with_inline_detaches_nested_view() -> None:
    """Detaching happens when the replacement is itself an inline table."""
    doc = tomlrt.loads("x = { a.c = 1, a.b = 2 }\n")
    inner = doc["x"]["a"]
    doc["x"] = {"p": 1}
    inner.sort()

    doc["y"] = inner
    out = tomlrt.dumps(doc)
    assert out == td("""
        x = { p = 1 }
        y = { b = 2, c = 1 }
        """)
    assert _reparses(out) == doc.to_dict()


def test_array_setitem_detaches_the_displaced_item_view() -> None:
    """Overwriting an array item detaches the view it displaced."""
    doc = tomlrt.loads("x = [ { c = 1 }, 2 ]\n")
    item = doc["x"][0]
    doc["x"][0] = 9
    assert tomlrt.dumps(doc) == "x = [ 9, 2 ]\n"

    item["d"] = 4
    doc["y"] = item
    out = tomlrt.dumps(doc)
    assert out == td("""
        x = [ 9, 2 ]
        y = { c = 1, d = 4 }
        """)
    assert doc["y"] is item
    assert _reparses(out) == doc.to_dict()


def test_array_clear_detaches_every_item_view() -> None:
    """Clearing an array detaches all of its item views at once."""
    doc = tomlrt.loads("x = [ { c = 1 }, { d = 2 } ]\n")
    first, second = doc["x"][0], doc["x"][1]
    doc["x"].clear()
    assert tomlrt.dumps(doc) == "x = [ ]\n"

    doc["y"] = first
    doc["z"] = second
    out = tomlrt.dumps(doc)
    assert out == td("""
        x = [ ]
        y = { c = 1 }
        z = { d = 2 }
        """)
    assert doc["y"] is first
    assert doc["z"] is second
    assert _reparses(out) == doc.to_dict()


def test_inline_overwrite_detaches_dotted_navigator() -> None:
    """A displaced dotted navigator must not write back into the table.

    The navigator kept resolving against the inline table it no longer
    belonged to, so a later write re-added its key alongside the value
    that replaced it — emitting a table with the key defined twice.
    """
    doc = tomlrt.loads("x = { n.a = 1, n.b = 2, z = 9 }\n")
    inner = doc["x"]["n"]
    doc["x"]["n"] = 5
    inner["c"] = 3
    out = tomlrt.dumps(doc)
    assert out == "x = { z = 9, n = 5 }\n"
    assert _reparses(out) == doc.to_dict()
    assert inner.to_dict() == {"a": 1, "b": 2, "c": 3}


def test_inline_delete_detaches_dotted_navigator() -> None:
    """Deleting a dotted prefix detaches its navigator too."""
    doc = tomlrt.loads("x = { n.a = 1, n.b = 2, z = 9 }\n")
    inner = doc["x"]["n"]
    del doc["x"]["n"]
    inner["c"] = 3
    out = tomlrt.dumps(doc)
    assert out == "x = { z = 9 }\n"
    assert _reparses(out) == doc.to_dict()


def test_inline_overwrite_detaches_displaced_array() -> None:
    """Replacing an inline entry detaches a displaced array too.

    The reset used to be reachable only when the displaced value was a
    table, so an array took the other replacement branch and stayed
    attached to an entry it no longer owned.
    """
    doc = tomlrt.loads("x = { n = [ { a = 1 } ], z = 9 }\n")
    held = doc["x"]["n"]
    nested = held[0]
    doc["x"]["n"] = 5
    assert tomlrt.dumps(doc) == "x = { n = 5, z = 9 }\n"

    held.append(2)
    doc["y"] = held
    out = tomlrt.dumps(doc)
    assert out == td("""
        x = { n = 5, z = 9 }
        y = [ { a = 1 }, 2 ]
        """)
    assert doc["y"] is held
    assert doc["y"][0] is nested
    assert _reparses(out) == doc.to_dict()


def test_array_item_assigned_to_itself_stays_attached() -> None:
    """``arr[i] = arr[i]`` re-uses the item view rather than displacing it.

    The identity short-circuit leaves the existing logical and CST
    binding untouched.
    """
    doc = tomlrt.loads(
        td("""
        [root]
        arr = [ { c = 1 }, 2 ]
        """)
    )
    orphan = doc.pop("root")
    arr = orphan["arr"]
    item = arr[0]
    arr[0] = item

    assert arr[0] is item
    item["d"] = 3
    doc["dest"] = orphan
    out = tomlrt.dumps(doc)
    assert out == td("""
        [dest]
        arr = [ { c = 1, d = 3 }, 2 ]
        """)
    assert _reparses(out) == doc.to_dict()


def test_array_same_position_slice_assignments_are_noops() -> None:
    src = "x = [ { a = 1 }, 2, [3] ]\n"
    doc = tomlrt.loads(src)
    arr = doc.array("x")
    original = list(arr)

    arr[0:1] = [arr[0]]
    arr[:] = arr
    arr[::2] = [arr[0], arr[2]]

    assert all(before is after for before, after in zip(original, arr, strict=True))
    assert tomlrt.dumps(doc) == src
    assert _reparses(src) == doc.to_dict()


def test_displaced_dotted_factory_frees_child_cst_roots() -> None:
    doc = tomlrt.loads("x = { n.a = {z=1}, n.b = [ 2, 3 ] }\n")
    inner = doc.table("x").table("n")
    table = inner.table("a")
    array = inner.array("b")

    del doc.table("x")["n"]
    inner["c"] = 3
    table["z"] = 9
    array.append(4)
    doc["y"] = inner

    assert doc["y"] is inner
    assert doc.table("y").table("a") is table
    assert doc.table("y").array("b") is array
    out = tomlrt.dumps(doc)
    assert out == "x = { }\ny = { a = {z=9}, b = [ 2, 3, 4 ], c = 3 }\n"
    assert _reparses(out) == doc.to_dict()


# ---------------------------------------------------------------------------
# A rejected ``Document(...)`` leaves the caller's views alone
# ---------------------------------------------------------------------------


def test_failed_document_init_leaves_section_detached() -> None:
    """A view is still the caller's to attach after the call raised.

    Everything is checked before anything is installed, so a later bad
    value cannot leave an earlier one wired into the document that the
    raise stopped us returning. The proof is black-box: an attach that
    really is the first one is *live*, so the later ``y`` shows up.
    """
    section = Table.section({"x": 1})
    with pytest.raises(TypeError):
        tomlrt.Document({"good": section, "bad": object()})

    # Still the caller's: assigning it is still a live attach.
    doc = tomlrt.Document()
    doc["s"] = section
    section["y"] = 2
    out = tomlrt.dumps(doc)
    assert out == td("""
        [s]
        x = 1
        y = 2
        """)
    assert _reparses(out) == doc.to_dict()


def test_failed_document_init_leaves_inline_and_array_detached() -> None:
    inline = Table.inline({"a": 1})
    array = Array([1, 2])
    with pytest.raises(TypeError):
        tomlrt.Document({"i": inline, "arr": array, "bad": object()})

    doc = tomlrt.Document()
    doc["i"] = inline
    doc["arr"] = array
    inline["b"] = 2
    array.append(3)
    out = tomlrt.dumps(doc)
    assert out == td("""
        i = { a = 1, b = 2 }
        arr = [1, 2, 3]
        """)
    assert _reparses(out) == doc.to_dict()


def test_failed_document_init_leaves_aot_detached() -> None:
    aot = AoT([{"k": 1}])
    with pytest.raises(TypeError):
        tomlrt.Document({"entries": aot, "bad": object()})

    doc = tomlrt.Document()
    doc["entries"] = aot
    aot.append({"k": 2})
    out = tomlrt.dumps(doc)
    assert out == td("""
        [[entries]]
        k = 1

        [[entries]]
        k = 2
        """)
    assert _reparses(out) == doc.to_dict()


def test_failed_document_init_leaves_popped_subtree_intact() -> None:
    """A popped subtree is the caller's, and a failed build must not take it."""
    source = tomlrt.loads(
        td("""
        [keep]
        k = 1

        [mine]
        x = 1
        """)
    )
    mine = source.table("mine")
    del source["mine"]

    with pytest.raises(TypeError):
        tomlrt.Document({"good": mine, "bad": object()})

    out = tomlrt.dumps(tomlrt.Document({"t": mine}))
    assert out == td("""
        [t]
        x = 1
        """)
    assert tomlrt.dumps(source) == td("""
        [keep]
        k = 1
        """)


def test_failed_document_init_checks_the_keys_it_installs() -> None:
    """A mapping may disagree with itself about which keys it holds.

    The keys checked on the way in are the ones ``__iter__`` yields; the
    ones installed come from ``items()``. Only the latter matter.
    """

    class Inconsistent(dict[Any, Any]):
        @override
        def __iter__(self) -> Iterator[Any]:
            return iter([k for k in dict.keys(self) if isinstance(k, str)])

    section = Table.section({"x": 1})
    data = Inconsistent()
    dict.__setitem__(data, "good", section)
    dict.__setitem__(data, 2, "two")

    with pytest.raises(TypeError, match="TOML keys must be str"):
        tomlrt.Document(data)

    doc = tomlrt.Document()
    doc["s"] = section
    section["y"] = 2
    assert tomlrt.dumps(doc) == td("""
        [s]
        x = 1
        y = 2
        """)


def _popped_with_inline_children() -> tuple[Table, Array, Table]:
    """A popped subtree, plus the inline views it still owns.

    Popping re-roots the whole subtree onto a private document of its
    own, inline descendants included -- so those are privately rooted
    too, and reachable without being the popped table itself.
    """
    src = tomlrt.loads(
        td("""
        [root]
        arr = [1, 2]
        it = { x = 1 }

        [root.sub]
        y = 2
        """)
    )
    orphan = src.pop("root")
    assert isinstance(orphan, Table)
    return orphan, orphan.array("arr"), orphan.table("it")


def test_update_through_a_plain_list_copies_a_popped_subtrees_array() -> None:
    """A popped subtree stays the caller's, however deeply it is wrapped.

    ``update`` reads its items first and installs them one at a time,
    and an install moves out of a private orphan rather than copying
    from it. A wrapper is rebuilt on the way in but the views inside it
    are installed as they are, so a source reachable only through one
    still has to be spared.
    """
    orphan, arr, _it = _popped_with_inline_children()
    dst = tomlrt.Document()
    dst.update({"wrapped": [arr]})
    arr.append(3)

    assert tomlrt.dumps(dst) == "wrapped = [[1, 2]]\n"
    assert tomlrt.dumps(tomlrt.Document(orphan)) == td("""
        arr = [1, 2, 3]
        it = { x = 1 }

        [sub]
        y = 2
        """)


def test_update_through_a_plain_mapping_copies_a_popped_subtrees_table() -> None:
    """The mapping half of the same rule.

    Left unspared, the install rehomes the inline table into ``dst``
    and the orphan is left saying one thing in its data and another in
    the source it renders.
    """
    orphan, _arr, it = _popped_with_inline_children()
    dst = tomlrt.Document()
    dst.update({"wrapped": {"it": it}})
    it["x"] = 9

    assert tomlrt.dumps(dst) == "wrapped = { it = { x = 1 } }\n"
    assert tomlrt.dumps(tomlrt.Document(orphan)) == td("""
        arr = [1, 2]
        it = { x = 9 }

        [sub]
        y = 2
        """)
    assert orphan.to_dict() == {"arr": [1, 2], "it": {"x": 9}, "sub": {"y": 2}}
