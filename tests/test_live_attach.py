"""Live-attach semantics for ``Table.inline`` (and later ``Array``, ``AoT``).

A typed-container assigned to a document attaches *live* when it is not
already attached elsewhere: the user's own reference becomes the view at
the assignment site, and subsequent mutations through that reference are
visible in the document. An already-attached typed container is cloned
on assignment instead, so a single object never lives in two CST
locations. Plain ``dict`` / ``list`` continue to be snapshot-synthesised
and are unchanged.
"""

from __future__ import annotations

import pytest

import tomlrt
from _helpers import reparses as _reparses
from tomlrt import AoT, Array, Table

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
    assert "b = 2" in rendered
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
    assert _reparses(tomlrt.dumps(doc)) == {"bar": {"x": 1, "y": 2, "z": 3}}


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
    assert _reparses(tomlrt.dumps(d1))["a"] == {"k": 99}
    assert _reparses(tomlrt.dumps(d2))["a"] == {"k": 1}


def test_intra_document_assignment_clones() -> None:
    doc = tomlrt.loads("")
    doc["a"] = Table.inline({"k": 1})
    doc["b"] = doc["a"]
    assert doc["b"] is not doc["a"]
    doc["a"]["k"] = 99
    parsed = _reparses(tomlrt.dumps(doc))
    assert parsed == {"a": {"k": 99}, "b": {"k": 1}}


# ---------------------------------------------------------------------------
# Plain dict assignment is still snapshot
# ---------------------------------------------------------------------------


def test_plain_dict_assignment_is_snapshot() -> None:
    doc = tomlrt.loads("")
    plain = {"a": 1}
    doc["foo"] = plain
    plain["b"] = 2  # plain dict mutation must not reach doc
    assert "b" not in _reparses(tomlrt.dumps(doc))["foo"]


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
    assert "b" not in _reparses(tomlrt.dumps(doc))["foo"]


def test_reassign_after_detach_attaches_again() -> None:
    doc = tomlrt.loads("")
    t = Table.inline({"a": 1})
    doc["foo"] = t
    doc["foo"] = Table.inline({"placeholder": True})
    # ``t`` is detached; it should be re-installable as live.
    doc["bar"] = t
    assert doc["bar"] is t
    t["c"] = 3
    assert _reparses(tomlrt.dumps(doc))["bar"] == {"a": 1, "c": 3}


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
    assert _reparses(tomlrt.dumps(doc)) == {"xs": [99, 2, 3, 4]}


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
    assert _reparses(tomlrt.dumps(doc)) == {"xs": [1, 2, 3, 4]}


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
    parsed = _reparses(tomlrt.dumps(doc))
    assert parsed == {"p": [1, 2, 99], "q": [1, 2]}


def test_array_cross_document_assignment_clones() -> None:
    d1 = tomlrt.loads("")
    d2 = tomlrt.loads("")
    arr = Array([1, 2, 3])
    d1["xs"] = arr
    d2["xs"] = d1["xs"]
    assert d2["xs"] is not d1["xs"]
    d1["xs"].append(99)
    assert _reparses(tomlrt.dumps(d1)) == {"xs": [1, 2, 3, 99]}
    assert _reparses(tomlrt.dumps(d2)) == {"xs": [1, 2, 3]}


def test_array_multiline_layout_preserved_through_live_attach() -> None:
    doc = tomlrt.loads("")
    arr = Array([1, 2, 3], multiline=True)
    doc["xs"] = arr
    assert doc["xs"] is arr
    out = tomlrt.dumps(doc)
    # Multiline format: items each on their own line.
    assert "\n    1" in out
    assert "\n    2" in out


def test_array_multiline_live_attach_into_crlf_document() -> None:
    doc = tomlrt.loads('name = "x"\r\n')
    doc["xs"] = Array(["foo", "bar"], multiline=True)
    out = tomlrt.dumps(doc)
    assert "\n" not in out.replace("\r\n", "")


def test_array_detached_from_crlf_reattached_into_lf_document() -> None:
    doc1 = tomlrt.loads("xs = [\r\n    1,\r\n    2,\r\n]\r\n")
    arr = doc1["xs"]
    doc1["xs"] = []  # detach arr
    doc2 = tomlrt.loads("")
    doc2["ys"] = arr
    out = tomlrt.dumps(doc2)
    assert "\r" not in out


def test_plain_list_assignment_is_snapshot() -> None:
    doc = tomlrt.loads("")
    plain = [1, 2, 3]
    doc["xs"] = plain
    plain.append(99)
    assert _reparses(tomlrt.dumps(doc))["xs"] == [1, 2, 3]


def test_detached_array_still_writable() -> None:
    doc = tomlrt.loads("")
    arr = Array([1, 2])
    doc["xs"] = arr
    doc["xs"] = Array([10, 20])  # arr is now detached
    arr.append(3)
    assert list(arr) == [1, 2, 3]
    assert _reparses(tomlrt.dumps(doc))["xs"] == [10, 20]


def test_reassign_array_after_detach_attaches_again() -> None:
    doc = tomlrt.loads("")
    arr = Array([1, 2])
    doc["xs"] = arr
    doc["xs"] = Array([99])
    doc["ys"] = arr  # arr is detached, so re-attaches live here
    assert doc["ys"] is arr
    arr.append(3)
    assert _reparses(tomlrt.dumps(doc))["ys"] == [1, 2, 3]


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
    parsed = _reparses(tomlrt.dumps(doc))
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
    parsed = _reparses(tomlrt.dumps(doc))
    assert parsed == {"servers": [{"name": "a"}, {"name": "b"}]}


def test_aot_entry_mutation_after_assignment_visible_in_document() -> None:
    doc = tomlrt.loads("")
    aot = tomlrt.AoT([{"name": "a"}])
    doc["servers"] = aot
    aot[0]["extra"] = 42
    parsed = _reparses(tomlrt.dumps(doc))
    assert parsed == {"servers": [{"name": "a", "extra": 42}]}


def test_empty_aot_attaches_then_appends_via_user_reference() -> None:
    doc = tomlrt.loads("")
    aot = tomlrt.AoT()
    doc["servers"] = aot
    aot.append({"name": "a"})
    aot.append({"name": "b"})
    parsed = _reparses(tomlrt.dumps(doc))
    assert parsed == {"servers": [{"name": "a"}, {"name": "b"}]}


def test_aot_double_assign_clones_second_slot() -> None:
    doc = tomlrt.loads("")
    aot = tomlrt.AoT([{"name": "a"}])
    doc["p"] = aot
    doc["q"] = aot
    assert doc["p"] is aot
    assert doc["q"] is not aot
    aot.append({"name": "b"})
    parsed = _reparses(tomlrt.dumps(doc))
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
    assert _reparses(tomlrt.dumps(d1)) == {
        "servers": [{"name": "a"}, {"name": "b"}],
    }
    assert _reparses(tomlrt.dumps(d2)) == {"servers": [{"name": "a"}]}


def test_aot_intra_document_assignment_clones() -> None:
    doc = tomlrt.loads("")
    aot = tomlrt.AoT([{"name": "a"}])
    doc["p"] = aot
    doc["q"] = doc["p"]
    assert doc["p"] is aot
    assert doc["q"] is not doc["p"]
    aot.append({"name": "b"})
    parsed = _reparses(tomlrt.dumps(doc))
    assert parsed == {"p": [{"name": "a"}, {"name": "b"}], "q": [{"name": "a"}]}


def test_detached_aot_reattaches_live() -> None:
    doc = tomlrt.loads("")
    aot = tomlrt.AoT([{"name": "a"}])
    doc["servers"] = aot
    doc["servers"] = tomlrt.AoT([{"name": "z"}])  # aot now detached
    doc["others"] = aot  # re-attaches live here
    assert doc["others"] is aot
    aot.append({"name": "b"})
    parsed = _reparses(tomlrt.dumps(doc))
    assert parsed == {
        "servers": [{"name": "z"}],
        "others": [{"name": "a"}, {"name": "b"}],
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
    parsed = _reparses(tomlrt.dumps(doc))
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
    parsed = _reparses(tomlrt.dumps(doc))
    assert parsed == {"servers": [{"name": "a", "extra": 1}]}


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
    parsed = _reparses(tomlrt.dumps(doc))
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
    parsed = _reparses(tomlrt.dumps(doc))
    assert parsed == {"x": {"y": {"z": 1, "extra": 99}}}


def test_array_inside_plain_dict_attaches_live() -> None:
    doc = tomlrt.loads("")
    arr = Array([1, 2])
    doc["x"] = {"xs": arr}
    arr.append(3)
    parsed = _reparses(tomlrt.dumps(doc))
    assert parsed == {"x": {"xs": [1, 2, 3]}}


def test_inline_inside_plain_list_attaches_live() -> None:
    doc = tomlrt.loads("")
    inner = Table.inline({"z": 1})
    doc["xs"] = [inner, {"q": 2}]
    inner["extra"] = 99
    parsed = _reparses(tomlrt.dumps(doc))
    assert parsed == {"xs": [{"z": 1, "extra": 99}, {"q": 2}]}


def test_array_inside_array_attaches_live() -> None:
    doc = tomlrt.loads("")
    inner = Array([1, 2])
    doc["xs"] = Array([inner, [3, 4]])
    inner.append(99)
    parsed = _reparses(tomlrt.dumps(doc))
    assert parsed == {"xs": [[1, 2, 99], [3, 4]]}


def test_inline_table_inside_unattached_array_attaches_live() -> None:
    doc = tomlrt.loads("")
    inner = Table.inline({"a": 1})
    doc["xs"] = Array([inner])
    inner["b"] = 2
    parsed = _reparses(tomlrt.dumps(doc))
    assert parsed == {"xs": [{"a": 1, "b": 2}]}


def test_outer_plain_dict_remains_snapshot() -> None:
    # The plain-dict outer is still a snapshot: mutating it after
    # assignment does *not* show up in the document, even though a
    # nested typed container inside it attaches live.
    doc = tomlrt.loads("")
    plain: dict[str, object] = {"y": Table.inline({"z": 1})}
    doc["x"] = plain
    plain["new"] = 42  # outer is snapshot — not visible in doc
    parsed = _reparses(tomlrt.dumps(doc))
    assert parsed == {"x": {"y": {"z": 1}}}


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
    assert _reparses(tomlrt.dumps(doc)) == {"a": {"x": 1}}


def test_section_pre_assign_population_carries_through() -> None:
    doc = tomlrt.loads("")
    t = Table.section()
    t["x"] = 1
    t["y"] = 2
    doc["a"] = t
    t["z"] = 3
    assert _reparses(tomlrt.dumps(doc)) == {"a": {"x": 1, "y": 2, "z": 3}}


def test_section_double_assign_clones_second_slot() -> None:
    doc = tomlrt.loads("")
    t = Table.section({"x": 1})
    doc["a"] = t
    doc["b"] = t
    t["x"] = 99
    parsed = _reparses(tomlrt.dumps(doc))
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
    parsed = _reparses(tomlrt.dumps(doc))
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
    assert out.count("[[tool.pkgs]]") == 2
    assert _reparses(out) == {"tool": {"pkgs": [{"name": "a"}, {"name": "b"}]}}
    assert doc["tool"]["pkgs"] is pkgs


def test_section_into_aot_entry_is_scoped_to_that_entry() -> None:
    doc = tomlrt.loads("")
    doc["pkg"] = AoT([{"name": "a"}, {"name": "b"}])
    src = Table.section({"url": "u1"})
    doc["pkg"][0]["source"] = src
    src["hash"] = "h"
    parsed = _reparses(tomlrt.dumps(doc))
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
    parsed = _reparses(tomlrt.dumps(doc))
    assert parsed == {"a": {"child": {"deep": {"z": 1}}}}


def test_section_install_multi_segment_path() -> None:
    doc = tomlrt.loads("")
    t = Table.section({"x": 1})
    doc.install(("a", "b"), t)
    assert doc["a"]["b"] is t
    t["y"] = 2
    assert _reparses(tomlrt.dumps(doc)) == {"a": {"b": {"x": 1, "y": 2}}}


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
    assert "__tomlrt_detached__" not in out


def test_section_replacement_preserves_prior_leading() -> None:
    src = """\
# comment above

[a]
x = 1
"""
    doc = tomlrt.loads(src)
    doc["a"] = Table.section({"x": 2})
    out = tomlrt.dumps(doc)
    assert out.startswith("# comment above\n\n[a]\n")
    assert "x = 2" in out


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
