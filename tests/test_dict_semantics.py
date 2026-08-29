"""Tables really are :class:`dict` subclasses now.

These tests pin the user-visible Pythonic behaviours that the
view-based predecessor could not provide: ``isinstance(t, dict)``,
``**t`` unpacking, ``dict``-typed APIs accepting tables directly,
held references behaving like ordinary Python references, and
identity stability for repeated lookups.
"""

from __future__ import annotations

import json
from collections import UserString

import pytest

import tomlrt
from _helpers import td
from tomlrt import AoT, Table


def _src() -> str:
    return td("""
        title = "demo"

        [tool]
        name = "x"

        [tool.poetry]
        version = "0.1"

        [[entries]]
        k = 1

        [[entries]]
        k = 2
        """)


def test_document_is_dict() -> None:
    doc = tomlrt.loads(_src())
    assert isinstance(doc, dict)
    assert isinstance(doc["tool"], dict)
    assert isinstance(doc["tool"]["poetry"], dict)


def test_inline_table_is_dict() -> None:
    doc = tomlrt.loads("t = { a = 1, b = 2 }\n")
    assert isinstance(doc["t"], dict)


def test_dict_unpack_spread() -> None:
    doc = tomlrt.loads(_src())
    spread = {**doc["tool"]}
    assert spread == {"name": "x", "poetry": doc["tool"]["poetry"]}


def test_json_dumps_via_to_dict() -> None:
    doc = tomlrt.loads(
        td("""
        a = 1
        b = "two"
        [c]
        x = 3
        """)
    )
    # Direct json.dumps still needs to_dict for datetime-free trees,
    # but now the result of to_dict is interchangeable with a plain dict.
    assert json.loads(json.dumps(doc.to_dict())) == {"a": 1, "b": "two", "c": {"x": 3}}


def test_repeated_lookup_returns_same_object() -> None:
    doc = tomlrt.loads(_src())
    assert doc["tool"] is doc["tool"]
    assert doc["tool"]["poetry"] is doc["tool"]["poetry"]
    assert doc["entries"] is doc["entries"]


def test_held_reference_orphans_on_delete() -> None:
    doc = tomlrt.loads(_src())
    held = doc["tool"]
    del doc["tool"]
    # Held reference still has its own data...
    assert held["name"] == "x"
    # ...but mutations through it are silently invisible to the document.
    held["new"] = 99
    assert tomlrt.dumps(doc) == td("""
        title = "demo"

        [[entries]]
        k = 1

        [[entries]]
        k = 2
        """)


def test_held_reference_does_not_resurrect() -> None:
    doc = tomlrt.loads('[tool]\nname = "x"\n')
    held = doc["tool"]
    del doc["tool"]
    doc["tool"] = {"name": "y"}
    # The held reference is *not* the same object as the new binding.
    assert held is not doc["tool"]
    assert held["name"] == "x"
    assert doc["tool"]["name"] == "y"


def test_install_section_returns_object_stored_in_dict() -> None:
    doc = tomlrt.loads("")
    t = doc.install("a.b", Table.section())
    assert t is doc["a"]["b"]


def test_assign_aot_returns_object_stored_in_dict() -> None:
    doc = tomlrt.loads("")
    doc["things"] = AoT()
    aot = doc["things"]
    assert aot is doc["things"]


def test_empty_aot_appendable_after_set() -> None:
    doc = tomlrt.loads("")
    doc["things"] = AoT()
    aot = doc["things"]
    aot.add({"k": 1})
    assert tomlrt.dumps(doc) == td("""
        [[things]]
        k = 1
        """)
    # And still the same object.
    assert aot is doc["things"]


def test_dict_typed_isinstance_check() -> None:
    """Common downstream guard: ``isinstance(x, dict)`` now passes."""
    doc = tomlrt.loads("[a]\nx = 1\n")

    def consumer(d: dict[str, object]) -> int:
        assert isinstance(d, dict)
        return len(d)

    assert consumer(doc["a"]) == 1


def test_update_via_kwargs_and_mapping() -> None:
    doc = tomlrt.loads("[a]\nx = 1\n")
    doc["a"].update({"y": 2}, z=3)
    assert doc["a"]["x"] == 1
    assert doc["a"]["y"] == 2
    assert doc["a"]["z"] == 3
    rendered = tomlrt.dumps(doc)
    assert rendered == td("""
        [a]
        x = 1
        y = 2
        z = 3
        """)


def test_update_via_kwargs_only() -> None:
    """No positional argument: only the kwargs branch should run."""
    doc = tomlrt.loads("[a]\nx = 1\n")
    doc["a"].update(y=2, z=3)
    assert doc["a"]["y"] == 2
    assert doc["a"]["z"] == 3


def test_update_via_iterable_of_pairs() -> None:
    doc = tomlrt.loads("")
    doc.update([("a", 1), ("b", 2)])
    assert doc["a"] == 1
    assert doc["b"] == 2


def test_setdefault_existing_returns_stored() -> None:
    doc = tomlrt.loads("[a]\nx = 1\n")
    a = doc["a"]
    result = a.setdefault("x", 99)
    assert result == 1
    assert a["x"] == 1


def test_setdefault_missing_inserts_and_returns() -> None:
    doc = tomlrt.loads("[a]\nx = 1\n")
    a = doc["a"]
    result = a.setdefault("y", 42)
    assert result == 42
    assert a["y"] == 42
    assert tomlrt.dumps(doc) == td("""
        [a]
        x = 1
        y = 42
        """)


def test_clear_removes_keys_from_cst() -> None:
    doc = tomlrt.loads("a = 1\nb = 2\n")
    doc.clear()
    assert dict(doc) == {}
    assert tomlrt.dumps(doc) == ""


def test_pop_returns_orphaned_table() -> None:
    doc = tomlrt.loads("[a]\nx = 1\n")
    held_first = doc["a"]
    popped = doc.pop("a")
    # Same object that was in dict storage; now orphaned.
    assert popped is held_first
    assert "a" not in doc
    assert popped["x"] == 1


def test_pop_default_when_missing() -> None:
    doc = tomlrt.loads("")
    sentinel = object()
    assert doc.pop("missing", sentinel) is sentinel


def test_pop_non_str_key_is_treated_as_missing() -> None:
    doc = tomlrt.loads("a = 1\n")
    sentinel = object()
    assert doc.pop(123, sentinel) is sentinel
    with pytest.raises(KeyError):
        doc.pop(123)
    # Unhashable keys raise TypeError, matching ``dict.pop``.
    with pytest.raises(TypeError):
        doc.pop([], sentinel)
    with pytest.raises(TypeError):
        doc.pop([])
    assert tomlrt.dumps(doc) == "a = 1\n"


def test_pop_string_equivalent_non_str_key_is_treated_as_missing() -> None:
    key = UserString("a")
    doc = tomlrt.loads("a = 1\n")
    sentinel = object()

    assert doc.pop(key, sentinel) is sentinel
    with pytest.raises(KeyError):
        doc.pop(key)
    assert tomlrt.dumps(doc) == "a = 1\n"


def test_popitem_lifo() -> None:
    doc = tomlrt.loads(
        td("""
        a = 1
        b = 2
        c = 3
        """)
    )
    k, v = doc.popitem()
    assert (k, v) == ("c", 3)


def test_or_operator_in_place() -> None:
    doc = tomlrt.loads("a = 1\n")
    doc |= {"b": 2}
    assert doc["a"] == 1
    assert doc["b"] == 2


def test_copy_returns_plain_dict() -> None:
    doc = tomlrt.loads("[a]\nx = 1\n")
    snap = doc.copy()
    assert isinstance(snap, dict)
    assert not isinstance(snap, tomlrt.Table)
    # Mutations to snap don't reach the document.
    snap["new"] = 9
    assert "new" not in doc


def test_round_trip_unchanged_after_parse() -> None:
    """Populating dict storage at parse-time must not perturb the CST."""
    src = _src()
    assert tomlrt.dumps(tomlrt.loads(src)) == src


def test_mutation_through_held_child_visible_via_parent() -> None:
    doc = tomlrt.loads("[a]\nx = 1\n")
    a = doc["a"]
    a["y"] = 2
    assert doc["a"]["y"] == 2
    assert a is doc["a"]


def test_replace_value_invalidates_old_wrapper() -> None:
    doc = tomlrt.loads("[a]\nx = 1\n")
    old_a = doc["a"]
    doc["a"] = {"y": 9}
    assert old_a is not doc["a"]
    assert doc["a"]["y"] == 9
    assert "x" not in doc["a"]


def test_inline_promotion_returns_dict_storage_object() -> None:
    doc = tomlrt.loads('pkg = { name = "x" }\n')
    previous = doc.table("pkg")
    promoted = doc.promote_inline("pkg")
    assert promoted is doc["pkg"]
    assert promoted is not previous
    assert isinstance(doc["pkg"], dict)


def test_aot_promotion_returns_dict_storage_object() -> None:
    doc = tomlrt.loads("xs = [{ k = 1 }, { k = 2 }]\n")
    previous = doc.array("xs")
    previous_entries = list(previous)
    promoted = doc.promote_array("xs")
    assert promoted is doc["xs"]
    assert all(
        new is not old for new, old in zip(promoted, previous_entries, strict=True)
    )


def test_del_then_assign_section_does_not_revive_held_ref() -> None:
    doc = tomlrt.loads("[a]\nx = 1\n")
    held = doc["a"]
    del doc["a"]
    doc["a"] = Table.section()
    new = doc["a"]
    assert new is doc["a"]
    assert held is not new


def test_table_equals_plain_dict_with_same_keys() -> None:
    doc = tomlrt.loads(
        td("""
        [a]
        x = 1
        y = 2
        """)
    )
    assert doc["a"] == {"x": 1, "y": 2}


def test_keys_values_items_match_dict_protocol() -> None:
    doc = tomlrt.loads("a = 1\nb = 2\n")
    assert list(doc.keys()) == ["a", "b"]
    assert list(doc.values()) == [1, 2]
    assert list(doc.items()) == [("a", 1), ("b", 2)]


def test_table_unhashable_like_dict() -> None:
    doc = tomlrt.loads("a = 1\n")
    with pytest.raises(TypeError):
        hash(doc)


def test_detached_assign_section_does_not_revive_in_doc() -> None:
    doc = tomlrt.loads("[outer]\nx = 1\n")
    held = doc["outer"]
    del doc["outer"]
    assert isinstance(held, tomlrt.Table)
    held["nested"] = Table.section({"deep": "data"})
    assert tomlrt.dumps(doc) == ""
    # Held subtree still reflects its own structural changes.
    assert dict(held["nested"]) == {"deep": "data"}


def test_detached_aot_add_does_not_revive_in_doc() -> None:
    doc = tomlrt.loads(
        td("""
        [[entries]]
        k = 1
        [[entries]]
        k = 2
        """)
    )
    held = doc.aot("entries")
    del doc["entries"]
    held.add({"k": 999})
    assert tomlrt.dumps(doc) == ""
    assert [dict(e) for e in held] == [{"k": 1}, {"k": 2}, {"k": 999}]


def test_detached_aot_entry_assign_section_does_not_revive() -> None:
    doc = tomlrt.loads("[[entries]]\nk = 1\n")
    held_aot = doc.aot("entries")
    held_entry = held_aot[0]
    del doc["entries"]
    held_entry["sub"] = Table.section({"a": 1})
    held_entry["new"] = 42
    assert tomlrt.dumps(doc) == ""
    assert held_entry["new"] == 42
    assert dict(held_entry["sub"]) == {"a": 1}


def test_detached_aot_replaced_does_not_revive() -> None:
    doc = tomlrt.loads("[[entries]]\nk = 1\n")
    held = doc.aot("entries")
    doc["entries"] = "REPLACED"
    held.add({"k": 999})
    assert tomlrt.dumps(doc) == 'entries = "REPLACED"\n'
