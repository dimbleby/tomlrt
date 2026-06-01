"""Mutation API tests."""

from __future__ import annotations

from collections import deque
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from _helpers import reparses as _reparses
from _helpers import td

if TYPE_CHECKING:
    from collections.abc import MutableMapping

import pytest

import tomlrt
from tomlrt import AoT, Array, Table

# ---------------------------------------------------------------------------
# Scalar set/get/del
# ---------------------------------------------------------------------------


def test_replace_scalar_preserves_surrounding_format() -> None:
    src = td("""
        # header comment
        name = 'old'  # inline
        port = 80
        """)
    doc = tomlrt.loads(src)
    doc["name"] = "new"
    out = tomlrt.dumps(doc)
    assert out == td("""
        # header comment
        name = "new"  # inline
        port = 80
        """)
    assert _reparses(out)["name"] == "new"


def test_add_top_level_key_appends() -> None:
    src = "name = 'foo'\n"
    doc = tomlrt.loads(src)
    doc["count"] = 3
    out = tomlrt.dumps(doc)
    assert out == "name = 'foo'\ncount = 3\n"


def test_add_top_level_key_when_only_section_exists() -> None:
    src = "[srv]\nport = 8080\n"
    doc = tomlrt.loads(src)
    doc["name"] = "demo"
    out = tomlrt.dumps(doc)
    # Pre-header section is created at index 0; a blank line separates
    # the new top-level key from the following ``[srv]`` header.
    assert out == td("""
        name = "demo"

        [srv]
        port = 8080
        """)
    assert _reparses(out) == {"name": "demo", "srv": {"port": 8080}}


def test_add_key_inside_existing_section() -> None:
    src = "[srv]\nport = 80\n"
    doc = tomlrt.loads(src)
    srv = doc.table("srv")
    srv["host"] = "127.0.0.1"
    out = tomlrt.dumps(doc)
    assert out == td("""
        [srv]
        port = 80
        host = "127.0.0.1"
        """)
    assert _reparses(out) == {"srv": {"port": 80, "host": "127.0.0.1"}}


def test_delete_scalar_removes_line_with_leading_trivia() -> None:
    src = td("""
        a = 1
        # this comment belongs to b
        b = 2
        c = 3
        """)
    doc = tomlrt.loads(src)
    del doc["b"]
    out = tomlrt.dumps(doc)
    assert out == "a = 1\nc = 3\n"


def test_delete_missing_key_raises_keyerror() -> None:
    doc = tomlrt.loads("a = 1\n")
    with pytest.raises(KeyError):
        del doc["missing"]


def test_set_overwrites_dotted_prefix() -> None:
    src = "[a]\nb.c = 1\n"
    doc = tomlrt.loads(src)
    a = doc.table("a")
    a["b"] = 2
    out = tomlrt.dumps(doc)
    assert _reparses(out) == {"a": {"b": 2}}


def test_set_overwrites_implicit_child_table() -> None:
    src = "[a.b]\nx = 1\n"
    doc = tomlrt.loads(src)
    doc["a"] = 5
    out = tomlrt.dumps(doc)
    assert _reparses(out) == {"a": 5}


def test_quoted_key_when_bare_invalid() -> None:
    doc = tomlrt.loads("")
    doc["weird key.com"] = 1
    out = tomlrt.dumps(doc)
    assert out == '"weird key.com" = 1\n'
    assert _reparses(out) == {"weird key.com": 1}


def test_quoted_key_escapes_special_chars() -> None:
    """Keys containing ``\\``, ``"`` or control chars are basic-quoted with escapes."""
    doc = tomlrt.loads("")
    doc['a"b'] = 1
    doc["c\\d"] = 2
    doc["e\tf"] = 3  # tab is a control char (U+0009)
    out = tomlrt.dumps(doc)
    assert _reparses(out) == {'a"b': 1, "c\\d": 2, "e\tf": 3}
    # Round-trip confirms the emitted form is parseable.
    assert tomlrt.loads(out) == doc


def test_delete_top_level_dotted_key_in_preamble() -> None:
    # ``a.b = 1`` lives in the headerless preamble: deleting ``a``
    # must classify it as a dotted KV in an unattached section.
    doc = tomlrt.loads("a.b = 1\nc = 2\n")
    del doc["a"]
    assert "a" not in doc
    assert tomlrt.dumps(doc) == "c = 2\n"


def test_delete_dotted_key_inside_aot_entry() -> None:
    # An AoT entry holds ``foo.bar = 1`` as a dotted KV: deleting
    # ``foo`` walks the AoT-anchored slow path in ``_classify``.
    doc = tomlrt.loads("[[items]]\nfoo.bar = 1\nkeep = 2\n")
    del doc.aot("items")[0]["foo"]
    assert "foo" not in doc.aot("items")[0]
    assert doc.aot("items")[0]["keep"] == 2


def test_overwrite_implicit_supertable_inside_aot_entry() -> None:
    # An AoT entry has a deeper ``[items.deep.nested]`` sub-section,
    # so ``items[0]['deep']`` is an implicit super-table; assigning
    # a scalar to it must purge the deeper section.
    doc = tomlrt.loads("[[items]]\n[items.deep.nested]\nx = 2\n")
    doc.aot("items")[0]["deep"] = "scalar"
    assert doc.aot("items")[0]["deep"] == "scalar"


# ---------------------------------------------------------------------------
# Inline table mutation
# ---------------------------------------------------------------------------


def test_inline_table_replace() -> None:
    src = "obj = { a = 1, b = 2 }\n"
    doc = tomlrt.loads(src)
    obj = doc.table("obj")
    obj["a"] = 99
    out = tomlrt.dumps(doc)
    assert out == "obj = { a = 99, b = 2 }\n"


def test_inline_table_append() -> None:
    src = "obj = { a = 1 }\n"
    doc = tomlrt.loads(src)
    obj = doc.table("obj")
    obj["b"] = 2
    out = tomlrt.dumps(doc)
    assert out == "obj = { a = 1, b = 2 }\n"
    assert _reparses(out) == {"obj": {"a": 1, "b": 2}}


def test_inline_table_delete_last_clears_trailing_comma() -> None:
    src = "obj = { a = 1, b = 2 }\n"
    doc = tomlrt.loads(src)
    obj = doc.table("obj")
    del obj["b"]
    out = tomlrt.dumps(doc)
    assert _reparses(out) == {"obj": {"a": 1}}


def test_inline_table_delete_last_preserves_surviving_eol_comment() -> None:
    src = td("""
        t = {
          a = 1, # aye
          b = 2, # bee
        }
        """)
    doc = tomlrt.loads(src)
    del doc.table("t")["b"]
    assert tomlrt.dumps(doc) == td("""
        t = {
          a = 1, # aye
        }
        """)
    assert _reparses(tomlrt.dumps(doc))


def test_inline_table_delete_last_no_trailing_comma_preserves_eol() -> None:
    src = td("""
        t = {
          a = 1, # aye
          b = 2 # bee
        }
        """)
    doc = tomlrt.loads(src)
    del doc.table("t")["b"]
    assert tomlrt.dumps(doc) == td("""
        t = {
          a = 1 # aye
        }
        """)
    assert _reparses(tomlrt.dumps(doc))


def test_inline_table_delete_dotted_prefix_preserves_surviving_eol() -> None:
    src = td("""
        t = {
          a = 1, # aye
          b.x = 1, # bx
          b.y = 2, # by
        }
        """)
    doc = tomlrt.loads(src)
    del doc.table("t")["b"]
    assert tomlrt.dumps(doc) == td("""
        t = {
          a = 1, # aye
        }
        """)
    assert _reparses(tomlrt.dumps(doc))


def test_inline_table_accepts_standalone_array_with_live_attach() -> None:
    # Standalone Array assigned into an inline table attaches live:
    # the user's reference is the value at the assignment site, and
    # the requested multiline layout is preserved (TOML 1.1 admits
    # multi-line arrays inside inline tables).
    src = "obj = { a = 1 }\n"
    doc = tomlrt.loads(src)
    obj = doc.table("obj")
    arr = Array([1, 2, 3], multiline=True)
    obj["xs"] = arr
    assert obj["xs"] is arr
    arr.append(4)
    out = tomlrt.dumps(doc)
    assert _reparses(out) == {"obj": {"a": 1, "xs": [1, 2, 3, 4]}}


def test_inline_table_rejects_section_spec() -> None:
    doc = tomlrt.loads("obj = { a = 1 }\n")
    obj = doc.table("obj")
    with pytest.raises(tomlrt.TOMLError, match="inline-style table"):
        obj["bad"] = Table.section({"x": 1})


def test_inline_table_rejects_aot_value() -> None:
    doc = tomlrt.loads("obj = { a = 1 }\n")
    obj = doc.table("obj")
    with pytest.raises(tomlrt.TOMLError, match="array-of-tables inside an inline"):
        obj["bad"] = tomlrt.AoT()


# ---------------------------------------------------------------------------
# Array mutation
# ---------------------------------------------------------------------------


def test_array_append() -> None:
    src = "xs = [1, 2, 3]\n"
    doc = tomlrt.loads(src)
    xs = doc.array("xs")
    xs.append(4)
    assert list(xs) == [1, 2, 3, 4]
    out = tomlrt.dumps(doc)
    assert _reparses(out) == {"xs": [1, 2, 3, 4]}


def test_array_append_to_empty_with_tab_indented_comment_preserves_tab() -> None:
    # The only indent signal in the empty container is the tab before
    # the comment line. Appending must reuse it instead of falling back
    # to the four-space default.
    doc = tomlrt.loads("a = [\n\t# hi\n]\n")
    arr = doc.array("a")
    arr.append(1)
    assert tomlrt.dumps(doc) == "a = [\n\t# hi\n\t1,\n]\n"


def test_array_pop() -> None:
    doc = tomlrt.loads("xs = [10, 20, 30]\n")
    xs = doc.array("xs")
    v = xs.pop()
    assert v == 30
    assert list(xs) == [10, 20]
    out = tomlrt.dumps(doc)
    assert _reparses(out) == {"xs": [10, 20]}


def test_array_pop_out_of_range_raises_indexerror() -> None:
    doc = tomlrt.loads("xs = [1, 2, 3]\n")
    xs = doc.array("xs")
    with pytest.raises(IndexError, match="out of range"):
        xs.pop(99)
    with pytest.raises(IndexError, match="out of range"):
        xs.pop(-99)


def test_array_remove_missing_raises_valueerror() -> None:
    doc = tomlrt.loads("xs = [1, 2, 3]\n")
    xs = doc.array("xs")
    with pytest.raises(ValueError, match="not in array"):
        xs.remove(99)


def test_array_delitem_out_of_range_raises_indexerror() -> None:
    doc = tomlrt.loads("xs = [1, 2]\n")
    xs = doc.array("xs")
    with pytest.raises(IndexError):
        del xs[99]


def test_array_delitem_on_empty_array_raises_indexerror() -> None:
    doc = tomlrt.loads("xs = []\n")
    xs = doc.array("xs")
    with pytest.raises(IndexError):
        del xs[0]


def test_array_setitem_slice_extended_size_mismatch_raises_valueerror() -> None:
    doc = tomlrt.loads("xs = [1, 2, 3, 4, 5]\n")
    xs = doc.array("xs")
    with pytest.raises(ValueError, match="extended slice"):
        xs[::2] = [10, 20]


def test_array_setitem_int() -> None:
    doc = tomlrt.loads("xs = [1, 2, 3]\n")
    xs = doc.array("xs")
    xs[1] = 22
    out = tomlrt.dumps(doc)
    assert _reparses(out) == {"xs": [1, 22, 3]}


def test_array_setitem_slice() -> None:
    doc = tomlrt.loads("xs = [1, 2, 3, 4]\n")
    xs = doc.array("xs")
    xs[1:3] = [22, 33, 44]
    out = tomlrt.dumps(doc)
    assert _reparses(out) == {"xs": [1, 22, 33, 44, 4]}


def test_array_setitem_slice_matches_list_semantics() -> None:
    # Array slice-assignment should accept any iterable (matching plain
    # ``list``), and reject non-iterables with TypeError. The previous
    # implementation used ``assert``, which silently did the wrong
    # thing under ``python -O``.
    doc = tomlrt.loads("xs = [1, 2, 3]\n")
    xs = doc.array("xs")
    # Strings iterate to characters, like list.__setitem__ does.
    xs[0:1] = "ab"
    assert list(xs) == ["a", "b", 2, 3]
    # Non-iterables raise TypeError, like list.__setitem__ does.
    with pytest.raises(TypeError):
        xs[0:1] = 5  # type: ignore[call-overload]  # ty: ignore[invalid-assignment]


def test_array_setitem_slice_empties_multiline() -> None:
    # Slice-assigning an empty iterable to a multi-line array should
    # leave the canonical empty form, the same as ``arr.clear()`` or
    # ``del arr[:]``.
    src = td(
        """
        arr = [
            1,
            2,
        ]
        """
    )
    doc = tomlrt.loads(src)
    doc.array("arr")[:] = []
    assert tomlrt.dumps(doc) == "arr = [\n]\n"


def test_array_delitem_slice() -> None:
    doc = tomlrt.loads("xs = [1, 2, 3, 4]\n")
    xs = doc.array("xs")
    del xs[1:3]
    out = tomlrt.dumps(doc)
    assert _reparses(out) == {"xs": [1, 4]}


def test_array_setitem_slice_preserves_eol_comments() -> None:
    src = td(
        """
        arr = [
            1, # one
            2, # two
            3, # three
        ]
        """
    )
    doc = tomlrt.loads(src)
    doc.array("arr")[1:2] = []
    assert tomlrt.dumps(doc) == td(
        """
        arr = [
            1, # one
            3, # three
        ]
        """
    )


def test_array_setitem_slice_preserves_leading_comments() -> None:
    src = td(
        """
        arr = [
            1,
            2,
            3,
        ]
        """
    )
    doc = tomlrt.loads(src)
    arr = doc.array("arr")
    arr.leading_comments[2] = ["above three"]
    arr[1:2] = []
    assert tomlrt.dumps(doc) == td(
        """
        arr = [
            1,
            # above three
            3,
        ]
        """
    )


def test_array_setitem_slice_preserves_bracket_eol_comment() -> None:
    src = td(
        """
        arr = [ # tail
            1,
            2,
            3,
        ]
        """
    )
    doc = tomlrt.loads(src)
    doc.array("arr")[1:2] = [99]
    assert tomlrt.dumps(doc) == td(
        """
        arr = [ # tail
            1,
            99,
            3,
        ]
        """
    )


def test_array_clear_and_append() -> None:
    doc = tomlrt.loads("xs = [1, 2, 3]\n")
    xs = doc.array("xs")
    xs.clear()
    xs.append("hi")
    out = tomlrt.dumps(doc)
    assert _reparses(out) == {"xs": ["hi"]}


def test_array_extend_iadd() -> None:
    doc = tomlrt.loads("xs = []\n")
    xs = doc.array("xs")
    xs.extend([1, 2])
    xs += [3, 4]
    out = tomlrt.dumps(doc)
    assert _reparses(out) == {"xs": [1, 2, 3, 4]}


def test_array_sort_reverse() -> None:
    doc = tomlrt.loads("xs = [3, 1, 2]\n")
    xs = doc.array("xs")
    xs.sort()
    assert list(xs) == [1, 2, 3]
    xs.reverse()
    out = tomlrt.dumps(doc)
    assert _reparses(out) == {"xs": [3, 2, 1]}


def test_array_imul() -> None:
    doc = tomlrt.loads("xs = [1, 2]\n")
    xs = doc.array("xs")
    xs *= 3
    out = tomlrt.dumps(doc)
    assert _reparses(out) == {"xs": [1, 2, 1, 2, 1, 2]}


def test_array_imul_by_one_is_identity() -> None:
    """``arr *= 1`` short-circuits — output is byte-identical to the input."""
    src = "xs = [1, 2]\n"
    doc = tomlrt.loads(src)
    xs = doc.array("xs")
    xs *= 1
    assert list(xs) == [1, 2]
    assert tomlrt.dumps(doc) == src


def test_array_remove() -> None:
    doc = tomlrt.loads("xs = [1, 2, 3, 2]\n")
    xs = doc.array("xs")
    xs.remove(2)
    out = tomlrt.dumps(doc)
    assert _reparses(out) == {"xs": [1, 3, 2]}


def test_array_insert() -> None:
    doc = tomlrt.loads("xs = [1, 3]\n")
    xs = doc.array("xs")
    xs.insert(1, 2)
    out = tomlrt.dumps(doc)
    assert _reparses(out) == {"xs": [1, 2, 3]}


def test_array_insert_negative_index_clamps_to_zero() -> None:
    doc = tomlrt.loads("xs = [1, 2, 3]\n")
    xs = doc.array("xs")
    xs.insert(-99, 0)
    assert tomlrt.dumps(doc) == "xs = [0, 1, 2, 3]\n"


def test_array_insert_past_end_appends() -> None:
    doc = tomlrt.loads("xs = [1, 2, 3]\n")
    xs = doc.array("xs")
    xs.insert(99, 4)
    assert tomlrt.dumps(doc) == "xs = [1, 2, 3, 4]\n"


def test_array_insert_at_zero_does_not_duplicate_leading_comment() -> None:
    # The header comment ``# head`` is anchored to the array's opening
    # bracket (no newline before it). On insert(0, ...) it must stay
    # there, and the new item must land on its own indented line.
    doc = tomlrt.loads(
        td("""
        a = [# head
         1,
        ]
        """)
    )
    arr = doc.array("a")
    arr.insert(0, 99)
    assert tomlrt.dumps(doc) == td("""
        a = [# head
         99,
         1,
        ]
        """)


def test_array_delete_zero_keeps_next_leading_comment_after_prior_eol() -> None:
    """Regression for #122: ``del arr[0]`` when ``items[1].leading`` is in
    post-EOL shape must migrate item 1's leading comment into ``header_trivia``
    rather than dropping it.
    """
    doc = tomlrt.loads(
        td("""
        arr = [
            # leading a
            "a", # eol a
            # leading b
            "b",
        ]
        """)
    )
    arr = doc.array("arr")
    del arr[0]
    assert tomlrt.dumps(doc) == td("""
        arr = [
            # leading b
            "b",
        ]
        """)


def test_array_insert_after_eol_item_does_not_duplicate_following_comment() -> None:
    """Regression for #122: when inserting between an EOL-bearing item
    and a follower whose ``leading`` carries an above-block in post-EOL
    shape, the follower's leading comment must remain attached to the
    follower (not be reassigned to the inserted item as an EOL).
    """
    doc = tomlrt.loads(
        td("""
        arr = [
            # leading a
            "a", # eol a
            # leading b
            "b",
        ]
        """)
    )
    arr = doc.array("arr")
    arr.insert(1, "z")
    assert tomlrt.dumps(doc) == td("""
        arr = [
            # leading a
            "a", # eol a
            "z",
            # leading b
            "b",
        ]
        """)


def test_array_append_after_eol_item_does_not_duplicate_prior_comment() -> None:
    """Regression for #122: appending after an EOL-bearing item must not
    leave the appended item sharing the EOL row of the previous item.
    """
    doc = tomlrt.loads(
        td("""
        arr = [
            # leading a
            "a", # eol a
            # leading b
            "b",
        ]
        """)
    )
    arr = doc.array("arr")
    arr.append("z")
    assert tomlrt.dumps(doc) == td("""
        arr = [
            # leading a
            "a", # eol a
            # leading b
            "b",
            "z",
        ]
        """)


# Every Array/AoT mutator must be wired through the CST so the
# rendered output stays in sync with in-memory mutations.
@pytest.mark.parametrize(
    "name",
    [
        "append",
        "extend",
        "insert",
        "pop",
        "remove",
        "clear",
        "sort",
        "reverse",
        "__setitem__",
        "__delitem__",
        "__iadd__",
        "__imul__",
    ],
)
def test_every_array_mutator_is_overridden(name: str) -> None:
    array_method = getattr(tomlrt.Array, name, None)
    list_method = getattr(list, name, None)
    assert array_method is not None
    assert list_method is not None
    assert array_method is not list_method, (
        f"Array.{name} must be overridden so mutation routes through CST"
    )


# ---------------------------------------------------------------------------
# Container assignment / deep clone
# ---------------------------------------------------------------------------


def test_assigning_array_deep_clones() -> None:
    src = "src = [1, 2, 3]\n"
    doc = tomlrt.loads(src)
    src_arr = doc.array("src")
    doc["dst"] = src_arr
    dst = doc.array("dst")
    dst.append(99)
    assert list(src_arr) == [1, 2, 3]
    assert list(dst) == [1, 2, 3, 99]
    out = tomlrt.dumps(doc)
    parsed = _reparses(out)
    assert parsed == {"src": [1, 2, 3], "dst": [1, 2, 3, 99]}


def test_assigning_dict_creates_inline_table() -> None:
    doc = tomlrt.loads("")
    doc["obj"] = {"a": 1, "b": "two"}
    out = tomlrt.dumps(doc)
    assert _reparses(out) == {"obj": {"a": 1, "b": "two"}}


def test_assigning_list_creates_inline_array() -> None:
    doc = tomlrt.loads("")
    doc["nums"] = [1, 2, 3]
    out = tomlrt.dumps(doc)
    assert _reparses(out) == {"nums": [1, 2, 3]}


def test_replace_scalar_with_array() -> None:
    doc = tomlrt.loads("x = 1\n")
    doc["x"] = [True, False]
    out = tomlrt.dumps(doc)
    assert _reparses(out) == {"x": [True, False]}


# ---------------------------------------------------------------------------
# AoT mutators (pop / clear / __delitem__) — list-style mutation surface
# ---------------------------------------------------------------------------


def _aot_doc() -> tomlrt.Document:
    return tomlrt.loads(
        '[[pkg]]\nname = "a"\n\n'
        "[pkg.dep]\nx = 1\n\n"
        '[[pkg]]\nname = "b"\n\n'
        '[[pkg]]\nname = "c"\n'
    )


def test_aot_pop_default_removes_last_entry_and_owned_subsections() -> None:
    doc = _aot_doc()
    aot = doc.aot("pkg")
    popped = aot.pop()
    assert isinstance(popped, tomlrt.Table)
    assert popped["name"] == "c"
    assert len(aot) == 2
    out = tomlrt.dumps(doc)
    assert _reparses(out) == {
        "pkg": [
            {"name": "a", "dep": {"x": 1}},
            {"name": "b"},
        ],
    }


def test_aot_pop_first_entry_takes_owned_subsections_with_it() -> None:
    doc = _aot_doc()
    aot = doc.aot("pkg")
    popped = aot.pop(0)
    assert popped["name"] == "a"
    out = tomlrt.dumps(doc)
    assert out == td("""
        [[pkg]]
        name = "b"

        [[pkg]]
        name = "c"
        """)
    assert _reparses(out) == {"pkg": [{"name": "b"}, {"name": "c"}]}


def test_aot_pop_negative_index() -> None:
    doc = _aot_doc()
    aot = doc.aot("pkg")
    popped = aot.pop(-2)
    assert popped["name"] == "b"
    assert _reparses(tomlrt.dumps(doc)) == {
        "pkg": [{"name": "a", "dep": {"x": 1}}, {"name": "c"}],
    }


def test_aot_pop_index_out_of_range_raises() -> None:
    doc = _aot_doc()
    aot = doc.aot("pkg")
    with pytest.raises(IndexError, match="pop index out of range"):
        aot.pop(99)
    with pytest.raises(IndexError, match="pop index out of range"):
        aot.pop(-99)


def test_aot_pop_returns_held_entry_object_and_remains_mutable() -> None:
    """Held entry references survive pop and remain usable.

    Mirrors `del doc[k]`'s orphan-transplant semantics: a user who
    held the entry pre-pop ends up with the same object the pop
    returns, detached from the doc, fully mutable, re-attachable.
    """
    doc = _aot_doc()
    aot = doc.aot("pkg")
    held = aot[0]
    popped = aot.pop(0)
    assert popped is held
    held["extra"] = 99
    assert popped["extra"] == 99
    new_doc = tomlrt.loads("")
    new_doc["pkg"] = popped
    assert _reparses(tomlrt.dumps(new_doc))["pkg"]["extra"] == 99


def test_aot_clear_removes_all_entries_and_owned_subsections() -> None:
    doc = _aot_doc()
    aot = doc.aot("pkg")
    aot.clear()
    assert len(aot) == 0
    out = tomlrt.dumps(doc)
    assert out == ""
    assert _reparses(out) == {}


def test_aot_delitem_index_pops_one() -> None:
    doc = _aot_doc()
    aot = doc.aot("pkg")
    del aot[1]
    assert _reparses(tomlrt.dumps(doc)) == {
        "pkg": [{"name": "a", "dep": {"x": 1}}, {"name": "c"}],
    }


def test_aot_delitem_slice_removes_range() -> None:
    doc = _aot_doc()
    aot = doc.aot("pkg")
    del aot[1:]
    assert _reparses(tomlrt.dumps(doc)) == {"pkg": [{"name": "a", "dep": {"x": 1}}]}


def test_aot_delitem_slice_with_step() -> None:
    doc = tomlrt.loads(
        td("""
            [[p]]
            n=1
            [[p]]
            n=2
            [[p]]
            n=3
            [[p]]
            n=4
            """),
    )
    aot = doc.aot("p")
    del aot[::2]
    assert _reparses(tomlrt.dumps(doc)) == {"p": [{"n": 2}, {"n": 4}]}


def test_aot_setitem_replaces_entry() -> None:
    doc = _aot_doc()
    aot = doc.aot("pkg")
    aot[0] = {"new": True}
    rendered = tomlrt.dumps(doc)
    assert _reparses(rendered)["pkg"][0] == {"new": True}
    assert len(doc.aot("pkg")) == 3


def test_aot_setitem_negative_index() -> None:
    doc = _aot_doc()
    aot = doc.aot("pkg")
    aot[-1] = {"replaced": True}
    rendered = tomlrt.dumps(doc)
    assert _reparses(rendered)["pkg"][-1] == {"replaced": True}


def test_aot_setitem_out_of_range_raises() -> None:
    doc = _aot_doc()
    aot = doc.aot("pkg")
    with pytest.raises(IndexError):
        aot[99] = {"x": 1}


def test_aot_setitem_slice_validates_before_mutating() -> None:
    doc = _aot_doc()
    original = tomlrt.dumps(doc)
    aot = doc.aot("pkg")
    with pytest.raises(TypeError):
        aot[0:2] = [{"ok": True}, "not a mapping"]  # type: ignore[list-item]  # ty: ignore[invalid-assignment]
    assert tomlrt.dumps(doc) == original


def test_aot_iadd_appends_entries_to_cst() -> None:
    doc = _aot_doc()
    aot = doc.aot("pkg")
    aot += [{"name": "d"}, {"name": "e"}]
    rendered = tomlrt.dumps(doc)
    assert _reparses(rendered)["pkg"] == [
        {"name": "a", "dep": {"x": 1}},
        {"name": "b"},
        {"name": "c"},
        {"name": "d"},
        {"name": "e"},
    ]


def test_aot_imul_replicates_entries_in_cst() -> None:
    doc = tomlrt.loads(
        td("""
        [[t]]
        x = 1
        [[t]]
        x = 2
        """)
    )
    aot = doc.aot("t")
    aot *= 3
    rendered = tomlrt.dumps(doc)
    assert _reparses(rendered)["t"] == [
        {"x": 1},
        {"x": 2},
        {"x": 1},
        {"x": 2},
        {"x": 1},
        {"x": 2},
    ]


def test_aot_imul_zero_clears() -> None:
    doc = tomlrt.loads(
        td("""
        [[t]]
        x = 1
        [[t]]
        x = 2
        """)
    )
    aot = doc.aot("t")
    aot *= 0
    assert "t" not in tomlrt.loads(tomlrt.dumps(doc))


def test_aot_imul_detached_replicates_entries() -> None:
    """Detached AoT must follow list ``*=`` semantics."""
    aot = AoT([{"a": 1}, {"a": 2}])
    aot *= 3
    assert [dict(t) for t in aot] == [
        {"a": 1},
        {"a": 2},
        {"a": 1},
        {"a": 2},
        {"a": 1},
        {"a": 2},
    ]


def test_aot_imul_detached_zero_clears() -> None:
    aot = AoT([{"a": 1}])
    aot *= 0
    assert list(aot) == []


def test_aot_imul_detached_one_is_noop() -> None:
    aot = AoT([{"a": 1}])
    aot *= 1
    assert [dict(t) for t in aot] == [{"a": 1}]


def test_aot_imul_detached_deep_copies_entries() -> None:
    aot = AoT([{"a": 1}])
    aot *= 2
    aot[0]["a"] = 99
    assert aot[1]["a"] == 1


def test_aot_detached_accepts_generic_mapping() -> None:
    proxy = MappingProxyType({"x": 1})

    aot = AoT()
    aot.append(proxy)
    assert list(aot) == [{"x": 1}]

    aot = AoT([{"x": 1}])
    aot.insert(0, MappingProxyType({"x": 0}))
    assert [dict(t) for t in aot] == [{"x": 0}, {"x": 1}]

    aot = AoT([{"x": 1}])
    aot[0] = MappingProxyType({"x": 9})
    assert list(aot) == [{"x": 9}]


def test_aot_reverse_reorders_cst() -> None:
    doc = tomlrt.loads(
        td("""
        [[t]]
        x = 1
        [[t]]
        x = 2
        [[t]]
        x = 3
        """)
    )
    aot = doc.aot("t")
    aot.reverse()
    assert _reparses(tomlrt.dumps(doc))["t"] == [{"x": 3}, {"x": 2}, {"x": 1}]


def test_aot_sort_reorders_cst() -> None:
    doc = tomlrt.loads(
        td("""
        [[t]]
        x = 3
        [[t]]
        x = 1
        [[t]]
        x = 2
        """)
    )
    aot = doc.aot("t")
    aot.sort(key=lambda e: e["x"])
    assert _reparses(tomlrt.dumps(doc))["t"] == [{"x": 1}, {"x": 2}, {"x": 3}]


def test_aot_imul_preserves_inter_entry_separator() -> None:
    src = td("""
        [[t]]
        x = 1  # one

        [[t]]
        x = 2  # two
        """)
    doc = tomlrt.loads(src)
    doc.aot("t").__imul__(2)
    assert tomlrt.dumps(doc) == (
        "[[t]]\nx = 1  # one\n\n[[t]]\nx = 2  # two\n\n"
        "[[t]]\nx = 1  # one\n\n[[t]]\nx = 2  # two\n"
    )


def test_aot_imul_preserves_per_entry_leading_comments() -> None:
    src = td("""
        # A
        [[t]]
        x = 1

        # B
        [[t]]
        x = 2
        """)
    doc = tomlrt.loads(src)
    doc.aot("t").__imul__(2)
    assert tomlrt.dumps(doc) == (
        "# A\n[[t]]\nx = 1\n\n# B\n[[t]]\nx = 2\n\n"
        "# A\n[[t]]\nx = 1\n\n# B\n[[t]]\nx = 2\n"
    )


def test_aot_sort_preserves_formatting_byte_exact() -> None:
    src = td("""
        [[t]]
        x = 3  # third

        [[t]]
        x = 1  # first

        [[t]]
        x = 2  # second
        """)
    doc = tomlrt.loads(src)
    doc.aot("t").sort(key=lambda e: e["x"])
    assert tomlrt.dumps(doc) == (
        td("""
            [[t]]
            x = 1  # first

            [[t]]
            x = 2  # second

            [[t]]
            x = 3  # third
            """)
    )


def test_aot_reverse_preserves_formatting_byte_exact() -> None:
    src = td("""
        [[t]]
        name = "a"  # first

        [[t]]
        name = "b"  # second
        """)
    doc = tomlrt.loads(src)
    doc.aot("t").reverse()
    assert tomlrt.dumps(doc) == (
        td("""
            [[t]]
            name = "b"  # second

            [[t]]
            name = "a"  # first
            """)
    )


def test_aot_reverse_preserves_owned_subtables() -> None:
    doc = tomlrt.loads(
        td("""
            [[t]]
            name = "a"
            [t.sub]
            y = 1
            [[t]]
            name = "b"
            [t.sub]
            y = 2
            """)
    )
    aot = doc.aot("t")
    aot.reverse()
    parsed = _reparses(tomlrt.dumps(doc))["t"]
    assert parsed == [
        {"name": "b", "sub": {"y": 2}},
        {"name": "a", "sub": {"y": 1}},
    ]


def test_aot_reverse_moves_leading_comments_with_entries() -> None:
    src = td("""
        # A
        [[t]]
        x = 1

        # B
        [[t]]
        x = 2

        # C
        [[t]]
        x = 3
        """)
    doc = tomlrt.loads(src)
    doc.aot("t").reverse()
    assert tomlrt.dumps(doc) == (
        td("""
            # C
            [[t]]
            x = 3

            # B
            [[t]]
            x = 2

            # A
            [[t]]
            x = 1
            """)
    )


def test_aot_sort_moves_leading_comments_with_entries() -> None:
    src = td("""
        # x=2
        [[t]]
        x = 2

        # x=3
        [[t]]
        x = 3

        # x=1
        [[t]]
        x = 1
        """)
    doc = tomlrt.loads(src)
    doc.aot("t").sort(key=lambda e: e["x"])
    assert tomlrt.dumps(doc) == (
        td("""
            # x=1
            [[t]]
            x = 1

            # x=2
            [[t]]
            x = 2

            # x=3
            [[t]]
            x = 3
            """)
    )


def test_aot_reverse_with_partial_leading_comments() -> None:
    # Only the middle entry has a leading comment; reversing should
    # carry it with that entry and leave the new first/last entries
    # commentless.
    src = td("""
        [[t]]
        x = 1

        # B
        [[t]]
        x = 2

        [[t]]
        x = 3
        """)
    doc = tomlrt.loads(src)
    doc.aot("t").reverse()
    assert tomlrt.dumps(doc) == (
        td("""
        [[t]]
        x = 3

        # B
        [[t]]
        x = 2

        [[t]]
        x = 1
        """)
    )


def test_aot_remove_drops_first_matching_entry_from_cst() -> None:
    doc = tomlrt.loads(
        td("""
        [[t]]
        x = 1
        [[t]]
        x = 2
        [[t]]
        x = 3
        """)
    )
    aot = doc.aot("t")
    aot.remove(aot[1])
    assert _reparses(tomlrt.dumps(doc))["t"] == [{"x": 1}, {"x": 3}]


def test_aot_remove_missing_raises_value_error() -> None:
    doc = tomlrt.loads("[[t]]\nx = 1\n")
    aot = doc.aot("t")
    with pytest.raises(ValueError, match="not in list"):
        aot.remove({"x": 999})


def test_aot_slice_replace_contiguous() -> None:
    doc = tomlrt.loads(
        td("""
            [[items]]
            name = "a"

            [[items]]
            name = "b"

            [[items]]
            name = "c"
            """)
    )
    items = doc.aot("items")
    items[1:3] = [
        tomlrt.Table.inline({"name": "B"}),
        tomlrt.Table.inline({"name": "C"}),
    ]
    assert [t["name"] for t in items] == ["a", "B", "C"]
    assert _reparses(tomlrt.dumps(doc))["items"] == [
        {"name": "a"},
        {"name": "B"},
        {"name": "C"},
    ]


def test_aot_slice_replace_extended_matching_length() -> None:
    doc = tomlrt.loads(
        td("""
            [[items]]
            name = "a"

            [[items]]
            name = "b"

            [[items]]
            name = "c"
            """)
    )
    items = doc.aot("items")
    items[::2] = [
        tomlrt.Table.inline({"name": "A"}),
        tomlrt.Table.inline({"name": "C"}),
    ]
    assert [t["name"] for t in items] == ["A", "b", "C"]


def test_aot_slice_replace_extended_mismatched_length_raises() -> None:
    doc = tomlrt.loads(
        td("""
            [[items]]
            name = "a"

            [[items]]
            name = "b"

            [[items]]
            name = "c"
            """)
    )
    items = doc.aot("items")
    with pytest.raises(ValueError, match="extended slice"):
        items[::2] = [tomlrt.Table.inline({"name": "only"})]


def test_aot_reverse_on_empty_is_noop() -> None:
    doc = tomlrt.loads("[[items]]\n")
    items = doc.aot("items")
    items.clear()
    items.reverse()
    assert list(items) == []


def test_aot_sort_on_empty_is_noop() -> None:
    doc = tomlrt.loads("[[items]]\n")
    items = doc.aot("items")
    items.clear()
    items.sort(key=lambda t: t.get("name", ""))
    assert list(items) == []


# ---------------------------------------------------------------------------
# Array.sort(key=...), Array *= n, Array.table() type-error
# ---------------------------------------------------------------------------


def test_array_sort_with_key_callable() -> None:
    doc = tomlrt.loads('xs = ["bb", "a", "ccc"]\n')
    xs = doc.array("xs")
    xs.sort(key=lambda v: len(str(v)))
    assert _reparses(tomlrt.dumps(doc)) == {"xs": ["a", "bb", "ccc"]}


def test_array_imul_zero_clears() -> None:
    doc = tomlrt.loads("xs = [1, 2, 3]\n")
    xs = doc.array("xs")
    xs *= 0
    assert list(xs) == []
    assert _reparses(tomlrt.dumps(doc)) == {"xs": []}


def test_array_imul_negative_clears() -> None:
    doc = tomlrt.loads("xs = [1, 2]\n")
    xs = doc.array("xs")
    xs *= -3
    assert list(xs) == []


def test_array_imul_repeats_items() -> None:
    doc = tomlrt.loads("xs = [1, 2]\n")
    xs = doc.array("xs")
    xs *= 3
    assert _reparses(tomlrt.dumps(doc)) == {"xs": [1, 2, 1, 2, 1, 2]}


def test_array_imul_preserves_no_trailing_comma() -> None:
    """Single-line array without trailing comma must not gain one."""
    doc = tomlrt.loads("xs = [1, 2, 3]\n")
    xs = doc.array("xs")
    xs *= 2
    out = tomlrt.dumps(doc)
    assert out == "xs = [1, 2, 3, 1, 2, 3]\n"


def test_array_imul_preserves_trailing_comma_when_present() -> None:
    doc = tomlrt.loads("xs = [1, 2, 3,]\n")
    xs = doc.array("xs")
    xs *= 2
    out = tomlrt.dumps(doc)
    assert out == "xs = [1, 2, 3, 1, 2, 3,]\n"


def test_array_imul_preserves_multiline_no_trailing_comma() -> None:
    src = "xs = [\n  1,\n  2,\n  3\n]\n"
    doc = tomlrt.loads(src)
    xs = doc.array("xs")
    xs *= 2
    out = tomlrt.dumps(doc)
    assert out == "xs = [\n  1,\n  2,\n  3,\n  1,\n  2,\n  3\n]\n"


def test_array_imul_inline_table_copies_render_mutations() -> None:
    doc = tomlrt.loads("xs = [{a = 1}]\n")
    arr = doc.array("xs")
    arr *= 2
    arr.table(1)["b"] = 2
    assert tomlrt.dumps(doc) == "xs = [{a = 1}, {a = 1, b = 2}]\n"


def test_array_imul_nested_array_copies_render_mutations() -> None:
    doc = tomlrt.loads("xs = [[1]]\n")
    arr = doc.array("xs")
    arr *= 2
    arr.array(1).append(2)
    assert tomlrt.dumps(doc) == "xs = [[1], [1, 2]]\n"


def test_array_table_typed_accessor_raises_on_non_table_item() -> None:
    doc = tomlrt.loads("xs = [1, 2]\n")
    xs = doc.array("xs")
    with pytest.raises(TypeError, match="not a Table"):
        xs.table(0)


def test_array_array_typed_accessor_raises_on_non_array_item() -> None:
    doc = tomlrt.loads("xs = [1, 2]\n")
    xs = doc.array("xs")
    with pytest.raises(TypeError, match="not an Array"):
        xs.array(0)


def test_array_append_aot_raises() -> None:
    # An AoT only renders as ``[[ ... ]]`` sections; trying to splice
    # one into an inline array has no valid serialisation.
    doc = tomlrt.loads("xs = [1]\n")
    with pytest.raises(tomlrt.TOMLError, match="Cannot store an array-of-tables"):
        doc.array("xs").append(tomlrt.AoT())


def test_dotted_subtable_delitem_missing_key_raises_keyerror() -> None:
    doc = tomlrt.loads("a.b = 1\n")
    sub = doc["a"]
    assert isinstance(sub, Table)
    with pytest.raises(KeyError, match="missing"):
        del sub["missing"]


# ---------------------------------------------------------------------------
# install / typed-accessor key-path validation
# ---------------------------------------------------------------------------


def test_install_rejects_empty_string_path() -> None:
    doc = tomlrt.loads("")
    with pytest.raises(tomlrt.TOMLError, match="must not be empty"):
        doc.install("", 1)


def test_install_rejects_empty_tuple_path() -> None:
    doc = tomlrt.loads("")
    with pytest.raises(tomlrt.TOMLError, match="must not be empty"):
        doc.install((), 1)


def test_install_rejects_string_path_with_empty_segment() -> None:
    doc = tomlrt.loads("")
    with pytest.raises(tomlrt.TOMLError, match="empty segment"):
        doc.install("a..b", 1)


def test_install_rejects_tuple_path_with_empty_segment() -> None:
    doc = tomlrt.loads("")
    with pytest.raises(tomlrt.TOMLError, match="empty segment"):
        doc.install(("a", ""), 1)


def test_install_rejects_non_string_path() -> None:
    doc = tomlrt.loads("")
    with pytest.raises(TypeError, match="key path must be str or sequence"):
        doc.install(123, 1)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]


def test_install_rejects_tuple_with_non_string_segment() -> None:
    doc = tomlrt.loads("")
    with pytest.raises(TypeError, match="segment must be str"):
        doc.install(("a", 1), 1)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]


def test_install_accepts_list_path() -> None:
    doc = tomlrt.loads("")
    doc.install(["tool", "ruff", "line-length"], 88)
    assert tomlrt.dumps(doc) == "[tool.ruff]\nline-length = 88\n"


def test_install_accepts_arbitrary_sequence_path() -> None:
    # The public type signature is ``str | Sequence[str]`` and the docs
    # say "any sequence of literal segments". A ``collections.deque``
    # is a ``Sequence`` and should be accepted.
    doc = tomlrt.loads("")
    doc.install(deque(["a", "b"]), 1)
    assert tomlrt.dumps(doc) == "[a]\nb = 1\n"


def test_ensure_table_accepts_arbitrary_sequence_path() -> None:
    doc = tomlrt.loads("")
    t = doc.ensure_table(deque(["a", "b"]))
    t["x"] = 1
    assert tomlrt.dumps(doc) == "[a.b]\nx = 1\n"


def test_ensure_table_accepts_list_path() -> None:
    doc = tomlrt.loads("")
    t = doc.ensure_table(["tool", "ruff"])
    t["line-length"] = 88
    assert tomlrt.dumps(doc) == "[tool.ruff]\nline-length = 88\n"


def test_insert_into_implicit_parent_after_chained_ensure_table() -> None:
    doc = tomlrt.Document()
    doc.ensure_table("t").ensure_table("u")["k"] = 1
    assert tomlrt.dumps(doc) == "[t.u]\nk = 1\n"
    doc["t"]["v"] = 1
    assert tomlrt.dumps(doc) == "[t]\nv = 1\n\n[t.u]\nk = 1\n"


def test_insert_into_implicit_parent_after_multi_ensure_table() -> None:
    doc = tomlrt.Document()
    a = doc.ensure_table("a")
    a.ensure_table(["b", "c"])["k"] = 1
    assert tomlrt.dumps(doc) == "[a.b.c]\nk = 1\n"
    doc["a"]["v"] = 1
    assert tomlrt.dumps(doc) == "[a]\nv = 1\n\n[a.b.c]\nk = 1\n"


def test_insert_into_implicit_parent_after_aot_attach() -> None:
    doc = tomlrt.Document()
    doc["tool"] = Table.section({})
    doc["tool"]["list"] = AoT([{"name": "foo"}])
    assert tomlrt.dumps(doc) == '[[tool.list]]\nname = "foo"\n'
    doc["tool"]["v"] = 1
    assert tomlrt.dumps(doc) == '[tool]\nv = 1\n\n[[tool.list]]\nname = "foo"\n'


def test_install_section_through_scalar_intermediate_raises_typeerror() -> None:
    """``install("a.b.c", section)`` where ``a`` is a scalar must fail loudly."""
    doc = tomlrt.loads("a = 1\n")
    with pytest.raises(TypeError, match="intermediate 'a' is not a table"):
        doc.install("a.b.c", Table.section({"x": 1}))


def test_install_section_walks_through_existing_section_intermediate() -> None:
    """``install`` reuses an existing section intermediate rather than recreating it."""
    doc = tomlrt.loads("[a]\nold = 1\n")
    doc.install("a.b.c", Table.section({"x": 1}))
    assert tomlrt.dumps(doc) == td("""
        [a]
        old = 1

        [a.b.c]
        x = 1
        """)


def test_aot_setitem_out_of_range_index_raises_indexerror() -> None:
    """``aot[99] = body`` must raise IndexError, not silently grow the AoT."""
    src = td("""
        [[t]]
        x = 1
        """)
    doc = tomlrt.loads(src)
    aot = doc.aot("t")
    with pytest.raises(IndexError, match="out of range"):
        aot[99] = {"x": 2}
    # AoT unchanged on failure.
    assert tomlrt.dumps(doc) == src


def test_aot_setitem_clone_path_out_of_range_index_raises_indexerror() -> None:
    """The trivia-preserving clone branch also bounds-checks the index."""
    src1 = td("""
        [[t]]
        x = 1
        """)
    src2 = td("""
        [[other]]
        y = 2
        """)
    doc1 = tomlrt.loads(src1)
    doc2 = tomlrt.loads(src2)
    foreign_entry = doc2.aot("other")[0]
    aot = doc1.aot("t")
    with pytest.raises(IndexError, match="out of range"):
        aot[99] = foreign_entry
    assert tomlrt.dumps(doc1) == src1


# ---------------------------------------------------------------------------
# AoT.add — append-and-return-handle convenience
# ---------------------------------------------------------------------------


def test_aot_add_returns_new_table_view() -> None:
    doc = tomlrt.loads("")
    doc["pkg"] = AoT()
    aot = doc["pkg"]
    pkg = aot.add({"name": "foo"})
    assert isinstance(pkg, tomlrt.Table)
    assert pkg["name"] == "foo"
    assert _reparses(tomlrt.dumps(doc)) == {"pkg": [{"name": "foo"}]}


def test_aot_add_default_empty_returns_blank_entry_for_population() -> None:
    doc = tomlrt.loads("")
    doc["pkg"] = AoT()
    aot = doc["pkg"]
    pkg = aot.add()
    assert dict(pkg) == {}
    pkg["name"] = "bar"
    pkg["dep"] = Table.section({"x": 1})
    assert _reparses(tomlrt.dumps(doc)) == {
        "pkg": [{"name": "bar", "dep": {"x": 1}}],
    }


def test_aot_add_returned_view_stays_live_across_subsequent_adds() -> None:
    doc = tomlrt.loads("")
    doc["pkg"] = AoT()
    aot = doc["pkg"]
    first = aot.add({"name": "a"})
    aot.add({"name": "b"})
    aot.add({"name": "c"})
    # The handle returned earlier still refers to the right entry.
    first["version"] = "1.0"
    assert _reparses(tomlrt.dumps(doc)) == {
        "pkg": [
            {"name": "a", "version": "1.0"},
            {"name": "b"},
            {"name": "c"},
        ],
    }


def test_aot_add_blank_separates_consecutive_entries() -> None:
    doc = tomlrt.loads("")
    doc["pkg"] = AoT()
    aot = doc["pkg"]
    aot.add({"name": "a"})
    aot.add({"name": "b"})
    out = tomlrt.dumps(doc)
    # Same blank-separation behaviour as append, since add wraps it.
    assert out == td("""
        [[pkg]]
        name = "a"

        [[pkg]]
        name = "b"
        """)


# ---------------------------------------------------------------------------
# Array.append/extend/insert/__setitem__ accept dict & list at type level
# ---------------------------------------------------------------------------


def test_array_append_dict_synthesises_inline_table() -> None:
    doc = tomlrt.loads("xs = []\n")
    arr = doc.array("xs")
    arr.append({"a": 1})
    out = tomlrt.dumps(doc)
    assert out == "xs = [{ a = 1 }]\n"
    parsed = _reparses(out)
    assert parsed == {"xs": [{"a": 1}]}


def test_array_append_list_synthesises_inline_array() -> None:
    doc = tomlrt.loads("xs = []\n")
    arr = doc.array("xs")
    arr.append([1, 2, 3])
    parsed = _reparses(tomlrt.dumps(doc))
    assert parsed == {"xs": [[1, 2, 3]]}


def test_array_extend_mixed_python_containers() -> None:
    doc = tomlrt.loads("xs = []\n")
    arr = doc.array("xs")
    arr.extend([{"a": 1}, [1, 2], "three"])
    parsed = _reparses(tomlrt.dumps(doc))
    assert parsed == {"xs": [{"a": 1}, [1, 2], "three"]}


def test_array_insert_dict() -> None:
    doc = tomlrt.loads("xs = [1, 3]\n")
    arr = doc.array("xs")
    arr.insert(1, {"k": "v"})
    parsed = _reparses(tomlrt.dumps(doc))
    assert parsed == {"xs": [1, {"k": "v"}, 3]}


def test_array_setitem_replaces_with_dict() -> None:
    doc = tomlrt.loads("xs = [1, 2, 3]\n")
    arr = doc.array("xs")
    arr[1] = {"k": "v"}
    parsed = _reparses(tomlrt.dumps(doc))
    assert parsed == {"xs": [1, {"k": "v"}, 3]}


# ---------------------------------------------------------------------------
# to_dict / to_list: deep snapshot helpers
# ---------------------------------------------------------------------------


def test_table_to_dict_returns_plain_dict() -> None:
    doc = tomlrt.loads(
        """
        title = "demo"
        [owner]
        name = "alice"
        """
    )
    snap = doc.to_dict()
    assert type(snap) is dict
    assert type(snap["owner"]) is dict
    assert snap == {"title": "demo", "owner": {"name": "alice"}}


def test_table_to_dict_recursively_flattens_aot_and_array() -> None:
    doc = tomlrt.loads(
        """
        xs = [1, [2, 3], { k = "v" }]

        [[pkg]]
        name = "a"
        deps = ["x", "y"]

        [[pkg]]
        name = "b"
        """
    )
    snap = doc.to_dict()
    assert snap == {
        "xs": [1, [2, 3], {"k": "v"}],
        "pkg": [
            {"name": "a", "deps": ["x", "y"]},
            {"name": "b"},
        ],
    }
    # Every container is a real dict/list, not a tomlrt view.
    assert type(snap["xs"]) is list
    assert type(snap["xs"][2]) is dict
    assert type(snap["pkg"]) is list
    assert type(snap["pkg"][0]) is dict
    assert type(snap["pkg"][0]["deps"]) is list


def test_table_to_dict_isinstance_dict() -> None:
    doc = tomlrt.loads("[tool]\nname = 'x'\n")
    snap = doc.to_dict()
    assert isinstance(snap, dict)
    assert isinstance(snap["tool"], dict)


def test_table_to_dict_independent_of_document_mutations() -> None:
    doc = tomlrt.loads(
        td("""
        a = 1
        [t]
        b = 2
        """)
    )
    snap = doc.to_dict()
    doc["a"] = 99
    doc.table("t")["b"] = 99
    assert snap == {"a": 1, "t": {"b": 2}}


def test_array_to_list_returns_plain_list() -> None:
    doc = tomlrt.loads('xs = [1, "two", { k = "v" }]\n')
    snap = doc.array("xs").to_list()
    assert type(snap) is list
    assert type(snap[2]) is dict
    assert snap == [1, "two", {"k": "v"}]


def test_aot_to_list_returns_list_of_dicts() -> None:
    doc = tomlrt.loads(
        """
        [[pkg]]
        name = "a"

        [[pkg]]
        name = "b"
        nested = { x = 1 }
        """
    )
    snap = doc.aot("pkg").to_list()
    assert type(snap) is list
    assert all(type(t) is dict for t in snap)
    assert snap == [{"name": "a"}, {"name": "b", "nested": {"x": 1}}]


def test_to_dict_round_trip_is_data_equivalent_to_tomllib() -> None:
    src = """
    title = "demo"
    xs = [1, 2, 3]

    [owner]
    name = "alice"

    [[pkg]]
    name = "a"
    """
    assert tomlrt.loads(src).to_dict() == _reparses(src)


# ---------------------------------------------------------------------------
# get_table / get_array / get_aot: typed-but-optional accessors
# ---------------------------------------------------------------------------


def test_table_get_table_returns_table_when_present() -> None:
    doc = tomlrt.loads("[t]\nx = 1\n")
    t = doc.get_table("t")
    assert t is not None
    assert t["x"] == 1


def test_table_get_table_returns_none_when_missing() -> None:
    doc = tomlrt.loads("a = 1\n")
    assert doc.get_table("nope") is None


def test_table_get_table_returns_default_when_missing() -> None:
    doc = tomlrt.loads("a = 1\n")
    sentinel: dict[str, int] = {}
    result = doc.get_table("nope", sentinel)
    assert result is sentinel


def test_table_get_table_raises_typeerror_on_wrong_type() -> None:
    doc = tomlrt.loads("a = 1\n")
    with pytest.raises(TypeError, match="not a Table"):
        doc.get_table("a")


def test_table_get_table_handles_dotted_path() -> None:
    doc = tomlrt.loads("[tool.poetry]\nname = 'x'\n")
    sub = doc.get_table("tool.poetry")
    assert sub is not None
    assert sub["name"] == "x"
    assert doc.get_table("tool.missing") is None


def test_table_typed_accessors_accept_sequence_path() -> None:
    doc = tomlrt.loads("[tool.poetry]\nname = 'x'\n[tool.'foo.bar']\nv = 1\n")
    assert doc.table(["tool", "poetry"])["name"] == "x"
    # Sequence form lets you address a key whose name contains a dot.
    inner = doc.table(("tool", "foo.bar"))
    assert inner["v"] == 1
    assert doc.get_table(["tool", "missing"]) is None


def test_table_entry_returns_value_at_path() -> None:
    doc = tomlrt.loads("[tool.poetry]\nname = 'x'\nxs = [1, 2]\n")
    assert doc.entry("tool.poetry.name") == "x"
    assert isinstance(doc.entry(("tool", "poetry")), tomlrt.Table)
    assert isinstance(doc.entry("tool.poetry.xs"), tomlrt.Array)


def test_table_entry_raises_keyerror_when_missing() -> None:
    doc = tomlrt.loads("a = 1\n")
    with pytest.raises(KeyError):
        doc.entry("nope")


def test_table_entry_raises_typeerror_on_descend_through_non_table() -> None:
    doc = tomlrt.loads("a = 1\n")
    with pytest.raises(TypeError, match="cannot descend into 'a'"):
        doc.entry("a.b")


def test_table_get_entry_returns_value_or_default() -> None:
    doc = tomlrt.loads("[tool.poetry]\nname = 'x'\n")
    assert doc.get_entry("tool.poetry.name") == "x"
    assert doc.get_entry("nope") is None
    sentinel: object = object()
    assert doc.get_entry(("tool", "missing"), sentinel) is sentinel


def test_table_get_entry_does_not_swallow_typeerror() -> None:
    doc = tomlrt.loads("a = 1\n")
    with pytest.raises(TypeError, match="cannot descend into 'a'"):
        doc.get_entry("a.b")


def test_table_get_array_returns_array_or_default() -> None:
    doc = tomlrt.loads("xs = [1, 2, 3]\n")
    arr = doc.get_array("xs")
    assert arr is not None
    assert list(arr) == [1, 2, 3]
    assert doc.get_array("nope") is None
    assert doc.get_array("nope", []) == []


def test_table_get_array_raises_typeerror_on_wrong_type() -> None:
    doc = tomlrt.loads("a = 1\n")
    with pytest.raises(TypeError, match="not an Array"):
        doc.get_array("a")


def test_table_get_aot_returns_aot_or_default() -> None:
    doc = tomlrt.loads("[[pkg]]\nname = 'a'\n")
    aot = doc.get_aot("pkg")
    assert aot is not None
    assert aot[0]["name"] == "a"
    assert doc.get_aot("nope") is None


def test_table_get_aot_raises_typeerror_on_wrong_type() -> None:
    doc = tomlrt.loads("[t]\nname = 'a'\n")
    with pytest.raises(TypeError, match="not an AoT"):
        doc.get_aot("t")


def test_array_get_array_in_range_and_default() -> None:
    doc = tomlrt.loads("xs = [[1, 2], [3, 4]]\n")
    arr = doc.array("xs")
    inner = arr.get_array(0)
    assert inner is not None
    assert list(inner) == [1, 2]
    assert arr.get_array(99) is None
    assert arr.get_array(99, "fallback") == "fallback"


def test_array_get_table_in_range_and_default() -> None:
    doc = tomlrt.loads("xs = [{ a = 1 }, { a = 2 }]\n")
    arr = doc.array("xs")
    t = arr.get_table(0)
    assert t is not None
    assert t["a"] == 1
    assert arr.get_table(99) is None


def test_array_get_table_raises_typeerror_on_wrong_type() -> None:
    doc = tomlrt.loads("xs = [1, 2]\n")
    arr = doc.array("xs")
    with pytest.raises(TypeError, match="not a Table"):
        arr.get_table(0)


# ---------------------------------------------------------------------------
# Loosened typing on __getitem__ and the MutableMapping/list parameter.
#
# These are partly type-ergonomics smoke tests (mypy --strict will catch
# regressions), and partly behavioural confirmations that loosening the
# annotations didn't change the runtime contract.
# ---------------------------------------------------------------------------


def test_chained_subscripts_typecheck_and_work() -> None:
    doc = tomlrt.loads(
        """
        [tool.poetry]
        name = "demo"
        """,
    )
    # Chained subscripts now type-check (return Any, not the strict union).
    name: str = doc["tool"]["poetry"]["name"]
    assert name == "demo"


def test_table_is_mutablemapping_str_any() -> None:
    doc = tomlrt.loads(
        td("""
        a = 1
        [t]
        b = 2
        """)
    )
    # Consumers typed against MutableMapping[str, Any] (which is most of
    # the ecosystem) now compose with Table without a cast.
    sink: MutableMapping[str, Any] = doc
    assert sink["a"] == 1
    sink["c"] = "hello"
    assert doc["c"] == "hello"


def test_array_is_list_any() -> None:
    doc = tomlrt.loads('xs = [1, "two", { k = "v" }]\n')
    arr = doc.array("xs")
    # An Array is a list (subclass), parameterised as list[Any].
    sink: list[Any] = arr
    assert sink[0] == 1
    assert sink[2]["k"] == "v"


def test_table_getitem_returns_any_pop_too() -> None:
    doc = tomlrt.loads("[t]\nname = 'x'\n")
    # Static type of the popped value is Any; runtime is a plain dict
    # snapshot (per Table.pop semantics).
    popped = doc.pop("t")
    assert popped == {"name": "x"}
    assert "t" not in doc


def test_non_string_keys_rejected() -> None:
    """``doc[k] = v`` and friends must reject keys that can't round-trip
    through TOML. TOML keys are strings; anything else (``None``, ``42``,
    ``True``, ``0.5``) should fail loudly rather than silently coerce to
    an empty ``""`` key and lie about the stored state.
    """
    for bad in (None, 42, 3.14, True, False, (1,), b"bytes"):
        doc = tomlrt.loads("")
        with pytest.raises(TypeError):
            doc[bad] = 1  # type: ignore[index]  # ty: ignore[invalid-assignment]

    # Empty string key IS valid TOML (``"" = 1``) and must still work.
    doc = tomlrt.loads("")
    doc[""] = 1
    assert tomlrt.dumps(doc) == '"" = 1\n'
    assert tomlrt.loads(tomlrt.dumps(doc))[""] == 1

    # install() should reject non-string segments too.
    doc = tomlrt.loads("")
    with pytest.raises((TypeError, tomlrt.TOMLError)):
        doc.install((None,), 1)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]


def test_setitem_tuple_key_rejected_for_section_value() -> None:
    for src in ("", "[a]\nx = 1\n", 'owner = { name = "tom" }\n'):
        doc = tomlrt.loads(src)
        with pytest.raises(TypeError, match="TOML keys must be str"):
            doc["a", "b"] = tomlrt.Table.section({"v": 1})  # type: ignore[index]  # ty: ignore[invalid-assignment]


# ---------------------------------------------------------------------------
# Header-less parent: adding direct keys synthesises a parent header
# ---------------------------------------------------------------------------


def test_set_direct_key_on_headerless_parent_no_leading_blank() -> None:
    src = td("""
        [fruit.apple]
        x = 1

        [fruit.banana]
        y = 2
        """)
    doc = tomlrt.loads(src)
    doc.table("fruit")["count"] = 5
    out = tomlrt.dumps(doc)
    assert not out.startswith("\n"), repr(out)
    assert out == td("""
        [fruit]
        count = 5

        [fruit.apple]
        x = 1

        [fruit.banana]
        y = 2
        """)
    assert _reparses(out) == {
        "fruit": {"count": 5, "apple": {"x": 1}, "banana": {"y": 2}},
    }


def test_set_direct_key_on_headerless_parent_preserves_compact_style() -> None:
    """When the existing document packs adjacent headers with no blank
    lines between them, the synthesised parent header must follow the
    same convention rather than imposing a blank line before the
    descendant header that follows it.

    The source is also genuinely *out-of-order* — ``[fruit.apple]`` and
    ``[fruit.banana]`` are split by an unrelated ``[other]`` block —
    to exercise the bug-prone case in tomlkit-style terminology.
    """
    src = td("""
        [meta]
        version = 1
        [fruit.apple]
        x = 1
        [other]
        z = 3
        [fruit.banana]
        y = 2
        """)
    doc = tomlrt.loads(src)
    doc.table("fruit")["count"] = 5
    out = tomlrt.dumps(doc)
    assert out == td("""
        [meta]
        version = 1
        [fruit]
        count = 5
        [fruit.apple]
        x = 1
        [other]
        z = 3
        [fruit.banana]
        y = 2
        """)


def test_install_table_into_compact_style_doc_stays_compact() -> None:
    """Installing a section into a doc whose existing headers are
    packed flush (no blank lines between them) must not inject a blank
    line before the new header, which would mix styles.
    """
    src = td("""
        [a]
        x = 1
        [c]
        z = 3
        """)
    doc = tomlrt.loads(src)
    src_b = td("""
        [b]
        y = 2
        """)
    doc["b"] = tomlrt.loads(src_b)["b"]
    out = tomlrt.dumps(doc)
    assert out == td("""
        [a]
        x = 1
        [c]
        z = 3
        [b]
        y = 2
        """)


def test_install_table_into_blank_line_doc_keeps_blank_line() -> None:
    """The companion to the compact-style test: when the existing doc
    separates headers with blank lines, an installed section should
    follow suit, preserving canonical TOML readability.
    """
    src = td("""
        [a]
        x = 1

        [c]
        z = 3
        """)
    doc = tomlrt.loads(src)
    src_b = td("""
        [b]
        y = 2
        """)
    doc["b"] = tomlrt.loads(src_b)["b"]
    out = tomlrt.dumps(doc)
    assert out == td("""
        [a]
        x = 1

        [c]
        z = 3

        [b]
        y = 2
        """)


def test_aot_replace_in_compact_doc_preserves_compact_style() -> None:
    """Replacing an AoT entry in a compact doc must not inject a blank."""
    src = td("""
        [[xs]]
        a=1
        [[xs]]
        c=3
        """)
    doc = tomlrt.loads(src)
    doc["xs"][0] = {"k": 9}
    assert tomlrt.dumps(doc) == td("""
        [[xs]]
        k = 9
        [[xs]]
        c=3
        """)


def test_aot_replace_in_compact_ooo_doc_preserves_compact_style() -> None:
    """Per-entry replace is targeted: it touches `xs[0]`'s body and
    nothing else. Interleaved doc layout is preserved verbatim around
    the replacement — no side-effect renormalisation of the AoT.
    """
    src = td("""
        [[xs]]
        a=1
        [other]
        b=2
        [[xs]]
        c=3
        """)
    doc = tomlrt.loads(src)
    doc["xs"][0] = {"k": 9}
    assert tomlrt.dumps(doc) == td("""
        [[xs]]
        k = 9
        [other]
        b=2
        [[xs]]
        c=3
        """)


def test_aot_replace_in_blank_styled_doc_keeps_blank() -> None:
    """Control: blank-line style is preserved across replace."""
    src = td("""
        [[xs]]
        a=1

        [[xs]]
        c=3
        """)
    doc = tomlrt.loads(src)
    doc["xs"][0] = {"k": 9}
    assert tomlrt.dumps(doc) == td("""
        [[xs]]
        k = 9

        [[xs]]
        c=3
        """)


def test_aot_append_adopts_sibling_kv_indent() -> None:
    """A new AoT entry should match the sibling entries' KV indent."""
    src = td("""
        [[xs]]
            a = 1
        [[xs]]
            b = 2
        """)
    doc = tomlrt.loads(src)
    doc["xs"].append({"c": 3})
    assert tomlrt.dumps(doc) == td("""
        [[xs]]
            a = 1
        [[xs]]
            b = 2
        [[xs]]
            c = 3
        """)


def test_aot_insert_at_zero_adopts_sibling_kv_indent() -> None:
    """Insert at index 0 also adopts the sibling KV indent."""
    src = td("""
        [[xs]]
            a = 1
        """)
    doc = tomlrt.loads(src)
    doc["xs"].insert(0, {"c": 3})
    assert tomlrt.dumps(doc) == td("""
        [[xs]]
            c = 3

        [[xs]]
            a = 1
        """)


def test_aot_append_with_no_sibling_indent_stays_flush() -> None:
    """Control: no sibling indent signal means no indent is invented."""
    src = td("""
        [[xs]]
        a = 1
        """)
    doc = tomlrt.loads(src)
    doc["xs"].append({"b": 2})
    assert tomlrt.dumps(doc) == td("""
        [[xs]]
        a = 1

        [[xs]]
        b = 2
        """)


def test_dotted_add_when_host_lacks_trailing_newline() -> None:
    """Adding a dotted sibling must not glue onto the previous KV."""
    src = "[s]\na.b = 1"
    doc = tomlrt.loads(src)
    doc["s"]["a"]["c"] = 2
    out = tomlrt.dumps(doc)
    assert out == "[s]\na.b = 1\na.c = 2\n"
    assert _reparses(out)["s"]["a"] == {"b": 1, "c": 2}


def test_dotted_add_at_top_level_when_no_trailing_newline() -> None:
    src = "a.b = 1"
    doc = tomlrt.loads(src)
    doc["a"]["c"] = 2
    out = tomlrt.dumps(doc)
    assert out == "a.b = 1\na.c = 2\n"
    assert _reparses(out)["a"] == {"b": 1, "c": 2}


def test_dotted_add_inherits_section_indent() -> None:
    src = td("""
        [s]
            a.b = 1
            a.c = 2
        """)
    doc = tomlrt.loads(src)
    doc["s"]["a"]["d"] = 3
    out = tomlrt.dumps(doc)
    assert out == td("""
        [s]
            a.b = 1
            a.c = 2
            a.d = 3
        """)


def test_dotted_add_respects_blank_line_policy() -> None:
    src = td("""
        [s]
        a.b = 1

        a.c = 2
        """)
    doc = tomlrt.loads(src)
    doc["s"]["a"]["d"] = 99
    out = tomlrt.dumps(doc)
    assert out == td("""
        [s]
        a.b = 1

        a.c = 2

        a.d = 99
        """)


def test_dotted_add_anchors_at_implicit_tail_not_host_tail() -> None:
    """A new dotted sibling must group with its implicit-region peers,
    not be appended after unrelated host-level KVs that follow them in
    doc-stream order.
    """
    src = td("""
        [x]
        v.w = 1
        y = 2
        """)
    doc = tomlrt.loads(src)
    doc["x"]["v"]["z"] = 3
    out = tomlrt.dumps(doc)
    assert out == td("""
        [x]
        v.w = 1
        v.z = 3
        y = 2
        """)


def test_dotted_add_anchors_at_implicit_tail_with_unrelated_dotted_trailer() -> None:
    """Even when the trailing host slot is another dotted KV (under a
    different implicit prefix), the new sibling must anchor at its
    own implicit region's tail, not host's.
    """
    src = td("""
        [s]
        a.b = 1
        q.r = 2
        """)
    doc = tomlrt.loads(src)
    doc["s"]["a"]["c"] = 3
    out = tomlrt.dumps(doc)
    assert out == td("""
        [s]
        a.b = 1
        a.c = 3
        q.r = 2
        """)


def test_clear_doc_with_sections_drops_all_and_keeps_doc_empty() -> None:
    src = td("""
        [a]
        x = 1
        [b]
        y = 2
        [c.d]
        z = 3
        """)
    doc = tomlrt.loads(src)
    doc.clear()
    assert dict(doc) == {}
    assert tomlrt.dumps(doc) == ""
    doc["new"] = 1
    assert tomlrt.dumps(doc) == "new = 1\n"


def test_clear_doc_with_aot_children_drops_all() -> None:
    src = td("""
        [[a]]
        n = 1
        [[a]]
        n = 2
        [b]
        x = 1
        """)
    doc = tomlrt.loads(src)
    doc.clear()
    assert dict(doc) == {}
    assert tomlrt.dumps(doc) == ""


def test_clear_doc_orphans_held_section_view() -> None:
    doc = tomlrt.loads(
        td("""
        [a]
        x = 1
        [a.sub]
        y = 2
        """)
    )
    held = doc["a"]
    doc.clear()
    assert dict(doc) == {}
    assert held["x"] == 1
    assert held["sub"]["y"] == 2
    held["x"] = 99
    assert held["x"] == 99
    assert "x" not in doc


def test_clear_doc_orphans_held_aot_view() -> None:
    doc = tomlrt.loads(
        td("""
        [[a]]
        n = 1
        [[a]]
        n = 2
        """)
    )
    held = doc["a"]
    doc.clear()
    assert tomlrt.dumps(doc) == ""
    assert [dict(e) for e in held] == [{"n": 1}, {"n": 2}]


def test_clear_nested_section_keeps_anchor_and_drops_subsections() -> None:
    src = td("""
        [a]
        x = 1
        [a.sub]
        y = 2
        [b]
        z = 3
        """)
    doc = tomlrt.loads(src)
    doc["a"].clear()
    assert dict(doc["a"]) == {}
    assert dict(doc["b"]) == {"z": 3}
    out = tomlrt.dumps(doc)
    assert out == td("""
        [a]
        [b]
        z = 3
        """)


def test_clear_aot_entry_does_not_touch_siblings() -> None:
    src = td("""
        [[items]]
        a = 1
        [items.sub]
        x = 1
        [[items]]
        a = 2
        [items.sub]
        x = 2
        """)
    doc = tomlrt.loads(src)
    doc["items"][0].clear()
    assert dict(doc["items"][0]) == {}
    assert dict(doc["items"][1]) == {"a": 2, "sub": {"x": 2}}


def test_clear_inline_table_empties_and_round_trips() -> None:
    src = "obj = { a = 1, b = 2, c = 3 }\n"
    doc = tomlrt.loads(src)
    obj = doc.table("obj")
    obj.clear()
    assert dict(obj) == {}
    out = tomlrt.dumps(doc)
    assert _reparses(out) == {"obj": {}}


def test_clear_inline_table_orphans_held_array() -> None:
    src = "obj = { xs = [1, 2, 3] }\n"
    doc = tomlrt.loads(src)
    obj = doc.table("obj")
    held = obj["xs"]
    assert isinstance(held, Array)
    obj.clear()
    assert dict(obj) == {}
    held.append(4)
    assert list(held) == [1, 2, 3, 4]


def test_clear_dotted_subtable_drops_only_its_subtree() -> None:
    src = td("""
        [s]
        a.b.x = 1
        a.b.y = 2
        a.c = 3
        d = 4
        """)
    doc = tomlrt.loads(src)
    sub = doc["s"]["a"]["b"]
    sub.clear()
    assert dict(sub) == {}
    assert doc["s"]["a"]["c"] == 3
    assert "x" not in sub
    assert doc["s"]["d"] == 4
    out = tomlrt.dumps(doc)
    assert _reparses(out) == {"s": {"a": {"c": 3}, "d": 4}}


def test_clear_empty_table_is_noop() -> None:
    doc = tomlrt.loads("")
    doc.clear()
    assert tomlrt.dumps(doc) == ""


def test_clear_doc_with_top_level_array_detaches_it() -> None:
    src = "xs = [1, 2, 3]\n[a]\nx = 1\n"
    doc = tomlrt.loads(src)
    held = doc["xs"]
    assert isinstance(held, Array)
    doc.clear()
    assert tomlrt.dumps(doc) == ""
    held.append(4)
    assert list(held) == [1, 2, 3, 4]


def test_clear_empty_inline_and_dotted_subtable_are_noops() -> None:
    doc = tomlrt.loads("obj = {}\n")
    doc.table("obj").clear()
    assert tomlrt.dumps(doc) == "obj = {}\n"

    doc2 = tomlrt.loads("[s]\na.b = 1\n")
    sub = doc2["s"]["a"]
    del sub["b"]
    sub.clear()
    assert dict(sub) == {}


def test_del_subtable_with_subsections_drops_all() -> None:
    src = td("""
        [a]
        x = 1
        [a.sub]
        y = 2
        [a.sub.deeper]
        z = 3
        [b]
        w = 4
        """)
    doc = tomlrt.loads(src)
    held = doc["a"]
    del doc["a"]
    assert "a" not in doc
    assert dict(doc["b"]) == {"w": 4}
    assert held["x"] == 1
    assert held["sub"]["y"] == 2
    assert held["sub"]["deeper"]["z"] == 3


def test_del_aot_drops_all_entries_and_orphans_held_view() -> None:
    src = td("""
        [[a]]
        n = 1
        [a.sub]
        x = 1
        [[a]]
        n = 2
        [b]
        z = 3
        """)
    doc = tomlrt.loads(src)
    held = doc["a"]
    del doc["a"]
    assert "a" not in doc
    assert dict(doc["b"]) == {"z": 3}
    assert [dict(e) for e in held] == [{"n": 1, "sub": {"x": 1}}, {"n": 2}]


def test_del_top_level_array_orphans_held_reference() -> None:
    src = "xs = [1, 2, 3]\n[a]\ny = 1\n"
    doc = tomlrt.loads(src)
    held = doc["xs"]
    assert isinstance(held, Array)
    del doc["xs"]
    assert "xs" not in doc
    held.append(4)
    assert list(held) == [1, 2, 3, 4]
    assert tomlrt.dumps(doc) == "[a]\ny = 1\n"


def test_del_loop_leaves_doc_empty() -> None:
    src = "".join(f"[s{i}]\nx = {i}\n" for i in range(20))
    doc = tomlrt.loads(src)
    for k in list(doc):
        del doc[k]
    assert dict(doc) == {}
    assert tomlrt.dumps(doc) == ""


def test_inline_append_does_not_steal_eol_comment_in_multiline() -> None:
    # TOML 1.1 multiline inline table with an inline comment on the
    # last entry's line. The comment must stay attached to `a`, not
    # migrate to the appended entry.
    src = "obj = { a = 1 # eol-on-a\n  }\n"
    doc = tomlrt.loads(src)
    obj = doc.table("obj")
    obj["b"] = 2
    out = tomlrt.dumps(doc)
    assert out.index("# eol-on-a") < out.index("b = 2")
    assert tomlrt.loads(out).table("obj").to_dict() == {"a": 1, "b": 2}


def test_inline_append_keeps_eol_after_comma_not_orphans_comma() -> None:
    # When appending to a last entry that has an EOL comment but no
    # trailing comma, the comma must land immediately after the value
    # (with the comment moving to after the comma). Earlier code left
    # the comment in place and put the comma on its own line, yielding
    # the orphan-comma layout ``a = 1 # eol-on-a\n,\n  b = 2\n  }``.
    src = "obj = { a = 1 # eol-on-a\n  }\n"
    doc = tomlrt.loads(src)
    doc.table("obj")["b"] = 2
    assert tomlrt.dumps(doc) == "obj = { a = 1, # eol-on-a\n    b = 2\n  }\n"
    assert tomlrt.loads(tomlrt.dumps(doc)).table("obj").to_dict() == {"a": 1, "b": 2}


def test_inline_append_migrates_above_bracket_comment_to_new_entry() -> None:
    # An above-`}` comment block conceptually belongs to the item
    # below it. When appending a new entry, the block must migrate
    # off the bracket pad and onto the new entry's leading — mirror
    # of the existing inline-array behaviour. Earlier code left the
    # comment after the new entry, which read as if it belonged to
    # the bracket, not to either entry.
    src = td("""
        obj = {
            a = 1,
            # comment above the closing brace
        }
    """)
    doc = tomlrt.loads(src)
    doc.table("obj")["b"] = 2
    assert tomlrt.dumps(doc) == td("""
        obj = {
            a = 1,
            # comment above the closing brace
            b = 2,
        }
    """)


def test_inline_delete_head_migrates_above_comment_to_new_head() -> None:
    # When deleting the head entry, any above-item comment block that
    # belonged to the next entry (now the new head) must migrate from
    # that entry's leading into header_trivia — mirror of the existing
    # inline-array behaviour. Earlier code dropped the comment entirely.
    src = td("""
        obj = {
            a = 1,
            # comment above b
            b = 2,
            c = 3,
        }
    """)
    doc = tomlrt.loads(src)
    del doc.table("obj")["a"]
    assert tomlrt.dumps(doc) == td("""
        obj = {
            # comment above b
            b = 2,
            c = 3,
        }
    """)


def test_inline_delete_dotted_prefix_removes_all_subentries() -> None:
    doc = tomlrt.loads("obj = { a.b = 1, a.c = 2, d = 3 }\n")
    obj = doc.table("obj")
    del obj["a"]
    assert "a" not in obj
    assert obj["d"] == 3
    assert tomlrt.loads(tomlrt.dumps(doc)).table("obj").to_dict() == {"d": 3}


def test_inline_delete_dotted_leaf_cleans_empty_prefix_container() -> None:
    doc = tomlrt.loads("obj = { a.b = 1 }\n")
    obj = doc.table("obj")
    inner_a = obj.table("a")
    del inner_a["b"]
    # Synthetic prefix container `a` is now empty and has no entry in
    # the backing InlineTableValue; outer dict view should drop it.
    assert "a" not in obj
    assert tomlrt.loads(tomlrt.dumps(doc)).table("obj").to_dict() == {}


def test_insert_into_comment_only_doc_migrates_preamble() -> None:
    # Slotless doc with preamble trivia: inserting migrates the
    # comment block onto the new slot's leading so it stays visually
    # at the top of the file.
    doc = tomlrt.loads("# preamble\n")
    doc["a"] = 1
    assert tomlrt.dumps(doc) == "# preamble\n\na = 1\n"
    assert tomlrt.loads(tomlrt.dumps(doc)).preamble == ("preamble",)


def test_aot_entry_body_insert_now_works() -> None:
    doc = tomlrt.loads("[[arr]]\nx = 1\n")
    doc.aot("arr")[0]["y"] = 2
    assert tomlrt.dumps(doc) == "[[arr]]\nx = 1\ny = 2\n"


def test_delete_only_kv_in_section_then_reinsert() -> None:
    # Body emptied then refilled: the section header survives and
    # the new KV lands inside it.
    src = td("""
        [s]
        only = 1
    """)
    doc = tomlrt.loads(src)
    del doc.table("s")["only"]
    assert tomlrt.dumps(doc) == "[s]\n"
    doc.table("s")["fresh"] = 99
    assert tomlrt.dumps(doc) == td("""
        [s]
        fresh = 99
    """)


def test_insert_into_implicit_table() -> None:
    # `a` exists implicitly via a dotted top-level key; dotted-KV
    # insert under an implicit container.
    doc = tomlrt.loads("a.x = 1\n")
    doc.table("a")["y"] = 2
    assert tomlrt.dumps(doc) == "a.x = 1\na.y = 2\n"


def test_insert_into_implicit_grandparent() -> None:
    doc = tomlrt.loads("a.b.c = 1\n")
    doc.table("a").table("b")["d"] = 2
    assert tomlrt.dumps(doc) == "a.b.c = 1\na.b.d = 2\n"


def test_delete_only_subsection_keeps_implicit_parent() -> None:
    # `a` is implicit (no [a] header). Deleting its only [a.b]
    # leaves `a` reachable as an empty implicit table — Python-dict
    # semantics: `del` removes only the named key.
    src = td("""
        [a.b]
        x = 1
    """)
    doc = tomlrt.loads(src)
    del doc.table("a")["b"]
    assert tomlrt.dumps(doc) == ""
    assert "a" in doc
    assert dict(doc.table("a")) == {}


def test_delete_one_of_two_implicit_subsections_keeps_parent() -> None:
    src = td("""
        [a.b]
        x = 1
        [a.c]
        y = 2
    """)
    doc = tomlrt.loads(src)
    del doc.table("a")["b"]
    assert tomlrt.dumps(doc) == td("""
        [a.c]
        y = 2
    """)


def test_delete_deep_implicit_inside_aot_keeps_implicit_chain() -> None:
    # Implicit ancestors (`foo`, `bar`) inside an AoT entry stay as
    # empty `Table` views — they have no rendering presence.
    doc = tomlrt.loads("[[arr]]\nfoo.bar.baz = 1\n")
    del doc.aot("arr")[0].table("foo")["bar"]
    assert tomlrt.dumps(doc) == "[[arr]]\n"
    assert doc.to_dict() == {"arr": [{"foo": {}}]}


def test_delete_deep_non_aot_implicit_keeps_chain() -> None:
    # `[a.b.c.d]\nx=1` → a, b, c are all implicit. Deleting `d`
    # removes only `d`; the implicit chain `a.b.c` survives as
    # nested empty `Table` views with no rendering presence.
    doc = tomlrt.loads("[a.b.c.d]\nx = 1\n")
    del doc.table("a").table("b").table("c")["d"]
    assert tomlrt.dumps(doc) == ""
    assert "a" in doc
    assert dict(doc.table("a").table("b").table("c")) == {}


def test_delete_header_only_section() -> None:
    doc = tomlrt.loads("[s]\n")
    del doc["s"]
    assert tomlrt.dumps(doc) == ""


def test_readd_into_emptied_aot_implicit_anchors_inside_entry() -> None:
    # After deleting the only descendant of an AoT-owned implicit
    # chain, re-adding under that chain must synthesise the
    # [arr.foo] header inside the owning entry's slot region.
    doc = tomlrt.loads("[[arr]]\nfoo.bar.baz = 1\n\n[[arr]]\nname = 2\n")
    foo = doc.aot("arr")[0].table("foo")
    del foo["bar"]
    foo["new"] = 1
    out = tomlrt.dumps(doc)
    assert out == td("""
        [[arr]]
        [arr.foo]
        new = 1

        [[arr]]
        name = 2
        """)
    assert (
        tomlrt.loads(out).to_dict()
        == doc.to_dict()
        == {"arr": [{"foo": {"new": 1}}, {"name": 2}]}
    )


def test_delete_then_reinsert_at_top_level_after_section_delete() -> None:
    src = td("""
        x = 1
        [a]
        inner = 2
    """)
    doc = tomlrt.loads(src)
    del doc["a"]
    doc["y"] = 3
    assert tomlrt.dumps(doc) == "x = 1\ny = 3\n"


def test_insert_two_top_level_kvs_into_section_only_doc() -> None:
    doc = tomlrt.loads("[s]\nx = 1\n")
    doc["a"] = 1
    doc["b"] = 2
    assert tomlrt.dumps(doc) == "a = 1\nb = 2\n\n[s]\nx = 1\n"


def test_insert_top_level_kv_crlf_doc() -> None:
    doc = tomlrt.loads("[s]\r\nx = 1\r\n")
    doc["new"] = 1
    assert tomlrt.dumps(doc) == "new = 1\r\n\r\n[s]\r\nx = 1\r\n"


def test_delete_inserted_top_level_kv_round_trips() -> None:
    doc = tomlrt.loads("[s]\nx = 1\n")
    doc["new"] = 1
    del doc["new"]
    # Blank-line residue is acceptable; reparse is what matters.
    assert tomlrt.loads(tomlrt.dumps(doc)).to_dict() == {"s": {"x": 1}}


def test_structural_only_implicit_promotes_to_section() -> None:
    # `a` exists only via the descendant header [a.b]; assigning a
    # KV under `a` promotes it to an explicit `[a]` section before
    # the descendant rather than emitting a top-level dotted KV.
    doc = tomlrt.loads("[a.b]\ny = 1\n")
    doc.table("a")["x"] = 2
    out = tomlrt.dumps(doc)
    assert out == "[a]\nx = 2\n\n[a.b]\ny = 1\n"
    re_parsed = tomlrt.loads(out)
    assert re_parsed.table("a")["x"] == 2
    assert re_parsed.table("a").table("b")["y"] == 1


def test_multiple_aot_entries_independent() -> None:
    src = "[[arr]]\na.x = 1\n\n[[arr]]\na.x = 2\n"
    doc = tomlrt.loads(src)
    doc.aot("arr")[0].table("a")["y"] = 9
    assert tomlrt.dumps(doc) == "[[arr]]\na.x = 1\na.y = 9\n\n[[arr]]\na.x = 2\n"


def test_insert_before_later_child_section() -> None:
    doc = tomlrt.loads("a.x = 1\n\n[a.b]\ny = 2\n")
    doc.table("a")["z"] = 3
    assert tomlrt.dumps(doc) == "a.x = 1\na.z = 3\n\n[a.b]\ny = 2\n"


# ---------------------------------------------------------------------------
# Coverage gaps: unattached AoT mutators
# ---------------------------------------------------------------------------


def test_unattached_aot_setitem_int() -> None:
    aot = AoT([{"a": 1}, {"a": 2}])
    aot[0] = {"a": 99}
    assert aot[0]["a"] == 99
    assert aot[1]["a"] == 2


def test_unattached_aot_setitem_slice() -> None:
    aot = AoT([{"a": 1}, {"a": 2}, {"a": 3}])
    aot[1:3] = [{"a": 20}, {"a": 30}]
    assert [t["a"] for t in aot] == [1, 20, 30]


def test_unattached_aot_setitem_slice_grow() -> None:
    aot = AoT([{"a": 1}])
    aot[1:1] = [{"a": 2}, {"a": 3}]
    assert [t["a"] for t in aot] == [1, 2, 3]


def test_unattached_aot_setitem_non_iterable_raises() -> None:
    aot = AoT([{"a": 1}])
    with pytest.raises(TypeError, match="iterable"):
        aot[0:1] = 5  # type: ignore[call-overload]  # ty: ignore[invalid-assignment]


def test_unattached_aot_append_via_list_api() -> None:
    aot = AoT()
    aot.append({"a": 1})
    aot.append({"a": 2})
    assert [t["a"] for t in aot] == [1, 2]


def test_unattached_aot_insert() -> None:
    aot = AoT([{"a": 1}, {"a": 3}])
    aot.insert(1, {"a": 2})
    assert [t["a"] for t in aot] == [1, 2, 3]


def test_unattached_aot_pop() -> None:
    aot = AoT([{"a": 1}, {"a": 2}])
    popped = aot.pop()
    assert popped["a"] == 2
    assert len(aot) == 1


def test_unattached_aot_delitem() -> None:
    aot = AoT([{"a": 1}, {"a": 2}, {"a": 3}])
    del aot[1]
    assert [t["a"] for t in aot] == [1, 3]


def test_unattached_aot_delitem_slice() -> None:
    aot = AoT([{"a": 1}, {"a": 2}, {"a": 3}])
    del aot[0:2]
    assert [t["a"] for t in aot] == [3]


def test_unattached_aot_clear() -> None:
    aot = AoT([{"a": 1}, {"a": 2}])
    aot.clear()
    assert len(aot) == 0


def test_unattached_aot_reverse() -> None:
    aot = AoT([{"a": 1}, {"a": 2}, {"a": 3}])
    aot.reverse()
    assert [t["a"] for t in aot] == [3, 2, 1]


def test_unattached_aot_sort() -> None:
    aot = AoT([{"a": 3}, {"a": 1}, {"a": 2}])
    aot.sort(key=lambda t: t["a"])
    assert [t["a"] for t in aot] == [1, 2, 3]


def test_unattached_aot_then_attach_preserves_contents() -> None:
    aot = AoT()
    aot.append({"name": "a"})
    aot.insert(0, {"name": "z"})
    aot[1] = {"name": "b"}
    doc = tomlrt.loads("")
    doc["pkg"] = aot
    assert _reparses(tomlrt.dumps(doc)) == {"pkg": [{"name": "z"}, {"name": "b"}]}


# ---------------------------------------------------------------------------
# Coverage gaps: ensure_table edge cases
# ---------------------------------------------------------------------------


def test_ensure_table_on_inline_view_raises() -> None:
    doc = tomlrt.loads("t = {a = 1}\n")
    inline = doc.table("t")
    with pytest.raises(tomlrt.TOMLError, match="inline"):
        inline.ensure_table("sub")


def test_ensure_table_through_aot_raises() -> None:
    doc = tomlrt.loads("[[arr]]\nx = 1\n")
    with pytest.raises(tomlrt.TOMLError, match="array-of-tables"):
        doc.ensure_table(["arr", "sub"])


def test_ensure_table_through_inline_value_raises() -> None:
    doc = tomlrt.loads("t = {a = 1}\n")
    with pytest.raises(tomlrt.TOMLError, match="inline table or non-table"):
        doc.ensure_table(["t", "sub"])


def test_ensure_table_on_detached_table_section() -> None:
    t = Table.section()
    sub = t.ensure_table(["a", "b", "c"])
    sub["x"] = 1
    assert dict(t["a"]["b"]["c"]) == {"x": 1}


def test_ensure_table_on_inline_leaf_returns_existing() -> None:
    """An existing inline-flavoured table at the leaf is a valid Table
    and is returned as-is, matching pre-rewrite behaviour. Only
    descending through an inline (or creating a new section under
    one) is spec-impossible."""
    doc = tomlrt.loads("")
    doc["foo"] = {"a": 1}
    t = doc.ensure_table("foo")
    assert t is doc.table("foo")
    assert dict(t) == {"a": 1}
    # Flavour preserved: ensure_table did not promote the inline
    # to a section, and the dump still emits inline syntax.
    assert tomlrt.dumps(doc) == "foo = { a = 1 }\n"


def test_ensure_table_on_inline_leaf_via_inline_self() -> None:
    """Same leaf-return behaviour when the call is dispatched through
    an inline self: the existing inline child is returned without
    requiring promote_inline first."""
    src = "t = { sub = { x = 1 } }\n"
    doc = tomlrt.loads(src)
    inner = doc.table("t").ensure_table("sub")
    assert dict(inner) == {"x": 1}
    assert tomlrt.dumps(doc) == src


# ---------------------------------------------------------------------------
# Coverage gaps: AoT clone-with-dotted-key + nested AoT cleanup
# ---------------------------------------------------------------------------


def test_aot_entry_with_dotted_key_clones() -> None:
    src = td(
        """
        [[arr]]
        a.b = 1
        a.c = 2
        """,
    )
    doc = tomlrt.loads(src)
    src_entry = doc.aot("arr")[0]
    # Re-attach into a new AoT key — exercises clone path with dotted KVs.
    doc["dst"] = AoT()
    doc.aot("dst").append(src_entry)
    assert _reparses(tomlrt.dumps(doc)) == {
        "arr": [{"a": {"b": 1, "c": 2}}],
        "dst": [{"a": {"b": 1, "c": 2}}],
    }


def test_delete_aot_entry_with_nested_aot() -> None:
    src = td(
        """
        [[outer]]
        x = 1

        [[outer.inner]]
        y = 10

        [[outer.inner]]
        y = 20

        [[outer]]
        x = 2
        """,
    )
    doc = tomlrt.loads(src)
    del doc.aot("outer")[0]
    out = tomlrt.dumps(doc)
    assert _reparses(out) == {"outer": [{"x": 2}]}
    assert out == td("""
        [[outer]]
        x = 2
        """)


# ---------------------------------------------------------------------------
# Coverage gaps: standalone Array multiline + comment inheritance
# ---------------------------------------------------------------------------


def test_standalone_array_multiline_property() -> None:
    arr_single = Array([1, 2])
    assert arr_single.multiline is False
    arr_multi = Array([1, 2], multiline=True)
    assert arr_multi.multiline is True


def test_standalone_array_set_multiline_then_attach() -> None:
    arr = Array([1, 2])
    arr.set_multiline(multiline=True, indent="  ")
    doc = tomlrt.loads("")
    doc["xs"] = arr
    out = tomlrt.dumps(doc)
    assert out == td("""
        xs = [
          1,
          2,
        ]
        """)
    assert _reparses(out) == {"xs": [1, 2]}


def test_set_multiline_true_preserves_eol_comment_when_synthesising_comma() -> None:
    # Regression: a last element without a trailing comma carries its
    # EOL comment in ``trailing``; set_multiline synthesises a comma
    # and used to clobber that channel, dropping the comment.
    src = td("""
        alist = [
          'w' # Comment
        ]
        """)
    doc = tomlrt.loads(src)
    doc.array("alist").set_multiline(multiline=True, indent="  ")
    assert tomlrt.dumps(doc) == td("""
        alist = [
          'w', # Comment
        ]
        """)


def test_set_multiline_true_no_blank_line_before_bracket_with_eol_comment() -> None:
    # Companion regression: when the last item ends with an EOL
    # comment (whether already-comma'd in source or not), the
    # bracket-pad must not add its own newline — that would insert a
    # spurious blank line between the comment and ``]``.
    src = td("""
        alist = [
          'w', # Comment
        ]
        """)
    doc = tomlrt.loads(src)
    doc.array("alist").set_multiline(multiline=True, indent="  ")
    assert tomlrt.dumps(doc) == td("""
        alist = [
          'w', # Comment
        ]
        """)


def test_set_multiline_true_preserves_embedded_comments() -> None:
    src = td("""
        alist = [
        # Orphan comment
        # Multiline
          'a',
          'g',
        # Comment attached
          'w',
        ]
        """)
    doc = tomlrt.loads(src)
    arr = doc.array("alist")
    arr.set_multiline(multiline=True, indent="    ")
    assert tomlrt.dumps(doc) == td("""
        alist = [
            # Orphan comment
            # Multiline
            'a',
            'g',
            # Comment attached
            'w',
        ]
        """)


def test_multiline_setter_is_noop_when_already_multiline() -> None:
    # Assigning the same value to the property must not silently reflow
    # the array (e.g., re-indent from 2-space to the default 4-space).
    src = td("""
        xs = [
          1,
          2,
        ]
        """)
    doc = tomlrt.loads(src)
    arr = doc.array("xs")
    assert arr.multiline is True
    arr.multiline = True
    assert tomlrt.dumps(doc) == src


def test_multiline_setter_is_noop_when_already_singleline() -> None:
    src = "xs = [1, 2, 3]\n"
    doc = tomlrt.loads(src)
    arr = doc.array("xs")
    assert arr.multiline is False
    arr.multiline = False
    assert tomlrt.dumps(doc) == src


def test_array_format_preserves_bracket_eol_comment() -> None:
    src = td("""
        xs = [ # opening
          1,
          2,
        ]
        """)
    doc = tomlrt.loads(src)
    doc.array("xs").format()
    assert tomlrt.dumps(doc) == td("""
        xs = [ # opening
          1,
          2,
        ]
        """)


def test_container_format_preserves_inline_table_bracket_eol_comment() -> None:
    src = td("""
        t = { # opening
          a = 1,
          b = 2,
        }
        """)
    doc = tomlrt.loads(src)
    doc.format()
    assert tomlrt.dumps(doc) == td("""
        t = { # opening
          a = 1,
          b = 2,
        }
        """)


def test_set_multiline_true_preserves_bracket_eol_comment() -> None:
    src = td("""
        xs = [ # opening
          1,
          2,
        ]
        """)
    doc = tomlrt.loads(src)
    doc.array("xs").set_multiline(multiline=True, indent="  ")
    assert tomlrt.dumps(doc) == td("""
        xs = [ # opening
          1,
          2,
        ]
        """)


def test_array_format_preserves_bracket_eol_on_empty_array() -> None:
    src = td("""
        xs = [ # opening
        ]
        """)
    doc = tomlrt.loads(src)
    doc.array("xs").format()
    assert tomlrt.dumps(doc) == td("""
        xs = [ # opening
        ]
        """)


def test_set_multiline_true_preserves_bracket_eol_on_empty_array() -> None:
    src = td("""
        xs = [ # opening
        ]
        """)
    doc = tomlrt.loads(src)
    doc.array("xs").set_multiline(multiline=True, indent="  ")
    assert tomlrt.dumps(doc) == td("""
        xs = [ # opening
        ]
        """)


def test_set_multiline_true_empty_array_aligns_close_to_outer_indent() -> None:
    src = "xs = []\n"
    doc = tomlrt.loads(src)
    doc.array("xs").set_multiline(multiline=True, indent="    ")
    assert tomlrt.dumps(doc) == td("""
        xs = [
        ]
        """)


def test_collapse_multiline_with_nested_array_comment_raises() -> None:
    src = td(
        """
        xs = [
            [1, 2, # nested-eol
            ],
            [3, 4],
        ]
        """,
    )
    doc = tomlrt.loads(src)
    arr = doc.array("xs")
    with pytest.raises(tomlrt.TOMLError):
        arr.set_multiline(multiline=False)


def test_collapse_multiline_with_nested_inline_table_comment_raises() -> None:
    src = td(
        """
        xs = [
            { a = 1, # eol-in-inline
              b = 2,
            },
            { a = 3, b = 4 },
        ]
        """,
    )
    doc = tomlrt.loads(src)
    arr = doc.array("xs")
    with pytest.raises(tomlrt.TOMLError):
        arr.set_multiline(multiline=False)


# ---------------------------------------------------------------------------
# Container.sort
# ---------------------------------------------------------------------------


def test_has_header_partitions_sections_from_leaves() -> None:
    src = td("""
        bare = 1
        dotted.x = 2

        [section]
        x = 1

        [[aot]]
        y = 1

        [[aot]]
        y = 2

        [inline_host]
        inline = { a = 1 }
        """)
    doc = tomlrt.loads(src)
    assert doc.has_header("bare") is False
    assert doc.has_header("dotted") is False
    assert doc.has_header("section") is True
    assert doc.has_header("aot") is True
    assert doc.has_header("inline_host") is True
    assert doc.has_header("missing") is False
    assert doc.table("inline_host").has_header("inline") is False
    assert doc.table("dotted").has_header("x") is False


def test_has_header_drives_custom_sort_key() -> None:
    src = td("""
        b = 1
        a = 2

        [zeta]
        x = 1

        [alpha]
        y = 1
        """)
    doc = tomlrt.loads(src)
    original = list(doc.keys())

    def keyfn(k: str) -> tuple[int, str]:
        # Sections sort by name; leaves keep their original order.
        if doc.has_header(k):
            return (1, k)
        return (0, str(original.index(k)))

    doc.sort(key=keyfn)
    assert tomlrt.dumps(doc) == td("""
        b = 1
        a = 2

        [alpha]
        y = 1

        [zeta]
        x = 1
        """)


def test_sort_leaf_kvs_preserves_leading_and_eol_comments() -> None:
    src = td("""
        # preamble

        # above b
        b = 1  # eol b
        # above a
        a = 2  # eol a

        # above c
        c = 3
        """)
    doc = tomlrt.loads(src)
    doc.sort()
    assert tomlrt.dumps(doc) == td("""
        # preamble

        # above a
        a = 2  # eol a
        # above b
        b = 1  # eol b

        # above c
        c = 3
        """)
    assert _reparses(tomlrt.dumps(doc))


def test_sort_reverse_leaf_kvs() -> None:
    src = td("""
        a = 1
        b = 2
        c = 3
        """)
    doc = tomlrt.loads(src)
    doc.sort(reverse=True)
    assert tomlrt.dumps(doc) == td("""
        c = 3
        b = 2
        a = 1
        """)


def test_sort_with_custom_key() -> None:
    src = td("""
        short = 1
        longer = 2
        longest_name = 3
        """)
    doc = tomlrt.loads(src)
    doc.sort(key=len)
    assert tomlrt.dumps(doc) == td("""
        short = 1
        longer = 2
        longest_name = 3
        """)
    doc.sort(key=len, reverse=True)
    assert tomlrt.dumps(doc) == td("""
        longest_name = 3
        longer = 2
        short = 1
        """)


def test_sort_sections_preserves_above_header_comments() -> None:
    src = td("""
        # preamble

        # above b
        [b]
        x = 1

        # above a
        [a]
        y = 2
        z = 3
        """)
    doc = tomlrt.loads(src)
    doc.sort()
    assert tomlrt.dumps(doc) == td("""
        # preamble

        # above a
        [a]
        y = 2
        z = 3

        # above b
        [b]
        x = 1
        """)
    assert _reparses(tomlrt.dumps(doc))


def test_sort_aot_at_root_moves_all_entries() -> None:
    src = td("""
        # header for b
        [[b]]
        x = 1
        [[b]]
        x = 2

        # header for a
        [[a]]
        y = 1
        """)
    doc = tomlrt.loads(src)
    doc.sort()
    assert tomlrt.dumps(doc) == td("""
        # header for a
        [[a]]
        y = 1

        # header for b
        [[b]]
        x = 1
        [[b]]
        x = 2
        """)
    assert _reparses(tomlrt.dumps(doc))


def test_sort_dotted_kvs_treated_as_one_block() -> None:
    src = td("""
        b = 1
        a.x = 2
        a.y = 3
        """)
    doc = tomlrt.loads(src)
    doc.sort()
    assert tomlrt.dumps(doc) == td("""
        a.x = 2
        a.y = 3
        b = 1
        """)
    assert _reparses(tomlrt.dumps(doc))


def test_sort_on_dotted_intermediate_reorders_kvs() -> None:
    # Sorting the implicit container reached through a dotted key
    # (here doc['parent']['a']) must reorder the underlying dotted
    # KV slots, not just the dict storage.
    src = td("""
        [parent]
        a.z = 1
        a.x = 2
        a.y = 3
        """)
    doc = tomlrt.loads(src)
    doc.table(("parent", "a")).sort()
    assert tomlrt.dumps(doc) == td("""
        [parent]
        a.x = 2
        a.y = 3
        a.z = 1
        """)
    assert _reparses(tomlrt.dumps(doc))


def test_sort_on_dotted_intermediate_at_root() -> None:
    src = td("""
        a.z = 1
        a.x = 2
        a.y = 3
        """)
    doc = tomlrt.loads(src)
    doc.table("a").sort()
    assert tomlrt.dumps(doc) == td("""
        a.x = 2
        a.y = 3
        a.z = 1
        """)
    assert _reparses(tomlrt.dumps(doc))


def test_sort_inside_section() -> None:
    src = td("""
        [s]
        c = 3
        # above a
        a = 1
        b = 2
        """)
    doc = tomlrt.loads(src)
    doc.table("s").sort()
    assert tomlrt.dumps(doc) == td("""
        [s]
        # above a
        a = 1
        b = 2
        c = 3
        """)


def test_sort_default_partitions_sections_last() -> None:
    # Default sort (no key) is TOML-aware: leaf keys come before
    # structural section/AoT keys regardless of name ordering.
    doc = tomlrt.loads(
        td("""
        [a]
        x = 1
        [b]
        y = 2
        """)
    )
    doc["leaf"] = "z"
    doc.sort()
    assert tomlrt.dumps(doc) == td("""
        leaf = "z"

        [a]
        x = 1
        [b]
        y = 2
        """)
    assert _reparses(tomlrt.dumps(doc))


def test_sort_partition_overrides_user_key() -> None:
    # Even with an explicit `key=` that would interleave a leaf
    # between two sections, the partition keeps all sections last.
    doc = tomlrt.loads(
        td("""
        [a]
        x = 1
        [b]
        y = 2
        """)
    )
    doc["leaf"] = "z"
    order_idx = {"a": 0, "leaf": 1, "b": 2}
    doc.sort(key=order_idx.__getitem__)
    assert tomlrt.dumps(doc) == td("""
        leaf = "z"

        [a]
        x = 1
        [b]
        y = 2
        """)
    assert _reparses(tomlrt.dumps(doc))


def test_sort_reverse_keeps_sections_last() -> None:
    # `reverse=True` reverses order *within* each partition but does
    # not flip the leaves-then-sections structural ordering.
    doc = tomlrt.loads(
        td("""
        a_leaf = 1
        b_leaf = 2
        [a_sec]
        x = 1
        [b_sec]
        y = 2
        """)
    )
    doc.sort(reverse=True)
    assert tomlrt.dumps(doc) == td("""
        b_leaf = 2
        a_leaf = 1
        [b_sec]
        y = 2
        [a_sec]
        x = 1
        """)
    assert _reparses(tomlrt.dumps(doc))


def test_sort_smart_key_leaves_before_sections() -> None:
    doc = tomlrt.loads(
        td("""
        [a]
        x = 1
        [b]
        y = 2
        """)
    )
    doc["leaf"] = "z"

    def _is_section(k: str) -> bool:
        v = doc[k]
        if isinstance(v, AoT):
            return True
        return isinstance(v, Table) and not v._inline  # noqa: SLF001

    doc.sort(key=lambda k: (_is_section(k), k))
    assert tomlrt.dumps(doc) == td("""
        leaf = "z"

        [a]
        x = 1
        [b]
        y = 2
        """)


def test_sort_handles_non_contiguous_key_blocks() -> None:
    # Key 'outer' has two runs ([outer.a] and [outer.b]) at root,
    # with 'other' in between. sort() should collect both runs and
    # splice them as one contiguous block at 'outer's new position.
    src = td("""
        [outer.a]
        x = 1

        [other]
        z = 0

        [outer.b]
        y = 2
        """)
    doc = tomlrt.loads(src)
    doc.sort()
    assert tomlrt.dumps(doc) == td("""
        [other]
        z = 0

        [outer.a]
        x = 1

        [outer.b]
        y = 2
        """)
    assert _reparses(tomlrt.dumps(doc))


def test_sort_empty_and_single_are_noops() -> None:
    empty = tomlrt.loads("")
    empty.sort()
    assert tomlrt.dumps(empty) == ""

    single = tomlrt.loads("a = 1\n")
    single.sort()
    assert tomlrt.dumps(single) == "a = 1\n"


def test_sort_already_ordered_is_noop() -> None:
    src = td("""
        a = 1
        b = 2
        c = 3
        """)
    doc = tomlrt.loads(src)
    doc.sort()
    assert tomlrt.dumps(doc) == src


def test_sort_detached_table_reorders_dict_storage() -> None:
    t = Table()
    t["c"] = 1
    t["a"] = 2
    t["b"] = 3
    t.sort()
    assert list(t.keys()) == ["a", "b", "c"]


def test_sort_inline_table_simple() -> None:
    doc = tomlrt.loads("t = {b = 1, a = 2}\n")
    doc.table("t").sort()
    assert tomlrt.dumps(doc) == "t = {a = 2, b = 1}\n"
    assert _reparses(tomlrt.dumps(doc))


def test_sort_inline_table_preserves_no_trailing_comma() -> None:
    # Sorting must keep the original "no trailing comma" style: the
    # entry that moves to the last position drops its comma.
    doc = tomlrt.loads("t = { c = 3, a = 1, b = 2 }\n")
    doc.table("t").sort()
    assert tomlrt.dumps(doc) == "t = { a = 1, b = 2, c = 3 }\n"
    assert _reparses(tomlrt.dumps(doc))


def test_sort_inline_table_with_dotted_keys() -> None:
    # 'a' is a dotted-key prefix at the inline root: a.x and a.y are
    # two separate entries grouped under the same direct child.
    doc = tomlrt.loads("t = {b = 1, a.x = 2, a.y = 3}\n")
    doc.table("t").sort()
    # Direct children of t are 'a' and 'b' → sorted: 'a' first.
    # Both a.x and a.y belong to 'a' and travel together.
    assert tomlrt.dumps(doc) == "t = {a.x = 2, a.y = 3, b = 1}\n"
    assert _reparses(tomlrt.dumps(doc))


def test_sort_inline_table_multiline_with_above_comments() -> None:
    src = td("""
        t = {
          # above b
          b = 2,
          # above a
          a = 1,
        }
        """)
    doc = tomlrt.loads(src)
    doc.table("t").sort()
    assert tomlrt.dumps(doc) == td("""
        t = {
          # above a
          a = 1,
          # above b
          b = 2,
        }
        """)
    assert _reparses(tomlrt.dumps(doc))


def test_sort_inline_table_eol_comments_travel_with_entries() -> None:
    src = td("""
        t = {
          b = 2, # bee
          a = 1, # aye
        }
        """)
    doc = tomlrt.loads(src)
    doc.table("t").sort()
    assert tomlrt.dumps(doc) == td("""
        t = {
          a = 1, # aye
          b = 2, # bee
        }
        """)
    assert _reparses(tomlrt.dumps(doc))


def test_sort_inline_table_eol_on_last_no_comma_travels() -> None:
    # EOL comment on a comma-less last entry must travel with that
    # entry when reordered.
    src = td("""
        t = {
          b = 2, # bee
          a = 1 # aye
        }
        """)
    doc = tomlrt.loads(src)
    doc.table("t").sort()
    # 'a' moves to position 0 (and gains a comma); 'b' moves to last
    # (and loses its comma). Each entry's EOL comment travels with it.
    assert tomlrt.dumps(doc) == td("""
        t = {
          a = 1, # aye
          b = 2 # bee
        }
        """)
    assert _reparses(tomlrt.dumps(doc))


def test_sort_reverse_inline_table_eol_comments_travel() -> None:
    src = td("""
        t = {
          a = 1, # aye
          b = 2, # bee
        }
        """)
    doc = tomlrt.loads(src)
    doc.table("t").sort(reverse=True)
    assert tomlrt.dumps(doc) == td("""
        t = {
          b = 2, # bee
          a = 1, # aye
        }
        """)
    assert _reparses(tomlrt.dumps(doc))


def test_sort_inline_dotted_inner_navigator() -> None:
    # Sort the navigator view for 'a' in {a.p, a.q}; entries outside
    # 'a' (x, y) keep their absolute positions.
    doc = tomlrt.loads("t = {x = 0, a.q = 2, a.p = 1, y = 3}\n")
    doc.table("t").table("a").sort()
    # Owned positions are 1 and 2 (the a.q and a.p entries). After
    # sort by ('p','q'): pos 1 ← a.p, pos 2 ← a.q. x and y unchanged.
    assert tomlrt.dumps(doc) == "t = {x = 0, a.p = 1, a.q = 2, y = 3}\n"
    assert _reparses(tomlrt.dumps(doc))


def test_sort_preserves_lexeme_styles_and_whitespace() -> None:
    src = td("""
        b = 'lit'
        a = "basic"
        """)
    doc = tomlrt.loads(src)
    doc.sort()
    # Each KV keeps its original lexeme style; sort only reorders blocks.
    assert tomlrt.dumps(doc) == td("""
        a = "basic"
        b = 'lit'
        """)


def test_sort_aot_element_does_not_merge_sibling_entries() -> None:
    # Regression: Container.sort on an AoT element used to bucket
    # slots from sibling entries (same path) into the sorted entry,
    # merging their KVs and leaving the siblings empty.
    src = td("""
        [[hello]]
        b = 2
        a = 1

        [[hello]]
        b = 4
        a = 3
        """)
    doc = tomlrt.loads(src)
    doc.aot("hello")[0].sort()
    assert tomlrt.dumps(doc) == td("""
        [[hello]]
        a = 1
        b = 2

        [[hello]]
        b = 4
        a = 3
        """)
    assert _reparses(tomlrt.dumps(doc))


def test_sort_aot_element_with_nested_section_preserves_siblings() -> None:
    # Sorting an AoT entry with its own nested [a.sub] keeps the
    # sub-section attached to this entry and leaves the sibling
    # entry intact.
    src = td("""
        [[a]]
        x = 1

        [a.sub]
        n = 2

        [[a]]
        x = 3
        """)
    doc = tomlrt.loads(src)
    e0 = doc.aot("a")[0]

    def key(k: str) -> tuple[int, str]:
        v = dict.__getitem__(e0, k)
        return (1 if isinstance(v, (AoT, Table)) else 0, k)

    e0.sort(key=key)
    assert tomlrt.dumps(doc) == td("""
        [[a]]
        x = 1

        [a.sub]
        n = 2

        [[a]]
        x = 3
        """)
    assert _reparses(tomlrt.dumps(doc))


def test_sort_super_table_with_aot_preserves_explicit_header() -> None:
    # Regression: Container.sort on a super-table that mixes a nested
    # AoT with direct KVs used to drop the [a] header from the splice;
    # the direct KVs then re-bound to the document root.
    src = td("""
        [[a.hello]]
        x = 1

        [a]
        date = "2019"
        name = "Bob"
        """)
    doc = tomlrt.loads(src)
    # Structural-last key (mirrors toml-sort's section ordering): puts
    # leaf KVs before structural sub-sections / AoTs.
    a = doc.table("a")

    def key(k: str) -> tuple[int, str]:
        v = dict.__getitem__(a, k)
        return (1 if isinstance(v, (AoT, Table)) else 0, k)

    a.sort(key=key)
    out = tomlrt.dumps(doc)
    assert _reparses(out)
    reparsed = tomlrt.loads(out)
    assert reparsed.table("a")["date"] == "2019"
    assert reparsed.table("a")["name"] == "Bob"
    assert reparsed.table("a").aot("hello")[0]["x"] == 1


def test_sort_when_aot_child_precedes_parent_in_source() -> None:
    # Regression: when an explicit [c] header was preceded in source by
    # one of its structural children (i.e. c-header was not the doc-
    # stream-earliest owned slot of c's region), Container.sort used
    # to drag c-header's own positional leading into its new top-of-
    # region position — producing a spurious leading blank line at the
    # head of the reordered region. The region-marker model now sets
    # c-header's new leading from the region's external structural
    # prefix (the doc preamble / above-region gap) plus c-header's own
    # attached-comment remainder.
    src = td("""
        [[a-section.hello]]
        ports = [80]

        [a-section]
        date = "2019"
        """)
    doc = tomlrt.loads(src)
    doc["a-section"].sort()
    out = tomlrt.dumps(doc)
    assert out == td("""
        [a-section]
        date = "2019"
        [[a-section.hello]]
        ports = [80]
        """)
    assert _reparses(out)


def test_sort_preserves_doc_preamble_when_aot_child_precedes_parent() -> None:
    # Companion to the test above: the doc preamble lived on the AoT
    # child header (because it was the doc-stream-earliest owned
    # slot); after sort, it must end up on [a-section] (the new
    # region head), not stay attached to where the child ended up.
    src = td("""
        # preamble

        [[a-section.hello]]
        x = 1
        [a-section]
        y = 1
        """)
    doc = tomlrt.loads(src)
    a = doc["a-section"]

    def key(k: str) -> tuple[int, str]:
        v = dict.__getitem__(a, k)
        return (1 if isinstance(v, (AoT, Table)) else 0, k)

    a.sort(key=key)
    out = tomlrt.dumps(doc)
    assert out == td("""
        # preamble

        [a-section]
        y = 1
        [[a-section.hello]]
        x = 1
        """)


def test_sort_moves_disjoint_above_blank_header_comment_with_section() -> None:
    # Regression for #131: a comment block separated from its section
    # header by a blank line is part of header_leading_block (per
    # #126), so it must travel with the section under Container.sort
    # rather than staying at the physical position.
    src = td("""
        [a]
        x = 1

        [b]
        y = 2

        # c comment

        [c]
        z = 3
        """)
    doc = tomlrt.loads(src)
    doc.sort(key=lambda k: {"a": 0, "c": 1, "b": 2}[k])
    assert tomlrt.dumps(doc) == td("""
        [a]
        x = 1

        # c comment

        [c]
        z = 3

        [b]
        y = 2
        """)
    assert _reparses(tomlrt.dumps(doc))
    # header_leading_block ownership stable across the sort.
    assert doc.table("c").header_leading_block == (None, "c comment", None)
    assert doc.table("b").header_leading_block == (None,)


def test_aot_sort_moves_disjoint_above_blank_header_comment_with_entry() -> None:
    # Regression for #131 (AoT-parallel fix in renormalise_aot_order):
    # a disjoint above-blank comment block on an AoT entry header must
    # travel with that entry under AoT.sort/reverse, not stay put.
    src = td("""
        [[t]]
        x = 2

        # t3

        [[t]]
        x = 3

        [[t]]
        x = 1
        """)
    doc = tomlrt.loads(src)
    doc.aot("t").sort(key=lambda e: e["x"])
    assert tomlrt.dumps(doc) == td("""
        [[t]]
        x = 1

        [[t]]
        x = 2

        # t3

        [[t]]
        x = 3
        """)
    assert _reparses(tomlrt.dumps(doc))


def test_sort_preserves_document_epilogue() -> None:
    # Regression: reorder_container that unlinks every owned slot
    # (region starts at doc head) used to trigger the empty-doc
    # preamble-migration branch of insert_before_head, draining
    # doc._trailing into the new head's leading. The re-stitch step
    # then overwrote that leading and the epilogue silently vanished.
    src = td("""
        [b]
        y = 2

        [a]
        x = 1

        # trailing comment
        """)
    doc = tomlrt.loads(src)
    doc.sort()
    assert tomlrt.dumps(doc) == td("""
        [a]
        x = 1

        [b]
        y = 2

        # trailing comment
        """)
    assert doc.epilogue == ("trailing comment",)
    assert _reparses(tomlrt.dumps(doc))


def test_aot_sort_preserves_document_epilogue() -> None:
    # Regression: same bug in renormalise_aot_order — when the AoT
    # starts at doc head, the transient empty-doc state during splice
    # used to drain doc._trailing into the new head's leading.
    src = td("""
        [[t]]
        x = 2

        [[t]]
        x = 1

        # trailing comment
        """)
    doc = tomlrt.loads(src)
    doc.aot("t").sort(key=lambda e: e["x"])
    assert tomlrt.dumps(doc) == td("""
        [[t]]
        x = 1

        [[t]]
        x = 2

        # trailing comment
        """)
    assert doc.epilogue == ("trailing comment",)
    assert _reparses(tomlrt.dumps(doc))


def test_sort_round_trips_for_repeated_sorts() -> None:
    src = td("""
        # head

        # above c
        c = 3
        # above a
        a = 1

        # above b
        b = 2
        """)
    doc = tomlrt.loads(src)
    doc.sort()
    once = tomlrt.dumps(doc)
    doc.sort()  # idempotent: already sorted
    twice = tomlrt.dumps(doc)
    assert once == twice


def test_delete_first_kv_preserves_doc_preamble() -> None:
    src = td("""
        # preamble

        x = 1
        y = 2
    """)
    doc = tomlrt.loads(src)
    del doc["x"]
    assert tomlrt.dumps(doc) == td("""
        # preamble

        y = 2
    """)


def test_delete_first_section_preserves_doc_preamble() -> None:
    src = td("""
        # preamble

        [a]
        x = 1
        [b]
        y = 2
    """)
    doc = tomlrt.loads(src)
    del doc["a"]
    assert tomlrt.dumps(doc) == td("""
        # preamble

        [b]
        y = 2
    """)


def test_delete_first_aot_entry_preserves_doc_preamble() -> None:
    src = td("""
        # preamble

        [[a]]
        x = 1
        [[a]]
        x = 2
    """)
    doc = tomlrt.loads(src)
    del doc.aot("a")[0]
    assert tomlrt.dumps(doc) == td("""
        # preamble

        [[a]]
        x = 2
    """)


def test_pop_all_items_clears_multiline_array_indent() -> None:
    src = td("""
        arr = [
            1,
            2,
        ]
    """)
    doc = tomlrt.loads(src)
    doc.array("arr").pop()
    doc.array("arr").pop()
    assert tomlrt.dumps(doc) == "arr = [\n]\n"


def test_delete_all_entries_clears_multiline_inline_table_indent() -> None:
    src = td("""
        x = {
            a = 1,
            b = 2,
        }
    """)
    doc = tomlrt.loads(src)
    del doc.table("x")["a"]
    del doc.table("x")["b"]
    assert tomlrt.dumps(doc) == "x = {\n}\n"


def test_array_clear_clears_multiline_indent() -> None:
    src = td("""
        arr = [
            1,
            2,
        ]
    """)
    doc = tomlrt.loads(src)
    doc.array("arr").clear()
    assert tomlrt.dumps(doc) == "arr = [\n]\n"


def test_add_to_empty_multiline_inline_table_indents() -> None:
    src = "x = {\n}\n"
    doc = tomlrt.loads(src)
    doc.table("x")["z"] = 99
    assert tomlrt.dumps(doc) == "x = {\n    z = 99,\n}\n"


def test_add_after_emptying_multiline_inline_table_indents() -> None:
    src = td("""
        x = {
            a = 1,
            b = 2,
        }
    """)
    doc = tomlrt.loads(src)
    del doc.table("x")["a"]
    del doc.table("x")["b"]
    doc.table("x")["z"] = 99
    assert tomlrt.dumps(doc) == "x = {\n    z = 99,\n}\n"


def test_pop_all_array_keeps_bracket_eol_comment_without_blank_line() -> None:
    src = td("""
        arr = [ # tail
            1,
            2,
        ]
    """)
    doc = tomlrt.loads(src)
    doc.array("arr").pop()
    doc.array("arr").pop()
    assert tomlrt.dumps(doc) == td("""
        arr = [ # tail
        ]
    """)


def test_append_after_empty_array_keeps_bracket_eol_comment() -> None:
    src = td("""
        arr = [ # tail
            1,
        ]
    """)
    doc = tomlrt.loads(src)
    doc.array("arr").pop()
    doc.array("arr").append(99)
    assert tomlrt.dumps(doc) == td("""
        arr = [ # tail
            99,
        ]
    """)


def test_delete_all_inline_keeps_bracket_eol_comment_without_blank_line() -> None:
    src = td("""
        x = { # tail
            a = 1,
        }
    """)
    doc = tomlrt.loads(src)
    del doc.table("x")["a"]
    assert tomlrt.dumps(doc) == td("""
        x = { # tail
        }
    """)


def test_add_after_empty_inline_keeps_bracket_eol_comment() -> None:
    src = td("""
        x = { # tail
            a = 1,
        }
    """)
    doc = tomlrt.loads(src)
    del doc.table("x")["a"]
    doc.table("x")["z"] = 99
    assert tomlrt.dumps(doc) == td("""
        x = { # tail
            z = 99,
        }
    """)


def test_append_to_multiline_inline_table_with_eol_on_last_entry() -> None:
    src = td("""
        t = {
            a = 1,
            b = 2,
            c = 3, # last
        }
    """)
    doc = tomlrt.loads(src)
    doc.table("t")["d"] = 4
    assert tomlrt.dumps(doc) == td("""
        t = {
            a = 1,
            b = 2,
            c = 3, # last
            d = 4,
        }
    """)


def test_delete_last_entry_of_multiline_inline_table_with_eol() -> None:
    src = td("""
        t = {
            a = 1,
            b = 2,
            c = 3, # last
        }
    """)
    doc = tomlrt.loads(src)
    del doc.table("t")["c"]
    assert tomlrt.dumps(doc) == td("""
        t = {
            a = 1,
            b = 2,
        }
    """)


def test_array_del_tail_preserves_survivor_eol_comment() -> None:
    src = td("""
        arr = [
            1, # one
            2, # two
            3, # last
        ]
    """)
    doc = tomlrt.loads(src)
    del doc.array("arr")[-1]
    assert tomlrt.dumps(doc) == td("""
        arr = [
            1, # one
            2, # two
        ]
    """)


# ---------------------------------------------------------------------------
# Container.update — error contract on >1 positional arg
# ---------------------------------------------------------------------------


def test_update_rejects_multiple_positional_args() -> None:
    doc = tomlrt.loads("a = 1\n")
    with pytest.raises(TypeError, match="at most 1 argument"):
        doc.update({"b": 2}, {"c": 3})


def test_inline_setitem_non_coerceable_matches_regular_table() -> None:
    """Inline-table assignment of a non-coerceable type raises the same
    ``TypeError: Cannot convert ...`` as the regular-table path, rather
    than the previous misleading ``NotImplementedError`` about
    "live-attach of typed Container/Array/AoT"."""
    doc = tomlrt.loads("t = {a = 1}\n")
    inline = doc.table("t")
    with pytest.raises(TypeError, match="Cannot convert set"):
        inline["x"] = {1, 2, 3}


# ---------------------------------------------------------------------------
# Table.promote_array — validation error paths
# ---------------------------------------------------------------------------


def test_promote_array_missing_key_raises() -> None:
    doc = tomlrt.loads("a = 1\n")
    with pytest.raises(KeyError, match="not in table"):
        doc.promote_array("missing")


def test_promote_array_not_an_array_raises() -> None:
    doc = tomlrt.loads("a = 1\n")
    with pytest.raises(tomlrt.TOMLError, match="not an array"):
        doc.promote_array("a")


def test_promote_array_empty_raises() -> None:
    doc = tomlrt.loads("a = []\n")
    with pytest.raises(tomlrt.TOMLError, match="empty array"):
        doc.promote_array("a")


def test_promote_array_non_inline_table_element_raises() -> None:
    doc = tomlrt.loads("a = [1, 2, 3]\n")
    with pytest.raises(tomlrt.TOMLError, match="non-inline-table"):
        doc.promote_array("a")


def test_promote_array_on_inline_host_raises() -> None:
    doc = tomlrt.loads("t = {arr = [{x = 1}]}\n")
    inline = doc.table("t")
    with pytest.raises(tomlrt.TOMLError, match="inline tables"):
        inline.promote_array("arr")


def test_promote_array_with_outer_comments_raises() -> None:
    src = td("""
        a = [
          # outer
          {x = 1},
        ]
    """)
    doc = tomlrt.loads(src)
    with pytest.raises(tomlrt.TOMLError, match="comments that would be lost"):
        doc.promote_array("a")


def test_promote_array_with_entry_inner_comments_raises() -> None:
    src = td("""
        a = [{
          x = 1, # inner
        }]
    """)
    doc = tomlrt.loads(src)
    with pytest.raises(tomlrt.TOMLError, match="inner comments"):
        doc.promote_array("a")


# ---------------------------------------------------------------------------
# AoT index-error / no-op contracts (closes _layout_ops coverage)
# ---------------------------------------------------------------------------


def test_aot_pop_positive_out_of_range() -> None:
    doc = tomlrt.loads("[[a]]\nx = 1\n")
    with pytest.raises(IndexError, match="out of range"):
        doc.aot("a").pop(99)


def test_aot_pop_negative_out_of_range() -> None:
    doc = tomlrt.loads("[[a]]\nx = 1\n")
    with pytest.raises(IndexError, match="out of range"):
        doc.aot("a").pop(-5)


def test_aot_delitem_out_of_range() -> None:
    doc = tomlrt.loads("[[a]]\nx = 1\n")
    with pytest.raises(IndexError, match="out of range"):
        del doc.aot("a")[99]


def test_aot_self_assign_is_noop() -> None:
    """``aot[i] = aot[i]`` is a no-op; bytes-stable."""
    doc = tomlrt.loads(
        td("""
        [[a]]
        x = 1
        [[a]]
        y = 2
    """)
    )
    src = tomlrt.dumps(doc)
    doc.aot("a")[0] = doc.aot("a")[0]
    assert tomlrt.dumps(doc) == src


def test_aot_cross_doc_assign_negative_index() -> None:
    """Assigning a foreign AoT entry to a negative index normalises the
    index (line 2769 in replace_aot_entry_with_clone)."""
    src_doc = tomlrt.loads("[[s]]\nv = 99\n")
    dst_doc = tomlrt.loads(
        td("""
        [[a]]
        x = 1
        [[a]]
        y = 2
    """)
    )
    dst_doc.aot("a")[-1] = src_doc.aot("s")[0]
    assert tomlrt.dumps(dst_doc) == td("""
        [[a]]
        x = 1
        [[a]]
        v = 99
    """)


def test_aot_sort_singleton_short_circuits() -> None:
    """``aot.sort()`` on a single-entry AoT takes the early-return path
    and leaves the doc byte-stable."""
    doc = tomlrt.loads("[[a]]\nx = 1\n")
    src = tomlrt.dumps(doc)
    doc.aot("a").sort(key=lambda t: t.get("x", 0))
    assert tomlrt.dumps(doc) == src


def test_aot_sort_empty_short_circuits() -> None:
    """``aot.sort()`` on an empty AoT also takes the early-return path."""
    doc = tomlrt.loads("")
    doc["a"] = AoT()
    doc.aot("a").sort(key=lambda _t: 0)


def test_demote_synthetic_placeholder_transfers_preamble() -> None:
    """Demoting an empty synthetic placeholder hands its leading trivia
    off to the successor so the doc preamble survives."""
    doc = tomlrt.loads("# preamble comment\n[existing]\nx = 1\n")
    doc["tool"] = Table.section()
    doc["tool"]["sub"] = Table.section({"k": 1})
    assert tomlrt.dumps(doc) == (
        "# preamble comment\n[existing]\nx = 1\n\n\n[tool.sub]\nk = 1\n"
    )


def test_demote_synthetic_placeholder_clears_stale_body_tail() -> None:
    """After a synthetic empty placeholder is demoted, subsequent direct
    KV appends on the (now-implicit) parent must reach the doc stream.

    Regression: demote left ``parent._body_tail`` pointing at the
    demoted, doc-unlinked header. The next ``parent[k] = scalar`` then
    spliced the new KV after a detached slot, so it never appeared in
    the rendered output even though the dict held it.
    """
    doc = tomlrt.loads("")
    doc["tool"] = Table.section({"a": 1})
    del doc["tool"]["a"]
    doc["tool"]["poetry"] = tomlrt.AoT([{"x": 1}])
    doc["tool"]["q"] = 99
    assert tomlrt.dumps(doc) == td("""
        [tool]
        q = 99

        [[tool.poetry]]
        x = 1
        """)
