"""Spacing heuristics: new entries mimic the local blank-line convention."""

from __future__ import annotations

import tomlrt
from _helpers import td
from tomlrt import AoT, Table

# ---------------------------------------------------------------------------
# KV append
# ---------------------------------------------------------------------------


def test_kv_append_to_packed_section_stays_packed() -> None:
    doc = tomlrt.loads("a = 1\nb = 2\n")
    doc["c"] = 3
    assert tomlrt.dumps(doc) == td("""
        a = 1
        b = 2
        c = 3
        """)


def test_kv_append_to_uniformly_spaced_section_adds_blank() -> None:
    doc = tomlrt.loads(
        td("""
        a = 1

        b = 2
        """)
    )
    doc["c"] = 3
    assert tomlrt.dumps(doc) == td("""
        a = 1

        b = 2

        c = 3
        """)


def test_kv_append_to_mixed_layout_matches_last_gap() -> None:
    doc = tomlrt.loads(
        td("""
        a = 1
        b = 2

        c = 3
        """)
    )
    doc["d"] = 4
    assert tomlrt.dumps(doc) == td("""
        a = 1
        b = 2

        c = 3

        d = 4
        """)


def test_kv_append_to_single_entry_does_not_add_blank() -> None:
    doc = tomlrt.loads("a = 1\n")
    doc["b"] = 2
    assert tomlrt.dumps(doc) == "a = 1\nb = 2\n"


def test_kv_append_to_empty_table_does_not_add_blank() -> None:
    doc = tomlrt.loads("[t]\n")
    tbl = doc.table("t")
    tbl["a"] = 1
    assert tomlrt.dumps(doc) == "[t]\na = 1\n"


def test_kv_append_preserves_indent_and_adds_blank() -> None:
    src = td("""
        [t]
            a = 1

            b = 2
        """)
    doc = tomlrt.loads(src)
    tbl = doc.table("t")
    tbl["c"] = 3
    assert tomlrt.dumps(doc) == td("""
        [t]
            a = 1

            b = 2

            c = 3
        """)


def test_kv_append_after_dotted_tail_mirrors_its_gap() -> None:
    """The nearest preceding KV sets the convention, dotted or not."""
    src = td("""
        [t]
        a = 1

        b.c = 2
        """)
    doc = tomlrt.loads(src)
    tbl = doc.table("t")
    tbl["d"] = 3
    assert tomlrt.dumps(doc) == td("""
        [t]
        a = 1

        b.c = 2

        d = 3
        """)


def test_kv_append_after_packed_dotted_tail_stays_packed() -> None:
    src = td("""
        [t]

        a = 1
        b.c = 2
        """)
    doc = tomlrt.loads(src)
    tbl = doc.table("t")
    tbl["d"] = 3
    assert tomlrt.dumps(doc) == td("""
        [t]

        a = 1
        b.c = 2
        d = 3
        """)


def test_kv_append_after_dotted_tail_inherits_indent() -> None:
    src = td("""
        [t]
            b.c = 1
        """)
    doc = tomlrt.loads(src)
    tbl = doc.table("t")
    tbl["d"] = 2
    assert tomlrt.dumps(doc) == td("""
        [t]
            b.c = 1
            d = 2
        """)


# ---------------------------------------------------------------------------
# AoT append / insert
# ---------------------------------------------------------------------------


def test_aot_append_to_spaced_aot_adds_blank() -> None:
    src = td("""
        [[i]]
        x = 1

        [[i]]
        x = 2
        """)
    doc = tomlrt.loads(src)
    aot = doc.aot("i")
    aot.append({"x": 3})
    assert tomlrt.dumps(doc) == (
        td("""
        [[i]]
        x = 1

        [[i]]
        x = 2

        [[i]]
        x = 3
        """)
    )


def test_aot_insert_middle_uniformly_spaced_adds_blank() -> None:
    src = td("""
        [[i]]
        x = 1

        [[i]]
        x = 3
        """)
    doc = tomlrt.loads(src)
    aot = doc.aot("i")
    aot.insert(1, {"x": 2})
    assert tomlrt.dumps(doc) == (
        td("""
        [[i]]
        x = 1

        [[i]]
        x = 2

        [[i]]
        x = 3
        """)
    )


def test_aot_append_to_mixed_follows_last_entry_style() -> None:
    # Mixed-style AoT: the last entry's separator decides what we do.
    # Here the last existing separator has a blank line, so we add one.
    src = td("""
        [[i]]
        x = 1
        [[i]]
        x = 2

        [[i]]
        x = 4
        """)
    doc = tomlrt.loads(src)
    aot = doc.aot("i")
    aot.append({"x": 5})
    assert tomlrt.dumps(doc) == (
        td("""
            [[i]]
            x = 1
            [[i]]
            x = 2

            [[i]]
            x = 4

            [[i]]
            x = 5
            """)
    )


def test_aot_append_to_single_entry_adds_blank() -> None:
    """With only one prior entry the style is ambiguous, so default to
    canonical blank-separated TOML — matches what most users expect and
    what re-parsing the result then dumping again produces."""
    src = "[[i]]\nx = 1\n"
    doc = tomlrt.loads(src)
    aot = doc.aot("i")
    aot.append({"x": 2})
    assert tomlrt.dumps(doc) == td("""
        [[i]]
        x = 1

        [[i]]
        x = 2
        """)


def test_aot_append_preserves_user_no_blank_style() -> None:
    """When the existing entries clearly show no-blank-line spacing
    (≥2 sibling gaps to learn from), respect the user's choice."""
    src = td("""
        [[i]]
        x = 1
        [[i]]
        x = 2
        """)
    doc = tomlrt.loads(src)
    aot = doc.aot("i")
    aot.append({"x": 3})
    assert tomlrt.dumps(doc) == td("""
        [[i]]
        x = 1
        [[i]]
        x = 2
        [[i]]
        x = 3
        """)


def test_aot_extend_inherits_blank_after_first_added_entry() -> None:
    src = td("""
        [[i]]
        x = 1

        [[i]]
        x = 2
        """)
    doc = tomlrt.loads(src)
    aot = doc.aot("i")
    aot.extend([{"x": 3}, {"x": 4}])
    assert tomlrt.dumps(doc) == (
        td("""
            [[i]]
            x = 1

            [[i]]
            x = 2

            [[i]]
            x = 3

            [[i]]
            x = 4
            """)
    )


def test_aot_entry_inherits_indent_from_dotted_sibling() -> None:
    """A sibling entry whose only KVs are dotted still sets the indent."""
    src = td("""
        [[i]]
          a.b = 1

        [[i]]
        """)
    doc = tomlrt.loads(src)
    doc.aot("i")[1]["z"] = 2
    assert tomlrt.dumps(doc) == td("""
        [[i]]
          a.b = 1

        [[i]]
          z = 2
        """)


# ---------------------------------------------------------------------------
# Inline array style preservation on append/sort/delete
# ---------------------------------------------------------------------------


def test_inline_array_append_preserves_single_space_separator() -> None:
    doc = tomlrt.loads("x = [1, 2, 3]\n")
    doc.array("x").append(4)
    assert tomlrt.dumps(doc) == "x = [1, 2, 3, 4]\n"


def test_inline_array_append_preserves_compact_separator() -> None:
    doc = tomlrt.loads("x = [1,2,3]\n")
    doc.array("x").append(4)
    assert tomlrt.dumps(doc) == "x = [1,2,3,4]\n"


def test_inline_array_append_preserves_bracket_padding() -> None:
    doc = tomlrt.loads("x = [ 1, 2, 3 ]\n")
    doc.array("x").append(4)
    assert tomlrt.dumps(doc) == "x = [ 1, 2, 3, 4 ]\n"


def test_inline_array_append_into_empty_padded() -> None:
    doc = tomlrt.loads("x = [ ]\n")
    doc.array("x").append(1)
    assert tomlrt.dumps(doc) == "x = [ 1 ]\n"


def test_inline_array_append_into_empty_flush() -> None:
    doc = tomlrt.loads("x = []\n")
    doc.array("x").append(1)
    assert tomlrt.dumps(doc) == "x = [1]\n"


def test_multiline_array_append_preserves_trailing_comma_layout() -> None:
    src = td("""
        x = [
            1,
            2,
            3,
        ]
        """)
    doc = tomlrt.loads(src)
    doc.array("x").append(4)
    assert tomlrt.dumps(doc) == td("""
        x = [
            1,
            2,
            3,
            4,
        ]
        """)


def test_multiline_array_append_without_trailing_comma() -> None:
    src = td("""
        x = [
            1,
            2,
            3
        ]
        """)
    doc = tomlrt.loads(src)
    doc.array("x").append(4)
    assert tomlrt.dumps(doc) == td("""
        x = [
            1,
            2,
            3,
            4
        ]
        """)


def test_array_of_inline_tables_append_preserves_separator() -> None:
    doc = tomlrt.loads("x = [{ a = 1 }, { a = 2 }]\n")
    template = tomlrt.loads("x = { a = 3 }\n").table("x")
    doc.array("x").append(template)
    assert tomlrt.dumps(doc) == "x = [{ a = 1 }, { a = 2 }, { a = 3 }]\n"


def test_array_sort_drops_dangling_trailing_comma() -> None:
    doc = tomlrt.loads("x = [3, 1, 2]\n")
    doc.array("x").sort()
    assert tomlrt.dumps(doc) == "x = [1, 2, 3]\n"


def test_array_sort_preserves_bracket_padding() -> None:
    doc = tomlrt.loads("x = [ 3, 1, 2 ]\n")
    doc.array("x").sort()
    assert tomlrt.dumps(doc) == "x = [ 1, 2, 3 ]\n"


def test_array_sort_compact_stays_compact() -> None:
    doc = tomlrt.loads("x = [3,1,2]\n")
    doc.array("x").sort()
    assert tomlrt.dumps(doc) == "x = [1,2,3]\n"


def test_array_pop_preserves_bracket_padding() -> None:
    doc = tomlrt.loads("x = [ 1, 2, 3 ]\n")
    doc.array("x").pop()
    assert tomlrt.dumps(doc) == "x = [ 1, 2 ]\n"


def test_array_del_first_preserves_bracket_padding() -> None:
    doc = tomlrt.loads("x = [ 1, 2, 3 ]\n")
    del doc.array("x")[0]
    assert tomlrt.dumps(doc) == "x = [ 2, 3 ]\n"


# ---------------------------------------------------------------------------
# Inline table style preservation on insert/delete
# ---------------------------------------------------------------------------


def test_inline_table_insert_preserves_padded_style() -> None:
    doc = tomlrt.loads("x = { a = 1, b = 2 }\n")
    doc.table("x")["c"] = 3
    assert tomlrt.dumps(doc) == "x = { a = 1, b = 2, c = 3 }\n"


def test_inline_table_insert_preserves_compact_style() -> None:
    doc = tomlrt.loads("x={a=1, b=2}\n")
    doc.table("x")["c"] = 3
    assert tomlrt.dumps(doc) == "x={a=1, b=2, c=3}\n"


def test_inline_table_insert_into_single_entry_compact() -> None:
    doc = tomlrt.loads("x={a=1}\n")
    doc.table("x")["b"] = 2
    assert tomlrt.dumps(doc) == "x={a=1, b=2}\n"


def test_inline_table_insert_into_empty_padded() -> None:
    doc = tomlrt.loads("x = { }\n")
    doc.table("x")["a"] = 1
    assert tomlrt.dumps(doc) == "x = { a = 1 }\n"


def test_inline_table_insert_into_empty_pads() -> None:
    # An empty ``{}`` carries no populated-style signal (it is just the
    # conventional spelling of "empty"), so inserting the first key
    # adopts the canonical inline-table padding rather than staying tight.
    doc = tomlrt.loads("x = {}\n")
    doc.table("x")["a"] = 1
    assert tomlrt.dumps(doc) == "x = { a = 1 }\n"


def test_inline_table_delete_last_preserves_padding() -> None:
    doc = tomlrt.loads("x = { a = 1, b = 2 }\n")
    del doc.table("x")["b"]
    assert tomlrt.dumps(doc) == "x = { a = 1 }\n"


def test_inline_table_delete_first_preserves_padding() -> None:
    doc = tomlrt.loads("x = { a = 1, b = 2 }\n")
    del doc.table("x")["a"]
    assert tomlrt.dumps(doc) == "x = { b = 2 }\n"


def test_multiline_inline_table_insert_preserves_layout() -> None:
    src = td("""
        x = {
          a = 1,
          b = 2,
        }
        """)
    doc = tomlrt.loads(src)
    doc.table("x")["c"] = 3
    assert tomlrt.dumps(doc) == td("""
        x = {
          a = 1,
          b = 2,
          c = 3,
        }
        """)


def test_multiline_inline_table_insert_does_not_duplicate_above_comment() -> None:
    """The above-entry comment block must stay attached to its entry.

    A naive inter-separator that clones ``entries[1].leading`` whole
    would replicate the comment block onto the appended entry.
    """
    src = td("""
        x = {
          a = 1,
          # above b
          b = 2,
        }
        """)
    doc = tomlrt.loads(src)
    doc.table("x")["c"] = 3
    assert tomlrt.dumps(doc) == td("""
        x = {
          a = 1,
          # above b
          b = 2,
          c = 3,
        }
        """)


# ---------------------------------------------------------------------------
# A7: blank line before the first ``[table]`` when inserting into the
# implicit pre-header section.
# ---------------------------------------------------------------------------


def test_insert_top_level_kv_adds_blank_before_first_table() -> None:
    doc = tomlrt.loads("[a]\nx = 1\n")
    doc["z"] = 99
    assert tomlrt.dumps(doc) == td("""
        z = 99

        [a]
        x = 1
        """)


def test_insert_top_level_kv_preserves_existing_blank() -> None:
    doc = tomlrt.loads(
        td("""
        [a]
        x = 1

        [b]
        y = 2
        """)
    )
    doc["z"] = 99
    assert tomlrt.dumps(doc) == td("""
        z = 99

        [a]
        x = 1

        [b]
        y = 2
        """)


def test_multiple_top_level_kv_inserts_share_one_blank() -> None:
    doc = tomlrt.loads("[a]\nx = 1\n")
    doc["y"] = 1
    doc["z"] = 2
    assert tomlrt.dumps(doc) == td("""
        y = 1
        z = 2

        [a]
        x = 1
        """)


def test_top_level_insert_into_existing_pre_header_section() -> None:
    doc = tomlrt.loads(
        td("""
        first = 0
        [a]
        x = 1
        """)
    )
    doc["z"] = 99
    # The user's existing layout had no blank between ``first`` and
    # ``[a]``; appending another KV must not silently inject one.
    assert tomlrt.dumps(doc) == td("""
        first = 0
        z = 99
        [a]
        x = 1
        """)


def test_aot_first_two_appends_into_empty_doc_blank_separate() -> None:
    """Bug report: programmatically-built AoT entries rendered glued."""
    doc = tomlrt.loads("")
    doc["package"] = AoT()
    aot = doc["package"]
    aot.append({"name": "foo"})
    aot.append({"name": "bar"})
    assert tomlrt.dumps(doc) == (
        td("""
            [[package]]
            name = "foo"

            [[package]]
            name = "bar"
            """)
    )


def test_aot_append_after_sub_section_blank_separates() -> None:
    """Bug report: an AoT entry following a previous entry's sub-section
    must still be blank-separated from that sub-section."""
    doc = tomlrt.loads("")
    doc["package"] = AoT()
    aot = doc["package"]
    aot.append({"name": "foo", "version": "1.0"})
    aot[-1]["dependencies"] = Table.section({"bar": "^1"})
    aot.append({"name": "baz", "version": "2.0"})
    assert tomlrt.dumps(doc) == (
        '[[package]]\nname = "foo"\nversion = "1.0"\n'
        "\n"
        '[package.dependencies]\nbar = "^1"\n'
        "\n"
        '[[package]]\nname = "baz"\nversion = "2.0"\n'
    )


# ---------------------------------------------------------------------------
# Blank-line detection across a comment block
# ---------------------------------------------------------------------------


def test_new_section_mirrors_a_gap_of_comments_plus_a_blank() -> None:
    """A gap counts as blank-separated even with comments in it.

    The blank line is what the new ``[c]`` mirrors; the comment lines
    above it neither hide it nor supply one of their own.
    """
    doc = tomlrt.loads(
        td("""
        [a]
        x = 1
        # note one
        # note two

        [b]
        y = 2
        """)
    )
    doc["c"] = Table.section({"z": 3})
    assert tomlrt.dumps(doc) == td("""
        [a]
        x = 1
        # note one
        # note two

        [b]
        y = 2

        [c]
        z = 3
        """)
