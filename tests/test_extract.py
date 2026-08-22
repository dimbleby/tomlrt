"""Extracting a section-backed table as a document of its own.

``tomlrt.dumps(table)`` wraps the table in a `Document`, which re-roots
the table's subtree: sub-tables lose the parent's path prefix, and the
layout the table holds — its own comments and spacing as much as its
children's — travels with it.
"""

from __future__ import annotations

import tomlrt
from _helpers import td
from tomlrt import Document, Table


def test_table_keeps_its_own_comments() -> None:
    src = td("""
        [tool.dg]
        a = 1  # note-a
        # lead-b
        b   =   2

        [tool.dg.sub]
        c = 3  # note-c
    """)
    t = tomlrt.loads(src).table("tool.dg")
    assert tomlrt.dumps(t) == td("""
        a = 1  # note-a
        # lead-b
        b   =   2

        [sub]
        c = 3  # note-c
    """)


def test_document_of_table_matches_dumps() -> None:
    src = td("""
        [t]
        a = 1  # note
    """)
    t = tomlrt.loads(src).table("t")
    assert tomlrt.dumps(Document(t)) == tomlrt.dumps(t)


def test_header_comments_become_the_preamble() -> None:
    src = td("""
        # what t is for
        [t]  # and a note
        k = 1
    """)
    doc = Document(tomlrt.loads(src).table("t"))
    assert doc.preamble == ("what t is for", "and a note")
    assert tomlrt.dumps(doc) == td("""
        # what t is for
        # and a note

        k = 1
    """)


def test_disjoint_block_above_the_header_travels() -> None:
    src = td("""
        [other]
        # note about t

        [t]
        k = 1
    """)
    doc = tomlrt.loads(src)
    # The block is t's own, as ``header_leading_block`` and ``sort`` agree.
    assert doc.table("t").header_leading_block == ("note about t", None)
    assert tomlrt.dumps(doc.table("t")) == td("""
        # note about t

        k = 1
    """)


def test_sub_section_separators_are_left_alone() -> None:
    # The source's own spacing wins over the document-wide convention,
    # in both directions.
    blank = td("""
        [a]
        y = 2

        [a.b]
        x = 1
        [c]
        z = 1
    """)
    assert tomlrt.dumps(tomlrt.loads(blank).table("a")) == td("""
        y = 2

        [b]
        x = 1
    """)
    tight = td("""
        [a]
        y = 2
        [a.b]
        x = 1

        [c]
        z = 1
    """)
    assert tomlrt.dumps(tomlrt.loads(tight).table("a")) == td("""
        y = 2
        [b]
        x = 1
    """)


def test_forward_declared_parent_keeps_its_keys_in_scope() -> None:
    src = td("""
        [a.b]
        x = 1

        [a]
        y = 2  # still a's
    """)
    t = tomlrt.loads(src).table("a")
    # ``y`` is a key of the extracted root, so it has to precede ``[b]``.
    assert tomlrt.dumps(t) == td("""
        y = 2  # still a's

        [b]
        x = 1
    """)
    # The seam is new, so it takes the document's own section spacing.
    tight = tomlrt.loads("[a.b]\nx = 1\n[a]\ny = 2\n")
    assert tomlrt.dumps(tight.table("a")) == "y = 2\n[b]\nx = 1\n"


def test_implicit_table_hosted_by_a_dotted_key() -> None:
    src = td("""
        a.x = 1  # dotted

        [a.sub]
        y = 2
    """)
    t = tomlrt.loads(src).table("a")
    assert tomlrt.dumps(t) == td("""
        x = 1  # dotted

        [sub]
        y = 2
    """)


def test_implicit_parent_of_a_section() -> None:
    src = td("""
        [tool.dg]
        a = 1  # note
    """)
    t = tomlrt.loads(src).table("tool")
    assert tomlrt.dumps(t) == td("""
        [dg]
        a = 1  # note
    """)


def test_aot_entries_keep_their_shape() -> None:
    src = td("""
        [t]
        k = 1

        [[t.e]]
        n = 1  # first

        [[t.e]]
        n = 2
    """)
    doc = tomlrt.loads(src)
    assert tomlrt.dumps(doc.table("t")) == td("""
        k = 1

        [[e]]
        n = 1  # first

        [[e]]
        n = 2
    """)
    assert tomlrt.dumps(doc.aot("t.e")[0]) == "n = 1  # first\n"


def test_crlf_and_missing_final_newline_survive() -> None:
    src = "[t]\r\na = 1  # c\r\nb = 2"
    t = tomlrt.loads(src).table("t")
    assert tomlrt.dumps(t) == "a = 1  # c\r\nb = 2"


def test_empty_section_extracts_to_an_empty_document() -> None:
    t = tomlrt.loads("[t]\n").table("t")
    assert tomlrt.dumps(t) == ""


def test_extraction_leaves_the_source_alone() -> None:
    src = td("""
        [t]
        a = 1  # note

        [t.sub]
        b = 2
    """)
    doc = tomlrt.loads(src)
    extracted = Document(doc.table("t"))
    extracted["a"] = 99
    extracted["sub"]["b"] = 99
    assert tomlrt.dumps(doc) == src
    assert tomlrt.dumps(extracted) == td("""
        a = 99  # note

        [sub]
        b = 99
    """)


def test_section_spacing_convention_is_inherited() -> None:
    src = td("""
        [t]
        a = 1
        [t.sub]
        b = 2
    """)
    extracted = Document(tomlrt.loads(src).table("t"))
    extracted["fresh"] = Table.section({"c": 3})
    assert tomlrt.dumps(extracted) == td("""
        a = 1
        [sub]
        b = 2
        [fresh]
        c = 3
    """)


def test_inline_and_detached_tables_are_rebuilt_from_data() -> None:
    src = "a = { x = 1, y = 2 }  # note\n"
    t = tomlrt.loads(src).table("a")
    assert t.is_inline
    assert tomlrt.dumps(t) == td("""
        x = 1
        y = 2
    """)
    assert tomlrt.dumps(Table.section({"k": "v"})) == 'k = "v"\n'
