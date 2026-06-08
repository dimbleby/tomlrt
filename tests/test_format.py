"""Tests for `Container.format()` and `Array.format()`."""

from __future__ import annotations

import pytest
import tomli

import tomlrt
from _helpers import td
from tomlrt import TOMLError


def _roundtrip(src: str, *, comments: bool = True) -> str:
    doc = tomlrt.loads(src)
    doc.format(comments=comments)
    return tomlrt.dumps(doc)


def test_idempotent() -> None:
    src = td("""
        [foo]
        a    =   1
        b=2
        arr = [1,2 ,3]
    """)
    once = _roundtrip(src)
    doc = tomlrt.loads(once)
    doc.format()
    twice = tomlrt.dumps(doc)
    assert once == twice


def test_kv_spacing() -> None:
    src = td("""
        a   =1
        b=2
    """)
    assert _roundtrip(src) == td("""
        a = 1
        b = 2
    """)


def test_dotted_key_normalised() -> None:
    src = "a . b   .  c = 1\n"
    assert _roundtrip(src) == "a.b.c = 1\n"


def test_header_brackets() -> None:
    src = td("""
        [  a . b  ]
        x = 1
    """)
    assert _roundtrip(src) == td("""
        [a.b]
        x = 1
    """)


def test_inter_kv_blank_collapses() -> None:
    src = td("""
        a = 1


        b = 2
    """)
    assert _roundtrip(src) == td("""
        a = 1
        b = 2
    """)


def test_inter_section_blank_canonical() -> None:
    src = td("""
        [a]
        x = 1



        [b]
        y = 2
    """)
    assert _roundtrip(src) == td("""
        [a]
        x = 1

        [b]
        y = 2
    """)


def test_two_sections_touching_get_blank() -> None:
    src = td("""
        [a]
        x = 1
        [b]
        y = 2
    """)
    assert _roundtrip(src) == td("""
        [a]
        x = 1

        [b]
        y = 2
    """)


def test_orphan_comment_block_preserved() -> None:
    src = td("""
        a = 1

        # orphan one
        # orphan two

        # attached
        b = 2
    """)
    assert _roundtrip(src) == td("""
        a = 1

        # orphan one
        # orphan two

        # attached
        b = 2
    """)


def test_blank_above_orphan_preserved_when_collapsing() -> None:
    # The structural blank between sibling KVs normally collapses to
    # zero, but if there's an orphan / attached comment block above
    # the next slot the blank stays so the block is visually
    # separated from the previous slot.
    src = td("""
        a = 1


        # orphan
        b = 2
    """)
    assert _roundtrip(src) == td("""
        a = 1

        # orphan
        b = 2
    """)


def test_single_line_inline_array() -> None:
    src = "arr = [1,2 ,3 ]\n"
    assert _roundtrip(src) == "arr = [1, 2, 3]\n"


def test_inline_table_single_line() -> None:
    src = "t = {a=1,b=2}\n"
    assert _roundtrip(src) == "t = { a = 1, b = 2 }\n"


def test_multiline_array_preserved_shape() -> None:
    src = td("""
        arr = [
          1,
          2,
          3,
        ]
    """)
    assert _roundtrip(src) == td("""
        arr = [
          1,
          2,
          3,
        ]
    """)


def test_comma_first_array_keeps_comment() -> None:
    # A comma-first layout parks the item's EOL comment in ``trailing``
    # *before* the comma, even though ``has_comma`` is set. format()
    # used to look only in ``post_comma_trivia`` and drop the comment;
    # it is now migrated to the canonical post-comma position.
    src = td("""
        a = [
              1 # comma is on the next line
             ,2
            ]
    """)
    assert _roundtrip(src) == td("""
        a = [
          1, # comma is on the next line
          2,
        ]
    """)


def test_comma_first_inline_table_keeps_comment() -> None:
    src = td("""
        t = {
              a = 1 # comma is on the next line
             ,b = 2
            }
    """)
    assert _roundtrip(src) == td("""
        t = {
          a = 1, # comma is on the next line
          b = 2,
        }
    """)


def test_nested_multiline_array_indents_per_depth() -> None:
    src = td("""
        outer = [
        [
        1,
        2,
        ],
        [
        3,
        4,
        ],
        ]
    """)
    assert _roundtrip(src) == td("""
        outer = [
          [
            1,
            2,
          ],
          [
            3,
            4,
          ],
        ]
    """)


def test_nested_multiline_table_indents_per_depth() -> None:
    src = td("""
        outer = {
        a = {
        x = 1,
        y = 2,
        },
        b = {
        x = 3,
        y = 4,
        },
        }
    """)
    assert _roundtrip(src) == td("""
        outer = {
          a = {
            x = 1,
            y = 2,
          },
          b = {
            x = 3,
            y = 4,
          },
        }
    """)


def test_nested_array_in_multiline_table_indents() -> None:
    src = td("""
        outer = {
        items = [
        1,
        2,
        ],
        }
    """)
    assert _roundtrip(src) == td("""
        outer = {
          items = [
            1,
            2,
          ],
        }
    """)


def test_nested_table_in_multiline_array_indents() -> None:
    src = td("""
        outer = [
        {
        x = 1,
        y = 2,
        },
        {
        x = 3,
        y = 4,
        },
        ]
    """)
    assert _roundtrip(src) == td("""
        outer = [
          {
            x = 1,
            y = 2,
          },
          {
            x = 3,
            y = 4,
          },
        ]
    """)


def test_aot_recursion() -> None:
    src = td("""
        [[a]]
        x=  1

        [[a]]
        x=2
    """)
    assert _roundtrip(src) == td("""
        [[a]]
        x = 1

        [[a]]
        x = 2
    """)


def test_comments_false_keeps_comment_text() -> None:
    src = td("""
        a = 1   #foo
        b = 2   #   bar
    """)
    assert _roundtrip(src, comments=False) == td("""
        a = 1 #foo
        b = 2 #   bar
    """)


def test_comments_true_normalises_text() -> None:
    src = "a = 1 #foo\nb = 2 #   bar\nc = 3 # baz   \n"
    assert _roundtrip(src, comments=True) == td("""
        a = 1 # foo
        b = 2 # bar
        c = 3 # baz
    """)


def test_bare_hash_comment_stays() -> None:
    assert _roundtrip("a = 1 #\n", comments=True) == "a = 1 #\n"


def test_crlf_newlines_retargeted() -> None:
    src = "a   =   1\r\nb=2\r\n"
    out = _roundtrip(src)
    assert out == "a = 1\r\nb = 2\r\n"


def test_preamble_preserved() -> None:
    src = td("""
        # preamble line one
        # preamble line two

        [a]
        x = 1
    """)
    assert _roundtrip(src) == td("""
        # preamble line one
        # preamble line two

        [a]
        x = 1
    """)


def test_detached_section_raises() -> None:
    t = tomlrt.Table.section()
    t["x"] = 1
    with pytest.raises(TOMLError):
        t.format()


def test_detached_inline_factory_raises() -> None:
    t = tomlrt.Table.inline()
    t["x"] = 1
    with pytest.raises(TOMLError):
        t.format()


def test_inline_root_format() -> None:
    src = "t = {x=1, y=2}\n"
    doc = tomlrt.loads(src)
    t = doc.table("t")
    t.format()
    assert tomlrt.dumps(doc) == "t = { x = 1, y = 2 }\n"


def test_array_format_direct() -> None:
    src = "arr = [1, 2,3 ]\n"
    doc = tomlrt.loads(src)
    arr = doc.array("arr")
    arr.format()
    assert tomlrt.dumps(doc) == "arr = [1, 2, 3]\n"


def test_section_scoped() -> None:
    src = td("""
        [keep]
        a   =1



        [target]
        b   =2
    """)
    doc = tomlrt.loads(src)
    doc.table("target").format()
    # 'keep' subtree is untouched (raw spacing, blank gap intact);
    # 'target' subtree is canonicalised, but the gap above its header
    # is owned by the parent so it stays as-is.
    assert tomlrt.dumps(doc) == td("""
        [keep]
        a   =1



        [target]
        b = 2
    """)


def test_format_canonicalises_epilogue() -> None:
    src = "key = 1\n#trailing  \n#  more  \n"
    assert _roundtrip(src) == "key = 1\n# trailing\n# more\n"


def test_format_canonicalises_epilogue_no_comments_flag() -> None:
    # comments=False strips trailing WS on blank lines, but leaves
    # comment text (including its trailing whitespace) alone.
    src = "key = 1\n#trailing  \n  \n"
    assert _roundtrip(src, comments=False) == "key = 1\n#trailing  \n\n"


def test_format_canonicalises_empty_doc_preamble() -> None:
    # Empty doc: everything lives in _trailing and surfaces as preamble.
    src = "#hello  \n#  world\n"
    assert _roundtrip(src) == "# hello\n# world\n"


def test_format_restores_newline_after_sort_of_last_section() -> None:
    # The original last section lacks a trailing newline; sort moves it
    # into the middle. format() must restore the eol newline so the
    # canonical inter-section blank line materialises.
    src = '# Header\n\ntitle = "X"\n\n[a]\nx = 1\n\n[b]\ny = 2'
    doc = tomlrt.loads(src)
    doc.sort(reverse=True)
    doc.format()
    assert tomlrt.dumps(doc) == td("""
        # Header

        title = "X"

        [b]
        y = 2

        [a]
        x = 1
    """)


def test_format_preserves_above_item_comment_indent_in_multiline_array() -> None:
    src = td("""
        arr = [
          # block comment
          'a',
        ]
        """)
    assert _roundtrip(src) == src


def test_format_preserves_above_item_comment_indent_in_multiline_inline_table() -> None:
    src = td("""
        t = {
          # block comment
          a = 1,
        }
        """)
    assert _roundtrip(src) == src


def test_format_preserves_orphan_comment_indent_in_multiline_array() -> None:
    src = td("""
        arr = [
          # orphan

          'a',
        ]
        """)
    assert _roundtrip(src) == src


def test_format_returns_none() -> None:
    doc = tomlrt.loads("a = 1\n")
    assert doc.format() is None  # type: ignore[func-returns-value]
    arr = tomlrt.loads("a = [1]\n").array("a")
    assert arr.format() is None  # type: ignore[func-returns-value]


def test_roundtrip_through_tomli() -> None:
    src = td("""
        # top
        [section]
        a = 1
        b = 2.5
        c = "hi"
        arr = [1, 2, 3]
        inl = { x = 1, y = 2 }

        [[aot]]
        n = 1

        [[aot]]
        n = 2
    """)
    out = _roundtrip(src)

    assert tomli.loads(out) == tomli.loads(src)


def test_format_aot_entry_does_not_leak_to_siblings() -> None:
    src = td("""
        [[a]]
        x=1
        [[a]]
        y    =    2
        [[a]]
        z=3
    """)
    doc = tomlrt.loads(src)
    doc["a"][1].format()
    assert tomlrt.dumps(doc) == td("""
        [[a]]
        x=1
        [[a]]
        y = 2
        [[a]]
        z=3
    """)


def test_format_first_aot_entry_does_not_touch_siblings() -> None:
    src = td("""
        [[a]]
        x   =   1
        [[a]]
        y=2
        [[a]]
        z=3
    """)
    doc = tomlrt.loads(src)
    doc["a"][0].format()
    assert tomlrt.dumps(doc) == td("""
        [[a]]
        x = 1
        [[a]]
        y=2
        [[a]]
        z=3
    """)


def test_format_multiline_inline_table_keeps_eol_comment_no_blank() -> None:
    src = td("""
        a = {
          x = 1, # eol
          y = 2
        }
    """)
    doc = tomlrt.loads(src)
    doc.table("a").format()
    assert tomlrt.dumps(doc) == td("""
        a = {
          x = 1, # eol
          y = 2,
        }
    """)


def test_format_multiline_array_keeps_eol_comment_no_blank() -> None:
    src = td("""
        a = [
          1, # eol
          2,
        ]
    """)
    doc = tomlrt.loads(src)
    doc.array("a").format()
    assert tomlrt.dumps(doc) == td("""
        a = [
          1, # eol
          2,
        ]
    """)


def test_format_implicit_section() -> None:
    """`format()` on an implicit-section view canonicalises all owned slots.

    Implicit sections have no `[a]` header in source — their slots are
    only reached via descendants (dotted KVs like `a.x = 1`, sub-section
    headers like `[a.b]`, AoT headers like `[[a.arr]]`) — so the path
    cannot reuse the contiguous subtree walk used for SECTION views.
    Exercise all three flavours so every branch of the implicit-section
    canonicaliser runs.
    """
    src = td("""
        a.x   =1
        [a.b]
        y=2
        [[a.arr]]
        z=3
    """)
    doc = tomlrt.loads(src)
    doc.table("a").format()
    assert tomlrt.dumps(doc) == td("""
        a.x = 1
        [a.b]
        y = 2
        [[a.arr]]
        z = 3
    """)


def test_format_implicit_section_detached_raises() -> None:
    """A detached implicit-section view cannot be formatted."""
    doc = tomlrt.loads("[a.b]\nx = 1\n")
    implicit = doc.table("a")
    del doc["a"]
    with pytest.raises(TOMLError, match="attached"):
        implicit.format()
