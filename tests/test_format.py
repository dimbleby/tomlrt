"""Tests for `Container.format()` and `Array.format()`."""

from __future__ import annotations

import warnings
from dataclasses import FrozenInstanceError
from inspect import Parameter, signature
from typing import Any

import pytest
import tomli

import tomlrt
from _helpers import reparses, td
from tomlrt import TOMLError


def _roundtrip(src: str, *, comments: bool = True) -> str:
    doc = tomlrt.loads(src)
    doc.format(options=tomlrt.FormatOptions(normalize_comments=comments))
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


def test_format_keeps_comment_after_last_array_item() -> None:
    # A comment block before `]`, after the last item's EOL comment, lives in
    # `final_trivia` with no leading newline; format used to drop its first
    # line (losing one comment per pass).
    src = td("""
        a = [
          1,
          2, # eol
          # trailing
        ]
    """)
    once = _roundtrip(src)
    assert once == src
    doc = tomlrt.loads(once)
    doc.format()
    assert tomlrt.dumps(doc) == once


def test_kv_spacing() -> None:
    src = td("""
        a   =1
        b=2
    """)
    assert _roundtrip(src) == td("""
        a = 1
        b = 2
    """)


def test_top_level_key_indent_is_removed() -> None:
    assert _roundtrip("  a=1\n") == "a = 1\n"


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


def test_blank_runs_between_comment_blocks_collapse() -> None:
    src = td("""
        a = 1


        # orphan



        # attached
        b = 2
    """)
    assert _roundtrip(src) == td("""
        a = 1

        # orphan

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


def test_multiline_array_drops_blank_only_line_before_closing_bracket() -> None:
    doc = tomlrt.loads(
        td("""
        a = [
          1,

        ]
        """)
    )

    doc.array("a").format()

    assert tomlrt.dumps(doc) == td("""
        a = [
          1,
        ]
    """)


def test_multiline_inline_table_drops_blank_only_line_before_closing_brace() -> None:
    doc = tomlrt.loads(
        td("""
        a = {
          x = 1,

        }
        """)
    )

    doc.table("a").format()

    assert tomlrt.dumps(doc) == td("""
        a = {
          x = 1,
        }
    """)


def test_format_options_omit_multiline_trailing_comma() -> None:
    src = td("""
        items = [
          1,
          2,
        ]
    """)
    doc = tomlrt.loads(src)
    options = tomlrt.FormatOptions(multiline_trailing_comma=False)
    doc.format(options=options)
    expected = td("""
        items = [
          1,
          2
        ]
    """)
    assert tomlrt.dumps(doc) == expected
    assert tomli.loads(expected) == {"items": [1, 2]}
    doc.format(options=options)
    assert tomlrt.dumps(doc) == expected


def test_no_trailing_comma_preserves_final_item_eol_comment() -> None:
    src = td("""
        items = [
          1,
          2, # final
        ]
    """)
    doc = tomlrt.loads(src)
    doc.array("items").format(
        options=tomlrt.FormatOptions(multiline_trailing_comma=False)
    )
    assert tomlrt.dumps(doc) == td("""
        items = [
          1,
          2 # final
        ]
    """)


def test_no_trailing_comma_recurses_through_arrays_and_inline_tables() -> None:
    src = td("""
        outer = [
          {
            values = [
              1,
            ],
            table = {
              value = 2,
            },
          },
        ]
    """)
    doc = tomlrt.loads(src)
    doc.format(options=tomlrt.FormatOptions(multiline_trailing_comma=False))
    assert tomlrt.dumps(doc) == td("""
        outer = [
          {
            values = [
              1
            ],
            table = {
              value = 2
            }
          }
        ]
    """)


def test_no_trailing_comma_preserves_empty_array_and_crlf() -> None:
    src = td("""
        items = [
        ]
    """).replace("\n", "\r\n")
    doc = tomlrt.loads(src)
    doc.format(options=tomlrt.FormatOptions(multiline_trailing_comma=False))
    assert tomlrt.dumps(doc) == src


def test_append_after_format_without_trailing_comma_preserves_style() -> None:
    src = td("""
        items = [
          1,
          2, # final
        ]
    """)
    doc = tomlrt.loads(src)
    items = doc.array("items")
    items.format(options=tomlrt.FormatOptions(multiline_trailing_comma=False))
    items.append(3)
    assert tomlrt.dumps(doc) == td("""
        items = [
          1,
          2, # final
          3
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


def test_comma_first_leading_blocks_remain_leading_when_formatted() -> None:
    src = td("""
        a = [
              1
              # array leading
             ,2
            ]
        t = {
              a = 1
              # inline leading
             ,b = 2
            }
    """)
    doc = tomlrt.loads(src)
    doc.format()
    assert tomlrt.dumps(doc) == td("""
        a = [
          1,
          # array leading
          2,
        ]
        t = {
          a = 1,
          # inline leading
          b = 2,
        }
    """)


def test_format_preserves_final_block_after_closed_item_row() -> None:
    src = td("""
        x = [
          1, # eol

          # closing
        ]
    """)
    doc = tomlrt.loads(src)
    doc.array("x").format()
    assert tomlrt.dumps(doc) == src


def test_format_preserves_final_block_after_open_item_row() -> None:
    src = td("""
        x = [
          1,

          # closing
        ]
    """)
    doc = tomlrt.loads(src)
    doc.array("x").format()
    assert tomlrt.dumps(doc) == src


@pytest.mark.parametrize(
    "options",
    [
        tomlrt.FormatOptions(multiline_trailing_comma=True),
        tomlrt.FormatOptions(multiline_trailing_comma=False),
    ],
)
def test_comma_first_array_keeps_post_comma_comment(
    options: tomlrt.FormatOptions,
) -> None:
    src = td("""
        a = [
          1
          , # after comma
          2
        ]
    """)
    doc = tomlrt.loads(src)
    doc.format(options=options)
    expected = (
        td("""
            a = [
              1, # after comma
              2,
            ]
        """)
        if options.multiline_trailing_comma
        else td("""
            a = [
              1, # after comma
              2
            ]
        """)
    )
    assert tomlrt.dumps(doc) == expected


@pytest.mark.parametrize(
    "options",
    [
        tomlrt.FormatOptions(multiline_trailing_comma=True),
        tomlrt.FormatOptions(multiline_trailing_comma=False),
    ],
)
def test_comma_first_inline_table_keeps_post_comma_comment(
    options: tomlrt.FormatOptions,
) -> None:
    src = td("""
        t = {
          a = 1
          , # after comma
          b = 2
        }
    """)
    doc = tomlrt.loads(src)
    doc.format(options=options)
    expected = (
        td("""
            t = {
              a = 1, # after comma
              b = 2,
            }
        """)
        if options.multiline_trailing_comma
        else td("""
            t = {
              a = 1, # after comma
              b = 2
            }
        """)
    )
    assert tomlrt.dumps(doc) == expected


@pytest.mark.parametrize(
    "options",
    [
        tomlrt.FormatOptions(multiline_trailing_comma=True),
        tomlrt.FormatOptions(multiline_trailing_comma=False),
    ],
)
def test_comma_first_array_keeps_comments_on_both_sides(
    options: tomlrt.FormatOptions,
) -> None:
    src = td("""
        a = [
          1 # before comma
          , # after comma
          2
        ]
    """)
    doc = tomlrt.loads(src)
    doc.format(options=options)
    expected = (
        td("""
            a = [
              1, # before comma
              # after comma
              2,
            ]
        """)
        if options.multiline_trailing_comma
        else td("""
            a = [
              1, # before comma
              # after comma
              2
            ]
        """)
    )
    assert tomlrt.dumps(doc) == expected


@pytest.mark.parametrize(
    "options",
    [
        tomlrt.FormatOptions(multiline_trailing_comma=True),
        tomlrt.FormatOptions(multiline_trailing_comma=False),
    ],
)
def test_comma_first_inline_table_keeps_comments_on_both_sides(
    options: tomlrt.FormatOptions,
) -> None:
    src = td("""
        t = {
          a = 1 # before comma
          , # after comma
          b = 2
        }
    """)
    doc = tomlrt.loads(src)
    doc.format(options=options)
    expected = (
        td("""
            t = {
              a = 1, # before comma
              # after comma
              b = 2,
            }
        """)
        if options.multiline_trailing_comma
        else td("""
            t = {
              a = 1, # before comma
              # after comma
              b = 2
            }
        """)
    )
    assert tomlrt.dumps(doc) == expected


def test_comma_first_multiple_comments_with_zero_indent() -> None:
    src = td("""
        a = [
          1 # before comma
          , # after comma
          2
        ]
    """)
    doc = tomlrt.loads(src)
    doc.format(
        options=tomlrt.FormatOptions(
            indent=0,
            multiline_trailing_comma=False,
        )
    )
    assert tomlrt.dumps(doc) == td("""
        a = [
        1, # before comma
        # after comma
        2
        ]
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


def test_format_options_indent_recurses_through_inline_values() -> None:
    src = td("""
        outer = [
        { nested = [
        1,
        ] },
        ]
    """)
    doc = tomlrt.loads(src)
    doc.format(options=tomlrt.FormatOptions(indent=4))
    assert tomlrt.dumps(doc) == td("""
        outer = [
            { nested = [
                1,
            ] },
        ]
    """)


def test_array_scoped_format_preserves_outer_indent_with_custom_step() -> None:
    src = td("""
        outer = [
          { nested = [
             1,
          ] },
        ]
    """)
    doc = tomlrt.loads(src)
    doc.array("outer").table(0).array("nested").format(
        options=tomlrt.FormatOptions(indent=4)
    )
    assert tomlrt.dumps(doc) == td("""
        outer = [
          { nested = [
              1,
          ] },
        ]
    """)


def test_table_scoped_format_preserves_outer_indent_with_custom_step() -> None:
    src = td("""
        outer = [
          { nested = {
             value=1,
          } },
        ]
    """)
    doc = tomlrt.loads(src)
    doc.array("outer").table(0).table("nested").format(
        options=tomlrt.FormatOptions(indent=4)
    )
    assert tomlrt.dumps(doc) == td("""
        outer = [
          { nested = {
              value = 1,
          } },
        ]
    """)


def test_format_options_zero_indent() -> None:
    src = td("""
        outer = [
          [
            1,
          ],
        ]
    """)
    doc = tomlrt.loads(src)
    doc.format(options=tomlrt.FormatOptions(indent=0))
    assert tomlrt.dumps(doc) == td("""
        outer = [
        [
        1,
        ],
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


def test_format_options_eol_comment_spaces_cover_supported_comments() -> None:
    src = td("""
        top=1#top
        array = [
          1,#array
        ]
        table = {
          value=1,#entry
        }
        [section]#header
        x=1
    """)
    doc = tomlrt.loads(src)
    doc.format(options=tomlrt.FormatOptions(eol_comment_spaces=3))
    assert tomlrt.dumps(doc) == td("""
        top = 1   # top
        array = [
          1,   # array
        ]
        table = {
          value = 1,   # entry
        }

        [section]   # header
        x = 1
    """)


def test_format_options_zero_eol_comment_spaces() -> None:
    src = td("""
        key = 1   # key
        values = [
          1,   # item
        ]
    """)
    doc = tomlrt.loads(src)
    doc.format(options=tomlrt.FormatOptions(eol_comment_spaces=0))
    assert tomlrt.dumps(doc) == td("""
        key = 1# key
        values = [
          1,# item
        ]
    """)


def test_comment_spacing_is_independent_of_text_normalization() -> None:
    src = td("""
        key=1#   text
        values = [
          1,#   item
        ]
    """)
    doc = tomlrt.loads(src)
    doc.format(
        options=tomlrt.FormatOptions(
            normalize_comments=False,
            eol_comment_spaces=2,
        )
    )
    assert tomlrt.dumps(doc) == td("""
        key = 1  #   text
        values = [
          1,  #   item
        ]
    """)


def test_eol_comment_spacing_preserves_opening_bracket_spacing() -> None:
    src = td("""
        values = [   # opening
          1, # item
        ]
    """)
    doc = tomlrt.loads(src)
    doc.format(options=tomlrt.FormatOptions(eol_comment_spaces=2))
    assert tomlrt.dumps(doc) == td("""
        values = [   # opening
          1,  # item
        ]
    """)


def test_eol_comment_spacing_preserves_crlf() -> None:
    src = td("""
        values = [
          1,# item
        ]
    """).replace("\n", "\r\n")
    doc = tomlrt.loads(src)
    doc.array("values").format(options=tomlrt.FormatOptions(eol_comment_spaces=2))
    assert tomlrt.dumps(doc) == td("""
        values = [
          1,  # item
        ]
    """).replace("\n", "\r\n")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        pytest.param(
            {"indent": -1},
            "indent must be non-negative",
            id="indent",
        ),
        pytest.param(
            {"eol_comment_spaces": -1},
            "eol_comment_spaces must be non-negative",
            id="eol-comment-spaces",
        ),
    ],
)
def test_format_options_rejects_negative_values(
    kwargs: Any,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        tomlrt.FormatOptions(**kwargs)


def test_format_options_is_frozen_and_keyword_only() -> None:
    options = tomlrt.FormatOptions()
    assert options.normalize_comments is True
    attribute = "normalize_comments"
    with pytest.raises(FrozenInstanceError):
        setattr(options, attribute, False)
    parameters = signature(tomlrt.FormatOptions).parameters
    assert all(p.kind is Parameter.KEYWORD_ONLY for p in parameters.values())


def test_all_format_options_interact_without_changing_data() -> None:
    src = td("""
        root=1#   root
        outer = [
          {
            values = [
              1,
              2,#   final
            ],
            table = {
              value=3,#   entry
            },
          },
        ]
        [section]#   header
        key=4
    """)
    doc = tomlrt.loads(src)
    before = doc.to_dict()
    options = tomlrt.FormatOptions(
        normalize_comments=False,
        indent=4,
        eol_comment_spaces=2,
        multiline_trailing_comma=False,
    )
    doc.format(options=options)
    expected = td("""
        root = 1  #   root
        outer = [
            {
                values = [
                    1,
                    2  #   final
                ],
                table = {
                    value = 3  #   entry
                }
            }
        ]

        [section]  #   header
        key = 4
    """)
    assert tomlrt.dumps(doc) == expected
    assert doc.to_dict() == before
    assert tomli.loads(expected) == before
    doc.format(options=options)
    assert tomlrt.dumps(doc) == expected


def test_legacy_comments_warns_at_caller_for_each_receiver() -> None:
    doc = tomlrt.loads(
        td("""
        a = { x=1 }
        b = [1,2 ]
    """)
    )
    receivers = (doc, doc.table("a"), doc.array("b"))
    for receiver in receivers:
        with pytest.warns(
            DeprecationWarning, match="comments= is deprecated"
        ) as caught:
            receiver.format(comments=False)
        assert caught[0].filename == __file__


def test_format_rejects_options_with_legacy_comments_without_mutating() -> None:
    src = "a   =1\n"
    doc = tomlrt.loads(src)
    with pytest.raises(ValueError, match="cannot specify both"):
        doc.format(options=tomlrt.FormatOptions(), comments=False)
    assert tomlrt.dumps(doc) == src


def test_recursive_format_with_options_emits_no_deprecation_warning() -> None:
    doc = tomlrt.loads(
        td("""
        a.x   =1
        [a.b]
        y=2
    """)
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        doc.table("a").format(options=tomlrt.FormatOptions())
    assert caught == []
    assert tomlrt.dumps(doc) == td("""
        a.x = 1
        [a.b]
        y = 2
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


def test_preamble_blank_run_collapses_before_kv() -> None:
    # A preamble separated from the first slot by 2+ blank lines must
    # collapse to a single blank line, like any other blank run.
    src = td("""
        # pre



        k = 1
    """)
    out = _roundtrip(src)
    assert out == td("""
        # pre

        k = 1
    """)
    assert reparses(out) == {"k": 1}


def test_preamble_blank_run_collapses_before_section() -> None:
    src = td("""
        # pre



        [a]
        x = 1
    """)
    out = _roundtrip(src)
    assert out == td("""
        # pre

        [a]
        x = 1
    """)
    assert reparses(out) == {"a": {"x": 1}}


def test_preamble_blank_run_collapses_before_orphan_comment() -> None:
    src = td("""
        # pre


        # orphan
        k = 1
    """)
    out = _roundtrip(src)
    assert out == td("""
        # pre

        # orphan
        k = 1
    """)
    assert reparses(out) == {"k": 1}


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


def test_format_collapses_document_boundary_blank_runs() -> None:
    assert _roundtrip("\n\n\nkey=1\n\n\n") == "\nkey = 1\n\n"


def test_format_canonicalises_empty_doc_preamble() -> None:
    # Empty doc: everything lives in _trailing and surfaces as preamble.
    src = "#hello  \n#  world\n"
    assert _roundtrip(src) == "# hello\n# world\n"


def test_format_and_sort_commute_when_last_section_lacks_newline() -> None:
    src = '# Header\n\ntitle = "X"\n\n[a]\nx = 1\n\n[b]\ny = 2'
    doc = tomlrt.loads(src)
    doc.format()
    doc.sort(reverse=True)
    expected = td("""
        # Header

        title = "X"

        [b]
        y = 2

        [a]
        x = 1
    """)
    assert tomlrt.dumps(doc) == expected

    doc = tomlrt.loads(src)
    doc.sort(reverse=True)
    doc.format()
    assert tomlrt.dumps(doc) == expected


def test_format_inserts_blank_before_commented_aot_header_after_sort() -> None:
    doc = tomlrt.loads(
        td("""
        [c]
        x = 1

        # start
        [[a.hello]]
        x = 1

        [a]
        name = "x"
        """)
    )

    doc.table("a").sort()
    doc.format()

    assert tomlrt.dumps(doc) == td("""
        [c]
        x = 1

        [a]
        name = "x"

        # start
        [[a.hello]]
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


def test_multiline_item_comment_separation_collapses_to_one_blank() -> None:
    src = td("""
        arr = [
          'g',



        # attached
          'w', # inline


          # orphan


          # attached
          'a', # final inline



          # trailing
        ]
        table = {
          g = 1,



        # attached
          w = 2, # inline


          # orphan
          a = 3, # final inline



          # trailing
        }
    """)
    assert _roundtrip(src) == td("""
        arr = [
          'g',

          # attached
          'w', # inline

          # orphan

          # attached
          'a', # final inline

          # trailing
        ]
        table = {
          g = 1,

          # attached
          w = 2, # inline

          # orphan
          a = 3, # final inline

          # trailing
        }
    """)


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
