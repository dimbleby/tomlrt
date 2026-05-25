"""Tests for the comments side-channel and inline-table promotion."""

from __future__ import annotations

import pytest

import tomlrt
from _helpers import td

# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


def test_eol_comment_present() -> None:
    src = 'name = "ada"  # the lovelace\n'
    doc = tomlrt.loads(src)
    assert doc.comments["name"] == "the lovelace"
    assert "name" in doc.comments


def test_eol_comment_absent_means_key_not_in_mapping() -> None:
    src = 'name = "ada"\n'
    doc = tomlrt.loads(src)
    assert "name" not in doc.comments
    with pytest.raises(KeyError):
        _ = doc.comments["name"]


def test_eol_comment_unknown_key_raises_keyerror() -> None:
    doc = tomlrt.loads("a = 1\n")
    with pytest.raises(KeyError):
        _ = doc.comments["missing"]


def test_eol_comment_iter_yields_only_commented_keys() -> None:
    src = td("""
        a = 1  # one
        b = 2
        c = 3  # three
        """)
    doc = tomlrt.loads(src)
    assert list(doc.comments) == ["a", "c"]
    assert dict(doc.comments) == {"a": "one", "c": "three"}
    assert len(doc.comments) == 2


def test_leading_comments_present() -> None:
    src = td("""
        # a section
        # of two lines
        name = 1
        """)
    doc = tomlrt.loads(src)
    assert doc.leading_comments["name"] == ("a section", "of two lines")
    assert "name" in doc.leading_comments


def test_leading_comments_absent_raises_on_get() -> None:
    doc = tomlrt.loads("name = 1\n")
    assert "name" not in doc.leading_comments
    with pytest.raises(KeyError):
        _ = doc.leading_comments["name"]


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------


def test_set_eol_comment_on_unknown_key_raises_keyerror() -> None:
    """Setter refuses to invent a key — ``key not in container`` → KeyError."""
    doc = tomlrt.loads("a = 1\n")
    with pytest.raises(KeyError, match="missing"):
        doc.comments["missing"] = "x"


def test_set_leading_comments_on_unknown_key_raises_keyerror() -> None:
    doc = tomlrt.loads("a = 1\n")
    with pytest.raises(KeyError, match="missing"):
        doc.leading_comments["missing"] = ("x",)


def test_eol_comment_contains_non_str_returns_false() -> None:
    """``__contains__`` on a non-str key must return False, not raise."""
    doc = tomlrt.loads("a = 1  # one\n")
    assert 1 not in doc.comments  # type: ignore[comparison-overlap]
    assert None not in doc.comments
    assert (1, 2) not in doc.comments  # type: ignore[comparison-overlap]


def test_leading_comments_contains_non_str_returns_false() -> None:
    doc = tomlrt.loads("# above\na = 1\n")
    assert 1 not in doc.leading_comments  # type: ignore[comparison-overlap]
    assert object() not in doc.leading_comments


def test_set_leading_comments_non_iterable_raises_typeerror() -> None:
    doc = tomlrt.loads("a = 1\n")
    with pytest.raises(TypeError, match="iterable of comment strings"):
        doc.leading_comments["a"] = 42  # type: ignore[assignment]  # ty: ignore[invalid-assignment]


def test_set_leading_comments_element_not_str_raises_typeerror() -> None:
    doc = tomlrt.loads("a = 1\n")
    with pytest.raises(TypeError, match="entries must be strings"):
        doc.leading_comments["a"] = ("ok", 5)  # type: ignore[assignment]  # ty: ignore[invalid-assignment]


def test_preamble_element_not_str_raises_typeerror() -> None:
    doc = tomlrt.loads("a = 1\n")
    with pytest.raises(TypeError, match="entries must be strings"):
        doc.preamble = ("ok", 7)  # type: ignore[assignment]  # ty: ignore[invalid-assignment]


def test_preamble_element_with_embedded_newline_rejected() -> None:
    doc = tomlrt.loads("a = 1\n")
    with pytest.raises(tomlrt.TOMLError, match="line terminator"):
        doc.preamble = ("a\nb",)


def test_set_eol_comment_on_uncommented_key() -> None:
    doc = tomlrt.loads('name = "ada"\n')
    doc.comments["name"] = "the lovelace"
    assert tomlrt.dumps(doc) == 'name = "ada" # the lovelace\n'
    assert doc.comments["name"] == "the lovelace"


def test_set_eol_comment_replaces_existing() -> None:
    src = 'name = "ada"  # old\n'
    doc = tomlrt.loads(src)
    doc.comments["name"] = "new"
    assert tomlrt.dumps(doc) == 'name = "ada"  # new\n'


def test_set_eol_comment_to_empty_string_writes_bare_hash() -> None:
    # An empty comment string is content (a bare '#'), not a delete:
    # the API is symmetric with the reader, which returns "" for a
    # parsed bare '#'. Use ``del`` (or assign ``None`` for the header
    # variants) to actually remove a comment.
    src = 'name = "ada"  # old\n'
    doc = tomlrt.loads(src)
    doc.comments["name"] = ""
    assert tomlrt.dumps(doc) == 'name = "ada"  #\n'
    assert doc.comments["name"] == ""


def test_del_eol_comment_removes_it() -> None:
    src = 'name = "ada"  # old\n'
    doc = tomlrt.loads(src)
    del doc.comments["name"]
    assert tomlrt.dumps(doc) == 'name = "ada"\n'
    assert "name" not in doc.comments
    with pytest.raises(KeyError):
        del doc.comments["name"]


def test_set_eol_comment_round_trips_text_with_hash_prefix() -> None:
    # The API takes comment *content*, never the '#' marker. A user
    # whose content genuinely starts with '#' (e.g. "#hashtag") gets
    # exactly that back on read; the renderer prepends its own marker.
    doc = tomlrt.loads("a = 1\n")
    doc.comments["a"] = "## emphasised"
    assert tomlrt.dumps(doc) == "a = 1 # ## emphasised\n"
    assert doc.comments["a"] == "## emphasised"


def test_comment_views_are_idempotent_under_self_assignment() -> None:
    # A comment view's getter and setter must round-trip: writing back
    # what we read must be a no-op for any present key, including
    # comments whose content starts with '#'.
    src = td("""
            a = 1  # plain
            b = 2  # "quoted"
            c = 3  # #hashtag
            d = 4  # ## emphasised
            """)
    doc = tomlrt.loads(src)
    for key in ("a", "b", "c", "d"):
        doc.comments[key] = doc.comments[key]
    re = tomlrt.loads(tomlrt.dumps(doc))
    assert dict(re.comments) == dict(doc.comments)
    assert dict(doc.comments) == {
        "a": "plain",
        "b": '"quoted"',
        "c": "#hashtag",
        "d": "## emphasised",
    }


def test_empty_comment_in_source_round_trips_through_view() -> None:
    # A bare '#' (empty comment) in the source must read as ''
    # *and* be present, and writing '' back must be a no-op. The
    # ``del``-via-empty-string shortcut would have broken this.
    doc = tomlrt.loads("a = 1  #\nb = 2\n")
    assert doc.comments["a"] == ""
    assert "a" in doc.comments
    doc.comments["a"] = doc.comments["a"]
    assert tomlrt.loads(tomlrt.dumps(doc)).comments["a"] == ""


def test_set_eol_comment_rejects_newline() -> None:
    doc = tomlrt.loads("a = 1\n")
    with pytest.raises(tomlrt.TOMLError):
        doc.comments["a"] = "no\nway"


def test_set_eol_comment_rejects_non_str_value() -> None:
    """The setter validates the value type up-front."""
    doc = tomlrt.loads("a = 1\n")
    with pytest.raises(TypeError, match="must be str"):
        doc.comments["a"] = 123  # type: ignore[assignment]  # ty: ignore[invalid-assignment]


@pytest.mark.parametrize("ch", ["\x00", "\x01", "\x1f", "\x0b", "\x0c", "\x7f"])
def test_set_eol_comment_rejects_other_control_chars(ch: str) -> None:
    # Comments may only contain TAB among the control characters; any
    # other control char would be rejected by the parser on round-trip.
    # The setter must refuse them up front rather than silently produce
    # output that no longer reparses.
    doc = tomlrt.loads("a = 1\n")
    with pytest.raises(tomlrt.TOMLError):
        doc.comments["a"] = f"bad{ch}stuff"


def test_set_eol_comment_allows_tab() -> None:
    doc = tomlrt.loads("a = 1\n")
    doc.comments["a"] = "with\ttab"
    out = tomlrt.dumps(doc)
    # round-trips cleanly
    assert tomlrt.loads(out)["a"] == 1


def test_set_leading_comments_replaces_block() -> None:
    src = "# old comment\nname = 1\n"
    doc = tomlrt.loads(src)
    doc.leading_comments["name"] = ("fresh", "block")
    assert tomlrt.dumps(doc) == td("""
        # fresh
        # block
        name = 1
        """)


def test_set_leading_comments_to_empty_clears_block() -> None:
    src = td("""
        # noisy
        # preamble
        name = 1
        """)
    doc = tomlrt.loads(src)
    doc.leading_comments["name"] = ()
    assert tomlrt.dumps(doc) == "name = 1\n"
    assert "name" not in doc.leading_comments


def test_del_leading_comments_clears_block() -> None:
    src = "# above\nname = 1\n"
    doc = tomlrt.loads(src)
    del doc.leading_comments["name"]
    assert tomlrt.dumps(doc) == "name = 1\n"


def test_set_leading_comments_preserves_indent_in_subtable() -> None:
    src = td("""
        [tbl]
            # explanation
            x = 1
        """)
    doc = tomlrt.loads(src)
    tbl = doc.table("tbl")
    assert tbl.leading_comments["x"] == ("explanation",)
    tbl.leading_comments["x"] = ("replaced",)
    assert tomlrt.dumps(doc) == td("""
        [tbl]
            # replaced
            x = 1
        """)


# ---------------------------------------------------------------------------
# Bulk-shaped operations (the reason side-channels exist)
# ---------------------------------------------------------------------------


def test_dict_of_comments_round_trips_via_update() -> None:
    src = td("""
        a = 1
        b = 2
        c = 3
        """)
    doc = tomlrt.loads(src)
    doc.comments.update({"a": "first", "c": "third"})
    assert dict(doc.comments) == {"a": "first", "c": "third"}
    assert tomlrt.dumps(doc) == td("""
        a = 1 # first
        b = 2
        c = 3 # third
        """)


def test_comments_view_is_live_not_snapshot() -> None:
    doc = tomlrt.loads("a = 1  # original\n")
    view = doc.comments
    doc.comments["a"] = "updated"
    assert view["a"] == "updated"


def test_comments_view_repr_shows_pairs() -> None:
    doc = tomlrt.loads("a = 1  # one\n")
    r = repr(doc.comments)
    assert "'a': 'one'" in r


# ---------------------------------------------------------------------------
# Inline tables and promotion
# ---------------------------------------------------------------------------


def test_comments_on_inline_table_raises_with_helpful_message() -> None:
    src = 'pkg = { name = "tomlrt", version = "0.1" }\n'
    doc = tomlrt.loads(src)
    pkg = doc.table("pkg")
    with pytest.raises(tomlrt.TOMLError, match="comment API"):
        pkg.comments["name"] = "x"
    with pytest.raises(tomlrt.TOMLError, match="comment API"):
        _ = pkg.leading_comments


def test_inline_table_promotion_basic() -> None:
    src = 'pkg = { name = "tomlrt", version = "0.1" }\n'
    doc = tomlrt.loads(src)
    promoted = doc.promote_inline("pkg")
    assert isinstance(promoted, tomlrt.Table)
    assert promoted["name"] == "tomlrt"
    assert promoted["version"] == "0.1"
    assert tomlrt.dumps(doc) == td("""
        [pkg]
        name = "tomlrt"
        version = "0.1"
        """)


def test_inline_table_promotion_preserves_leading_comments() -> None:
    src = '# the package\npkg = { name = "tomlrt" }\n'
    doc = tomlrt.loads(src)
    doc.promote_inline("pkg")
    assert tomlrt.dumps(doc) == td("""
        # the package
        [pkg]
        name = "tomlrt"
        """)


def test_inline_table_promotion_preserves_eol_comment_on_header() -> None:
    src = 'pkg = { name = "tomlrt" }  # describes pkg\n'
    doc = tomlrt.loads(src)
    doc.promote_inline("pkg")
    assert tomlrt.dumps(doc) == '[pkg]  # describes pkg\nname = "tomlrt"\n'


def test_inline_promotion_then_set_comment_on_member() -> None:
    src = 'pkg = { name = "tomlrt", version = "0.1" }\n'
    doc = tomlrt.loads(src)
    promoted = doc.promote_inline("pkg")
    promoted.comments["version"] = "calver soon"
    assert tomlrt.dumps(doc) == (
        td("""
            [pkg]
            name = "tomlrt"
            version = "0.1" # calver soon
            """)
    )


def test_promote_inline_refuses_when_inner_comments_would_be_lost() -> None:
    src = td("""
        pkg = {
            # inner
            x = 1,
        }
        """)
    doc = tomlrt.loads(src)
    with pytest.raises(tomlrt.TOMLError, match="inner comments"):
        doc.promote_inline("pkg")
    # Nothing was mutated.
    assert tomlrt.dumps(doc) == src


def test_promote_inline_refuses_on_eol_comment_inside_entry() -> None:
    src = td("""
        pkg = {
            x = 1, # inner-eol
            y = 2,
        }
        """)
    doc = tomlrt.loads(src)
    with pytest.raises(tomlrt.TOMLError, match="inner comments"):
        doc.promote_inline("pkg")


def test_promote_array_refuses_when_item_eol_comment_would_be_lost() -> None:
    src = td("""
        a = [
            {x=1}, # one
            {x=2},
        ]
        """)
    doc = tomlrt.loads(src)
    with pytest.raises(tomlrt.TOMLError, match="comments that would be lost"):
        doc.promote_array("a")
    assert tomlrt.dumps(doc) == src


def test_promote_array_refuses_when_array_final_comment_would_be_lost() -> None:
    src = td("""
        a = [
            {x=1},
            # trailing
        ]
        """)
    doc = tomlrt.loads(src)
    with pytest.raises(tomlrt.TOMLError, match="comments that would be lost"):
        doc.promote_array("a")


def test_promote_array_refuses_when_inner_inline_table_has_comments() -> None:
    src = td("""
        a = [
            {
                # inner
                x = 1,
            },
        ]
        """)
    doc = tomlrt.loads(src)
    with pytest.raises(tomlrt.TOMLError, match="inner comments"):
        doc.promote_array("a")


def test_inline_promotion_inserts_after_parent_block() -> None:
    src = td("""
        [parent]
        a = 1
        pkg = { x = 10 }
        [other]
        b = 2
        """)
    doc = tomlrt.loads(src)
    parent = doc.table("parent")
    parent.promote_inline("pkg")
    # A blank line separates the parent's direct entries from the
    # promoted child header, matching ``promote_array`` and other
    # section-installing operations.
    assert tomlrt.dumps(doc) == (
        td("""
            [parent]
            a = 1

            [parent.pkg]
            x = 10
            [other]
            b = 2
            """)
    )


def test_promote_non_inline_raises() -> None:
    doc = tomlrt.loads("a = 1\n")
    with pytest.raises(tomlrt.TOMLError, match="not an inline table"):
        doc.promote_inline("a")


def test_promote_unknown_key_raises_keyerror() -> None:
    doc = tomlrt.loads("a = 1\n")
    with pytest.raises(KeyError):
        doc.promote_inline("missing")


# ---------------------------------------------------------------------------
# Inline tables reject the comment / header / promotion APIs
# ---------------------------------------------------------------------------


def _inline_in_doc() -> tomlrt.Table:
    doc = tomlrt.loads("t = { a = 1 }\n")
    return doc.table("t")


def test_inline_table_comments_raises() -> None:
    t = _inline_in_doc()
    with pytest.raises(tomlrt.TOMLError, match="comment API"):
        _ = t.comments


def test_inline_table_leading_comments_raises() -> None:
    t = _inline_in_doc()
    with pytest.raises(tomlrt.TOMLError, match="comment API"):
        _ = t.leading_comments


def test_inline_table_header_comment_get_raises() -> None:
    t = _inline_in_doc()
    with pytest.raises(tomlrt.TOMLError, match="header comment API"):
        _ = t.header_comment


def test_inline_table_header_comment_set_raises() -> None:
    t = _inline_in_doc()
    with pytest.raises(tomlrt.TOMLError, match="header comment API"):
        t.header_comment = "x"


def test_inline_table_header_comment_del_raises() -> None:
    t = _inline_in_doc()
    with pytest.raises(tomlrt.TOMLError, match="header comment API"):
        del t.header_comment


def test_inline_table_header_leading_comments_get_raises() -> None:
    t = _inline_in_doc()
    with pytest.raises(tomlrt.TOMLError, match="header comment API"):
        _ = t.header_leading_comments


def test_inline_table_header_leading_comments_set_raises() -> None:
    t = _inline_in_doc()
    with pytest.raises(tomlrt.TOMLError, match="header comment API"):
        t.header_leading_comments = ("x",)


def test_inline_table_header_leading_comments_del_raises() -> None:
    t = _inline_in_doc()
    with pytest.raises(tomlrt.TOMLError, match="header comment API"):
        del t.header_leading_comments


def test_inline_table_promote_inline_raises() -> None:
    t = _inline_in_doc()
    with pytest.raises(tomlrt.TOMLError, match="inline-table promotion"):
        t.promote_inline("a")


def test_inline_table_promote_array_raises() -> None:
    t = _inline_in_doc()
    with pytest.raises(tomlrt.TOMLError, match="array-of-tables promotion"):
        t.promote_array("a")


def test_inline_table_install_section_raises() -> None:
    doc = tomlrt.loads("t = { a = 1 }\n")
    with pytest.raises(tomlrt.TOMLError, match="not section-backed"):
        doc.install("t.x.y", 99)


# ---------------------------------------------------------------------------
# Round-trips
# ---------------------------------------------------------------------------


def test_round_trip_after_set_and_clear() -> None:
    doc = tomlrt.loads("x = 1\n")
    doc.comments["x"] = "trailing"
    doc.leading_comments["x"] = ("above",)
    out = tomlrt.dumps(doc)
    assert out == "# above\nx = 1 # trailing\n"
    again = tomlrt.loads(out)
    assert again.comments["x"] == "trailing"
    assert again.leading_comments["x"] == ("above",)


# ---------------------------------------------------------------------------
# Header comment API
# ---------------------------------------------------------------------------


def test_header_comment_present() -> None:
    src = "[server] # in DC1\nhost = 'a'\n"
    doc = tomlrt.loads(src)
    assert doc.table("server").header_comment == "in DC1"


def test_header_comment_absent() -> None:
    src = "[server]\nhost = 'a'\n"
    doc = tomlrt.loads(src)
    assert doc.table("server").header_comment is None


def test_header_comment_set_round_trips() -> None:
    doc = tomlrt.loads("[server]\nhost = 'a'\n")
    doc.table("server").header_comment = "in DC1"
    assert tomlrt.dumps(doc) == "[server] # in DC1\nhost = 'a'\n"


def test_header_comment_replace_existing() -> None:
    doc = tomlrt.loads("[server] # old\nhost = 'a'\n")
    doc.table("server").header_comment = "new"
    assert tomlrt.dumps(doc) == "[server] # new\nhost = 'a'\n"


def test_header_comment_empty_string_writes_bare_hash() -> None:
    # An empty header_comment is a bare '#', not a clear: pass None
    # (or use ``del``) to actually remove it.
    doc = tomlrt.loads("[server] # old\nhost = 'a'\n")
    doc.table("server").header_comment = ""
    assert tomlrt.dumps(doc) == "[server] #\nhost = 'a'\n"
    assert doc.table("server").header_comment == ""


def test_header_comment_clear_with_none() -> None:
    doc = tomlrt.loads("[server] # old\nhost = 'a'\n")
    doc.table("server").header_comment = None
    assert tomlrt.dumps(doc) == "[server]\nhost = 'a'\n"


def test_header_comment_del() -> None:
    doc = tomlrt.loads("[server] # old\nhost = 'a'\n")
    del doc.table("server").header_comment
    assert doc.table("server").header_comment is None


def test_header_comment_del_no_comment_preserves_trailing_whitespace() -> None:
    """Deleting a header comment that doesn't exist must be a no-op."""
    src = "[server]   \nhost = 'a'\n"
    doc = tomlrt.loads(src)
    del doc.table("server").header_comment
    assert tomlrt.dumps(doc) == src


def test_header_comment_set_none_no_comment_preserves_trailing_whitespace() -> None:
    """Same for the setter when value is None."""
    src = "[server]   \nhost = 'a'\n"
    doc = tomlrt.loads(src)
    doc.table("server").header_comment = None
    assert tomlrt.dumps(doc) == src


def test_header_leading_comments_extract_block_only() -> None:
    src = td("""
        # old archived note

        # active 1
        # active 2
        [server]
        host = 'a'
        """)
    doc = tomlrt.loads(src)
    # Only the *contiguous* block above the header counts.
    assert doc.table("server").header_leading_comments == ("active 1", "active 2")


def test_header_leading_comments_round_trip() -> None:
    src = td("""
        # above
        [server]
        host = 'a'
        """)
    doc = tomlrt.loads(src)
    assert tomlrt.dumps(doc) == src


def test_header_leading_comments_set_preserves_older_block() -> None:
    src = td("""
        # old archived note

        # active
        [server]
        host = 'a'
        """)
    doc = tomlrt.loads(src)
    doc.table("server").header_leading_comments = ("brand new",)
    out = tomlrt.dumps(doc)
    # Older blank-separated comment must remain untouched.
    assert out == (
        td("""
        # old archived note

        # brand new
        [server]
        host = 'a'
        """)
    )


def test_header_leading_comments_set_on_empty() -> None:
    doc = tomlrt.loads("[server]\nhost = 'a'\n")
    doc.table("server").header_leading_comments = ("hello", "world")
    assert tomlrt.dumps(doc) == td("""
        # hello
        # world
        [server]
        host = 'a'
        """)


def test_header_leading_comments_clear_with_empty_tuple() -> None:
    doc = tomlrt.loads(
        td("""
        # above
        [server]
        host = 'a'
        """)
    )
    doc.table("server").header_leading_comments = ()
    assert tomlrt.dumps(doc) == "[server]\nhost = 'a'\n"


def test_header_leading_comments_del() -> None:
    doc = tomlrt.loads(
        td("""
        # above
        [server]
        host = 'a'
        """)
    )
    del doc.table("server").header_leading_comments
    assert tomlrt.dumps(doc) == "[server]\nhost = 'a'\n"


def test_header_comment_on_aot_entry() -> None:
    src = td("""
        [[items]]
        name = 'a'

        [[items]]
        name = 'b'
        """)
    doc = tomlrt.loads(src)
    items = doc.aot("items")
    items[0].header_comment = "first"
    items[1].header_leading_comments = ("about the second",)
    out = tomlrt.dumps(doc)
    assert out == (
        td("""
            [[items]] # first
            name = 'a'

            # about the second
            [[items]]
            name = 'b'
            """)
    )


def test_header_comment_on_document_returns_empty() -> None:
    # The document root has no header line; mirror the implicit-section
    # behaviour: getters yield the empty state, setters still raise.
    doc = tomlrt.loads("a = 1\n")
    assert doc.header_comment is None
    assert doc.header_leading_comments == ()
    assert doc.header_leading_block == ()
    with pytest.raises(tomlrt.TOMLError):
        doc.header_comment = "x"
    with pytest.raises(tomlrt.TOMLError):
        doc.header_leading_comments = ("x",)
    with pytest.raises(tomlrt.TOMLError):
        doc.header_leading_block = ("x",)


def test_header_comment_on_inline_table_raises() -> None:
    doc = tomlrt.loads("a = { x = 1, y = 2 }\n")
    a = doc.table("a")
    with pytest.raises(tomlrt.TOMLError):
        _ = a.header_comment
    with pytest.raises(tomlrt.TOMLError):
        _ = a.header_leading_comments


def test_header_comment_on_implicit_parent_returns_empty() -> None:
    # `parent` exists logically but has no `[parent]` section in source.
    # Getters return the empty state (None / ()); setters still raise,
    # because silently dropping a write would be a footgun.
    doc = tomlrt.loads("[parent.child]\nx = 1\n")
    parent = doc.table("parent")
    assert parent.header_comment is None
    assert parent.header_leading_comments == ()
    assert parent.header_leading_block == ()
    with pytest.raises(tomlrt.TOMLError):
        parent.header_comment = "x"
    with pytest.raises(tomlrt.TOMLError):
        parent.header_leading_comments = ("x",)
    with pytest.raises(tomlrt.TOMLError):
        parent.header_leading_block = ("x",)


# ---------------------------------------------------------------------------
# Pre-existing leading_comments bug fix: only the trailing block counts
# ---------------------------------------------------------------------------


def test_leading_comments_extract_block_only() -> None:
    src = td("""
        # old archived note

        # active 1
        # active 2
        name = 'x'
        """)
    doc = tomlrt.loads(src)
    assert doc.leading_comments["name"] == ("active 1", "active 2")


def test_leading_comments_set_preserves_older_block() -> None:
    src = td("""
        # old archived note

        # active
        name = 'x'
        """)
    doc = tomlrt.loads(src)
    doc.leading_comments["name"] = ("brand new",)
    assert tomlrt.dumps(doc) == (
        td("""
        # old archived note

        # brand new
        name = 'x'
        """)
    )


# ---------------------------------------------------------------------------
# Array element comments
# ---------------------------------------------------------------------------


def test_array_eol_comments_read_multiline() -> None:
    src = td("""
        arr = [
          1, # one
          2, # two
          3, # three
        ]
        """)
    doc = tomlrt.loads(src)
    arr = doc.array("arr")
    assert dict(arr.comments) == {0: "one", 1: "two", 2: "three"}


def test_array_eol_comment_read_last_no_trailing_comma() -> None:
    src = td("""
        arr = [
          1,
          2 # last
        ]
        """)
    doc = tomlrt.loads(src)
    arr = doc.array("arr")
    assert dict(arr.comments) == {1: "last"}


def test_array_leading_comments_read() -> None:
    src = td("""
        arr = [
          # before 0
          1,
          # before 1a
          # before 1b
          2,
        ]
        """)
    doc = tomlrt.loads(src)
    arr = doc.array("arr")
    assert dict(arr.leading_comments) == {
        0: ("before 0",),
        1: ("before 1a", "before 1b"),
    }


def test_array_round_trip_with_comments() -> None:
    src = td("""
        arr = [
          1, # one
          2, # two
        ]
        """)
    doc = tomlrt.loads(src)
    assert tomlrt.dumps(doc) == src


def test_array_set_eol_on_single_line_promotes_to_multiline() -> None:
    doc = tomlrt.loads("arr = [1, 2, 3]\n")
    arr = doc.array("arr")
    arr.comments[1] = "two"
    out = tomlrt.dumps(doc)
    re = tomlrt.loads(out)
    re_arr = re.array("arr")
    assert list(re_arr) == [1, 2, 3]
    assert dict(re_arr.comments) == {1: "two"}


def test_array_set_eol_on_last_item_no_comma_breaks_before_close() -> None:
    doc = tomlrt.loads("arr = [1, 2, 3]\n")
    arr = doc.array("arr")
    arr.comments[2] = "last"
    out = tomlrt.dumps(doc)
    # `]` must not be on the same line as the comment.
    assert "# last\n" in out
    assert out.rstrip().endswith("]")
    re = tomlrt.loads(out)
    re_arr = re.array("arr")
    assert list(re_arr) == [1, 2, 3]
    assert dict(re_arr.comments) == {2: "last"}


def test_array_set_eol_on_last_item_with_trailing_comma() -> None:
    doc = tomlrt.loads(
        td("""
        arr = [
          1,
          2,
        ]
        """)
    )
    arr = doc.array("arr")
    arr.comments[1] = "second"
    out = tomlrt.dumps(doc)
    re = tomlrt.loads(out)
    re_arr = re.array("arr")
    assert list(re_arr) == [1, 2]
    assert dict(re_arr.comments) == {1: "second"}


def test_array_replace_existing_eol_comment() -> None:
    doc = tomlrt.loads(
        td("""
        arr = [
          1, # old
          2,
        ]
        """)
    )
    arr = doc.array("arr")
    arr.comments[0] = "new"
    out = tomlrt.dumps(doc)
    assert "# new" in out
    assert "# old" not in out


def test_array_delete_eol_comment() -> None:
    doc = tomlrt.loads(
        td("""
        arr = [
          1, # one
          2,
        ]
        """)
    )
    arr = doc.array("arr")
    del arr.comments[0]
    out = tomlrt.dumps(doc)
    assert "# one" not in out
    re = tomlrt.loads(out)
    assert list(re.array("arr")) == [1, 2]


def test_array_set_leading_on_single_line_promotes_to_multiline() -> None:
    doc = tomlrt.loads("arr = [1, 2, 3]\n")
    arr = doc.array("arr")
    arr.leading_comments[1] = ("before two",)
    out = tomlrt.dumps(doc)
    re = tomlrt.loads(out)
    re_arr = re.array("arr")
    assert list(re_arr) == [1, 2, 3]
    assert dict(re_arr.leading_comments) == {1: ("before two",)}


def test_array_set_leading_on_first_item() -> None:
    doc = tomlrt.loads("arr = [1, 2, 3]\n")
    arr = doc.array("arr")
    arr.leading_comments[0] = ("first",)
    out = tomlrt.dumps(doc)
    re = tomlrt.loads(out)
    re_arr = re.array("arr")
    assert list(re_arr) == [1, 2, 3]
    assert dict(re_arr.leading_comments) == {0: ("first",)}


def test_array_set_multiple_leading_lines() -> None:
    doc = tomlrt.loads(
        td("""
        arr = [
          1,
          2,
        ]
        """)
    )
    arr = doc.array("arr")
    arr.leading_comments[1] = ("line one", "line two")
    out = tomlrt.dumps(doc)
    re = tomlrt.loads(out)
    re_arr = re.array("arr")
    assert dict(re_arr.leading_comments) == {1: ("line one", "line two")}


def test_array_delete_leading_comments() -> None:
    src = td("""
        arr = [
          # before
          1,
          2,
        ]
        """)
    doc = tomlrt.loads(src)
    arr = doc.array("arr")
    del arr.leading_comments[0]
    out = tomlrt.dumps(doc)
    assert "# before" not in out
    re = tomlrt.loads(out)
    assert list(re.array("arr")) == [1, 2]


def test_array_append_migrates_last_eol_comment() -> None:
    doc = tomlrt.loads(
        td("""
        arr = [
          1,
          2 # last
        ]
        """)
    )
    arr = doc.array("arr")
    assert dict(arr.comments) == {1: "last"}
    arr.append(3)
    out = tomlrt.dumps(doc)
    re = tomlrt.loads(out)
    re_arr = re.array("arr")
    assert list(re_arr) == [1, 2, 3]
    # The EOL must still belong to item 1, not item 2.
    assert dict(re_arr.comments) == {1: "last"}


def test_array_comments_view_contains_iter_len() -> None:
    src = td("""
        arr = [
          1, # one
          2,
          3, # three
        ]
        """)
    doc = tomlrt.loads(src)
    arr = doc.array("arr")
    assert 0 in arr.comments
    assert 1 not in arr.comments
    assert 2 in arr.comments
    assert 99 not in arr.comments
    assert sorted(arr.comments) == [0, 2]
    assert len(arr.comments) == 2


def test_array_comments_view_empty_array() -> None:
    doc = tomlrt.loads("arr = []\n")
    arr = doc.array("arr")
    assert len(arr.comments) == 0
    assert list(arr.comments) == []
    assert len(arr.leading_comments) == 0
    with pytest.raises(KeyError):
        _ = arr.comments[0]
    with pytest.raises(KeyError):
        _ = arr.leading_comments[0]


def test_array_comments_non_int_key_raises() -> None:
    doc = tomlrt.loads("arr = [1, 2]\n")
    arr = doc.array("arr")
    with pytest.raises(TypeError):
        _ = arr.comments["x"]  # type: ignore[index]  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError):
        arr.comments["x"] = "v"  # type: ignore[index]  # ty: ignore[invalid-assignment]


def test_array_comments_out_of_range_raises() -> None:
    doc = tomlrt.loads("arr = [1, 2]\n")
    arr = doc.array("arr")
    with pytest.raises(KeyError):
        arr.comments[5] = "nope"
    with pytest.raises(KeyError):
        arr.leading_comments[5] = ("nope",)
    with pytest.raises(KeyError):
        del arr.comments[5]


def test_array_comment_with_hash_prefix_round_trips() -> None:
    doc = tomlrt.loads("arr = [1, 2]\n")
    arr = doc.array("arr")
    arr.comments[0] = "#hashtag"
    re = tomlrt.loads(tomlrt.dumps(doc))
    # Content that happens to start with '#' is preserved verbatim.
    assert dict(re.array("arr").comments) == {0: "#hashtag"}


def test_array_set_value_via_indexing_preserves_eol_comment() -> None:
    src = td("""
        arr = [
          1, # one
          2, # two
        ]
        """)
    doc = tomlrt.loads(src)
    arr = doc.array("arr")
    arr[0] = 99
    re = tomlrt.loads(tomlrt.dumps(doc))
    re_arr = re.array("arr")
    assert list(re_arr) == [99, 2]
    # Comment ownership shouldn't change.
    assert dict(re_arr.comments) == {0: "one", 1: "two"}


# ---------------------------------------------------------------------------
# Typed accessors: Table.array / .table / .aot, Array.array / .table
# ---------------------------------------------------------------------------


def test_table_array_returns_array() -> None:
    doc = tomlrt.loads("xs = [1, 2]\n")
    arr = doc.array("xs")
    assert isinstance(arr, tomlrt.Array)
    arr.comments[0] = "first"
    assert "# first" in tomlrt.dumps(doc)


def test_table_table_returns_table() -> None:
    doc = tomlrt.loads("[server]\nport = 80\n")
    tbl = doc.table("server")
    assert isinstance(tbl, tomlrt.Table)
    tbl.header_comment = "production"
    assert "# production" in tomlrt.dumps(doc)


def test_table_aot_returns_aot() -> None:
    doc = tomlrt.loads(
        td("""
        [[products]]
        name = 'a'
        [[products]]
        name = 'b'
        """)
    )
    aot = doc.aot("products")
    assert isinstance(aot, tomlrt.AoT)
    assert len(aot) == 2


def test_table_array_wrong_kind_raises_typeerror() -> None:
    doc = tomlrt.loads("x = 1\n")
    with pytest.raises(TypeError, match="not an Array"):
        doc.array("x")


def test_table_table_wrong_kind_raises_typeerror() -> None:
    doc = tomlrt.loads("x = 1\n")
    with pytest.raises(TypeError, match="not a Table"):
        doc.table("x")


def test_table_aot_wrong_kind_raises_typeerror() -> None:
    doc = tomlrt.loads("x = 1\n")
    with pytest.raises(TypeError, match="not an AoT"):
        doc.aot("x")


def test_table_typed_accessors_propagate_keyerror() -> None:
    doc = tomlrt.loads("x = 1\n")
    with pytest.raises(KeyError):
        doc.array("missing")
    with pytest.raises(KeyError):
        doc.table("missing")
    with pytest.raises(KeyError):
        doc.aot("missing")


def test_array_array_returns_nested_array() -> None:
    doc = tomlrt.loads("xs = [[1, 2], [3, 4]]\n")
    inner = doc.array("xs").array(0)
    assert isinstance(inner, tomlrt.Array)
    assert list(inner) == [1, 2]


def test_array_table_returns_nested_inline_table() -> None:
    doc = tomlrt.loads("xs = [{a = 1}, {a = 2}]\n")
    tbl = doc.array("xs").table(0)
    assert isinstance(tbl, tomlrt.Table)
    assert tbl["a"] == 1


def test_array_array_wrong_kind_raises_typeerror() -> None:
    doc = tomlrt.loads("xs = [1, 2]\n")
    with pytest.raises(TypeError, match="not an Array"):
        doc.array("xs").array(0)


def test_array_table_wrong_kind_raises_typeerror() -> None:
    doc = tomlrt.loads("xs = [1, 2]\n")
    with pytest.raises(TypeError, match="not a Table"):
        doc.array("xs").table(0)


def test_table_typed_dotted_descent_through_non_table_raises() -> None:
    doc = tomlrt.loads("x = 1\n")
    with pytest.raises(TypeError, match="cannot descend into 'x'"):
        doc.table("x.y")


def test_table_get_typed_returns_default_for_missing_key() -> None:
    doc = tomlrt.loads("x = 1\n")
    assert doc.get_table("missing") is None
    assert doc.get_array("missing", "fallback") == "fallback"
    assert doc.get_aot("missing") is None


def test_table_get_typed_wrong_kind_raises_typeerror() -> None:
    doc = tomlrt.loads("x = 1\n")
    with pytest.raises(TypeError, match="not a Table"):
        doc.get_table("x")
    with pytest.raises(TypeError, match="not an Array"):
        doc.get_array("x")
    with pytest.raises(TypeError, match="not an AoT"):
        doc.get_aot("x")


def test_array_get_typed_returns_default_for_out_of_range() -> None:
    doc = tomlrt.loads("xs = [1, 2]\n")
    arr = doc.array("xs")
    assert arr.get_array(99) is None
    assert arr.get_table(99, "fallback") == "fallback"


def test_array_get_typed_wrong_kind_raises_typeerror() -> None:
    doc = tomlrt.loads("xs = [1, 2]\n")
    arr = doc.array("xs")
    with pytest.raises(TypeError, match="not an Array"):
        arr.get_array(0)
    with pytest.raises(TypeError, match="not a Table"):
        arr.get_table(0)


# ---------------------------------------------------------------------------
# Document.preamble / Document.epilogue
# ---------------------------------------------------------------------------


def test_preamble_empty_doc_set_and_get() -> None:
    doc = tomlrt.loads("")
    assert doc.preamble == ()
    doc.preamble = ("hello", "world")
    assert doc.preamble == ("hello", "world")
    assert tomlrt.dumps(doc) == "# hello\n# world\n"


def test_preamble_set_on_doc_with_content_adds_blank_separator() -> None:
    doc = tomlrt.loads("a = 1\n")
    doc.preamble = ("top",)
    assert tomlrt.dumps(doc) == td("""
        # top

        a = 1
        """)
    assert doc.preamble == ("top",)


def test_preamble_distinguishes_attached_leading_comment() -> None:
    """A comment immediately above the first key is leading, not preamble."""
    doc = tomlrt.loads("# attached\nkey = 1\n")
    assert doc.preamble == ()
    assert doc.leading_comments["key"] == ("attached",)


def test_preamble_set_preserves_attached_leading_comment() -> None:
    doc = tomlrt.loads("# attached\nkey = 1\n")
    doc.preamble = ("preamble",)
    assert tomlrt.dumps(doc) == td("""
        # preamble

        # attached
        key = 1
        """)
    assert doc.preamble == ("preamble",)
    assert doc.leading_comments["key"] == ("attached",)


def test_preamble_blank_separated_from_attached() -> None:
    doc = tomlrt.loads(
        td("""
        # pre

        # attached
        key = 1
        """)
    )
    assert doc.preamble == ("pre",)
    assert doc.leading_comments["key"] == ("attached",)


def test_preamble_works_when_doc_starts_with_section_header() -> None:
    doc = tomlrt.loads("[t]\nx = 1\n")
    doc.preamble = ("hi",)
    assert tomlrt.dumps(doc) == td("""
        # hi

        [t]
        x = 1
        """)


def test_preamble_delete() -> None:
    doc = tomlrt.loads(
        td("""
        # pre

        key = 1
        """)
    )
    doc.preamble = ()
    assert tomlrt.dumps(doc) == "key = 1\n"
    assert doc.preamble == ()


def test_preamble_replace_existing() -> None:
    doc = tomlrt.loads(
        td("""
        # old

        key = 1
        """)
    )
    doc.preamble = ("new1", "new2")
    assert tomlrt.dumps(doc) == td("""
        # new1
        # new2

        key = 1
        """)


def test_epilogue_empty_doc_returns_empty() -> None:
    doc = tomlrt.loads("")
    assert doc.epilogue == ()


def test_epilogue_set_on_doc_with_content() -> None:
    doc = tomlrt.loads("a = 1\n")
    doc.epilogue = ("bye",)
    assert tomlrt.dumps(doc) == "a = 1\n# bye\n"
    assert doc.epilogue == ("bye",)


def test_epilogue_replace_existing() -> None:
    doc = tomlrt.loads("a = 1\n# old\n")
    assert doc.epilogue == ("old",)
    doc.epilogue = ("new",)
    assert tomlrt.dumps(doc) == "a = 1\n# new\n"


def test_epilogue_delete() -> None:
    doc = tomlrt.loads("a = 1\n# old\n")
    doc.epilogue = ()
    assert tomlrt.dumps(doc) == "a = 1\n"
    assert doc.epilogue == ()


def test_del_preamble_clears_block() -> None:
    """``del doc.preamble`` is equivalent to ``doc.preamble = ()``."""
    doc = tomlrt.loads(
        td("""
            # one
            # two

            a = 1
            """)
    )
    del doc.preamble
    assert tomlrt.dumps(doc) == "a = 1\n"
    assert doc.preamble == ()


def test_del_epilogue_clears_block() -> None:
    """``del doc.epilogue`` is equivalent to ``doc.epilogue = ()``."""
    doc = tomlrt.loads("a = 1\n# bye\n")
    del doc.epilogue
    assert tomlrt.dumps(doc) == "a = 1\n"
    assert doc.epilogue == ()


def test_del_preamble_on_empty_doc() -> None:
    """Empty-doc preamble lives in `_trailing`; `del` must clear it there too."""
    doc = tomlrt.loads("# only\n")
    del doc.preamble
    assert tomlrt.dumps(doc) == ""
    assert doc.preamble == ()


def test_epilogue_set_on_empty_doc_raises() -> None:
    doc = tomlrt.loads("")
    with pytest.raises(tomlrt.TOMLError, match="no structural content"):
        doc.epilogue = ("x",)


def test_preamble_and_epilogue_independent() -> None:
    doc = tomlrt.loads(
        td("""
        # top

        a = 1
        # bottom
        """)
    )
    assert doc.preamble == ("top",)
    assert doc.epilogue == ("bottom",)
    assert tomlrt.dumps(doc) == td("""
        # top

        a = 1
        # bottom
        """)


def test_preamble_round_trips_through_reparse() -> None:
    doc = tomlrt.loads("")
    doc.preamble = ("a", "b")
    doc["k"] = 1
    doc.epilogue = ("z",)
    rendered = tomlrt.dumps(doc)
    assert tomlrt.dumps(tomlrt.loads(rendered)) == rendered


def test_preamble_rejects_embedded_newline() -> None:
    doc = tomlrt.loads("")
    with pytest.raises(tomlrt.TOMLError, match="line terminator"):
        doc.preamble = ("a\nb",)


# ---------------------------------------------------------------------------
# Comment-view error & repr paths (Array + Table)
# ---------------------------------------------------------------------------


def test_table_comments_delitem_missing_raises_keyerror() -> None:
    doc = tomlrt.loads("a = 1\n")
    with pytest.raises(KeyError, match="a"):
        del doc.comments["a"]


def test_table_leading_comments_delitem_missing_raises_keyerror() -> None:
    doc = tomlrt.loads("a = 1\n")
    with pytest.raises(KeyError, match="a"):
        del doc.leading_comments["a"]


def test_table_comments_repr_lists_only_present_keys() -> None:
    doc = tomlrt.loads("a = 1  # alpha\nb = 2\n")
    body = repr(doc.comments)
    assert "'a': 'alpha'" in body
    assert "'b'" not in body


def test_array_comments_typeerror_on_non_int_key() -> None:
    doc = tomlrt.loads("xs = [1, 2, 3]\n")
    arr = doc.array("xs")
    with pytest.raises(TypeError, match="must be int"):
        _ = arr.comments["zero"]  # type: ignore[index]  # ty: ignore[invalid-argument-type]


def test_array_comments_keyerror_on_out_of_range() -> None:
    doc = tomlrt.loads("xs = [1, 2]\n")
    arr = doc.array("xs")
    with pytest.raises(KeyError):
        _ = arr.comments[5]
    with pytest.raises(KeyError):
        _ = arr.comments[-3]


def test_array_comments_negative_index_mirrors_array_indexing() -> None:
    doc = tomlrt.loads("xs = [1, 2, 3]\n")
    arr = doc.array("xs")
    arr.comments[-1] = "last"
    assert arr.comments[-1] == "last"
    assert arr.comments[2] == "last"
    assert -1 in arr.comments
    arr.leading_comments[-2] = ("middle",)
    assert arr.leading_comments[-2] == ("middle",)
    assert arr.leading_comments[1] == ("middle",)
    del arr.comments[-1]
    assert 2 not in arr.comments


def test_array_comments_delitem_missing_raises_keyerror() -> None:
    doc = tomlrt.loads("xs = [1, 2]\n")
    arr = doc.array("xs")
    with pytest.raises(KeyError):
        del arr.comments[0]


def test_array_comments_setitem_empty_string_writes_bare_hash() -> None:
    # An empty array-item comment is a bare '#', not a delete: use
    # ``del`` to actually remove it.
    doc = tomlrt.loads(
        td("""
        xs = [
          1,
          2,  # tail
        ]
        """)
    )
    arr = doc.array("xs")
    assert arr.comments[1] == "tail"
    arr.comments[1] = ""
    assert arr.comments[1] == ""
    re = tomlrt.loads(tomlrt.dumps(doc))
    assert re.array("xs").comments[1] == ""


def test_array_comments_repr_lists_only_present_indices() -> None:
    doc = tomlrt.loads("xs = [1, 2  # mid\n]\n")
    arr = doc.array("xs")
    body = repr(arr.comments)
    assert "1: 'mid'" in body
    assert "0:" not in body


def test_array_leading_comments_typeerror_on_non_int_key() -> None:
    doc = tomlrt.loads("xs = [1, 2]\n")
    arr = doc.array("xs")
    with pytest.raises(TypeError, match="must be int"):
        _ = arr.leading_comments["zero"]  # type: ignore[index]  # ty: ignore[invalid-argument-type]


def test_array_leading_comments_keyerror_out_of_range() -> None:
    doc = tomlrt.loads("xs = [1, 2]\n")
    arr = doc.array("xs")
    with pytest.raises(KeyError):
        _ = arr.leading_comments[5]


def test_array_leading_comments_keyerror_when_absent() -> None:
    doc = tomlrt.loads("xs = [1, 2]\n")
    arr = doc.array("xs")
    with pytest.raises(KeyError):
        _ = arr.leading_comments[1]


def test_array_leading_comments_delitem_missing_raises_keyerror() -> None:
    doc = tomlrt.loads("xs = [1, 2]\n")
    arr = doc.array("xs")
    with pytest.raises(KeyError):
        del arr.leading_comments[0]


def test_array_leading_comments_set_empty_on_no_comments_is_noop() -> None:
    """Setting ``()`` when no leading comments exist must not raise.

    The setter still runs ``_ensure_multiline`` (so an inline array is
    promoted), but the no-op delete path is the branch we care about.
    """
    doc = tomlrt.loads(
        td("""
            xs = [
              1,
              2,
            ]
            """)
    )
    arr = doc.array("xs")
    arr.leading_comments[0] = ()
    assert 0 not in arr.leading_comments
    assert tomlrt.dumps(doc) == td("""
        xs = [
          1,
          2,
        ]
        """)


def test_array_eol_comment_getitem_raises_keyerror_when_absent() -> None:
    """Index exists but item has no EOL: KeyError, not None / IndexError."""
    doc = tomlrt.loads("xs = [1, 2]\n")
    arr = doc.array("xs")
    with pytest.raises(KeyError):
        _ = arr.comments[0]


def test_array_leading_comments_set_on_inline_array_synthesises_indent() -> None:
    """Setting leading comments on a single-line inline array promotes it.

    The array has no pre-existing pad (header_trivia is empty) so the
    setter must synthesise ``[NL, WS(indent)]`` rather than copy from
    a non-existent template.
    """
    doc = tomlrt.loads("xs = [1, 2]\n")
    arr = doc.array("xs")
    arr.leading_comments[1] = ("about two",)
    out = tomlrt.dumps(doc)
    # Round-trips and the comment is recovered.
    assert tomlrt.loads(out).array("xs").leading_comments[1] == ("about two",)


def test_array_leading_comments_delitem_after_prior_eol() -> None:
    """Regression for #122: ``del arr.leading_comments[i]`` for i > 0.

    When the prior item carries an EOL comment, the structural newline
    is hoisted into its ``post_comma_trivia``, so ``items[i].leading``
    lacks a leading NL. The mutation path must handle that shape.
    """
    src = td("""
        arr = [
            # leading z
            "z",
            "a", # eol a
            # leading m
            "M",
        ]
        """)
    doc = tomlrt.loads(src)
    arr = doc.array("arr")
    del arr.leading_comments[2]
    assert 2 not in arr.leading_comments
    assert tomlrt.dumps(doc) == td("""
        arr = [
            # leading z
            "z",
            "a", # eol a
            "M",
        ]
        """)


def test_array_leading_comments_setitem_after_prior_eol() -> None:
    """Regression for #122: setting ``leading_comments[i]`` for i > 0
    when item ``i-1`` has an EOL must not duplicate the indent or
    corrupt the surrounding shape.
    """
    src = td("""
        arr = [
            "a", # eol a
            "b",
        ]
        """)
    doc = tomlrt.loads(src)
    arr = doc.array("arr")
    arr.leading_comments[1] = ("about b",)
    assert tomlrt.dumps(doc) == td("""
        arr = [
            "a", # eol a
            # about b
            "b",
        ]
        """)


def test_array_leading_comments_clear_after_prior_eol() -> None:
    """Regression for #122: ``.clear()`` must terminate (no infinite
    loop from a silently-failing ``__delitem__``).
    """
    src = td("""
        arr = [
            # leading z
            "z",
            "a", # eol a
            # leading m
            "M",
        ]
        """)
    doc = tomlrt.loads(src)
    arr = doc.array("arr")
    arr.leading_comments.clear()
    assert dict(arr.leading_comments) == {}
    assert tomlrt.dumps(doc) == td("""
        arr = [
            "z",
            "a", # eol a
            "M",
        ]
        """)


def test_array_eol_comment_del_on_last_no_comma_item() -> None:
    """Deleting an EOL on a trailing item without a comma needs no NL restore."""
    src = td("""
        xs = [
          1,
          2  # bye
        ]
        """)
    doc = tomlrt.loads(src)
    arr = doc.array("xs")
    assert arr.comments[1] == "bye"
    del arr.comments[1]
    assert 1 not in arr.comments
    # Round-trips through the parser.
    assert tomlrt.loads(tomlrt.dumps(doc)) == {"xs": [1, 2]}


def test_array_eol_comment_set_on_internal_item_strips_structural_newline() -> None:
    """Setting an EOL on a multiline item replaces the row's structural newline.

    The synthesised EOL carries its own newline; the structural newline
    that previously terminated the row must be dropped or the row would
    render with a blank line after the comment.
    """
    src = td("""
        xs = [
          1,
          2,
        ]
        """)
    doc = tomlrt.loads(src)
    arr = doc.array("xs")
    arr.comments[0] = "first"
    out = tomlrt.dumps(doc)
    assert out == td("""
        xs = [
          1, # first
          2,
        ]
        """)


def test_array_leading_comments_repr_lists_only_present_indices() -> None:
    doc = tomlrt.loads(
        td("""
        xs = [
          # first
          1,
          2,
        ]
        """)
    )
    arr = doc.array("xs")
    body = repr(arr.leading_comments)
    assert "0:" in body
    assert "['first']" in body


def test_array_comments_on_last_no_comma_forces_bracket_to_new_line() -> None:
    doc = tomlrt.loads("xs = [1, 2]\n")
    arr = doc.array("xs")
    arr.comments[1] = "tail"
    out = tomlrt.dumps(doc)
    # ] must drop to its own line so the EOL comment doesn't swallow it.
    assert "# tail\n]" in out


def test_leading_comments_setter_rejects_str() -> None:
    """A bare ``str`` is technically a ``Sequence[str]`` of single chars
    in Python; passing one to a leading-comments setter would silently
    iterate it character-by-character and produce a stack of
    one-character ``# x`` lines. Refuse it instead."""
    doc = tomlrt.loads("[a]\nx = 1\n")
    with pytest.raises(TypeError, match="iterable of comment strings"):
        doc["a"].leading_comments["x"] = "# above"


def test_preamble_setter_rejects_str() -> None:
    """Same str-as-Sequence footgun applies to the document preamble."""
    doc = tomlrt.loads("x = 1\n")
    with pytest.raises(TypeError, match="iterable of comment strings"):
        doc.preamble = "# top"  # type: ignore[assignment]  # ty: ignore[invalid-assignment]


def test_header_leading_comments_setter_rejects_str() -> None:
    doc = tomlrt.loads("[a]\nx = 1\n")
    with pytest.raises(TypeError, match="iterable of comment strings"):
        doc["a"].header_leading_comments = "# above"


def test_epilogue_setter_rejects_str() -> None:
    doc = tomlrt.loads("x = 1\n")
    with pytest.raises(TypeError, match="iterable of comment strings"):
        doc.epilogue = "# bottom"  # type: ignore[assignment]  # ty: ignore[invalid-assignment]


# ---------------------------------------------------------------------------
# AoT mutations preserve source-table trivia
# ---------------------------------------------------------------------------


def test_aot_append_preserves_source_table_comments() -> None:
    """``aot.append(other_aot[i])`` carries over per-KV trivia and any
    sub-section structure from the source entry."""
    src = tomlrt.loads(
        "[[a]]\n# inner-leading\nx = 1  # inner-eol\ny = 2\n"
        "\n[a.sub]\n# sub-leading\nz = 3\n",
    )
    dst = tomlrt.loads("[[a]]\nfirst = 0\n")
    dst.aot("a").append(src.aot("a")[0])
    out = tomlrt.dumps(dst)
    assert "# inner-leading" in out
    assert "# inner-eol" in out
    assert "# sub-leading" in out
    assert "[a.sub]" in out


def test_aot_insert_preserves_source_table_comments() -> None:
    src = tomlrt.loads(
        td("""
        [[b]]
        # top
        q = 1  # eol
        """)
    )
    dst = tomlrt.loads("[[b]]\nx = 0\n")
    dst.aot("b").insert(0, src.aot("b")[0])
    out = tomlrt.dumps(dst)
    assert "# top" in out
    assert "# eol" in out
    # Original entry survived too
    assert "x = 0" in out


def test_aot_setitem_preserves_source_table_and_slot_leading() -> None:
    """``aot[i] = other_aot[j]`` must preserve the source's per-KV trivia
    and the destination slot's header leading (the comments above the
    original ``[[path]]`` line)."""
    src = tomlrt.loads(
        td("""
        [[a]]
        # inner
        x = 1  # eol
        """)
    )
    dst = tomlrt.loads(
        td("""
        # slot-leading
        [[a]]
        old = 1
        """)
    )
    dst.aot("a")[0] = src.aot("a")[0]
    out = tomlrt.dumps(dst)
    assert "# slot-leading" in out
    assert "# inner" in out
    assert "# eol" in out
    assert "old" not in out


def test_aot_append_same_doc_duplicates_with_comments() -> None:
    """Same-document duplication via ``append`` must clone (not alias)
    and preserve comments on both copies."""
    doc = tomlrt.loads(
        td("""
        [[a]]
        # c1
        x = 1  # c2
        """)
    )
    doc.aot("a").append(doc.aot("a")[0])
    out = tomlrt.dumps(doc)
    assert out.count("# c1") == 2
    assert out.count("# c2") == 2


def test_aot_append_std_section_table_preserves_comments() -> None:
    """The source can be any ``Table`` view, not just an AoT entry."""
    src = tomlrt.loads(
        td("""
        [s]
        # leading
        k = 1  # eol
        """)
    )
    dst = tomlrt.loads("[[a]]\nx = 0\n")
    dst.aot("a").append(src["s"])
    out = tomlrt.dumps(dst)
    assert "# leading" in out
    assert "# eol" in out


def test_aot_entry_assigned_as_std_table_renders_as_table_header() -> None:
    """``doc[k] = aot[i]`` must produce a ``[k]`` header, not ``[[k]]``."""
    src = tomlrt.loads(
        td("""
        [[a]]
        # c
        x = 1  # eol
        """)
    )
    dst = tomlrt.loads("")
    dst["t"] = src.aot("a")[0]
    out = tomlrt.dumps(dst)
    assert "[t]" in out
    assert "[[t]]" not in out
    assert "# c" in out
    assert "# eol" in out


def test_cross_doc_aot_assignment_preserves_subsections() -> None:
    """``dst[k] = src.aot(k)`` carries over each entry's owned
    sub-sections (``[k.sub]`` etc.) and their data, not just the
    ``[[k]]`` headers."""
    src = tomlrt.loads(
        td("""
            [[a]]
            # leading
            x = 1
            [a.sub]
            # nested
            y = 2
            [[a]]
            z = 3
            """),
    )
    dst = tomlrt.loads("")
    dst["a"] = src.aot("a")
    out = tomlrt.dumps(dst)
    assert "# leading" in out
    assert "[a.sub]" in out
    assert "# nested" in out
    assert "y = 2" in out
    assert "z = 3" in out


def test_same_doc_aot_assigned_under_new_key_preserves_subsections() -> None:
    """Same-document copy under a new key must rebase sub-section paths
    too: ``[a.sub]`` becomes ``[b.sub]`` when the AoT is copied to ``b``."""
    doc = tomlrt.loads(
        td("""
        [[a]]
        x = 1
        [a.sub]
        y = 2
        """)
    )
    doc["b"] = doc.aot("a")
    out = tomlrt.dumps(doc)
    assert "[b.sub]" in out
    assert "y = 2" in out
    # Source is unchanged.
    assert "[a.sub]" in out


# ---------------------------------------------------------------------------
# Dotted-key comments are reached through the dotted-parent container.
# ---------------------------------------------------------------------------


def test_dotted_key_comments_exposed_via_dotted_parent() -> None:
    src = td("""
        [project]
        # leading comment on urls.homepage
        urls.homepage = "x"  # eol comment
        """)
    doc = tomlrt.loads(src)
    project = doc.table("project")
    # Not visible from the explicit ancestor — "urls" is implicit, not a leaf.
    assert dict(project.comments) == {}
    assert dict(project.leading_comments) == {}
    urls = project.table("urls")
    assert urls.comments["homepage"] == "eol comment"
    assert urls.leading_comments["homepage"] == ("leading comment on urls.homepage",)


def test_dotted_key_comments_write_round_trips() -> None:
    src = td("""
        [project]
        # old leading
        urls.homepage = "x"  # old eol
        """)
    doc = tomlrt.loads(src)
    urls = doc.table("project").table("urls")
    urls.comments["homepage"] = "new eol"
    urls.leading_comments["homepage"] = ("new leading 1", "new leading 2")
    assert tomlrt.dumps(doc) == td("""
        [project]
        # new leading 1
        # new leading 2
        urls.homepage = "x"  # new eol
        """)


def test_dotted_key_comments_delete_round_trips() -> None:
    src = td("""
        [project]
        # leading
        urls.homepage = "x"  # eol
        """)
    doc = tomlrt.loads(src)
    urls = doc.table("project").table("urls")
    del urls.comments["homepage"]
    del urls.leading_comments["homepage"]
    assert tomlrt.dumps(doc) == td("""
        [project]
        urls.homepage = "x"
        """)


def test_dotted_key_comments_only_exposed_at_immediate_parent() -> None:
    src = td("""
        [section]
        # c-comment
        a.b.c = 1  # eol-c
        """)
    doc = tomlrt.loads(src)
    section = doc.table("section")
    a = section.table("a")
    b = a.table("b")
    assert dict(section.comments) == {}
    assert dict(a.comments) == {}
    assert b.comments["c"] == "eol-c"
    assert b.leading_comments["c"] == ("c-comment",)


# ---------------------------------------------------------------------------
# leading_block / header_leading_block — full-fidelity above-blank access (#123)
# ---------------------------------------------------------------------------


def test_header_leading_block_exposes_orphan_between_sections() -> None:
    src = td("""
        [a]
        x = 1

        # orphan

        [b]
        y = 2
        """)
    doc = tomlrt.loads(src)
    assert doc["b"].header_leading_block == (None, "orphan", None)
    assert doc["b"].header_leading_comments == ()


def test_header_leading_block_round_trips_attached_and_above_blank() -> None:
    src = td("""
        [first]
        x = 1
        # above-blank line 1
        # above-blank line 2

        # attached line 1
        # attached line 2
        [a]
        x = 1
        """)
    doc = tomlrt.loads(src)
    assert doc["a"].header_leading_block == (
        "above-blank line 1",
        "above-blank line 2",
        None,
        "attached line 1",
        "attached line 2",
    )
    assert doc["a"].header_leading_comments == ("attached line 1", "attached line 2")


def test_header_leading_block_set_writes_blank_lines_faithfully() -> None:
    doc = tomlrt.loads("[a]\nx = 1\n[b]\ny = 2\n")
    doc["b"].header_leading_block = ("orphan-1", None, "orphan-2", None, "attached")
    assert tomlrt.dumps(doc) == td("""
        [a]
        x = 1
        # orphan-1

        # orphan-2

        # attached
        [b]
        y = 2
        """)


def test_header_leading_block_first_section_excludes_document_preamble() -> None:
    src = td("""
        # preamble

        [a]
        x = 1
        """)
    doc = tomlrt.loads(src)
    assert doc.preamble == ("preamble",)
    assert doc["a"].header_leading_block == ()
    # Writing through header_leading_block leaves preamble intact.
    doc["a"].header_leading_block = ("attached",)
    assert doc.preamble == ("preamble",)
    assert doc["a"].header_leading_comments == ("attached",)
    assert tomlrt.dumps(doc) == td("""
        # preamble

        # attached
        [a]
        x = 1
        """)


def test_header_leading_block_round_trip_preserves_document_preamble() -> None:
    src = td("""
        # preamble line 1
        # preamble line 2

        # section a comment
        [a]
        x = 1

        # section b comment
        [b]
        y = 2
        """)
    doc = tomlrt.loads(src)
    assert doc.preamble == ("preamble line 1", "preamble line 2")
    assert doc["a"].header_leading_block == ("section a comment",)
    assert doc["b"].header_leading_block == (None, "section b comment")
    # In-place read+write of the first section's block must not migrate
    # the document preamble into the section body.
    doc["a"].header_leading_block = doc["a"].header_leading_block
    assert tomlrt.dumps(doc) == src


def test_header_leading_block_delete_first_section_preserves_preamble() -> None:
    src = td("""
        # preamble

        # attached
        [a]
        x = 1
        """)
    doc = tomlrt.loads(src)
    assert doc["a"].header_leading_block == ("attached",)
    del doc["a"].header_leading_block
    assert doc.preamble == ("preamble",)
    assert tomlrt.dumps(doc) == td("""
        # preamble

        [a]
        x = 1
        """)


def test_leading_block_first_kv_excludes_document_preamble() -> None:
    src = td("""
        # preamble

        # attached to x
        x = 1
        y = 2
        """)
    doc = tomlrt.loads(src)
    assert doc.preamble == ("preamble",)
    assert doc.leading_block["x"] == ("attached to x",)
    # Round-trip leaves preamble intact.
    doc.leading_block["x"] = doc.leading_block["x"]
    assert tomlrt.dumps(doc) == src
    del doc.leading_block["x"]
    assert doc.preamble == ("preamble",)
    assert tomlrt.dumps(doc) == td("""
        # preamble

        x = 1
        y = 2
        """)


def test_header_leading_block_delete_clears_above_blank_and_attached() -> None:
    src = td("""
        [a]
        x = 1

        # orphan
        # attached
        [b]
        y = 2
        """)
    doc = tomlrt.loads(src)
    assert doc["b"].header_leading_block == (None, "orphan", "attached")
    del doc["b"].header_leading_block
    assert doc["b"].header_leading_block == ()
    assert tomlrt.dumps(doc) == td("""
        [a]
        x = 1
        [b]
        y = 2
        """)


def test_leading_block_on_kv_round_trips_above_blank() -> None:
    src = td("""
        x = 1

        # orphan

        # attached
        y = 2
        """)
    doc = tomlrt.loads(src)
    assert doc.leading_block["y"] == (None, "orphan", None, "attached")
    assert doc.leading_comments["y"] == ("attached",)


def test_leading_block_setitem_round_trips_orphan_and_attached() -> None:
    doc = tomlrt.loads("x = 1\ny = 2\n")
    doc.leading_block["y"] = ("orphan", None, "attached")
    assert tomlrt.dumps(doc) == td("""
        x = 1
        # orphan

        # attached
        y = 2
        """)


def test_leading_block_preserves_indent_on_nested_key() -> None:
    src = td("""
        [section]
            # original
            nested = 1
        """)
    doc = tomlrt.loads(src)
    section = doc.table("section")
    section.leading_block["nested"] = ("first", None, "second")
    assert tomlrt.dumps(doc) == td("""
        [section]
            # first

            # second
            nested = 1
        """)


def test_leading_block_absent_key_is_missing_from_view() -> None:
    doc = tomlrt.loads("x = 1\n")
    assert "x" not in doc.leading_block
    with pytest.raises(KeyError):
        _ = doc.leading_block["x"]


def test_leading_block_rejects_non_string_non_none_entries() -> None:
    doc = tomlrt.loads("x = 1\n")
    with pytest.raises(TypeError):
        doc.leading_block["x"] = ("ok", 5)  # type: ignore[assignment]  # ty: ignore[invalid-assignment]


def test_leading_block_rejects_newline_inside_entry() -> None:
    doc = tomlrt.loads("x = 1\n")
    with pytest.raises(tomlrt.TOMLError):
        doc.leading_block["x"] = ("a\nb",)


def test_header_leading_block_unavailable_on_inline_table() -> None:
    doc = tomlrt.loads("x = { a = 1 }\n")
    with pytest.raises(tomlrt.TOMLError):
        _ = doc.table("x").header_leading_block


def test_leading_block_unavailable_on_inline_table() -> None:
    doc = tomlrt.loads("x = { a = 1 }\n")
    with pytest.raises(tomlrt.TOMLError):
        _ = doc.table("x").leading_block


def test_reorder_via_block_preserves_orphan_between_sections() -> None:
    src = td("""
        [a]
        x = 1

        # orphan

        [b]
        y = 2
        """)
    doc = tomlrt.loads(src)
    a, b = doc["a"], doc["b"]
    a_block = doc["a"].header_leading_block
    b_block = doc["b"].header_leading_block
    del doc["a"]
    del doc["b"]
    doc["b"] = b
    doc["a"] = a
    # User decides to anchor the orphan above the new [a] (was below it).
    doc["b"].header_leading_block = a_block
    doc["a"].header_leading_block = b_block
    assert tomlrt.dumps(doc) == td("""
        [b]
        y = 2

        # orphan

        [a]
        x = 1
        """)


# ---------------------------------------------------------------------------
# Detached-container comment mutation: clear error message
# ---------------------------------------------------------------------------


def test_detached_table_comments_setitem_raises_clear_error() -> None:
    """Setting a comment on a detached Table should raise TOMLError, not KeyError."""
    elem = tomlrt.Table.section()
    elem["x"] = 1
    assert "x" in elem
    with pytest.raises(tomlrt.TOMLError, match="detached container"):
        elem.comments["x"] = "eol"
    with pytest.raises(tomlrt.TOMLError, match="detached container"):
        elem.leading_comments["x"] = ("above",)
    with pytest.raises(tomlrt.TOMLError, match="detached container"):
        elem.leading_block["x"] = (None, "above")


def test_detached_table_comments_delitem_raises_clear_error() -> None:
    elem = tomlrt.Table.section()
    elem["x"] = 1
    with pytest.raises(tomlrt.TOMLError, match="detached container"):
        del elem.comments["x"]
    with pytest.raises(tomlrt.TOMLError, match="detached container"):
        del elem.leading_comments["x"]
    with pytest.raises(tomlrt.TOMLError, match="detached container"):
        del elem.leading_block["x"]


def test_detached_table_comments_reads_are_forgiving() -> None:
    """Reads still return empty / None on detached containers."""
    elem = tomlrt.Table.section()
    elem["x"] = 1
    assert elem.comments.get("x") is None
    assert list(elem.comments) == []
    assert len(elem.comments) == 0
    assert "x" not in elem.comments
    assert list(elem.leading_comments) == []
    assert list(elem.leading_block) == []


def test_detached_aot_element_comments_documented_workaround() -> None:
    """Attach-first workaround from #127: detached AoT element comments."""
    aot = tomlrt.AoT()
    src = tomlrt.Table.section()
    src["x"] = 1
    aot.append(src)
    doc = tomlrt.loads("")
    doc["items"] = aot
    doc["items"][-1].comments["x"] = "eol"
    assert tomlrt.dumps(doc) == "[[items]]\nx = 1 # eol\n"
