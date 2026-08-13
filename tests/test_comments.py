"""Tests for the comments side-channel and inline-table promotion."""

from __future__ import annotations

import pytest

import tomlrt
from _helpers import reparses, td

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
    with pytest.raises(ValueError, match="line terminator"):
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


def test_set_adjacent_eol_comment_without_final_newline() -> None:
    doc = tomlrt.loads("a = 1#old")
    doc.comments["a"] = "new"
    assert tomlrt.dumps(doc) == "a = 1 # new\n"


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


def test_del_eol_comment_with_no_preceding_whitespace_removes_it() -> None:
    """No gap-whitespace piece to drop when the comment abuts the value directly."""
    src = 'name = "ada"# old\n'
    doc = tomlrt.loads(src)
    del doc.comments["name"]
    assert tomlrt.dumps(doc) == 'name = "ada"\n'
    assert "name" not in doc.comments


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
    out = tomlrt.dumps(doc)
    assert out == src
    re = tomlrt.loads(out)
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
    out = tomlrt.dumps(doc)
    assert out == "a = 1  #\nb = 2\n"
    assert tomlrt.loads(out).comments["a"] == ""


def test_set_eol_comment_rejects_newline() -> None:
    doc = tomlrt.loads("a = 1\n")
    with pytest.raises(ValueError, match="single-line"):
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
    with pytest.raises(ValueError, match="control character"):
        doc.comments["a"] = f"bad{ch}stuff"


def test_set_eol_comment_allows_tab() -> None:
    doc = tomlrt.loads("a = 1\n")
    doc.comments["a"] = "with\ttab"
    out = tomlrt.dumps(doc)
    assert out == "a = 1 # with\ttab\n"
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


def test_del_leading_comments_preserves_older_detached_block() -> None:
    doc = tomlrt.loads(
        td("""
            first = 0

            # older

            # attached
            name = 1
            """)
    )
    del doc.leading_comments["name"]
    assert tomlrt.dumps(doc) == td("""
        first = 0

        # older

        name = 1
        """)


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


def test_comments_on_inline_table_now_supported() -> None:
    src = 'pkg = { name = "tomlrt", version = "0.1" }\n'
    doc = tomlrt.loads(src)
    pkg = doc.table("pkg")
    pkg.comments["name"] = "the package name"
    assert tomlrt.dumps(doc) == td(
        """
        pkg = {
            name = "tomlrt", # the package name
            version = "0.1",
        }
        """,
    )


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
    # The promoted header takes the parent KV's place and follows this
    # document's header-spacing convention (headers here are not
    # blank-separated), matching ``promote_array`` and other
    # section-installing operations.
    expected = td("""
        [parent]
        a = 1
        [parent.pkg]
        x = 10
        [other]
        b = 2
        """)
    assert tomlrt.dumps(doc) == expected

    array_doc = tomlrt.loads(src.replace("{ x = 10 }", "[{ x = 10 }]"))
    array_doc.table("parent").promote_array("pkg")
    assert tomlrt.dumps(array_doc) == expected.replace("[parent.pkg]", "[[parent.pkg]]")


def test_promote_non_inline_raises() -> None:
    doc = tomlrt.loads("a = 1\n")
    with pytest.raises(TypeError, match="not an inline table"):
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


def test_inline_table_comments_now_supported() -> None:
    t = _inline_in_doc()
    t.comments["a"] = "an a"
    assert dict(t.comments) == {"a": "an a"}


def test_inline_table_leading_comments_now_supported() -> None:
    t = _inline_in_doc()
    t.leading_comments["a"] = ("note",)
    assert dict(t.leading_comments) == {"a": ("note",)}


def test_retained_inline_comment_view_tracks_structural_mutations() -> None:
    doc = tomlrt.loads(
        td("""
        t = {
            a = 1, # an a
            b = 2, # a b
            c = 3, # a c
        }
        """)
    )
    t = doc.table("t")
    comments = t.comments
    assert dict(comments) == {"a": "an a", "b": "a b", "c": "a c"}

    t.sort(reverse=True)
    assert dict(comments) == {"c": "a c", "b": "a b", "a": "an a"}

    del t["b"]
    assert dict(comments) == {"c": "a c", "a": "an a"}
    with pytest.raises(KeyError):
        _ = comments["b"]

    t["b"] = 4
    comments["b"] = "new b"
    assert dict(comments) == {"c": "a c", "a": "an a", "b": "new b"}


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


def test_inline_table_install_promotes_inline_ancestor() -> None:
    """``install`` promotes an inline-table ancestor to a section, even
    for a scalar leaf: ``t`` needs an explicit ``[t.x]`` header either
    way to hold the new ``y``, so there's nothing left to forbid.
    """
    doc = tomlrt.loads("t = { a = 1 }\n")
    doc.install("t.x.y", 99)
    assert tomlrt.dumps(doc) == td("""
        [t]
        a = 1

        [t.x]
        y = 99
        """)


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


def test_header_comment_del_with_no_preceding_whitespace_removes_it() -> None:
    """No gap-whitespace piece to drop when the comment abuts the header directly."""
    src = "[server]# old\nhost = 'a'\n"
    doc = tomlrt.loads(src)
    del doc.table("server").header_comment
    assert tomlrt.dumps(doc) == "[server]\nhost = 'a'\n"


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
    assert out == td("""
        arr = [
            1,
            2, # two
            3,
        ]
        """)
    re = tomlrt.loads(out)
    re_arr = re.array("arr")
    assert list(re_arr) == [1, 2, 3]
    assert dict(re_arr.comments) == {1: "two"}


def test_array_set_eol_on_last_item_no_comma_breaks_before_close() -> None:
    doc = tomlrt.loads("arr = [1, 2, 3]\n")
    arr = doc.array("arr")
    arr.comments[2] = "last"
    out = tomlrt.dumps(doc)
    assert out == td("""
        arr = [
            1,
            2,
            3, # last
        ]
        """)
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
    assert out == td("""
        arr = [
          1,
          2, # second
        ]
        """)
    re = tomlrt.loads(out)
    re_arr = re.array("arr")
    assert list(re_arr) == [1, 2]
    assert dict(re_arr.comments) == {1: "second"}


def test_array_set_eol_on_multiline_last_item_no_comma_no_blank_line() -> None:
    """Setting an EOL on the last item of a multi-line, no-trailing-comma
    array must not leak a blank line before ``]``.

    Regression: the strip-downstream-NL pass was gated on
    ``item.has_comma``, so for the no-comma terminal item the
    structural NL in ``value.final_trivia`` was left in place, ending
    up alongside the synthesised EOL's own NL.
    """
    doc = tomlrt.loads(
        td("""
        a = [
            1
        ]
        """)
    )
    arr = doc.array("a")
    arr.comments[0] = "hi"
    assert tomlrt.dumps(doc) == td("""
        a = [
            1 # hi
        ]
        """)


def test_array_set_eol_on_multiline_last_item_no_comma_multi_item() -> None:
    """Same fix for a multi-item array whose last item lacks a comma."""
    doc = tomlrt.loads(
        td("""
        a = [
            1,
            2
        ]
        """)
    )
    arr = doc.array("a")
    arr.comments[1] = "two"
    assert tomlrt.dumps(doc) == td("""
        a = [
            1,
            2 # two
        ]
        """)


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
    assert out == td("""
        arr = [
          1, # new
          2,
        ]
        """)


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
    assert out == td("""
        arr = [
          1,
          2,
        ]
        """)
    re = tomlrt.loads(out)
    assert list(re.array("arr")) == [1, 2]


def test_array_leading_and_trailing_comma_on_one_line_round_trips() -> None:
    # The physical line ``,2,`` carries item 0's comma (a comma-first
    # predecessor breaks before it) *and* item 1's own trailing comma.
    # The two belong to different items, so the dense line needs no
    # special handling.
    src = td("""
        a = [
          1
          ,2, # two
          3
        ]
        """)
    doc = tomlrt.loads(src)
    assert tomlrt.dumps(doc) == src
    arr = doc.array("a")
    assert dict(arr.comments) == {1: "two"}
    arr.comments[0] = "one"
    assert tomlrt.dumps(doc) == td("""
        a = [
          1 # one
          ,2, # two
          3
        ]
        """)


def test_array_comma_first_eol_comment_read() -> None:
    # A comma-first item carries has_comma=True but parks its EOL comment
    # in `trailing`, before the comma. The view must still find it.
    doc = tomlrt.loads(
        td("""
        a = [
              1 # comma is on the next line
             ,2
            ]
        """)
    )
    arr = doc.array("a")
    assert dict(arr.comments) == {0: "comma is on the next line"}


def test_array_comma_first_eol_comment_replace_keeps_layout() -> None:
    doc = tomlrt.loads(
        td("""
        a = [
              1 # comma is on the next line
             ,2
            ]
        """)
    )
    arr = doc.array("a")
    arr.comments[0] = "changed"
    assert tomlrt.dumps(doc) == td("""
        a = [
              1 # changed
             ,2
            ]
        """)


def test_array_comma_first_eol_comment_add_to_uncommented_row() -> None:
    # A comma-first row with no comment still parks its row break in
    # `trailing`, before the comma. Adding a comment must attach it to the
    # value's row there, not after the comma.
    doc = tomlrt.loads(
        td("""
        a = [
              1
             ,2
            ]
        """)
    )
    arr = doc.array("a")
    arr.comments[0] = "added"
    assert tomlrt.dumps(doc) == td("""
        a = [
              1 # added
             ,2
            ]
        """)


def test_array_comma_first_eol_comment_delete_keeps_layout() -> None:
    # Deleting a comma-first EOL must keep the row break (which lives in
    # the item's own trailing, ahead of the comma) rather than reflowing
    # the comma and the following item.
    doc = tomlrt.loads(
        td("""
        a = [
              1 # comma is on the next line
             ,2
            ]
        """)
    )
    arr = doc.array("a")
    del arr.comments[0]
    out = tomlrt.dumps(doc)
    assert out == td("""
        a = [
              1
             ,2
            ]
        """)
    assert list(tomlrt.loads(out).array("a")) == [1, 2]


def test_array_comma_first_leading_comment_read() -> None:
    # A comma-first predecessor parks the next item's above-block in its
    # own `trailing`, after the row break and ahead of the comma. The
    # leading-comment view must still attribute it to the following item.
    doc = tomlrt.loads(
        td("""
        a = [
              1
            # above two
             ,2
            ]
        """)
    )
    arr = doc.array("a")
    assert dict(arr.leading_comments) == {1: ("above two",)}


def test_array_comma_first_leading_comment_add_to_uncommented_row() -> None:
    # Adding a leading comment to a comma-first item must place it on its
    # own line ahead of the comma, not reflow the comma onto a new line.
    doc = tomlrt.loads(
        td("""
        a = [
              1
             ,2
            ]
        """)
    )
    arr = doc.array("a")
    arr.leading_comments[1] = ("above two",)
    out = tomlrt.dumps(doc)
    assert out == td("""
        a = [
              1
              # above two
             ,2
            ]
        """)
    assert list(tomlrt.loads(out).array("a")) == [1, 2]


def test_array_comma_first_leading_comment_replace_keeps_layout() -> None:
    doc = tomlrt.loads(
        td("""
        a = [
              1
            # above two
             ,2
            ]
        """)
    )
    arr = doc.array("a")
    arr.leading_comments[1] = ("changed",)
    assert tomlrt.dumps(doc) == td("""
        a = [
              1
              # changed
             ,2
            ]
        """)


def test_array_comma_first_leading_comment_delete_keeps_layout() -> None:
    doc = tomlrt.loads(
        td("""
        a = [
              1
            # above two
             ,2
            ]
        """)
    )
    arr = doc.array("a")
    del arr.leading_comments[1]
    out = tomlrt.dumps(doc)
    assert out == td("""
        a = [
              1
             ,2
            ]
        """)
    assert list(tomlrt.loads(out).array("a")) == [1, 2]


def test_array_comma_first_leading_comment_coexists_with_eol() -> None:
    # The predecessor's own EOL comment and the next item's above-block
    # share the predecessor's trailing; adding the latter must not
    # disturb the former.
    doc = tomlrt.loads(
        td("""
        a = [
              1 # eol one
             ,2
            ]
        """)
    )
    arr = doc.array("a")
    arr.leading_comments[1] = ("above two",)
    out = tomlrt.dumps(doc)
    assert out == td("""
        a = [
              1 # eol one
              # above two
             ,2
            ]
        """)
    reparsed = tomlrt.loads(out).array("a")
    assert dict(reparsed.comments) == {0: "eol one"}
    assert dict(reparsed.leading_comments) == {1: ("above two",)}


def test_array_comma_first_leading_comment_eol_predecessor_no_indent() -> None:
    # Predecessor carries an EOL comment and the comma sits at column zero,
    # so the next item's above-region is empty and its only line break is
    # the EOL section's own terminating newline. Adding the above-block must
    # reuse that break, not stack a second one.
    doc = tomlrt.loads("a = [\n      1 # eol\n,2\n]\n")
    arr = doc.array("a")
    arr.leading_comments[1] = ("above",)
    out = tomlrt.dumps(doc)
    assert out == "a = [\n      1 # eol\n      # above\n      ,2\n]\n"
    reparsed = tomlrt.loads(out).array("a")
    assert dict(reparsed.comments) == {0: "eol"}
    assert dict(reparsed.leading_comments) == {1: ("above",)}


def test_array_comma_first_leading_comment_first_item_on_open_line() -> None:
    # The first value shares the opening-bracket line, so header_trivia is
    # empty and the comma-first items sit at column zero; a leading comment
    # lines up with them rather than picking up an arbitrary indent.
    doc = tomlrt.loads(
        td(
            """
            a = [1
            ,2
            ,3]
            """,
        ),
    )
    arr = doc.array("a")
    arr.leading_comments[1] = ("above",)
    out = tomlrt.dumps(doc)
    assert out == td(
        """
        a = [1
        # above
        ,2
        ,3]
        """,
    )
    assert list(tomlrt.loads(out).array("a")) == [1, 2, 3]


def test_array_leading_comment_on_first_item_sharing_open_line() -> None:
    # Item 0 sits on the opening-bracket line, so its above-region
    # (header_trivia) is empty. A leading comment must be framed onto its
    # own line above the value, lined up at column zero with the items.
    doc = tomlrt.loads(
        td(
            """
            a = [1
            ,2
            ]
            """,
        ),
    )
    arr = doc.array("a")
    arr.leading_comments[0] = ("first",)
    out = tomlrt.dumps(doc)
    assert out == td(
        """
        a = [
        # first
        1
        ,2
        ]
        """,
    )
    reparsed = tomlrt.loads(out).array("a")
    assert list(reparsed) == [1, 2]
    assert dict(reparsed.leading_comments) == {0: ("first",)}


def test_array_first_item_leading_excludes_open_bracket_comment() -> None:
    # A comment on the opening-bracket line trails the `[`; it is framing,
    # not item 0's leading block. Reading item 0's leading must omit it, and a
    # read-write round-trip must not duplicate it.
    src = td("""
        more = [ # hdr
          # lead0
          42,
        ]
        """)
    doc = tomlrt.loads(src)
    arr = doc.array("more")
    assert arr.leading_comments[0] == ("lead0",)
    arr.leading_comments[0] = arr.leading_comments[0]
    assert tomlrt.dumps(doc) == src
    # Replacing the leading block keeps the bracket comment intact.
    arr.leading_comments[0] = ("changed",)
    assert tomlrt.dumps(doc) == td("""
        more = [ # hdr
          # changed
          42,
        ]
        """)


def test_array_delete_eol_on_multiline_last_item_no_comma_keeps_break() -> None:
    """Deleting an EOL on the last item of a multi-line, no-trailing-comma
    array must keep the structural newline before ``]``.

    Regression: the restore-downstream-NL pass was gated on
    ``item.has_comma``, so for the no-comma terminal item the
    structural NL the deleted EOL had been providing was simply
    dropped — the closing bracket collapsed onto the value's line.
    """
    doc = tomlrt.loads(
        td("""
        a = [
            1 # hi
        ]
        """)
    )
    arr = doc.array("a")
    del arr.comments[0]
    assert tomlrt.dumps(doc) == td("""
        a = [
            1
        ]
        """)


def test_array_delete_eol_on_multiline_last_item_no_comma_multi_item() -> None:
    """Same fix for a multi-item array whose last item lacks a comma."""
    doc = tomlrt.loads(
        td("""
        a = [
            1, # one
            2 # two
        ]
        """)
    )
    arr = doc.array("a")
    del arr.comments[1]
    assert tomlrt.dumps(doc) == td("""
        a = [
            1, # one
            2
        ]
        """)


def test_array_set_leading_on_single_line_promotes_to_multiline() -> None:
    doc = tomlrt.loads("arr = [1, 2, 3]\n")
    arr = doc.array("arr")
    arr.leading_comments[1] = ("before two",)
    out = tomlrt.dumps(doc)
    assert out == td("""
        arr = [
            1,
            # before two
            2,
            3,
        ]
        """)
    re = tomlrt.loads(out)
    re_arr = re.array("arr")
    assert list(re_arr) == [1, 2, 3]
    assert dict(re_arr.leading_comments) == {1: ("before two",)}


def test_array_set_empty_leading_does_not_promote_to_multiline() -> None:
    # Assigning an empty sequence means "no leading comments" — a
    # semantic delete-if-present. Earlier code unconditionally called
    # _ensure_multiline before the empty-check, so this used to
    # silently restamp a single-line array as multi-line.
    doc = tomlrt.loads("arr = [1, 2, 3]\n")
    arr = doc.array("arr")
    arr.leading_comments[0] = ()
    assert tomlrt.dumps(doc) == "arr = [1, 2, 3]\n"
    # And the same on a multi-line array — should be a no-op when the
    # item had no leading comments to begin with.
    src = td("""
        arr = [
            1,
            2,
        ]
        """)
    doc = tomlrt.loads(src)
    doc.array("arr").leading_comments[0] = ()
    assert tomlrt.dumps(doc) == src


def test_array_set_leading_on_first_item() -> None:
    doc = tomlrt.loads("arr = [1, 2, 3]\n")
    arr = doc.array("arr")
    arr.leading_comments[0] = ("first",)
    out = tomlrt.dumps(doc)
    assert out == td("""
        arr = [
            # first
            1,
            2,
            3,
        ]
        """)
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
    assert out == td("""
        arr = [
          1,
          # line one
          # line two
          2,
        ]
        """)
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
    assert out == td("""
        arr = [
          1,
          2,
        ]
        """)
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
    assert out == td("""
        arr = [
          1,
          2, # last
          3
        ]
        """)
    re = tomlrt.loads(out)
    re_arr = re.array("arr")
    assert list(re_arr) == [1, 2, 3]
    # The EOL must still belong to item 1, not item 2.
    assert dict(re_arr.comments) == {1: "last"}


@pytest.mark.parametrize("eol", ["", " # last"])
def test_array_append_adopts_dangling_comment_as_leading(eol: str) -> None:
    # A comment on its own row before "]" belongs to the item below it,
    # so an append adopts it as the new item's leading block -- whether
    # or not the previous item already closed its row with an EOL
    # comment.
    doc = tomlrt.loads(
        td(f"""
        arr = [
          1,{eol}
          # dangling
        ]
        """)
    )
    arr = doc.array("arr")
    assert dict(arr.leading_comments) == {}
    arr.append(2)
    assert dict(arr.leading_comments) == {1: ("dangling",)}
    out = tomlrt.dumps(doc)
    assert out == td(f"""
        arr = [
          1,{eol}
          # dangling
          2,
        ]
        """)
    assert dict(tomlrt.loads(out).array("arr").leading_comments) == {1: ("dangling",)}


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


def test_array_eol_setitem_out_of_range_does_not_promote_to_multiline() -> None:
    """A failed EOL assignment must not promote a single-line array.

    Regression: ``ArrayEolView.__setitem__`` called ``_ensure_multiline``
    before validating the index, so an out-of-range key raised
    ``KeyError`` *but* still left the array reformatted into multi-line
    form — a silent round-trip break behind a failed operation.
    """
    doc = tomlrt.loads("xs = [1, 2]\n")
    arr = doc.array("xs")
    with pytest.raises(KeyError):
        arr.comments[5] = "oops"
    assert tomlrt.dumps(doc) == "xs = [1, 2]\n"


def test_array_comment_with_hash_prefix_round_trips() -> None:
    doc = tomlrt.loads("arr = [1, 2]\n")
    arr = doc.array("arr")
    arr.comments[0] = "#hashtag"
    out = tomlrt.dumps(doc)
    assert out == td("""
        arr = [
            1, # #hashtag
            2,
        ]
        """)
    re = tomlrt.loads(out)
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
    out = tomlrt.dumps(doc)
    assert out == td("""
        arr = [
          99, # one
          2, # two
        ]
        """)
    re = tomlrt.loads(out)
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
    assert tomlrt.dumps(doc) == td("""
        xs = [
            1, # first
            2,
        ]
        """)


def test_table_table_returns_table() -> None:
    doc = tomlrt.loads("[server]\nport = 80\n")
    tbl = doc.table("server")
    assert isinstance(tbl, tomlrt.Table)
    tbl.header_comment = "production"
    assert tomlrt.dumps(doc) == td("""
        [server] # production
        port = 80
        """)


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


def test_epilogue_full_fidelity_round_trips_blank_separated_groups() -> None:
    # The epilogue keeps blank lines (as None), including the blank that
    # separates it from the last value, so a read-then-write is a no-op
    # (it used to collapse blank-separated groups).
    src = td("""
        x = 1

        # group one

        # group two
        """)
    doc = tomlrt.loads(src)
    assert doc.epilogue == (None, "group one", None, "group two")
    doc.epilogue = doc.epilogue
    assert tomlrt.dumps(doc) == src


def test_epilogue_set_with_blank_lines() -> None:
    doc = tomlrt.loads("x = 1\n")
    doc.epilogue = ("group one", None, "group two")
    assert tomlrt.dumps(doc) == td("""
        x = 1
        # group one

        # group two
        """)
    assert tomlrt.loads(tomlrt.dumps(doc)).epilogue == ("group one", None, "group two")


def test_epilogue_set_terminates_unterminated_last_line() -> None:
    """A last line with no newline gets one before the epilogue lands.

    Otherwise the first epilogue comment would be appended to that line
    and read back as its end-of-line comment.
    """
    doc = tomlrt.loads("x = 1")
    doc.epilogue = ("bye", "two")
    out = tomlrt.dumps(doc)
    assert out == td("""
        x = 1
        # bye
        # two
        """)
    reloaded = tomlrt.loads(out)
    assert reloaded.epilogue == ("bye", "two")
    assert dict(reloaded.comments) == {}

    # An empty epilogue adds nothing, terminator included.
    unchanged = tomlrt.loads("x = 1")
    unchanged.epilogue = ()
    assert tomlrt.dumps(unchanged) == "x = 1"


def test_epilogue_set_does_not_merge_into_unterminated_eol_comment() -> None:
    doc = tomlrt.loads("x = 1 # c")
    doc.epilogue = ("bye",)
    out = tomlrt.dumps(doc)
    assert out == td("""
        x = 1 # c
        # bye
        """)
    reloaded = tomlrt.loads(out)
    assert reloaded.epilogue == ("bye",)
    assert dict(reloaded.comments) == {"x": "c"}


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
    with pytest.raises(ValueError, match="line terminator"):
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
    out = tomlrt.dumps(doc)
    assert out == td("""
        xs = [
          1,
          2, #
        ]
        """)
    re = tomlrt.loads(out)
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
    assert out == td("""
        xs = [
            1,
            # about two
            2,
        ]
        """)
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
    out = tomlrt.dumps(doc)
    assert out == td("""
        xs = [
          1,
          2
        ]
        """)
    # Round-trips through the parser.
    assert tomlrt.loads(out) == {"xs": [1, 2]}


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
    assert repr(arr.leading_comments) == "{0: ('first',)}"


def test_array_comments_on_last_no_comma_forces_bracket_to_new_line() -> None:
    doc = tomlrt.loads("xs = [1, 2]\n")
    arr = doc.array("xs")
    arr.comments[1] = "tail"
    out = tomlrt.dumps(doc)
    assert out == td("""
        xs = [
            1,
            2, # tail
        ]
        """)


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
    assert out == td("""
        [[a]]
        first = 0

        [[a]]
        # inner-leading
        x = 1  # inner-eol
        y = 2

        [a.sub]
        # sub-leading
        z = 3
        """)


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
    assert out == td("""
        [[b]]
        # top
        q = 1  # eol

        [[b]]
        x = 0
        """)


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
    assert out == td("""
        # slot-leading
        [[a]]
        # inner
        x = 1  # eol
        """)


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
    assert out == td("""
        [[a]]
        # c1
        x = 1  # c2

        [[a]]
        # c1
        x = 1  # c2
        """)


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
    assert out == td("""
        [[a]]
        x = 0

        [[a]]
        # leading
        k = 1  # eol
        """)


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
    assert out == td("""
        [t]
        # c
        x = 1  # eol
        """)


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
    assert out == td("""
        [[a]]
        # leading
        x = 1
        [a.sub]
        # nested
        y = 2
        [[a]]
        z = 3
        """)


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
    assert out == td("""
        [[a]]
        x = 1
        [a.sub]
        y = 2
        [[b]]
        x = 1
        [b.sub]
        y = 2
        """)


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


def test_preamble_is_opening_paragraph_and_round_trips() -> None:
    # Preamble is the opening paragraph; later groups are the first
    # construct's block. Re-assigning preamble is a no-op (was lossy).
    src = td("""
        # preamble

        # orphan

        # attached
        [a]
        x = 1
        """)
    doc = tomlrt.loads(src)
    assert doc.preamble == ("preamble",)
    assert doc["a"].header_leading_block == ("orphan", None, "attached")
    doc.preamble = doc.preamble
    assert tomlrt.dumps(doc) == src


def test_comment_above_indented_first_construct_is_leading_not_preamble() -> None:
    # TOML allows leading whitespace before a key/header. A comment directly
    # above an indented first construct (no blank line) is that construct's
    # leading comment, not the preamble: the line after the comment is the
    # slot's indent, not a blank separator.
    for src in ("# comment\n  key = 1\n", "# comment\n  [a]\n  x = 1\n"):
        doc = tomlrt.loads(src)
        assert doc.preamble == ()
        assert tomlrt.dumps(doc) == src
    kv = tomlrt.loads("# comment\n  key = 1\n")
    assert kv.leading_comments["key"] == ("comment",)


def test_blank_block_under_preamble_round_trips_without_drift() -> None:
    # An existing preamble's blank line shields the first construct's block,
    # so a blank-bearing block there round-trips exactly.
    doc = tomlrt.loads(
        td("""
        # license

        [a]
        x = 1
        """)
    )
    doc["a"].header_leading_block = ("orphan", None, "attached")
    out = tomlrt.dumps(doc)
    assert out == td("""
        # license

        # orphan

        # attached
        [a]
        x = 1
        """)
    reparsed = tomlrt.loads(out)
    assert reparsed.preamble == ("license",)
    assert reparsed["a"].header_leading_block == ("orphan", None, "attached")


def test_blank_block_on_first_construct_without_preamble_is_byte_idempotent() -> None:
    # With no preamble, a blank-bearing block on the first construct renders
    # fine and is byte-idempotent, but on reload its opening paragraph reads
    # back as the preamble (the two are textually identical).
    doc = tomlrt.loads("[a]\nx = 1\n")
    doc["a"].header_leading_block = ("orphan", None, "attached")
    out = tomlrt.dumps(doc)
    assert out == td("""
        # orphan

        # attached
        [a]
        x = 1
        """)
    assert tomlrt.dumps(tomlrt.loads(out)) == out
    reparsed = tomlrt.loads(out)
    assert reparsed.preamble == ("orphan",)
    assert reparsed["a"].header_leading_block == ("attached",)


def test_blank_block_on_non_head_key_round_trips() -> None:
    doc = tomlrt.loads("x = 1\ny = 2\n")
    doc.leading_block["y"] = ("orphan", None, "attached")
    assert doc.leading_block["y"] == ("orphan", None, "attached")
    assert tomlrt.dumps(doc) == td("""
        x = 1
        # orphan

        # attached
        y = 2
        """)


def test_delete_absent_leading_block_raises_keyerror() -> None:
    doc = tomlrt.loads("x = 1\n")
    with pytest.raises(KeyError, match="x"):
        del doc.leading_block["x"]


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
    with pytest.raises(ValueError, match="line terminator"):
        doc.leading_block["x"] = ("a\nb",)


def test_header_leading_block_unavailable_on_inline_table() -> None:
    doc = tomlrt.loads("x = { a = 1 }\n")
    with pytest.raises(tomlrt.TOMLError):
        _ = doc.table("x").header_leading_block


def test_array_leading_block_distinguishes_attached_comments() -> None:
    src = td("""
        x = [
          # orphan

          # attached
          1,
        ]
        """)
    doc = tomlrt.loads(src)
    arr = doc.array("x")
    assert arr.leading_block[0] == ("orphan", None, "attached")
    assert arr.leading_comments[0] == ("attached",)
    arr.leading_block[0] = arr.leading_block[0]
    assert tomlrt.dumps(doc) == src


def test_array_leading_block_excludes_preceding_row_break() -> None:
    src = "x = [1,  \n  2]\n"
    doc = tomlrt.loads(src)
    arr = doc.array("x")
    assert 1 not in arr.leading_block
    arr.leading_block[1] = ()
    assert tomlrt.dumps(doc) == src


def test_comma_leading_blocks_include_blank_after_predecessor_eol() -> None:
    src = td("""
        array = [
          1, # array eol

          # array attached
          2,
        ]
        inline = {
          a = 1, # inline eol

          # inline attached
          b = 2,
        }
        """)
    doc = tomlrt.loads(src)
    array = doc.array("array")
    inline = doc.table("inline")

    assert array.leading_block[1] == (None, "array attached")
    assert array.leading_comments[1] == ("array attached",)
    assert inline.leading_block["b"] == (None, "inline attached")
    assert inline.leading_comments["b"] == ("inline attached",)

    array.leading_block[1] = array.leading_block[1]
    inline.leading_block["b"] = inline.leading_block["b"]
    assert tomlrt.dumps(doc) == src

    del array.leading_block[1]
    del inline.leading_block["b"]
    assert tomlrt.dumps(doc) == td("""
        array = [
          1, # array eol
          2,
        ]
        inline = {
          a = 1, # inline eol
          b = 2,
        }
        """)


def test_array_leading_block_after_eol_uses_document_newline() -> None:
    src = "x = [\r\n1, # eol\r\n\r\n2\r\n]\r\n"
    doc = tomlrt.loads(src)
    arr = doc.array("x")
    assert arr.leading_block[1] == (None,)
    arr.leading_block[1] = ("attached",)
    assert tomlrt.dumps(doc) == "x = [\r\n1, # eol\r\n# attached\r\n2\r\n]\r\n"


def test_array_orphan_only_is_absent_from_leading_comments() -> None:
    doc = tomlrt.loads(
        td("""
        x = [
          # orphan

          1,
        ]
        """)
    )
    arr = doc.array("x")
    assert arr.leading_block[0] == ("orphan", None)
    assert 0 not in arr.leading_comments


def test_array_leading_comments_preserve_older_block() -> None:
    src = td("""
        x = [
          # orphan

          # old
          1,
        ]
        """)
    doc = tomlrt.loads(src)
    arr = doc.array("x")
    arr.leading_comments[0] = ("new",)
    assert tomlrt.dumps(doc) == td("""
        x = [
          # orphan

          # new
          1,
        ]
        """)
    del arr.leading_comments[0]
    assert tomlrt.dumps(doc) == td("""
        x = [
          # orphan

          1,
        ]
        """)


def test_array_leading_block_item_zero_blank_and_bracket_comment() -> None:
    src = td("""
        x = [ # bracket

          1,
        ]
        """)
    doc = tomlrt.loads(src)
    arr = doc.array("x")
    assert arr.leading_block[0] == (None,)
    assert 0 not in arr.leading_comments
    arr.leading_comments[0] = ("attached",)
    assert tomlrt.dumps(doc) == td("""
        x = [ # bracket

          # attached
          1,
        ]
        """)
    del arr.leading_block[0]
    assert tomlrt.dumps(doc) == td("""
        x = [ # bracket
          1,
        ]
        """)


def test_array_leading_block_set_promotes_and_delete_clears() -> None:
    doc = tomlrt.loads("x = [1, 2]\n")
    arr = doc.array("x")
    arr.leading_block[1] = ("orphan", None, "attached")
    assert tomlrt.dumps(doc) == td("""
        x = [
            1,
            # orphan

            # attached
            2,
        ]
        """)
    reparsed = tomlrt.loads(tomlrt.dumps(doc)).array("x")
    assert reparsed.leading_block[1] == ("orphan", None, "attached")
    assert reparsed.leading_comments[1] == ("attached",)
    del arr.leading_block[1]
    assert tomlrt.dumps(doc) == td("""
        x = [
            1,
            2,
        ]
        """)


def test_array_leading_block_empty_does_not_promote() -> None:
    doc = tomlrt.loads("x = [1, 2]\n")
    arr = doc.array("x")
    arr.leading_block[0] = ()
    assert tomlrt.dumps(doc) == "x = [1, 2]\n"
    arr.leading_block[0] = (None,)
    assert tomlrt.dumps(doc) == td("""
        x = [

            1,
            2,
        ]
        """)


def test_array_leading_block_empty_assignment_clears_existing() -> None:
    doc = tomlrt.loads(
        td("""
        x = [
            # note
            1,
        ]
        """)
    )
    arr = doc.array("x")
    arr.leading_block[0] = ()
    assert tomlrt.dumps(doc) == td("""
        x = [
            1,
        ]
        """)


def test_array_leading_block_delete_absent_raises() -> None:
    arr = tomlrt.loads("x = [\n    1,\n]\n").array("x")
    with pytest.raises(KeyError, match="0"):
        del arr.leading_block[0]


def test_array_leading_block_repr_lists_only_present_indices() -> None:
    arr = tomlrt.loads(
        td("""
        x = [
            # note

            1,
        ]
        """)
    ).array("x")
    assert repr(arr.leading_block) == "{0: ('note', None)}"


def test_array_leading_views_resolve_before_validation() -> None:
    arr = tomlrt.loads("x = [1]\n").array("x")
    with pytest.raises(KeyError, match="2"):
        arr.leading_comments[2] = "invalid"  # type: ignore[assignment]  # ty: ignore[invalid-assignment]
    with pytest.raises(KeyError, match="2"):
        arr.leading_block[2] = "invalid"  # type: ignore[assignment]  # ty: ignore[invalid-assignment]


def test_inline_leading_block_distinguishes_attached_comments() -> None:
    src = td("""
        x = {
          # orphan

          # attached
          a = 1,
        }
        """)
    doc = tomlrt.loads(src)
    table = doc.table("x")
    assert table.leading_block["a"] == ("orphan", None, "attached")
    assert table.leading_comments["a"] == ("attached",)
    table.leading_block["a"] = table.leading_block["a"]
    assert tomlrt.dumps(doc) == src


def test_inline_leading_block_excludes_preceding_row_break() -> None:
    src = "x = { a = 1,  \n  # note\n  b = 2 }\n"
    doc = tomlrt.loads(src)
    table = doc.table("x")
    assert table.leading_block["b"] == ("note",)
    table.leading_block["b"] = table.leading_block["b"]
    assert tomlrt.dumps(doc) == src


def test_inline_leading_comments_preserve_older_block() -> None:
    doc = tomlrt.loads(
        td("""
        x = {
          # orphan

          # old
          a = 1,
        }
        """)
    )
    table = doc.table("x")
    table.leading_comments["a"] = ("new",)
    assert tomlrt.dumps(doc) == td("""
        x = {
          # orphan

          # new
          a = 1,
        }
        """)
    del table.leading_comments["a"]
    assert tomlrt.dumps(doc) == td("""
        x = {
          # orphan

          a = 1,
        }
        """)


def test_inline_leading_block_set_promotes_and_delete_clears() -> None:
    doc = tomlrt.loads("x = { a = 1, b = 2 }\n")
    table = doc.table("x")
    table.leading_block["b"] = ("orphan", None, "attached")
    assert tomlrt.dumps(doc) == td("""
        x = {
            a = 1,
            # orphan

            # attached
            b = 2,
        }
        """)
    reparsed = tomlrt.loads(tomlrt.dumps(doc)).table("x")
    assert reparsed.leading_block["b"] == ("orphan", None, "attached")
    assert reparsed.leading_comments["b"] == ("attached",)
    del table.leading_block["b"]
    assert tomlrt.dumps(doc) == td("""
        x = {
            a = 1,
            b = 2,
        }
        """)


def test_inline_leading_block_comma_first_preserves_predecessor_eol() -> None:
    doc = tomlrt.loads(
        td("""
        x = {
              a = 1 # eol
             ,b = 2
        }
        """)
    )
    table = doc.table("x")
    table.leading_block["b"] = ("orphan", None, "attached")
    out = tomlrt.dumps(doc)
    assert out == td("""
        x = {
              a = 1 # eol
              # orphan

              # attached
             ,b = 2
        }
        """)
    reparsed = tomlrt.loads(out).table("x")
    assert reparsed.comments["a"] == "eol"
    assert reparsed.leading_block["b"] == ("orphan", None, "attached")


def test_array_leading_block_comma_first_eol_exposes_blank() -> None:
    src = td("""
        x = [
          1 # eol

         ,2
        ]
        """)
    doc = tomlrt.loads(src)
    arr = doc.array("x")
    assert arr.leading_block[1] == (None,)
    arr.leading_block[1] = arr.leading_block[1]
    assert tomlrt.dumps(doc) == src
    del arr.leading_block[1]
    assert tomlrt.dumps(doc) == td("""
        x = [
          1 # eol
         ,2
        ]
        """)


def test_array_leading_block_spans_comma_first_regions() -> None:
    src = td("""
        x = [
          1 # value eol
          # before comma
         , # comma eol

          # attached
          2
        ]
        """)
    doc = tomlrt.loads(src)
    arr = doc.array("x")
    assert arr.leading_block[1] == ("before comma", None, "attached")
    assert arr.leading_comments[1] == ("attached",)
    arr.leading_block[1] = arr.leading_block[1]
    assert tomlrt.dumps(doc) == src
    del arr.leading_block[1]
    assert tomlrt.dumps(doc) == td("""
        x = [
          1 # value eol
         , # comma eol
          2
        ]
        """)


def test_array_delete_post_comma_eol_preserves_comma_first_layout() -> None:
    doc = tomlrt.loads(
        td("""
        x = [
          1
         , # comma eol
          2
        ]
        """)
    )
    del doc.array("x").comments[0]
    assert tomlrt.dumps(doc) == td("""
        x = [
          1
         ,
          2
        ]
        """)


def test_array_delete_terminal_eol_then_append_keeps_conventional_layout() -> None:
    doc = tomlrt.loads(
        td("""
        x = [
          1 # terminal eol
        ]
        """)
    )
    arr = doc.array("x")
    del arr.comments[0]
    arr.append(2)
    assert tomlrt.dumps(doc) == td("""
        x = [
          1,
          2
        ]
        """)


def test_array_comma_first_leading_block_survives_insert() -> None:
    doc = tomlrt.loads(
        td("""
        x = [
          1
          # attached to two
         ,2
        ]
        """)
    )
    doc.array("x").insert(1, 9)
    assert tomlrt.dumps(doc) == td("""
        x = [
          1
         ,9
          # attached to two
         ,2
        ]
        """)


def test_array_comma_first_leading_block_survives_predecessor_delete() -> None:
    doc = tomlrt.loads(
        td("""
        x = [
          1
          # attached to two
         ,2
         ,3
        ]
        """)
    )
    del doc.array("x")[0]
    assert tomlrt.dumps(doc) == td("""
        x = [
          # attached to two
          2
         ,3
        ]
        """)


def test_comma_first_leading_blocks_follow_sorted_items() -> None:
    doc = tomlrt.loads(
        td("""
        array = [
          2
          # attached to three
         ,3
         ,1
        ]
        inline = {
          b = 2
          # attached to c
         ,c = 3
         ,a = 1
        }
        """)
    )
    doc.array("array").sort()
    doc.table("inline").sort()
    assert tomlrt.dumps(doc) == td("""
        array = [
          1
         ,2
          # attached to three
         ,3
        ]
        inline = {
          a = 1
         ,b = 2
          # attached to c
         ,c = 3
        }
        """)


def test_sort_snapshots_leading_blocks_before_moving_eol_comments() -> None:
    doc = tomlrt.loads(
        td("""
        x = [
          2, # eol two

          # attached to three
          3,
          1,
        ]
        """)
    )
    doc.array("x").sort()
    assert tomlrt.dumps(doc) == td("""
        x = [
          1,
          2, # eol two

          # attached to three
          3,
        ]
        """)


def test_sort_realigns_successor_when_comma_break_placement_changes() -> None:
    doc = tomlrt.loads(
        td("""
        x = [
          2
         , # eol two
          3
         ,1
        ]
        """)
    )
    doc.array("x").sort()
    assert tomlrt.dumps(doc) == td("""
        x = [
          1
         ,
          2
         , # eol two
          3
        ]
        """)
    assert dict(doc.array("x").comments) == {1: "eol two"}


def test_sort_preserves_standalone_comma_row_without_creating_blank() -> None:
    doc = tomlrt.loads(
        td("""
        x = [
          2 # eol two
         ,
          3,
          1
        ]
        """)
    )
    doc.array("x").sort()
    assert tomlrt.dumps(doc) == td("""
        x = [
          1
         ,
          2, # eol two
          3
        ]
        """)
    assert dict(doc.array("x").leading_block) == {}


def test_sort_eol_replaces_destination_pre_comma_break() -> None:
    doc = tomlrt.loads("x = [2 # two\n , 1\n , 3\n]\n")
    doc.array("x").sort()
    assert tomlrt.dumps(doc) == "x = [1\n , 2 # two\n , 3\n]\n"
    assert dict(doc.array("x").leading_block) == {}


def test_sort_preserves_blank_after_pre_comma_eol() -> None:
    # The item pushed onto a row of its own takes the two-space indent of
    # the comma row -- the only row the value opens -- not a fallback.
    doc = tomlrt.loads("x = [2, 1 # one\n\n  ,3]\n")
    arr = doc.array("x")
    assert arr.leading_block[2] == (None,)
    arr.sort()
    assert tomlrt.dumps(doc) == "x = [1, # one\n  2\n\n  ,3]\n"
    assert arr.leading_block[2] == (None,)


def test_partial_sort_keeps_foreign_successor_leading_block() -> None:
    doc = tomlrt.loads(
        td("""
        t = {a.q=3, x=0, a.p=1
         , # ep
         # ay
         y=2
        }
        """)
    )
    table = doc.table("t")
    table.table("a").sort()
    out = tomlrt.dumps(doc)
    assert out == td("""
        t = {a.p=1, # ep
         x=0, a.q=3
         ,
         # ay
         y=2
        }
        """)
    reparsed = tomlrt.loads(out).table("t")
    assert reparsed.leading_block["y"] == ("ay",)
    assert reparsed.table("a").comments["p"] == "ep"


def test_comma_first_head_insert_keeps_block_with_displaced_item() -> None:
    doc = tomlrt.loads(
        td("""
        x = [
         # attached to one
         1
        ,2
        ]
        """)
    )
    doc.array("x").insert(0, 9)
    assert tomlrt.dumps(doc) == td("""
        x = [
         9
         # attached to one
        ,1
        ,2
        ]
        """)
    assert dict(doc.array("x").leading_block) == {1: ("attached to one",)}


def test_delete_head_drops_removed_items_leading_block() -> None:
    doc = tomlrt.loads(
        td("""
        x = [
          # attached to one
          1,
          2,
        ]
        """)
    )
    del doc.array("x")[0]
    assert tomlrt.dumps(doc) == td("""
        x = [
          2,
        ]
        """)


def test_delete_head_keeps_surviving_blank_block() -> None:
    doc = tomlrt.loads(
        td("""
        x = [
          1,

          2,
        ]
        """)
    )
    del doc.array("x")[0]
    assert tomlrt.dumps(doc) == td("""
        x = [

          2,
        ]
        """)
    assert doc.array("x").leading_block[0] == (None,)


def test_identity_sort_preserves_shared_rows() -> None:
    src = td("""
        x = [1, 2,
          3]
        """)
    doc = tomlrt.loads(src)
    doc.array("x").sort()
    assert tomlrt.dumps(doc) == src


def test_sort_preserves_shared_row_boundaries() -> None:
    doc = tomlrt.loads("x = [3, 1, 2\n]\n")
    doc.array("x").sort()
    assert tomlrt.dumps(doc) == "x = [1, 2, 3\n]\n"


def test_leading_comment_write_breaks_shared_row() -> None:
    """The comment and the item it broke off land at the value's indent.

    The only row this value opens is the closing bracket's, flush at
    column zero, so column zero is where both go -- the space that
    separated the two items is intra-row padding, not an indent.
    """
    doc = tomlrt.loads(
        td("""
        x = [1, 2
        ]
        """)
    )
    doc.array("x").leading_comments[1] = ("two",)
    assert tomlrt.dumps(doc) == td("""
        x = [1,
        # two
        2
        ]
        """)
    assert doc.array("x").leading_comments[1] == ("two",)


def test_sort_combines_positional_blank_with_moved_comment() -> None:
    doc = tomlrt.loads(
        td("""
        x = [
          3,

          1,
          # attached to two
          2,
        ]
        """)
    )
    doc.array("x").sort()
    assert tomlrt.dumps(doc) == td("""
        x = [
          1,

          # attached to two
          2,
          3,
        ]
        """)


def test_tail_delete_keeps_pre_comma_space_before_eol() -> None:
    doc = tomlrt.loads(
        td("""
        x = [
          1  , # eol one
          2
        ]
        """)
    )
    del doc.array("x")[1]
    assert tomlrt.dumps(doc) == td("""
        x = [
          1   # eol one
        ]
        """)


def test_tail_delete_drops_obsolete_comma_first_break() -> None:
    doc = tomlrt.loads(
        td("""
        x = [
          1
         , # eol one
          2
        ]
        """)
    )
    del doc.array("x")[1]
    assert tomlrt.dumps(doc) == td("""
        x = [
          1 # eol one
        ]
        """)


def test_tail_delete_uses_final_boundary_leading_block() -> None:
    doc = tomlrt.loads(
        td("""
        x = [
          1
          # attached to two
         ,2
          # closing
         ,
        ]
        """)
    )
    del doc.array("x")[1]
    assert tomlrt.dumps(doc) == td("""
        x = [
          1
          # closing
         ,
        ]
        """)


def test_middle_delete_preserves_attached_lane_across_comma_eol() -> None:
    doc = tomlrt.loads(
        td("""
        x = [
          3
         , # eol three
          1
          # attached to two
         ,2
        ]
        """)
    )
    del doc.array("x")[1]
    assert tomlrt.dumps(doc) == td("""
        x = [
          3
         , # eol three
          # attached to two
          2
        ]
        """)
    assert doc.array("x").leading_comments[1] == ("attached to two",)


def test_middle_delete_realigns_comma_row_follower() -> None:
    doc = tomlrt.loads("x = [1\n, 2, 3]\n")
    del doc.array("x")[1]
    assert tomlrt.dumps(doc) == "x = [1\n,3]\n"


def test_tail_delete_drops_removed_block_before_eol_classification() -> None:
    doc = tomlrt.loads(
        td("""
        x = [
          3,
          1 # eol one
          # attached to two
         ,2,
        ]
        """)
    )
    del doc.array("x")[2]
    assert tomlrt.dumps(doc) == td("""
        x = [
          3,
          1 , # eol one
        ]
        """)
    assert dict(doc.array("x").comments) == {1: "eol one"}


def test_array_leading_comments_after_zero_indent_eol_adds_no_blank() -> None:
    doc = tomlrt.loads("x = [\n1, # eol\n2\n]\n")
    arr = doc.array("x")
    arr.leading_comments[1] = ("attached",)
    assert tomlrt.dumps(doc) == "x = [\n1, # eol\n# attached\n2\n]\n"


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


# ---------------------------------------------------------------------------
# Inline-table comments (TOML 1.1)
# ---------------------------------------------------------------------------


def test_inline_eol_comment_set_promotes_to_multiline() -> None:
    doc = tomlrt.loads("t = { a = 1, b = 2 }\n")
    t = doc.table("t")
    before = t.multiline
    t.comments["a"] = "first"
    after = t.multiline
    assert before is False
    assert after is True
    assert tomlrt.dumps(doc) == td(
        """
        t = {
            a = 1, # first
            b = 2,
        }
        """,
    )


def test_inline_leading_comment_set_promotes_to_multiline() -> None:
    doc = tomlrt.loads("t = { a = 1, b = 2 }\n")
    t = doc.table("t")
    t.leading_comments["b"] = ("note one", "note two")
    assert tomlrt.dumps(doc) == td(
        """
        t = {
            a = 1,
            # note one
            # note two
            b = 2,
        }
        """,
    )


def test_inline_comment_read_roundtrips_byte_exact() -> None:
    src = td(
        """
        t = { a = 1, # eol-a
              # above-b
              b = 2 }
        """,
    )
    doc = tomlrt.loads(src)
    t = doc.table("t")
    assert tomlrt.dumps(doc) == src
    assert dict(t.comments) == {"a": "eol-a"}
    assert dict(t.leading_comments) == {"b": ("above-b",)}


def test_inline_comment_membership_and_iteration() -> None:
    doc = tomlrt.loads("t = { a = 1, b = 2, c = 3 }\n")
    t = doc.table("t")
    t.comments["b"] = "bee"
    assert "b" in t.comments
    assert "a" not in t.comments
    assert "missing" not in t.comments
    assert list(t.comments) == ["b"]
    assert len(t.comments) == 1


def test_inline_eol_comment_delete_restores_single_break() -> None:
    src = td(
        """
        t = {
            a = 1, # gone
            b = 2,
        }
        """,
    )
    doc = tomlrt.loads(src)
    t = doc.table("t")
    del t.comments["a"]
    assert tomlrt.dumps(doc) == td(
        """
        t = {
            a = 1,
            b = 2,
        }
        """,
    )


def test_inline_eol_comment_delete_on_tail_without_comma() -> None:
    src = td(
        """
        t = {
            a = 1,
            b = 2 # gone
        }
        """,
    )
    doc = tomlrt.loads(src)
    t = doc.table("t")
    del t.comments["b"]
    assert tomlrt.dumps(doc) == td(
        """
        t = {
            a = 1,
            b = 2
        }
        """,
    )


def test_inline_leading_comment_delete() -> None:
    src = td(
        """
        t = {
            a = 1,
            # bye
            b = 2,
        }
        """,
    )
    doc = tomlrt.loads(src)
    t = doc.table("t")
    del t.leading_comments["b"]
    assert tomlrt.dumps(doc) == td(
        """
        t = {
            a = 1,
            b = 2,
        }
        """,
    )


def test_inline_dotted_prefix_not_a_comment_key() -> None:
    doc = tomlrt.loads("t = { a.b = 1, a.c = 2 }\n")
    t = doc.table("t")
    assert "a" not in t.comments
    assert "a" not in t.leading_comments


def test_inline_dotted_comment_via_navigator() -> None:
    doc = tomlrt.loads("t = { a.b = 1, a.c = 2 }\n")
    inner = doc.table("t")["a"]
    inner.comments["b"] = "dotted"
    assert tomlrt.dumps(doc) == td(
        """
        t = {
            a.b = 1, # dotted
            a.c = 2,
        }
        """,
    )
    assert dict(inner.comments) == {"b": "dotted"}


def test_inline_quoted_dotted_key_comment() -> None:
    doc = tomlrt.loads('t = { "a.b" = 1, c = 2 }\n')
    t = doc.table("t")
    t.comments["a.b"] = "quoted"
    assert tomlrt.dumps(doc) == td(
        """
        t = {
            "a.b" = 1, # quoted
            c = 2,
        }
        """,
    )


def test_inline_nested_inline_table_comment() -> None:
    doc = tomlrt.loads("t = { a = { b = 1 }, c = 2 }\n")
    inner = doc.table("t").table("a")
    inner.comments["b"] = "nested"
    assert tomlrt.dumps(doc) == td(
        """
        t = { a = {
            b = 1, # nested
        }, c = 2 }
        """,
    )


def test_inline_comment_promotion_keeps_nested_value_text() -> None:
    # Setting a comment promotes the table to multi-line; that is a
    # shape change, and must not reformat the entries themselves.
    doc = tomlrt.loads("t = {a=1,  b={x=1,  y=2}}\n")
    doc.table("t").comments["a"] = "note"
    assert tomlrt.dumps(doc) == td(
        """
        t = {
            a=1, # note
            b={x=1,  y=2},
        }
        """,
    )


def test_array_comment_promotion_keeps_nested_value_text() -> None:
    doc = tomlrt.loads("a = [1,  {x=1,  y=2}]\n")
    doc.array("a").comments[0] = "note"
    assert tomlrt.dumps(doc) == td(
        """
        a = [
            1, # note
            {x=1,  y=2},
        ]
        """,
    )


def test_inline_set_multiline_then_collapse() -> None:
    doc = tomlrt.loads("t = { a = 1, b = 2 }\n")
    t = doc.table("t")
    t.set_multiline(multiline=True)
    assert tomlrt.dumps(doc) == td(
        """
        t = {
            a = 1,
            b = 2,
        }
        """,
    )
    t.set_multiline(multiline=False)
    assert tomlrt.dumps(doc) == "t = { a = 1, b = 2 }\n"


def test_inline_collapse_with_comment_raises() -> None:
    doc = tomlrt.loads("t = { a = 1, b = 2 }\n")
    t = doc.table("t")
    t.comments["a"] = "stuck"
    with pytest.raises(tomlrt.TOMLError):
        t.set_multiline(multiline=False)


def test_inline_multiline_on_non_inline_raises() -> None:
    doc = tomlrt.loads("[t]\na = 1\n")
    section = doc.table("t")
    with pytest.raises(tomlrt.TOMLError, match="only available on inline"):
        _ = section.multiline
    with pytest.raises(tomlrt.TOMLError, match="only available on inline"):
        section.set_multiline(multiline=True)


def test_inline_multiline_on_detached_factory_raises() -> None:
    t = tomlrt.Table.inline({"a": 1})
    with pytest.raises(tomlrt.TOMLError, match="detached inline"):
        _ = t.multiline


def test_inline_set_multiline_on_navigator_raises() -> None:
    doc = tomlrt.loads("t = { a.b = 1 }\n")
    inner = doc.table("t")["a"]
    with pytest.raises(tomlrt.TOMLError, match="whole inline table"):
        inner.set_multiline(multiline=True)


def test_inline_comments_detached_factory_raises() -> None:
    t = tomlrt.Table.inline({"a": 1})
    with pytest.raises(tomlrt.TOMLError, match="detached inline"):
        t.comments["a"] = "x"


def test_inline_comment_set_missing_key_raises() -> None:
    doc = tomlrt.loads("t = { a = 1 }\n")
    t = doc.table("t")
    with pytest.raises(KeyError):
        t.comments["missing"] = "x"
    # A failed set must not promote the table to multi-line.
    assert tomlrt.dumps(doc) == "t = { a = 1 }\n"


def test_inline_leading_comment_empty_assignment_is_delete() -> None:
    doc = tomlrt.loads("t = { a = 1, b = 2 }\n")
    t = doc.table("t")
    t.leading_comments["b"] = ()
    # No comments to add, so no promotion.
    assert tomlrt.dumps(doc) == "t = { a = 1, b = 2 }\n"


def test_inline_multiline_property_setter_round_trip() -> None:
    doc = tomlrt.loads("t = { a = 1, b = 2 }\n")
    t = doc.table("t")
    t.multiline = True
    assert tomlrt.dumps(doc) == td(
        """
        t = {
            a = 1,
            b = 2,
        }
        """,
    )
    # Setting to the current value is a no-op.
    t.multiline = True
    # The setter also collapses back.
    t.multiline = False
    assert tomlrt.dumps(doc) == "t = { a = 1, b = 2 }\n"


def test_array_comment_membership_wrong_type_key() -> None:
    doc = tomlrt.loads("a = [1, 2]\n")
    arr = doc.array("a")
    bad: object = "x"
    assert bad not in arr.comments
    assert bad not in arr.leading_comments


def test_inline_comment_membership_non_str_key() -> None:
    doc = tomlrt.loads("t = { a = 1 }\n")
    t = doc.table("t")
    bad: object = 5
    assert bad not in t.comments
    assert bad not in t.leading_comments


def test_inline_leading_comment_empty_assignment_clears_existing() -> None:
    src = td(
        """
        t = {
            a = 1,
            # above b
            b = 2,
        }
        """,
    )
    doc = tomlrt.loads(src)
    t = doc.table("t")
    t.leading_comments["b"] = ()
    assert tomlrt.dumps(doc) == td(
        """
        t = {
            a = 1,
            b = 2,
        }
        """,
    )


def test_inline_leading_comment_delete_absent_raises() -> None:
    doc = tomlrt.loads("t = { a = 1, b = 2 }\n")
    t = doc.table("t")
    with pytest.raises(KeyError):
        del t.leading_comments["b"]
    # Deleting a key that is not even an entry also raises.
    with pytest.raises(KeyError):
        del t.leading_comments["missing"]
    # A failed delete must not promote the table to multi-line.
    assert tomlrt.dumps(doc) == "t = { a = 1, b = 2 }\n"


def test_inline_navigator_iteration_skips_non_prefix_entries() -> None:
    doc = tomlrt.loads("u = { x.y = 1, z = 2 }\n")
    u = doc.table("u")
    u["x"].comments["y"] = "wye"
    assert list(u["x"].comments) == ["y"]
    assert len(u["x"].comments) == 1
    assert tomlrt.dumps(doc) == td(
        """
        u = {
            x.y = 1, # wye
            z = 2,
        }
        """,
    )


def test_inline_comment_set_on_already_multiline_does_not_repromote() -> None:
    src = td(
        """
        t = {
            a = 1,
            b = 2,
        }
        """,
    )
    doc = tomlrt.loads(src)
    t = doc.table("t")
    assert t.multiline is True
    t.comments["a"] = "first"
    assert tomlrt.dumps(doc) == td(
        """
        t = {
            a = 1, # first
            b = 2,
        }
        """,
    )


# The trivia-geometry branches in the shared comma-comment core only fire on
# irregular physical layouts. These tests pin them.

_COMMA_FIRST = td(
    """
    t = {
          a = 1
        , b = 2
    }
    """,
)


def test_inline_comma_first_eol_set_and_delete_round_trip() -> None:
    doc = tomlrt.loads(_COMMA_FIRST)
    t = doc.table("t")
    t.comments["a"] = "x"
    t.comments["b"] = "x"
    assert tomlrt.dumps(doc) == td(
        """
        t = {
              a = 1 # x
            , b = 2 # x
        }
        """,
    )
    del t.comments["a"]
    del t.comments["b"]
    assert tomlrt.dumps(doc) == _COMMA_FIRST


def test_inline_comma_first_leading_comment_set() -> None:
    doc = tomlrt.loads(_COMMA_FIRST)
    t = doc.table("t")
    t.leading_comments["a"] = ("lead a",)
    t.leading_comments["b"] = ("lead b",)
    assert tomlrt.dumps(doc) == td(
        """
        t = {
              # lead a
              a = 1
              # lead b
            , b = 2
        }
        """,
    )


def test_inline_eol_on_item_sharing_a_line() -> None:
    # Two items on one physical row: setting an EOL comment on the first
    # forces a break, and the item pushed onto its own line is re-indented
    # to the value's indent.
    doc = tomlrt.loads(
        td(
            """
            t = {
                a = 1, b = 2,
                c = 3,
            }
            """,
        ),
    )
    doc.table("t").comments["a"] = "x"
    assert tomlrt.dumps(doc) == td(
        """
        t = {
            a = 1, # x
            b = 2,
            c = 3,
        }
        """,
    )


def test_inline_delete_eol_keeps_existing_downstream_break() -> None:
    # Deleting the EOL comment keeps the item's own row break, so an
    # adjacent blank line below it is preserved.
    doc = tomlrt.loads(
        td(
            """
            t = {
                a = 1, # x

                b = 2,
            }
            """,
        ),
    )
    del doc.table("t").comments["a"]
    assert tomlrt.dumps(doc) == td(
        """
        t = {
            a = 1,

            b = 2,
        }
        """,
    )


def test_inline_leading_comment_on_zero_indent_item() -> None:
    # A zero-indent item whose above-region already holds a comment: the
    # replacement comment block lines up at column zero with the items.
    doc = tomlrt.loads(
        td(
            """
            t = {
            # c
            a = 1,
            b = 2,
            }
            """,
        ),
    )
    doc.table("t").leading_comments["a"] = ("L",)
    assert tomlrt.dumps(doc) == td(
        """
        t = {
        # L
        a = 1,
        b = 2,
        }
        """,
    )


def test_array_eol_trailing_ws_after_comma_no_blank_line() -> None:
    # Item 0's row ends with whitespace after its comma. Stamping an EOL
    # comment must not leave that stray whitespace behind as a blank,
    # space-only line. ("@" marks the significant trailing whitespace.)
    doc = tomlrt.loads(
        td(
            """
            a = [
              1,@@
              2,
            ]
            """,
        ).replace("@", " "),
    )
    doc["a"].comments[0] = "c"
    assert tomlrt.dumps(doc) == td(
        """
        a = [
          1, # c
          2,
        ]
        """,
    )


def test_inline_eol_trailing_ws_after_comma_no_blank_line() -> None:
    # Same trailing-whitespace gap for inline tables: the shared comma-edit
    # machinery must drop the stray whitespace with the row terminator.
    doc = tomlrt.loads(
        td(
            """
            t = {
              a = 1,@@
              b = 2,
            }
            """,
        ).replace("@", " "),
    )
    doc.table("t").comments["a"] = "c"
    assert tomlrt.dumps(doc) == td(
        """
        t = {
          a = 1, # c
          b = 2,
        }
        """,
    )


def test_array_append_trailing_ws_after_comma_no_blank_line() -> None:
    # A structural mutation maintains row breaks per boundary. Item 0's row
    # ends with whitespace after its comma; appending a new item must not be
    # fooled into treating that row as unterminated and inserting a spurious
    # blank line. The trailing whitespace is preserved (we are not editing
    # that row).
    doc = tomlrt.loads(
        td(
            """
            a = [
              1,@@
              2,
            ]
            """,
        ).replace("@", " "),
    )
    doc["a"].append(3)
    assert tomlrt.dumps(doc) == td(
        """
        a = [
          1,@@
          2,
          3,
        ]
        """,
    ).replace("@", " ")


def test_array_sort_trailing_ws_after_comma_no_blank_line() -> None:
    # Reordering items also normalises row breaks; the trailing whitespace
    # after the first row's comma must not spawn a blank line.
    doc = tomlrt.loads(
        td(
            """
            a = [
              3,@@
              1,
              2,
            ]
            """,
        ).replace("@", " "),
    )
    doc["a"].sort()
    assert tomlrt.dumps(doc) == td(
        """
        a = [
          1,@@
          2,
          3,
        ]
        """,
    ).replace("@", " ")


def test_inline_insert_trailing_ws_after_comma_no_blank_line() -> None:
    # Same structural path for inline tables: inserting a new entry must not
    # turn the trailing whitespace after the first comma into a blank line.
    doc = tomlrt.loads(
        td(
            """
            t = {
              a = 1,@@
              b = 2,
            }
            """,
        ).replace("@", " "),
    )
    doc.table("t")["c"] = 3
    assert tomlrt.dumps(doc) == td(
        """
        t = {
          a = 1,@@
          b = 2,
          c = 3,
        }
        """,
    ).replace("@", " ")


def test_array_mutation_preserves_blank_before_bracket_with_trailing_ws() -> None:
    # The closing-bracket gap (final_trivia) carries a deliberate blank line
    # after a self-terminated item (it has an EOL comment), and that blank
    # line's newline is masked behind the row's trailing indent ("  \n"). A
    # structural mutation must preserve the blank line (we are
    # format-preserving) rather than collapsing it or leaving a stray break.
    doc = tomlrt.loads(
        td(
            """
            a = [
              1,
              2, # c
            @@
            ]
            """,
        ).replace("@", " "),
    )
    doc["a"].insert(0, 0)
    assert tomlrt.dumps(doc) == td(
        """
        a = [
          0,
          1,
          2, # c
        @@
        ]
        """,
    ).replace("@", " ")


def test_array_append_preserves_blank_before_bracket() -> None:
    # A deliberate blank line before "]" (after a self-terminated item)
    # survives an append; the new item lands above the blank.
    doc = tomlrt.loads(
        td("""
        a = [
          1,
          2, # c

        ]
        """),
    )
    doc["a"].append(3)
    assert tomlrt.dumps(doc) == td("""
        a = [
          1,
          2, # c
          3,

        ]
        """)


def test_array_append_preserves_interior_blank() -> None:
    # A blank line between two existing items is untouched by an append at
    # the tail; only the tail boundary changes.
    doc = tomlrt.loads(
        td("""
        a = [
          1,

          2,
        ]
        """),
    )
    doc["a"].append(3)
    assert tomlrt.dumps(doc) == td("""
        a = [
          1,

          2,
          3,
        ]
        """)


def test_array_insert_interior_preserves_later_blank() -> None:
    doc = tomlrt.loads(
        td("""
        a = [
          1,
          2,

          3,
        ]
        """),
    )
    doc["a"].insert(1, 9)
    assert tomlrt.dumps(doc) == td("""
        a = [
          1,
          9,
          2,

          3,
        ]
        """)


def test_array_insert_head_preserves_blank_before_bracket() -> None:
    doc = tomlrt.loads(
        td("""
        a = [
          1,
          2, # c

        ]
        """),
    )
    doc["a"].insert(0, 0)
    assert tomlrt.dumps(doc) == td("""
        a = [
          0,
          1,
          2, # c

        ]
        """)


def test_array_delete_tail_preserves_blank_before_bracket() -> None:
    doc = tomlrt.loads(
        td("""
        a = [
          1, # c
          2,

        ]
        """),
    )
    del doc["a"][1]
    assert tomlrt.dumps(doc) == td("""
        a = [
          1, # c

        ]
        """)


def test_array_delete_interior_preserves_blank_at_seam() -> None:
    doc = tomlrt.loads(
        td("""
        a = [
          1,
          2,

          3,
        ]
        """),
    )
    del doc["a"][1]
    assert tomlrt.dumps(doc) == td("""
        a = [
          1,

          3,
        ]
        """)


def test_array_sort_preserves_blank_position() -> None:
    # The blank line is a positional pad: it stays at its boundary while the
    # items are permuted around it.
    doc = tomlrt.loads(
        td("""
        a = [
          3,

          1,
          2,
        ]
        """),
    )
    doc["a"].sort()
    assert tomlrt.dumps(doc) == td("""
        a = [
          1,

          2,
          3,
        ]
        """)


def test_array_delete_shared_row_predecessor_realigns_follower() -> None:
    # Deleting the predecessor of a shared-row item promotes that item to a
    # row leader: its leftover one-space inline separator is replaced by the
    # canonical row indent so it aligns with its siblings.
    doc = tomlrt.loads(
        td("""
        a = [
          1, # x
          2, 3,
        ]
        """),
    )
    del doc["a"][1]
    assert tomlrt.dumps(doc) == td("""
        a = [
          1, # x
          3,
        ]
        """)


def test_inline_table_delete_shared_row_predecessor_realigns_follower() -> None:
    # Same row-leader promotion as the array case, through the shared
    # _comma_ops primitive: deleting a shared-row entry realigns the
    # follower to the canonical row indent.
    doc = tomlrt.loads(
        td("""
        a = {
          b = 1, # x
          c = 2, d = 3,
        }
        """),
    )
    del doc["a"]["c"]
    assert tomlrt.dumps(doc) == td("""
        a = {
          b = 1, # x
          d = 3,
        }
        """)


def test_array_sort_promotes_shared_row_follower_to_leader() -> None:
    # Reorder path through the same _comma_ops primitive: sorting moves the
    # terminated `1, # x` item ahead of the former shared-row follower `2`,
    # which must be re-indented from its one-space separator to the row
    # indent (otherwise it would render ` 2,`).
    doc = tomlrt.loads(
        td("""
        a = [
          2, 1, # x
          3,
        ]
        """),
    )
    doc["a"].sort()
    assert tomlrt.dumps(doc) == td("""
        a = [
          1, # x
          2,
          3,
        ]
        """)


def test_array_sort_after_eol_delete_keeps_sibling_eol_attached() -> None:
    # Deleting an EOL comment must re-home the row break downstream (matching
    # a fresh parse) rather than leaving a bare break in the item's own EOL
    # channel. Otherwise the next reorder treats that channel as positional
    # and orphans a surviving sibling's EOL comment onto its own line, where
    # a re-parse would read it back as a leading comment.
    doc = tomlrt.loads(
        td("""
        xs = [
            3, # a
            2, # b
        ]
        """),
    )
    del doc["xs"].comments[1]
    doc["xs"].sort()
    assert tomlrt.dumps(doc) == td("""
        xs = [
            2,
            3, # a
        ]
        """)


def test_inline_table_sort_after_eol_delete_keeps_sibling_eol_attached() -> None:
    # Same re-home of the row break through the shared _comma_comments path,
    # exercised on an inline table's keyed EOL view + sort.
    doc = tomlrt.loads(
        td("""
        t = {
            b = 3, # a
            a = 2, # b
        }
        """),
    )
    del doc["t"].comments["a"]
    doc["t"].sort()
    assert tomlrt.dumps(doc) == td("""
        t = {
            a = 2,
            b = 3, # a
        }
        """)


def test_inline_table_append_preserves_blank_before_bracket() -> None:
    doc = tomlrt.loads(
        td("""
        t = {
          a = 1,
          b = 2, # c

        }
        """),
    )
    doc["t"]["e"] = 3
    assert tomlrt.dumps(doc) == td("""
        t = {
          a = 1,
          b = 2, # c
          e = 3,

        }
        """)


def test_inline_table_delete_preserves_blank_at_seam() -> None:
    doc = tomlrt.loads(
        td("""
        t = {
          a = 1,
          b = 2,

          c = 3,
        }
        """),
    )
    del doc["t"]["b"]
    assert tomlrt.dumps(doc) == td("""
        t = {
          a = 1,

          c = 3,
        }
        """)


# ---------------------------------------------------------------------------
# Comments on a value whose rows pack several items
# ---------------------------------------------------------------------------


def test_array_leading_comment_matches_a_packed_row_indent() -> None:
    """The block aligns with the value's rows, not with a post-comma pad.

    The indent is sampled the same way an append samples it: from the
    first row the value opens, never from the space that separates two
    items sharing a row.
    """
    doc = tomlrt.loads(
        td("""
        a = [1, 2,
             3, 4]
        """)
    )
    doc.array("a").leading_comments[2] = ("about three",)
    out = tomlrt.dumps(doc)
    assert out == td("""
        a = [1, 2,
             # about three
             3, 4]
        """)
    assert reparses(out) == {"a": [1, 2, 3, 4]}


def test_array_leading_comment_promotes_a_row_follower() -> None:
    """An item that shared a row joins the block at the value indent."""
    doc = tomlrt.loads(
        td("""
        a = [1, 2,
             3, 4]
        """)
    )
    doc.array("a").leading_comments[1] = ("about two",)
    out = tomlrt.dumps(doc)
    assert out == td("""
        a = [1,
             # about two
             2,
             3, 4]
        """)
    assert reparses(out) == {"a": [1, 2, 3, 4]}

    doc = tomlrt.loads(
        td("""
        a = [1, 2,
             3, 4]
        """)
    )
    doc.array("a").leading_comments[3] = ("about four",)
    assert tomlrt.dumps(doc) == td("""
        a = [1, 2,
             3,
             # about four
             4]
        """)


def test_array_leading_comment_on_first_item_of_a_packed_row() -> None:
    """The head boundary opens its row at the sampled indent too."""
    doc = tomlrt.loads(
        td("""
        a = [1, 2,
             3, 4]
        """)
    )
    doc.array("a").leading_comments[0] = ("about one",)
    assert tomlrt.dumps(doc) == td("""
        a = [
             # about one
             1, 2,
             3, 4]
        """)


def test_array_eol_comment_promotes_a_row_follower() -> None:
    """The break an EOL comment forces leaves the next item a row leader."""
    doc = tomlrt.loads(
        td("""
        a = [1, 2,
             3, 4]
        """)
    )
    doc.array("a").comments[0] = "one"
    out = tomlrt.dumps(doc)
    assert out == td("""
        a = [1, # one
             2,
             3, 4]
        """)
    assert reparses(out) == {"a": [1, 2, 3, 4]}


def test_inline_table_leading_comment_matches_a_packed_row_indent() -> None:
    """Inline tables sample the row indent the same way arrays do."""
    doc = tomlrt.loads(
        td("""
        t = { x = 1, y = 2,
              z = 3 }
        """)
    )
    doc.table("t").leading_comments["y"] = ("about y",)
    out = tomlrt.dumps(doc)
    assert out == td("""
        t = { x = 1,
              # about y
              y = 2,
              z = 3 }
        """)
    assert reparses(out) == {"t": {"x": 1, "y": 2, "z": 3}}


def test_array_sort_carries_a_block_onto_a_packed_row() -> None:
    """A block carried by a sort takes its item to the value indent with it."""
    doc = tomlrt.loads(
        td("""
        a = [4, 3,
             # about two
             2, 1]
        """)
    )
    doc.array("a").sort()
    out = tomlrt.dumps(doc)
    assert out == td("""
        a = [1,
             # about two
             2,
             3, 4]
        """)
    assert reparses(out) == {"a": [1, 2, 3, 4]}


def test_array_comma_first_leading_comment_on_a_packed_row() -> None:
    """Comma-first values promote a packed-row follower the same way."""
    doc = tomlrt.loads(
        td("""
        x = [
            1
            ,2, 3
            ]
        """)
    )
    doc.array("x").leading_comments[2] = ("about three",)
    out = tomlrt.dumps(doc)
    assert out == td("""
        x = [
            1
            ,2,
            # about three
            3
            ]
        """)
    assert reparses(out) == {"x": [1, 2, 3]}
