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

# A value with no TOML representation, for rejection tests.
_OPAQUE: Any = object()

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


def test_add_top_level_key_before_first_section_with_blank_attached_block() -> None:
    """Adding the first root KV ahead of a section keeps an existing
    opening blank-separated block on that section intact."""
    src = td("""

        # attached
        [srv]
        port = 8080
        """)
    doc = tomlrt.loads(src)
    doc["name"] = "demo"
    out = tomlrt.dumps(doc)
    assert out == td("""
        name = "demo"

        # attached
        [srv]
        port = 8080
        """)
    assert _reparses(out) == {"name": "demo", "srv": {"port": 8080}}


def test_add_top_level_key_before_first_section_with_attached_comment() -> None:
    """If the first section already owns a leading comment block, the
    inserted root KV still leaves one blank separator before it."""
    src = td("""
        # attached
        [srv]
        port = 8080
        """)
    doc = tomlrt.loads(src)
    doc["name"] = "demo"
    out = tomlrt.dumps(doc)
    assert out == td("""
        name = "demo"

        # attached
        [srv]
        port = 8080
        """)
    assert _reparses(out) == {"name": "demo", "srv": {"port": 8080}}


def test_add_top_level_key_before_first_section_with_indented_comment() -> None:
    """Leading indentation before the first section's attached comment
    should survive when a new root KV is inserted ahead of it."""
    src = "  # attached\n[srv]\nport = 8080\n"
    doc = tomlrt.loads(src)
    doc["name"] = "demo"
    out = tomlrt.dumps(doc)
    assert out == td("""
        name = "demo"

          # attached
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
    assert out == td("""
        [a]
        b = 2
        """)
    assert _reparses(out) == {"a": {"b": 2}}


def test_overwrite_non_last_key_with_section_keeps_following_kvs() -> None:
    """Replacing a non-last key with a section must place the new header
    *after* the table's remaining direct KVs, not at the old key's slot.

    Regression: ``reposition_install`` always repositioned a
    header-bearing replacement to the overwritten key's old position. If
    that key was not last, the new ``[a]`` header landed ahead of the
    sibling KVs (``b = 2``), which re-parse then attributed to ``a``.
    """
    doc = tomlrt.loads("a = 1\nb = 2\n")
    doc["a"] = Table.section({"x": 1})
    out = tomlrt.dumps(doc)
    assert out == td("""
        b = 2

        [a]
        x = 1
        """)
    assert _reparses(out) == doc.to_dict()


def test_overwrite_non_last_key_with_aot_keeps_following_kvs() -> None:
    """Same fix when the replacement is an array-of-tables."""
    doc = tomlrt.loads("[t]\na = 1\nb = 2\n")
    doc["t"]["a"] = AoT([{"x": 1}])
    out = tomlrt.dumps(doc)
    assert out == td("""
        [t]
        b = 2

        [[t.a]]
        x = 1
        """)
    assert _reparses(out) == doc.to_dict()


def test_overwrite_non_last_key_in_aot_entry_with_section() -> None:
    """The same hazard inside an AoT entry: the new sub-section header
    must follow the entry's remaining direct KVs."""
    doc = tomlrt.loads("[[p]]\na = 1\nb = 2\n")
    doc["p"][0]["a"] = Table.section({"x": 1})
    out = tomlrt.dumps(doc)
    assert out == td("""
        [[p]]
        b = 2

        [p.a]
        x = 1
        """)
    assert _reparses(out) == doc.to_dict()


def test_overwrite_section_with_aot_only_section_does_not_wipe_doc() -> None:
    """Overwriting an existing key with a detached body-less section.

    Regression: replacing ``[tool.example]`` with a ``Table.section``
    whose only child is an array-of-tables (no direct KVs) made
    ``reposition_install`` record the synthetic ``[tool.example]``
    header in ``new_slots``; ``_maybe_demote_synthetic_empty_header``
    then unlinked that header (its body is the AoT, so it has no direct
    KV). The orphaned slot stayed in ``new_slots``, and
    ``_move_slots_to_anchor`` spliced an unlinked slot back in — setting
    ``doc._head`` to ``None`` and rendering the whole document empty.
    """
    doc = tomlrt.loads(
        td("""
        [project]
        name = "demo"

        [[tool.example.items]]
        key = "existing"
        """)
    )
    doc2 = tomlrt.loads('[[items]]\nkey = "template"\n')
    new_section = Table.section(doc2)
    for entry in doc["tool"]["example"]["items"]:
        new_section["items"].append(entry)
    doc["tool"]["example"] = new_section
    out = tomlrt.dumps(doc)
    assert out == td("""
        [project]
        name = "demo"

        [[tool.example.items]]
        key = "template"

        [[tool.example.items]]
        key = "existing"
        """)
    assert _reparses(out) == doc.to_dict()


def test_overwrite_section_moving_block_earlier_keeps_later_edits_in_order() -> None:
    """A structural overwrite moves the new block back to the old position.

    The block is installed at the end of the document and moved earlier;
    everything filed after it must keep its doc-stream order so later
    edits still land in the right section.
    """
    doc = tomlrt.loads(
        td("""
        # preamble
        [a]
        x = 1

        [b]
        w = 2

        # comment for c
        [c]
        u = 3
        tail = 4
        """)
    )
    doc["a"] = Table.section({"x": 9, "y": 10})
    doc["a"]["z"] = 11
    doc["c"]["v"] = 12
    out = tomlrt.dumps(doc)
    assert out == td("""
        # preamble
        [a]
        x = 9
        y = 10
        z = 11

        [b]
        w = 2

        # comment for c
        [c]
        u = 3
        tail = 4
        v = 12
        """)
    assert _reparses(out) == doc.to_dict()


def test_overwrite_subsection_moving_block_later_keeps_ancestor_order() -> None:
    """The mirror case: the replacement block moves *later*.

    ``[a.b]`` reinstalls directly after ``[a]``'s body and is moved back
    down past ``[m]``, so both ``a``'s and the document's ref lists have
    to follow it; a later ``a`` key must still land in the ``[a]`` block.
    """
    doc = tomlrt.loads(
        td("""
        [a]
        x = 1

        [m]
        z = 0

        [a.b]
        y = 2
        """)
    )
    doc["a"]["b"] = Table.section({"q": 9})
    doc["a"]["b"]["r"] = 10
    doc["a"]["x2"] = 3
    out = tomlrt.dumps(doc)
    assert out == td("""
        [a]
        x = 1
        x2 = 3

        [m]
        z = 0

        [a.b]
        q = 9
        r = 10
        """)
    assert _reparses(out) == doc.to_dict()


def test_overwrite_in_aot_entry_with_trailing_entries_keeps_order() -> None:
    """The moved block is owned by an AoT entry and trailing entries follow."""
    doc = tomlrt.loads(
        td("""
        [[p]]
        a = 1
        keep = 2

        [[p]]
        a = 3

        [tail]
        t = 1
        """)
    )
    doc["p"][0]["a"] = Table.section({"x": 1})
    doc["p"][0]["more"] = 5
    doc["tail"]["u"] = 6
    out = tomlrt.dumps(doc)
    assert out == td("""
        [[p]]
        keep = 2
        more = 5

        [p.a]
        x = 1

        [[p]]
        a = 3

        [tail]
        t = 1
        u = 6
        """)
    assert _reparses(out) == doc.to_dict()


def test_overwrite_section_before_bulk_trailing_content() -> None:
    """Trailing content after the moved block is untouched by the move.

    The repositioning cost is proportional to the moved block, but what
    matters here is that the rest of the stream — and the ref order of
    every container that spans it — comes through unchanged.
    """
    trailing = "".join(f"\n[s{i}]\nv = {i}\n" for i in range(3))
    doc = tomlrt.loads("[a]\nx = 1\n" + trailing)
    doc["a"] = Table.section({"x": 2})
    doc["s1"]["extra"] = 7
    out = tomlrt.dumps(doc)
    assert out == td("""
        [a]
        x = 2

        [s0]
        v = 0

        [s1]
        v = 1
        extra = 7

        [s2]
        v = 2
        """)
    assert _reparses(out) == doc.to_dict()


def test_long_runs_of_appends_keep_landing_in_the_right_section() -> None:
    """Hundreds of appends into the same two spots, in document order.

    Each new key is placed relative to what is physically around it, and
    the bookkeeping that supports that has to be re-laid as a region
    fills up — repeatedly, and over a widening span as it gets denser.
    Every append must still land at the end of its own section.
    """
    doc = tomlrt.loads(
        td("""
        [a]
        x = 1

        [b]
        y = 2
        """)
    )
    for i in range(120):
        doc["a"][f"k{i:03d}"] = i
    # ``[b]`` is last, so its appends have the open end of the document
    # after them rather than another section's content.
    for i in range(20):
        doc["b"][f"j{i:02d}"] = i

    out = tomlrt.dumps(doc)
    assert out == (
        td("""
        [a]
        x = 1
        """)
        + "".join(f"k{i:03d} = {i}\n" for i in range(120))
        + td("""

        [b]
        y = 2
        """)
        + "".join(f"j{i:02d} = {i}\n" for i in range(20))
    )
    assert _reparses(out) == doc.to_dict()


def test_overwrite_moves_a_block_into_a_crowded_seam() -> None:
    """A block move claims order-key room for the whole block at once.

    Sustained appends in the middle of a document pack its order keys
    tight, so a structural overwrite there can want more room between
    two neighbours than is left. Re-laying the neighbourhood has to
    account for the whole incoming block, not just one slot of it.
    """
    sections = 100
    fill, replacement = 800, 100
    mid = sections // 2
    doc = tomlrt.loads("".join(f"[s{i:04d}]\nx = {i}\n\n" for i in range(sections)))
    for i in range(fill):
        doc[f"s{mid:04d}"][f"k{i:04d}"] = i
    doc[f"s{mid + 1:04d}"] = Table.section({f"m{j:03d}": j for j in range(replacement)})

    frames = [f"[s{i:04d}]\nx = {i}\n\n" for i in range(sections)]
    frames[mid] = (
        f"[s{mid:04d}]\nx = {mid}\n"
        + "".join(f"k{i:04d} = {i}\n" for i in range(fill))
        + "\n"
    )
    frames[mid + 1] = (
        f"[s{mid + 1:04d}]\n"
        + "".join(f"m{j:03d} = {j}\n" for j in range(replacement))
        + "\n"
    )
    out = tomlrt.dumps(doc)
    assert out == "".join(frames)
    assert _reparses(out) == doc.to_dict()

    # Keep editing around the move: a later insert or delete has to
    # find its place relative to the block that moved.
    doc[f"s{mid + 1:04d}"]["extra"] = -1
    doc[f"s{mid + 2:04d}"]["extra"] = -2
    del doc[f"s{mid + 1:04d}"]["m050"]

    frames[mid + 1] = (
        f"[s{mid + 1:04d}]\n"
        + "".join(f"m{j:03d} = {j}\n" for j in range(replacement) if j != 50)
        + "extra = -1\n\n"
    )
    frames[mid + 2] = f"[s{mid + 2:04d}]\nx = {mid + 2}\nextra = -2\n\n"
    out = tomlrt.dumps(doc)
    assert out == "".join(frames)
    assert _reparses(out) == doc.to_dict()
    assert tomlrt.dumps(tomlrt.loads(out)) == out


def test_mixed_inserts_deletes_and_moves_keep_the_document_consistent() -> None:
    """Every kind of splice, interleaved, then edits on top of the result.

    Inserts, deletes and the block move of a structural overwrite all
    reshuffle where later edits belong, so the document has to keep
    telling itself the same story about its own physical order: the
    rendered bytes, a re-parse of them, and the logical view must all
    agree once the dust settles.
    """
    doc = tomlrt.loads(
        td("""
        top = 0

        [a]
        x = 1
        y = 2

        [[p]]
        n = 1

        [[p]]
        n = 2

        [z]
        q = 3
        """)
    )
    doc["a"] = Table.section({"x": 10})
    del doc["z"]["q"]
    doc["a"]["w"] = 4
    doc["p"].append({"n": 3})
    doc["b"] = Table.section({"k": 5})
    del doc["p"][0]
    doc["top"] = 99

    out = tomlrt.dumps(doc)
    assert out == td("""
        top = 99

        [a]
        x = 10
        w = 4

        [[p]]
        n = 2

        [[p]]
        n = 3

        [z]

        [b]
        k = 5
        """)
    assert _reparses(out) == doc.to_dict()

    # Edit each container the shuffle touched: a new key lands next to
    # its container's existing content, wherever that ended up.
    doc["a"]["v"] = 6
    doc["z"]["r"] = 7
    doc["b"]["l"] = 8
    doc["p"][0]["m"] = 9
    doc["p"][1]["m"] = 10

    out = tomlrt.dumps(doc)
    assert out == td("""
        top = 99

        [a]
        x = 10
        w = 4
        v = 6

        [[p]]
        n = 2
        m = 9

        [[p]]
        n = 3
        m = 10

        [z]
        r = 7

        [b]
        k = 5
        l = 8
        """)
    assert _reparses(out) == doc.to_dict()
    assert tomlrt.dumps(tomlrt.loads(out)) == out


def test_overwrite_dotted_intermediate_keeps_dotted_form() -> None:
    """Overwriting a dotted intermediate (whose value is a subtable) with a
    scalar keeps the dotted form rather than promoting to an ``[a]`` header.

    ``a.b.c = 1`` is written with dotted keys, so replacing ``a.b`` with a
    scalar should stay dotted — ``a.b = "str"`` — preserving both the
    style and the position, instead of synthesising a section header
    (which also used to re-parent the trailing root key ``k``).
    """
    doc = tomlrt.loads("a.b.c = 1\nk = 2\n")
    doc["a"]["b"] = "str"
    out = tomlrt.dumps(doc)
    assert out == td("""
        a.b = "str"
        k = 2
        """)
    assert _reparses(out) == doc.to_dict()


def test_overwrite_dotted_subtable_before_sibling_header_under_same_key() -> None:
    # Overwriting a dotted sub-table re-files a ref under the shared ancestor
    # key (`apple` on `fruit`) ahead of the `[fruit.apple.texture]` header,
    # which is also filed under `apple`. The new ref precedes every existing
    # `apple` ref, so the index-projection places it at the front of the
    # bucket; the result must still round-trip and keep the header in place.
    doc = tomlrt.loads(
        td("""
        [fruit]
        apple.taste.sweet = true
        k = { a = 1 }

        [fruit.apple.texture]
        smooth = true
        """),
    )
    doc["fruit"]["apple"]["taste"] = {"a": 1}
    out = tomlrt.dumps(doc)
    assert out == td("""
        [fruit]
        apple.taste = { a = 1 }
        k = { a = 1 }

        [fruit.apple.texture]
        smooth = true
        """)
    assert _reparses(out) == doc.to_dict()


def test_overwrite_header_intermediate_keeps_header_form() -> None:
    """The mirror case: an intermediate written with a header keeps a header.

    ``[foo.bar.baz]`` uses header form, so replacing ``foo.bar`` with a
    scalar synthesises a ``[foo]`` header (not a dotted ``foo.bar``) and
    leaves it in place, after the sibling ``[other]`` section.
    """
    doc = tomlrt.loads(
        td("""
            [other]
            z = 3

            [foo.bar.baz]
            quux = 1
            """)
    )
    doc["foo"]["bar"] = 7
    out = tomlrt.dumps(doc)
    assert out == td("""
        [other]
        z = 3

        [foo]
        bar = 7
        """)
    assert _reparses(out) == doc.to_dict()


def test_overwrite_key_with_inline_when_subsection_follows() -> None:
    """Overwriting a key whose old value was a sub-section / nested AoT
    with an inline value must keep the new direct KV ahead of the
    parent's sub-section headers.

    Regression: the header-less reposition check treated any position
    inside the parent's whole subtree as safe, so the inline KV was
    repositioned to the old binding's slot — *after* a sibling
    ``[t.physical]`` sub-header — which re-parse then captured.
    """
    doc = tomlrt.loads(
        td("""
            [t]
            a = 1

            [t.physical]
            color = "red"

            [[t.varieties]]
            name = "x"
            """)
    )
    doc["t"]["varieties"] = {"a": 1}
    out = tomlrt.dumps(doc)
    assert out == td("""
        [t]
        a = 1
        varieties = { a = 1 }

        [t.physical]
        color = "red"
        """)
    assert _reparses(out) == doc.to_dict()
    assert doc.to_dict() == {
        "t": {"a": 1, "physical": {"color": "red"}, "varieties": {"a": 1}}
    }


def test_inline_overwrite_intermediate_dotted_node_keeps_view_linked() -> None:
    """Overwriting an intermediate dotted node of an inline table with a
    scalar must keep the logical view in sync with the rendered CST.

    Regression: the overwrite did ``del self[key]; self[key] = value``.
    The ``del`` emptied the navigator momentarily, so its empty-prefix
    cleanup unlinked the navigator (and detached ancestors) from the
    parent dict chain. The re-add fixed the CST but not those dict
    links, so ``to_dict()`` collapsed the whole branch to ``{}`` even
    though the rendered output was correct.
    """
    doc = tomlrt.loads("t = {a.b.c = 1, a.b.d = 2}\n")
    doc["t"]["a"]["b"] = 7
    out = tomlrt.dumps(doc)
    assert out == "t = {a.b = 7}\n"
    assert doc.to_dict() == {"t": {"a": {"b": 7}}}
    assert _reparses(out) == doc.to_dict()


def test_inline_overwrite_deep_intermediate_dotted_node() -> None:
    """Same fix across multiple detached navigator levels."""
    doc = tomlrt.loads("t = {a.b.c.d = 1, a.b.c.e = 2}\n")
    doc["t"]["a"]["b"]["c"] = 9
    out = tomlrt.dumps(doc)
    assert out == "t = {a.b.c = 9}\n"
    assert doc.to_dict() == {"t": {"a": {"b": {"c": 9}}}}
    assert _reparses(out) == doc.to_dict()


def test_inline_overwrite_sole_prefix_preserves_bracket_pad() -> None:
    """Overwriting a table's sole-content dotted prefix keeps its pad.

    The outer table is only transiently emptied while the old prefix
    entries are dropped, so an authored padded ``{ … }`` must stay
    padded (and a tight one stays tight) rather than being re-stamped
    to the canonical first-insert padding.
    """
    doc = tomlrt.loads("t = { a.b.c.d = 1 }\n")
    doc["t"]["a"]["b"]["c"] = 9
    out = tomlrt.dumps(doc)
    assert out == "t = { a.b.c = 9 }\n"
    assert _reparses(out) == doc.to_dict()


def test_inline_overwrite_dotted_prefix_on_multiline_table_skips_pad_restore() -> None:
    """No single-line pad to restore when the table is already multiline.

    ``set_multiline`` forces `is_multiline()` true, so overwriting a
    dotted prefix must skip the single-line bracket-pad save/restore
    entirely and let ``splice_in`` alone re-lay the multiline rows.
    """
    doc = tomlrt.loads("t = { a.b = 1, a.c = 2 }\n")
    doc.table("t").multiline = True
    doc.table("t")["a"] = 5
    out = tomlrt.dumps(doc)
    assert out == td("""
        t = {
            a = 5,
        }
        """)
    assert _reparses(out) == doc.to_dict()


def test_set_overwrites_implicit_child_table() -> None:
    src = "[a.b]\nx = 1\n"
    doc = tomlrt.loads(src)
    doc["a"] = 5
    out = tomlrt.dumps(doc)
    assert out == "a = 5\n"
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
    assert out == td(
        r"""
        "a\"b" = 1
        "c\\d" = 2
        "e\u0009f" = 3
        """
    )
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
    assert out == "obj = { a = 1 }\n"
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
    assert out == td("""
        obj = { a = 1, xs = [
            1,
            2,
            3,
            4,
        ] }
        """)
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


def test_inline_replacement_of_out_of_order_subtable() -> None:
    """Replacing an out-of-order subsection with an inline value must
    keep the new KV inside its parent's body region.

    ``[foo.bar]`` precedes ``[foo]`` in the source, so the existing
    primary slot for ``foo['bar']`` sits before ``[foo]``'s own
    header. Repositioning the synthesised inline KV onto that anchor
    would land it at top level, silently re-parenting ``bar`` out of
    ``foo``. The structural-overwrite path must detect this and fall
    back to delete + reinsert at ``[foo]``'s body tail.
    """
    src = td("""
        [foo.bar]
        here = true

        [foo]
        a = 1
        b = 2
        """)
    doc = tomlrt.loads(src)
    doc["foo"]["bar"] = {"x": 1}
    assert tomlrt.dumps(doc) == td("""
        [foo]
        a = 1
        b = 2
        bar = { x = 1 }
        """)


def test_inline_replacement_of_subtable_after_foreign_section() -> None:
    """Same out-of-order hazard, but the subsection sits *after* a
    foreign sibling header rather than before parent's own header.

    The structural-overwrite walks backward from ``[foo.sub]`` past
    ``z = 1`` and hits ``[other]`` — a header whose path is not
    under ``[foo]`` — so the captured anchor is rejected and the
    new inline lands at ``[foo]``'s body tail instead of after
    ``[other]`` (where it would silently reparent under ``other``).
    """
    src = td("""
        [foo]
        a = 1

        [other]
        z = 1

        [foo.sub]
        x = 1
        """)
    doc = tomlrt.loads(src)
    doc["foo"]["sub"] = {"y": 2}
    assert tomlrt.dumps(doc) == td("""
        [foo]
        a = 1
        sub = { y = 2 }

        [other]
        z = 1
        """)


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
    assert out == "xs = [1, 2, 3, 4]\n"
    assert _reparses(out) == {"xs": [1, 2, 3, 4]}


def test_array_append_to_empty_with_tab_indented_comment_preserves_tab() -> None:
    # The only indent signal in the empty container is the tab before
    # the comment line. Appending must reuse it instead of falling back
    # to the four-space default.
    doc = tomlrt.loads("a = [\n\t# hi\n]\n")
    arr = doc.array("a")
    arr.append(1)
    assert tomlrt.dumps(doc) == "a = [\n\t# hi\n\t1,\n]\n"


def test_array_append_to_empty_with_unindented_comment() -> None:
    doc = tomlrt.loads(
        td("""
            a = [
            # hi
            ]
            """)
    )
    doc.array("a").append(1)
    assert tomlrt.dumps(doc) == td("""
        a = [
        # hi
            1,
        ]
        """)


def test_array_pop() -> None:
    doc = tomlrt.loads("xs = [10, 20, 30]\n")
    xs = doc.array("xs")
    v = xs.pop()
    assert v == 30
    assert list(xs) == [10, 20]
    out = tomlrt.dumps(doc)
    assert out == "xs = [10, 20]\n"
    assert _reparses(out) == {"xs": [10, 20]}


def test_array_pop_out_of_range_raises_indexerror() -> None:
    doc = tomlrt.loads("xs = [1, 2, 3]\n")
    xs = doc.array("xs")
    with pytest.raises(IndexError, match="out of range"):
        xs.pop(99)
    with pytest.raises(IndexError, match="out of range"):
        xs.pop(-99)


def test_array_indices_and_repeat_counts_require_supports_index() -> None:
    doc = tomlrt.loads("a = [1, 2]\n")
    arr = doc.array("a")
    invalid: Any = 0.9

    with pytest.raises(TypeError):
        arr.insert(invalid, 9)
    with pytest.raises(TypeError):
        arr[invalid] = 9
    with pytest.raises(TypeError):
        del arr[invalid]
    with pytest.raises(TypeError):
        arr.pop(invalid)
    with pytest.raises(TypeError):
        arr *= invalid

    assert list(arr) == [1, 2]
    assert tomlrt.dumps(doc) == "a = [1, 2]\n"


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
    assert out == "xs = [1, 22, 3]\n"
    assert _reparses(out) == {"xs": [1, 22, 3]}


def test_array_setitem_int_negative_index() -> None:
    doc = tomlrt.loads("xs = [1, 2, 3]\n")
    xs = doc.array("xs")
    xs[-1] = 33
    out = tomlrt.dumps(doc)
    assert out == "xs = [1, 2, 33]\n"
    assert _reparses(out) == {"xs": [1, 2, 33]}


def test_array_setitem_int_oob_index_does_not_corrupt_cst() -> None:
    # Regression: an out-of-range index (negative or positive) must raise
    # IndexError *without* mutating any item's value CST. The negative case
    # used to normalise into a valid slot and silently overwrite it before
    # ``list.__setitem__`` raised, leaving the rendered output disagreeing
    # with the logical view.
    for bad in (-4, 3):
        doc = tomlrt.loads("xs = [1, 2, 3]\n")
        xs = doc.array("xs")
        with pytest.raises(IndexError, match="list assignment index out of range"):
            xs[bad] = 99
        assert list(xs) == [1, 2, 3]
        out = tomlrt.dumps(doc)
        assert out == "xs = [1, 2, 3]\n"
        assert _reparses(out) == {"xs": [1, 2, 3]}


def test_array_setitem_slice_extended_matching_length() -> None:
    doc = tomlrt.loads("xs = [1, 2, 3, 4, 5]\n")
    xs = doc.array("xs")
    xs[::2] = [10, 30, 50]
    out = tomlrt.dumps(doc)
    assert out == "xs = [10, 2, 30, 4, 50]\n"
    assert _reparses(out) == {"xs": [10, 2, 30, 4, 50]}


@pytest.mark.parametrize("index", [slice(0, 2), slice(None, None, 2)])
def test_array_setitem_slice_invalid_value_is_atomic(index: slice) -> None:
    src = td("""
        xs = [ # values
            1, # one
            3,
            5,
        ]
        """)
    doc = tomlrt.loads(src)
    xs = doc.array("xs")

    with pytest.raises(TypeError):
        xs[index] = [2, object()]

    assert list(xs) == [1, 3, 5]
    rendered = tomlrt.dumps(doc)
    assert rendered == src
    assert _reparses(rendered) == {"xs": [1, 3, 5]}


def test_array_setitem_slice() -> None:
    doc = tomlrt.loads("xs = [1, 2, 3, 4]\n")
    xs = doc.array("xs")
    xs[1:3] = [22, 33, 44]
    out = tomlrt.dumps(doc)
    assert out == "xs = [1, 22, 33, 44, 4]\n"
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
    assert out == "xs = [1, 4]\n"
    assert _reparses(out) == {"xs": [1, 4]}


def test_array_delitem_empty_slice_is_noop() -> None:
    src = "xs = [1, 2, 3]\n"
    doc = tomlrt.loads(src)
    del doc.array("xs")[1:1]
    assert tomlrt.dumps(doc) == src


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
    arr.leading_comments[2] = ("above three",)
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
    assert out == 'xs = ["hi"]\n'
    assert _reparses(out) == {"xs": ["hi"]}


def test_array_multiline_tracks_shape_across_add_and_remove() -> None:
    # `multiline` is derived from the value's rendered shape and memoised.
    # The memo must stay correct as items are added (shape preserved) and
    # removed (removing the item that carries the sole newline flips it to
    # single-line, so a later append must not resurrect the multi-line form).
    doc = tomlrt.loads("xs = [1, # c\n    2]\n")
    assert doc.array("xs").multiline is True
    del doc.array("xs")[0]
    doc.array("xs").append(3)
    assert doc.array("xs").multiline is False
    assert tomlrt.dumps(doc) == "xs = [2, 3]\n"


def test_array_extend_iadd() -> None:
    doc = tomlrt.loads("xs = []\n")
    xs = doc.array("xs")
    xs.extend([1, 2])
    xs += [3, 4]
    out = tomlrt.dumps(doc)
    assert out == "xs = [1, 2, 3, 4]\n"
    assert _reparses(out) == {"xs": [1, 2, 3, 4]}


def test_array_extend_multiline_lays_one_item_per_line() -> None:
    # extend snapshots the layout style once and reuses it, so a multi-line
    # array still gets each appended item on its own line (matching repeated
    # append) rather than collapsing onto a single row.
    doc = tomlrt.loads(
        td("""
        xs = [
            1,
        ]
        """),
    )
    doc["xs"].extend([2, 3])
    assert tomlrt.dumps(doc) == td("""
        xs = [
            1,
            2,
            3,
        ]
        """)


def test_array_extend_rejects_aot_atomically() -> None:
    doc = tomlrt.loads("xs = [1]\n")
    with pytest.raises(tomlrt.TOMLError):
        doc["xs"].extend([2, AoT([{"a": 1}])])
    # The whole extend is rejected up front; no partial mutation.
    assert tomlrt.dumps(doc) == "xs = [1]\n"


@pytest.mark.parametrize(
    "invalid",
    [{"nested": AoT([{"a": 1}])}, {"nested": Table.section({"a": 1})}],
)
def test_array_extend_rejects_nested_structural_value_atomically(
    invalid: Any,
) -> None:
    doc = tomlrt.loads("xs = [1]\n")

    with pytest.raises(tomlrt.TOMLError):
        doc.array("xs").extend([2, invalid])

    assert tomlrt.dumps(doc) == "xs = [1]\n"


def test_array_extend_invalid_value_is_atomic() -> None:
    src = td("""
        xs = [ # values
            1, # one
        ]
        """)
    doc = tomlrt.loads(src)
    xs = doc.array("xs")

    with pytest.raises(TypeError):
        xs.extend([2, object()])

    assert list(xs) == [1]
    rendered = tomlrt.dumps(doc)
    assert rendered == src
    assert _reparses(rendered) == {"xs": [1]}


@pytest.mark.parametrize(
    ("value", "error", "message"),
    [
        pytest.param((1, 2), TypeError, "cannot assign tuple; use a list", id="tuple"),
        pytest.param(
            b"value", TypeError, "cannot assign bytes; use a string", id="bytes"
        ),
        pytest.param(
            AoT([{"a": 1}]),
            tomlrt.TOMLError,
            "cannot store an array-of-tables inside an inline array",
            id="aot",
        ),
        pytest.param(
            Table.section({"a": 1}),
            tomlrt.TOMLError,
            "cannot store a section-style table inside an inline-style table",
            id="section",
        ),
        pytest.param(
            {"nested": AoT([{"a": 1}])},
            tomlrt.TOMLError,
            "cannot store an array-of-tables inside an inline table",
            id="nested-aot",
        ),
    ],
)
def test_array_append_rejects_unstorable_value_atomically(
    value: Any, error: type[Exception], message: str
) -> None:
    """`append` synthesises before splicing, so a value an inline array
    cannot hold is rejected with the array left untouched."""
    src = "xs = [1]\n"
    doc = tomlrt.loads(src)
    xs = doc.array("xs")

    with pytest.raises(error, match=message):
        xs.append(value)

    assert list(xs) == [1]
    assert tomlrt.dumps(doc) == src


@pytest.mark.parametrize(
    ("value", "error"),
    [
        pytest.param(_OPAQUE, TypeError, id="opaque"),
        pytest.param(AoT([{"a": 1}]), tomlrt.TOMLError, id="aot"),
        pytest.param(Table.section({"a": 1}), tomlrt.TOMLError, id="section"),
    ],
)
def test_array_constructor_rejects_unstorable_value(
    value: Any, error: type[Exception]
) -> None:
    with pytest.raises(error):
        Array([value])


@pytest.mark.parametrize("value", [Array([2]), Table.inline({"x": 2})])
def test_array_failed_bulk_mutation_does_not_attach_input(value: Any) -> None:
    doc = tomlrt.loads("xs = []\n")

    with pytest.raises(TypeError):
        doc.array("xs").extend([value, object()])

    doc["kept"] = value
    assert doc["kept"] is value
    expected = (
        "xs = []\nkept = [2]\n"
        if isinstance(value, Array)
        else "xs = []\nkept = { x = 2 }\n"
    )
    assert tomlrt.dumps(doc) == expected


def _apply_array_single_mutation(arr: Array, operation: str, value: Any) -> None:
    if operation == "append":
        arr.append(value)
    elif operation == "insert":
        arr.insert(0, value)
    else:
        arr[0] = value


def _reject_array_synthesis(failure_site: str, destination: Array, value: Any) -> None:
    invalid: list[Any] = [value, object()]
    if failure_site == "constructor":
        Array(invalid)
    else:
        destination.append(invalid)


@pytest.mark.parametrize("operation", ["append", "insert", "replace"])
@pytest.mark.parametrize("value_kind", ["array", "table"])
def test_array_failed_single_mutation_does_not_attach_nested_input(
    operation: str, value_kind: str
) -> None:
    src = "xs = [1]\n"
    doc = tomlrt.loads(src)
    xs = doc.array("xs")
    value = Array([2]) if value_kind == "array" else Table.inline({"x": 2})
    invalid = [value, object()]

    with pytest.raises(TypeError):
        _apply_array_single_mutation(xs, operation, invalid)

    assert tomlrt.dumps(doc) == src
    doc["kept"] = value
    assert doc["kept"] is value
    expected = (
        td("""
            xs = [1]
            kept = [2]
            """)
        if isinstance(value, Array)
        else td("""
            xs = [1]
            kept = { x = 2 }
            """)
    )
    rendered = tomlrt.dumps(doc)
    assert rendered == expected
    assert _reparses(rendered) == doc.to_dict()


@pytest.mark.parametrize("failure_site", ["constructor", "append"])
def test_failed_array_synthesis_preserves_detached_inline_table(
    failure_site: str,
) -> None:
    value = Table.inline({"x": 1})
    doc = tomlrt.loads("xs = []\n")

    with pytest.raises(TypeError, match="cannot convert object to a TOML value"):
        _reject_array_synthesis(failure_site, doc.array("xs"), value)

    with pytest.raises(
        tomlrt.TOMLError, match="unavailable on a detached inline table"
    ):
        value.set_multiline(multiline=True)
    value["y"] = 2
    del value["x"]
    doc["kept"] = value
    assert doc["kept"] is value
    value["z"] = 3
    rendered = tomlrt.dumps(doc)
    assert rendered == td("""
        xs = []
        kept = { y = 2, z = 3 }
        """)
    assert _reparses(rendered) == doc.to_dict()


@pytest.mark.parametrize("failure_site", ["constructor", "append"])
def test_failed_array_synthesis_preserves_detached_array_layout(
    failure_site: str,
) -> None:
    value = Array([1], multiline=True, indent=2)
    value.comments[0] = "kept"
    doc = tomlrt.loads("xs = []\r\n")

    with pytest.raises(TypeError, match="cannot convert object to a TOML value"):
        _reject_array_synthesis(failure_site, doc.array("xs"), value)

    assert value.multiline
    assert value.comments[0] == "kept"
    value.append(2)
    doc["kept"] = value
    assert doc["kept"] is value
    rendered = tomlrt.dumps(doc)
    assert rendered == "xs = []\r\nkept = [\r\n  1, # kept\r\n  2,\r\n]\r\n"
    assert _reparses(rendered) == doc.to_dict()


def test_array_single_mutation_live_attaches_array_input() -> None:
    doc = tomlrt.loads("xs = [1]\n")
    xs = doc.array("xs")
    value = Array([2], multiline=True, indent=2)

    xs.append(value)

    assert xs[1] is value
    rendered = tomlrt.dumps(doc)
    assert rendered == td("""
        xs = [1, [
          2,
        ]]
    """)
    assert _reparses(rendered) == doc.to_dict()


def test_array_extend_self_duplicates_once() -> None:
    """``arr.extend(arr)`` matches list semantics: duplicate once, no hang.

    Regression: the implementation iterated ``values`` while appending to
    ``self``. When ``values is self`` the iteration kept seeing the
    just-appended items and never terminated.
    """
    doc = tomlrt.loads("xs = [1, 2]\n")
    xs = doc.array("xs")
    xs.extend(xs)
    assert list(xs) == [1, 2, 1, 2]
    assert tomlrt.dumps(doc) == "xs = [1, 2, 1, 2]\n"


def test_array_extend_empty_is_noop() -> None:
    src = "xs = [1, 2]\n"
    doc = tomlrt.loads(src)
    doc.array("xs").extend([])
    assert tomlrt.dumps(doc) == src


def test_array_iadd_self_duplicates_once() -> None:
    """``arr += arr`` matches list semantics: duplicate once, no hang."""
    doc = tomlrt.loads("xs = [1, 2]\n")
    xs = doc.array("xs")
    xs += xs
    assert list(xs) == [1, 2, 1, 2]
    assert tomlrt.dumps(doc) == "xs = [1, 2, 1, 2]\n"


def test_array_sort_reverse() -> None:
    doc = tomlrt.loads("xs = [3, 1, 2]\n")
    xs = doc.array("xs")
    xs.sort()
    assert list(xs) == [1, 2, 3]
    xs.reverse()
    out = tomlrt.dumps(doc)
    assert out == "xs = [3, 2, 1]\n"
    assert _reparses(out) == {"xs": [3, 2, 1]}


def test_array_imul() -> None:
    doc = tomlrt.loads("xs = [1, 2]\n")
    xs = doc.array("xs")
    xs *= 3
    out = tomlrt.dumps(doc)
    assert out == "xs = [1, 2, 1, 2, 1, 2]\n"
    assert _reparses(out) == {"xs": [1, 2, 1, 2, 1, 2]}


def test_array_imul_by_one_is_identity() -> None:
    """``arr *= 1`` short-circuits — output is byte-identical to the input."""
    src = "xs = [1, 2]\n"
    doc = tomlrt.loads(src)
    xs = doc.array("xs")
    xs *= 1
    assert list(xs) == [1, 2]
    assert tomlrt.dumps(doc) == src


def test_detached_array_imul_clones_inline_table_elements() -> None:
    # A standalone (unattached) array of inline tables has no binding, so
    # its `__imul__` clones decode with ``parent=None``. The cloned views
    # round-trip once attached.
    arr = Array([{"a": 1}])
    arr *= 2
    doc = tomlrt.loads("")
    doc["k"] = arr
    out = tomlrt.dumps(doc)
    assert out == "k = [{ a = 1 }, { a = 1 }]\n"
    assert _reparses(out) == {"k": [{"a": 1}, {"a": 1}]}


def test_array_remove() -> None:
    doc = tomlrt.loads("xs = [1, 2, 3, 2]\n")
    xs = doc.array("xs")
    xs.remove(2)
    out = tomlrt.dumps(doc)
    assert out == "xs = [1, 3, 2]\n"
    assert _reparses(out) == {"xs": [1, 3, 2]}


def test_array_insert() -> None:
    doc = tomlrt.loads("xs = [1, 3]\n")
    xs = doc.array("xs")
    xs.insert(1, 2)
    out = tomlrt.dumps(doc)
    assert out == "xs = [1, 2, 3]\n"
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


# ---------------------------------------------------------------------------
# Removal seam repair (shared by inline arrays and inline tables)
# ---------------------------------------------------------------------------

# Every item owns an above-block and an EOL comment, and a dangling comment
# sits before the closing bracket, so a seam repaired against the wrong
# boundary shows up as a moved, duplicated, or lost comment.
_SEAM_ARRAY = td("""
    arr = [
        # above a
        "a", # eol a
        # above b
        "b", # eol b
        # above c
        "c", # eol c
        # above d
        "d", # eol d
        # above e
        "e", # eol e
        # dangling
    ]
    """)

_SEAM_TABLE = td("""
    t = {
        # above a
        a = 1, # eol a
        # above x
        p.x = 2, # eol x
        # above y
        p.y = 3, # eol y
        # above d
        d = 4, # eol d
        # dangling
    }
    """)


def test_array_delete_first_item_rehomes_head_comments() -> None:
    doc = tomlrt.loads(_SEAM_ARRAY)
    del doc.array("arr")[0]
    out = tomlrt.dumps(doc)
    assert out == td("""
        arr = [
            # above b
            "b", # eol b
            # above c
            "c", # eol c
            # above d
            "d", # eol d
            # above e
            "e", # eol e
            # dangling
        ]
        """)
    assert _reparses(out) == {"arr": ["b", "c", "d", "e"]}


def test_array_delete_middle_item_keeps_seam_comments() -> None:
    doc = tomlrt.loads(_SEAM_ARRAY)
    del doc.array("arr")[2]
    out = tomlrt.dumps(doc)
    assert out == td("""
        arr = [
            # above a
            "a", # eol a
            # above b
            "b", # eol b
            # above d
            "d", # eol d
            # above e
            "e", # eol e
            # dangling
        ]
        """)
    assert _reparses(out) == {"arr": ["a", "b", "d", "e"]}


def test_array_delete_last_item_keeps_dangling_comment() -> None:
    doc = tomlrt.loads(_SEAM_ARRAY)
    del doc.array("arr")[-1]
    assert tomlrt.dumps(doc) == td("""
        arr = [
            # above a
            "a", # eol a
            # above b
            "b", # eol b
            # above c
            "c", # eol c
            # above d
            "d", # eol d
            # dangling
        ]
        """)


def test_array_delete_contiguous_slice_keeps_seam_comments() -> None:
    doc = tomlrt.loads(_SEAM_ARRAY)
    del doc.array("arr")[1:3]
    assert tomlrt.dumps(doc) == td("""
        arr = [
            # above a
            "a", # eol a
            # above d
            "d", # eol d
            # above e
            "e", # eol e
            # dangling
        ]
        """)


def test_array_delete_strided_slice_keeps_every_seam_comment() -> None:
    """Two interior seams in one removal: each survivor keeps its own block."""
    doc = tomlrt.loads(_SEAM_ARRAY)
    del doc.array("arr")[1::2]
    out = tomlrt.dumps(doc)
    assert out == td("""
        arr = [
            # above a
            "a", # eol a
            # above c
            "c", # eol c
            # above e
            "e", # eol e
            # dangling
        ]
        """)
    assert _reparses(out) == {"arr": ["a", "c", "e"]}


def test_array_delete_first_and_last_item_together() -> None:
    """A removal touching both ends and no interior seam."""
    doc = tomlrt.loads(_SEAM_ARRAY)
    del doc.array("arr")[::4]
    assert tomlrt.dumps(doc) == td("""
        arr = [
            # above b
            "b", # eol b
            # above c
            "c", # eol c
            # above d
            "d", # eol d
            # dangling
        ]
        """)


def test_array_delete_all_items_keeps_bracket_comments() -> None:
    doc = tomlrt.loads(_SEAM_ARRAY)
    del doc.array("arr")[:]
    assert tomlrt.dumps(doc) == td("""
        arr = [
            # above a
            # dangling
        ]
        """)


def test_inline_table_delete_first_entry_rehomes_head_comments() -> None:
    doc = tomlrt.loads(_SEAM_TABLE)
    del doc.table("t")["a"]
    out = tomlrt.dumps(doc)
    assert out == td("""
        t = {
            # above x
            p.x = 2, # eol x
            # above y
            p.y = 3, # eol y
            # above d
            d = 4, # eol d
            # dangling
        }
        """)
    assert _reparses(out) == {"t": {"p": {"x": 2, "y": 3}, "d": 4}}


def test_inline_table_delete_dotted_prefix_keeps_seam_comments() -> None:
    """A dotted prefix removes a run of entries between two survivors."""
    doc = tomlrt.loads(_SEAM_TABLE)
    del doc.table("t")["p"]
    out = tomlrt.dumps(doc)
    assert out == td("""
        t = {
            # above a
            a = 1, # eol a
            # above d
            d = 4, # eol d
            # dangling
        }
        """)
    assert _reparses(out) == {"t": {"a": 1, "d": 4}}


def test_inline_table_delete_last_entry_keeps_dangling_comment() -> None:
    doc = tomlrt.loads(_SEAM_TABLE)
    del doc.table("t")["d"]
    assert tomlrt.dumps(doc) == td("""
        t = {
            # above a
            a = 1, # eol a
            # above x
            p.x = 2, # eol x
            # above y
            p.y = 3, # eol y
            # dangling
        }
        """)


def test_inline_table_delete_every_entry_keeps_bracket_comments() -> None:
    doc = tomlrt.loads(_SEAM_TABLE)
    t = doc.table("t")
    for key in ("a", "p", "d"):
        del t[key]
    assert tomlrt.dumps(doc) == td("""
        t = {
            # above d
            # dangling
        }
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
    assert out == td("""
        src = [1, 2, 3]
        dst = [1, 2, 3, 99]
        """)
    parsed = _reparses(out)
    assert parsed == {"src": [1, 2, 3], "dst": [1, 2, 3, 99]}


def test_assigning_dict_creates_inline_table() -> None:
    doc = tomlrt.loads("")
    doc["obj"] = {"a": 1, "b": "two"}
    out = tomlrt.dumps(doc)
    assert out == 'obj = { a = 1, b = "two" }\n'
    assert _reparses(out) == {"obj": {"a": 1, "b": "two"}}


def test_assigning_list_creates_inline_array() -> None:
    doc = tomlrt.loads("")
    doc["nums"] = [1, 2, 3]
    out = tomlrt.dumps(doc)
    assert out == "nums = [1, 2, 3]\n"
    assert _reparses(out) == {"nums": [1, 2, 3]}


def test_overwrite_aot_with_implicit_table_containing_empty_aot() -> None:
    doc = tomlrt.loads(
        td("""
            source.zchild.leaf = 1

            [[target]]
            x = 1
            """)
    )
    source = doc["source"]
    source["aempty"] = AoT()
    source.sort()

    doc["target"] = source

    out = tomlrt.dumps(doc)
    assert out == td("""
        source.aempty = []
        source.zchild.leaf = 1

        [target]
        aempty = []
        zchild.leaf = 1
        """)
    assert _reparses(out) == doc.to_dict()


def test_replace_scalar_with_array() -> None:
    doc = tomlrt.loads("x = 1\n")
    doc["x"] = [True, False]
    out = tomlrt.dumps(doc)
    assert out == "x = [true, false]\n"
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
    assert out == td("""
        [[pkg]]
        name = "a"

        [pkg.dep]
        x = 1

        [[pkg]]
        name = "b"
        """)
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
    out = tomlrt.dumps(doc)
    assert out == td("""
        [[pkg]]
        name = "a"

        [pkg.dep]
        x = 1

        [[pkg]]
        name = "c"
        """)
    assert _reparses(out) == {
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
    out = tomlrt.dumps(new_doc)
    assert out == td("""
        [pkg]
        name = "a"
        extra = 99

        [pkg.dep]
        x = 1
        """)
    assert _reparses(out)["pkg"]["extra"] == 99


def test_aot_clear_removes_all_entries_and_owned_subsections() -> None:
    doc = _aot_doc()
    aot = doc.aot("pkg")
    aot.clear()
    assert len(aot) == 0
    out = tomlrt.dumps(doc)
    assert out == "pkg = []\n"
    assert _reparses(out) == {"pkg": []}


def test_aot_delitem_index_pops_one() -> None:
    doc = _aot_doc()
    aot = doc.aot("pkg")
    del aot[1]
    out = tomlrt.dumps(doc)
    assert out == td("""
        [[pkg]]
        name = "a"

        [pkg.dep]
        x = 1

        [[pkg]]
        name = "c"
        """)
    assert _reparses(out) == {
        "pkg": [{"name": "a", "dep": {"x": 1}}, {"name": "c"}],
    }


def test_aot_delitem_slice_removes_range() -> None:
    doc = _aot_doc()
    aot = doc.aot("pkg")
    del aot[1:]
    out = tomlrt.dumps(doc)
    assert out == td("""
        [[pkg]]
        name = "a"

        [pkg.dep]
        x = 1
        """)
    assert _reparses(out) == {"pkg": [{"name": "a", "dep": {"x": 1}}]}


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
    out = tomlrt.dumps(doc)
    assert out == td("""
        [[p]]
        n=2
        [[p]]
        n=4
        """)
    assert _reparses(out) == {"p": [{"n": 2}, {"n": 4}]}


def test_aot_setitem_replaces_entry() -> None:
    doc = _aot_doc()
    aot = doc.aot("pkg")
    aot[0] = {"new": True}
    rendered = tomlrt.dumps(doc)
    assert rendered == td("""
        [[pkg]]
        new = true

        [[pkg]]
        name = "b"

        [[pkg]]
        name = "c"
        """)
    assert _reparses(rendered)["pkg"][0] == {"new": True}
    assert len(doc.aot("pkg")) == 3


def test_aot_setitem_negative_index() -> None:
    doc = _aot_doc()
    aot = doc.aot("pkg")
    aot[-1] = {"replaced": True}
    rendered = tomlrt.dumps(doc)
    assert rendered == td("""
        [[pkg]]
        name = "a"

        [pkg.dep]
        x = 1

        [[pkg]]
        name = "b"

        [[pkg]]
        replaced = true
        """)
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
    assert rendered == td("""
        [[pkg]]
        name = "a"

        [pkg.dep]
        x = 1

        [[pkg]]
        name = "b"

        [[pkg]]
        name = "c"

        [[pkg]]
        name = "d"

        [[pkg]]
        name = "e"
        """)
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
    assert rendered == td("""
        [[t]]
        x = 1
        [[t]]
        x = 2
        [[t]]
        x = 1
        [[t]]
        x = 2
        [[t]]
        x = 1
        [[t]]
        x = 2
        """)
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
    out = tomlrt.dumps(doc)
    assert out == "t = []\n"
    assert tomlrt.loads(out)["t"] == []


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


def test_aot_detached_add_without_body_returns_empty_entry() -> None:
    aot = AoT()
    entry = aot.add()
    assert entry == {}
    assert list(aot) == [{}]


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


def test_aot_insert_invalid_index_does_not_mutate_document() -> None:
    doc = tomlrt.loads(
        td("""
            [[a]]
            x = 1
            """)
    )
    aot = doc.aot("a")
    invalid_index: Any = "bad"

    with pytest.raises(TypeError):
        aot.insert(invalid_index, {"x": 2})

    assert tomlrt.dumps(doc) == td("""
        [[a]]
        x = 1
        """)
    assert len(aot) == 1


def test_attached_aot_indices_and_repeat_counts_require_supports_index() -> None:
    doc = tomlrt.loads(
        td("""
            [[a]]
            x = 1

            [[a]]
            x = 2
            """)
    )
    aot = doc.aot("a")
    invalid: Any = 0.9

    with pytest.raises(TypeError):
        del aot[invalid]
    with pytest.raises(TypeError):
        aot.pop(invalid)
    with pytest.raises(TypeError):
        aot.insert(invalid, {"x": 3})
    with pytest.raises(TypeError):
        aot *= invalid

    assert aot.to_list() == [{"x": 1}, {"x": 2}]
    assert tomlrt.dumps(doc) == td("""
        [[a]]
        x = 1

        [[a]]
        x = 2
        """)


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
    out = tomlrt.dumps(doc)
    assert out == td("""
        [[t]]
        x = 3
        [[t]]
        x = 2
        [[t]]
        x = 1
        """)
    assert _reparses(out)["t"] == [{"x": 3}, {"x": 2}, {"x": 1}]


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
    out = tomlrt.dumps(doc)
    assert out == td("""
        [[t]]
        x = 1
        [[t]]
        x = 2
        [[t]]
        x = 3
        """)
    assert _reparses(out)["t"] == [{"x": 1}, {"x": 2}, {"x": 3}]


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


def test_aot_sort_terminates_former_document_tail() -> None:
    doc = tomlrt.loads(
        td("""
        [[t]]
        x = 2

        [[t]]
        x = 1""")
    )
    doc.aot("t").sort(key=lambda entry: entry["x"])
    assert tomlrt.dumps(doc) == td("""
        [[t]]
        x = 1

        [[t]]
        x = 2
    """)


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
    out = tomlrt.dumps(doc)
    assert out == td("""
        [[t]]
        name = "b"
        [t.sub]
        y = 2
        [[t]]
        name = "a"
        [t.sub]
        y = 1
        """)
    parsed = _reparses(out)["t"]
    assert parsed == [
        {"name": "b", "sub": {"y": 2}},
        {"name": "a", "sub": {"y": 1}},
    ]


def test_append_aot_entry_after_sorting_an_entrys_keys() -> None:
    """Appending an AoT entry after reordering a prior entry's keys must
    anchor the new entry after the entry's true last slot.

    Regression: ``reorder_container`` permuted the entry's KV slots in
    the doc-stream but left ``AoTEntry.entry_slots`` in its old order.
    ``_aot_append_anchor``'s predecessor trusted ``entry_slots[-1]`` as the
    append anchor, so the new ``[[p]]`` header was spliced *inside* the
    reordered entry, splitting it and re-parenting the trailing key on
    re-parse.
    """
    doc = tomlrt.loads("[[p]]\nb = 2\na = 1\n")
    doc["p"][0].sort()
    doc["p"].append({"new": True})
    out = tomlrt.dumps(doc)
    assert out == td("""
        [[p]]
        a = 1
        b = 2

        [[p]]
        new = true
        """)
    assert _reparses(out) == doc.to_dict()


def test_insert_aot_entry_at_end_after_sorting_an_entrys_keys() -> None:
    """Same fix via ``insert`` at the tail rather than ``append``."""
    doc = tomlrt.loads("[[p]]\nx = 0\n\n[[p]]\nb = 2\na = 1\n")
    doc["p"][1].sort()
    doc["p"].insert(2, {"new": True})
    out = tomlrt.dumps(doc)
    assert out == td("""
        [[p]]
        x = 0

        [[p]]
        a = 1
        b = 2

        [[p]]
        new = true
        """)
    assert _reparses(out) == doc.to_dict()


def test_append_aot_entry_when_last_entry_owns_nested_aot() -> None:
    """Appending a new ``[[p]]`` must anchor after the last entry's whole
    subtree, including a nested ``[[p.sub]]`` it owns.

    Regression: ``_aot_append_anchor``'s predecessor returned
    ``entry_slots[-1]``, which excludes slots owned by nested AoT entries,
    so the new ``[[p]]`` header was spliced *before* the nested
    ``[[p.sub]]`` block — which re-parse then attributed to the new entry.
    """
    doc = tomlrt.loads(
        td("""
            [[p]]
            a = 1

              [[p.sub]]
              x = 1
            """)
    )
    doc["p"].append({"new": True})
    out = tomlrt.dumps(doc)
    assert out == td("""
        [[p]]
        a = 1

          [[p.sub]]
          x = 1

        [[p]]
        new = true
        """)
    assert _reparses(out) == doc.to_dict()


def test_append_aot_entry_when_nested_aot_is_non_contiguous() -> None:
    """The append anchor must follow a nested ``[[p.sub]]`` even when an
    unrelated ``[other]`` physically separates it from its parent entry."""
    doc = tomlrt.loads(
        td("""
            [[p]]
            x = 1

            [other]
            y = 1

            [[p.sub]]
            z = 1
            """)
    )
    doc["p"].append({"new": True})
    out = tomlrt.dumps(doc)
    assert out == td("""
        [[p]]
        x = 1

        [other]
        y = 1

        [[p.sub]]
        z = 1

        [[p]]
        new = true
        """)
    assert _reparses(out) == doc.to_dict()


def test_append_to_empty_nested_aot_anchors_within_parent_entry() -> None:
    """The first entry materialised in an *empty* nested AoT must land
    inside its owning parent entry, not at the document tail.

    Regression: when a nested ``[[p.sub]]`` AoT held no entries, the
    append anchor fell through to the document tail. With a later
    sibling ``[[p]]`` present, the new ``[[p.sub]]`` header was spliced
    after it, so a re-parse attributed the nested block to the wrong
    parent entry. The append anchor now falls back to the parent
    entry's subtree tail.
    """
    doc = tomlrt.loads(
        td("""
            [[fruits]]
            name = "apple"

            [[fruits]]
            name = "banana"

            [[fruits.varieties]]
            name = "plantain"
            """)
    )
    doc["fruits"][1]["varieties"].pop(0)
    doc["fruits"].append({"new": True})
    doc["fruits"][1]["varieties"].append({"new": -7})
    out = tomlrt.dumps(doc)
    assert out == td("""
        [[fruits]]
        name = "apple"

        [[fruits]]
        name = "banana"

        [[fruits.varieties]]
        new = -7

        [[fruits]]
        new = true
        """)
    assert _reparses(out) == doc.to_dict()


def test_append_to_empty_nested_aot_under_dotted_key_clears_host_body() -> None:
    """An empty nested AoT whose logical parent is a *dotted* (header-less)
    container anchors after its host section's whole body, so a direct
    KV of that section is not captured by the new sub-section.

    Regression: the empty-AoT fallback anchored at the immediate
    (dotted-inner) parent's tail, which sat *before* a sibling direct
    KV of the enclosing ``[fruit]`` section; the new ``[[fruit.apple.seeds]]``
    header then captured that KV on re-parse. The anchor now walks up to
    the host header.
    """
    doc = tomlrt.loads(
        td("""
            [fruit]
            apple.color = "red"

            [[fruit.apple.seeds]]
            size = 2
            """)
    )
    doc["fruit"]["apple"]["seeds"].pop(0)
    doc["fruit"]["k23"] = 1
    doc["fruit"]["apple"]["seeds"].append({"new": 1})
    out = tomlrt.dumps(doc)
    assert out == td("""
        [fruit]
        apple.color = "red"
        k23 = 1

        [[fruit.apple.seeds]]
        new = 1
        """)
    assert _reparses(out) == doc.to_dict()


def test_sort_after_rematerialising_a_subsection_at_a_new_position() -> None:
    """Sorting a container after a dotted leaf re-materialises in place.

    Deleting ``a.b``'s only content then re-adding under it
    re-materialises ``a.b`` as an inline table in place (``a.b = {}`` →
    ``a.b = { k14 = true }``). Because the inline binding never moves, no
    foreign section is straddled and sorting ``tbl.a`` simply orders its
    dotted leaves.
    """
    doc = tomlrt.loads(
        td("""
            [tbl]
            a.b.c = 1
            a.k45 = ""

            [tbl.x]
            y = 1
            """)
    )
    del doc["tbl"]["a"]["b"]["c"]
    doc["tbl"]["a"]["b"]["k14"] = True
    doc["tbl"]["a"].sort()
    out = tomlrt.dumps(doc)
    assert out == td("""
        [tbl]
        a.b = { k14 = true }
        a.k45 = ""

        [tbl.x]
        y = 1
        """)
    assert _reparses(out) == doc.to_dict()


def test_aot_reverse_carries_nested_aot_blocks() -> None:
    """Reversing an AoT must move each entry's nested ``[[t.sub]]`` blocks
    with it.

    Regression: ``renormalise_aot_order`` gathered each entry's
    physical block from ``AoTEntry.entry_slots``, which holds only the
    entry's *own* header + KV slots. A nested ``[[t.sub]]`` lives in
    its own ``AoTEntry``, so its slots were left physically behind and
    re-parented onto whichever entry landed at the old position —
    silently corrupting the data on round-trip.
    """
    src = td("""
        [[t]]
        name = "a"

          [[t.sub]]
          y = 1

        [[t]]
        name = "b"

          [[t.sub]]
          y = 2
        """)
    doc = tomlrt.loads(src)
    doc.aot("t").reverse()
    assert tomlrt.dumps(doc) == (
        td("""
            [[t]]
            name = "b"

              [[t.sub]]
              y = 2

            [[t]]
            name = "a"

              [[t.sub]]
              y = 1
            """)
    )


def test_aot_sort_carries_nested_aot_blocks() -> None:
    """Sorting an AoT must move each entry's nested ``[[t.sub]]`` blocks."""
    src = td("""
        [[t]]
        x = 2

          [[t.sub]]
          y = 20

        [[t]]
        x = 1

          [[t.sub]]
          y = 10
        """)
    doc = tomlrt.loads(src)
    doc.aot("t").sort(key=lambda e: e["x"])
    assert tomlrt.dumps(doc) == (
        td("""
            [[t]]
            x = 1

              [[t.sub]]
              y = 10

            [[t]]
            x = 2

              [[t.sub]]
              y = 20
            """)
    )


def test_aot_sort_carries_varying_nested_aot_counts_and_subtable() -> None:
    """Sorting must carry each entry's own count of nested blocks: zero,
    one, or several ``[[t.sub]]`` entries, and an unrelated ``[t.s]``
    plain sub-table -- not just the uniform one-sub-each shape the
    sibling tests above pin.
    """
    src = td("""
        [[t]]
        x = 3

          [[t.sub]]
          y = 30

          [[t.sub]]
          y = 31

          [t.s]
          z = 3

        [[t]]
        x = 1

        [[t]]
        x = 2

          [[t.sub]]
          y = 20
        """)
    doc = tomlrt.loads(src)
    doc.aot("t").sort(key=lambda e: e["x"])
    out = tomlrt.dumps(doc)
    assert out == (
        td("""
            [[t]]
            x = 1

            [[t]]
            x = 2

              [[t.sub]]
              y = 20

            [[t]]
            x = 3

              [[t.sub]]
              y = 30

              [[t.sub]]
              y = 31

              [t.s]
              z = 3
            """)
    )
    assert _reparses(out) == doc.to_dict()


def test_sort_container_with_synthetic_header_binding_keeps_kv_in_scope() -> None:
    """Sorting a table whose header is synthetic (promoted by an overwrite)
    must keep the synthetic header as the region marker so its body KV
    stays bound to it.

    Regression: ``reorder_container`` skipped synthetic headers when
    choosing the region marker, so the direct KV the synthetic ``[fruit]``
    header binds was spliced ahead of every header — re-parse then
    rebound it to the document root.
    """
    doc = tomlrt.loads(
        td("""
            [fruit.apple]
            [animal]
            [fruit.orange]
            """)
    )
    doc["fruit"]["orange"] = [1, 2]
    doc["fruit"].sort()
    out = tomlrt.dumps(doc)
    assert out == td("""
        [fruit]
        orange = [1, 2]
        [fruit.apple]
        [animal]
        """)
    assert _reparses(out) == doc.to_dict()
    assert doc.to_dict() == {
        "fruit": {"apple": {}, "orange": [1, 2]},
        "animal": {},
    }


def test_sort_container_with_synthetic_header_binding_dotted_kv() -> None:
    """Same fix when the synthetic header binds a *dotted* body KV — the
    header is non-elidable and must stay the region marker."""
    doc = tomlrt.loads(
        td("""
            [fruit.apple]
            [fruit.orange]
            """)
    )
    doc["fruit"]["orange"] = {"deep": {"v": 1}}
    doc["fruit"].sort()
    out = tomlrt.dumps(doc)
    assert out == td("""
        [fruit]
        orange = { deep = { v = 1 } }
        [fruit.apple]
        """)
    assert _reparses(out) == doc.to_dict()
    assert doc.to_dict() == {"fruit": {"apple": {}, "orange": {"deep": {"v": 1}}}}


def test_sort_container_skips_empty_synthetic_header_region_marker() -> None:
    """An empty synthetic header stays put while real children reorder.

    Promote ``fruit`` to a synthetic ``[fruit]`` header, then delete the
    body KV that made it non-elidable. Sorting the remaining child
    sections must leave that now-empty placeholder header untouched
    while still reordering the real children beneath it.
    """
    doc = tomlrt.loads(
        td("""
            [fruit.apple]
            x = 1

            [fruit.banana]
            y = 2
            """)
    )
    doc["fruit"]["orange"] = {"deep": {"v": 1}}
    del doc["fruit"]["orange"]
    doc["fruit"].sort(reverse=True)
    out = tomlrt.dumps(doc)
    assert out == td("""
        [fruit]

        [fruit.banana]
        y = 2

        [fruit.apple]
        x = 1
        """)
    assert _reparses(out) == doc.to_dict()
    assert doc.to_dict() == {"fruit": {"banana": {"y": 2}, "apple": {"x": 1}}}


def test_sort_aot_entry_with_nested_aot_child_reorders_physically() -> None:
    """Sorting an AoT entry whose children include a *nested AoT* must
    reorder the physical CST, not just dict storage.

    Regression: ``reorder_container`` gated membership on
    ``owner_aot_entry is c_owner``, which excludes a nested-AoT child
    (owned by its own entry). The child's block was left in place while
    dict storage was reordered, so the logical key order diverged from
    the re-parsed order. Membership is now bounded by the entry's
    physical extent, so nested descendants move with their key while
    sibling entries stay excluded.
    """
    doc = tomlrt.loads(
        td("""
            [[a]]
            [a.c]
            y = 2
            [[a.b]]
            x = 1
            """)
    )
    doc["a"][0].sort()
    out = tomlrt.dumps(doc)
    assert out == td("""
        [[a]]
        [[a.b]]
        x = 1
        [a.c]
        y = 2
        """)
    assert _reparses(out) == doc.to_dict()
    assert list(doc["a"][0].keys()) == list(tomlrt.loads(out)["a"][0].keys())


def test_sort_aot_entry_excludes_sibling_entries() -> None:
    """The physical-extent bound must still exclude *sibling* AoT entries:
    sorting entry 0 reorders only its own children and leaves later
    entries of the same array untouched."""
    doc = tomlrt.loads(
        td("""
            [[a]]
            [a.c]
            y = 2
            [[a.b]]
            x = 1
            [[a]]
            [a.z]
            w = 9
            """)
    )
    doc["a"][0].sort()
    out = tomlrt.dumps(doc)
    assert out == td("""
        [[a]]
        [[a.b]]
        x = 1
        [a.c]
        y = 2
        [[a]]
        [a.z]
        w = 9
        """)
    assert _reparses(out) == doc.to_dict()
    assert list(doc["a"][1].keys()) == ["z"]


def test_overwrite_section_with_value_preserves_neighbour_leading() -> None:
    """Overwriting a section with a value synthesises a ``[fruit]`` header
    and relocates it back to the replaced binding's position. The
    untouched sibling section it was momentarily synthesised in front of
    must keep its original leading — no spurious inter-section blank line
    injected before it (and none stranded at the top of the file)."""
    doc = tomlrt.loads(
        td("""
            z = 0
            [fruit.apple]
            [fruit.orange]
            """)
    )
    doc["fruit"]["orange"] = {"deep": {"v": 1}}
    out = tomlrt.dumps(doc)
    assert out == td("""
        z = 0
        [fruit.apple]
        [fruit]
        orange = { deep = { v = 1 } }
        """)
    assert _reparses(out) == doc.to_dict()


def test_overwrite_first_section_with_value_successor_is_synth_anchor() -> None:
    """The replaced binding is the *first* descendant, so its captured
    successor (``[fruit.apple]``) is the very sibling synthesis displaces.
    The successor-restore and the displaced-restore target the same slot
    but are mutually exclusive (block-before vs block-not-before), so the
    blank line the user wrote between the two sections survives exactly
    once and ``[fruit.apple]`` is otherwise untouched."""
    doc = tomlrt.loads(
        td("""
            [fruit.orange]

            [fruit.apple]
            """)
    )
    doc["fruit"]["orange"] = {"deep": {"v": 1}}
    out = tomlrt.dumps(doc)
    assert out == td("""
        [fruit]
        orange = { deep = { v = 1 } }

        [fruit.apple]
        """)
    assert _reparses(out) == doc.to_dict()
    """Sorting a container whose owned span has an interleaved foreign
    (outer-scope) key must not push that key into a sub-section's scope.

    Regression: ``reorder_container`` gathered the non-contiguous
    ``many.dots`` runs across the root key ``kfor``, shoving ``kfor``
    after ``[many.dots.sub]`` — re-parse then bound it under
    ``many.dots.sub``. The foreign key is now hoisted to the region
    head and ``many`` is gathered contiguously.
    """
    doc = tomlrt.loads(
        td("""
            many.dots.a = 1
            many.k = 2
            kfor = 99
            [many.dots.sub]
            x = 1
            """)
    )
    doc["many"].sort()
    out = tomlrt.dumps(doc)
    assert out == td("""
        kfor = 99
        many.k = 2
        many.dots.a = 1
        [many.dots.sub]
        x = 1
        """)
    assert _reparses(out) == doc.to_dict()
    assert doc.to_dict() == {
        "many": {"k": 2, "dots": {"a": 1, "sub": {"x": 1}}},
        "kfor": 99,
    }


def test_sort_container_sinks_interleaved_foreign_header() -> None:
    """An interleaved foreign *header* must sink past the reordered run,
    not hoist ahead of it.

    A foreign KV in c's containing scope hoists to the region head to
    keep its scope (see the ``kfor`` test). A foreign *header*
    establishes its own scope and must instead stay after the run: a
    genuine reorder of ``tbl.a``'s two dotted leaves, with ``[tbl.x]``
    sitting between the leaves and ``tbl.a``'s own sub-section, must
    move ``[tbl.x]`` to the back. Hoisting it to the front (the old
    blanket rule) pulled it ahead of the dotted leaves ``a.k45`` /
    ``a.k46``, which a re-parse then captured under ``[tbl.x]``.
    """
    doc = tomlrt.loads(
        td("""
            [tbl]
            a.k46 = 1
            a.k45 = 2

            [tbl.x]
            y = 9

            [tbl.a.b]
            k14 = true
            """)
    )
    doc["tbl"]["a"].sort()
    out = tomlrt.dumps(doc)
    assert out == td("""
        [tbl]
        a.k45 = 2
        a.k46 = 1

        [tbl.a.b]
        k14 = true

        [tbl.x]
        y = 9
        """)
    assert _reparses(out) == doc.to_dict()
    assert doc.to_dict() == {
        "tbl": {"a": {"k45": 2, "k46": 1, "b": {"k14": True}}, "x": {"y": 9}},
    }


def test_sort_container_keeps_every_mixed_keys_leaf_ahead_of_every_section() -> None:
    """A reorder of two mixed keys must not sink a leaf under a header.

    ``b`` and ``c`` each own a dotted leaf *and* a section, so each
    contributes both a leaf block and a structural block to the reorder,
    and the two blocks of one key straddle the other key's. Keeping
    every leaf ahead of every section is the only thing stopping
    ``c.x = 1`` from being emitted inside ``[t.b.z]``, where a re-parse
    binds it to ``t.b.z`` rather than to ``t.c``. The logical view would
    go on claiming otherwise, so the round-trip is what catches it.
    """
    doc = tomlrt.loads(
        td("""
            [t]
            c.x = 1
            b.x = 2

            [t.c.z]
            k = 3

            [t.b.z]
            k = 4
            """)
    )
    doc["t"].sort()
    out = tomlrt.dumps(doc)
    assert out == td("""
        [t]
        b.x = 2
        c.x = 1

        [t.b.z]
        k = 4

        [t.c.z]
        k = 3
        """)
    assert _reparses(out) == doc.to_dict()
    assert doc.to_dict() == {
        "t": {"b": {"x": 2, "z": {"k": 4}}, "c": {"x": 1, "z": {"k": 3}}},
    }


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
    out = tomlrt.dumps(doc)
    assert out == td("""
        [[t]]
        x = 1
        [[t]]
        x = 3
        """)
    assert _reparses(out)["t"] == [{"x": 1}, {"x": 3}]


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
    out = tomlrt.dumps(doc)
    assert out == td("""
        [[items]]
        name = "a"

        [[items]]
        name = "B"

        [[items]]
        name = "C"
        """)
    assert _reparses(out)["items"] == [
        {"name": "a"},
        {"name": "B"},
        {"name": "C"},
    ]


def test_aot_slice_replace_before_surviving_entries() -> None:
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
    doc.aot("items")[0:1] = [{"name": "A"}]
    assert tomlrt.dumps(doc) == td("""
        [[items]]
        name = "A"

        [[items]]
        name = "b"

        [[items]]
        name = "c"
        """)


def test_aot_slice_insert_before_surviving_entries() -> None:
    doc = tomlrt.loads(
        td("""
            [[items]]
            name = "a"

            [[items]]
            name = "c"
            """)
    )
    doc.aot("items")[1:1] = [{"name": "b"}]
    assert tomlrt.dumps(doc) == td("""
        [[items]]
        name = "a"

        [[items]]
        name = "b"

        [[items]]
        name = "c"
        """)


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


def test_aot_empty_mutations_are_noops() -> None:
    doc = tomlrt.loads("[[items]]\n")
    items = doc.aot("items")
    items.clear()
    src = tomlrt.dumps(doc)
    del items[:]
    items.clear()
    assert tomlrt.dumps(doc) == src


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
    out = tomlrt.dumps(doc)
    assert out == 'xs = ["a", "bb", "ccc"]\n'
    assert _reparses(out) == {"xs": ["a", "bb", "ccc"]}


def test_array_sort_empty_is_noop() -> None:
    src = "xs = []\n"
    doc = tomlrt.loads(src)
    doc.array("xs").sort()
    assert tomlrt.dumps(doc) == src


def test_array_imul_zero_clears() -> None:
    doc = tomlrt.loads("xs = [1, 2, 3]\n")
    xs = doc.array("xs")
    xs *= 0
    assert list(xs) == []
    out = tomlrt.dumps(doc)
    assert out == "xs = []\n"
    assert _reparses(out) == {"xs": []}


def test_array_imul_negative_clears() -> None:
    doc = tomlrt.loads("xs = [1, 2]\n")
    xs = doc.array("xs")
    xs *= -3
    assert list(xs) == []


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
    with pytest.raises(tomlrt.TOMLError, match="cannot store an array-of-tables"):
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
    with pytest.raises(ValueError, match="must not be empty"):
        doc.install("", 1)


def test_install_rejects_empty_tuple_path() -> None:
    doc = tomlrt.loads("")
    with pytest.raises(ValueError, match="must not be empty"):
        doc.install((), 1)


def test_install_rejects_string_path_with_empty_segment() -> None:
    doc = tomlrt.loads("")
    with pytest.raises(ValueError, match="empty segment"):
        doc.install("a..b", 1)


def test_install_rejects_tuple_path_with_empty_segment() -> None:
    doc = tomlrt.loads("")
    with pytest.raises(ValueError, match="empty segment"):
        doc.install(("a", ""), 1)


def test_install_rejects_non_string_path() -> None:
    doc = tomlrt.loads("")
    with pytest.raises(TypeError, match="key path must be str or sequence"):
        doc.install(123, 1)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]


def test_entry_rejects_empty_string_path() -> None:
    doc = tomlrt.loads("x = 1\n")
    with pytest.raises(ValueError, match="must not be empty"):
        doc.entry("")


def test_entry_rejects_empty_sequence_path() -> None:
    doc = tomlrt.loads("x = 1\n")
    with pytest.raises(ValueError, match="must not be empty"):
        doc.entry([])


def test_entry_rejects_string_path_with_empty_segment() -> None:
    doc = tomlrt.loads("[a]\ny = 1\n")
    with pytest.raises(ValueError, match="empty segment"):
        doc.entry("a..y")


def test_entry_rejects_sequence_path_with_empty_segment() -> None:
    doc = tomlrt.loads("[a]\ny = 1\n")
    with pytest.raises(ValueError, match="empty segment"):
        doc.entry(["a", "", "y"])


def test_typed_accessors_reject_empty_path() -> None:
    doc = tomlrt.loads("x = 1\n")
    with pytest.raises(ValueError, match="must not be empty"):
        doc.table("")
    with pytest.raises(ValueError, match="must not be empty"):
        doc.array([])
    with pytest.raises(ValueError, match="must not be empty"):
        doc.aot("")


def test_get_entry_propagates_empty_path_error() -> None:
    # An empty path is a malformed argument, not a missing key, so the
    # ``get_*`` variants must raise rather than swallow it as a default.
    doc = tomlrt.loads("x = 1\n")
    with pytest.raises(ValueError, match="must not be empty"):
        doc.get_entry("", default=None)
    with pytest.raises(ValueError, match="must not be empty"):
        doc.get_table("", default=None)


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


def test_ensure_table_under_implicit_parent_lands_after_section_body() -> None:
    # ``ensure_table(("s","a","b"))`` synthesises ``[s.a.b]``. ``a`` is
    # implicit, with its dotted KV ``a.c`` interleaved among ``[s]``'s
    # other KVs (``d``). The header re-parents everything after it, so it
    # must land after ``[s]``'s whole body (``d = 4``), not between the
    # dotted-key siblings — otherwise ``d`` is captured under ``[s.a.b]``.
    doc = tomlrt.loads(
        td("""
            [s]
            a.c = 3
            d = 4
            """)
    )
    doc.ensure_table(("s", "a", "b"))
    out = tomlrt.dumps(doc)
    assert out == td("""
        [s]
        a.c = 3
        d = 4

        [s.a.b]
        """)
    assert _reparses(out) == doc.to_dict() == {"s": {"a": {"c": 3, "b": {}}, "d": 4}}


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


def test_install_section_through_scalar_intermediate_raises_tomlerror() -> None:
    """``install("a.b.c", section)`` where ``a`` is a scalar must fail loudly.

    Raises the same `TOMLError`, with the same message, that
    `ensure_table` raises for the identical scenario: both walk the
    same existing-prefix logic and neither can descend through a
    non-table value.
    """
    doc = tomlrt.loads("a = 1\n")
    with pytest.raises(
        tomlrt.TOMLError, match="existing value at 'a' is not section-backed"
    ):
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


def test_aot_setitem_clone_terminates_foreign_document_tail() -> None:
    src = tomlrt.loads(
        td("""
        [[s]]
        x = 9""")
    )
    dst = tomlrt.loads(
        td("""
        [[d]]
        x = 1

        [[d]]
        x = 2
        """)
    )
    dst.aot("d")[0] = src.aot("s")[0]
    assert tomlrt.dumps(dst) == td("""
        [[d]]
        x = 9

        [[d]]
        x = 2
    """)


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
    out = tomlrt.dumps(doc)
    assert out == td("""
        [[pkg]]
        name = "foo"
        """)
    assert _reparses(out) == {"pkg": [{"name": "foo"}]}


def test_aot_add_default_empty_returns_blank_entry_for_population() -> None:
    doc = tomlrt.loads("")
    doc["pkg"] = AoT()
    aot = doc["pkg"]
    pkg = aot.add()
    assert dict(pkg) == {}
    pkg["name"] = "bar"
    pkg["dep"] = Table.section({"x": 1})
    out = tomlrt.dumps(doc)
    assert out == td("""
        [[pkg]]
        name = "bar"

        [pkg.dep]
        x = 1
        """)
    assert _reparses(out) == {
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
    out = tomlrt.dumps(doc)
    assert out == td("""
        [[pkg]]
        name = "a"
        version = "1.0"

        [[pkg]]
        name = "b"

        [[pkg]]
        name = "c"
        """)
    assert _reparses(out) == {
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
    out = tomlrt.dumps(doc)
    assert out == "xs = [[1, 2, 3]]\n"
    parsed = _reparses(out)
    assert parsed == {"xs": [[1, 2, 3]]}


def test_array_extend_mixed_python_containers() -> None:
    doc = tomlrt.loads("xs = []\n")
    arr = doc.array("xs")
    arr.extend([{"a": 1}, [1, 2], "three"])
    out = tomlrt.dumps(doc)
    assert out == 'xs = [{ a = 1 }, [1, 2], "three"]\n'
    parsed = _reparses(out)
    assert parsed == {"xs": [{"a": 1}, [1, 2], "three"]}


def test_array_insert_dict() -> None:
    doc = tomlrt.loads("xs = [1, 3]\n")
    arr = doc.array("xs")
    arr.insert(1, {"k": "v"})
    out = tomlrt.dumps(doc)
    assert out == 'xs = [1, { k = "v" }, 3]\n'
    parsed = _reparses(out)
    assert parsed == {"xs": [1, {"k": "v"}, 3]}


def test_array_setitem_replaces_with_dict() -> None:
    doc = tomlrt.loads("xs = [1, 2, 3]\n")
    arr = doc.array("xs")
    arr[1] = {"k": "v"}
    out = tomlrt.dumps(doc)
    assert out == 'xs = [1, { k = "v" }, 3]\n'
    parsed = _reparses(out)
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


def test_aot_insert_negative_index_matches_list_semantics() -> None:
    """``AoT.insert(-1, x)`` inserts BEFORE the last entry, like ``list``.

    Regression: ``insert`` appended the new entry first and then
    normalised a negative index against the post-append length, landing
    one slot too high. ``insert(-1, x)`` was silently equivalent to
    ``append(x)``.
    """
    src = td("""
        [[a]]
        x = 1

        [[a]]
        x = 2

        [[a]]
        x = 3
        """)
    doc = tomlrt.loads(src)
    new = tomlrt.Table()
    new["x"] = 99
    doc.aot("a").insert(-1, new)
    assert tomlrt.dumps(doc) == td("""
        [[a]]
        x = 1

        [[a]]
        x = 2

        [[a]]
        x = 99

        [[a]]
        x = 3
        """)


def test_aot_insert_negative_two_matches_list_semantics() -> None:
    """``insert(-2, x)`` lands one slot earlier than ``insert(-1, x)``."""
    src = td("""
        [[a]]
        x = 1

        [[a]]
        x = 2

        [[a]]
        x = 3
        """)
    doc = tomlrt.loads(src)
    new = tomlrt.Table()
    new["x"] = 99
    doc.aot("a").insert(-2, new)
    assert tomlrt.dumps(doc) == td("""
        [[a]]
        x = 1

        [[a]]
        x = 99

        [[a]]
        x = 2

        [[a]]
        x = 3
        """)


def test_aot_insert_before_value_equal_entry() -> None:
    """``insert`` places by index even when entries compare equal.

    Regression: the reorder was guarded by ``new_order != list(self)``,
    a value comparison, while the reorder itself is keyed on identity.
    Value-equal entries made the guard report "already in order" and the
    new entry stayed where it had been appended, at the end.
    """
    src = td("""
        [[a]] # first
        x = 1

        [[a]] # second
        x = 1
        """)
    doc = tomlrt.loads(src)
    doc.aot("a").insert(0, {"x": 1})
    assert tomlrt.dumps(doc) == td("""
        [[a]]
        x = 1

        [[a]] # first
        x = 1

        [[a]] # second
        x = 1
        """)


def test_aot_slice_assign_before_value_equal_entry() -> None:
    """Slice assignment places by index even when entries compare equal."""
    src = td("""
        [[a]] # first
        x = 1

        [[a]] # second
        x = 1
        """)
    doc = tomlrt.loads(src)
    aot = doc.aot("a")
    aot[0:1] = [{"x": 1}]
    assert tomlrt.dumps(doc) == td("""
        [[a]]
        x = 1

        [[a]] # second
        x = 1
        """)


def test_aot_tail_insert_leaves_interleaved_section_alone() -> None:
    """An insert at the tail must not renormalise the AoT's layout.

    Entries need not be physically contiguous; reordering gathers them
    together and pushes intervening sections after them. An insert that
    needs no reordering must not trigger that.
    """
    src = td("""
        [[a]]
        x = 1

        [b]
        y = 9

        [[a]]
        x = 2
        """)
    doc = tomlrt.loads(src)
    doc.aot("a").insert(2, {"x": 3})
    assert tomlrt.dumps(doc) == td("""
        [[a]]
        x = 1

        [b]
        y = 9

        [[a]]
        x = 2

        [[a]]
        x = 3
        """)


def test_aot_slice_assign_at_tail_leaves_interleaved_section_alone() -> None:
    """A slice assignment already at the tail must not renormalise layout.

    The "does this need reordering?" flag has to be captured after the
    deletion and before the append, when ``len(self)`` is the survivor
    count; taking it at any other moment gathers the entries together
    and pushes the intervening section past them.
    """
    src = td("""
        [[a]]
        x = 1

        [b]
        y = 9

        [[a]]
        x = 2

        [[a]]
        x = 3
        """)
    doc = tomlrt.loads(src)
    doc.aot("a")[2:3] = [{"x": 9}]
    assert tomlrt.dumps(doc) == td("""
        [[a]]
        x = 1

        [b]
        y = 9

        [[a]]
        x = 2

        [[a]]
        x = 9
        """)


def test_aot_slice_assign_multiple_at_negative_bounds() -> None:
    """A multi-entry replacement at negative bounds matches list order.

    ``start`` is relative to the pre-deletion length, and the new
    entries go in at consecutive offsets from it.
    """
    src = td("""
        [[a]]
        x = 0

        [[a]]
        x = 1

        [[a]]
        x = 2

        [[a]]
        x = 3
        """)
    doc = tomlrt.loads(src)
    doc.aot("a")[-2:-1] = [{"x": 8}, {"x": 9}]
    assert tomlrt.dumps(doc) == td("""
        [[a]]
        x = 0

        [[a]]
        x = 1

        [[a]]
        x = 8

        [[a]]
        x = 9

        [[a]]
        x = 3
        """)


def test_aot_slice_delete_leaves_interleaved_section_alone() -> None:
    """Assigning an empty slice deletes without renormalising layout."""
    src = td("""
        [[a]]
        x = 1

        [b]
        y = 9

        [[a]]
        x = 2
        """)
    doc = tomlrt.loads(src)
    aot = doc.aot("a")
    aot[0:1] = []
    assert tomlrt.dumps(doc) == td("""
        [b]
        y = 9

        [[a]]
        x = 2
        """)
    assert [t["x"] for t in aot] == [2]


def test_aot_extend_self_duplicates_once() -> None:
    """``aot.extend(aot)`` matches list semantics: duplicate once, no hang.

    Regression: the implementation iterated ``values`` while appending
    to ``self``. When ``values is self`` the iteration kept seeing the
    just-appended entries and never terminated.
    """
    src = td("""
        [[t]]
        a = 1

        [[t]]
        a = 2
        """)
    doc = tomlrt.loads(src)
    aot = doc.aot("t")
    aot.extend(aot)
    assert len(aot) == 4
    assert [t["a"] for t in aot] == [1, 2, 1, 2]


_BAD_BODY: Any = {"q": object()}
"""A mapping body carrying a value TOML cannot represent."""


@pytest.mark.parametrize(
    "invalid",
    [
        {"x": object()},
        {"t": {"u": object()}},
        {"xs": [1, [2, object()]]},
        {"sub": AoT([_BAD_BODY])},
        {"sub": Table.section(_BAD_BODY)},
    ],
    ids=["scalar", "inline-table", "array", "aot", "section-table"],
)
def test_aot_extend_invalid_entry_is_atomic(invalid: Any) -> None:
    """An invalid leaf anywhere in a later entry rejects the whole call."""
    src = td("""
        [[a]]
        x = 1
        """)
    doc = tomlrt.loads(src)
    aot = doc.aot("a")
    values: Any = [{"x": 2}, invalid]

    with pytest.raises(TypeError):
        aot.extend(values)

    assert aot.to_list() == [{"x": 1}]
    rendered = tomlrt.dumps(doc)
    assert rendered == src
    assert _reparses(rendered) == {"a": [{"x": 1}]}


def test_aot_setitem_slice_invalid_entry_is_atomic() -> None:
    src = td("""
        [[a]]
        x = 1

        [[a]]
        x = 2
        """)
    doc = tomlrt.loads(src)
    aot = doc.aot("a")
    values: Any = [{"x": 3}, {"x": object()}]

    with pytest.raises(TypeError):
        aot[:] = values

    assert aot.to_list() == [{"x": 1}, {"x": 2}]
    rendered = tomlrt.dumps(doc)
    assert rendered == src
    assert _reparses(rendered) == {"a": [{"x": 1}, {"x": 2}]}


def test_aot_iadd_self_duplicates_once() -> None:
    """``aot += aot`` matches list semantics: duplicate once, no hang."""
    src = td("""
        [[t]]
        a = 1

        [[t]]
        a = 2
        """)
    doc = tomlrt.loads(src)
    aot = doc.aot("t")
    aot += aot
    assert len(aot) == 4
    assert [t["a"] for t in aot] == [1, 2, 1, 2]


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


def test_aot_append_skips_structural_only_sibling_when_inheriting_indent() -> None:
    """A new entry's first KV inherits indent from the nearest earlier
    sibling that actually has a direct KV."""
    src = td("""
        [[t]]
            z = 0

        [[t]]
        [t.sub]
        x = 1

        [[t]]
        """)
    doc = tomlrt.loads(src)
    doc.aot("t")[2]["y"] = 2
    out = tomlrt.dumps(doc)
    assert out == td("""
        [[t]]
            z = 0

        [[t]]
        [t.sub]
        x = 1

        [[t]]
            y = 2
        """)
    assert _reparses(out) == doc.to_dict()


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


# ---------------------------------------------------------------------------
# Dotted-KV insert: leading trivia comes from the host's last body KV
# ---------------------------------------------------------------------------


def test_dotted_add_at_document_head_ahead_of_sections() -> None:
    """A root-hosted dotted KV inherits from the root's last KV.

    The sections that follow are the bulk of the root's bookkeeping but
    none of them is a body KV, so they contribute nothing to the choice.
    """
    src = td("""
        a.b = 1

        [s0]
        x = 1

        [s1]
        y = 2
        """)
    doc = tomlrt.loads(src)
    doc["a"]["c"] = 3
    out = tomlrt.dumps(doc)
    assert out == td("""
        a.b = 1
        a.c = 3

        [s0]
        x = 1

        [s1]
        y = 2
        """)
    assert _reparses(out) == {"a": {"b": 1, "c": 3}, "s0": {"x": 1}, "s1": {"y": 2}}


def test_dotted_add_in_middle_section_with_subsections_following() -> None:
    """Descendant headers filed on the host don't disturb the indent."""
    src = td("""
        [s0]
        x = 1

        [s1]
          a.b = 1

        [s1.sub]
        y = 2

        [s2]
        z = 3
        """)
    doc = tomlrt.loads(src)
    doc["s1"]["a"]["c"] = 4
    out = tomlrt.dumps(doc)
    assert out == td("""
        [s0]
        x = 1

        [s1]
          a.b = 1
          a.c = 4

        [s1.sub]
        y = 2

        [s2]
        z = 3
        """)
    assert _reparses(out)["s1"] == {"a": {"b": 1, "c": 4}, "sub": {"y": 2}}


def test_dotted_add_at_document_tail_mirrors_host_kv_after_dotted_body() -> None:
    """The host's *last* KV sets the spacing, even when it is not the
    dotted region's own tail: the new slot groups with its peers but
    keeps the host's most recent blank-line convention.
    """
    src = td("""
        [s0]
        x = 1

        [s1]
        a.b = 1

        z = 2
        """)
    doc = tomlrt.loads(src)
    doc["s1"]["a"]["c"] = 3
    out = tomlrt.dumps(doc)
    assert out == td("""
        [s0]
        x = 1

        [s1]
        a.b = 1

        a.c = 3

        z = 2
        """)
    assert _reparses(out)["s1"] == {"a": {"b": 1, "c": 3}, "z": 2}


def test_dotted_add_mirrors_blank_gap_but_not_comment_block() -> None:
    """A comment block in the peer's leading is separation, not content
    to duplicate: the new slot inherits the blank line above it and the
    peer's indent, and nothing else.
    """
    src = td("""
        [s]
          a.b = 1

          # why c matters
          a.c = 2
        """)
    doc = tomlrt.loads(src)
    doc["s"]["a"]["d"] = 3
    out = tomlrt.dumps(doc)
    assert out == td("""
        [s]
          a.b = 1

          # why c matters
          a.c = 2

          a.d = 3
        """)
    assert _reparses(out)["s"] == {"a": {"b": 1, "c": 2, "d": 3}}


def test_dotted_add_after_commented_peer_without_blank_gap() -> None:
    src = td("""
        [s]
        \t# about b
        \ta.b = 1
        """)
    doc = tomlrt.loads(src)
    doc["s"]["a"]["c"] = 2
    out = tomlrt.dumps(doc)
    assert out == td("""
        [s]
        \t# about b
        \ta.b = 1
        \ta.c = 2
        """)
    assert _reparses(out)["s"] == {"a": {"b": 1, "c": 2}}


def test_promote_inline_installs_dotted_entries_into_empty_section() -> None:
    """The first dotted entry lands in a section with no body KV yet, so
    it has no peer to inherit from; the rest follow it.
    """
    src = td("""
        [s]
        a = {b.c = 1, b.d = 2}
        z = 3
        """)
    doc = tomlrt.loads(src)
    doc["s"].promote_inline("a")
    out = tomlrt.dumps(doc)
    assert out == td("""
        [s]
        z = 3

        [s.a]
        b.c = 1
        b.d = 2
        """)
    assert _reparses(out)["s"] == {"z": 3, "a": {"b": {"c": 1, "d": 2}}}


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
    assert out == "obj = { }\n"
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
    # ``s.a.b`` is now empty but still a live key; its dotted origin
    # re-materialises it as an inline table ``a.b = {}`` in place — an
    # inline binding re-parents nothing, so it stays among ``[s]``'s KVs.
    assert out == td("""
        [s]
        a.b = {}
        a.c = 3
        d = 4
        """)
    assert _reparses(out) == {"s": {"a": {"b": {}, "c": 3}, "d": 4}}


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


def test_clone_section_with_forward_declared_nested_table_keeps_content() -> None:
    # `[a.b.c]` is written *before* `a`'s own header — legal TOML — so
    # `a`'s doc-stream-first owned slot is the nested header, not its own.
    src = td("""
        [a.b.c]
        answer = 42

        [a]
        better = 43
        """)
    doc = tomlrt.loads(src)
    doc2 = tomlrt.loads("")
    doc2["moved"] = doc["a"]
    out = tomlrt.dumps(doc2)
    assert out == td("""
        [moved.b.c]
        answer = 42

        [moved]
        better = 43
        """)
    assert _reparses(out) == {"moved": {"better": 43, "b": {"c": {"answer": 42}}}}


def test_clone_with_forward_declared_nested_supports_later_insert_into_it() -> None:
    """A cloned forward-declared nested descendant's binding must reach
    all the way to the document root, not just up to the clone's own
    section — otherwise a later insert into the (implicit) intermediate
    between the two can't find its anchor among the root's refs.
    """
    src = td("""
        [a.b.c]
        answer = 42

        [a]
        better = 43
        """)
    doc = tomlrt.loads(src)
    doc2 = tomlrt.loads("x = 1\n")
    doc2["y"] = doc["a"]
    doc2["y"]["b"]["newkey"] = 99
    out = tomlrt.dumps(doc2)
    assert out == td("""
        x = 1

        [y.b]
        newkey = 99

        [y.b.c]
        answer = 42

        [y]
        better = 43
        """)
    assert _reparses(out) == doc2.to_dict()


def test_clone_section_with_forward_declared_nested_past_foreign_section() -> None:
    # As above, but with an unrelated section physically between the
    # forward-declared nested descendant and `a`'s own header, so
    # recovering doc-stream order must skip a foreign slot while
    # walking backward from `a`'s header, not just forward from it.
    src = td("""
        [a.b]
        x = 1

        [unrelated]
        y = 2

        [a]
        better = 2
        """)
    doc = tomlrt.loads(src)
    doc2 = tomlrt.loads("")
    doc2["moved"] = doc["a"]
    out = tomlrt.dumps(doc2)
    assert out == td("""
        [moved.b]
        x = 1

        [moved]
        better = 2
        """)
    assert _reparses(out) == {"moved": {"better": 2, "b": {"x": 1}}}


def test_overwrite_scalar_anchors_past_forward_declared_nested_predecessor() -> None:
    """A forward-declared descendant cannot be the replacement anchor."""
    doc = tomlrt.loads(
        td("""
        [a.b.c]
        answer = 42

        [a]
        better = 43
        """)
    )
    del doc["a"]["b"]["c"]
    doc["a"] = 999
    out = tomlrt.dumps(doc)
    assert out == "\na = 999\n"
    assert _reparses(out) == doc.to_dict()


def test_overwrite_scalar_survives_reinstall_demoting_its_own_header() -> None:
    """Overwriting the sole direct KV of a *synthetic* header with a
    section-style value must not lose that value when the reinstall
    itself empties and demotes the header the overwrite was anchored to.

    ``t["k66"] = True`` on an implicit table synthesises a `[t]` header
    to host it (its only body content). Overwriting `k66` with a section
    deletes that KV, leaving `[t]` empty; the reinstall's clone install
    then demotes that now-empty synthetic header as a matter of course.
    ``reposition_install`` must detect that its saved anchor was
    unlinked by this and leave the freshly-installed content where the
    reinstall placed it, rather than anchoring it to a dead slot.
    """
    doc = tomlrt.loads(
        td("""
        [[albums.songs]]
        new = ""

        [[albums.songs]]
        name = "Glory Days"
        """)
    )
    foreign = tomlrt.loads(
        td("""
        [[people]]
        first_name = "Bruce"
        last_name = "Springsteen"

        [[people]]
        first_name = "Bob"
        last_name = "Seger"
        """)
    )
    albums = doc["albums"]
    albums["k66"] = True
    albums["k66"] = foreign["people"][1]
    out = tomlrt.dumps(doc)
    assert out == td("""
        [[albums.songs]]
        new = ""

        [[albums.songs]]
        name = "Glory Days"

        [albums.k66]
        first_name = "Bob"
        last_name = "Seger"
        """)
    assert _reparses(out) == doc.to_dict()


def test_overwrite_ancestor_into_own_descendant_snapshots_before_delete() -> None:
    """``t[k] = ancestor`` as an *overwrite* (not a new key) where
    ``ancestor`` is one of ``t``'s own ancestors must not lose the part
    of ``ancestor`` that overwriting ``t[k]`` itself deletes.

    The structural-overwrite path deletes the old ``t[k]`` subtree
    before cloning from the source. When the source is ``t``'s own
    ancestor and ``k`` is (transitively) one of that ancestor's own
    children, the delete removes content out from under the source
    before it's read, silently truncating the clone. Must snapshot the
    source before anything is deleted.
    """
    doc = tomlrt.loads("[x.k16]\nw = {}\n")
    node = doc["x"]["k16"]
    node["w"] = node
    out = tomlrt.dumps(doc)
    assert out == td("""
        [x.k16]

        [x.k16.w]
        [x.k16.w.w]
        """)
    assert doc.to_dict() == {"x": {"k16": {"w": {"w": {}}}}}
    assert _reparses(out) == doc.to_dict()


def test_clone_section_from_no_final_newline_source_gets_separator() -> None:
    """A cloned section whose source document had no final newline (it
    was previously the very last thing there) must get one when spliced
    anywhere but the destination's own new tail — otherwise it runs
    into whatever follows on the same physical line.
    """
    doc = tomlrt.loads(
        td("""
        [d.e.f]
        [g.h.i]
        """)
    )
    other = tomlrt.loads("# No newline at end of file.\n[table]")
    doc["d"]["e"]["k27"] = other["table"]
    out = tomlrt.dumps(doc)
    assert out == td("""
        [d.e.f]
        # No newline at end of file.
        [d.e.k27]
        [g.h.i]
        """)
    assert _reparses(out) == doc.to_dict()


def test_overwrite_with_no_final_newline_clone_moved_to_anchor_gets_separator() -> None:
    """As above, but the overwrite's position-preserving move (not the
    initial clone-install) is what relocates the no-final-newline
    content away from the document's tail.

    Overwriting `a.b.c` (the doc's very first binding) with a clone
    from a no-final-newline source installs the clone at whatever
    position is natural for a fresh key first, then
    ``reposition_install`` moves it back to `a.b.c`'s original anchor
    (doc head here) to preserve its physical position — that move must
    also ensure a trailing newline if something now follows.
    """
    doc = tomlrt.loads('[a.b.c]\n[a."b.c"]\n')
    other = tomlrt.loads("[table]")
    doc["a"]["b"]["c"] = other["table"]
    out = tomlrt.dumps(doc)
    assert out == '[a.b.c]\n[a."b.c"]\n'
    assert _reparses(out) == doc.to_dict()


def test_overwrite_with_scattered_implicit_source_skips_reposition() -> None:
    """Overwriting an existing key with an implicit source that has both
    direct KVs and structural children must not crash.

    ``_install_attached_subtree`` hosts the source's direct KVs at the
    destination's nearest header but gives structural children their
    own section anchor — the two kinds land in physically disjoint
    doc-stream regions, not one contiguous block.
    ``reposition_install``'s position-preserving move assumes a single
    contiguous span; it must detect this and leave the install where it
    landed instead of asserting.
    """
    doc = tomlrt.loads(
        td("""
        [name]
        first = "Tom"
        last = "Preston-Werner"

        [animal]
        type.name = "pug"
        type.k56 = 1

        [animal.type.k20.personal]
        email = "a@b.com"

        [animal.type.k20.work]
        name = "x"
        """)
    )
    doc["name"]["first"] = doc["animal"]["type"]
    out = tomlrt.dumps(doc)
    assert out == td("""
        [name]
        last = "Preston-Werner"
        first.name = "pug"
        first.k56 = 1

        [name.first.k20.personal]
        email = "a@b.com"

        [name.first.k20.work]
        name = "x"

        [animal]
        type.name = "pug"
        type.k56 = 1

        [animal.type.k20.personal]
        email = "a@b.com"

        [animal.type.k20.work]
        name = "x"
        """)
    assert _reparses(out) == doc.to_dict()
    assert doc["name"]["first"].to_dict() == doc["animal"]["type"].to_dict()


def test_overwrite_first_root_key_with_implicit_source_skips_root_reposition() -> None:
    """Replacing the first root KV with an implicit source whose cloned
    block starts with a dotted KV must not try to move that dotted KV
    back to the doc head across a header boundary."""
    src = tomlrt.loads(
        td("""
        src.q = 2

        [src.child]
        z = 1
        """)
    )
    dst = tomlrt.loads(
        td("""
        a = 0

        [tail]
        y = 1
        """)
    )
    dst["a"] = src["src"]
    out = tomlrt.dumps(dst)
    assert out == td("""
        a.q = 2

        [a.child]
        z = 1

        [tail]
        y = 1
        """)
    assert _reparses(out) == dst.to_dict()


def test_overwrite_ancestor_with_own_nested_aot_preserves_nested_entries() -> None:
    """Overwriting a key with its own descendant AoT (Case A: descendant
    into ancestor) must preserve nested `[[a.x]]` entries living inside
    that AoT's own entries, not just their own direct/dotted content.

    ``_attach_aot``'s private-orphan rehome path gathers each preserved
    entry's slots via ``clone_aot_entry``'s bare-``AoTEntry`` branch,
    which only sees the entry's *own* ``entry_slots`` membership, not
    slots owned by AoT entries nested inside its body. The full subtree
    must be gathered while the entry is still live, before
    ``_reset_table_for_rehome`` clears the ``_refs`` that gathering
    depends on.
    """
    doc = tomlrt.loads(
        td("""
        [fruit]

        [[fruit.apple.seeds]]
        [fruit.apple.seeds.size]
        color = "red"

        [[fruit.apple.seeds.size.seeds]]

        [[fruit.apple.seeds.size.seeds]]
        new = true

        [[fruit.apple.seeds]]
        new = true
        """)
    )
    doc["fruit"]["apple"] = doc["fruit"]["apple"]["seeds"]
    out = tomlrt.dumps(doc)
    assert out == td("""
        [fruit]

        [[fruit.apple]]
        [fruit.apple.size]
        color = "red"

        [[fruit.apple.size.seeds]]

        [[fruit.apple.size.seeds]]
        new = true

        [[fruit.apple]]
        new = true
        """)
    assert _reparses(out) == doc.to_dict()


def test_clone_section_into_aot_entry_registers_it_in_entry_slots() -> None:
    """A section-style value cloned as a *new key inside an existing AoT
    entry* (not the entry itself) must be registered in that entry's
    ``entry_slots`` membership, not just have ``owner_aot_entry`` set.

    ``remove_aot_entries`` enumerates an entry's owned slots via
    ``entry.entry_slots``; a slot with the right ``owner_aot_entry`` but
    missing from ``entry_slots`` survives entry removal as an orphaned,
    still-rendered block even though the logical model correctly shows
    the AoT as empty.
    """
    doc = tomlrt.loads("[[arr]]\n")
    other = tomlrt.loads("[k40]\nval = 2\n")
    doc["arr"][0]["k40"] = other["k40"]
    assert tomlrt.dumps(doc) == td("""
        [[arr]]

        [arr.k40]
        val = 2
        """)
    doc["arr"].pop(0)
    assert tomlrt.dumps(doc) == "arr = []\n"
    assert doc.to_dict() == {"arr": []}


def test_clone_as_aot_entry_hoists_own_content_past_forward_declared_nested() -> None:
    # Same forward-declaration hazard as above, but the destination is a
    # *new AoT entry* rather than a plain section: an array-of-tables
    # entry can never be reopened (unlike a plain table), so the source's
    # own direct content (`better`) must be hoisted ahead of the nested
    # descendant's block (`b`) rather than kept in true doc-stream order.
    src = td("""
        [a.b]
        x = 1

        [a]
        better = 2
        """)
    doc = tomlrt.loads(src)
    doc["arr"] = AoT()
    doc["arr"].append(doc["a"])
    out = tomlrt.dumps(doc)
    assert out == td("""
        [a.b]
        x = 1

        [a]
        better = 2

        [[arr]]
        better = 2
        [arr.b]
        x = 1
        """)
    assert doc["arr"][0].to_dict() == {"better": 2, "b": {"x": 1}}
    assert _reparses(out)["arr"] == [{"better": 2, "b": {"x": 1}}]


def test_aot_append_inline_table_nested_in_other_aot_entry() -> None:
    """Appending a value physically nested inside an AoT entry (but not
    itself the entry) must synthesise it as ordinary content, not
    dispatch through the AoT-entry clone path (which requires a header)."""
    src = td("""
        [[people]]
        x = {a = 1}
        """)
    doc = tomlrt.loads(src)
    inline = doc.aot("people")[0]["x"]
    doc["arr"] = AoT()
    doc["arr"].append(inline)
    out = tomlrt.dumps(doc)
    assert out == td("""
        [[people]]
        x = {a = 1}

        [[arr]]
        a = 1
        """)
    assert doc["arr"][0].to_dict() == {"a": 1}
    assert _reparses(out)["arr"] == [{"a": 1}]


def test_assign_nested_section_of_aot_entry_as_plain_section() -> None:
    """A section nested inside an AoT entry is not itself the entry.

    Assigning it elsewhere must clone just that section, not the whole
    owning entry with the nested section's key kept as an extra level.
    """
    src = td("""
        [[people]]
        x = 1

        [people.nested]
        y = 2
        """)
    doc = tomlrt.loads(src)
    nested = doc.aot("people")[0]["nested"]
    doc2 = tomlrt.loads("")
    doc2["copied"] = nested
    out = tomlrt.dumps(doc2)
    assert out == td("""
        [copied]
        y = 2
        """)
    assert doc2["copied"].to_dict() == {"y": 2}
    assert _reparses(out) == {"copied": {"y": 2}}


def test_assign_dotted_key_navigator_view_synthesises_fresh_inline_table() -> None:
    """A dotted-key navigator view (e.g. the `a` in `t = {a.b = 1}`) owns
    no CST of its own — it's a live projection over its parent's inline
    value — so it must be synthesised fresh, not cloned, when used as a
    value elsewhere."""
    doc = tomlrt.loads("t = {a.b = 1, a.c = 2}\n")
    inner = doc["t"]["a"]
    doc2 = tomlrt.loads("")
    doc2["x"] = inner
    out = tomlrt.dumps(doc2)
    assert out == "x = { b = 1, c = 2 }\n"
    assert doc2["x"].to_dict() == {"b": 1, "c": 2}
    assert _reparses(out) == {"x": {"b": 1, "c": 2}}


def test_aot_append_dotted_key_navigator_view() -> None:
    """As above, but appending the navigator view into an AoT."""
    doc = tomlrt.loads("t = {a.b = 1}\n")
    inner = doc["t"]["a"]
    doc["arr"] = AoT()
    doc["arr"].append(inner)
    out = tomlrt.dumps(doc)
    assert out == td("""
        t = {a.b = 1}

        [[arr]]
        b = 1
        """)
    assert doc["arr"][0].to_dict() == {"b": 1}
    assert _reparses(out)["arr"] == [{"b": 1}]


def test_pop_last_entry_of_dotted_nested_aot_inside_aot_entry() -> None:
    """Emptying an AoT reached only via dotted keys, itself nested inside
    another AoT entry, must synthesise the `key = []` placeholder's owning
    header even with no active install transaction (unlike the
    structural-overwrite path, there is nothing to reposition afterward)."""
    src = td("""
        [[fruit.apple.seeds]]
        size = 2

        [[fruit.apple.seeds]]

        [[fruit.apple.seeds.apple.seeds]]
        size = 2
        """)
    doc = tomlrt.loads(src)
    nested = doc["fruit"]["apple"]["seeds"][1]["apple"]["seeds"]
    nested.pop(0)
    out = tomlrt.dumps(doc)
    assert out == td("""
        [[fruit.apple.seeds]]
        size = 2

        [[fruit.apple.seeds]]
        [fruit.apple.seeds.apple]
        seeds = []
        """)
    assert doc.to_dict() == {
        "fruit": {"apple": {"seeds": [{"size": 2}, {"apple": {"seeds": []}}]}}
    }
    assert _reparses(out) == doc.to_dict()


def test_del_and_readd_with_forward_declared_nested_table_keeps_content() -> None:
    src = td("""
        [t5.t1]

        [t5]
        t2 = 3.5
        """)
    doc = tomlrt.loads(src)
    v = doc["t5"]
    del doc["t5"]
    doc["t5"] = v
    out = tomlrt.dumps(doc)
    assert out == src
    assert _reparses(out) == {"t5": {"t1": {}, "t2": 3.5}}


def test_del_loop_leaves_doc_empty() -> None:
    src = "".join(f"[s{i}]\nx = {i}\n" for i in range(20))
    doc = tomlrt.loads(src)
    for k in list(doc):
        del doc[k]
    assert dict(doc) == {}
    assert tomlrt.dumps(doc) == ""


def test_inline_append_to_entry_with_eol_comment() -> None:
    # Appending to a last entry that has an EOL comment but no trailing
    # comma: the comma must land immediately after the value with the
    # comment moving to after the comma (not orphaned on its own line as
    # ``a = 1 # eol-on-a\n,\n  b = 2\n  }``), and the comment must stay
    # attached to `a` rather than migrating to the appended entry.
    # The new entry takes the two-space indent of the only row the value
    # opens -- here the closing bracket's, parked in the entry's EOL
    # section -- rather than the four-space no-signal fallback.
    src = "obj = { a = 1 # eol-on-a\n  }\n"
    doc = tomlrt.loads(src)
    doc.table("obj")["b"] = 2
    out = tomlrt.dumps(doc)
    assert out == td("""
        obj = { a = 1, # eol-on-a
          b = 2
          }
        """)
    assert tomlrt.loads(out).table("obj").to_dict() == {"a": 1, "b": 2}


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
    out = tomlrt.dumps(doc)
    assert out == "obj = { d = 3 }\n"
    assert tomlrt.loads(out).table("obj").to_dict() == {"d": 3}


def test_inline_delete_dotted_leaf_cleans_empty_prefix_container() -> None:
    doc = tomlrt.loads("obj = { a.b = 1 }\n")
    obj = doc.table("obj")
    inner_a = obj.table("a")
    del inner_a["b"]
    # Synthetic prefix container `a` is now empty and has no entry in
    # the backing InlineTableValue; outer dict view should drop it.
    assert "a" not in obj
    out = tomlrt.dumps(doc)
    assert out == "obj = { }\n"
    assert tomlrt.loads(out).table("obj").to_dict() == {}


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
    # `a` is implicit (no [a] header). Deleting its only [a.b] empties
    # `a`, which survives as a live key — Python-dict semantics: `del`
    # removes only the named key. The now-empty `a` re-materialises its
    # own `[a]` header so the surviving table still renders.
    src = td("""
        [a.b]
        x = 1
    """)
    doc = tomlrt.loads(src)
    del doc.table("a")["b"]
    assert tomlrt.dumps(doc) == "[a]\n"
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


def test_emptied_implicit_header_materialises_in_place() -> None:
    # The re-materialised header stays at the emptied descendant's old
    # position rather than being shoved to end-of-doc.
    src = td("""
        [a.b]
        x = 1

        [c.d]
        y = 2
    """)
    doc = tomlrt.loads(src)
    del doc.table("a")["b"]
    out = tomlrt.dumps(doc)
    assert out == td("""
        [a]

        [c.d]
        y = 2
    """)
    assert _reparses(out) == {"a": {}, "c": {"d": {"y": 2}}}


def test_emptied_implicit_header_keeps_leading_comment() -> None:
    # A comment on the emptied descendant's leading carries onto the
    # re-materialised header.
    src = td("""
        # keep me
        [a.b]
        x = 1
    """)
    doc = tomlrt.loads(src)
    del doc.table("a")["b"]
    assert tomlrt.dumps(doc) == td("""
        # keep me
        [a]
    """)


def test_emptied_implicit_header_keeps_eol_comment() -> None:
    # An end-of-line comment on the emptied descendant's header line
    # carries onto the re-materialised (shorter) header.
    src = td("""
        [a.b]  # note
        x = 1
    """)
    doc = tomlrt.loads(src)
    del doc.table("a")["b"]
    assert tomlrt.dumps(doc) == td("""
        [a]  # note
    """)


def test_emptied_implicit_deep_header_materialises_in_place() -> None:
    # The descendant's anchor is a deep header `[a.b.c.d]`; emptying `a`
    # materialises `[a]` at that header's old position, carrying its
    # leading separator.
    src = td("""
        z = 0

        [a.b.c.d]
        x = 1
    """)
    doc = tomlrt.loads(src)
    del doc.table("a")["b"]
    assert tomlrt.dumps(doc) == td("""
        z = 0

        [a]
    """)
    assert _reparses(tomlrt.dumps(doc)) == {"z": 0, "a": {}}


def test_emptied_aot_header_origin_materialises_table_header() -> None:
    # The emptied implicit table's only anchor was an AoT header
    # (`[[a.b]]`). A plain `[a]` table header must be materialised (not
    # `[[a]]`), matching the surviving table model.
    doc = tomlrt.loads("[[a.b]]\nx = 1\n")
    del doc.table("a")["b"]
    out = tomlrt.dumps(doc)
    assert out == "[a]\n"
    assert _reparses(out) == {"a": {}}


def test_emptied_at_head_dotted_stays_in_place_as_inline() -> None:
    # The emptied binding was the doc head. Its dotted origin
    # re-materialises it as an inline table ``a = {}`` in place — it stays
    # at the head, so the following blank-line separator is undisturbed.
    src = td("""
        a.b = 1

        c = 2
    """)
    doc = tomlrt.loads(src)
    del doc.table("a")["b"]
    assert tomlrt.dumps(doc) == td("""
        a = {}

        c = 2
    """)


def test_emptied_dotted_stays_in_place_among_sibling_kvs() -> None:
    # The deleted dotted subtree's KVs are interleaved with surviving
    # sibling KVs. The inline ``a = {}`` re-parents nothing, so it stays
    # exactly where the dotted binding was rather than moving past siblings.
    src = td("""
        a.b.x = 1
        c = 2
        a.b.y = 3
        d = 4
    """)
    doc = tomlrt.loads(src)
    del doc.table("a")["b"]
    assert tomlrt.dumps(doc) == td("""
        a = {}
        c = 2
        d = 4
    """)
    assert _reparses(tomlrt.dumps(doc)) == {"a": {}, "c": 2, "d": 4}


def test_delete_deep_implicit_inside_aot_keeps_implicit_chain() -> None:
    # Emptying an AoT-owned implicit table (`foo`) re-materialises it as
    # an inline table `foo = {}` inside the owning entry's slot region, so
    # the surviving empty table still renders and round-trips.
    doc = tomlrt.loads("[[arr]]\nfoo.bar.baz = 1\n")
    del doc.aot("arr")[0].table("foo")["bar"]
    assert tomlrt.dumps(doc) == "[[arr]]\nfoo = {}\n"
    assert doc.to_dict() == {"arr": [{"foo": {}}]}


def test_delete_deep_non_aot_implicit_keeps_chain() -> None:
    # `[a.b.c.d]\nx=1` → a, b, c are all implicit. Deleting `d` empties
    # `c`; the surviving empty `c` re-materialises as `[a.b.c]` (a, b
    # stay implicit, carried by the dotted header path).
    doc = tomlrt.loads("[a.b.c.d]\nx = 1\n")
    del doc.table("a").table("b").table("c")["d"]
    assert tomlrt.dumps(doc) == "[a.b.c]\n"
    assert "a" in doc
    assert dict(doc.table("a").table("b").table("c")) == {}


def test_delete_header_only_section() -> None:
    doc = tomlrt.loads("[s]\n")
    del doc["s"]
    assert tomlrt.dumps(doc) == ""


def test_readd_into_emptied_aot_implicit_anchors_inside_entry() -> None:
    # After deleting the only descendant of an AoT-owned implicit
    # chain, re-adding under that chain mutates the in-place inline
    # table inside the owning entry's slot region.
    doc = tomlrt.loads("[[arr]]\nfoo.bar.baz = 1\n\n[[arr]]\nname = 2\n")
    foo = doc.aot("arr")[0].table("foo")
    del foo["bar"]
    foo["new"] = 1
    out = tomlrt.dumps(doc)
    assert out == td("""
        [[arr]]
        foo = { new = 1 }

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


def test_append_multiline_array_pad_free_newline_matches_its_rows() -> None:
    # The array is multi-line but its only newline lives in item 0's EOL
    # comment section, not the bracket pads. A synthesised inter-item
    # separator matches the break the array's own rows use, not a
    # hardcoded LF.
    doc = tomlrt.loads("xs = [1, # c\r\n    2]\r\n")
    doc["xs"].append(3)
    assert tomlrt.dumps(doc) == "xs = [1, # c\r\n    2,\r\n    3]\r\n"


def test_append_matches_a_row_break_the_document_does_not_use() -> None:
    # Sampling the value's own rows rather than the document's newline
    # keeps a mixed-ending document's array internally consistent.
    doc = tomlrt.loads("xs = [1, 2,\r\n  3]\n")
    doc["xs"].append(4)
    assert tomlrt.dumps(doc) == "xs = [1, 2,\r\n  3,\r\n  4]\n"


def test_emptied_multiline_array_refill_collapses_to_single_line() -> None:
    # An array's multi-line shape is derived from its bracket pads, not a
    # sticky flag. Emptying it (the closing break lived in the deleted item's
    # EOL section, so the pads collapse) leaves a single-line ``[]``; refilling
    # is then cleanly single-line in both LF and CRLF documents rather than a
    # half-expanded shape with a stray LF.
    doc = tomlrt.loads("xs = [1]\r\n")
    doc["xs"].comments[0] = "c"  # promote to multi-line
    del doc["xs"][0]  # empty: bracket pads collapse
    doc["xs"].append(2)
    doc["xs"].append(3)
    assert tomlrt.dumps(doc) == "xs = [2, 3]\r\n"


def test_delete_inserted_top_level_kv_round_trips() -> None:
    doc = tomlrt.loads("[s]\nx = 1\n")
    doc["new"] = 1
    del doc["new"]
    # Blank-line residue is acceptable; reparse is what matters.
    out = tomlrt.dumps(doc)
    assert out == td("""
        [s]
        x = 1
        """)
    assert tomlrt.loads(out).to_dict() == {"s": {"x": 1}}


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
    out = tomlrt.dumps(doc)
    assert out == td("""
        [[pkg]]
        name = "z"

        [[pkg]]
        name = "b"
        """)
    assert _reparses(out) == {"pkg": [{"name": "z"}, {"name": "b"}]}


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


def test_ensure_table_through_scalar_value_raises() -> None:
    doc = tomlrt.loads("a = 1\n")
    with pytest.raises(tomlrt.TOMLError, match="inline table or non-table"):
        doc.ensure_table("a")


def test_install_dotted_path_leaves_doc_untouched_when_blocked_by_comments() -> None:
    """A dotted `install()` that must promote an ancestor, but is then
    blocked by a *later* ancestor's inner comments, must not leave the
    first ancestor promoted: the whole call is atomic."""
    src = td("""
        a = {b = { # comment
        c = 1 }, sibling = 2}
        """)
    doc = tomlrt.loads(src)
    with pytest.raises(tomlrt.TOMLError, match="inner comments"):
        doc.install("a.b.x", 9)
    assert tomlrt.dumps(doc) == src
    assert _reparses(tomlrt.dumps(doc)) == {"a": {"b": {"c": 1}, "sibling": 2}}


def test_install_dotted_path_leaves_doc_untouched_when_blocked_by_scalar() -> None:
    """Same atomicity guarantee when the blocking component is simply a
    non-table value rather than a comment-bearing inline table."""
    src = "a = {arr = 1}\n"
    doc = tomlrt.loads(src)
    with pytest.raises(tomlrt.TOMLError, match="inline table or non-table"):
        doc.install("a.arr.x", 9)
    assert tomlrt.dumps(doc) == src
    assert _reparses(tomlrt.dumps(doc)) == {"a": {"arr": 1}}


def test_install_dotted_path_leaves_doc_untouched_when_blocked_by_aot() -> None:
    """The AoT-blocks-the-walk branch of the preflight also leaves the
    document untouched (an inline table can never itself hold an AoT,
    so the AoT here is necessarily a sibling, not a promoted
    descendant, of the inline table also present under ``x``)."""
    src = td("""
        [x]
        a = {b = 1}
        [[x.c]]
        y = 1
        """)
    doc = tomlrt.loads(src)
    with pytest.raises(tomlrt.TOMLError, match="array-of-tables"):
        doc.install("x.c.d", 9)
    assert tomlrt.dumps(doc) == src
    assert _reparses(tomlrt.dumps(doc)) == {"x": {"a": {"b": 1}, "c": [{"y": 1}]}}


def test_install_dotted_path_rolls_back_multiple_promotions_when_later_blocked() -> (
    None
):
    """Two ancestors are both promotable, but the third path component
    is a plain scalar: neither of the first two must end up promoted."""
    src = "a = {b = {c = 1}, other = 2}\n"
    doc = tomlrt.loads(src)
    with pytest.raises(tomlrt.TOMLError, match="inline table or non-table"):
        doc.install("a.b.c.x", 9)
    assert tomlrt.dumps(doc) == src
    assert _reparses(tomlrt.dumps(doc)) == {"a": {"b": {"c": 1}, "other": 2}}


def test_install_dotted_path_promotes_every_promotable_ancestor() -> None:
    """Every promotable ancestor on the path is promoted and the leaf
    is installed."""
    doc = tomlrt.loads("a = {b = {c = {d = 1}}, other = 2}\n")
    doc.install("a.b.c.x", 9)
    out = tomlrt.dumps(doc)
    assert out == td("""
        [a]
        other = 2

        [a.b.c]
        d = 1
        x = 9
        """)
    assert _reparses(out) == {"a": {"other": 2, "b": {"c": {"d": 1, "x": 9}}}}


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(_OPAQUE, id="opaque"),
        pytest.param({"x": _OPAQUE}, id="mapping-with-opaque"),
        pytest.param((1, 2), id="tuple"),
        pytest.param(b"bytes", id="bytes"),
        pytest.param(Table.section({"x": _OPAQUE}), id="section"),
        pytest.param(AoT([{"x": _OPAQUE}]), id="aot"),
    ],
)
def test_install_invalid_value_leaves_document_untouched(value: Any) -> None:
    """A leaf value `install()` cannot convert is rejected before any
    part of the path is synthesised."""
    src = td("""
        a = 1
        """)
    doc = tomlrt.loads(src)
    with pytest.raises(TypeError):
        doc.install("p.q.r", value)
    assert tomlrt.dumps(doc) == src
    assert list(doc) == ["a"]
    assert _reparses(tomlrt.dumps(doc)) == {"a": 1}


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(_OPAQUE, id="opaque"),
        pytest.param(Table.section({"x": _OPAQUE}), id="section"),
        pytest.param(AoT([{"x": _OPAQUE}]), id="aot"),
    ],
)
def test_install_invalid_value_does_not_promote_inline_ancestor(value: Any) -> None:
    """The value check runs before the ancestor promotion `install()`
    would otherwise perform, so a rejected value leaves inline
    ancestors inline."""
    src = td("""
        a = {b = 1}
        """)
    doc = tomlrt.loads(src)
    with pytest.raises(TypeError):
        doc.install("a.c", value)
    assert tomlrt.dumps(doc) == src
    assert _reparses(tomlrt.dumps(doc)) == {"a": {"b": 1}}


def test_install_invalid_value_messages() -> None:
    """The leaf check keeps `__setitem__`'s key-aware advice."""
    doc = tomlrt.loads("")
    pair: Any = (1, 2)
    raw: Any = bytearray(b"x")
    with pytest.raises(TypeError, match="cannot convert object to a TOML value"):
        doc.install("p.q", _OPAQUE)
    with pytest.raises(TypeError, match="cannot assign tuple to TOML key 'q'"):
        doc.install("p.q", pair)
    with pytest.raises(TypeError, match="cannot assign bytes to TOML key 'q'"):
        doc.install("p.q", raw)
    assert tomlrt.dumps(doc) == ""
    assert list(doc) == []


def test_install_nested_section_chain_renders_deepest_header_only() -> None:
    """Installing a chain of section tables emits one header for the chain.

    Intermediate components stay implicit, so only the deepest table
    that carries a body gets an explicit header.
    """
    value = Table.section({"leaf": {"x": 1}})
    for i in range(2):
        value = Table.section({f"k{i}": value})
    doc = tomlrt.loads("[a]\nx = 1\n")
    doc.install(("root",), value)
    assert tomlrt.dumps(doc) == td("""
        [a]
        x = 1

        [root.k1.k0]
        leaf = { x = 1 }
        """)


def test_nested_invalid_value_messages_name_their_own_key() -> None:
    """Validation reports the key the offending value is bound to.

    The recursive walk carries each child's key down with it, so a
    rejection inside a nested mapping is as actionable as one at the
    top level. A list item has no key of its own, so it keeps the
    unqualified advice.
    """
    doc = tomlrt.loads("")
    pair: Any = (1, 2)
    raw: Any = b"x"
    with pytest.raises(TypeError, match="cannot assign tuple to TOML key 'inner'"):
        doc["q"] = {"inner": pair}
    with pytest.raises(TypeError, match="cannot assign bytes to TOML key 'inner'"):
        doc["q"] = {"inner": raw}
    with pytest.raises(TypeError, match="cannot assign tuple to TOML key 'deep'"):
        doc["q"] = Table.section({"deep": pair})
    with pytest.raises(TypeError, match=r"cannot assign tuple; use a list"):
        doc["q"] = [pair]
    assert tomlrt.dumps(doc) == ""
    assert list(doc) == []


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
    out = tomlrt.dumps(doc)
    assert out == td("""
        [[arr]]
        a.b = 1
        a.c = 2

        [[dst]]
        a.b = 1
        a.c = 2
        """)
    assert _reparses(out) == {
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


def test_delete_nested_aot_section_scans_past_earlier_inline_value() -> None:
    """Deleting a nested section from one AoT entry should still find
    surviving sibling-entry ownership even when an earlier inline table
    is visited first during the reachability scan."""
    doc = tomlrt.loads(
        td("""
        meta = { a = 1 }

        [[fruit]]
        name = "a"

        [fruit.extra]
        x = 1

        [[fruit]]
        name = "b"
        """)
    )
    del doc.aot("fruit")[0]["extra"]
    out = tomlrt.dumps(doc)
    assert out == td("""
        meta = { a = 1 }

        [[fruit]]
        name = "a"

        [[fruit]]
        name = "b"
        """)
    assert _reparses(out) == doc.to_dict()


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
    arr.set_multiline(multiline=True, indent=2)
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
    doc.array("alist").set_multiline(multiline=True, indent=2)
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
    doc.array("alist").set_multiline(multiline=True, indent=2)
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
    arr.set_multiline(multiline=True, indent=4)
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
    doc.array("xs").set_multiline(multiline=True, indent=2)
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
    doc.array("xs").set_multiline(multiline=True, indent=2)
    assert tomlrt.dumps(doc) == td("""
        xs = [ # opening
        ]
        """)


def test_set_multiline_true_empty_array_aligns_close_to_outer_indent() -> None:
    src = "xs = []\n"
    doc = tomlrt.loads(src)
    doc.array("xs").set_multiline(multiline=True, indent=4)
    assert tomlrt.dumps(doc) == td("""
        xs = [
        ]
        """)


def test_set_multiline_true_preserves_nested_array_outer_indent() -> None:
    src = td("""
        outer = [
              { nested = [
            1, # item
              ] },
        ]
        """)
    doc = tomlrt.loads(src)
    doc.array("outer").table(0).array("nested").set_multiline(multiline=True, indent=4)
    assert tomlrt.dumps(doc) == src


def test_set_multiline_true_preserves_empty_nested_table_outer_indent() -> None:
    src = td("""
        outer = [
            { nested = {
            } },
        ]
        """)
    doc = tomlrt.loads(src)
    doc.array("outer").table(0).table("nested").set_multiline(multiline=True)
    assert tomlrt.dumps(doc) == src


def test_set_multiline_true_closes_nested_value_at_its_row_indent() -> None:
    src = td("""
        outer = [
          { nested = [1, 2] },
        ]
        """)
    doc = tomlrt.loads(src)
    doc.array("outer").table(0).array("nested").set_multiline(multiline=True, indent=4)
    out = tomlrt.dumps(doc)
    assert out == td("""
        outer = [
          { nested = [
            1,
            2,
          ] },
        ]
        """)
    assert _reparses(out) == {"outer": [{"nested": [1, 2]}]}


def test_set_multiline_true_closes_indented_kv_at_its_own_indent() -> None:
    src = td("""
        [s]
          a = [1, 2]
        """)
    doc = tomlrt.loads(src)
    doc["s"]["a"].set_multiline(multiline=True, indent=4)
    out = tomlrt.dumps(doc)
    assert out == td("""
        [s]
          a = [
            1,
            2,
          ]
        """)
    assert _reparses(out) == {"s": {"a": [1, 2]}}


def test_set_multiline_true_closes_empty_indented_value_at_its_own_indent() -> None:
    src = td("""
        [s]
          a = []
        """)
    doc = tomlrt.loads(src)
    doc["s"]["a"].set_multiline(multiline=True, indent=4)
    out = tomlrt.dumps(doc)
    assert out == td("""
        [s]
          a = [
          ]
        """)
    assert _reparses(out) == {"s": {"a": []}}


def test_set_multiline_true_closes_value_after_multiline_sibling() -> None:
    src = td("""
        outer = [ [
            1,
          ] , { nested = { a = 1 } },
             { other = { b = 2 } } ]
        """)
    doc = tomlrt.loads(src)
    # The first target's row is opened by the preceding item's own line
    # break, the second's by the break in its own leading trivia. The pad
    # after the sibling's closing bracket is not row indent, so the first
    # target starts where that bracket left the row, not past it.
    doc.array("outer").table(1).table("nested").set_multiline(multiline=True, indent=4)
    doc.array("outer").table(2).table("other").set_multiline(multiline=True, indent=7)
    out = tomlrt.dumps(doc)
    assert out == td("""
        outer = [ [
            1,
          ] , { nested = {
            a = 1,
          } },
             { other = {
               b = 2,
             } } ]
        """)
    assert _reparses(out) == {"outer": [[1], {"nested": {"a": 1}}, {"other": {"b": 2}}]}


def test_set_multiline_true_closes_value_after_a_multiline_string() -> None:
    """A sibling's own line breaks move the row the value starts on.

    A multi-line string carries its breaks in its lexeme rather than in
    any trivia, so the row is tracked through what each item renders,
    not through the trivia alone.
    """
    src = td('''
        a = [ """
          x""", [1] ]
        ''')
    doc = tomlrt.loads(src)
    doc["a"].array(1).set_multiline(multiline=True)
    out = tomlrt.dumps(doc)
    assert out == td('''
        a = [ """
          x""", [
            1,
          ] ]
        ''')
    assert _reparses(out) == {"a": ["  x", [1]]}


def test_comment_write_promotes_indented_inline_table_at_its_own_indent() -> None:
    src = td("""
        [s]
          t = { a = 1 }
        """)
    doc = tomlrt.loads(src)
    doc["s"]["t"].comments["a"] = "note"
    out = tomlrt.dumps(doc)
    assert out == td("""
        [s]
          t = {
            a = 1, # note
          }
        """)
    assert _reparses(out) == {"s": {"t": {"a": 1}}}


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


def test_sort_moves_blank_separator_with_trailing_whitespace() -> None:
    # A blank-line separator whose own line carries trailing whitespace
    # (not itself the slot's own indent, since another blank line
    # follows it) is purely positional, like any comment-free
    # separator, and travels with the reorder like the plain-newline
    # case above.
    src = "a = 1\nb = 2\n   \n\nc = 3\n"
    doc = tomlrt.loads(src)
    doc.sort(reverse=True)
    out = tomlrt.dumps(doc)
    assert out == "c = 3\nb = 2\n   \n\na = 1\n"
    assert _reparses(out)


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


def test_sort_keeps_mixed_key_leaf_ahead_of_a_sorted_section() -> None:
    """A key that owns both a leaf and a sub-section keeps its leaf ahead
    of every section header when sorting.

    Regression: ``z`` owns a dotted leaf (``z.k``) and a sub-section
    (``[z.m]``). ``sort`` classified it wholly as a section (it has a
    header), so it sorted after the array-of-tables ``x`` — placing
    ``z.k`` after ``[[x]]``, which a re-parse captured into the AoT
    entry. ``z``'s leaf part now sorts into the leaf region.
    """
    doc = tomlrt.loads(
        td("""
            z.k = 7

            [[x]]

            [z.m]
            n = 1
            """)
    )
    doc.sort()
    out = tomlrt.dumps(doc)
    assert out == td("""
        z.k = 7

        [[x]]

        [z.m]
        n = 1
        """)
    assert _reparses(out) == doc.to_dict()


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


def test_sort_headerless_child_refreshes_ancestor_primary_ref() -> None:
    doc = tomlrt.loads(
        td("""
            [outer.z]
            value = 1

            [other]
            value = 0

            [outer.a]
            value = 2
            """)
    )
    doc.table("outer").sort()
    doc["outer"] = Table.section({"value": 3})

    out = tomlrt.dumps(doc)
    assert out == td("""
        [outer]
        value = 3

        [other]
        value = 0
        """)
    assert _reparses(out) == doc.to_dict()


def test_sort_headerless_child_refreshes_ancestor_body_tail() -> None:
    doc = tomlrt.loads(
        td("""
            outer.z = 1
            foreign = 0
            outer.a = 2

            [tail]
            value = 3
            """)
    )
    doc.table("outer").sort()
    doc["new"] = 4

    out = tomlrt.dumps(doc)
    assert out == td("""
        foreign = 0
        outer.a = 2
        outer.z = 1
        new = 4

        [tail]
        value = 3
        """)
    assert _reparses(out) == doc.to_dict()


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


def test_sort_detached_inline_factory_table() -> None:
    """A detached ``Table.inline()`` sorts its dict order.

    There is no backing inline value to permute yet, so the sort only
    has to survive until attachment, where dict order decides what is
    emitted.
    """
    table = Table.inline()
    table["b"] = 1
    table["a"] = 2
    table.sort()
    assert list(table) == ["a", "b"]
    doc = tomlrt.Document()
    doc["x"] = table
    assert tomlrt.dumps(doc) == "x = { a = 2, b = 1 }\n"
    assert _reparses(tomlrt.dumps(doc)) == doc.to_dict()


def test_sort_detached_inline_factory_table_reverse_and_key() -> None:
    """``key`` / ``reverse`` apply to a detached inline table too."""
    table = Table.inline()
    table["bb"] = 1
    table["a"] = 2
    table["ccc"] = 3
    table.sort(key=len, reverse=True)
    doc = tomlrt.Document()
    doc["x"] = table
    assert tomlrt.dumps(doc) == "x = { ccc = 3, bb = 1, a = 2 }\n"
    assert _reparses(tomlrt.dumps(doc)) == doc.to_dict()


def test_sort_nested_detached_inline_factory_table() -> None:
    """Sorting a factory table already nested in another factory works."""
    outer = Table.inline()
    inner = Table.inline()
    inner["b"] = 1
    inner["a"] = 2
    outer["n"] = inner
    inner.sort()
    doc = tomlrt.Document()
    doc["x"] = outer
    assert tomlrt.dumps(doc) == "x = { n = { a = 2, b = 1 } }\n"
    assert _reparses(tomlrt.dumps(doc)) == doc.to_dict()


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


def test_repeated_sort_of_aot_entry_refiles_noncontiguous_ancestor_refs() -> None:
    src = td("""
        [[arr]]
        [arr.z]
        value = 1
        [arr.a]
        value = 2

        [[arr]]
        name = "sibling"
        """)
    doc = tomlrt.loads(src)
    entry = doc.aot("arr")[0]

    entry.sort()
    entry.sort(reverse=True)
    entry.sort()
    doc["tail"] = 3

    out = tomlrt.dumps(doc)
    assert out == td("""
        tail = 3

        [[arr]]
        [arr.a]
        value = 2
        [arr.z]
        value = 1

        [[arr]]
        name = "sibling"
        """)
    assert _reparses(out) == doc.to_dict()


def test_live_section_clone_files_ancestor_refs_at_physical_position() -> None:
    doc = tomlrt.loads(
        td("""
            [x.y.z.w]

            [x]

            [tail]
            """)
    )
    w = doc.table(("x", "y", "z", "w"))
    w["k68"] = w
    doc.table("x")["a"] = 1

    doc.table("x").sort()

    out = tomlrt.dumps(doc)
    assert out == td("""
        [x]
        a = 1

        [x.y.z.w]

        [x.y.z.w.k68]

        [tail]
        """)
    assert _reparses(out) == doc.to_dict()


def test_forward_declared_clone_files_own_refs_in_physical_order() -> None:
    doc = tomlrt.loads(
        td("""
            [src.child]
            z = 1
            [src]
            x = 2
            [tail]
            """)
    )
    doc["clone"] = doc.table("src")

    doc.table("clone").sort()
    out = tomlrt.dumps(doc)
    assert out == td("""
        [src.child]
        z = 1
        [src]
        x = 2
        [tail]
        [clone]
        x = 2
        [clone.child]
        z = 1
        """)
    assert _reparses(out) == doc.to_dict()


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
    assert out == td("""
        [a]
        date = "2019"
        name = "Bob"

        [[a.hello]]
        x = 1
        """)
    assert _reparses(out)
    reparsed = tomlrt.loads(out)
    assert reparsed.table("a")["date"] == "2019"
    assert reparsed.table("a")["name"] == "Bob"
    assert reparsed.table("a").aot("hello")[0]["x"] == 1


def test_sort_preserves_parent_separator_when_aot_child_precedes_it() -> None:
    src = td("""
        [[a-section.hello]]
        ports = [80]

        [a-section]
        date = "2019"
        """)
    expected = td("""
        [a-section]
        date = "2019"

        [[a-section.hello]]
        ports = [80]
    """)
    doc = tomlrt.loads(src)
    doc.format()
    doc["a-section"].sort()
    assert tomlrt.dumps(doc) == expected

    doc = tomlrt.loads(src)
    doc["a-section"].sort()
    doc.format()
    assert tomlrt.dumps(doc) == expected


def test_sort_preserves_separators_for_multiple_forward_aot_children() -> None:
    src = td("""
        [[a.x]]
        x = 1

        [[a.y]]
        y = 2

        [a]
        z = 3
        """)
    expected = td("""
        [a]
        z = 3

        [[a.x]]
        x = 1

        [[a.y]]
        y = 2
    """)
    doc = tomlrt.loads(src)
    doc.format()
    doc.table("a").sort()
    assert tomlrt.dumps(doc) == expected

    doc = tomlrt.loads(src)
    doc.table("a").sort()
    doc.format()
    assert tomlrt.dumps(doc) == expected


def test_sort_hoists_mixed_leaf_before_forward_structural_content() -> None:
    src = td("""
        [a.x.m]
        u = 1

        [a]
        x.foo = 2
        z = 3
    """)
    expected = td("""
        [a]
        z = 3
        x.foo = 2

        [a.x.m]
        u = 1
    """)
    doc = tomlrt.loads(src)
    doc.format()
    doc.table("a").sort()
    out = tomlrt.dumps(doc)
    assert out == expected
    assert _reparses(out) == doc.to_dict()

    doc = tomlrt.loads(src)
    doc.table("a").sort()
    doc.format()
    assert tomlrt.dumps(doc) == expected


def test_sort_groups_all_mixed_leaves_before_their_structural_content() -> None:
    src = td("""
        [a.x.m]
        u = 1

        [a.y.m]
        v = 2

        [a]
        x.foo = 3
        y.foo = 4
        z = 5
        """)
    doc = tomlrt.loads(src)
    doc.table("a").sort()
    out = tomlrt.dumps(doc)
    assert out == td("""
        [a]
        z = 5
        x.foo = 3
        y.foo = 4

        [a.x.m]
        u = 1

        [a.y.m]
        v = 2
    """)
    assert _reparses(out) == doc.to_dict()

    doc = tomlrt.loads(src)
    doc.table("a").sort(reverse=True)
    out = tomlrt.dumps(doc)
    assert out == td("""
        [a]
        z = 5
        y.foo = 4
        x.foo = 3

        [a.y.m]
        v = 2

        [a.x.m]
        u = 1
    """)
    assert _reparses(out) == doc.to_dict()


def test_sort_groups_mixed_parts_when_parent_header_is_already_first() -> None:
    doc = tomlrt.loads(
        td("""
        [a]
        x.foo = 3
        y.foo = 4

        [a.x.m]
        u = 1

        [a.y.m]
        v = 2
        """)
    )
    doc.table("a").sort(reverse=True)
    out = tomlrt.dumps(doc)
    assert out == td("""
        [a]
        y.foo = 4
        x.foo = 3

        [a.y.m]
        v = 2

        [a.x.m]
        u = 1
    """)
    assert _reparses(out) == doc.to_dict()


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
    assert doc.epilogue == (None, "trailing comment")
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
    assert doc.epilogue == (None, "trailing comment")
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


def test_pop_sole_item_closing_bracket_on_same_line_keeps_eol_comment() -> None:
    """Item and closing bracket share a physical line (no newline between them)."""
    doc = tomlrt.loads("a = [ # tail\n    1]\n")
    doc.array("a").pop()
    assert tomlrt.dumps(doc) == "a = [ # tail\n]\n"


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


def test_delete_sole_entry_closing_brace_on_same_line_keeps_eol_comment() -> None:
    """Entry and closing brace share a physical line (no newline between them)."""
    doc = tomlrt.loads("a = { # tail\n    x = 1}\n")
    del doc.table("a")["x"]
    assert tomlrt.dumps(doc) == "a = { # tail\n}\n"


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
    ``TypeError: cannot convert ...`` as the regular-table path, rather
    than the previous misleading ``NotImplementedError`` about
    "live-attach of typed Container/Array/AoT"."""
    doc = tomlrt.loads("t = {a = 1}\n")
    inline = doc.table("t")
    with pytest.raises(TypeError, match="cannot convert set"):
        inline["x"] = {1, 2, 3}


def test_insert_new_key_non_coerceable_type_raises() -> None:
    doc = tomlrt.loads("a = 1\n")
    with pytest.raises(TypeError, match="cannot convert set"):
        doc["b"] = {1, 2, 3}


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


def test_promote_refusals_are_all_toml_errors() -> None:
    """Every wrong-shape refusal from `promote_inline` / `promote_array` is a
    `TOMLError`, so the handler the errors documentation shows actually
    catches them. A missing key still raises `KeyError`, mirroring `dict`.
    """
    doc = tomlrt.loads(
        td("""
        [project]
        [project.authors]
        name = "me"
    """)
    )
    with pytest.raises(tomlrt.TOMLError, match="not an inline table"):
        doc.table("project").promote_inline("authors")

    doc = tomlrt.loads("a = 1\nb = [1, 2]\n")
    for call in (
        lambda: doc.promote_inline("a"),
        lambda: doc.promote_array("a"),
        lambda: doc.promote_array("b"),
    ):
        with pytest.raises(tomlrt.TOMLError):
            call()

    with pytest.raises(KeyError, match="not in table"):
        doc.promote_inline("missing")


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


def test_aot_cross_doc_assign_empty_entry_clears_body() -> None:
    """Replacing an attached entry with an empty foreign one keeps only
    the destination header."""
    src_doc = tomlrt.loads("[[s]]\n")
    dst_doc = tomlrt.loads("[[a]]\nx = 1\n")
    dst_doc.aot("a")[0] = src_doc.aot("s")[0]
    out = tomlrt.dumps(dst_doc)
    assert out == "[[a]]\n"
    assert _reparses(out) == dst_doc.to_dict()


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
    assert tomlrt.dumps(doc) == td("""
        # preamble comment
        [existing]
        x = 1

        [tool.sub]
        k = 1
        """)


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


def test_demote_synthetic_placeholder_then_reassign_parent() -> None:
    """Demote then ``doc[parent] = ...`` round-trips the new value.

    Exercises the cleanup path after a demote: replacing the parent key
    must scrub the surviving descendants and rewrite the slot stream
    without tripping over any residual bookkeeping left by the demoted
    header.
    """
    doc = tomlrt.loads("")
    doc["tool"] = Table.section({})
    doc["tool"]["poetry"] = tomlrt.AoT([{"x": 1}])
    doc["tool"] = Table.section({"y": 2})
    assert tomlrt.dumps(doc) == td("""
        [tool]
        y = 2
        """)


def test_demote_synthetic_placeholder_then_sort_aot() -> None:
    """Sorting the AoT that triggered the demote round-trips cleanly.

    AoT sort drives ``_scrub_owned_slots_via_backptrs`` over each
    entry's slots. The demoted header is not in any entry's slot set,
    but exercising the sort post-demote pins that the cleanup paths
    operate on a self-consistent slot graph.
    """
    doc = tomlrt.loads("")
    doc["tool"] = Table.section({})
    doc["tool"]["poetry"] = tomlrt.AoT([{"x": 2}, {"x": 1}, {"x": 3}])
    doc.aot("tool.poetry").sort(key=lambda t: int(t["x"]))
    assert tomlrt.dumps(doc) == td("""
        [[tool.poetry]]
        x = 1

        [[tool.poetry]]
        x = 2

        [[tool.poetry]]
        x = 3
        """)


def test_demote_synthetic_placeholder_inside_aot_entry_updates_membership() -> None:
    """Demoting a synthetic placeholder inside an AoT entry must also
    remove the orphaned header from that entry's slot membership."""
    doc = tomlrt.loads('[[pkg]]\nname = "a"\n')
    entry = doc.aot("pkg")[0]
    entry["tool"] = Table.section({})
    entry["tool"]["sub"] = Table.section({"x": 1})
    out = tomlrt.dumps(doc)
    assert out == td("""
        [[pkg]]
        name = "a"

        [pkg.tool.sub]
        x = 1
        """)
    assert _reparses(out) == doc.to_dict()


def test_aot_clone_append_past_first_entry_keeps_trailing_placeholder() -> None:
    """A placeholder header with nothing after it survives an AoT append.

    Overwriting ``[fruit.orange]`` with a scalar synthesises a
    ``[fruit]`` header at the document tail, and deleting the scalar
    leaves it an empty placeholder. The appended entry anchors after its
    predecessor's subtree — ahead of that placeholder — so there is no
    successor to hand the placeholder's trivia to and it stays put.
    """
    doc = tomlrt.loads(
        td("""
        [[fruit.k99]]
        v = 1

        [animal]

        [fruit.orange]
        """)
    )
    doc["fruit"]["orange"] = "str"
    del doc["fruit"]["orange"]
    doc.aot("fruit.k99").append(doc.table("animal"))
    out = tomlrt.dumps(doc)
    assert out == td("""
        [[fruit.k99]]
        v = 1

        [[fruit.k99]]

        [animal]

        [fruit]
        """)
    assert _reparses(out) == doc.to_dict()


def test_aot_append_past_first_entry_demotes_emptied_placeholder() -> None:
    """A placeholder that empties after entry 0 is demoted by a later entry.

    Every ``[[path]]`` header makes a synthetic placeholder parent
    redundant, not just the first: the placeholder here still had a KV
    body when the AoT was created.
    """
    doc = tomlrt.loads("")
    doc["tool"] = Table.section({"a": 1})
    doc["tool"]["poetry"] = tomlrt.AoT([{"x": 1}])
    del doc["tool"]["a"]
    doc.aot("tool.poetry").append({"x": 2})
    out = tomlrt.dumps(doc)
    assert out == td("""
        [[tool.poetry]]
        x = 1

        [[tool.poetry]]
        x = 2
        """)
    assert _reparses(out) == doc.to_dict()


# ---------------------------------------------------------------------------
# Non-tail AoT append / section creation
# ---------------------------------------------------------------------------
#
# Appending to an AoT, or creating a new section, splices the new header
# relative to its own target's position — via ``_parent_subtree_tail``
# for AoT append and the document's stable section-spacing convention —
# independent of how much unrelated content follows the target elsewhere
# in the document. These tests pin correctness of those code paths;
# performance is covered by ``benchmarks/bench_mutate.py``.


def test_aot_append_before_nested_subtree_and_trailing_section() -> None:
    """Appending to a non-tail AoT lands after the last entry's whole
    subtree — including a nested AoT — and before unrelated trailing
    content."""
    src = td("""
        [[items]]
        name = "first"

        [[items]]
        name = "last"

        [[items.tags]]
        label = "x"

        [other]
        k = 1
        """)
    doc = tomlrt.loads(src)
    doc.aot("items").append({"name": "new"})
    assert tomlrt.dumps(doc) == td("""
        [[items]]
        name = "first"

        [[items]]
        name = "last"

        [[items.tags]]
        label = "x"

        [[items]]
        name = "new"

        [other]
        k = 1
        """)


def test_aot_append_before_trailing_section_reparses_identically() -> None:
    """A non-tail append round-trips and re-parses to the same model."""
    src = td("""
        [[items]]
        name = "first"

        [other]
        k = 1
        """)
    doc = tomlrt.loads(src)
    doc.aot("items").append({"name": "second"})
    out = tomlrt.dumps(doc)
    assert _reparses(out)
    assert tomlrt.loads(out).to_dict() == doc.to_dict()


def test_promote_implicit_table_inside_non_tail_aot_entry() -> None:
    """Promoting an implicit table inside a non-last-in-doc AoT entry
    lands inside that entry, ahead of unrelated trailing content."""
    src = td("""
        [[items]]
        name = "x"

        [other]
        k = 1
        """)
    doc = tomlrt.loads(src)
    entry = doc.aot("items")[0]
    entry.install(("sub",), Table.section({"y": 1}))
    assert tomlrt.dumps(doc) == td("""
        [[items]]
        name = "x"

        [items.sub]
        y = 1

        [other]
        k = 1
        """)


def test_attach_section_inside_non_last_nested_aot_entry_then_reverse() -> None:
    """A host-tail insert files refs before later same-path AoT siblings."""
    doc = tomlrt.loads(
        td("""
        [[a.b]]
        x = 1

        [[a.b]]
        [a.b.c]
        z = 3
        """)
    )
    doc.aot(("a", "b"))[0]["c"] = Table.section({"w": 4})
    doc.aot(("a", "b")).reverse()
    assert tomlrt.dumps(doc) == td("""
        [[a.b]]
        [a.b.c]
        z = 3

        [[a.b]]
        x = 1
        [a.b.c]
        w = 4
        """)


def test_overwrite_section_inside_aot_owned_implicit_table() -> None:
    """A structural overwrite stages and restores its replacement block."""
    doc = tomlrt.loads(
        td("""
        [[a]]
        [a.b.c]
        x = 1
        """)
    )
    doc.aot("a")[0].table("b")["c"] = 5
    assert tomlrt.dumps(doc) == td("""
        [[a]]
        [a.b]
        c = 5
        """)


def test_new_section_header_after_mid_doc_header_insert() -> None:
    """A new section created after an earlier, non-tail header insertion
    follows the document's section-spacing convention.
    """
    src = td("""
        [a]
        x = 1

        [middle]
        y = 1
        [b]
        z = 2
        """)
    doc = tomlrt.loads(src)
    # Mid-document: promote an implicit table under [a], physically
    # inserted before [middle]/[b], not at the doc tail.
    doc["a"].install(("sub",), Table.section({"w": 1}))
    doc.install(("c",), Table.section({"v": 3}))
    out = tomlrt.dumps(doc)
    assert _reparses(out)
    reloaded = tomlrt.loads(out)
    assert reloaded["a"]["sub"]["w"] == 1
    assert reloaded["c"]["v"] == 3
    assert out == td("""
        [a]
        x = 1
        [a.sub]
        w = 1

        [middle]
        y = 1
        [b]
        z = 2
        [c]
        v = 3
        """)


def test_section_spacing_convention_survives_last_header_delete() -> None:
    """Section spacing remains the convention learned from the source."""
    doc = tomlrt.loads(
        td("""
        [a]
        x = 1
        [b]
        y = 2

        [c]
        z = 3
        """)
    )
    del doc["c"]
    doc.install(("d",), Table.section({"w": 4}))
    assert tomlrt.dumps(doc) == td("""
        [a]
        x = 1
        [b]
        y = 2

        [d]
        w = 4
        """)


def test_compact_section_spacing_survives_deleting_to_one_header() -> None:
    """A learned compact convention does not depend on header count."""
    doc = tomlrt.loads(
        td("""
        [a]
        x = 1
        [b]
        y = 2
        """)
    )
    del doc["b"]
    doc.install(("c",), Table.section({"z": 3}))
    assert tomlrt.dumps(doc) == td("""
        [a]
        x = 1
        [c]
        z = 3
        """)


def test_new_section_on_freshly_parsed_document_with_no_headers() -> None:
    """A parsed document with only root KVs (no headers at all) still
    computes the first synthesised header's leading correctly."""
    doc = tomlrt.loads("x = 1\n")
    doc.install(("a",), Table.section({"y": 2}))
    assert tomlrt.dumps(doc) == td("""
        x = 1

        [a]
        y = 2
        """)


def test_repeated_section_install_after_large_trailing_content() -> None:
    """Bulk section creation stays correct regardless of how much
    unrelated content follows the insertion point."""
    src = "[a]\nx = 1\n\n[other]\n" + "".join(f"k{i} = {i}\n" for i in range(500))
    doc = tomlrt.loads(src)
    for i in range(10):
        doc.install((f"s{i}",), Table.section({"v": i}))
    out = tomlrt.dumps(doc)
    assert _reparses(out)
    reloaded = tomlrt.loads(out)
    for i in range(10):
        assert reloaded[f"s{i}"]["v"] == i


def test_pop_aot_entry_removes_self_reference_section() -> None:
    """An entry key assigned to the entry itself (turning a scalar into
    a nested section snapshot via ``clone_document_as_section``) must be
    fully removed, header included, when the owning entry is popped."""
    doc = tomlrt.loads(
        td("""
        [fruit]
        apple.color = "red"

        [[fruit.apple.seeds]]
        size = 2
        """)
    )
    entry = doc["fruit"]["apple"]["seeds"][0]
    entry["size"] = entry
    doc["fruit"]["apple"]["seeds"].pop(0)
    assert tomlrt.dumps(doc) == td("""
        [fruit]
        apple.color = "red"
        apple.seeds = []
        """)
    assert doc.to_dict() == {"fruit": {"apple": {"color": "red", "seeds": []}}}


def test_clone_section_into_headerless_implicit_parent_anchors_past_siblings() -> None:
    """Cloning a live section into a key of a headerless implicit table
    must not land the new header between that table's own last KV and
    an unrelated sibling implicit table's later KVs — doing so would
    silently re-scope the sibling's keys under the new header."""
    doc = tomlrt.loads(
        td("""
        apple.type = "fruit"
        orange.type = "fruit"

        apple.skin = "thin"
        orange.skin = "thick"

        apple.color = "red"
        apple.k47 = -7

        orange.color = -7
        k20 = true
        """)
    )
    foreign = tomlrt.loads(
        td("""
        [clients]
        data = [[1, 2]]
        """)
    )
    doc["orange"]["color"] = foreign["clients"]
    assert tomlrt.dumps(doc) == td("""
        apple.type = "fruit"
        orange.type = "fruit"

        apple.skin = "thin"
        orange.skin = "thick"

        apple.color = "red"
        apple.k47 = -7
        k20 = true

        [orange.color]
        data = [[1, 2]]
        """)
    assert doc.to_dict() == {
        "apple": {"type": "fruit", "skin": "thin", "color": "red", "k47": -7},
        "orange": {"type": "fruit", "skin": "thick", "color": {"data": [[1, 2]]}},
        "k20": True,
    }


def test_overwrite_with_own_grandchild_then_clone_elsewhere() -> None:
    """Overwriting a key with one of its own (same-document) descendants
    must snapshot before the old subtree is deleted — deleting it would
    otherwise unlink the descendant's own backing slots before they are
    read, corrupting the clone or (if it later becomes a clone source
    itself) leaving stale host-path bookkeeping behind."""
    doc = tomlrt.loads(
        td("""
        name.first = "Arthur"
        "name".'last' = "Dent"

        many.dots.dot.dot.dot = 42
        """)
    )
    doc["k9"] = -7
    doc["many"]["dots"] = doc["many"]["dots"]["dot"]["dot"]
    doc["many"]["k96"] = 1
    doc["name"]["k75"] = doc["many"]
    out = tomlrt.dumps(doc)
    assert out == td("""
        name.first = "Arthur"
        "name".'last' = "Dent"

        k9 = -7

        [name.k75]
        k96 = 1

        [name.k75.dots]
        dot = 42

        [many]
        k96 = 1

        [many.dots]
        dot = 42
        """)
    assert _reparses(out) == doc.to_dict()


def test_overwrite_aot_entry_key_with_own_grandchild_then_pop_owning_entry() -> None:
    """Overwriting an AoT entry's key with a nested (same-entry) section
    stales the content's old owning entry when that key's whole old
    binding is deleted first. The adopted content must be re-owned by
    the *destination*'s entry, not left with no owner (or the deleted
    one) — otherwise popping the destination entry later leaves the
    adopted section behind instead of removing it."""
    doc = tomlrt.loads(
        td("""
        [[a]]
        [[a.b]]
        [a.b.c]
        d = 1
        [[a.b]]
        [a.b.c]
        d = 2
        """)
    )
    doc["a"][0]["b"] = doc["a"][0]["b"][0]["c"]
    assert tomlrt.dumps(doc) == td("""
        [[a]]
        [a.b]
        d = 1
        """)
    doc["a"].pop(0)
    assert tomlrt.dumps(doc) == "a = []\n"
    assert doc.to_dict() == {"a": []}


def test_clone_section_into_fresh_implicit_intermediate_anchors_locally() -> None:
    """Cloning a live header-bearing section into a fresh implicit
    intermediate (no ``_refs`` of its own yet, e.g. a key of an AoT
    entry that has no other content) must anchor via the nearest
    header-bearing ancestor's own extent, not fall through to a
    ``None`` anchor that lands the block at the document's absolute
    tail — letting a later sibling AoT entry capture it on re-parse."""
    doc = tomlrt.loads(
        td("""
        [[a]]
        a.b.c = 1
        a.b.d = 2

        [[a]]
        a.b = { x = 1 }
        """)
    )
    doc["a"][0]["a"]["k34"] = doc["a"][0]["a"]
    assert tomlrt.dumps(doc) == td("""
        [[a]]
        a.b.c = 1
        a.b.d = 2

        [a.a.k34.b]
        c = 1
        d = 2

        [[a]]
        a.b = { x = 1 }
        """)
    assert doc.to_dict() == {
        "a": [
            {"a": {"b": {"c": 1, "d": 2}, "k34": {"b": {"c": 1, "d": 2}}}},
            {"a": {"b": {"x": 1}}},
        ]
    }


def test_adopt_section_into_fresh_implicit_aot_parent_anchors_locally() -> None:
    """A private section adopted into an earlier AoT entry stays in that entry."""
    doc = tomlrt.loads(
        td("""
        [[arr]]

        [[arr]]
        y = 2

        [donor]
        d = 10
        """)
    )
    orphan = doc["donor"]
    doc["donor"] = {"replaced": True}
    doc.aot("arr")[0].install(("sub", "leaf"), orphan)
    out = tomlrt.dumps(doc)
    assert out == td("""
        donor = { replaced = true }

        [[arr]]

        [arr.sub.leaf]
        d = 10

        [[arr]]
        y = 2
        """)
    assert _reparses(out) == doc.to_dict()


def test_adopt_private_section_unfiles_stale_bindings_for_nested_headers() -> None:
    """``adopt_private_section`` unfiled the stale ancestor bindings for
    the orphan's own header, but not for any nested header within its
    subtree.

    Here ``songs[0]["name"] = songs[1]`` clones sibling entry 1 into a
    new ``[albums.songs.0.name]`` section nested inside entry 0; then
    ``doc["albums"] = songs[0]`` promotes that same entry (header and
    all) to become the new ``[albums]``, detaching it (and everything
    nested inside it, including "name") as one coherent private-orphan
    unit before adopting it back in place.

    "name"'s own ancestor bindings (onto its old host chain — the
    entry / AoT / table hierarchy as it existed before the promotion)
    are never scrubbed by that detach-and-adopt round trip, the way
    ``delete_key`` scrubs the live tree. Left in place, they coexist
    with the fresh bindings ``adopt_private_section`` files after the
    rehome, corrupting "name"'s own back-pointer list with dangling
    entries pointing at containers that are about to be discarded.
    Deleting "name"'s old content and giving it a new section child
    (forcing ``_maybe_demote_synthetic_empty_header`` to scrub one of
    those dangling entries) then crashes trying to unfile a ref that
    isn't where it's expected — silently swallowed if reached through
    the public API's broader exception surface, but a hard crash here.
    """
    doc = tomlrt.loads(
        td("""
        [[albums]]
          [[albums.songs]]
          name = "Jungleland"
          [[albums.songs]]
          name = "Dancing in the Dark"
        """)
    )
    songs = doc["albums"][0]["songs"]
    songs[0]["name"] = songs[1]
    doc["albums"] = songs[0]
    del doc["albums"]["name"]["name"]
    foreign = tomlrt.loads(
        td("""
        [y]
        w = 1
        """)
    )
    doc["albums"]["name"]["k2"] = foreign["y"]
    out = tomlrt.dumps(doc)
    assert out == td("""
        [albums]
          [albums.name.k2]
        w = 1
        """)
    assert _reparses(out) == doc.to_dict()


def test_overwrite_with_leading_dotted_kvs_anchors_past_foreign_scope() -> None:
    """Overwriting a header-bearing key with an implicit source whose
    direct KVs precede its structural children must not anchor those
    leading KVs at the destination's old position when that position
    is physically inside some *other* table's own body.

    ``breed.k40`` used to be header-bearing (``[breed.k40.apple]``),
    sitting right after ``[breed.apple.taste.k78.taste]``'s own body —
    a safe place for a *header* to land (headers carry their own
    scope), but not for the new value's *leading dotted KVs*
    (``sweet``, ``k78.color``), which take their scope from whatever
    header precedes them. Reusing the old anchor for those bare KVs
    would silently re-parent them under ``breed.apple.taste.k78.taste``
    on re-parse.
    """
    doc = tomlrt.loads(
        td("""
        breed.apple.taste.sweet = true
        breed.apple.taste.k78.color = "red"

        [breed.apple.taste.k78.taste]
        sweet = true

        [breed.k40.apple]
        color = "red"
        """)
    )
    doc["breed"]["k40"] = doc["breed"]["apple"]["taste"]
    out = tomlrt.dumps(doc)
    assert out == td("""
        breed.apple.taste.sweet = true
        breed.apple.taste.k78.color = "red"
        breed.k40.sweet = true
        breed.k40.k78.color = "red"

        [breed.k40.k78.taste]
        sweet = true

        [breed.apple.taste.k78.taste]
        sweet = true
        """)
    assert _reparses(out) == doc.to_dict()


def test_clone_implicit_source_with_empty_string_header_child() -> None:
    """``_install_attached_subtree`` recursively installs a headerless
    source's structural children via ``Container.install()``'s tuple-
    path API. That API validates each path *segment*, rejecting an
    empty one — reasonable for a human-supplied dotted path (an empty
    segment there is almost certainly a typo), but wrong for this
    internal recursive use: the key comes from an already-live source
    Container, where an empty string is a legal (if unusual) TOML key,
    not user input to second-guess."""
    doc = tomlrt.loads(
        td("""
        x = 1
        src.k = 1

        [src.""]
        y = 1
        """)
    )
    doc["a"] = doc["src"]
    out = tomlrt.dumps(doc)
    assert out == td("""
        x = 1
        src.k = 1
        a.k = 1

        [a.""]
        y = 1

        [src.""]
        y = 1
        """)
    assert _reparses(out) == doc.to_dict()


def test_adopt_private_section_anchors_past_unrelated_trailing_kv() -> None:
    """``adopt_private_section`` had no safety check on its anchor at
    all: the block it splices in always brings its own header, so
    landing it right after a destination's own last KV — when an
    unrelated bare KV (belonging to some other table entirely) happens
    to sit physically right after that point — silently re-parents
    that unrelated KV under the new header on re-parse."""
    doc = tomlrt.loads(
        td("""
        site.k97 = -7
        k63 = { a = 1 }

        [tmp]
        x = 1
        """)
    )
    tmp = doc.pop("tmp")
    doc["site"]["google"] = tmp
    out = tomlrt.dumps(doc)
    assert out == td("""
        site.k97 = -7
        k63 = { a = 1 }

        [site.google]
        x = 1
        """)
    assert _reparses(out) == doc.to_dict()


def test_adopt_private_implicit_leaves_body_tail_stale() -> None:
    """``adopt_private_implicit`` re-files ancestor-chain refs for the
    rebased dotted KVs it adopts, but was not advancing those
    ancestors' ``_body_tail``: a later direct append to the same
    implicit ancestor then wrongly saw an empty body and tried to
    synthesise a fresh header for it, crashing when its anchor slot
    (deep inside a header-less chain) couldn't be found on the
    document root's own refs."""
    doc = tomlrt.loads(
        td("""
        [a]
        k61 = { a = 1 }

        [a.few.dots]
        polka.dot = "again?"
        polka.dance-with = "Dot"
        """)
    )
    doc["a"]["few"]["dots"] = doc["a"]["few"]["dots"]["polka"]
    doc["a"]["few"]["k29"] = [1, 2]
    out = tomlrt.dumps(doc)
    assert out == td("""
        [a]
        k61 = { a = 1 }

        few.dots.dot = "again?"
        few.dots.dance-with = "Dot"
        few.k29 = [1, 2]
        """)
    assert _reparses(out) == doc.to_dict()


def test_adopt_private_implicit_transfers_aot_entry_ownership() -> None:
    doc = tomlrt.loads(
        td("""
            [tbl]
            a.b.c = 42

            [[arr]]
            a.b.c = 3
            a.b.d = 4
            """)
    )
    target = doc["tbl"]["a"]
    target["b"] = doc["arr"]
    target["b"] = target["b"][0]["a"]

    target["k"] = "str"

    out = tomlrt.dumps(doc)
    assert out == td("""
        [tbl]

        a.b.b.c = 3
        a.b.b.d = 4
        a.k = "str"

        [[arr]]
        a.b.c = 3
        a.b.d = 4
        """)
    assert _reparses(out) == doc.to_dict()


def test_materialise_empty_aot_anchors_within_owning_entry() -> None:
    """When popping an AoT's last entry leaves its container empty,
    ``_materialise_empty_aot`` synthesises a ``key = []`` placeholder
    which, if the container had no header of its own, promotes it to
    an explicit section via ``_synthesise_header_then_insert_kv``. That
    promotion fell back to the document's absolute tail whenever no
    anchor slot was available, ignoring that the container is nested
    inside an AoT entry that is *not* the last thing in the document —
    landing the new header after a later sibling entry, which a
    re-parse then misattributes to that sibling."""
    doc = tomlrt.loads(
        td("""
        [[a.b]]
        [[a.b.x.songs]]
        name = 1

        [[a.b]]
        name = 2
        """)
    )
    doc["a"]["b"][0]["x"]["songs"].pop(0)
    out = tomlrt.dumps(doc)
    assert out == td("""
        [[a.b]]
        [a.b.x]
        songs = []

        [[a.b]]
        name = 2
        """)
    assert _reparses(out) == doc.to_dict()


def test_dotted_kv_chain_ref_files_before_unrelated_descendant_header() -> None:
    """``install_dotted_kv_slot``'s ancestor-chain ref-filing reused one
    shared anchor slot (computed at ``host``'s level) for every ancestor
    in the chain, including ones that don't themselves hold a ref to
    it. When such an ancestor's only existing ref was a nested
    descendant's header, the fallback filed the new ref *after* that
    header — even though a headerless ancestor's own dotted content
    always physically precedes any of its descendant headers. The
    ancestor's ``_refs`` ended up out of doc-stream order, and a later
    structural replacement keyed off that ordering silently detached
    an entire cloned AoT from the document."""
    doc = tomlrt.loads(
        td("""
        top.key = 1

        [a.few.dots]
        polka.dot = "again?"

        [tbl]
        a.b.c = 42.666
        """)
    )
    doc["a"]["k6"] = doc["tbl"]["a"]["b"]
    doc["a"]["few"]["k46"] = doc["a"]["k6"]
    foreign = tomlrt.loads(
        td("""
        [[arr]]
        a.b.c = 1
        a.b.d = 2
        [[arr]]
        a.b.c = 3
        a.b.d = 4
        """)
    )
    doc["a"]["few"] = foreign["arr"]
    out = tomlrt.dumps(doc)
    assert out == td("""
        top.key = 1
        a.k6.c = 42.666

        [[a.few]]
        a.b.c = 1
        a.b.d = 2
        [[a.few]]
        a.b.c = 3
        a.b.d = 4

        [tbl]
        a.b.c = 42.666
        """)
    assert _reparses(out) == doc.to_dict()


def test_dotted_kv_chain_body_tail_propagation_ignores_unrelated_root_content() -> None:
    """``install_dotted_kv_slot``'s ancestor-chain loop advanced each
    ancestor's ``_body_tail`` only when it matched the *host*-level
    anchor computed once at the top of the function. For an
    intermediate ancestor whose own body_tail already diverged from
    host's (e.g. host is the document root and some unrelated sibling
    key's content is root's own latest, while this ancestor tracks only
    its own subtree's latest), that condition never matched, so the
    ancestor's ``_body_tail`` went stale after the very first KV filed
    under it. A later structural overwrite of that ancestor's key,
    which reads ``_body_tail`` to find its anchor, then silently
    dropped the replacement content instead of installing it."""
    doc = tomlrt.loads(
        td("""
        physical.color = "orange"
        site.k1 = true
        """)
    )
    foreign1 = tomlrt.loads('k74."google.com" = true\n')
    doc["physical"]["k74"] = foreign1["k74"]
    foreign2 = tomlrt.loads(
        td("""
        sub.name = "a"
        sub.color = "b"
        sub.flavor = "c"
        """)
    )
    doc["physical"]["k74"]["google.com"] = foreign2["sub"]
    doc["physical"]["k74"] = "x y"
    out = tomlrt.dumps(doc)
    assert out == td("""
        physical.color = "orange"
        site.k1 = true
        physical.k74 = "x y"
        """)
    assert _reparses(out) == doc.to_dict()


def test_dotted_kv_chain_anchors_by_physical_position_not_cached_tail() -> None:
    """``install_dotted_kv_slot``'s ancestor-chain ref-filing used each
    ancestor's own cached ``_body_tail`` as its anchor. That cache can
    be *later* (physically) than the slot actually being spliced —
    here the document root's cached tail is a sibling key positioned
    after the dotted branch a nested clone is extending — so filing the
    new ref there rewound the root's already-correct tail instead of
    advancing it, leaving the ancestor's refs (and thus a later
    structural overwrite's saved anchor) out of doc-stream order and
    silently dropping the replacement content on render."""
    doc = tomlrt.loads(
        td("""
        a.p.q = 8
        d1.a.x = 9

        [tbl]
        k = 12
        """)
    )
    doc["a"]["p"]["r"] = -7
    doc["k47"] = 1
    doc["d1"]["a"]["k76"] = doc["a"]
    doc["d1"]["a"] = doc["a"]["p"]
    out = tomlrt.dumps(doc)
    assert out == td("""
        a.p.q = 8
        a.p.r = -7
        d1.a.q = 8
        d1.a.r = -7
        k47 = 1

        [tbl]
        k = 12
        """)
    assert _reparses(out) == doc.to_dict()


def test_sort_mixed_headerless_keys_keeps_all_leaves_before_headers() -> None:
    doc = tomlrt.loads(
        td("""
        3.14159 = "x y"
        3.k91.14159 = "x y"
        3.k91.k24.14159 = "x y"
        3.k91.k24.k81 = -7
        3.k61.14159 = "x y"

        [3.k61.k91]
        14159 = "x y"
        k81 = -7

        [3.k61.k91.k24]
        14159 = "x y"
        k81 = -7
        k49 = 3.5

        [3.k91.k56]
        14159 = "x y"
        k81 = -7
        k5 = 1

        [3.k91.k56.k24]
        14159 = "x y"
        k81 = -7
        k49 = 3.5
        """)
    )
    doc["3"].sort()
    out = tomlrt.dumps(doc)
    assert out == td("""
        3.14159 = "x y"
        3.k61.14159 = "x y"
        3.k91.14159 = "x y"
        3.k91.k24.14159 = "x y"
        3.k91.k24.k81 = -7

        [3.k61.k91]
        14159 = "x y"
        k81 = -7

        [3.k61.k91.k24]
        14159 = "x y"
        k81 = -7
        k49 = 3.5

        [3.k91.k56]
        14159 = "x y"
        k81 = -7
        k5 = 1

        [3.k91.k56.k24]
        14159 = "x y"
        k81 = -7
        k49 = 3.5
    """)
    assert _reparses(out) == doc.to_dict()


def test_reposition_install_leaf_kv_with_no_preceding_header_is_left_unmoved() -> None:
    """``_effective_header_path_before`` walks backward from the saved
    anchor to find the header currently governing that position, so
    ``reposition_install`` can tell whether a replacement block's
    leading bare KV would keep its original scope if moved back there.
    When the anchor sits at the very start of the document (nothing
    precedes it, so no header governs it at all), the walk must
    terminate at doc head and report root scope rather than looping or
    raising."""
    doc = tomlrt.loads(
        td("""
        top = 1
        a.x = 1

        [a.sub]
        y = 2
        """)
    )
    doc["top"] = doc["a"]
    out = tomlrt.dumps(doc)
    assert out == td("""
        a.x = 1
        top.x = 1

        [top.sub]
        y = 2

        [a.sub]
        y = 2
        """)
    assert _reparses(out) == doc.to_dict()


def test_adopt_private_implicit_splices_before_head_when_host_body_empty() -> None:
    """``adopt_private_implicit`` anchors a headerless orphan's rebased
    dotted content at its host's own direct-KV extent — but when the
    host (here the document root) has no direct KVs of its own yet
    (only section headers), that anchor is ``None`` even though the
    document isn't empty. Splicing at the doc's absolute tail in that
    case would be unsafe (it could land inside an unrelated later
    header's scope), so it must splice before the doc head instead."""
    doc = tomlrt.loads(
        td("""
        animal.type.name = "pug"
        animal.type.breed = "corgi"

        [name]
        first = "Tom"
        """)
    )
    doc["animal"] = doc["animal"]["type"]
    out = tomlrt.dumps(doc)
    assert out == td("""
        animal.name = "pug"
        animal.breed = "corgi"

        [name]
        first = "Tom"
        """)
    assert _reparses(out) == doc.to_dict()


def test_adopt_private_implicit_anchors_before_unrelated_later_header() -> None:
    """``adopt_private_implicit`` anchored a rehomed headerless block at
    ``_parent_subtree_tail(host)``. For a headerless ``host`` with no
    direct-KV extent of its own (the document root, once some other
    unrelated section already has content), that function walks the
    *whole* physical subtree reachable via the ancestor-chain ref every
    slot files on every ancestor — spanning the entire document, not
    just ``host``'s own body. Anchoring there lands the rehomed block
    after an unrelated later header's own content, silently
    re-parenting it under that header on re-parse.
    """
    doc = tomlrt.loads(
        td("""
        a.b = 1
        a.c = 2

        [other]
        x = 1
        """)
    )
    orphan = doc.pop("a")
    doc["z"] = orphan
    out = tomlrt.dumps(doc)
    assert out == td("""
        z.b = 1
        z.c = 2

        [other]
        x = 1
        """)
    assert _reparses(out) == doc.to_dict()


def test_adopt_private_section_adds_terminator_when_not_at_doc_tail() -> None:
    """A private orphan detached via ``pop()`` may have lost its own
    trailing newline (it used to be the very last thing in its source
    document). ``adopt_private_section`` must add one back when the
    orphan's content lands anywhere but this document's new tail —
    exercised here by adopting it in between two other sections."""
    doc = tomlrt.loads(
        td("""
        [dest]
        k = 1

        [other]
        x = 1

        [trailing]
        y = 1
        """)
    )
    orphan = doc.pop("other")
    doc["dest"]["sub"] = orphan
    out = tomlrt.dumps(doc)
    assert out == td("""
        [dest]
        k = 1

        [dest.sub]
        x = 1

        [trailing]
        y = 1
        """)
    assert _reparses(out) == doc.to_dict()


def test_reposition_install_scattered_source_via_disjoint_span_fallback() -> None:
    """``_recorded_install_span`` detects when an implicit source's
    recorded slots don't form one contiguous doc-stream span — the
    direct KVs and structural children land at different anchors, e.g.
    because the destination promoted from headerless to header-bearing
    partway through — and signals ``reposition_install`` to leave the
    fresh install where it landed rather than risk moving the wrong
    range. Exercises both of its rejection paths: more than one
    candidate span head, and a span head whose forward walk doesn't
    reach every recorded slot."""
    doc = tomlrt.loads(
        td("""
        name = "Orange"
        physical.color = "orange"
        physical.shape = "round"
        site."google.com" = true
        """)
    )
    doc["site"]["k96"] = doc["site"]
    doc["physical"]["shape"] = doc["site"]["k96"]
    doc["site"]["k96"]["google.com"] = doc["site"]
    doc["site"].sort()
    doc["name"] = doc["site"]
    out = tomlrt.dumps(doc)
    assert out == td("""
        physical.color = "orange"
        physical.shape."google.com" = true
        site."google.com" = true
        name."google.com" = true

        [site.k96."google.com"]
        "google.com" = true

        [site.k96."google.com".k96]
        "google.com" = true

        [name.k96."google.com"]
        "google.com" = true

        [name.k96."google.com".k96]
        "google.com" = true
        """)
    assert _reparses(out) == doc.to_dict()


def test_overwrite_aot_entry_key_with_ancestor_aot_snapshots_first() -> None:
    """``_snapshot_for_overlapping_install`` must snapshot an AoT
    value that is an *ancestor* of the destination being overwritten
    (not just a headerless Table descendant) before
    ``reposition_install`` deletes the old binding — otherwise deleting
    the destination's old content would unlink slots the AoT value
    itself still needs to read, corrupting the clone."""
    doc = tomlrt.loads(
        td("""
        [[a]]
        b = 1
        [[a]]
        b = 2
        """)
    )
    doc["a"][0]["b"] = doc["a"]
    out = tomlrt.dumps(doc)
    assert out == td("""
        [[a]]
        [[a.b]]
        b = 1

        [[a.b]]
        b = 2
        [[a]]
        b = 2
        """)
    assert _reparses(out) == doc.to_dict()
