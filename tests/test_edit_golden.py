"""Golden tests for editing operations.

Each test exercises a mutation pathway and asserts the *exact* rendered
output, so any regression in trivia handling is immediately visible.

Coverage targets:
- discontiguous tables (multiple physical sections for one logical table)
- dotted-key tables (logical table created via dotted keys)
- AoT middle/append/insert ops; AoT entries with nested sub-sections
- inline-table edits round-tripping
- cross-document assignment with deep-clone semantics
- mutation interaction with logical-view scoping for AoT entries
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from _helpers import reparses as _reparses
from _helpers import td

if TYPE_CHECKING:
    from collections.abc import Callable

import pytest

import tomlrt
from tomlrt import AoT, Array, Document, Table

# ---------------------------------------------------------------------------
# Discontiguous tables: [a] / [a.sub] / [b] / [a] is forbidden, but a
# logical table can still aggregate keys from the [a] header *and* from
# any [a.x] sub-section headers.
# ---------------------------------------------------------------------------


def test_table_with_sub_section_iter_includes_subtable() -> None:
    src = td("""
        [a]
        x = 1
        [a.sub]
        y = 2
        """)
    doc = tomlrt.loads(src)
    a = doc.table("a")
    assert a["x"] == 1
    sub = a.table("sub")
    assert sub["y"] == 2


def test_table_with_sub_section_modify_subtable_value() -> None:
    src = td("""
        [a]
        x = 1
        [a.sub]
        y = 2
        """)
    doc = tomlrt.loads(src)
    a = doc.table("a")
    sub = a.table("sub")
    sub["y"] = 99
    out = tomlrt.dumps(doc)
    assert out == td("""
        [a]
        x = 1
        [a.sub]
        y = 99
        """)


def test_table_with_sub_section_add_to_parent_appends_in_parent_block() -> None:
    src = td("""
        [a]
        x = 1
        [a.sub]
        y = 2
        """)
    doc = tomlrt.loads(src)
    a = doc.table("a")
    a["z"] = 3
    out = tomlrt.dumps(doc)
    # New parent-level key must land in the [a] block, BEFORE [a.sub] —
    # putting it after would make it semantically belong to [a.sub] under
    # TOML's "headers terminate a section" rule.
    assert out == td("""
        [a]
        x = 1
        z = 3
        [a.sub]
        y = 2
        """)
    assert _reparses(out) == {"a": {"x": 1, "z": 3, "sub": {"y": 2}}}


# ---------------------------------------------------------------------------
# Dotted-key tables (logical table only ever lives as dotted keys)
# ---------------------------------------------------------------------------


def test_dotted_key_table_read() -> None:
    src = "a.b = 1\na.c = 2\n"
    doc = tomlrt.loads(src)
    a = doc.table("a")
    assert dict(a) == {"b": 1, "c": 2}


def test_dotted_key_table_set_via_subtable_adds_dotted_entry() -> None:
    """Setting a new key on a dotted-only table appends a new dotted KV."""
    src = "a.b = 1\na.c = 2\n"
    doc = tomlrt.loads(src)
    a = doc.table("a")
    a["d"] = 3
    assert dict(a) == {"b": 1, "c": 2, "d": 3}
    assert tomlrt.dumps(doc) == td("""
        a.b = 1
        a.c = 2
        a.d = 3
        """)


def test_dotted_key_table_overwrite_via_subtable() -> None:
    """Overwriting a dotted leaf preserves position and surrounding trivia."""
    src = td("""
        # header
        a.b = 1 # cmt
        a.c = 2
        """)
    doc = tomlrt.loads(src)
    a = doc.table("a")
    a["b"] = 99
    assert dict(a) == {"b": 99, "c": 2}
    assert tomlrt.dumps(doc) == td("""
        # header
        a.b = 99 # cmt
        a.c = 2
        """)


def test_dotted_key_deep_overwrite_via_subtable() -> None:
    """Overwriting a deep dotted leaf is also in-place."""
    src = td("""
        a.b.x = 1 # leaf
        a.b.y = 2
        """)
    doc = tomlrt.loads(src)
    b = doc.table("a").table("b")
    b["x"] = 99
    assert dict(b) == {"x": 99, "y": 2}
    assert tomlrt.dumps(doc) == td("""
        a.b.x = 99 # leaf
        a.b.y = 2
        """)


def test_dotted_key_table_delete_via_subtable() -> None:
    src = td("""
        a.b = 1
        a.c = 2
        a.d = 3
        """)
    doc = tomlrt.loads(src)
    a = doc.table("a")
    del a["c"]
    assert dict(a) == {"b": 1, "d": 3}
    assert "c" not in tomlrt.dumps(doc)


def test_dotted_key_table_delete_missing_raises() -> None:
    src = "a.b = 1\n"
    doc = tomlrt.loads(src)
    a = doc.table("a")
    with pytest.raises(KeyError):
        del a["nope"]


def test_dotted_key_table_set_overwrites_subtree() -> None:
    src = td("""
        a.b.x = 1
        a.b.y = 2
        a.c = 3
        """)
    doc = tomlrt.loads(src)
    a = doc.table("a")
    a["b"] = 99
    assert dict(a) == {"c": 3, "b": 99}


def test_dotted_key_nested_subtable_set() -> None:
    """Setting a key on a deeply-nested dotted view works too."""
    src = "a.b.x = 1\na.b.y = 2\n"
    doc = tomlrt.loads(src)
    a = doc.table("a")
    b = a.table("b")
    b["z"] = 3
    assert dict(b) == {"x": 1, "y": 2, "z": 3}
    assert tomlrt.dumps(doc) == td("""
        a.b.x = 1
        a.b.y = 2
        a.b.z = 3
        """)


def test_inline_dotted_subtable_set() -> None:
    """Same thing inside an inline table."""
    src = "t = { a.b = 1, a.c = 2 }\n"
    doc = tomlrt.loads(src)
    t = doc.table("t")
    a = t.table("a")
    a["d"] = 3
    assert dict(a) == {"b": 1, "c": 2, "d": 3}
    assert tomlrt.dumps(doc) == "t = { a.b = 1, a.c = 2, a.d = 3 }\n"


def test_inline_dotted_subtable_overwrite() -> None:
    """Overwriting a dotted leaf inside an inline table is in-place too."""
    src = "t = { a.b = 1, a.c = 2 }\n"
    doc = tomlrt.loads(src)
    a = doc.table("t").table("a")
    a["b"] = 99
    assert dict(a) == {"b": 99, "c": 2}
    assert tomlrt.dumps(doc) == "t = { a.b = 99, a.c = 2 }\n"


def test_inline_dotted_subtable_delete() -> None:
    src = "t = { a.b = 1, a.c = 2 }\n"
    doc = tomlrt.loads(src)
    t = doc.table("t")
    a = t.table("a")
    del a["b"]
    assert dict(a) == {"c": 2}


# ---------------------------------------------------------------------------
# Arrays-of-tables (AoT) — middle ops and entries with sub-sections
# ---------------------------------------------------------------------------


def test_aot_basic_iteration() -> None:
    src = td("""
        [[users]]
        name = "alice"
        [[users]]
        name = "bob"
        """)
    doc = tomlrt.loads(src)
    users = doc.aot("users")
    assert [u["name"] for u in users] == ["alice", "bob"]


def test_aot_append_entry_via_dict() -> None:
    src = '[[users]]\nname = "alice"\n'
    doc = tomlrt.loads(src)
    users = doc.aot("users")
    users.append({"name": "bob"})
    out = tomlrt.dumps(doc)
    assert _reparses(out) == {"users": [{"name": "alice"}, {"name": "bob"}]}


def test_aot_modify_field_in_first_entry() -> None:
    src = td("""
        [[users]]
        name = "alice"
        [[users]]
        name = "bob"
        """)
    doc = tomlrt.loads(src)
    users = doc.aot("users")
    users[0]["name"] = "ALICE"
    out = tomlrt.dumps(doc)
    assert out == td("""
        [[users]]
        name = "ALICE"
        [[users]]
        name = "bob"
        """)


def test_aot_modify_field_in_middle_entry() -> None:
    src = td("""
        [[users]]
        name = "a"
        [[users]]
        name = "b"
        [[users]]
        name = "c"
        """)
    doc = tomlrt.loads(src)
    users = doc.aot("users")
    users[1]["name"] = "B"
    out = tomlrt.dumps(doc)
    assert out == td("""
            [[users]]
            name = "a"
            [[users]]
            name = "B"
            [[users]]
            name = "c"
            """)


def test_aot_entry_sub_section_read() -> None:
    """[[arr]] / [arr.sub] — sub belongs to the AoT entry."""
    src = td("""
        [[arr]]
        x = 1
        [arr.sub]
        y = 2
        [[arr]]
        x = 10
        [arr.sub]
        y = 20
        """)
    doc = tomlrt.loads(src)
    arr = doc.aot("arr")
    assert len(arr) == 2
    assert arr[0]["x"] == 1
    sub0 = arr[0].table("sub")
    assert sub0["y"] == 2
    assert arr[1]["x"] == 10
    sub1 = arr[1].table("sub")
    assert sub1["y"] == 20


def test_aot_entry_sub_section_modify_value() -> None:
    src = td("""
        [[arr]]
        x = 1
        [arr.sub]
        y = 2
        [[arr]]
        x = 10
        [arr.sub]
        y = 20
        """)
    doc = tomlrt.loads(src)
    arr = doc.aot("arr")
    sub = arr[1].table("sub")
    sub["y"] = 999
    out = tomlrt.dumps(doc)
    assert out == (
        td("""
            [[arr]]
            x = 1
            [arr.sub]
            y = 2
            [[arr]]
            x = 10
            [arr.sub]
            y = 999
            """)
    )
    assert _reparses(out) == {
        "arr": [
            {"x": 1, "sub": {"y": 2}},
            {"x": 10, "sub": {"y": 999}},
        ]
    }


def test_aot_entry_install_subsection_does_not_overwrite_sibling() -> None:
    doc = tomlrt.loads("")
    doc["package"] = tomlrt.AoT()
    e1 = doc["package"].add({"name": "foo"})
    e1["dependencies"] = tomlrt.Table.section({"req-foo": ">=1"})
    e2 = doc["package"].add({"name": "bar"})
    e2["dependencies"] = tomlrt.Table.section({"req-bar": ">=1"})
    out = tomlrt.dumps(doc)
    assert out == (
        '[[package]]\nname = "foo"\n\n'
        '[package.dependencies]\nreq-foo = ">=1"\n\n'
        '[[package]]\nname = "bar"\n\n'
        '[package.dependencies]\nreq-bar = ">=1"\n'
    )
    assert tomlrt.dumps(tomlrt.loads(out)) == out


def test_root_section_after_aot_does_not_split_entry_body() -> None:
    doc = tomlrt.loads("")
    doc["package"] = tomlrt.AoT()
    doc["package"].add({"name": "poetry-core", "version": "2.2.1"})
    doc["metadata"] = tomlrt.Table.section({"lock-version": "2.1"})
    out = tomlrt.dumps(doc)
    assert out == (
        "[[package]]\n"
        'name = "poetry-core"\n'
        'version = "2.2.1"\n\n'
        "[metadata]\n"
        'lock-version = "2.1"\n'
    )
    assert tomlrt.dumps(tomlrt.loads(out)) == out


def test_root_section_after_multi_entry_aot() -> None:
    """The walk to parent's subtree tail must traverse multiple AoT entry
    owners and land after the *last* entry's body."""
    doc = tomlrt.loads("[[package]]\nname='a'\n\n[[package]]\nname='b'\n")
    doc["metadata"] = tomlrt.Table.section({"v": 1})
    out = tomlrt.dumps(doc)
    assert "[metadata]" in out
    assert out.index("[metadata]") > out.index("name='b'")
    assert tomlrt.dumps(tomlrt.loads(out)) == out


def test_clone_section_into_aot_entry_with_sibling_entry() -> None:
    """Guard against the lax fix: when parent is entry 0 of [[a]] and
    entry 1 exists, the new sub-section must land between the two
    entries — not after entry 1."""
    src = tomlrt.loads("[src]\nz = 3\n")
    doc = tomlrt.loads("[[a]]\nx = 1\n\n[[a]]\ny = 2\n")
    doc["a"][0]["child"] = src["src"]
    out = tomlrt.dumps(doc)
    assert out == ("[[a]]\nx = 1\n\n[a.child]\nz = 3\n\n[[a]]\ny = 2\n")
    assert tomlrt.dumps(tomlrt.loads(out)) == out


def test_subsection_into_aot_entry_with_nested_sub_aot() -> None:
    """Strict-descendant rule: when entry 0 already has a nested [[a.b]]
    sub-AoT, a new [a.child] under entry 0 must not split the nested
    AoT header from its body."""
    doc = tomlrt.loads("[[a]]\nx = 1\n\n[[a.b]]\ny = 2\n")
    doc["a"][0]["child"] = tomlrt.Table.section({"z": 3})
    out = tomlrt.dumps(doc)
    # Whatever the chosen ordering, [a.child] must NOT land between
    # [[a.b]] and its body.
    parsed = tomlrt.loads(out)
    assert parsed["a"][0]["b"][0]["y"] == 2
    assert parsed["a"][0]["child"]["z"] == 3
    assert tomlrt.dumps(parsed) == out


def test_ensure_table_multi_component_anchors_under_parent() -> None:
    doc = tomlrt.loads("[tool]\nx = 1\n\n[other]\ny = 1\n")
    doc.ensure_table(("tool", "a", "b"))
    out = tomlrt.dumps(doc)
    assert out.index("[tool.a.b]") < out.index("[other]")
    assert tomlrt.dumps(tomlrt.loads(out)) == out


def test_cross_doc_update_preserves_dotted_super_table() -> None:
    src = tomlrt.loads("[tool.poetry]\npackages = []\n")

    dst1 = tomlrt.loads("")
    dst1.update(src)
    out1 = tomlrt.dumps(dst1)
    assert "[tool]\n" not in out1
    assert "[tool.poetry]" in out1

    dst2 = tomlrt.loads("")
    for k, v in src.items():
        dst2[k] = v
    out2 = tomlrt.dumps(dst2)
    assert "[tool]\n" not in out2
    assert "[tool.poetry]" in out2


def test_cross_doc_array_assignment_preserves_multiline_layout() -> None:
    src = tomlrt.loads(
        '[project]\ndependencies = [\n    "foo>=2.0",\n    "bar>=1.0",\n]\n'
    )
    dst = tomlrt.loads('[project]\ndependencies = ["x"]\n')
    dst["project"]["dependencies"] = src["project"]["dependencies"]
    assert (
        tomlrt.dumps(dst)
        == '[project]\ndependencies = [\n    "foo>=2.0",\n    "bar>=1.0",\n]\n'
    )
    # Source is independent of subsequent destination mutation.
    dst["project"]["dependencies"].remove("foo>=2.0")
    assert "foo>=2.0" in tomlrt.dumps(src)


def test_cross_doc_inline_table_assignment_preserves_spacing() -> None:
    src = tomlrt.loads('owner = { name = "tom",  dob = 1979 }\n')
    dst = tomlrt.loads("owner = { x = 1 }\n")
    dst["owner"] = src["owner"]
    assert tomlrt.dumps(dst) == 'owner = { name = "tom",  dob = 1979 }\n'


# ---------------------------------------------------------------------------
# Inline tables and arrays — round-trip edits
# ---------------------------------------------------------------------------


def test_inline_table_modify_preserves_spacing() -> None:
    src = "owner = { name = 'tom', dob = 1979 }\n"
    doc = tomlrt.loads(src)
    owner = doc.table("owner")
    owner["name"] = "tim"
    out = tomlrt.dumps(doc)
    # Style of the replaced scalar regenerates as basic-quoted (default),
    # but surrounding spacing/comma trivia is preserved.
    assert out == 'owner = { name = "tim", dob = 1979 }\n'


def test_inline_array_modify_preserves_brackets() -> None:
    src = "ports = [ 80, 443, 8080 ]\n"
    doc = tomlrt.loads(src)
    ports = doc.array("ports")
    ports[1] = 444
    out = tomlrt.dumps(doc)
    assert out == "ports = [ 80, 444, 8080 ]\n"


def test_array_insert_then_pop_round_trips() -> None:
    src = "ports = [80, 443]\n"
    doc = tomlrt.loads(src)
    ports = doc.array("ports")
    ports.insert(1, 8080)
    assert list(ports) == [80, 8080, 443]
    ports.pop(1)
    out = tomlrt.dumps(doc)
    assert out == "ports = [80, 443]\n"


# ---------------------------------------------------------------------------
# Cross-document assignment — must deep-clone, never share state
# ---------------------------------------------------------------------------


def test_cross_doc_table_assign_deep_clones() -> None:
    src1 = td("""
        [srv]
        host = "a.example"
        port = 80
        """)
    src2 = ""
    a = tomlrt.loads(src1)
    b = tomlrt.loads(src2)
    b["srv"] = a["srv"]
    # Mutating `a` must not affect `b`.
    a_srv = a.table("srv")
    a_srv["port"] = 9999
    out_a = tomlrt.dumps(a)
    out_b = tomlrt.dumps(b)
    assert _reparses(out_a) == {"srv": {"host": "a.example", "port": 9999}}
    assert _reparses(out_b) == {"srv": {"host": "a.example", "port": 80}}


def test_cross_doc_aot_assign_deep_clones() -> None:
    src1 = td("""
        [[users]]
        name = "alice"
        [[users]]
        name = "bob"
        """)
    src2 = ""
    a = tomlrt.loads(src1)
    b = tomlrt.loads(src2)
    b["users"] = a["users"]
    a_users = a.aot("users")
    a_users[0]["name"] = "MUT"
    assert _reparses(tomlrt.dumps(a))["users"][0]["name"] == "MUT"
    assert _reparses(tomlrt.dumps(b))["users"][0]["name"] == "alice"


def test_cross_doc_array_assign_deep_clones() -> None:
    src1 = "ports = [80, 443]\n"
    src2 = ""
    a = tomlrt.loads(src1)
    b = tomlrt.loads(src2)
    b["ports"] = a["ports"]
    a_ports = a.array("ports")
    a_ports.append(8080)
    assert _reparses(tomlrt.dumps(a))["ports"] == [80, 443, 8080]
    assert _reparses(tomlrt.dumps(b))["ports"] == [80, 443]


def test_cross_doc_table_assign_with_nested_aot() -> None:
    """Cross-doc copy of a section that contains an AoT in its subtree."""
    src = (
        '[project]\nname = "foo"\n\n'
        '[[tool.poetry.source]]\nname = "pypi"\n'
        'url = "https://pypi.org/simple"\n\n'
        '[build-system]\nrequires = ["poetry-core"]\n'
    )
    a = tomlrt.loads(src)
    b = Document()
    for k, v in a.items():
        b[k] = v
    out = tomlrt.dumps(b)
    # Logical content matches.
    assert _reparses(out) == _reparses(src)
    # The nested AoT lands as ``[[..]]`` rather than getting flattened
    # to an inline table or raising.
    assert "[[tool.poetry.source]]" in out
    # Implicit super-tables stay implicit: no empty ``[tool]`` /
    # ``[tool.poetry]`` headers polluting the output.
    for line in out.splitlines():
        stripped = line.strip()
        assert stripped not in ("[tool]", "[tool.poetry]")
    # And mutating the source must not bleed into the destination.
    a_src = a["tool"]["poetry"].aot("source")
    a_src[0]["name"] = "MUT"
    assert "MUT" not in tomlrt.dumps(b)


def test_cross_doc_table_assign_with_explicit_header_and_nested_aot() -> None:
    """Issue #108: cross-doc whole-section assignment must preserve nested AoT.

    The source has an explicit ``[a]`` header (so cross-doc copy of ``a``
    goes through ``clone_section_as_section`` rather than the implicit
    subtree-walk branch). The body contains an AoT (``[[a.x]]``). The
    AoT must survive as ``[[a.x]]`` in the destination, not be downgraded
    to two ``[a.x]`` headers (invalid TOML).
    """
    src = tomlrt.loads("[a]\n[[a.x]]\ny = 1\n\n[[a.x]]\ny = 2\n")
    dst = Document()
    dst["a"] = src["a"]
    out = dst.render()
    assert "[[a.x]]" in out
    assert _reparses(out) == _reparses("[a]\n[[a.x]]\ny = 1\n\n[[a.x]]\ny = 2\n")


def test_aot_append_entry_preserves_nested_aot() -> None:
    """Appending a parent AoT entry must preserve its nested AoT children.

    ``aot.append(some_entry)`` (``clone_aot_entry``) historically copied
    only the entry's own ``entry_slots``, which excludes slots owned by
    nested ``[[a.x]]`` entries living physically inside the parent
    entry. The nested entries were silently dropped — output stayed
    valid TOML but ``q`` values were lost. Same class as #108.
    """
    src = tomlrt.loads(
        "[[outer]]\nv = 1\n[outer.sub]\ns = 2\n"
        "[[outer.aot]]\nq = 3\n[[outer.aot]]\nq = 4\n",
    )
    dst = tomlrt.loads("[[outer]]\nv = 10\n")
    dst["outer"].append(src["outer"][0])
    out = dst.render()
    reparsed = _reparses(out)
    assert reparsed["outer"][1] == {
        "v": 1,
        "sub": {"s": 2},
        "aot": [{"q": 3}, {"q": 4}],
    }


def test_aot_replace_entry_preserves_nested_aot() -> None:
    """Replacing an AoT entry (``aot[i] = src_entry``) preserves nested AoTs.

    ``replace_aot_entry_with_clone`` historically copied only
    ``src_entry.entry_slots``, which excludes slots owned by nested
    ``[[a.x]]`` entries living physically inside the source entry —
    those entries were silently dropped. Same class as the
    ``aot.append(...)`` bug fixed in #108.
    """
    src = tomlrt.loads(
        "[[outer]]\nv = 1\n[outer.sub]\ns = 2\n"
        "[[outer.aot]]\nq = 3\n[[outer.aot]]\nq = 4\n",
    )
    dst = tomlrt.loads("[[outer]]\nv = 99\n")
    dst["outer"][0] = src["outer"][0]
    out = dst.render()
    reparsed = _reparses(out)
    assert reparsed["outer"][0] == {
        "v": 1,
        "sub": {"s": 2},
        "aot": [{"q": 3}, {"q": 4}],
    }


def test_cross_doc_assign_repeats_subsection_under_distinct_aot_entries() -> None:
    """Repeated ``[a.x.sub]`` under separate ``[[a.x]]`` entries must
    materialise as distinct view containers, not share one.

    Each ``[[a.x]]`` opens a fresh entry, and the following ``[a.x.sub]``
    belongs to that entry only. Without per-entry container scoping, a
    cross-doc clone would conflate the two ``sub`` containers — the
    first entry's ``sub`` would end up holding the second's value, and
    the second entry would lose its ``sub`` key entirely.
    """
    src = tomlrt.loads(
        "[a]\n[[a.x]]\ny = 1\n[a.x.sub]\nz = 1\n[[a.x]]\ny = 2\n[a.x.sub]\nz = 2\n",
    )
    dst = Document()
    dst["a"] = src["a"]
    entries = list(dst["a"]["x"])
    assert dict(entries[0]) == {"y": 1, "sub": {"z": 1}}
    assert dict(entries[1]) == {"y": 2, "sub": {"z": 2}}
    dst["a"]["x"][0]["sub"]["z"] = 99
    assert "z = 99" in dst.render()
    assert "z = 2" in dst.render()


def test_cross_doc_table_assign_preserves_header_leading_comments() -> None:
    """Cross-doc ``dst[k] = src[k]`` preserves the source header's
    leading comment block (and its EOL comment).

    The clone path rewrites the structural prefix of the head's
    leading trivia (so the destination doc's spacing convention wins)
    but keeps any comment pieces — they belong to the section being
    copied.
    """
    src = tomlrt.loads("# bee\n[b]  # eol\nval = 2\n")
    dst = Document()
    dst["b"] = src["b"]
    table = dst.get_table("b")
    assert table is not None
    assert table.header_leading_comments == ("bee",)
    assert table.header_comment == "eol"
    out = dst.render()
    assert "# bee" in out
    assert "# eol" in out


def test_cross_doc_implicit_parent_preserves_child_header_comments() -> None:
    """Issue #117: when source parent is implicit, child sub-tables'
    ``header_leading_comments`` / ``header_comment`` must survive the move.
    """
    src = tomlrt.loads("# alpha header\n[servers.alpha]  # eol\nx = 1\n")
    dst = Document()
    dst["servers"] = src["servers"]
    servers = dst.get_table("servers")
    assert servers is not None
    alpha = servers.get_table("alpha")
    assert alpha is not None
    assert alpha.header_leading_comments == ("alpha header",)
    assert alpha.header_comment == "eol"


def test_install_multi_component_attached_section_preserves_comments() -> None:
    """``install("a.b", attached_section)`` must clone the source's CST
    trivia rather than synthesise a fresh ``[a.b]`` header.

    Same root cause as #117: the multi-component install path was routing
    attached header-bearing sections through the synthesis path
    (``attach_section_at``) instead of the clone path
    (``clone_section_as_section``).
    """
    src = tomlrt.loads("# c\n[x]  # eol\ny = 1\n")
    dst = Document()
    dst.install("a.b", src["x"])
    a = dst.get_table("a")
    assert a is not None
    b = a.get_table("b")
    assert b is not None
    assert b.header_leading_comments == ("c",)
    assert b.header_comment == "eol"


def test_cross_doc_table_assign_preserves_comments() -> None:
    """Cross-doc copy of a section preserves its comments and layout."""
    src = td("""
        # top comment
        [srv]
        # inner
        host = "a.example"
        port = 80
        """)
    a = tomlrt.loads(src)
    b = Document()
    b["srv"] = a["srv"]
    out = tomlrt.dumps(b)
    assert "# inner" in out
    assert 'host = "a.example"' in out


def test_cross_doc_assign_whole_document() -> None:
    """Assigning a whole ``Document`` as a value snapshots its full content.

    Exercises the ``splen == 0`` branch in ``_clone_table_sections``: the
    source's implicit pre-header entries must survive as dotted KVs under
    the new host section, and any ``[X]`` / ``[[X]]`` sections must be
    re-rooted under the destination key.
    """
    src = td("""
        top = 1
        lit = "x"
        [s]
        x = 1
        [[a]]
        n = 1
        """)
    a = tomlrt.loads(src)
    b = Document()
    b["wrap"] = a
    out = tomlrt.dumps(b)
    assert _reparses(out) == {
        "wrap": {"top": 1, "lit": "x", "s": {"x": 1}, "a": [{"n": 1}]},
    }
    # Nested AoT survives as `[[..]]`, not flattened.
    assert "[[wrap.a]]" in out


def test_cross_doc_table_assign_dotted_kv_only_source() -> None:
    """Source table backed solely by ancestor dotted KVs (no own header).

    Exercises the ``host is None`` branch in ``_clone_table_sections``:
    the source's contents live entirely as dotted KVs under an ancestor
    section, so the cloned block has to synthesise a host section.
    """
    src = td("""
        [a]
        b.c = 1
        b.d = 2
        """)
    a = tomlrt.loads(src)
    b = Document()
    inner = a["a"].table("b")
    b["x"] = inner
    out = tomlrt.dumps(b)
    assert _reparses(out) == {"x": {"c": 1, "d": 2}}


def test_cross_doc_table_assign_merges_dotted_and_own_section() -> None:
    """Source has both a pre-header section and an own header at full_path.

    Exercises the branch where ``host`` is a cloned own-section (rather
    than a freshly synthesised one) into which extras must be merged.
    Achievable via the ``Document``-as-value path: the implicit
    pre-header entries become extras, while a top-level ``[k]`` section
    in the source clones to the host at the destination's ``[k]``.
    """
    src = td("""
        pre = 1
        [k]
        x = 2
        """)
    a = tomlrt.loads(src)
    b = Document()
    b["k"] = a
    out = tomlrt.dumps(b)
    assert _reparses(out) == {"k": {"pre": 1, "k": {"x": 2}}}


def test_self_overlap_assign_replaces_with_child_block() -> None:
    """``doc[k] = doc[k]["child"]`` lifts the child to a ``[k]`` block."""
    doc = tomlrt.loads(
        td("""
        [a]
        x = 1
        [a.b]
        y = 2
        [[a.b.list]]
        n = 1
        """)
    )
    doc["a"] = doc["a"]["b"]
    out = tomlrt.dumps(doc)
    assert _reparses(out) == {"a": {"y": 2, "list": [{"n": 1}]}}
    assert "[[a.list]]" in out
    # And the simple (no-AoT) variant stays a section, not an inline table.
    doc2 = tomlrt.loads(
        td("""
        [a]
        x = 1
        [a.b]
        y = 2
        """)
    )
    doc2["a"] = doc2["a"]["b"]
    assert tomlrt.dumps(doc2).startswith("[a]\n")


def test_cross_doc_splice_no_doubled_blank_lines() -> None:
    """Sequential cross-doc copies don't double the blank line between sections.

    Cloned sections retain their original leading blank-line trivia;
    ``_insert_section_block`` must avoid prepending another one.
    """
    src = tomlrt.loads(
        td("""
        [a]
        x = 1

        [b]
        y = 2
        """)
    )
    dst = Document()
    for k, v in src.items():
        dst[k] = v
    assert tomlrt.dumps(dst) == td("""
        [a]
        x = 1

        [b]
        y = 2
        """)


def test_delete_first_section_strips_top_blank() -> None:
    """``del doc[k]`` (where ``[k]`` was first) doesn't leave a stray blank.

    The successor section's leading blank-line trivia was a separator
    from the now-removed first section; after removal it must not show
    up as a top-of-file blank line.
    """
    doc = tomlrt.loads(
        td("""
        [a]
        x = 1

        [b]
        y = 2
        """)
    )
    del doc["a"]
    assert tomlrt.dumps(doc) == "[b]\ny = 2\n"
    # Deleting a middle section preserves separation between survivors.
    doc2 = tomlrt.loads(
        td("""
        [a]
        x = 1

        [b]
        y = 2

        [c]
        z = 3
        """)
    )
    del doc2["b"]
    assert tomlrt.dumps(doc2) == td("""
        [a]
        x = 1

        [c]
        z = 3
        """)


def test_delete_first_aot_entry_strips_top_blank() -> None:
    """Removing the first ``[[t]]`` entry must not leave a stray top blank.

    Same shape as :func:`test_delete_first_section_strips_top_blank`,
    but driven through the AoT mutation API. The successor entry's
    header carries a blank-line separator from its previous neighbour;
    once that neighbour is gone the blank renders as a top-of-file
    artefact unless the removal path normalises it.
    """
    src = td("""
        [[items]]
        n = 1

        [[items]]
        n = 2
        """)
    expected = "[[items]]\nn = 2\n"

    doc = tomlrt.loads(src)
    del doc.aot("items")[0]
    assert tomlrt.dumps(doc) == expected

    doc = tomlrt.loads(src)
    doc.aot("items").pop(0)
    assert tomlrt.dumps(doc) == expected

    doc = tomlrt.loads(src)
    aot = doc.aot("items")
    aot.remove(aot[0])
    assert tomlrt.dumps(doc) == expected

    doc = tomlrt.loads(src)
    del doc.aot("items")[:1]
    assert tomlrt.dumps(doc) == expected

    # Owned sub-sections of the popped entry are removed too, and the
    # next entry — now first in the document — must still render
    # flush against the top.
    doc = tomlrt.loads(
        td("""
            [[items]]
            n = 1
            [items.sub]
            v = 1

            [[items]]
            n = 2
            """),
    )
    doc.aot("items").pop(0)
    assert tomlrt.dumps(doc) == "[[items]]\nn = 2\n"


def test_delete_first_top_level_kv_strips_top_blank() -> None:
    """``del doc[k]`` (k the first top-level KV) doesn't leave a stray blank.

    The successor entry's leading blank-line trivia was a separator
    from the now-removed first KV; after removal it must not show up
    as a top-of-file blank line.
    """
    doc = tomlrt.loads(
        td("""
        x = 1

        y = 2
        """)
    )
    del doc["x"]
    assert tomlrt.dumps(doc) == "y = 2\n"

    # Same when the survivor is a section, not a KV.
    doc = tomlrt.loads(
        td("""
        x = 1

        [a]
        y = 2
        """)
    )
    del doc["x"]
    assert tomlrt.dumps(doc) == "[a]\ny = 2\n"

    # Same via Table.pop.
    doc = tomlrt.loads(
        td("""
        x = 1

        y = 2
        """)
    )
    doc.pop("x")
    assert tomlrt.dumps(doc) == "y = 2\n"


def test_aot_imul_inserts_blank_separator_when_no_sibling_to_sample() -> None:
    """``aot *= n`` on a single-entry AoT must blank-separate the copies.

    With one block there is no inter-entry separator to copy, so the
    repeat path used to fall back to empty trivia, gluing the new
    headers directly under the original (``[[t]]\\n[[t]]\\n``). The
    canonical-style fallback inserts a blank line between repetitions.
    """
    doc = tomlrt.loads("[[t]]\nn = 1\n")
    doc.aot("t").__imul__(2)
    assert tomlrt.dumps(doc) == td("""
        [[t]]
        n = 1

        [[t]]
        n = 1
        """)

    doc = tomlrt.loads("[[t]]\nn = 1\n")
    doc.aot("t").__imul__(3)
    assert tomlrt.dumps(doc) == (
        td("""
        [[t]]
        n = 1

        [[t]]
        n = 1

        [[t]]
        n = 1
        """)
    )


def test_install_through_aot_rejects_cleanly() -> None:
    """``install`` rejects a path that threads through an AoT, untouched.

    AoT entries don't have a single addressable child container, so a
    multi-segment install whose intermediate is ``[[t]]`` has no
    well-defined target. Reject up-front with a clear ``TOMLError``
    rather than letting downstream code trip an ``AssertionError``
    after partially mutating the document.
    """
    src = "[[t]]\nn = 1\n"
    doc = tomlrt.loads(src)
    with pytest.raises(tomlrt.TOMLError, match="array-of-tables"):
        doc.install(("t", "sub"), Table.section({"k": 1}))
    # Document must be unchanged after the rejected install.
    assert tomlrt.dumps(doc) == src

    # Single-segment install at the AoT key still replaces it.
    doc = tomlrt.loads(src)
    doc.install(("t",), Table.section({"k": 1}))
    assert tomlrt.dumps(doc) == "[t]\nk = 1\n"


def test_chained_supertable_assignment_drops_empty_parent() -> None:
    """``doc[t] = Table.section({}); doc[t][c] = ...`` doesn't leave ``[t]``.

    A synthesised empty parent header is redundant once a child section
    gives it a ``[t.c]`` sibling — the parent table is implied by the
    dotted child key. Mirrors the long-standing behaviour of
    ``Document.install(("t", "c"), Table.section({}))``.
    """
    doc = Document()
    doc["tool"] = Table.section({})
    doc["tool"]["poetry"] = Table.section({"name": "foo"})
    assert tomlrt.dumps(doc) == '[tool.poetry]\nname = "foo"\n'

    # Same behaviour with an AoT child.
    doc2 = Document()
    doc2["tool"] = Table.section({})
    doc2["tool"]["list"] = AoT([{"n": 1}])
    assert tomlrt.dumps(doc2) == "[[tool.list]]\nn = 1\n"

    # Non-empty parent must be preserved.
    doc3 = Document()
    doc3["tool"] = Table.section({"extra": 1})
    doc3["tool"]["poetry"] = Table.section({"name": "foo"})
    assert tomlrt.dumps(doc3) == td("""
        [tool]
        extra = 1

        [tool.poetry]
        name = "foo"
        """)

    # Parser-authored empty header must be preserved.
    doc4 = tomlrt.loads("[product]\n")
    doc4.table("product")["variant"] = AoT([{"sku": "X"}])
    assert tomlrt.dumps(doc4) == td("""
        [product]

        [[product.variant]]
        sku = "X"
        """)

    # An empty parent installed alone (no child) stays as the user wrote it.
    doc5 = Document()
    doc5["tool"] = Table.section({})
    assert tomlrt.dumps(doc5) == "[tool]\n"


def test_subsection_under_non_last_aot_entry_lands_in_owned_range() -> None:
    """``aot[i][k] = Table.section(...)`` lands inside entry ``i``'s range,
    not after the last sibling sharing the parent prefix."""
    doc = Document()
    doc["package"] = AoT([{"n": "a"}, {"n": "b"}, {"n": "c"}])
    doc["package"][0]["source"] = Table.section({"x": 1})
    doc["package"][1]["source"] = Table.section({"y": 2})
    expected = (
        '[[package]]\nn = "a"\n\n'
        "[package.source]\nx = 1\n\n"
        '[[package]]\nn = "b"\n\n'
        "[package.source]\ny = 2\n\n"
        '[[package]]\nn = "c"\n'
    )
    assert tomlrt.dumps(doc) == expected
    # Round-trip preserves the per-entry attribution.
    parsed = tomlrt.loads(tomlrt.dumps(doc))
    assert parsed["package"][0]["source"]["x"] == 1
    assert parsed["package"][1]["source"]["y"] == 2
    assert "source" not in parsed["package"][2]


def test_aot_constructor_preserves_nested_section_specs() -> None:
    """``AoT([Table.section({..., k: Table.section({...})})])`` must
    install the nested section as ``[path.k]`` under the entry, not
    silently inline it via the synthesiser."""
    doc = Document()
    doc["package"] = AoT(
        [
            Table.section(
                {"name": "A", "dependencies": Table.section({"b": "*"})},
            ),
            Table.section(
                {"name": "B", "dependencies": Table.section({"c": "*"})},
            ),
        ],
    )
    rendered = tomlrt.dumps(doc)
    expected = (
        '[[package]]\nname = "A"\n\n'
        '[package.dependencies]\nb = "*"\n\n'
        '[[package]]\nname = "B"\n\n'
        '[package.dependencies]\nc = "*"\n'
    )
    assert rendered == expected
    assert tomlrt.loads(rendered).render() == rendered


def test_aot_constructor_preserves_nested_aot() -> None:
    doc = Document()
    doc["pkg"] = AoT(
        [
            Table.section({"name": "A", "tags": AoT([{"k": 1}, {"k": 2}])}),
        ],
    )
    rendered = tomlrt.dumps(doc)
    assert "tags = [" not in rendered
    assert rendered.count("[[pkg.tags]]") == 2
    assert tomlrt.loads(rendered).render() == rendered


def test_ior_on_subscripted_table_preserves_position() -> None:
    """``doc[k] |= other`` keeps ``[k]``'s position in the document.

    Python compiles augmented assignment to a subscripted target as
    ``tmp = doc[k]; tmp.__ior__(other); doc[k] = tmp`` — the third
    step rebinds even though ``tmp`` is the same object already at
    ``doc[k]``. The default ``Table.__setitem__`` would detach the
    "old" value (which is also the "new" value, so it moves the CST
    sections into an orphan doc) and then re-clone them back via
    ``_install_attached_table``, losing the original position and
    surrounding blank-line trivia. The early-return for ``old is
    value`` short-circuits that round-trip.
    """
    doc = tomlrt.loads(
        td("""
            [tool.black]
            line-length = 88

            [other]
            x = 1
            """),
    )
    addition = tomlrt.loads('[tool.poetry]\nname = "foo"\n')
    doc["tool"] |= addition["tool"]

    assert tomlrt.dumps(doc) == (
        "[tool.black]\nline-length = 88\n\n"
        '[tool.poetry]\nname = "foo"\n\n'
        "[other]\nx = 1\n"
    )


def test_self_assignment_is_a_noop() -> None:
    """``doc[k] = doc[k]`` does not mutate the document or detach the view.

    Plain Python dict semantics: re-binding a key to its own current
    value is a no-op.
    """
    doc = tomlrt.loads(
        td("""
        [t]
        a = 1
        [u]
        b = 2
        """)
    )
    t = doc["t"]
    before = tomlrt.dumps(doc)
    doc["t"] = doc["t"]
    assert tomlrt.dumps(doc) == before
    # Held reference still tracks live state.
    t["c"] = 3
    assert "c" in doc["t"]


def test_section_replace_preserves_position() -> None:
    """``doc[k] = Table.section({...})`` keeps ``[k]`` where it was.

    Replacing an existing section used to purge the old block then
    splice the new one after the last sibling sharing the parent
    prefix, which moved the section to the end of the document.
    The slot lookup now remembers the position of the first matching
    section before purge and reuses that index.
    """
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
    doc["b"] = Table.section({"q": 9})
    assert tomlrt.dumps(doc) == td("""
        [a]
        x = 1

        [b]
        q = 9

        [c]
        z = 3
        """)


def test_section_replace_preserves_position_for_implicit_parent() -> None:
    """Replacing an implicit super-table key preserves the subtree's slot.

    ``[b.c]`` exists with no explicit ``[b]`` header. Assigning a
    ``Table.section`` to ``b`` purges the implicit subtree and lands a
    fresh ``[b]`` block where ``[b.c]`` used to live.
    """
    doc = tomlrt.loads(
        td("""
        [a]
        x = 1

        [b.c]
        y = 2

        [d]
        z = 3
        """)
    )
    doc["b"] = Table.section({"q": 9})
    assert tomlrt.dumps(doc) == td("""
        [a]
        x = 1

        [b]
        q = 9

        [d]
        z = 3
        """)


def test_aot_assign_purges_implicit_supertable_aot_subtree() -> None:
    """Assigning an AoT to a key that names an implicit super-table
    over an existing ``[[a.b]]`` block must purge the old subtree.
    """
    src = tomlrt.loads(
        td("""
        [[a.b]]
        x = 1
        [[a.b]]
        x = 2
        """)
    )
    src["a"] = src.aot("a.b")
    assert tomlrt.dumps(src) == td("""
        [[a]]
        x = 1
        [[a]]
        x = 2
        """)
    assert src.to_dict() == {"a": [{"x": 1}, {"x": 2}]}

    dst = tomlrt.loads("[[a.b]]\nx = 1\n")
    other = tomlrt.loads(
        td("""
        [[t]]
        z = 99
        [[t]]
        z = 100
        """)
    )
    dst["a"] = other.aot("t")
    assert tomlrt.dumps(dst) == td("""
        [[a]]
        z = 99
        [[a]]
        z = 100
        """)
    assert dst.to_dict() == {"a": [{"z": 99}, {"z": 100}]}


def test_install_aot_same_doc_overlap_preserves_source() -> None:
    """``doc.install(k, doc.aot(k.sub))`` must not silently lose data.

    ``__setitem__`` snapshots an in-cache ``old`` value via ``_detach``
    before any structural mutation, which incidentally protected the
    same-document overlap case. ``install`` skips that path and called
    ``_install_attached_aot``, which used to clone from the source
    *after* purging the destination slot — so the source sections were
    already gone by the time we tried to copy them, leaving an empty
    result. Aligning the AoT install ordering with the table install
    (clone-before-purge) closes the gap.
    """
    src = tomlrt.loads(
        td("""
        [[a.inner]]
        x = 1
        [[a.inner]]
        x = 2
        """)
    )
    src.install("a", src.aot("a.inner"))
    assert tomlrt.dumps(src) == td("""
        [[a]]
        x = 1
        [[a]]
        x = 2
        """)
    assert src.to_dict() == {"a": [{"x": 1}, {"x": 2}]}


def test_replacing_value_detaches_old_view_for_all_setters() -> None:
    """User-held references to a replaced view must be cleanly detached
    so writes through them stop reaching the document.

    ``__setitem__`` only detached the old value on the *non-flavoured*
    path. Assigning a ``Table.section(...)`` (or a standalone ``AoT`` /
    ``Array``) skipped the detach, so the old view kept writing into
    the document under its old path. ``Document.install`` skipped the
    detach entirely. Both now detach before installing.
    """
    # __setitem__ with a Table.section
    doc = tomlrt.loads("[a]\nx = 1\n")
    old = doc["a"]
    doc["a"] = Table.section({"z": 99})
    old["w"] = 777
    assert tomlrt.dumps(doc) == "[a]\nz = 99\n"
    assert old["w"] == 777

    # Document.install at top level
    doc = tomlrt.loads("[a]\nx = 1\n")
    old = doc["a"]
    doc.install("a", Table.section({"z": 99}))
    old["w"] = 777
    assert tomlrt.dumps(doc) == "[a]\nz = 99\n"

    # Document.install at a dotted path
    doc = tomlrt.loads("[a.b]\nx = 1\n")
    old = doc["a"]["b"]
    doc.install(("a", "b"), Table.section({"q": 9}))
    old["w"] = 555
    assert tomlrt.dumps(doc) == "[a.b]\nq = 9\n"


def test_aot_entry_subsection_replace_preserves_position() -> None:
    """``aot[i][k] = Table.section({...})`` keeps the sub-section in place.

    Inside an AoT entry the slot lookup must be scoped to that entry
    so a sibling entry's same-named sub-section is not mistaken for a
    prior — and the new block must land where the *entry's own* prior
    sub-section sat, not at the end of the entry's owned range.
    """
    doc = tomlrt.loads(
        td("""
            [[pkg]]
            n = 1

            [pkg.a]
            x = 1

            [pkg.b]
            y = 2

            [pkg.c]
            z = 3
            """),
    )
    doc["pkg"][0]["b"] = Table.section({"q": 9})
    assert tomlrt.dumps(doc) == (
        td("""
            [[pkg]]
            n = 1

            [pkg.a]
            x = 1

            [pkg.b]
            q = 9

            [pkg.c]
            z = 3
            """)
    )


def test_section_subkey_across_aot_entries_keeps_values_separate() -> None:
    """Setting the *same* sub-section key on multiple AoT entries does not
    leak values across entries.

    The freshly inserted ``[aot.k]`` view used to be created without an
    ``owner_anchor``, so its scope spanned the whole document. When two
    entries set the same sub-key, scalar writes through the second view
    found the first entry's section as a "direct" hit and silently
    overwrote it — corrupting earlier entries and leaving the later
    [aot.k] section partly empty.
    """
    doc = Document()
    doc["package"] = AoT(
        [{"n": "git1"}, {"n": "git2"}, {"n": "url1"}, {"n": "url2"}],
    )
    doc["package"][0]["source"] = Table.section(
        {"type": "git", "url": "g1", "ref": "develop"},
    )
    doc["package"][1]["source"] = Table.section(
        {"type": "git", "url": "g2", "subdir": "s"},
    )
    doc["package"][2]["source"] = Table.section({"type": "url", "url": "u1"})
    doc["package"][3]["source"] = Table.section({"type": "url", "url": "u2"})

    expected_sources = [
        {"type": "git", "url": "g1", "ref": "develop"},
        {"type": "git", "url": "g2", "subdir": "s"},
        {"type": "url", "url": "u1"},
        {"type": "url", "url": "u2"},
    ]
    for i, want in enumerate(expected_sources):
        assert dict(doc["package"][i]["source"]) == want

    # Round-trip parses the same way: each [package.source] stays
    # attached to its own [[package]] entry.
    parsed = tomlrt.loads(tomlrt.dumps(doc))
    for i, want in enumerate(expected_sources):
        assert dict(parsed["package"][i]["source"]) == want


def test_section_subkey_across_identical_aot_entries() -> None:
    """Adjacent AoT entries with identical content keep their sub-sections
    distinct.

    ``_prepare_section_slot`` used ``list.index`` (``==``) to locate the
    owning ``[[..]]`` anchor inside ``self._doc_node.sections``. When
    two siblings had identical entries, that returned the first
    matching position for both, so installing a sub-section under the
    later sibling spliced an empty placeholder into the *earlier*
    sibling's range and put the new content next to its existing
    ``[..source]`` — corrupting both on round-trip.
    """
    doc = Document()
    doc["package"] = AoT([{"n": "a"}, {"n": "b"}, {"n": "b"}])
    doc["package"][0]["source"] = Table.section({"x": "prev"})
    doc["package"][1]["source"] = Table.section({"x": "c"})
    doc["package"][2]["source"] = Table.section({"x": "d"})

    expected = (
        '[[package]]\nn = "a"\n\n'
        '[package.source]\nx = "prev"\n\n'
        '[[package]]\nn = "b"\n\n'
        '[package.source]\nx = "c"\n\n'
        '[[package]]\nn = "b"\n\n'
        '[package.source]\nx = "d"\n'
    )
    assert tomlrt.dumps(doc) == expected

    parsed = tomlrt.loads(tomlrt.dumps(doc))
    assert [dict(p["source"]) for p in parsed["package"]] == [
        {"x": "prev"},
        {"x": "c"},
        {"x": "d"},
    ]


def test_del_after_emptying_descendant_succeeds() -> None:
    """A cached implicit-table view stays deletable after its only descendant
    is removed.

    Holding ``bar = group['bar']`` and then ``del bar['dependencies']``
    leaves ``bar`` reachable through ``group`` as an empty table — same
    as plain Python dict semantics — and ``del group['bar']`` (or
    ``group.pop('bar')``) must succeed rather than raising ``KeyError``
    because the underlying CST chain is already gone.
    """
    doc = tomlrt.loads('[tool.poetry.group.bar.dependencies]\nfoo = "1"\n')
    bar = doc["tool"]["poetry"]["group"]["bar"]
    del bar["dependencies"]
    group = doc["tool"]["poetry"]["group"]
    assert "bar" in group
    assert dict(group["bar"]) == {}
    del group["bar"]
    assert "bar" not in group
    assert tomlrt.dumps(doc) == ""

    # ``pop`` is the same code path; verify it too returns the empty view.
    doc2 = tomlrt.loads('[tool.poetry.group.bar.dependencies]\nfoo = "1"\n')
    bar2 = doc2["tool"]["poetry"]["group"]["bar"]
    del bar2["dependencies"]
    group2 = doc2["tool"]["poetry"]["group"]
    popped = group2.pop("bar")
    assert dict(popped) == {}
    assert "bar" not in group2

    # A genuinely-absent key still raises, exactly as a plain dict would.
    with pytest.raises(KeyError):
        del group["nope"]


def test_install_attached_aot_preserves_comments() -> None:
    # `install` and `__setitem__` should both deep-clone the source CST
    # when given an attached AoT from another document. The previous
    # `install` implementation always routed through `to_dict()`, which
    # silently stripped comments and formatting, diverging from the
    # subscript path.
    src = td("""
        [[t]]
        # leading
        a = 1  # eol
        [[t]]
        b = 2
        """)
    a = tomlrt.loads(src)
    b = tomlrt.loads("")
    b.install("y", a["t"])
    assert tomlrt.dumps(b) == (
        td("""
        [[y]]
        # leading
        a = 1  # eol
        [[y]]
        b = 2
        """)
    )


def test_install_attached_aot_at_dotted_path_preserves_comments() -> None:
    src = td("""
        [[t]]
        # leading
        a = 1
        """)
    a = tomlrt.loads(src)
    b = tomlrt.loads("")
    b.install("p.q", a["t"])
    assert tomlrt.dumps(b) == td("""
        [[p.q]]
        # leading
        a = 1
        """)


def test_install_attached_aot_is_independent_of_source() -> None:
    src = '[[t]]\nname = "alice"\n'
    a = tomlrt.loads(src)
    b = tomlrt.loads("")
    b.install("y", a["t"])
    a["t"][0]["name"] = "MUT"
    assert _reparses(tomlrt.dumps(b))["y"][0]["name"] == "alice"


# ---------------------------------------------------------------------------
# Cross-section conflict on mutation
# ---------------------------------------------------------------------------


def test_set_value_overwriting_existing_subsection() -> None:
    """Assigning a scalar to a name that's currently a sub-table.

    Matches plain-dict semantics: the [a.b] section (and anything nested
    under it) is silently removed and replaced with ``b = 99`` inside
    ``[a]``.
    """
    src = td("""
        [a]
        x = 1
        [a.b]
        y = 2
        [a.b.c]
        z = 3
        """)
    doc = tomlrt.loads(src)
    a = doc.table("a")
    a["b"] = 99
    out = tomlrt.dumps(doc)
    assert out == td("""
        [a]
        x = 1
        b = 99
        """)
    assert _reparses(out) == {"a": {"x": 1, "b": 99}}


def test_set_value_overwriting_existing_aot() -> None:
    src = td("""
        [a]
        x = 1
        [[a.items]]
        name = "first"
        [[a.items]]
        name = "second"
        """)
    doc = tomlrt.loads(src)
    a = doc.table("a")
    a["items"] = 5
    out = tomlrt.dumps(doc)
    assert out == td("""
        [a]
        x = 1
        items = 5
        """)
    assert _reparses(out) == {"a": {"x": 1, "items": 5}}


def test_set_value_overwriting_dotted_subtree() -> None:
    src = td("""
        [a]
        b.c = 1
        b.d = 2
        x = 9
        """)
    doc = tomlrt.loads(src)
    a = doc.table("a")
    a["b"] = 99
    out = tomlrt.dumps(doc)
    assert _reparses(out) == {"a": {"x": 9, "b": 99}}


def test_set_value_overwriting_top_level_table() -> None:
    src = td("""
        [a]
        x = 1
        [b]
        y = 2
        """)
    doc = tomlrt.loads(src)
    doc["a"] = 99
    out = tomlrt.dumps(doc)
    assert _reparses(out) == {"a": 99, "b": {"y": 2}}


def test_del_subtable() -> None:
    src = td("""
        [a]
        x = 1
        [a.b]
        y = 2
        [a.b.c]
        z = 3
        [other]
        q = 1
        """)
    doc = tomlrt.loads(src)
    a = doc.table("a")
    del a["b"]
    out = tomlrt.dumps(doc)
    assert _reparses(out) == {"a": {"x": 1}, "other": {"q": 1}}


def test_del_aot() -> None:
    src = td("""
        [a]
        x = 1
        [[a.items]]
        name = "first"
        [[a.items]]
        name = "second"
        """)
    doc = tomlrt.loads(src)
    a = doc.table("a")
    del a["items"]
    out = tomlrt.dumps(doc)
    assert _reparses(out) == {"a": {"x": 1}}


def test_del_dotted_subtree() -> None:
    src = td("""
        [a]
        b.c = 1
        b.d = 2
        x = 9
        """)
    doc = tomlrt.loads(src)
    a = doc.table("a")
    del a["b"]
    out = tomlrt.dumps(doc)
    assert _reparses(out) == {"a": {"x": 9}}


def test_del_missing_raises_keyerror() -> None:
    doc = tomlrt.loads("[a]\nx = 1\n")
    a = doc.table("a")
    with pytest.raises(KeyError):
        del a["missing"]


def test_pop_returns_subtable_snapshot() -> None:
    src = td("""
        [a]
        x = 1
        [a.b]
        y = 2
        [a.b.c]
        z = 3
        """)
    doc = tomlrt.loads(src)
    a = doc.table("a")
    popped = a.pop("b")
    assert popped == {"y": 2, "c": {"z": 3}}
    assert _reparses(tomlrt.dumps(doc)) == {"a": {"x": 1}}


def test_pop_returns_aot_snapshot() -> None:
    doc = tomlrt.loads(
        td("""
        [[items]]
        name = "a"
        [[items]]
        name = "b"
        """)
    )
    popped = doc.pop("items")
    assert popped == [{"name": "a"}, {"name": "b"}]
    assert tomlrt.dumps(doc) == ""


def test_pop_with_default() -> None:
    doc = tomlrt.loads("")
    assert doc.pop("missing", "fallback") == "fallback"
    with pytest.raises(KeyError):
        doc.pop("missing")


def test_popitem_is_lifo() -> None:
    doc = tomlrt.loads(
        td("""
        a = 1
        b = 2
        c = 3
        """)
    )
    assert doc.popitem() == ("c", 3)
    assert doc.popitem() == ("b", 2)
    assert _reparses(tomlrt.dumps(doc)) == {"a": 1}


def test_popitem_empty_raises() -> None:
    doc = tomlrt.loads("")
    with pytest.raises(KeyError):
        doc.popitem()


def test_setitem_into_implicit_parent() -> None:
    """Adding a new key to an implicit-only parent materialises [a]."""
    src = "[a.b]\ny = 2\n"
    doc = tomlrt.loads(src)
    a = doc.table("a")
    a["new"] = 1
    out = tomlrt.dumps(doc)
    assert _reparses(out) == {"a": {"new": 1, "b": {"y": 2}}}


def test_setitem_into_implicit_grandparent() -> None:
    src = "[a.b.c]\nz = 3\n"
    doc = tomlrt.loads(src)
    ab = doc.table("a").table("b")
    ab["new"] = 1
    out = tomlrt.dumps(doc)
    assert _reparses(out) == {"a": {"b": {"new": 1, "c": {"z": 3}}}}


def test_inline_table_setitem_overwrites_dotted_group() -> None:
    src = 'config = { server.host = "x", server.port = 80, name = "y" }\n'
    doc = tomlrt.loads(src)
    config = doc.table("config")
    config["server"] = "newval"
    out = tomlrt.dumps(doc)
    assert _reparses(out) == {"config": {"name": "y", "server": "newval"}}


def test_inline_table_delitem_removes_dotted_group() -> None:
    src = 'config = { server.host = "x", server.port = 80, name = "y" }\n'
    doc = tomlrt.loads(src)
    config = doc.table("config")
    del config["server"]
    out = tomlrt.dumps(doc)
    assert _reparses(out) == {"config": {"name": "y"}}


def test_inline_table_delitem_missing_raises_keyerror() -> None:
    doc = tomlrt.loads("config = { a = 1 }\n")
    config = doc.table("config")
    with pytest.raises(KeyError):
        del config["missing"]


# ---------------------------------------------------------------------------
# Sub-table access through AoT entry (uses the new owned_scope path)
# ---------------------------------------------------------------------------


def test_aot_entry_owned_scope_isolates_sibling_sub_sections() -> None:
    """[[arr]] / [arr.sub] / x=1 / [[arr]] / [arr.sub] / x=2

    Each entry's sub.x must be independent; mutating arr[0].sub.x must
    not affect arr[1].sub.x.
    """
    src = td("""
        [[arr]]
        [arr.sub]
        x = 1
        [[arr]]
        [arr.sub]
        x = 2
        """)
    doc = tomlrt.loads(src)
    arr = doc.aot("arr")
    s0 = arr[0].table("sub")
    s1 = arr[1].table("sub")
    assert s0["x"] == 1
    assert s1["x"] == 2
    s0["x"] = 100
    assert s1["x"] == 2  # unchanged
    out = tomlrt.dumps(doc)
    assert _reparses(out) == {"arr": [{"sub": {"x": 100}}, {"sub": {"x": 2}}]}


# ---------------------------------------------------------------------------
# Array.set_multiline / multiline property
# ---------------------------------------------------------------------------


def test_array_set_multiline_true_wraps_with_default_indent() -> None:
    doc = tomlrt.loads("a = [1, 2, 3]\n")
    arr = doc.array("a")
    assert not arr.multiline
    arr.set_multiline(multiline=True)
    assert tomlrt.dumps(doc) == td("""
        a = [
            1,
            2,
            3,
        ]
        """)
    assert arr.multiline


def test_array_set_multiline_false_collapses() -> None:
    doc = tomlrt.loads(
        td("""
        a = [
            1,
            2,
            3,
        ]
        """)
    )
    arr = doc.array("a")
    assert arr.multiline
    arr.set_multiline(multiline=False)
    assert tomlrt.dumps(doc) == "a = [1, 2, 3]\n"
    assert not arr.multiline


def test_array_set_multiline_false_with_eol_comment_raises() -> None:
    doc = tomlrt.loads(
        td("""
        a = [
            1,  # one
            2,
        ]
        """)
    )
    arr = doc.array("a")
    with pytest.raises(tomlrt.TOMLError, match="comments"):
        arr.set_multiline(multiline=False)
    assert tomlrt.dumps(doc) == td("""
        a = [
            1,  # one
            2,
        ]
        """)


def test_array_set_multiline_false_with_leading_comment_raises() -> None:
    doc = tomlrt.loads(
        td("""
        a = [
            # leading
            1,
            2,
        ]
        """)
    )
    arr = doc.array("a")
    with pytest.raises(tomlrt.TOMLError, match="comments"):
        arr.set_multiline(multiline=False)
    assert tomlrt.dumps(doc) == td("""
        a = [
            # leading
            1,
            2,
        ]
        """)


def test_array_set_multiline_false_after_clearing_comments_works() -> None:
    doc = tomlrt.loads(
        td("""
        a = [
            1,  # one
            2,
        ]
        """)
    )
    arr = doc.array("a")
    del arr.comments[0]
    arr.set_multiline(multiline=False)
    assert tomlrt.dumps(doc) == "a = [1, 2]\n"


def test_array_set_multiline_false_refuses_when_inner_inline_has_comment() -> None:
    src = td("""
        a = [
            {
                # inner
                x = 1,
            },
        ]
        """)
    doc = tomlrt.loads(src)
    arr = doc.array("a")
    with pytest.raises(tomlrt.TOMLError, match="EOL or leading comments"):
        arr.set_multiline(multiline=False)
    assert tomlrt.dumps(doc) == src


def test_array_set_multiline_custom_indent() -> None:
    doc = tomlrt.loads("a = [1, 2]\n")
    doc.array("a").set_multiline(multiline=True, indent="  ")
    assert tomlrt.dumps(doc) == td("""
        a = [
          1,
          2,
        ]
        """)


def test_array_set_multiline_preserves_crlf_newlines() -> None:
    doc = tomlrt.loads("a = [1, 2]\r\n")
    doc.array("a").set_multiline(multiline=True, indent="  ")
    out = tomlrt.dumps(doc)
    assert "\n" not in out.replace("\r\n", "")
    assert out == "a = [\r\n  1,\r\n  2,\r\n]\r\n"


def test_array_multiline_property_setter() -> None:
    doc = tomlrt.loads("a = [1, 2]\n")
    arr = doc.array("a")
    arr.multiline = True
    assert tomlrt.dumps(doc) == td("""
        a = [
            1,
            2,
        ]
        """)
    arr.multiline = False
    assert tomlrt.dumps(doc) == "a = [1, 2]\n"


def test_array_set_multiline_then_append() -> None:
    doc = tomlrt.loads("a = [1]\n")
    arr = doc.array("a")
    arr.set_multiline(multiline=True)
    arr.append(2)
    assert tomlrt.dumps(doc) == td("""
        a = [
            1,
            2,
        ]
        """)


def test_array_set_multiline_returns_self() -> None:
    doc = tomlrt.loads("a = [1]\n")
    arr = doc.array("a")
    assert arr.set_multiline(multiline=True) is arr


def test_array_set_multiline_survives_view_refetch() -> None:
    doc = tomlrt.loads("a = []\n")
    doc.array("a").set_multiline(multiline=True)
    refetched = doc.array("a")
    assert refetched.multiline
    refetched.append(1)
    assert tomlrt.dumps(doc) == td("""
        a = [
            1,
        ]
        """)


def test_array_set_multiline_indent_preserved_on_install() -> None:
    # Calling set_multiline(indent=...) on a standalone Array and then
    # installing it should honour the requested indent, not silently
    # revert to the indent passed to the Array constructor.
    arr = tomlrt.Array([1, 2, 3])
    arr.set_multiline(multiline=True, indent="  ")
    doc = Document()
    doc["x"] = arr
    assert tomlrt.dumps(doc) == td("""
        x = [
          1,
          2,
          3,
        ]
        """)


def test_append_to_multiline_array_with_eol_comments() -> None:
    # When every existing item carries an inline comment, the
    # separator-style sampler used to give up and fall back to
    # ", " for the inter-item separator, and to drag the last
    # item's trailing comment into the close-pad. Newly appended
    # items must instead inherit the structural indent and leave
    # the existing comments alone.
    src = td("""
        a = [
            1,  # one
            2,  # two
        ]
        """)
    doc = tomlrt.loads(src)
    doc.array("a").append(3)
    assert tomlrt.dumps(doc) == (
        td("""
        a = [
            1,  # one
            2,  # two
            3,
        ]
        """)
    )


def test_array_parsed_empty_with_newline_is_multiline() -> None:
    doc = tomlrt.loads("a = [\n]\n")
    arr = doc.array("a")
    assert arr.multiline
    arr.append(1)
    assert tomlrt.dumps(doc) == td("""
        a = [
            1,
        ]
        """)


def test_array_parsed_empty_with_newline_indent_is_inferred() -> None:
    doc = tomlrt.loads("a = [\n  ]\n")
    arr = doc.array("a")
    arr.append(1)
    assert tomlrt.dumps(doc) == td("""
        a = [
          1,
        ]
        """)


def test_append_preserves_empty_array_inner_comment() -> None:
    # An empty multiline array with only a comment inside used to lose
    # the comment entirely on first append. The comment should survive
    # as leading trivia of the newly inserted first item.
    src = td("""
        a = [
            # placeholder
        ]
        """)
    doc = tomlrt.loads(src)
    doc.array("a").append(1)
    assert tomlrt.dumps(doc) == td("""
        a = [
            # placeholder
            1,
        ]
        """)


def test_append_preserves_trailing_comment_in_single_item_array() -> None:
    # A single-item multiline array whose last-item post-comma slot
    # carries a comment used to have that comment collapse onto the
    # same line as the new item, producing valid-but-ugly output.
    src = td("""
        a = [
            1,
            # tail
        ]
        """)
    doc = tomlrt.loads(src)
    doc.array("a").append(2)
    assert tomlrt.dumps(doc) == td("""
        a = [
            1,
            # tail
            2,
        ]
        """)


def test_append_preserves_leading_comment_in_single_item_array() -> None:
    # A single-item multiline array with a leading comment used to
    # collapse to single-line layout on append because the inter-item
    # separator could not be sampled from items[:-1] (which is empty).
    src = td("""
        a = [
            # head
            1,
        ]
        """)
    doc = tomlrt.loads(src)
    doc.array("a").append(2)
    assert tomlrt.dumps(doc) == td("""
        a = [
            # head
            1,
            2,
        ]
        """)


# ---------------------------------------------------------------------------
# AoT assignment / Table.promote_array
# ---------------------------------------------------------------------------


def test_assign_aot_creates_repeated_headers() -> None:
    doc = tomlrt.loads("")
    doc["packages"] = AoT([{"name": "a", "version": "1.0"}, {"name": "b"}])
    aot = doc.aot("packages")
    assert len(aot) == 2
    assert "[[packages]]" in tomlrt.dumps(doc)


def test_assign_empty_aot_returns_appendable_view() -> None:
    doc = tomlrt.loads("")
    doc["servers"] = AoT()
    aot = doc["servers"]
    assert tomlrt.dumps(doc) == ""
    aot.append({"host": "localhost"})
    rendered = tomlrt.dumps(doc)
    assert "[[servers]]" in rendered
    assert 'host = "localhost"' in rendered


def test_assign_aot_overwrites_existing_key() -> None:
    doc = tomlrt.loads("foo = 1\n")
    doc["foo"] = AoT([{"x": 1}])
    rendered = tomlrt.dumps(doc)
    assert "foo = 1" not in rendered
    assert "[[foo]]" in rendered


def test_assign_aot_nested_path() -> None:
    doc = tomlrt.loads("[product]\n")
    doc.table("product")["variant"] = AoT([{"sku": "X"}])
    rendered = tomlrt.dumps(doc)
    assert "[[product.variant]]" in rendered
    assert 'sku = "X"' in rendered
    assert isinstance(doc.table("product").aot("variant"), tomlrt.AoT)


def test_assign_aot_blank_separated_entries() -> None:
    doc = tomlrt.loads("")
    doc["p"] = AoT([{"x": 1}, {"x": 2}, {"x": 3}])
    assert tomlrt.dumps(doc) == (
        td("""
        [[p]]
        x = 1

        [[p]]
        x = 2

        [[p]]
        x = 3
        """)
    )


def test_assign_aot_blank_before_first_when_preceded_by_content() -> None:
    doc = tomlrt.loads("top = 1\n")
    doc["p"] = AoT([{"x": 1}])
    assert tomlrt.dumps(doc) == td("""
        top = 1

        [[p]]
        x = 1
        """)


def test_assign_aot_blank_after_section_header() -> None:
    doc = tomlrt.loads("[product]\n")
    doc.table("product")["variant"] = AoT([{"sku": "X"}])
    assert tomlrt.dumps(doc) == (
        td("""
        [product]

        [[product.variant]]
        sku = "X"
        """)
    )


def test_promote_array_converts_inline_to_aot() -> None:
    doc = tomlrt.loads('packages = [{name = "a"}, {name = "b"}]\n')
    aot = doc.promote_array("packages")
    assert isinstance(aot, tomlrt.AoT)
    assert len(aot) == 2
    rendered = tomlrt.dumps(doc)
    assert "packages = [" not in rendered
    assert rendered.count("[[packages]]") == 2
    assert isinstance(doc.aot("packages"), tomlrt.AoT)


def test_promote_array_rejects_empty_array() -> None:
    doc = tomlrt.loads("a = []\n")
    with pytest.raises(tomlrt.TOMLError, match="empty array"):
        doc.promote_array("a")


def test_promote_array_rejects_non_table_elements() -> None:
    doc = tomlrt.loads("a = [1, 2]\n")
    with pytest.raises(tomlrt.TOMLError, match="non-inline-table"):
        doc.promote_array("a")


def test_promote_array_rejects_non_array() -> None:
    doc = tomlrt.loads("a = 1\n")
    with pytest.raises(tomlrt.TOMLError, match="not an array"):
        doc.promote_array("a")


def test_promote_inline_rejects_non_inline_for_present_keys() -> None:
    """When ``key`` is present but isn't a simple inline-table KV, the
    user should see a clear "nothing to promote" message, not a bare
    ``KeyError`` that contradicts ``key in self``.
    """
    for src, target in [
        ("[a]\nb.c = 1\n", "b"),  # dotted-key subtable
        ("[a.b]\nc = 1\n", "b"),  # subsection
        ("[[a.b]]\nc = 1\n", "b"),  # array-of-tables
    ]:
        doc = tomlrt.loads(src)
        assert target in doc["a"]
        with pytest.raises(tomlrt.TOMLError, match="not an inline table"):
            doc["a"].promote_inline(target)


def test_promote_array_rejects_non_array_for_present_keys() -> None:
    for src, target in [
        ("[a]\nb.c = 1\n", "b"),
        ("[a.b]\nc = 1\n", "b"),
        ("[[a.b]]\nc = 1\n", "b"),
    ]:
        doc = tomlrt.loads(src)
        assert target in doc["a"]
        with pytest.raises(tomlrt.TOMLError, match="not an array"):
            doc["a"].promote_array(target)


def test_promote_inline_across_aot_entries() -> None:
    """Promoting the same key in each entry of an AoT must not trip
    the "section already exists" guard with the section just emitted
    inside a sibling entry's block.
    """
    doc = tomlrt.loads(
        '[[package]]\nname = "A"\ndependencies = { b = "*" }\n\n'
        '[[package]]\nname = "B"\ndependencies = { c = "*" }\n',
    )
    for entry in doc.aot("package"):
        entry.promote_inline("dependencies")
    rendered = tomlrt.dumps(doc)
    assert rendered.count("[package.dependencies]") == 2
    assert "dependencies = {" not in rendered
    assert tomlrt.loads(rendered).render() == rendered


def test_promote_array_across_aot_entries() -> None:
    doc = tomlrt.loads(
        '[[package]]\nname = "A"\ntags = [{k = 1}]\n\n'
        '[[package]]\nname = "B"\ntags = [{k = 2}]\n',
    )
    for entry in doc.aot("package"):
        entry.promote_array("tags")
    rendered = tomlrt.dumps(doc)
    assert rendered.count("[[package.tags]]") == 2
    assert "tags = [" not in rendered
    assert tomlrt.loads(rendered).render() == rendered


# ---------------------------------------------------------------------------
# Section assignment / Table.ensure_table / dotted-path navigation
# ---------------------------------------------------------------------------


def test_assign_section_creates_section_directly() -> None:
    doc = tomlrt.loads("")
    doc["tool"] = Table.section({"name": "x"})
    t = doc["tool"]
    assert isinstance(t, tomlrt.Table)
    assert tomlrt.dumps(doc) == '[tool]\nname = "x"\n'


def test_install_section_dotted_omits_super_table_headers() -> None:
    doc = tomlrt.loads("")
    doc.install("tool.poetry", Table.section({"name": "x", "version": "0.1"}))
    rendered = tomlrt.dumps(doc)
    assert "[tool]\n" not in rendered
    assert rendered == td("""
        [tool.poetry]
        name = "x"
        version = "0.1"
        """)


def test_install_section_implicit_super_table_navigable() -> None:
    doc = tomlrt.loads("")
    doc.install("tool.poetry", Table.section({"name": "x"}))
    assert doc.table("tool").table("poetry")["name"] == "x"
    tool = doc.table("tool")
    poetry = tool.table("poetry")
    assert poetry["name"] == "x"
    assert doc.table("tool.poetry")["name"] == "x"


def test_install_section_sibling_does_not_disturb_existing() -> None:
    doc = tomlrt.loads("")
    doc.install("tool.poetry", Table.section({"name": "x"}))
    doc.install("tool.poetry.dependencies", Table.section({"requests": "^2.0"}))
    assert tomlrt.dumps(doc) == (
        td("""
            [tool.poetry]
            name = "x"

            [tool.poetry.dependencies]
            requests = "^2.0"
            """)
    )


def test_install_section_replaces_existing_and_purges_children() -> None:
    doc = tomlrt.loads(
        td("""
        [tool.poetry]
        name = "x"
        [tool.poetry.foo]
        bar = 1
        """)
    )
    doc.install("tool.poetry", Table.section({"version": "2.0"}))
    rendered = tomlrt.dumps(doc)
    assert "name" not in rendered
    assert "[tool.poetry.foo]" not in rendered
    assert rendered == '[tool.poetry]\nversion = "2.0"\n'


def test_install_section_overwrites_inline_value() -> None:
    doc = tomlrt.loads('tool = {poetry = {name = "x"}}\n')
    doc.install("tool.poetry", Table.section({"version": "2.0"}))
    rendered = tomlrt.dumps(doc)
    assert "name" not in rendered
    assert "[tool.poetry]" in rendered
    assert "version" in rendered


def test_install_empty_section() -> None:
    doc = tomlrt.loads("")
    t = doc.install("tool.poetry", Table.section())
    assert tomlrt.dumps(doc) == "[tool.poetry]\n"
    t["name"] = "x"
    assert tomlrt.dumps(doc) == '[tool.poetry]\nname = "x"\n'


def test_ensure_table_creates_when_absent() -> None:
    doc = tomlrt.loads("")
    deps = doc.ensure_table("tool.poetry.dependencies")
    deps["pytest"] = "^7.0"
    assert tomlrt.dumps(doc) == ('[tool.poetry.dependencies]\npytest = "^7.0"\n')


def test_ensure_table_navigates_existing_explicit_section() -> None:
    doc = tomlrt.loads('[tool.poetry]\nname = "x"\n')
    t = doc.ensure_table("tool.poetry")
    t["version"] = "0.1"
    assert tomlrt.dumps(doc) == (
        td("""
        [tool.poetry]
        name = "x"
        version = "0.1"
        """)
    )


def test_ensure_table_navigates_implicit_super_table() -> None:
    doc = tomlrt.loads('[tool.poetry]\nname = "x"\n')
    t = doc.ensure_table("tool")
    assert isinstance(t, tomlrt.Table)
    # No new [tool] header created.
    assert tomlrt.dumps(doc) == '[tool.poetry]\nname = "x"\n'


def test_ensure_table_creates_only_missing_tail() -> None:
    doc = tomlrt.loads('[tool.poetry]\nname = "x"\n')
    t = doc.ensure_table("tool.poetry.dependencies")
    t["requests"] = "^2.0"
    assert tomlrt.dumps(doc) == (
        td("""
            [tool.poetry]
            name = "x"

            [tool.poetry.dependencies]
            requests = "^2.0"
            """)
    )


def test_ensure_table_rejects_non_table_value() -> None:
    doc = tomlrt.loads("tool = 1\n")
    with pytest.raises(tomlrt.TOMLError, match=r"existing value"):
        doc.ensure_table("tool")


def test_install_aot_dotted_path() -> None:
    doc = tomlrt.loads("")
    doc.install(
        "tool.poetry.source",
        AoT(
            [{"name": "pypi"}, {"name": "private"}],
        ),
    )
    rendered = tomlrt.dumps(doc)
    assert "[tool]" not in rendered
    assert "[tool.poetry]" not in rendered
    assert rendered.count("[[tool.poetry.source]]") == 2


def test_install_section_rejects_empty_path() -> None:
    doc = tomlrt.loads("")
    with pytest.raises(tomlrt.TOMLError, match="must not be empty"):
        doc.install("", Table.section())


def test_install_section_rejects_empty_segment() -> None:
    doc = tomlrt.loads("")
    with pytest.raises(tomlrt.TOMLError, match="empty segment"):
        doc.install("tool..poetry", Table.section())


def test_install_scalar_at_dotted_path() -> None:
    doc = tomlrt.loads("")
    doc.install("tool.poetry.version", "0.1.0")
    rendered = tomlrt.dumps(doc)
    assert rendered == '[tool.poetry]\nversion = "0.1.0"\n'
    # Repeated install at a sibling under the same parent should reuse
    # the existing [tool.poetry] section rather than create a new one.
    doc.install(("tool", "poetry", "name"), "x")
    rendered = tomlrt.dumps(doc)
    assert rendered.count("[tool.poetry]") == 1


def test_install_scalar_at_literal_dot_segment() -> None:
    doc = tomlrt.loads("")
    doc.install(("tool", "weird.key"), 1)
    assert tomlrt.dumps(doc) == '[tool]\n"weird.key" = 1\n'


def test_install_plain_dict_at_dotted_path() -> None:
    doc = tomlrt.loads("")
    doc.install("tool.xy", {"x": 1, "y": 2})
    assert tomlrt.dumps(doc) == "[tool]\nxy = { x = 1, y = 2 }\n"


def test_install_scalar_on_inline_table() -> None:
    doc = tomlrt.loads("it = { a = 1 }\n")
    inline = doc.table("it")
    inline.install("b", 2)
    assert tomlrt.dumps(doc) == "it = { a = 1, b = 2 }\n"


def test_install_multi_segment_on_inline_table_errors() -> None:
    doc = tomlrt.loads("it = { a = 1 }\n")
    inline = doc.table("it")
    with pytest.raises(tomlrt.TOMLError, match="inline-style table"):
        inline.install("a.b", 1)


def test_table_accepts_dotted_path() -> None:
    doc = tomlrt.loads('[tool.poetry]\nname = "x"\n')
    assert doc.table("tool.poetry")["name"] == "x"


def test_aot_accepts_dotted_path() -> None:
    doc = tomlrt.loads('[[tool.poetry.source]]\nname = "pypi"\n')
    aot = doc.aot("tool.poetry.source")
    assert isinstance(aot, tomlrt.AoT)
    assert aot[0]["name"] == "pypi"


def test_install_section_round_trips() -> None:
    doc = tomlrt.loads("")
    doc.install("tool.poetry", Table.section({"name": "x"}))
    doc.install("tool.poetry.dependencies", Table.section({"requests": "^2.0"}))
    rendered = tomlrt.dumps(doc)
    # Re-parse and re-dump must produce identical bytes.
    assert tomlrt.dumps(tomlrt.loads(rendered)) == rendered


# ---------------------------------------------------------------------------
# Array assignment
# ---------------------------------------------------------------------------


def test_assign_array_creates_inline_array() -> None:
    doc = tomlrt.loads("")
    doc["authors"] = Array(["A", "B"])
    arr = doc["authors"]
    assert isinstance(arr, tomlrt.Array)
    assert tomlrt.dumps(doc) == 'authors = ["A", "B"]\n'


def test_assign_array_multiline_lays_out_one_per_line() -> None:
    doc = tomlrt.loads("")
    doc["authors"] = Array(["A", "B", "C"], multiline=True)
    assert tomlrt.dumps(doc) == (
        td("""
        authors = [
            "A",
            "B",
            "C",
        ]
        """)
    )


def test_assign_array_custom_indent() -> None:
    doc = tomlrt.loads("")
    doc["x"] = Array([1, 2], multiline=True, indent="  ")
    assert tomlrt.dumps(doc) == td("""
        x = [
          1,
          2,
        ]
        """)


def test_assign_array_empty_multiline_appendable() -> None:
    doc = tomlrt.loads("")
    doc["x"] = Array(multiline=True)
    arr = doc["x"]
    arr.append(1)
    assert tomlrt.dumps(doc) == td("""
        x = [
            1,
        ]
        """)


def test_install_array_dotted_path_creates_parent_section() -> None:
    doc = tomlrt.loads("")
    doc.install("tool.poetry.authors", Array(["A", "B"], multiline=True))
    rendered = tomlrt.dumps(doc)
    assert "[tool]\n" not in rendered
    assert rendered == (
        td("""
        [tool.poetry]
        authors = [
            "A",
            "B",
        ]
        """)
    )


def test_install_array_dotted_path_uses_existing_section() -> None:
    doc = tomlrt.loads('[tool.poetry]\nname = "x"\n')
    doc.install("tool.poetry.authors", Array(["A", "B"]))
    assert tomlrt.dumps(doc) == (
        td("""
        [tool.poetry]
        name = "x"
        authors = ["A", "B"]
        """)
    )


def test_assign_array_replaces_existing_value() -> None:
    doc = tomlrt.loads("a = 1\n")
    doc["a"] = Array([1, 2, 3])
    assert tomlrt.dumps(doc) == "a = [1, 2, 3]\n"


def test_install_array_round_trips() -> None:
    doc = tomlrt.loads("")
    doc.install("tool.poetry.authors", Array(["A", "B"], multiline=True))
    rendered = tomlrt.dumps(doc)
    assert tomlrt.dumps(tomlrt.loads(rendered)) == rendered


def test_detached_aot_preserves_entry_array_multiline_layout() -> None:
    doc = tomlrt.loads("")
    aot = tomlrt.AoT()
    pkg = aot.add({"name": "foo"})
    pkg["files"] = Array([1, 2, 3], multiline=True)
    doc["package"] = aot
    assert tomlrt.dumps(doc) == (
        td("""
            [[package]]
            name = "foo"
            files = [
                1,
                2,
                3,
            ]
            """)
    )


def test_install_detached_aot_preserves_entry_array_multiline_layout() -> None:
    doc = tomlrt.loads("")
    aot = tomlrt.AoT()
    pkg = aot.add({"name": "bar"})
    pkg["files"] = Array([1, 2], multiline=True, indent="  ")
    doc.install("pkgs", aot)
    assert tomlrt.dumps(doc) == (
        td("""
        [[pkgs]]
        name = "bar"
        files = [
          1,
          2,
        ]
        """)
    )


def test_assign_over_aot_keeps_dict_view_in_sync() -> None:
    src = td("""
        [tool]

        [[tool.source]]
        name = "foo"
        """)
    doc = tomlrt.loads(src)
    doc["tool"]["source"] = {}
    assert isinstance(doc["tool"]["source"], tomlrt.Table)
    assert dict(doc["tool"]["source"]) == {}
    assert tomlrt.dumps(doc) == "[tool]\n\nsource = {}\n"


def test_del_then_assign_keeps_dict_view_in_sync() -> None:
    src = td("""
        [tool]

        [[tool.source]]
        name = "foo"
        """)
    doc = tomlrt.loads(src)
    del doc["tool"]["source"]
    doc["tool"]["source"] = {}
    assert isinstance(doc["tool"]["source"], tomlrt.Table)
    assert dict(doc["tool"]["source"]) == {}
    assert tomlrt.dumps(doc) == "[tool]\nsource = {}\n"


def test_pop_then_assign_keeps_dict_view_in_sync() -> None:
    src = td("""
            [tool.poetry]
            name = "x"

            [tool.poetry.extras]
            a = ["one"]
            b = ["two"]
            """)
    doc = tomlrt.loads(src)
    poetry = doc["tool"]["poetry"]
    poetry.pop("extras")
    poetry["extras"] = {"a-norm": ["one"], "b-norm": ["two"]}
    extras = poetry.table("extras")
    assert dict(extras) == {"a-norm": ["one"], "b-norm": ["two"]}


def test_pop_inherited_dotted_key_from_ancestor_section() -> None:
    src = td("""
        [tool]
        poetry.name = "x"

        [tool.poetry.extras]
        a = ["one"]
        """)
    doc = tomlrt.loads(src)
    poetry = doc["tool"]["poetry"]
    assert poetry["name"] == "x"
    poetry.pop("name")
    assert "name" not in poetry
    rendered = tomlrt.dumps(doc)
    assert "poetry.name" not in rendered
    assert "[tool.poetry.extras]" in rendered


def test_set_inherited_dotted_key_mutates_in_place() -> None:
    """Assigning to an inherited dotted entry should update the existing KV,
    not create a duplicate in a new section.
    """
    src = '[tool]\npoetry.name = "x"\n'
    doc = tomlrt.loads(src)
    doc["tool"]["poetry"]["name"] = "y"
    rendered = tomlrt.dumps(doc)
    assert rendered == '[tool]\npoetry.name = "y"\n'


def test_preamble_set_on_empty_doc_renders_before_added_content() -> None:
    doc = Document()
    doc.preamble = ("This is a comment",)
    doc["x"] = 1
    rendered = tomlrt.dumps(doc)
    assert rendered == td("""
        # This is a comment

        x = 1
        """)
    # The migrated comment must remain visible as preamble for round-trip.
    assert doc.preamble == ("This is a comment",)
    assert tomlrt.loads(rendered).preamble == ("This is a comment",)


def test_preamble_migration_for_install_section_and_aot() -> None:
    cases: list[tuple[str, Callable[[tomlrt.Document], object]]] = [
        ("install_section", lambda d: d.install("a", Table.section({"k": 1}))),
        ("install_aot", lambda d: d.install("a", AoT([{"k": 1}]))),
        ("inline_mapping", lambda d: d.__setitem__("a", {"b": 1})),
        ("nested_install_section", lambda d: d.install("a.b", Table.section({"k": 1}))),
    ]
    for op_name, build in cases:
        doc = Document()
        doc.preamble = ("Top",)
        build(doc)
        rendered = tomlrt.dumps(doc)
        assert rendered.startswith("# Top\n\n"), (op_name, rendered)
        assert tomlrt.loads(rendered).preamble == ("Top",), op_name


def test_preamble_preserved_when_empty_section_promoted_to_implicit() -> None:
    """Setting a sub-section on an empty placeholder section promotes
    the parent to an implicit super-table; the demoted parent header
    used to carry the file preamble in its leading trivia, and the
    promotion silently dropped it.
    """
    doc = Document()
    doc.preamble = ("hi",)
    doc["project"] = Table.section()
    assert doc.preamble == ("hi",)
    project = doc.get_table("project")
    assert project is not None
    project["urls"] = Table.section()
    assert doc.preamble == ("hi",)
    rendered = tomlrt.dumps(doc)
    assert rendered == td("""
        # hi

        [project.urls]
        """)
    assert tomlrt.loads(rendered).preamble == ("hi",)


def test_cross_doc_section_assign_demotes_empty_parent_to_implicit() -> None:
    """Cross-doc clone of a sub-section under an empty placeholder
    parent used to skip the synthetic-header demotion, leaving a
    spurious bare ``[a]`` line for what is now an implicit super-table.
    """
    src = tomlrt.loads("[a.b]\nx = 1\n")
    dst = Document()
    dst["a"] = Table.section()
    dst["a"]["b"] = src["a"]["b"]
    assert tomlrt.dumps(dst) == "[a.b]\nx = 1\n"


def test_aot_insert_on_empty_doc_migrates_preamble() -> None:
    """``AoT.insert`` was bypassing the preamble-migration choke-point,
    so on an empty doc with a preamble the comment ended up after the
    inserted ``[[..]]`` section instead of before it.
    """
    doc = tomlrt.loads("")
    doc.preamble = ("Copyright",)
    doc["products"] = AoT()
    aot = doc["products"]
    aot.insert(0, {"name": "x"})
    rendered = tomlrt.dumps(doc)
    assert rendered.startswith(
        td("""
        # Copyright

        [[products]]
        """)
    ), rendered
    assert tomlrt.loads(rendered).preamble == ("Copyright",)


def test_promote_array_preserves_source_kv_leading_and_trailing() -> None:
    """``promote_array`` carries the inline KV's leading comments /
    blank lines onto the first new ``[[..]]`` header, and its trailing
    EOL comment onto the last entry's tail.
    """
    src = td("""
        # header comment

        servers = [{ name = "a" }, { name = "b" }]  # tail
        """)
    doc = tomlrt.loads(src)
    doc.promote_array("servers")
    rendered = tomlrt.dumps(doc)
    assert "# header comment" in rendered
    assert rendered.startswith("# header comment\n")
    assert "# tail" in rendered


def test_aot_insert_at_zero_separates_from_following_entry() -> None:
    """``AoT.insert(0, ...)`` separates the new ``[[..]]`` from both the
    preceding content and the following existing entry, defaulting to
    blank-separated sibling-uniformity.
    """
    doc = tomlrt.loads("[[a]]\nx = 1\n")
    doc["a"].insert(0, {"x": 0})
    assert tomlrt.dumps(doc) == td("""
        [[a]]
        x = 0

        [[a]]
        x = 1
        """)

    # Tight existing layout: don't impose a blank.
    doc = tomlrt.loads(
        td("""
        [[a]]
        x = 1
        [[a]]
        x = 2
        """)
    )
    doc["a"].insert(0, {"x": 0})
    assert tomlrt.dumps(doc) == td("""
        [[a]]
        x = 0
        [[a]]
        x = 1
        [[a]]
        x = 2
        """)


def test_replace_section_preserves_leading_comments() -> None:
    """Replacing a section in place via ``doc[k] = Table.section({...})``
    used to silently drop the comment block that sat above the original
    ``[k]`` header. The slot was reused (since 5527097) but the leading
    trivia was not. Now the prior header's leading is transplanted onto
    the replacement so the surrounding visual context survives.
    """
    src = td("""
        # leader
        [a]
        x=1
        [b]
        y=2
        """)
    doc = tomlrt.loads(src)
    doc["a"] = tomlrt.Table.section({"new": 99})
    assert tomlrt.dumps(doc) == td("""
        # leader
        [a]
        new = 99
        [b]
        y=2
        """)

    # Mid-document, multi-line comment block: also preserved.
    src = td("""
        [a]
        x=1

        # big
        # block
        [b]
        y=2
        """)
    doc = tomlrt.loads(src)
    doc["b"] = tomlrt.Table.section({"new": 99})
    assert tomlrt.dumps(doc) == td("""
        [a]
        x=1

        # big
        # block
        [b]
        new = 99
        """)


def test_replace_aot_preserves_leading_comments() -> None:
    """Same shape for ``doc[k] = AoT([...])`` over an existing AoT."""
    src = td("""
        # top
        # block
        [[a]]
        n=1
        [[a]]
        n=2
        [b]
        x=1
        """)
    doc = tomlrt.loads(src)
    doc["a"] = tomlrt.AoT([{"n": 9}])
    assert tomlrt.dumps(doc) == td("""
        # top
        # block
        [[a]]
        n = 9
        [b]
        x=1
        """)


def test_replace_section_with_aot_preserves_leading_comments() -> None:
    """Cross-flavour replacement still preserves the prior header's
    leading: the slot is the same, only the body changes shape."""
    src = td("""
        # block
        [a]
        x=1
        [b]
        y=2
        """)
    doc = tomlrt.loads(src)
    doc["a"] = tomlrt.AoT([{"n": 1}])
    assert tomlrt.dumps(doc) == td("""
        # block
        [[a]]
        n = 1
        [b]
        y=2
        """)


def test_array_reverse_with_eol_comments_keeps_close_bracket_unindented() -> None:
    """Reordering items in a multi-line array used to leak the
    "indent-for-next-item" trivia onto the new last item, indenting the
    closing bracket. The shared trivia rewriter now strips that tail
    whenever the trailing slot carries a comment."""
    src = td("""
        a = [
          1, # one
          2, # two
          3, # three
        ]
        """)
    doc = tomlrt.loads(src)
    doc["a"].reverse()
    assert tomlrt.dumps(doc) == td("""
        a = [
          3, # three
          2, # two
          1, # one
        ]
        """)
    doc = tomlrt.loads(src)
    doc["a"].sort()
    assert tomlrt.dumps(doc) == src


def test_array_reverse_with_leading_comments_follows_items() -> None:
    """Leading comments are anchored to their item, not their slot:
    reversing the array reverses the comments alongside the values."""
    src = td("""
        a = [
          # for 1
          1,
          # for 2
          2,
          # for 3
          3,
        ]
        """)
    doc = tomlrt.loads(src)
    doc["a"].reverse()
    expected = td("""
        a = [
          # for 3
          3,
          # for 2
          2,
          # for 1
          1,
        ]
        """)
    assert tomlrt.dumps(doc) == expected


def test_array_sort_with_leading_comments_follows_items() -> None:
    src = td("""
        a = [
          # for c
          3,
          # for a
          1,
          # for b
          2,
        ]
        """)
    doc = tomlrt.loads(src)
    doc["a"].sort()
    expected = td("""
        a = [
          # for a
          1,
          # for b
          2,
          # for c
          3,
        ]
        """)
    assert tomlrt.dumps(doc) == expected


def test_array_sort_normalises_mismatched_indents() -> None:
    """``sort`` / ``reverse`` must not leave item 0 carrying the old
    position-0 indent while items[1..] inherit the canonical one."""
    src = "x = [\n    'z',\n  # comment\n      'a',\n   'm',\n]\n"
    doc = tomlrt.loads(src)
    doc.array("x").sort()
    expected = td("""
        x = [
              # comment
              'a',
              'm',
              'z',
        ]
        """)
    assert tomlrt.dumps(doc) == expected
    doc = tomlrt.loads(src)
    doc.array("x").reverse()
    expected_rev = td("""
        x = [
              'm',
              # comment
              'a',
              'z',
        ]
        """)
    assert tomlrt.dumps(doc) == expected_rev


def test_array_insert_zero_pushes_existing_leading_comment_to_new_position() -> None:
    """``insert(0, x)`` must not duplicate the leading-of-(formerly) item-0
    onto both the new item and its old (now position-1) item."""
    src = td("""
        a = [
          # above 1
          1,
          2,
        ]
        """)
    doc = tomlrt.loads(src)
    doc["a"].insert(0, 99)
    expected = td("""
        a = [
          99,
          # above 1
          1,
          2,
        ]
        """)
    assert tomlrt.dumps(doc) == expected


def test_array_pop_drops_the_popped_items_leading_comment() -> None:
    src = td("""
        a = [
          # for 1
          1,
          # for 2
          2,
          # for 3
          3,
        ]
        """)
    doc = tomlrt.loads(src)
    doc["a"].pop(1)
    expected = td("""
        a = [
          # for 1
          1,
          # for 3
          3,
        ]
        """)
    assert tomlrt.dumps(doc) == expected


def test_leading_comments_view_does_not_bleed_eol_comment() -> None:
    """``leading_comments[i]`` for ``i > 0`` is read out of
    ``items[i-1].post_comma_trivia``, which also holds item ``i-1``'s
    EOL comment. The reader used to consume the EOL line as part of the
    leading block, so users saw a phantom extra line."""
    src = td("""
        a = [
          1, # eol
          # above 2
          2,
        ]
        """)
    doc = tomlrt.loads(src)
    assert dict(doc["a"].leading_comments) == {1: ("above 2",)}
    assert dict(doc["a"].comments) == {0: "eol"}


def test_aot_clear_renders_empty_but_keeps_key() -> None:
    """Clearing an AoT empties it like a regular Python list value:
    the key stays on the host (so ``in`` / ``len`` / ``keys`` keep
    behaving like a dict), but render skips it because empty AoTs
    have no syntax in TOML. A subsequent re-parse will not see the
    key — that's an acceptable mutation-time cost; held references
    keep working as plain (now-empty) lists."""
    doc = tomlrt.loads(
        td("""
        [[a]]
        n=1
        [[a]]
        n=2
        """)
    )
    doc["a"].clear()
    assert "a" in doc
    assert len(doc["a"]) == 0
    assert tomlrt.dumps(doc) == ""


def test_aot_pop_last_renders_empty_but_keeps_key() -> None:
    doc = tomlrt.loads(
        td("""
        x=0
        [[a]]
        n=1
        """)
    )
    doc["a"].pop()
    assert "a" in doc
    assert tomlrt.dumps(doc) == "x=0\n"


def test_replace_section_preserves_blank_before_next_section() -> None:
    """Replacing a section in place must not strip the leading blank
    line from the *next* section. The purge step normalised the doc's
    top-blank before the replacement was spliced in, which silently ate
    the inter-section separator carried on the next section's header."""
    src = td("""
        [a]
        x=1

        # next section
        [b]
        y=2
        """)
    doc = tomlrt.loads(src)
    doc["a"] = tomlrt.Table.section({"new": 1})
    assert tomlrt.dumps(doc) == td("""
        [a]
        new = 1

        # next section
        [b]
        y=2
        """)


def test_replace_section_with_aot_preserves_blank_before_next_section() -> None:
    """Same shape for the section -> AoT replacement path."""
    src = td("""
        [a]
        x=1

        [b]
        y=2
        """)
    doc = tomlrt.loads(src)
    doc["a"] = tomlrt.AoT([{"n": 99}])
    assert tomlrt.dumps(doc) == td("""
        [[a]]
        n = 99

        [b]
        y=2
        """)


def test_replace_dotted_subtable_with_value_no_stray_top_blank() -> None:
    """Overwriting a dotted-key sub-table with a scalar must not leave
    a stray blank above the materialised parent header. The new ``[a]``
    header was unconditionally prefixed with a ``\\n`` whenever the doc
    still had any sections, but an empty preamble doesn't count as
    preceding content."""
    doc = tomlrt.loads("[a.b]\nx=1\n")
    doc["a"]["b"] = 99
    assert tomlrt.dumps(doc) == "[a]\nb = 99\n"


def test_setting_eol_comment_does_not_double_indent_next_item() -> None:
    """Adding an EOL comment to a multi-line array item must not push
    the following item's indent. The parser stores the inter-item
    ``\\n  `` on the *previous* item's ``post_comma_trivia``; the
    comment-setter then unconditionally seeded ``next_item.leading``
    with another indent run, so the next item rendered at double the
    original indent."""
    doc = tomlrt.loads(
        td("""
        arr = [
          1,
          2,
          3,
        ]
        """)
    )
    doc["arr"].comments[0] = "z"
    assert tomlrt.dumps(doc) == td("""
        arr = [
          1, # z
          2,
          3,
        ]
        """)


def test_setting_eol_comment_on_consecutive_items_keeps_indent() -> None:
    doc = tomlrt.loads(
        td("""
        arr = [
          1,
          2,
        ]
        """)
    )
    doc["arr"].comments[0] = "zero"
    doc["arr"].comments[1] = "one"
    assert tomlrt.dumps(doc) == td("""
        arr = [
          1, # zero
          2, # one
        ]
        """)


def test_aot_pop_preserves_owned_sub_sections_in_orphan() -> None:
    """``AoT.pop`` returns a view that still carries its [a.sub] children."""
    src = td("""
        [[a]]
        x = 1

        [a.sub]
        y = 2

        [[a]]
        x = 9
        """)
    doc = tomlrt.loads(src)
    popped = doc["a"].pop(0)
    new_doc = tomlrt.loads("")
    new_doc["a"] = popped
    assert tomlrt.dumps(new_doc) == td("""
        [a]
        x = 1

        [a.sub]
        y = 2
        """)


def test_aot_clear_preserves_owned_sub_sections_in_cached_views() -> None:
    """``AoT.clear`` orphans cached entries with their full sub-section run.

    Same root cause as the ``pop`` regression: every dying entry's
    detach must run while the live doc still indexes the owned range,
    or the cached Table view loses its nested ``[a.sub]`` sections.
    """
    src = td("""
        [[a]]
        x = 1

        [a.sub]
        y = 2

        [[a]]
        x = 9
        """)
    doc = tomlrt.loads(src)
    e0 = doc["a"][0]
    doc["a"].clear()
    new_doc = tomlrt.loads("")
    new_doc["a"] = e0
    assert tomlrt.dumps(new_doc) == td("""
        [a]
        x = 1

        [a.sub]
        y = 2
        """)


def test_inline_table_rejects_section_table_value() -> None:
    """Foreign section-Table into an inline-table key raises like AoT.

    The three flavour-bearing "give me a [section] here" requests
    (``Table.section({})``, ``AoT(...)``, an attached section-backed
    Table) all encode the same intent. The first two were already
    rejected by an inline host; the third silently flattened into an
    inline-table-of-inline-tables, dropping the [section] semantics.
    Reject all three uniformly so the inline / standard split stays
    on the producing side, never hidden in a quiet flatten.
    """
    src = tomlrt.loads(
        td("""
        [b]
        x = 1
        [b.sub]
        y = 2
        """)
    )
    nd = tomlrt.loads("")
    nd["a"] = {}
    with pytest.raises(tomlrt.TOMLError, match="inline-style table"):
        nd["a"]["b"] = src["b"]

    # Inline-Table into inline still flattens (legitimate: both sides
    # are inline-style).
    inner_src = tomlrt.loads("inner = { x = 1 }")
    nd2 = tomlrt.loads("")
    nd2["a"] = {}
    nd2["a"]["b"] = inner_src["inner"]
    assert tomlrt.dumps(nd2) == "a = {b = { x = 1 }}\n"


def test_cross_doc_update_retargets_eol_to_dst() -> None:
    """``Document.update(src)`` adopts the destination's line ending.

    Grafting tables from a CRLF source into an LF destination (or
    vice versa) used to leak the source's NewlineNode pieces into
    the destination's slot stream, producing mixed-EOL output.
    Cloning now retargets all structural newlines to the
    destination's detected line ending, so the merged document is
    consistently LF or CRLF.
    """
    dst = tomlrt.loads("\r\n[tool.black]\r\nline-length = 88\r\n")
    src = tomlrt.loads('[project]\nname = "foo"\n\n[build-system]\nrequires = []\n')
    dst.update(src)
    out = tomlrt.dumps(dst)
    # Every newline must be CRLF.
    assert "\n" not in out.replace("\r\n", "")
    assert "[project]" in out
    assert "[build-system]" in out

    # Symmetric: CRLF source into LF destination → all LF.
    dst2 = tomlrt.loads("[a]\nx = 1\n")
    src2 = tomlrt.loads("[b]\r\ny = 2\r\n")
    dst2.update(src2)
    out2 = tomlrt.dumps(dst2)
    assert "\r" not in out2


def test_cross_doc_array_assignment_retargets_eol() -> None:
    """Cross-doc inline-Array assignment retargets multi-line EOL too."""
    dst = tomlrt.loads("a = [1]\r\n")
    src = tomlrt.loads("a = [\n  2,\n  3,\n]\n")
    dst["a"] = src["a"]
    out = tomlrt.dumps(dst)
    assert "\n" not in out.replace("\r\n", "")


def test_mixed_eol_parse_still_roundtrips_byte_exact() -> None:
    """Render-time normalisation would break this; graft-time doesn't.

    Parsing a document whose physical line endings are inconsistent
    must still produce byte-exact output on a no-op round-trip. The
    EOL retargeting fix runs only at clone time, so this invariant
    is preserved.
    """
    mixed = "[a]\r\nx = 1\n[b]\r\ny = 2\n"
    assert tomlrt.dumps(tomlrt.loads(mixed)) == mixed


def test_displaced_inline_view_detaches_on_overwrite() -> None:
    """Held inline-table reference is detached when its hosting KV is dropped.

    The structural-overwrite path (assigning a section/AoT into a
    key currently bound to an inline value, or ``del`` of the
    holding key) used to leave inline Containers and Arrays in
    the displaced subtree with stale ``_layout_root`` /
    ``_attached`` state. That broke identity preservation on
    re-assignment: a held reference would silently be cloned
    instead of live-attached.

    Sections and AoTs already detached correctly via the
    private-orphan rehome dance; this test pins the same
    behaviour for inline views.
    """
    # 1. Inline displaced by section overwrite.
    doc = tomlrt.loads("k = {x = 1}\n")
    it = doc["k"]
    doc["k"] = tomlrt.Table.section({"y": 2})
    doc["other"] = it
    assert doc["other"] is it

    # 2. Array displaced by section overwrite.
    doc = tomlrt.loads("k = [1, 2]\n")
    arr = doc["k"]
    doc["k"] = tomlrt.Table.section({"y": 2})
    doc["x"] = arr
    assert doc["x"] is arr

    # 3. Nested inline-in-inline displaced via del of outer.
    doc = tomlrt.loads("outer = {inner = {x = 1}}\n")
    inner = doc["outer"]["inner"]
    del doc["outer"]
    doc["new"] = inner
    assert doc["new"] is inner

    # 4. Array nested in inline displaced.
    doc = tomlrt.loads("o = {a = [1, 2]}\n")
    arr = doc["o"]["a"]
    del doc["o"]
    doc["x"] = arr
    assert doc["x"] is arr

    # 5. Inline nested in array displaced.
    doc = tomlrt.loads("a = [{x = 1}]\n")
    it = doc["a"][0]
    del doc["a"]
    doc["new"] = it
    assert doc["new"] is it


def test_scalar_overwrite_of_implicit_table_preserves_position() -> None:
    """Replacing an implicit-parent's table-shaped child with a scalar."""
    src = td("""
        [foo.bar.baz]
        quux = 1

        [this]
        that = 2
        """)
    doc = tomlrt.loads(src)
    doc["foo"]["bar"] = 3
    assert tomlrt.dumps(doc) == td("""
        [foo]
        bar = 3

        [this]
        that = 2
        """)


def test_scalar_overwrite_of_explicit_subsection_preserves_position() -> None:
    """Scalar over an explicit ``[foo.bar]`` lands in ``[foo]``'s body.

    The blank line that originally preceded ``[foo.bar]`` is preserved
    on the new binding's leading — the structural-overwrite path
    restores the captured anchor's leading verbatim.
    """
    src = td("""
        [foo]
        x = 1

        [foo.bar]
        y = 2

        [other]
        z = 3
        """)
    doc = tomlrt.loads(src)
    doc["foo"]["bar"] = 99
    assert tomlrt.dumps(doc) == td("""
        [foo]
        x = 1

        bar = 99

        [other]
        z = 3
        """)


def test_synth_inline_overwrite_of_implicit_table_preserves_position() -> None:
    """Same shape as the scalar case but with a ``{}`` (synth-inline) value."""
    src = td("""
        [foo.bar.baz]
        quux = 1

        [this]
        that = 2
        """)
    doc = tomlrt.loads(src)
    doc["foo"]["bar"] = {"q": 1}
    assert tomlrt.dumps(doc) == td("""
        [foo]
        bar = { q = 1 }

        [this]
        that = 2
        """)


def test_scalar_overwrite_at_doc_tail_preserves_position() -> None:
    """Implicit-parent at doc tail: the synthesised block stays at tail."""
    src = td("""
        [other]
        z = 3

        [foo.bar.baz]
        quux = 1
        """)
    doc = tomlrt.loads(src)
    doc["foo"]["bar"] = 7
    assert tomlrt.dumps(doc) == td("""
        [other]
        z = 3

        [foo]
        bar = 7
        """)


def test_scalar_overwrite_of_doc_root_section_relocates_before_headers() -> None:
    """Replacing a top-level section with a scalar must lift the scalar
    above all sibling section headers.

    A top-level KV (or inline-value KV) is only legal *before* any
    section header. Preserving the original `[b]` position would put
    `b = ""` after `[a]` and re-parse as `a.b`, producing a duplicate
    key. The overwrite must therefore drop position and reinsert at
    the top-of-doc seam.
    """
    src = td("""
        [a]
        b = ""

        [b]
        """)
    doc = tomlrt.loads(src)
    doc["b"] = ""
    out = tomlrt.dumps(doc)
    assert out == td("""
        b = ""

        [a]
        b = ""
        """)
    # Must round-trip: a top-level `b = ""` after `[a]` would parse
    # as a duplicate `a.b` and raise.
    assert tomlrt.dumps(tomlrt.loads(out)) == out


def test_synth_inline_overwrite_of_doc_root_section_relocates_before_headers() -> None:
    """Same shape as the scalar case but with an inline-table value."""
    src = td("""
        [a]
        x = 1

        [b]
        y = 2
        """)
    doc = tomlrt.loads(src)
    doc["b"] = {"q": 1}
    out = tomlrt.dumps(doc)
    assert out == td("""
        b = { q = 1 }

        [a]
        x = 1
        """)
    assert tomlrt.dumps(tomlrt.loads(out)) == out


def test_scalar_overwrite_of_doc_root_aot_relocates_before_headers() -> None:
    """AoT → scalar at the doc root must also lift above sibling headers."""
    src = td("""
        [a]
        x = 1

        [[b]]
        y = 2

        [[b]]
        y = 3
        """)
    doc = tomlrt.loads(src)
    doc["b"] = 99
    out = tomlrt.dumps(doc)
    assert out == td("""
        b = 99

        [a]
        x = 1
        """)
    assert tomlrt.dumps(tomlrt.loads(out)) == out


def test_scalar_overwrite_of_discontiguous_implicit_binding() -> None:
    """Replacing an implicit binding whose region spans interleaved sections.

    ``a.b`` is built from ``[a.b]`` plus the later ``[a.b.c]``, with
    an unrelated ``[d]`` in between. The structural overwrite captures
    the original successor of the *first* contiguous run (``[d]``)
    and restores its leading after the move so the visual gap between
    ``a.b``'s former physical location and ``[d]`` survives.
    """
    src = td("""
        [a.b]
        x = 1

        [d]
        y = 2

        [a.b.c]
        z = 3
        """)
    doc = tomlrt.loads(src)
    doc["a"]["b"] = 99
    assert tomlrt.dumps(doc) == td("""
        [a]
        b = 99

        [d]
        y = 2
        """)


def test_scalar_overwrite_inside_aot_entry_preserves_position() -> None:
    """Scalar over a nested ``[a.sub]`` inside the second ``[[a]]`` entry.

    The structural-overwrite machinery routes through
    ``Container._structural_overwrite`` on the AoT-entry's table view
    (the parent), and the synthesised KV must land where ``[a.sub]``
    used to be — between the second and third ``[[a]]`` entries.
    """
    src = td("""
        [[a]]
        x = 1

        [[a]]
        x = 2

        [a.sub]
        z = 9

        [[a]]
        x = 3
        """)
    doc = tomlrt.loads(src)
    doc["a"][1]["sub"] = 7
    assert tomlrt.dumps(doc) == td("""
        [[a]]
        x = 1

        [[a]]
        x = 2

        sub = 7

        [[a]]
        x = 3
        """)


def test_section_overwrite_inside_aot_entry_preserves_position() -> None:
    """Same shape as the scalar case but replacing with a section table."""
    src = td("""
        [[a]]
        x = 1

        [a.sub]
        z = 9

        [[a]]
        x = 2
        """)
    doc = tomlrt.loads(src)
    doc["a"][0]["sub"] = {"q": 1}
    assert tomlrt.dumps(doc) == td("""
        [[a]]
        x = 1

        sub = { q = 1 }

        [[a]]
        x = 2
        """)
