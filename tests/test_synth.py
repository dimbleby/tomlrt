"""Building a document from a mapping.

`Document(mapping)` synthesises the document's slots in one pass. There
is no second implementation to compare against, so these check
invariants instead: that the result round-trips, that it is a fixed
point, and that it agrees with the same content assigned key by key --
which is a genuinely separate path through the mutation layer.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Mapping
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import tomlrt
from _helpers import deep_equal, td
from tomlrt import AoT, Array, Table, TOMLError

if TYPE_CHECKING:
    from collections.abc import Iterator

    from tomlrt._container import TomlInput

if sys.version_info >= (3, 12):
    from typing import override
else:  # pragma: no cover -- backport for Python < 3.12
    from typing_extensions import override

CORPUS = Path(__file__).resolve().parent.parent / "vendor" / "toml-test" / "tests"


def _assigned(data: dict[str, Any]) -> str:
    """Render ``data`` built by assignment, the mutation layer's own path.

    Each value is given the shape `Document(mapping)` would choose, so
    the two should agree; the route there shares nothing but `_build`.
    """
    doc = tomlrt.Document()
    for key, value in data.items():
        doc[key] = _shaped(value)
    return tomlrt.dumps(doc)


def _shaped(value: object) -> TomlInput:
    """``value`` with its structural shape spelled out explicitly."""
    if isinstance(value, (Table, Array, AoT)):
        return value
    if isinstance(value, dict):
        return Table.section({k: _shaped(v) for k, v in value.items()})
    if isinstance(value, list) and value and all(isinstance(x, dict) for x in value):
        return AoT([{k: _shaped(v) for k, v in entry.items()} for entry in value])
    if isinstance(value, (str, int, float, date, time, list)):
        return value
    msg = f"unexpected test value: {value!r}"
    raise AssertionError(msg)


SHAPES: dict[str, dict[str, Any]] = {
    "empty": {},
    "scalars": {"s": "x", "i": 1, "f": 1.5, "b": True},
    "dates": {
        "d": date(2020, 1, 2),
        "t": time(3, 4, 5),
        "dt": datetime(2020, 1, 2, 3, 4, tzinfo=timezone.utc),
    },
    "special floats": {"inf": math.inf, "ninf": -math.inf},
    "section": {"a": {"x": 1}},
    "section after key": {"k": 1, "a": {"x": 1}},
    "key after section": {"a": {"x": 1}, "k": 1},
    "nested sections": {"a": {"b": {"c": {"d": 1}}}},
    "header-less parent": {"tool": {"cov": {"x": 1}}},
    "empty section": {"a": {}},
    "empty section then key": {"a": {}, "z": 1},
    "empty nested": {"a": {"b": {}, "c": 1}},
    "aot": {"items": [{"k": 1}, {"k": 2}]},
    "aot then key": {"a": [{"k": 1}], "z": 2},
    "aot in aot": {"a": [{"b": [{"c": 1}]}]},
    "section in aot": {"a": [{"b": {"c": 1}}]},
    "aot of empty": {"a": [{}]},
    "inline array": {"a": [1, 2, 3]},
    "empty array": {"a": []},
    "nested arrays": {"a": [[1, 2], [3]]},
    "inline table in array": {"a": [1, {"x": 2}]},
    "array of inline tables in array": {"a": [[{"x": 1}]]},
    "quoted keys": {"a b": 1, "c.d": {"e": 1}},
    "unicode key": {"\u043a\u043b\u044e\u0447": 1},
    "empty key": {"": 1},
    "string escapes": {"a": 'q"\\\n\t'},
}


@pytest.mark.parametrize("shape", list(SHAPES))
def test_matches_the_same_content_assigned(shape: str) -> None:
    data = SHAPES[shape]
    assert tomlrt.dumps(tomlrt.Document(data)) == _assigned(data)


@pytest.mark.parametrize("shape", list(SHAPES))
def test_round_trips(shape: str) -> None:
    data = SHAPES[shape]
    assert deep_equal(tomlrt.loads(tomlrt.dumps(tomlrt.Document(data))).to_dict(), data)


def test_corpus_round_trips_and_is_a_fixed_point() -> None:
    """Every valid toml-test document, rebuilt from its own data."""
    files = sorted((CORPUS / "valid").rglob("*.toml"))
    if not files:
        pytest.skip("toml-test corpus is not vendored")
    checked = 0
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        data = tomlrt.loads(text).to_dict()
        out = tomlrt.dumps(tomlrt.Document(data))
        reloaded = tomlrt.loads(out).to_dict()
        assert deep_equal(reloaded, data), path.name
        assert tomlrt.dumps(tomlrt.Document(reloaded)) == out, path.name
        checked += 1
    assert checked > 100


# ---------------------------------------------------------------------------
# Layout the synthesiser has to get right on its own
# ---------------------------------------------------------------------------


def test_section_with_only_subsections_gets_no_header() -> None:
    doc = tomlrt.Document({"tool": {"a": {"x": 1}, "b": {"y": 2}}})
    assert tomlrt.dumps(doc) == td("""
        [tool.a]
        x = 1

        [tool.b]
        y = 2
        """)


def test_keys_precede_subsections_whatever_the_mapping_order() -> None:
    doc = tomlrt.Document({"s": {"x": 1}, "k": 2, "t": {"y": 3}})
    assert tomlrt.dumps(doc) == td("""
        k = 2

        [s]
        x = 1

        [t]
        y = 3
        """)


def test_nested_aot_entries_keep_their_owners() -> None:
    doc = tomlrt.Document({"a": [{"k": 1, "b": [{"c": 2}]}, {"k": 3}]})
    out = tomlrt.dumps(doc)
    assert out == td("""
        [[a]]
        k = 1

        [[a.b]]
        c = 2

        [[a]]
        k = 3
        """)
    assert tomlrt.loads(out).to_dict() == doc.to_dict()


# ---------------------------------------------------------------------------
# Views contribute their contents and their shape, and nothing else
# ---------------------------------------------------------------------------


def test_section_view_is_copied_and_keeps_its_shape() -> None:
    section = Table.section({"x": 1})
    doc = tomlrt.Document({"s": section, "k": 1})
    section["y"] = 2
    assert doc["s"] is not section
    assert tomlrt.dumps(doc) == td("""
        k = 1

        [s]
        x = 1
        """)


def test_inline_view_nested_in_plain_data_stays_inline() -> None:
    inline = Table.inline({"x": 1})
    doc = tomlrt.Document({"outer": {"deep": {"i": inline}}})
    inline["y"] = 2
    assert tomlrt.dumps(doc) == td("""
        [outer.deep]
        i = { x = 1 }
        """)


def test_array_view_of_tables_stays_an_inline_array() -> None:
    doc = tomlrt.Document({"a": Array([{"x": 1}]), "b": [{"y": 2}]})
    assert tomlrt.dumps(doc) == td("""
        a = [{ x = 1 }]

        [[b]]
        y = 2
        """)


def test_inline_table_in_a_list_keeps_the_list_inline() -> None:
    """An explicit inline table is a value, so its list is an array."""
    doc = tomlrt.Document({"a": [Table.inline({"x": 1}), {"y": 2}]})
    assert tomlrt.dumps(doc) == "a = [{ x = 1 }, { y = 2 }]\n"


def test_aot_view_is_copied() -> None:
    aot = AoT([{"k": 1}])
    doc = tomlrt.Document({"e": aot})
    aot.append({"k": 2})
    assert doc["e"] is not aot
    assert tomlrt.dumps(doc) == td("""
        [[e]]
        k = 1
        """)


def test_attached_view_keeps_its_comments_and_spacing() -> None:
    """A view holding source layout is cloned, not rebuilt from its data."""
    src = tomlrt.loads(
        td("""
        [a]
        # keep me
        x = 1  # trailing
        y = 'literal'
        """)
    )
    doc = tomlrt.Document({"k": 1, "b": src.table("a")})
    assert tomlrt.dumps(doc) == td("""
        k = 1

        [b]
        # keep me
        x = 1  # trailing
        y = 'literal'
        """)


# ---------------------------------------------------------------------------
# Rejected input
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("data", "exc", "match"),
    [
        ({"a": object()}, TypeError, "cannot convert object"),
        ({"a": (1, 2)}, TypeError, "cannot assign tuple"),
        ({"a": b"x"}, TypeError, "cannot assign bytes"),
        ({"a": None}, TypeError, "cannot convert NoneType"),
        ({1: "x"}, TypeError, "TOML keys must be str"),
        ({"a": {2: "x"}}, TypeError, "TOML keys must be str"),
        ({"a": [{"b": object()}]}, TypeError, "cannot convert object"),
        ({"a": time(1, 2, tzinfo=timezone.utc)}, ValueError, "cannot represent"),
    ],
)
def test_rejected_input_is_reported(
    data: dict[Any, Any], exc: type[Exception], match: str
) -> None:
    with pytest.raises(exc, match=match):
        tomlrt.Document(data)


def test_first_bad_value_is_reported_whichever_kind_it_is() -> None:
    """A bad value under a subsection precedes a later bad direct key.

    The synthesiser writes keys before subsections, but it checks in the
    mapping's own order, so the report does not depend on where a value
    ends up in the document.
    """
    data = {"a": {"inner": object()}, "z": time(1, 2, tzinfo=timezone.utc)}
    with pytest.raises(TypeError, match="cannot convert object"):
        tomlrt.Document(data)


def test_mapping_keys_are_validated_before_values() -> None:
    data = {"a": [0, {"ok": object(), 1: 2}]}
    with pytest.raises(TypeError, match="TOML keys must be str"):
        tomlrt.Document(data)


# ---------------------------------------------------------------------------
# A synthesised document is an ordinary one
# ---------------------------------------------------------------------------


def test_synthesised_document_is_mutable() -> None:
    doc = tomlrt.Document({"a": {"x": 1}, "items": [{"k": 1}]})
    doc["z"] = 2
    doc.table("a")["y"] = 3
    doc.aot("items").append({"k": 2})
    del doc.table("a")["x"]
    out = tomlrt.dumps(doc)
    assert out == td("""
        z = 2

        [a]
        y = 3

        [[items]]
        k = 1

        [[items]]
        k = 2
        """)
    assert tomlrt.loads(out).to_dict() == doc.to_dict()


def test_synthesised_document_sorts_and_reparses() -> None:
    doc = tomlrt.Document({"b": 2, "a": 1, "s": {"y": 2, "x": 1}})
    doc.sort()
    doc.table("s").sort()
    out = tomlrt.dumps(doc)
    assert out == td("""
        a = 1
        b = 2

        [s]
        x = 1
        y = 2
        """)
    assert tomlrt.loads(out).to_dict() == doc.to_dict()


def test_dumps_of_a_plain_mapping_round_trips() -> None:
    data: dict[str, Any] = {"k": 1, "s": {"x": "y"}, "e": [{"n": 1}]}
    assert tomlrt.loads(tomlrt.dumps(data)).to_dict() == data


# ---------------------------------------------------------------------------
# `dumps` renders the run without building the views over it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", list(SHAPES))
def test_dumps_matches_the_document_it_would_have_built(shape: str) -> None:
    """Asked for text, `dumps` skips the views -- and must not show it."""
    data = SHAPES[shape]
    assert tomlrt.dumps(data) == tomlrt.Document(data).render()


def test_dumps_of_a_graft_matches_the_document_it_would_have_built() -> None:
    """Including the one value whose slots are cloned rather than made."""

    def data() -> dict[str, Any]:
        src = tomlrt.loads(
            td("""
            # above a
            [a]
            x = 1  # eol

            [a.sub]
            y = 2
            """)
        )
        return {"first": 1, "g": src.table("a"), "last": 2}

    assert tomlrt.dumps(data()) == tomlrt.Document(data()).render()


@pytest.mark.parametrize(
    "value",
    [object(), b"x", (1, 2), {1: "int key"}, {"a": object()}, [object()]],
    ids=["object", "bytes", "tuple", "int key", "nested object", "in a list"],
)
def test_dumps_rejects_what_document_rejects(value: object) -> None:
    """The same complaint, whichever of the two the caller reached for."""
    with pytest.raises((TypeError, TOMLError)) as from_document:
        tomlrt.Document({"k": value})
    with pytest.raises((TypeError, TOMLError)) as from_dumps:
        tomlrt.dumps({"k": value})
    assert str(from_dumps.value) == str(from_document.value)
    assert type(from_dumps.value) is type(from_document.value)


def test_dumps_rejects_a_non_mapping() -> None:
    """``None`` used to render as an empty document, silently.

    `dumps` reached it through ``Document(None)``, whose argument is
    optional because a document can be built empty. Nothing else it is
    handed gets that reading, and its own signature does not offer it.
    """
    not_a_mapping: Any = None
    with pytest.raises(TypeError, match="must be a Mapping"):
        tomlrt.dumps(not_a_mapping)


def test_dumps_of_an_attached_table_keeps_its_layout() -> None:
    """A table that owns section layout is cloned, not rebuilt.

    It has comments and spacing a mapping cannot describe, so `dumps`
    builds the document that knows how to carry them across.
    """
    src = tomlrt.loads(
        td("""
        # above a
        [a]
        x = 1  # eol

        [a.sub]
        y = 2
        """)
    )
    assert tomlrt.dumps(src.table("a")) == td("""
        # above a

        x = 1  # eol

        [sub]
        y = 2
        """)


# ---------------------------------------------------------------------------
# Property differential
# ---------------------------------------------------------------------------

_KEYS = st.from_regex(r"\A[A-Za-z][A-Za-z0-9_-]{0,6}\Z")
_LEAVES = st.one_of(
    st.text(max_size=8),
    st.integers(min_value=-(2**31), max_value=2**31 - 1),
    st.booleans(),
    st.sampled_from([1.5, -3.25, math.inf]),
    st.dates(),
)
_INLINE = st.recursive(
    _LEAVES,
    lambda children: st.one_of(
        st.lists(children, max_size=3),
        st.dictionaries(_KEYS, children, max_size=3),
    ),
    max_leaves=6,
)
_TREE = st.recursive(
    _INLINE,
    lambda children: st.one_of(
        st.dictionaries(_KEYS, children, max_size=3),
        st.lists(st.dictionaries(_KEYS, children, max_size=3), min_size=1, max_size=2),
    ),
    max_leaves=8,
)


@pytest.mark.slow
@settings(max_examples=400, suppress_health_check=[HealthCheck.too_slow])
@given(st.dictionaries(_KEYS, _TREE, max_size=4))
def test_property_round_trips_and_matches_assignment(data: dict[str, Any]) -> None:
    out = tomlrt.dumps(tomlrt.Document(data))
    assert deep_equal(tomlrt.loads(out).to_dict(), data)
    assert out == _assigned(data)


def test_array_of_tables_inside_an_inline_value_is_rejected() -> None:
    with pytest.raises(TOMLError, match="array-of-tables inside an inline table"):
        tomlrt.Document({"a": [AoT([{"k": 1}])]})


def test_section_table_inside_an_inline_value_is_rejected() -> None:
    with pytest.raises(TOMLError, match="section-style table inside an inline"):
        tomlrt.Document({"a": [1, Table.section({"x": 1})]})
    with pytest.raises(TOMLError, match="section-style table inside an inline"):
        tomlrt.Document({"a": Table.inline({"b": Table.section({"x": 1})})})


def test_attached_view_inside_an_aot_entry_keeps_its_layout() -> None:
    src = tomlrt.loads(
        td("""
        [a]
        # keep me
        x = 1
        """)
    )
    doc = tomlrt.Document({"items": [{"k": 1, "sub": src.table("a")}, {"k": 2}]})
    out = tomlrt.dumps(doc)
    assert out == td("""
        [[items]]
        k = 1

        [items.sub]
        # keep me
        x = 1

        [[items]]
        k = 2
        """)
    assert tomlrt.loads(out).to_dict() == doc.to_dict()


# ---------------------------------------------------------------------------
# Empty containers keep their keys
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "rendered"),
    [
        (AoT([]), "a = []\n"),
        (Array([]), "a = []\n"),
        ([], "a = []\n"),
        (Table.inline({}), "a = {}\n"),
        ({}, "[a]\n"),
        (Table.section({}), "[a]\n"),
    ],
)
def test_empty_containers_keep_their_key(value: object, rendered: str) -> None:
    assert tomlrt.dumps(tomlrt.Document({"a": value})) == rendered


def test_empty_aot_inside_a_section_stays_in_the_body() -> None:
    doc = tomlrt.Document({"s": {"a": AoT([]), "sub": {"x": 1}}})
    assert tomlrt.dumps(doc) == td("""
        [s]
        a = []

        [s.sub]
        x = 1
        """)


# ---------------------------------------------------------------------------
# A value holding source layout keeps its place as well as its layout
# ---------------------------------------------------------------------------


def test_cloned_section_keeps_the_mapping_order() -> None:
    src = tomlrt.loads(
        td("""
        [g]
        # lead
        q = 1
        """)
    )
    doc = tomlrt.Document({"g": src.table("g"), "plain": {"x": 1}})
    assert list(doc) == ["g", "plain"]
    assert tomlrt.dumps(doc) == td("""
        [g]
        # lead
        q = 1

        [plain]
        x = 1
        """)


def test_cloned_array_keeps_its_place_among_keys() -> None:
    src = tomlrt.loads("arr = [ 1,  2 ]\n")
    doc = tomlrt.Document({"k": 1, "g": src.array("arr"), "h": 2})
    assert list(doc) == ["k", "g", "h"]
    assert tomlrt.dumps(doc) == td("""
        k = 1
        g = [ 1,  2 ]
        h = 2
        """)


def test_list_of_attached_tables_becomes_an_aot() -> None:
    src = tomlrt.loads(
        td("""
        [sec]
        x = 1
        """)
    )
    doc = tomlrt.Document({"xs": [src.table("sec")]})
    assert tomlrt.dumps(doc) == td("""
        [[xs]]
        x = 1
        """)


def test_children_of_a_passed_table_take_document_shape() -> None:
    """A detached table holds plain data; its children get section shape.

    ``Document(mapping)`` turns nested mappings into ``[section]``
    blocks, and that rule reaches inside a `Table` handed to it -- the
    table itself chose only its own shape.
    """
    doc = tomlrt.Document({"t": Table.section({"k": {"a": 1}})})
    assert tomlrt.dumps(doc) == td("""
        [t.k]
        a = 1
        """)
    inline = tomlrt.Document({"t": Table.section({"k": Table.inline({"a": 1})})})
    assert tomlrt.dumps(inline) == td("""
        [t]
        k = { a = 1 }
        """)


def test_attached_empty_aot_needs_no_cloning() -> None:
    """With no entries there are no headers, so there is nothing to clone."""
    src = tomlrt.loads("k = 1\n")
    src["a"] = AoT([])
    doc = tomlrt.Document({"k": 1, "a": src.aot("a")})
    assert tomlrt.dumps(doc) == td("""
        k = 1
        a = []
        """)


def test_key_order_follows_the_mapping_with_and_without_a_graft() -> None:
    src = tomlrt.loads(
        td("""
        [g]
        q = 1
        """)
    )
    plain: dict[str, Any] = {"sub": {"y": 2}, "k": 1}
    assert list(tomlrt.Document(plain)) == ["sub", "k"]
    assert list(tomlrt.Document({**plain, "g": src.table("g")})) == ["sub", "k", "g"]
    assert list(tomlrt.Document({"s": {"inner": {"z": 1}, "j": 2}})["s"]) == [
        "inner",
        "j",
    ]


def test_mapping_that_repeats_a_key_takes_the_last_value() -> None:
    """A `Mapping` may hand out the same key twice; a document may not."""

    class Repeats(Mapping[str, Any]):
        @override
        def __iter__(self) -> Iterator[str]:
            return iter(["a", "b"])

        @override
        def __len__(self) -> int:
            return 2

        @override
        def __getitem__(self, key: str) -> Any:
            return {"a": 2, "b": 3}[key]

        @override
        def items(self) -> Any:
            yield ("a", 1)
            yield ("a", 2)
            yield ("b", 3)

    out = tomlrt.dumps(Repeats())
    assert out == td("""
        a = 2
        b = 3
        """)
    assert tomlrt.loads(out).to_dict() == {"a": 2, "b": 3}


def test_cloned_section_keeps_the_comments_above_its_header() -> None:
    src = tomlrt.loads(
        td("""
        # note: this section matters
        [sec]
        x = 1
        """)
    )
    assert tomlrt.dumps(tomlrt.Document({"sec": src.table("sec")})) == td("""
        # note: this section matters
        [sec]
        x = 1
        """)
    # and after a key, where the destination supplies the blank line
    assert tomlrt.dumps(tomlrt.Document({"z": 1, "sec": src.table("sec")})) == td("""
        z = 1

        # note: this section matters
        [sec]
        x = 1
        """)


def test_cloned_section_keeps_comments_when_its_header_is_not_first() -> None:
    """The install takes the leading of the block's *first* slot.

    A forward-declared subtable comes before its parent's header, and a
    super-table declared last comes after it; either way the comments
    that travel are the first slot's.
    """
    forward = tomlrt.loads(
        td("""
        # about a.b
        [a.b]
        x = 1
        [a]
        y = 2
        """)
    )
    assert tomlrt.dumps(tomlrt.Document({"a": forward.table("a")})) == td("""
        # about a.b
        [a.b]
        x = 1
        [a]
        y = 2
        """)

    last = tomlrt.loads(
        td("""
        [a.b]
        x = 1
        # about a
        [a]
        y = 2
        """)
    )
    assert tomlrt.dumps(tomlrt.Document({"a": last.table("a")})) == td("""
        [a.b]
        x = 1
        # about a
        [a]
        y = 2
        """)


def test_cloned_dotted_section_keeps_its_own_key_comments() -> None:
    """Dotted keys install as body, each keeping the comment above it."""
    src = tomlrt.loads(
        td("""
        # makes fruit a table
        fruit.apple.smooth = true

        # adds to it
        fruit.orange = 2
        """)
    )
    out = tomlrt.dumps(tomlrt.Document({"fruit": src.table("fruit")}))
    assert "# makes fruit a table" in out
    assert "# adds to it" in out
    assert out.count("# makes fruit a table") == 1
    assert tomlrt.loads(out).to_dict() == {"fruit": src["fruit"].to_dict()}


def test_cloned_aot_keeps_the_comments_above_its_first_header() -> None:
    src = tomlrt.loads(
        td("""
        # lead beta
        [[b]]
        k = 1

        [[b]]
        k = 2
        """)
    )
    assert tomlrt.dumps(tomlrt.Document({"b": src.aot("b")})) == td("""
        # lead beta
        [[b]]
        k = 1

        [[b]]
        k = 2
        """)


def test_repeating_mapping_inside_an_inline_value_takes_the_last_value() -> None:
    """The inline builder takes one reading of a mapping, as the plan does."""

    class Repeats(Mapping[str, Any]):
        _pairs: ClassVar[list[tuple[str, int]]] = [("a", 1), ("a", 2), ("b", 3)]

        @override
        def __iter__(self) -> Iterator[str]:
            return (k for k, _ in self._pairs)

        @override
        def __len__(self) -> int:
            return len(self._pairs)

        @override
        def __getitem__(self, key: str) -> Any:
            return next(v for k, v in self._pairs if k == key)

        @override
        def items(self) -> Any:
            return iter(self._pairs)

    out = tomlrt.dumps({"k": [Repeats(), 1]})
    assert out == "k = [{ a = 2, b = 3 }, 1]\n"
    assert tomlrt.loads(out).to_dict() == {"k": [{"a": 2, "b": 3}, 1]}


def test_grafts_without_a_header_of_their_own() -> None:
    """A `Document` installs under a header synthesised for its key.

    An implicit section spelled by dotted keys installs as body.
    """
    src = tomlrt.loads("q = 1\n")
    assert tomlrt.dumps(tomlrt.Document({"d": src})) == td("""
        [d]
        q = 1
        """)
    implicit = tomlrt.loads("a.b.x = 1\n")
    assert tomlrt.dumps(tomlrt.Document({"g": implicit.table("a")})) == "g.b.x = 1\n"


def test_graft_materialises_the_section_holding_it() -> None:
    """A graft is content, so its enclosing section still needs a header."""
    dotted = tomlrt.loads("fruit.apple = 1\n")
    assert tomlrt.dumps(
        tomlrt.Document({"outer": {"fruit": dotted.table("fruit")}})
    ) == td("""
        [outer]
        fruit.apple = 1
        """)
    sub = tomlrt.loads("x = 1\n")
    assert tomlrt.dumps(tomlrt.Document({"o1": {"o2": {"a": sub}}})) == td("""
        [o1.o2.a]
        x = 1
        """)


def test_document_graft_keeps_its_place_and_its_comments_once() -> None:
    """A `Document` is always a header block, and carries no comments of its own."""
    doc = tomlrt.Document({"o": tomlrt.loads("x = 1\n"), "s": {"q": 1}})
    assert list(doc) == ["o", "s"]
    assert tomlrt.dumps(doc) == td("""
        [o]
        x = 1

        [s]
        q = 1
        """)
    headed = tomlrt.Document({"o": tomlrt.loads("# note\n[a]\nx = 1\n")})
    out = tomlrt.dumps(headed)
    assert out.count("# note") == 1
    assert out == td("""
        [o]
        # note
        [o.a]
        x = 1
        """)


def test_dotted_graft_keeps_its_place_among_plain_keys() -> None:
    """A graft is written where the mapping puts it, like any other key."""
    src = tomlrt.loads("[h]\n# above\na.c = 1\na.d = 2\n")
    doc = tomlrt.Document({"first": 1, "g": src.table("h").table("a"), "last": 2})
    assert list(doc) == ["first", "g", "last"]
    assert tomlrt.dumps(doc) == td("""
        first = 1
        # above
        g.c = 1
        g.d = 2
        last = 2
        """)


def test_implicit_graft_splits_its_body_from_its_subsections() -> None:
    """A header-less section spells itself in both regions of its host.

    Its own keys are dotted lines of the body; its sub-sections are
    blocks below it, after whatever else the body holds.
    """
    src = tomlrt.loads("# top\na.x = 1\n\n# sub\n[a.sub]\ny = 2\n")
    doc = tomlrt.Document({"g": src.table("a"), "last": 2})
    assert tomlrt.dumps(doc) == td("""
        # top
        g.x = 1
        last = 2

        # sub
        [g.sub]
        y = 2
        """)


def test_graft_at_the_tail_keeps_a_source_that_ends_without_a_newline() -> None:
    """Only a slot written after it needs a terminator adding."""

    def source() -> Table:
        return tomlrt.loads("[a]\nx = 1").table("a")

    assert tomlrt.dumps(tomlrt.Document({"g": source()})) == "[g]\nx = 1"
    assert tomlrt.dumps(tomlrt.Document({"g": source(), "z": {"q": 2}})) == td("""
        [g]
        x = 1

        [z]
        q = 2
        """)
