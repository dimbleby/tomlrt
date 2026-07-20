"""Smoke tests: parse, value access, exact round-trip."""

from __future__ import annotations

from datetime import date, datetime, time, timezone
from textwrap import dedent

import pytest

import tomlrt
from _helpers import td

UTC = timezone.utc

# ---------------------------------------------------------------------------
# Round-trip corpus: dumps(parse(s)) == s, byte-for-byte.
# ---------------------------------------------------------------------------

ROUND_TRIP_CORPUS: list[str] = [
    "",
    "\n",
    "# just a comment\n",
    "key = 1\n",
    "key = 1",  # no trailing newline
    'name = "Tom"\n',
    "title = 'literal'\n",
    "pi = 3.14\n",
    "neg = -7\n",
    "hex = 0xCAFE_BABE\n",
    "oct = 0o755\n",
    "bin = 0b1010\n",
    "huge = 1_000_000\n",
    "flag = true\nother = false\n",
    "arr = [1, 2, 3]\n",
    'mixed = [1, "two", 3.0]\n',
    "nested = [[1, 2], [3, 4]]\n",
    "point = { x = 1, y = 2 }\n",
    dedent(
        """\
        # leading file comment
        title = "TOML Example"

        [owner]
        name = "Tom Preston-Werner"
        dob = 1979-05-27T07:32:00-08:00 # First class dates

        [database]
        enabled = true
        ports = [ 8000, 8001, 8002 ]
        data = [ ["delta", "phi"], [3.14] ]
        temp_targets = { cpu = 79.5, case = 72.0 }

        [servers]

          [servers.alpha]
          ip = "10.0.0.1"
          role = "frontend"

          [servers.beta]
          ip = "10.0.0.2"
          role = "backend"
        """,
    ),
    dedent(
        """\
        [[products]]
        name = "Hammer"
        sku = 738594937

        [[products]]  # empty table within the array

        [[products]]
        name = "Nail"
        sku = 284758393

        color = "gray"
        """,
    ),
    dedent(
        """\
        # dotted keys
        physical.color = "orange"
        physical.shape = "round"
        site."google.com" = true
        """,
    ),
    dedent(
        '''\
        str1 = """
        Roses are red
        Violets are blue"""
        str2 = """\\
            The quick brown \\
            fox jumps over \\
            the lazy dog.\\
            """
        path = 'C:\\Users\\nodejs\\templates'
        regex2 = \'\'\'I [dw]on\'t need \\d{2} apples\'\'\'
        '''
    ),
    dedent(
        """\
        odt1 = 1979-05-27T07:32:00Z
        odt2 = 1979-05-27T00:32:00-07:00
        ldt1 = 1979-05-27T07:32:00
        ldt2 = 1979-05-27T00:32:00.999999
        ld1 = 1979-05-27
        lt1 = 07:32:00
        lt2 = 00:32:00.999999
        """,
    ),
    # CRLF immediately after a multi-line string opening triple is
    # trimmed by the parser, so it must be re-emitted verbatim too.
    # CRLF inside the body must also pass through unchanged.
    'a = """\r\nfirst\r\nsecond\r\n"""\n',
    "b = '''\r\nfirst\r\nsecond\r\n'''\n",
    # CRLF inside an array body — exercises the scanner's CRLF
    # newline-detection path.
    "arr = [\r\n  1,\r\n  2,\r\n]\r\n",
]


@pytest.mark.parametrize("src", ROUND_TRIP_CORPUS)
def test_round_trip(src: str) -> None:
    doc = tomlrt.loads(src)
    assert tomlrt.dumps(doc) == src


# ---------------------------------------------------------------------------
# Value access
# ---------------------------------------------------------------------------


def test_basic_value_access() -> None:
    doc = tomlrt.loads(
        dedent(
            """\
            title = "Example"
            count = 42
            ratio = 0.75
            on = true

            [owner]
            name = "Tom"
            """,
        ),
    )
    assert doc["title"] == "Example"
    assert doc["count"] == 42
    assert doc["ratio"] == 0.75
    assert doc["on"] is True
    owner = doc["owner"]
    assert isinstance(owner, tomlrt.Table)
    assert owner["name"] == "Tom"


def test_arrays_are_real_lists() -> None:
    doc = tomlrt.loads("xs = [1, 2, 3]\n")
    xs = doc["xs"]
    assert isinstance(xs, list)
    assert xs == [1, 2, 3]


def test_inline_table_access() -> None:
    doc = tomlrt.loads("p = { x = 1, y = 2 }\n")
    p = doc["p"]
    assert isinstance(p, tomlrt.Table)
    assert dict(p) == {"x": 1, "y": 2}


def test_table_is_inline_distinguishes_section_from_inline() -> None:
    doc = tomlrt.loads(
        td(
            """
            inline = { y = 2 }

            [section]
            x = 1
            """,
        ),
    )
    section = doc["section"]
    inline = doc["inline"]
    assert isinstance(section, tomlrt.Table)
    assert isinstance(inline, tomlrt.Table)
    assert section.is_inline is False
    assert inline.is_inline is True


def test_table_is_inline_for_factory_constructors() -> None:
    assert tomlrt.Table.section().is_inline is False
    assert tomlrt.Table.inline().is_inline is True


def test_nested_tables_and_dotted_keys() -> None:
    src = dedent(
        """\
        [a.b.c]
        v = 1

        [a]
        x = 2
        """,
    )
    doc = tomlrt.loads(src)
    a = doc["a"]
    assert isinstance(a, tomlrt.Table)
    assert a["x"] == 2
    b = a["b"]
    assert isinstance(b, tomlrt.Table)
    c = b["c"]
    assert isinstance(c, tomlrt.Table)
    assert c["v"] == 1


def test_array_of_tables() -> None:
    src = dedent(
        """\
        [[products]]
        name = "Hammer"

        [[products]]
        name = "Nail"
        """,
    )
    doc = tomlrt.loads(src)
    products = doc["products"]
    assert isinstance(products, list)
    assert len(products) == 2
    p0 = products[0]
    p1 = products[1]
    assert isinstance(p0, tomlrt.Table)
    assert isinstance(p1, tomlrt.Table)
    assert p0["name"] == "Hammer"
    assert p1["name"] == "Nail"


def test_datetime_values() -> None:
    doc = tomlrt.loads(
        dedent(
            """\
            odt = 1979-05-27T07:32:00Z
            ldt = 1979-05-27T07:32:00
            ld = 1979-05-27
            lt = 07:32:00
            """,
        ),
    )
    assert doc["odt"] == datetime(1979, 5, 27, 7, 32, 0, tzinfo=UTC)
    assert doc["ldt"] == datetime(1979, 5, 27, 7, 32, 0)  # noqa: DTZ001 - local datetime is naive by spec
    assert doc["ld"] == date(1979, 5, 27)
    assert doc["lt"] == time(7, 32, 0)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "src",
    [
        "key =\n",
        "= 1\n",
        '"unterminated\n',
        "x = 01\n",
        "x = 1__0\n",
        "x = 1.\n",
        "x = .1\n",
        # Inline table missing '=' between key and value.
        "t = { a 1 }\n",
        # Hex/oct/bin integers with consecutive underscores.
        "x = 0xff__ff\n",
        "x = 0o7__7\n",
        "x = 0b1__1\n",
        # Signed integer with no body.
        "x = +\n",
        "x = -\n",
        # Float with no mantissa, and float with leading-zero mantissa.
        "x = +e5\n",
        "x = 01e1\n",
        # Misplaced underscore between exponent sign and digits.
        "x = 1e+_1\n",
        "x = 1e-_1\n",
        "x = 1E+_1\n",
        "x = 1E-_1\n",
        "x = 1.2e+_3\n",
        # Unterminated multi-line strings after a non-empty body.
        'x = """abc',
        "x = '''abc",
        # Stray CR (no following LF) inside multi-line strings.
        'x = """ab\rcd"""\n',
        "x = '''ab\rcd'''\n",
        # Malformed time literals.
        "t = 12:30:\n",
        "t = 12:30T05\n",
        "t = 12:30:00.\n",
        # Control character U+007F is not allowed in basic strings.
        's = "ab\x7fcd"\n',
        # TOML grammar restricts every digit position in date/time
        # literals to ASCII 0-9. ``str.isdigit`` and bare ``int``
        # accept other Unicode decimal digits (Arabic-Indic,
        # Devanagari, ...) — they must be rejected.
        "a = \u0662\u0660\u0662\u0664-01-01\n",  # Arabic-Indic year
        "a = 2024-\u0660\u0661-01\n",  # Arabic-Indic month
        "a = 2024-01-\u0660\u0661\n",  # Arabic-Indic day
        "a = \u0660\u0660:00:00\n",  # Arabic-Indic hour
        "a = 00:\u0660\u0660:00\n",  # Arabic-Indic minute
        "a = 00:00:\u0660\u0660\n",  # Arabic-Indic second
        "a = 00:00:00.\u0660\u0660\u0660\n",  # Arabic-Indic fraction
        "a = 2024-01-01T00:00:00+\u0660\u0661:00\n",  # offset hour
        "a = 2024-01-01T00:00:00+01:\u0660\u0660\n",  # offset minute
        "a = 2024-01-01 \u0660\u0660:00:00\n",  # space-separator look-ahead
    ],
)
def test_parse_errors(src: str) -> None:
    with pytest.raises(tomlrt.TOMLParseError):
        tomlrt.loads(src)


def test_deep_array_nesting_raises_parse_error_not_recursionerror() -> None:
    payload = "x = " + "[" * 500 + "1" + "]" * 500 + "\n"
    with pytest.raises(tomlrt.TOMLParseError, match="nesting exceeds"):
        tomlrt.loads(payload)


def test_deep_inline_table_nesting_raises_parse_error() -> None:
    payload = "x = " + "{a=" * 500 + "1" + "}" * 500 + "\n"
    with pytest.raises(tomlrt.TOMLParseError, match="nesting exceeds"):
        tomlrt.loads(payload)


def test_overlong_integer_raises_parse_error_not_valueerror() -> None:
    # CPython caps int<->str conversion at 4300 digits by default; a
    # longer decimal literal must surface as a TOMLParseError, not the
    # bare ValueError that ``int()`` raises. The message forwards the
    # underlying cause rather than presuming it.
    payload = "x = " + "1" * 4301 + "\n"
    with pytest.raises(tomlrt.TOMLParseError, match="invalid integer") as exc_info:
        tomlrt.loads(payload)
    assert isinstance(exc_info.value.__cause__, ValueError)


def test_inline_table_dotted_key_conflict_reports_inline_position() -> None:
    # The conflict between `x = 1` and `x.y = 2` lives on line 1 inside
    # the inline table; the error must point there, not at the start of
    # the next line.
    src = "a = { x = 1, x.y = 2 }\n"
    with pytest.raises(tomlrt.TOMLParseError) as exc_info:
        tomlrt.loads(src)
    assert exc_info.value.line == 1
    # The conflicting "x.y" key starts at column 14 (1-based).
    assert exc_info.value.col == 14


def test_parse_error_is_value_error() -> None:
    # `tomllib.TOMLDecodeError` extends `ValueError`; tomlrt should be
    # catchable the same way for drop-in compatibility.
    with pytest.raises(ValueError, match="expected"):
        tomlrt.loads("a =")


# ---------------------------------------------------------------------------
# Validator rules — pin the user-facing error message text for each rule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("src", "message"),
    [
        # Header redefinition rules.
        ("[a]\n[a]\n", r"redefinition of table 'a'"),
        ("[a]\n[[a]]\n", r"cannot redefine table 'a' as an array-of-tables"),
        ("[[a]]\n[a]\n", r"cannot redefine array-of-tables 'a' as a normal table"),
        ("[a.b]\n[[a]]\n", r"already used as an implicit table"),
        (
            "[x]\na.b = 1\n[x.a.e]\n[x.a]\n",
            r"cannot define 'x\.a' as a table: already created via dotted keys",
        ),
        # Header / value-path conflicts.
        ("a = 1\n[a]\n", r"cannot define 'a' as a table: already defined as a value"),
        ("a = 1\n[a.b]\n", r"cannot use 'a' as a table: already defined as a value"),
        # Key/value rules.
        ("x = 1\nx = 2\n", r"duplicate key 'x'"),
        ("[a.b]\n[a]\nb = 1\n", r"key 'a\.b' already defined as a table"),
        (
            "[a.b]\n[a]\nb.c = 1\n",
            r"cannot extend explicitly-defined table 'a\.b' via dotted keys",
        ),
        (
            "[[a.b]]\n[a]\nb.c = 1\n",
            r"cannot extend array-of-tables 'a\.b' via dotted keys",
        ),
        # Datetime / time value errors.
        ("t = 07:32:00x\n", r"bad fractional seconds"),
        # Inline-table key conflicts.
        ("t = { x = 1, x = 2 }\n", r"duplicate key 'x' in inline table"),
        (
            "t = { a.b = 1, a = 2 }\n",
            r"key 'a' in inline table conflicts with an existing dotted-key prefix",
        ),
        (
            "t = { a = true }\n[t.a]\n",
            r"cannot use 't' as a table: already defined as a value",
        ),
    ],
)
def test_parse_error_messages(src: str, message: str) -> None:
    with pytest.raises(tomlrt.TOMLParseError, match=message):
        tomlrt.loads(src)


@pytest.mark.parametrize(
    "src",
    [
        # An AoT entry's bound keys / sub-headers must not collide with
        # those of a sibling entry: the validator resets per entry.
        "[[h]]\nx = 1\n[[h]]\nx = 2\n",
        "[[h]]\n[h.sub]\n[[h]]\n[h.sub]\n",
        # Nested AoTs reset their sub-tables along with the outer entry.
        ("[[h]]\n[[h.inner]]\n[h.inner.leaf]\n[[h]]\n[[h.inner]]\n[h.inner.leaf]\n"),
    ],
)
def test_aot_scope_resets_between_entries(src: str) -> None:
    tomlrt.loads(src)


def test_moderate_array_nesting_still_parses() -> None:
    payload = "x = " + "[" * 50 + "1" + "]" * 50 + "\n"
    doc = tomlrt.loads(payload)
    assert tomlrt.dumps(doc) == payload


# ---------------------------------------------------------------------------
# Out-of-order tables — iteration order should match tomllib (first-appearance)
# ---------------------------------------------------------------------------


def test_iteration_order_child_section_before_parent() -> None:
    src = td("""
        [a.b]
        x = 1
        [a]
        y = 2
        """)
    doc = tomlrt.loads(src)
    a = doc.table("a")
    assert list(a) == ["b", "y"]


def test_iteration_order_parent_then_child_then_more_direct_keys() -> None:
    # With a single [a] block plus a sub-section after it, direct keys
    # come first because they appear first physically.
    src = td("""
        [a]
        x = 1
        [a.b]
        y = 2
        """)
    doc = tomlrt.loads(src)
    assert list(doc.table("a")) == ["x", "b"]


def test_iteration_order_sibling_interleaved_between_parent_and_child() -> None:
    src = td("""
        [a]
        x = 1
        [b]
        y = 2
        [a.sub]
        z = 3
        """)
    doc = tomlrt.loads(src)
    assert list(doc) == ["a", "b"]
    assert list(doc.table("a")) == ["x", "sub"]


def test_iteration_order_aot_then_sibling_then_more_aot() -> None:
    src = td("""
        [[fruits]]
        name = "apple"
        [[other]]
        n = 1
        [[fruits]]
        name = "banana"
        """)
    doc = tomlrt.loads(src)
    assert list(doc) == ["fruits", "other"]
    fruits = doc.aot("fruits")
    assert [t["name"] for t in fruits] == ["apple", "banana"]
