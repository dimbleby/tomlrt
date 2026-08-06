"""Run the official BurntSushi/toml-lang ``toml-test`` corpus.

The test data lives in ``vendor/toml-test`` (a ``git clone`` of
https://github.com/toml-lang/toml-test, gitignored). If the directory
is missing the whole module is skipped so the suite still runs in
clean checkouts.

For every entry in the TOML 1.1.0 manifest:

* ``valid/X.toml`` is parsed; its decoded values are compared against
  the tagged JSON in ``valid/X.json`` (the toml-test "tagged" format).
* ``invalid/X.toml`` must raise ``TOMLParseError``.

We further assert byte-exact round-trip on every valid case.
"""

from __future__ import annotations

import json
from datetime import date, datetime, time
from pathlib import Path

import pytest

import tomlrt

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TOML_TEST_ROOT = _REPO_ROOT / "vendor" / "toml-test"
_MANIFEST = _TOML_TEST_ROOT / "tests" / "files-toml-1.1.0"

if not _MANIFEST.is_file():
    pytest.skip(
        "toml-test corpus not vendored; run "
        "`git clone --depth 1 https://github.com/toml-lang/toml-test "
        "vendor/toml-test`",
        allow_module_level=True,
    )


def _load_manifest() -> tuple[list[str], list[str]]:
    valid: list[str] = []
    invalid: list[str] = []
    for raw_line in _MANIFEST.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("valid/") and line.endswith(".toml"):
            valid.append(line)
        elif line.startswith("invalid/") and line.endswith(".toml"):
            invalid.append(line)
    return valid, invalid


_VALID, _INVALID = _load_manifest()


# ---------------------------------------------------------------------------
# Tagged-JSON decoding (toml-test format)
# ---------------------------------------------------------------------------


def _decode_tagged(obj: object) -> object:
    """Convert a toml-test "tagged" JSON value to a plain Python value."""
    if isinstance(obj, dict):
        if "type" in obj and "value" in obj and len(obj) == 2:
            t = obj["type"]
            v = obj["value"]
            assert isinstance(v, str)
            if t == "string":
                return v
            if t == "integer":
                return int(v)
            if t == "float":
                if v in ("inf", "+inf"):
                    return float("inf")
                if v == "-inf":
                    return float("-inf")
                if v in ("nan", "+nan", "-nan"):
                    return float("nan")
                return float(v)
            if t == "bool":
                return v == "true"
            if t == "datetime":
                return _parse_iso_datetime(v)
            if t == "datetime-local":
                return _parse_iso_datetime(v)
            if t == "date-local":
                return date.fromisoformat(v)
            if t == "time-local":
                return _parse_iso_time(v)
            msg = f"unknown tagged type: {t!r}"
            raise AssertionError(msg)
        return {k: _decode_tagged(val) for k, val in obj.items()}
    if isinstance(obj, list):
        return [_decode_tagged(v) for v in obj]
    msg = f"unexpected JSON node: {obj!r}"
    raise AssertionError(msg)


def _parse_iso_datetime(s: str) -> datetime:
    # Python 3.11+ ``fromisoformat`` accepts a trailing ``Z`` for UTC and
    # arbitrary fractional-second precision; on 3.10 we have to normalise.
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(_normalise_fractional_seconds(s))


def _parse_iso_time(s: str) -> time:
    return time.fromisoformat(_normalise_fractional_seconds(s))


def _normalise_fractional_seconds(s: str) -> str:
    # On 3.10 ``fromisoformat`` requires exactly 3 or 6 fractional digits.
    dot = s.rfind(".")
    if dot < 0:
        return s
    end = dot + 1
    while end < len(s) and s[end].isdigit():
        end += 1
    digits = s[dot + 1 : end]
    if len(digits) in (3, 6):
        return s
    digits = digits.ljust(6, "0") if len(digits) < 6 else digits[:6]
    return s[: dot + 1] + digits + s[end:]


# ---------------------------------------------------------------------------
# Recursive comparison that treats NaN as equal and floats as approx
# ---------------------------------------------------------------------------


def _equal(a: object, b: object) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        if a != a and b != b:  # both NaN
            return True
        return a == b
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return False
        return all(_equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b, strict=True))
    return type(a) is type(b) and a == b


def _materialise(value: object) -> object:
    """Recursively convert a Document/Table/Array into plain Python values."""
    if isinstance(value, tomlrt.Table):
        return {k: _materialise(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_materialise(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("relpath", _VALID, ids=lambda p: p)
def test_valid(relpath: str) -> None:
    toml_path = _TOML_TEST_ROOT / "tests" / relpath
    json_path = toml_path.with_suffix(".json")
    src = toml_path.read_bytes().decode("utf-8")

    # 1. Parses without error.
    doc = tomlrt.loads(src)

    # 2. Round-trip is byte-exact.
    assert tomlrt.dumps(doc) == src, f"round-trip differs for {relpath}"

    # 3. Decoded values match the corpus' tagged JSON.
    expected = _decode_tagged(json.loads(json_path.read_text(encoding="utf-8")))
    actual = _materialise(doc)
    assert _equal(actual, expected), (
        f"decoded values differ for {relpath}\n"
        f"  expected={expected!r}\n  actual={actual!r}"
    )


@pytest.mark.parametrize("relpath", _INVALID, ids=lambda p: p)
def test_invalid(relpath: str) -> None:
    toml_path = _TOML_TEST_ROOT / "tests" / relpath
    raw = toml_path.read_bytes()
    try:
        src = raw.decode("utf-8")
    except UnicodeDecodeError:
        # Invalid UTF-8 is itself a rejection of the TOML document.
        return
    with pytest.raises(tomlrt.TOMLParseError):
        tomlrt.loads(src)
