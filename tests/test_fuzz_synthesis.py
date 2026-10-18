"""Property tests for synthesis from plain Python trees."""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import tomlrt
from _helpers import deep_equal, dumps_by_assignment

pytestmark = pytest.mark.slow

_KEYS = st.from_regex(r"\A[A-Za-z][A-Za-z0-9_-]{0,15}\Z")
_LEAVES = st.one_of(
    st.text(max_size=20),
    st.integers(min_value=-(2**31), max_value=2**31 - 1),
    st.booleans(),
    st.floats(allow_nan=False, width=64),
    st.dates(),
)
_INLINE = st.recursive(
    _LEAVES,
    lambda children: st.one_of(
        st.lists(children, max_size=4),
        st.dictionaries(_KEYS, children, max_size=4),
    ),
    max_leaves=6,
)
_TREE = st.recursive(
    _INLINE,
    lambda children: st.one_of(
        st.dictionaries(_KEYS, children, max_size=4),
        st.lists(st.dictionaries(_KEYS, children, max_size=4), min_size=1, max_size=2),
    ),
    max_leaves=16,
)


@settings(
    max_examples=500,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(st.dictionaries(_KEYS, _TREE, max_size=4))
def test_property_round_trips_and_matches_assignment(data: dict[str, Any]) -> None:
    out = tomlrt.dumps(tomlrt.Document(data))
    assert deep_equal(tomlrt.loads(out).to_dict(), data)
    assert tomlrt.dumps(data) == out == dumps_by_assignment(data)
