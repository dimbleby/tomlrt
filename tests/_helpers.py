"""Shared test helpers.

* :func:`td` — write TOML fixtures as indented triple-quoted literals
  so tests don't degenerate into walls of ``\\n``-escaped strings.
* :func:`reparses` — re-parse a rendered document with ``tomli`` to
  sanity-check that the output is still valid TOML carrying the
  expected logical values. ``tomli`` is preferred over the stdlib
  ``tomllib`` because as of writing (Python 3.14) ``tomllib`` is TOML
  1.0 only, whereas ``tomli`` 2.4+ accepts TOML 1.1 syntax (multi-line
  inline tables, etc.).
* :func:`deep_equal` — structural equality (dict/list-recursive,
  NaN-as-equal-to-itself, type-exact at the leaves) used by the
  property/fuzz suites to compare a decoded `Document` against a
  `tomli`-parsed oracle or a Python-dict shadow model.
* :func:`fuzz_context` — attach the seed to whatever a fuzz program
  raises, so a CI failure is reproducible.
* :func:`fuzz_seeds` — the fuzzers' seed source, honouring
  ``TOMLRT_FUZZ_SEED`` so a reported seed can be replayed.
* :func:`check_slot_chain` / :func:`check_view_caches` — structural
  oracles over a document's physical slot stream and the projections
  of it that every container caches. A corrupt projection renders
  perfectly, so the model oracles cannot see one; these can.
"""

from __future__ import annotations

import itertools
import math
import os
import secrets
from contextlib import contextmanager
from textwrap import dedent
from typing import TYPE_CHECKING, Any

import tomli

from tomlrt import AoT
from tomlrt._container import _is_section
from tomlrt._slots import KVSlot, StructuralHeaderSlot, slot_local_key

if TYPE_CHECKING:
    from collections.abc import Iterator

    from tomlrt._container import Container, Document
    from tomlrt._slots import Slot


def td(src: str) -> str:
    """Dedent ``src`` and strip a single leading newline.

    Lets tests embed TOML fixtures as indented triple-quoted literals::

        src = td('''
            [a]
            x = 1
            [a.sub]
            y = 2
        ''')

    A single leading newline (the one immediately after the opening
    ``\"\"\"``) is stripped so the first content line starts at column 0,
    then :func:`textwrap.dedent` removes the common leading whitespace.
    The result is byte-identical to ``"[a]\\nx = 1\\n[a.sub]\\ny = 2\\n"``,
    which matters because everything in this project is round-tripped
    byte-for-byte.
    """
    return dedent(src).removeprefix("\n")


def reparses(src: str) -> dict[str, Any]:
    """Parse ``src`` with ``tomli`` and return the result."""
    return tomli.loads(src)


def deep_equal(a: object, b: object) -> bool:
    """Structural equality treating NaN as equal to itself.

    Recurses through ``dict`` and ``list`` values; leaves compare with
    ``==`` gated on exact type match (so e.g. ``1`` and ``True`` are
    not conflated). Used across the property/fuzz suites to compare a
    decoded `Document` against a `tomli`-parsed oracle or a plain
    Python-dict shadow model.
    """
    if isinstance(a, float) and isinstance(b, float):
        return (math.isnan(a) and math.isnan(b)) or a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(deep_equal(v, b[k]) for k, v in a.items())
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(
            deep_equal(x, y) for x, y in zip(a, b, strict=True)
        )
    return type(a) is type(b) and a == b


@contextmanager
def fuzz_context(ctx: str) -> Iterator[None]:
    """Re-raise whatever the block raises with ``ctx`` prefixed to it.

    A seed-driven fuzzer only earns its keep if a failure says which
    seed reproduces it. The oracle assertions could carry that in their
    own messages, but the usual way a fuzzer finds a bug is a raise from
    inside the library itself, which would otherwise report a traceback
    with the seed nowhere in it. Wrapping the whole program covers both.
    """
    try:
        yield
    except Exception as exc:
        msg = f"{ctx}: {type(exc).__name__}: {exc}"
        raise AssertionError(msg) from exc


def fuzz_seeds(count: int) -> Iterator[int]:
    """Yield ``count`` fresh 64-bit seeds, or the one pinned by the environment.

    Fresh seeds every run keep the fuzzers exploring new programs rather
    than re-checking a frozen grid. Setting ``TOMLRT_FUZZ_SEED=<n>``
    replays exactly one program instead, which is how a seed reported by
    :func:`fuzz_context` is reproduced::

        TOMLRT_FUZZ_SEED=<n> uv run pytest -m slow <the failing test id>
    """
    override = os.environ.get("TOMLRT_FUZZ_SEED")
    if override is not None:
        yield int(override)
        return
    for _ in range(count):
        yield secrets.randbits(64)


def _chain(doc: Document) -> list[Slot]:
    """The document's slots, walked forward from its head."""
    out: list[Slot] = []
    seen: set[int] = set()
    cur = doc._head  # noqa: SLF001
    while cur is not None:
        assert id(cur) not in seen, "cycle in slot chain"
        seen.add(id(cur))
        out.append(cur)
        cur = cur._next  # noqa: SLF001
    return out


def check_slot_chain(doc: Document, ctx: str) -> None:
    """Assert the document's physical slot stream is well formed.

    Cheap enough to run after every fuzz step. These are the checks
    whose violation has been seen to render correctly and only fail
    later, on an unrelated operation:

    * the chain is acyclic and its links are symmetric;
    * ``_head`` / ``_tail`` really are its ends;
    * the order keys that place refs in doc order increase along it.
    """
    slots = _chain(doc)

    if slots:
        assert doc._tail is slots[-1], f"{ctx}: _tail is not the chain end"  # noqa: SLF001
        assert slots[0]._prev is None, f"{ctx}: head has a predecessor"  # noqa: SLF001
    else:
        assert doc._tail is None, f"{ctx}: empty chain with a _tail"  # noqa: SLF001

    for a, b in itertools.pairwise(slots):
        assert b._prev is a, f"{ctx}: broken back-link"  # noqa: SLF001
        assert a._order < b._order, f"{ctx}: order keys are not increasing"  # noqa: SLF001


def _containers(doc: Document) -> list[Container]:
    """Every section-backed container reachable from ``doc``."""
    out: list[Container] = []

    def visit(c: Container) -> None:
        out.append(c)
        for child in c.values():
            if _is_section(child):
                visit(child)
            elif isinstance(child, AoT):
                for entry in child:
                    visit(entry)

    visit(doc)
    return out


def _expected_body_tail(c: Container) -> Slot | None:
    """The body tail ``_layout_ops._recompute_body_tail`` would derive."""
    owner = c._owner_aot_entry  # noqa: SLF001
    for slot in reversed(c._refs):  # noqa: SLF001
        if isinstance(slot, KVSlot) and slot.owner_aot_entry is owner:
            return slot
    return c._header  # noqa: SLF001


def check_view_caches(doc: Document, ctx: str) -> None:
    """Assert every container's projections of the slot stream agree with it.

    ``_refs``, ``_index`` and ``_body_tail`` are caches over the one
    source of physical order, the doc-stream linked list. A mutation
    that files a ref out of order, or leaves a bucket naming a slot the
    walk no longer reaches, still renders correctly -- the renderer
    walks the stream, not the caches -- and only fails later, when an
    insertion consults a cache to decide where a slot belongs.
    """
    pos = {id(s): i for i, s in enumerate(_chain(doc))}
    for c in _containers(doc):
        where = f"{ctx}: {c._path}"  # noqa: SLF001
        assert len({id(s) for s in c._refs}) == len(c._refs), (  # noqa: SLF001
            f"{where}: slot is indexed more than once"
        )
        for slot in c._refs:  # noqa: SLF001
            assert id(slot) in pos, f"{where}: ref names a slot off the chain"
        order = [pos[id(slot)] for slot in c._refs]  # noqa: SLF001
        assert order == sorted(order), f"{where}: _refs is not in doc order"

        # `_body_tail` is maintained incrementally on every append and
        # only fully recomputed on a body-affecting delete, so what is
        # caught here is the incremental path drifting from the
        # recomputation that is meant to agree with it.
        want = _expected_body_tail(c)
        assert c._body_tail is want, (  # noqa: SLF001
            f"{where}: _body_tail is stale (got {c._body_tail!r}, want {want!r})"  # noqa: SLF001
        )

        for slot in c._refs:  # noqa: SLF001
            owners = slot._containers  # noqa: SLF001
            assert len({id(owner) for owner in owners}) == len(owners), (
                f"{where}: duplicate container back-pointer"
            )
            assert any(back is c for back in slot._containers), (  # noqa: SLF001
                f"{where}: slot does not back-point to this container"
            )
        own_header = c._header  # noqa: SLF001
        if own_header is not None:
            own_path = c._path  # noqa: SLF001
            assert isinstance(own_header, StructuralHeaderSlot), (
                f"{where}: _header does not name a header"
            )
            assert own_header.path == own_path, f"{where}: _header path mismatch"

        for key, bucket in c._index.items():  # noqa: SLF001
            expected = [s for s in c._refs if slot_local_key(s, c) == key]  # noqa: SLF001
            assert bucket == expected, f"{where}: _index[{key!r}] is not its projection"
        for slot in c._refs:  # noqa: SLF001
            local = slot_local_key(slot, c)
            if local is None:
                # The one ref with no local key is the container's own
                # header, which lives in `_header`, not `_index`.
                assert c._header is slot, f"{where}: keyless ref is not the header"  # noqa: SLF001
                continue
            assert slot in c._index.get(local, []), f"{where}: ref missing from _index"  # noqa: SLF001
