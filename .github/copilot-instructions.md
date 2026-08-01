# Instructions for AI coding agents

This file is read by GitHub Copilot, Copilot CLI, and similar agents
when they work in this repository. Humans are welcome to read it too —
it doubles as a high-signal contributor guide.

## What this project is

`tomlrt` is a **pure-Python, format-preserving** TOML parser and
writer. The non-negotiable invariant is:

> Parsing a document and dumping it again, with no mutations in
> between, must return the **exact same bytes** — including comments,
> whitespace, string style (literal vs basic, single vs multiline),
> number formatting, and line endings (LF vs CRLF).

If a change you are about to make could break that, stop and rethink.

## Toolchain

- **`uv`** is the only supported package/dependency manager. Do not
  introduce `pip`, `poetry`, `pipenv`, `tox`, `nox`, `setuptools`, or
  `requirements.txt`.
- Build backend is **`hatchling`**.
- Supported Python versions: **3.10 – 3.14**.

## Common commands

```bash
uv sync                          # install dev deps
uv run pytest -q                 # run the test suite (~10s)
uv run pytest --cov              # tests + branch coverage
uv run pytest -m slow            # property + bytes-level fuzz suite
uv run mypy                      # strict type-check src/ and tests/
uvx ty check                     # second (Astral) type-checker
uv run ruff check .              # lint
uv run ruff format .             # apply formatting
uv run ruff format --check .     # CI-style format check
```

A `Makefile` wraps the most common invocations (`make test`, `make
fuzz`, `make coverage`, `make lint`, `make docs`, `make docs-serve`,
`make bench`, `make clean`); use it if you prefer.

All five checks (`pytest`, `mypy`, `ty check`, `ruff check`, `ruff
format --check`) must pass before any commit. CI runs the same set on
Python 3.10–3.14. `ty` is a second, independent type-checker (run via
`uvx`; it is not a declared dev dependency) — it sometimes flags things
`mypy` does not, so keep both green.

## Coding standards

- **`mypy --strict`** clean — no `# type: ignore` without a specific
  error code, and ideally no ignores at all. Prefer fixing the type.
- **`ruff` with `select = ["ALL"]`** clean — see `[tool.ruff.lint.ignore]`
  in `pyproject.toml` for the curated exceptions. Do not add new
  per-line `# noqa` without a strong reason.
- **`ruff format`** is the source of truth for formatting. Run it
  before committing.
- **No runtime dependencies beyond conditional stdlib backports.**
  The only declared runtime dep is `typing_extensions` on
  Python < 3.12 (for `Self` / `override`), behind a `python_version`
  marker. Don't add others. `dependency-groups.dev` and
  `dependency-groups.docs` may grow, but only with care.
- **No `cast()` in user-facing code paths.** Tests should not need
  `cast()` either; the typed accessors `Table.array(k)`,
  `Table.table(k)`, `Table.aot(k)`, `Array.array(i)`, `Array.table(i)`
  exist precisely to avoid this.
- **`from __future__ import annotations`** at the top of every module
  (enforced by ruff's isort `required-imports`).
- Do not add comments that merely restate the code. Comment intent and
  invariants, not mechanics.
- **Validate user input at user-facing boundaries; trust typed
  signatures below them.** Public factories / mutators (`loads`,
  `Document(data=…)`, `Table.section`, `Table.inline`, `AoT.__init__`
  / `append` / `insert` / `__setitem__` / `add`,
  `Container.__setitem__`) each call `_validate_key` /
  `_validate_mapping` exactly once. Helpers below the API boundary
  (`_make_unattached_entry`, `_populate_unattached`, …) trust their
  signatures. Don't re-validate in the middle of a chain.

## Architecture (in `src/tomlrt/`)

The codebase is layered: **physical slot stream** at the bottom,
**logical dict/list views** on top, **mutation primitives** between
them. Read roughly in this order:

### Foundation

- **`_errors.py`** — public exception hierarchy.
- **`_paths.py`** — key-path argument parsing and validation
  (the `t["a", "b"]` / `t[("a", "b")]` shapes used by the public
  API).
- **`_typecheck.py`** — runtime type-checks for user-supplied keys
  and mappings (`_validate_key`, `_validate_mapping`). Pure helpers
  with no dependency on the
  container / array layers, so any module that accepts user-supplied
  data at a public boundary can import them without participating
  in the circular-import graph that `_container` / `_array` form
  among themselves. Distinct from `_validator.py` (parse-time
  semantic validator) — this module is API-boundary plumbing.
- **`_trivia.py`** — `Trivia` / `TriviaPiece` types and pure helpers
  over them (whitespace, newlines, comments). Depends only on
  `_errors`.
- **`_scanner.py`** — the `(src, end, pos)` cursor and the `scan_*`
  primitives the parser drives. String scanning is *semantic*:
  escapes are decoded, surrogate code points rejected, and the
  resulting node carries both the raw lexeme and the decoded value.
  Performance-sensitive: prefer bulk `str` scans over per-character
  loops.
- **`_values.py`** — the inline-value layer. Every TOML value
  (scalar, `ArrayValue`, `InlineTableValue`) carries enough source
  text to re-emit byte-exactly. Pure data; no slot-stream awareness.
  `ArrayValue` and `InlineTableValue` share a generic `CommaValue`
  base that owns the `items` list plus a pair of bracket-pad
  anchors — `header_trivia` (gap immediately after `[` / `{`) and
  `final_trivia` (gap before the closing bracket) — so that the
  above-item region of item 0 and the post-comma trivia of item -1
  have a single canonical owner; per-item `leading` only owns the
  region above items 1..n-1. Each item is one of two sibling
  concrete leaves of a `CommaItem` base: `ArrayItem` (bare value,
  used by `ArrayValue`) or `InlineTableEntry` (with a ``key = ``
  prefix, used by `InlineTableValue`). Annotate with `CommaItem`
  when working polymorphically over both flavours and with the
  concrete leaf when the code is flavour-specific —
  `ArrayValue.items: list[ArrayItem]` narrows away
  `InlineTableEntry` at the type level.
- **`_scalar.py`** — Python-to-TOML scalar predicates / coercion
  helpers (`is_scalar`, etc.). Depends on `_values` only.
- **`_slots.py`** — the **physical slot stream**:
  - `Slot` — base; carries `leading: Trivia`, `_prev` / `_next`
    intrusive linked-list pointers, `owner_aot_entry`, and a
    `_refs` back-pointer list (every `SlotRef` that targets the
    slot — bounded by path depth, used for O(depth) ref scrub on
    AoT removal).
  - `KVSlot` — one `key = value` line (`host_path`, `key_parts`,
    `key_seps`, `value`, `eol`).
  - `StructuralHeaderSlot` — one `[a.b]` / `[[a.b]]` header (`path`,
    `kind`, `entry`, `synthetic`).
  - `AoTEntry` — bookkeeping for an `[[a]]` entry (its
    `entry_slots`, the table view it backs).
  - `SlotRef` — a per-container *occurrence* of a slot.
    `local_key` is a derived `@property` of `(slot, container)`
    geometry — never store it. Registers itself on the target
    slot's `_refs` list at construction; `unfile_ref` unregisters.

### Parser

- **`_parser.py`** — hand-written recursive-descent parser,
  TOML 1.0 + 1.1, that drives `_Scanner` to produce a flat slot
  stream and feeds each header / `key = value` / inline-table key
  into `_Validator`.
- **`_validator.py`** — semantic validator for the cross-section /
  cross-line rules that the per-line slot stream cannot express on
  its own (a key bound as a value cannot later be opened as a
  table, `[H]` cannot redefine an already-opened table, dotted
  keys cannot extend an explicitly defined table / AoT, inline-
  table local key rules). Owned and invoked by `_parser.py`.
  Separate from `_typecheck.py`, which is the runtime
  input-validation layer for the public mutation API.

### Logical view construction & mutation

- **`_build.py`** — single linear pass over the parser's slot
  stream that constructs the `Document` body and all nested
  `Table` / `Array` / `AoT` views, populating dict storage in
  doc-stream-first-occurrence order. The *one* place that derives
  implicit containers from slot paths.
- **`_layout_ops.py`** — section-side mutation primitives: insert
  / delete / sort on the doc-stream linked list; `_index` and `_refs`
  bookkeeping; KV / section / AoT-entry append; subtree rehome.
  By far the largest file. Internal hot-path conventions:
  - **Reverse-walks of `c._refs`** go through `_last_kv(c, predicate)`.
    The wrappers (`_last_direct_kv`, `_recompute_body_tail`) exist
    where the same predicate composition is reused or carries a
    fast-path; one-off walks pass the predicate inline. Don't add a
    third ad-hoc walk.
  - **Ordered ref filing** goes through `_file_ordered_ref`; physical
    region permutations finish through `_finish_region_permutation`,
    which updates every affected `_refs` / `_index` projection.
  - **Container sorting is region-local** and keeps leaf / dotted-KV
    blocks before structural section / AoT blocks so re-parsing cannot
    change ownership. `key` / `reverse` apply within those partitions.
  - **Bulk ref removal** walks each slot's back-pointers through
    `_scrub_owned_slots_via_backptrs`, not ancestor-wide cache scans.
  - **`Container._body_tail`** is the cached doc-stream-tail of
    the container's region; treat it as ground truth for
    "what's the latest body slot of `c`?". `_last_direct_kv`
    uses it as an O(1) fast path before falling back to a
    reverse-walk.
- **`_inline_ops.py`** — inline-table mutation primitives. Inline
  tables are decoupled from the doc-stream linked list: a top-
  level inline table is one `KVSlot` whose `value` is an
  `InlineTableValue`, and mutation operates on
  `InlineTableValue.items` directly. Thin orchestration only —
  every structural splice / reorder / boundary fix-up is delegated
  to `_comma_ops` so inline-array and inline-table mutation share
  one canonical implementation; this module just resolves the
  outermost inline-table for a dotted key, files / unfiles the
  matching `InlineTableEntry`, and forwards to `splice_in` /
  `splice_out` / `reorder_owned`. Also exposes the inline-table
  multi-line control (`set_inline_multiline` / `ensure_inline_multiline`)
  backing `Table.multiline` / `Table.set_multiline`, delegating the
  actual expand / collapse to `_format.set_comma_value_multiline`.
- **`_comma_ops.py`** — structural mutation primitives shared by
  inline arrays (`Array`) and inline tables (`_inline_ops`).
  Owns the canonical layout invariants for any `CommaValue`:
  per-item trivia ownership, the single-row-break rule, EOL
  section attachment, and trailing-comma policy. `Boundary` is the
  canonical capture / compose / restore model for trivia spanning
  adjacent items. The cross-module
  surface is intentionally small — a few `splice_*` / `reorder_owned`
  entry points consumed by `Array` / `_inline_ops`, plus the
  row-break primitives (`shift_pieces`, `boundary_break_holder`)
  shared with `_comma_comments`; the lower-level boundary-flip and
  EOL-section helpers stay module-private. The
  shared bracket-pad re-anchoring uses `split_above_block` /
  `join_above_block` from `_trivia.py`. A future change to the
  canonical inline-value model only needs to land here.
- **`_format.py`** — pure-function canonicaliser invoked by the
  public `Container.format()` / `Array.format()` methods. Walks a
  subtree of slots / values and rewrites trivia to a canonical
  shape (KV `key = value` spacing, header inner-pad, sibling-
  spacing rules, single-line vs multi-line inline shape, EOL
  comment placement), configured by the public `FormatOptions` (the
  old `comments=` argument is deprecated). Shape-preserving for
  inline values (single-line stays single-line; multi-line stays
  multi-line) and idempotent. The structural counterpart is
  `_comma_ops`, which owns *changing* layout; this module owns
  *canonicalising* the layout you already have. Re-uses
  `flip_to_*` / `_take_eol` / `_put_eol` from `_comma_ops` for
  the bits that touch the comma-value boundary. Also owns
  `set_comma_value_multiline` — the shared single ↔ multi-line
  expand / collapse for any `CommaValue`, used by both
  `Array.set_multiline` and the inline-table multi-line control.
- **`_render.py`** — pure linear walk of the doc-stream slot list
  + trailing trivia → source string. Byte-exact for any
  unmodified parse.

### Logical views

- **`_container.py`** — `Container` (the abstract base, a `dict`
  subclass), `Document`, and `Table`. Holds `_refs`, `_index`,
  `_path`, `_parent`, `_layout_root`, `_owner_aot_entry`,
  `_body_tail`, `_value`, `_header_ref`, `_inline`. Exposes
  `_wire(layout_root=, parent=, path=, owner=)` — every container
  construction site goes through it for the four common attachment
  fields; flavour-specific bits (`_inline`, `_value`, `_header_ref`,
  `_body_tail`) stay explicit at the call site so the table's kind
  is visible. `_doc_newline` is the canonical "newline of the
  owning document, or `\n` if detached" accessor — prefer it over
  reaching into `_layout_root._newline`. Public `Mapping` /
  `MutableMapping` API; mutation is delegated to `_layout_ops` /
  `_inline_ops` (which in turn delegates to `_comma_ops` for
  inline-value structural fix-ups).
- **`_array.py`** — `Array(list)` (inline arrays) and
  `AoT(list[Table])` (array-of-tables) views, plus the `AoTEntry`
  glue that connects an entry's slots to its Table view. Inline-
  array structural mutation is forwarded to `_comma_ops`.
- **`_comments.py`** — the `MutableMapping`-shaped EOL / leading-
  comment side-channel views over **section** `Container` slot trivia
  (`Container.comments` / `leading_comments` / `leading_block` when the
  container is section-backed). These operate on the physical slot
  stream (`KVSlot` / `StructuralHeaderSlot`). Also owns the shared
  comment encode/decode + validation helpers (`_encode_comment`,
  `_decode_comment`, `_validate_comment_str`, `_validate_comment_seq`)
  reused by the comma-value views below.
- **`_comma_comments.py`** — the flavour-agnostic comment core shared
  by inline **arrays** and inline **tables**, both of which carry
  comments on a `CommaValue` / `CommaItem` (not the slot stream). Owns
  the per-item EOL / above-block read-write plumbing **and** the
  generic keyed mapping views: `CommaCommentAdapter` (per-flavour hooks
  `value` / `resolve` / `promote` / `newline` / `candidates`) plus
  `CommaEolView` / `CommaLeadingView`, which hold all the
  get / set / del / iterate logic once. A future change to comma-value
  comment behaviour lands here.
- **`_array_comments.py`** — the int-keyed `Array` adapter
  (`_ArrayAdapter`) plus one-line `ArrayEolView` / `ArrayLeadingView`
  subclasses of the generic views (`Array.comments`,
  `Array.leading_comments`).
- **`_inline_comments.py`** — the str-keyed inline-**table** adapter
  (`_InlineAdapter`, leaf-key → entry resolution via `_inline_ops`)
  plus `InlineEolView` / `InlineLeadingView`. Backs
  `Table.comments` / `leading_comments` when the table is inline;
  setting a comment auto-promotes a single-line table to multi-line
  (TOML 1.1), and a detached `Table.inline()` factory raises until it
  is attached.

### Public API

- **`_public.py`** — top-level `loads` / `load` / `dumps` / `dump`.
  `load` / `dump` require **binary** file objects (`IO[bytes]`);
  text mode would silently translate newlines on Windows and
  break round-tripping.
- **`__init__.py`** — re-exports the public API; keep `__all__`
  alphabetised.

When in doubt: a change that touches only one of these layers is
usually right; a change that has to touch all of them is usually
wrong.

### Invariants worth knowing

- **Comma-value boundaries** may span predecessor `trailing`,
  predecessor `post_comma_trivia`, and successor `leading`. EOL
  payload belongs to the left item; comment-containing above blocks
  belong to the right item; blank-only regions stay positional.
  `leading_block` hides exactly one structural row break.
- **`Boundary` transforms mutate with copy-on-write.** Call `copy()`
  before transforming a snapshot that remains a source. Reorders
  capture affected seams before moving items, compose each final
  seam completely, then restore it once. Keep reorder linear; do not
  mutate then recapture or deep-copy every trivia channel.
- **Slot-stream linked list** is the single source of physical
  ordering. Mutation primitives splice exactly one slot at a time
  and update `_prev` / `_next`. Never rebuild the list.
- **`SlotRef.local_key` is derived** from `(slot, container)` —
  never assigned, never stored. The property asserts the
  geometric invariant on every read; an out-of-place ref fails
  fast at the property boundary rather than corrupting an
  `_index` bucket.
- **`Container._index[k]`** is the in-order list of refs in
  `_refs` whose `local_key == k`. File mid-stream refs through
  `_file_ordered_ref`; after block reordering, update the contiguous
  region and its per-key projections through
  `_finish_region_permutation`.
- **`Container._body_tail`** ≡ "the most recent slot in `_refs`
  belonging to the body region" (KV with matching owner; or, for
  a header-bearing container with no body, the header itself).
  Maintained eagerly on every body-region append, recomputed by
  `_recompute_body_tail` on body-affecting deletes.
- **`Slot.owner_aot_entry`** lives on the base `Slot`, not on the
  subclasses. Use direct attribute access — never `getattr(slot,
  "owner_aot_entry", None)`.
- **`Slot._refs`** is the back-pointer list from a slot to every
  `SlotRef` that targets it. Bounded by path depth + 1. Maintained
  by `SlotRef.__init__` (registers) and `unfile_ref`
  (unregisters). AoT removal uses it to scrub refs in O(depth) per
  slot instead of O(siblings) per container — don't bypass it
  with ad-hoc walks of ancestor `_index` buckets.
- **`AoTEntry.entry_slots`** is a **membership** list, not a
  doc-ordered one: it records which slots belong to the entry (with
  its `[[a]]` header kept first because it is appended first), but
  its order is *not* the doc-stream order. Anything that needs the
  entry's doc-stream order or subtree tail must **derive** it from
  the linked stream (`_owned_slots_ordered`, `_parent_subtree_tail`,
  `_aot_append_position`), never read `entry_slots[-1]`. Clone
  bookkeeping, in turn, relies on append-order slices and
  `entry_slots[0]` being the header.
- **Container shape** is named explicitly by the `_Kind` enum in
  `_kind.py` and surfaced as `Container._kind`. The six kinds —
  `DOCUMENT`, `SECTION`, `IMPLICIT_SECTION`, `INLINE_ROOT`,
  `INLINE_FACTORY`, `INLINE_DOTTED_INNER` — pick out the
  combinations of `_inline` / `_value` / `_layout_root` /
  `_header_ref` that previously had to be re-derived at every
  call site. In particular, `INLINE_FACTORY` (a detached
  `Table.inline()` not yet assigned anywhere) and
  `INLINE_DOTTED_INNER` (the navigator view for the `a` in
  `{a.b = 1}`) share `_inline=True, _value=None` and differ only in
  `_layout_root`; dispatch on `_kind` rather than re-discovering
  the discriminator.

## Tests

- `tests/test_basic.py`, `test_spacing.py`, `test_edit_golden.py` —
  parser and writer regressions, including byte-exact round-trip
  fixtures.
- `tests/test_comments.py` — the comment manipulation API.
- `tests/test_compliance.py` — the official **`toml-test`** suite
  (vendored under `vendor/`). Do not edit fixtures there to make
  failures pass.
- `tests/test_dict_semantics.py` — pins the user-visible behaviours
  that come from `Table` actually being a `dict` subclass
  (`isinstance`, ``**t`` unpacking, identity stability of lookups).
- `tests/test_toml11.py` — TOML 1.1-specific coverage.
- `tests/test_fuzz_roundtrip.py` — property-based round-trip tests
  (Hypothesis). If you break round-tripping, this will usually catch
  it; add new strategies here when you add a new construct.
- `tests/test_fuzz_parser.py` — bytes-level grammar fuzzer that feeds the
  parser arbitrary / near-valid input and asserts it either raises
  `TOMLParseError` or accepts and round-trips byte-exactly. Marked
  `slow`, so it is only picked up by `pytest -m slow` (`make fuzz`).
- `tests/test_fuzz_mutation.py` — mutation fuzzer over the vendored
  `toml-test` corpus, and from an empty document: runs random edit
  programs (set / delete / overwrite / sort / array + AoT ops,
  including building fresh `[section]` / `[[AoT]]` structure from
  scratch) over each parsed document, and once starting from empty,
  asserting the model stays self-consistent — valid TOML out, a
  dump→load→dump fixed point, and (the important oracle)
  `tomli.loads(dumps(doc))` matching `doc.to_dict()`, which catches a
  mutation that places a slot where a re-parse attributes it to a
  different owner than the logical view says. Draws **fresh random
  seeds** each run and reports the failing seed for reproduction.
  Marked `slow`; skips if the corpus is not vendored.
- `tests/test_mutation.py` — the dict/list mutation API.
- `tests/test_format.py` — the `Container.format()` /
  `Array.format()` canonicaliser.
- `tests/test_live_attach.py` — live-attach semantics for
  `Table.inline`, `Array`, and `AoT` when assigned into a document.
- `tests/test_synthesise_and_io.py` — value synthesis and binary I/O.
- `tests/test_scanner.py` — pins the cursor + diagnostics contract
  on `_Scanner` that the higher-level `scan_*` helpers build on.
- `tests/_helpers.py` — shared test helpers: `td(""" … """)` for writing
  TOML fixtures as indented triple-quoted literals (prefer it over walls
  of `\n`-escaped strings in new tests), and `reparses(src)` for the
  re-parse sanity check via `tomli` (used unconditionally because, as of
  writing, stdlib `tomllib` is TOML 1.0 only whereas `tomli` 2.4+ accepts
  TOML 1.1 syntax).

When adding behaviour, add a focused unit test in the relevant file
**and** consider whether the property tests should grow.

### Test-writing conventions

- **Assert on the full rendered document, not substrings.** Compare
  `tomlrt.dumps(doc)` to the complete expected output with `==`.
  Substring checks (`"foo" in out`, `"\n[bar]\n" not in out`) and
  dict-membership checks (`doc["a"]["b"] == 1`) silently miss the
  whitespace, comment, and trivia regressions the format-preserving
  invariant exists to catch. The few exceptions are pure error-path
  tests (`pytest.raises(...)`) that don't render anything.
- **Use `td(""" … """)` for both input fixtures and expected output.**
  Indented triple-quoted literals read top-to-bottom in TOML
  syntax; walls of `\n`-escaped strings hide structure and make
  byte-level diffs unreadable when an assertion fails. Reach for
  `\n`-escaped strings only for very short single-line fixtures
  where the noise of `td()` would dominate.
- **Don't reference line numbers in test docstrings or comments.**
  Source line numbers shift on every refactor; an explanatory note
  that names the function and the behaviour it pins ages well, a
  line-number reference doesn't.

## Documentation

User-facing prose docs live under `docs/` and are published at
<https://dimbleby.github.io/tomlrt/>. The site is built with
[Zensical](https://zensical.org/) (a static site generator from the
creators of Material for MkDocs) plus the `mkdocstrings` Python
handler. Site config lives in `zensical.toml` (the nav lives there
too — update it when you add or rename a page). The dependency group
is `docs`:

```bash
uv run --group docs zensical serve     # preview locally
uv run --group docs zensical build     # what CI runs
```

The API reference page (`docs/api.md`) is generated from docstrings
via `mkdocstrings`, so docstring changes flow through automatically.
The task-oriented pages (`quickstart.md`, `building.md`, `reading.md`,
`editing.md`, `comments.md`, `layout.md`, `errors.md`) are
hand-written — update them when you add, rename, or change behaviour
of any public API.

## Things to avoid

- Adding an unconditional runtime dependency.
- Reaching into `_slots` / the doc-stream linked list from
  user-facing code instead of going through `_layout_ops` /
  `_inline_ops`.
- Storing data on a `SlotRef` other than `slot` and `container` —
  `local_key` is derived; if you need another piece of state,
  derive it too or push it onto the slot itself.
- Adding a fourth ad-hoc reverse-walk of `c._refs` instead of
  expressing the predicate to `_last_kv`.
- "Fixing" formatting differences in the writer's output without
  adding a round-trip test that proves it.
- Touching `vendor/` (it is third-party, vendored verbatim).
- Editing `uv.lock` by hand — let `uv` regenerate it.
- Bumping action versions in `.github/workflows/*.yml` to a tag instead
  of a 40-char commit SHA. The workflows are **`zizmor` clean** and
  must stay that way (`uv tool run zizmor .`).

## Commit conventions

- Subject line: imperative mood, ≤ ~70 chars, no trailing period.
- Body: wrap around 72 chars; explain *why* not *what*. Bullets are
  fine.
- One logical change per commit. Keep mechanical reformat passes
  separate from substantive changes.
- Append a `Co-authored-by` trailer when an AI agent did the work, e.g.
  `Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>`.
