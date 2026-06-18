# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- The `indent` argument of `Array(...)`, `Array.set_multiline`, and
  `Table.set_multiline` is now an `int` (number of spaces) instead of a
  string, so callers can no longer pass arbitrary indent text.
- Invalid-argument errors now use the standard built-in exceptions
  instead of `TOMLError`: an empty or malformed key path, and a comment
  string containing a newline or other control character, now raise
  `ValueError`; `promote_inline` / `promote_array` raise `TypeError`
  when the target value is the wrong kind. `TOMLError` is reserved for
  tomlrt-domain conditions.

### Removed

- The deprecated `tomlrt.document()` and `tomlrt.parse()` aliases. Use
  `Document()` and `loads()` instead.

### Fixed

- Key-path lookups now raise on an empty path (e.g. `""`, `[]`,
  `"a..b"`) instead of silently returning the whole container.

## [1.8.3] - 2026-06-17

### Fixed

- `format()` no longer drops a comment placed before a multi-line array's
  closing bracket.
- Fixed `Document.preamble` dropping blank lines: reading and re-assigning it
  no longer collapses blank-separated comment groups. `preamble` is now the
  opening comment paragraph (up to the first blank line); comments below that
  blank belong to the first key or section.
- `Document.epilogue` now preserves blank lines between trailing comment
  groups: it is a `tuple[str | None, ...]` (with `None` per blank line), so
  reading and re-assigning it no longer collapses blank-separated groups.
- Sorting a multi-line inline array or table after deleting one of its
  end-of-line comments no longer detaches another item's comment.
- A multi-line inline array's layout is now derived consistently from its
  rendered shape (like inline tables): emptying and refilling one no longer
  leaves a half-collapsed layout, and appends in a CRLF document no longer
  emit a stray LF.
- `Array.extend()` now computes its layout style once instead of per item,
  making a bulk extend linear rather than quadratic in the array's length.
- Multi-line detection for inline arrays and tables is now memoised, so
  building one with repeated `append` / `update` / `t[k] = v` is linear
  rather than quadratic in the number of items.
- Adding keys to a section table is now linear rather than quadratic, so
  building a large table key-by-key is much faster.

## [1.8.2] - 2026-06-14

### Fixed

- An empty array-of-tables now renders as an empty array (`key = []`) instead of
  being omitted, so a dumped document matches its dict view.

## [1.8.1] - 2026-06-14

### Fixed

- `Array.sort()` no longer drops or collapses leading comments when another
  item in the array carries an end-of-line comment (#185).

## [1.8.0] - 2026-06-13

### Added

- Inline tables now support the comment-manipulation API (`comments`,
  `leading_comments`) and multi-line layout control (`multiline`,
  `set_multiline`), matching inline arrays. Setting a comment promotes a
  single-line inline table to multi-line (TOML 1.1).

### Fixed

- Comment editing on multi-line inline arrays and tables now lays out
  irregular layouts more cleanly: a displaced item is re-indented, deleting an
  end-of-line comment keeps an adjacent blank line, and a comment block above
  unindented items lines up at column zero.
- Trailing whitespace after a comma no longer masks the row-terminating
  newline, so editing or restructuring such a row no longer leaves a stray
  space-only blank line.
- Structural edits (append, insert, delete, sort) on multi-line inline arrays
  and tables now preserve deliberate blank lines elsewhere in the value instead
  of collapsing them.
- Deleting the predecessor of a shared-row item in a multi-line inline array
  or table now re-indents the surviving follower to the canonical row indent
  instead of leaving its one-space inline separator in place.


## [1.7.6] - 2026-06-13

### Fixed

- Deleting the only content of an implicit table no longer makes the emptied
  table vanish from the output.
- Synthesising a section header no longer captures a trailing sibling key under
  the wrong table.
- Filling a previously-empty inline table now pads the braces (`{ a = 1 }`) to
  match synthesis and `.format()`.
- Assigning a parsed document or table into another document now keeps its
  comments and formatting, instead of dropping them.
- Overwriting a section that has array-of-tables children now keeps those
  children under the section, instead of moving them to the end of the
  document.
- Re-attaching a section that was removed (via `del`, `pop`, or overwrite) now
  keeps its comments and formatting, instead of dropping them.

## [1.7.5] - 2026-06-11

### Fixed

- Overwriting a key with a body-less section (one whose only child is an
  array-of-tables) no longer wipes the whole document on dump.
- Overwriting a section or array-of-tables and then re-using an entry read from
  the displaced value no longer silently drops the entry's body.

## [1.7.4] - 2026-06-09

### Fixed

- Sorting an array-of-tables entry that has a nested array-of-tables child no
  longer leaves the rendered order out of sync with the logical key order.
- Formatting fixes when dealing with arrays whose elements use leading rather
  than trailing commas

## [1.7.3] - 2026-06-07

### Fixed

- Various in-place edits — `sort`, overwriting a key, and appending, inserting,
  reversing or sorting array-of-tables entries — could place a value where
  re-reading the file would attribute it to the wrong table, silently corrupting
  the document or desyncing it from the rendered output.
  These cases (covering nested arrays-of-tables, dotted keys, inline tables, and
  out-of-order or auto-promoted section headers) are fixed.
- Replacing an out-of-order subsection (e.g.
  `[foo.bar]` defined before `[foo]`) with a dict / inline value no longer emits
  the new binding outside its parent section.
- `Array.extend(arr)` / `AoT.extend(aot)` (and the corresponding `+=` forms) no
  longer hang when extending a sequence with itself.
- A failed out-of-range `Array.comments[i] = ...` no longer silently promotes a
  single-line array to multi-line form.

## [1.7.2] - 2026-06-01

### Fixed

- Assorted mutation, comment, and scanner edge-case fixes.

## [1.7.1] - 2026-05-31

### Fixed

- Appending to, or deleting the tail of, a multi-line inline value whose last
  item carries an EOL comment no longer mangles the closing bracket layout.
- Reordering or deleting the last entry of a multi-line inline table no longer
  swaps EOL comments between entries.
- Assigning `Array.multiline` to its current value is now a no-op.
- `Array.format` / `Container.format` and `Array.set_multiline(True)` no longer
  drop a row-attached comment on the opening `[` / `{` line.
- `Array.set_multiline(True)` on an empty array no longer drops a bracket-line
  comment and now aligns the closing `]` to the outer indent.
- `Table.format()` on an AoT entry no longer reformats sibling entries of the
  same array.
- `format()` on a multi-line inline value no longer inserts a spurious blank
  line after an item that carries an EOL comment.
- Deleting the first KV, section, or AoT entry no longer drops the document
  preamble.
- Deleting all items from a multi-line inline array or inline table no longer
  leaves a stray indent line.
- Slice-assigning to a multi-line inline array no longer leaves a stray indent
  inside the brackets when emptied, drops bracket-EOL comments, or wipes
  above-block comments on surviving items.
- Adding an entry to an empty multi-line inline table now restores the canonical
  indent and trailing comma.
- Emptying then re-adding to a bracketed value with a bracket-EOL comment no
  longer drops the comment or leaves a blank line.
- Inline-table mutation (append / delete / sort) no longer drops above-bracket
  and above-entry comment blocks; `Array.sort()` and `Array.reverse()` now
  preserve per-position indents and above-`]` comments rather than
  wholesale-restamping them.
- `Array.leading_comments[i] = ()` no longer silently promotes a single-line
  array to multi-line (the empty assignment is a semantic delete-if-present).

## [1.7.0] - 2026-05-25

### Added

- `Container.has_header(key)` predicate: does this child render with a
  `[header]` line.

### Changed

- `Table.header_comment` and `Table.header_leading_comments` getters return
  `None` / `()` on headerless containers (implicit sections, document root)
  instead of raising.

### Fixed

- `Container.sort()` no longer fabricates a leading blank line when a nested AoT
  child appeared before its explicit parent header in source.
- `format()` no longer strips the indent of full-line comments inside multi-line
  inline arrays / inline tables.
- `Array.set_multiline(multiline=True, ...)` no longer drops the last element's
  EOL comment when synthesising a trailing comma, and no longer inserts a blank
  line before the closing bracket when the last item carries an EOL comment.
- `Document.format()` now canonicalises the document epilogue as well as the
  body.

## [1.6.1] - 2026-05-25

### Changed

- `Container.sort` now keeps sections and arrays-of-tables after bare keys
  instead of raising `ValueError`.

## [1.6.0] - 2026-05-25

### Added

- `Container.format(*, comments: bool = True)` and `Array.format(*, comments:
bool = True)` reformat a section, document, table, or inline array in place.
  See [docs/formatting.md](docs/formatting.md) for details.

## [1.5.0] - 2026-05-24

### Added

- `Container.sort(*, key=None, reverse=False)` reorders a section, document,
  table, or inline table's direct child keys in place, preserving per-key
  trivia.
  Mirrors `list.sort` / `AoT.sort`.
- `Container.leading_block` and `Table.header_leading_block` expose the full
  leading-trivia region (including blank-line-separated "orphan" comments) as a
  tuple of `str | None`, where `None` is a blank line.
- `Table.is_inline` property to distinguish inline `{...}` tables from
  `[section]` blocks when walking a parsed document.

### Fixed

- Mutating `comments`, `leading_comments`, or `leading_block` on a detached
  `Table.section()` / `Table.inline()` now raises a clear `TOMLError` pointing
  at the attach-first workaround instead of a misleading `KeyError("key '...'
not in container")`.
- `Container.leading_block` and `Table.header_leading_block` no longer overlap
  with `Document.preamble` at the document head slot.
  Round-tripping or reordering sections no longer migrates the preamble into a
  section body.
- Per-key clone of a sub-table that came from dotted-key form (e.g.
  `dst["x"]["v"] = src["x"]["v"]` where the source was `[x]\nv.w = 1`) now
  preserves the dotted form on the destination instead of silently promoting it to
  an explicit `[x.v]` header.
- Cross-document `dst["k"] = src["k"]` now preserves the source section header's
  indent even when no leading comment is attached.
- Multi-component `install("a.b", value)` where the implicit parent must be
  synthesised now preserves source header trivia (leading comments) on the
  installed child, by routing through the standard `__setitem__` clone dispatch
  rather than the synthesis fallback.
  Same fix applies to cross-document `dst["a"] = src["a"]` when the source has
  only implicit parents.
- Cross-document `dst[k] = src[k]`, `aot.append(entry)` and `aot[i] = entry` now
  preserve nested arrays-of-tables inside the cloned subtree.
- Cross-document `dst[k] = src[k]` now preserves the source table header's
  leading comment block.
- Adding a sub-section to an empty placeholder section no longer clears
  `Document.preamble`.
- `Array.sort` / `Array.reverse` now keep per-item leading comments with their
  items and stop the closing `]` from gluing to the new last item.
- Cross-document `dst[k] = src_section` under an empty placeholder parent now
  demotes the parent to an implicit super-table instead of leaving a stray bare
  `[parent]` header line.
- Assigning a section table or `AoT` as a value of an inline table now
  consistently raises `TOMLError` with a clear message, including for detached
  `Table.inline()` factories (previously: silently accepted, then
  `NotImplementedError` at attach time).
- Comments on dotted-key entries are now reachable through the dotted-parent
  container (`project["urls"].comments["homepage"]`).
- `Array.sort()` / `Array.reverse()` no longer leave the new item 0 carrying the
  old position-0 indent in multi-line arrays.
- Cross-document `dst[k] = src[k]` and `AoT.sort()` no longer drag the source
  document's preamble onto the destination or surviving entries.
- `Array.leading_comments` del/pop/clear at non-zero indices, and insert/append
  next to items carrying EOL comments, no longer misattribute or duplicate
  adjacent comments on re-render.

## [1.4.3] - 2026-05-24

### Fixed

- Overwriting an implicit table with a scalar or inline value now preserves the
  original position in the document instead of moving the binding to the end.
- Mutations to inline tables and arrays nested inside an `Array` are now
  rendered.
- `Array.set_multiline` no longer inserts `\n` newlines into a CRLF document.
- `Array.__imul__` (`arr *= n`) now keeps replicated inline tables and arrays
  live: mutations through the copies render.
- `Document.install` / `Document.ensure_table` now accept any `Sequence[str]`
  (e.g.
  `collections.deque`), matching the documented contract and the public type
  signature.
- Detached `AoT.append` / `AoT.insert` / `AoT[idx] = ...` now accept any
  `Mapping` (e.g.
  `MappingProxyType`), matching their declared type signature and the attached
  paths.
- `Document(data=...)` no longer coerces a user-supplied `Array` of mappings
  into an `AoT`; the caller's explicit inline-array choice is preserved.
- `Table.section(mapping)` and `AoT(entries)` now reject non-string keys at
  construction with a clear `TypeError`, matching `Table.inline` and
  `Document(data=...)`.
  Attached `AoT.append` / `.add` / `.insert` and inline-table synthesis from a
  plain `dict` now produce the same unified `"TOML keys must be str"` error
  instead of crashing deep in the layout pipeline.
- `copy.copy` / `copy.deepcopy` of an inline `Table` now returns an inline
  `Table` instead of silently converting to a section table.

## [1.4.2] - 2026-05-10

### Fixed

- Live-attaching an `Array` into a document with a different line ending no
  longer produces mixed `\n` / `\r\n` newlines.

## [1.4.1] - 2026-05-09

### Fixed

- Mutating an implicit table whose only descendants were created via chained
  `ensure_table` / nested AoT attaches no longer trips an internal
  anchor-not-found assertion.

## [1.4.0] - 2026-05-08

- Rewrite internals for improved performance

## [1.3.2] - 2026-05-04

### Changed

- Faster `install` on deep dotted paths in large documents.
- `Table.clear` no longer scales quadratically with document size.
- Faster delete of sub-tables and AoT children in large documents.
- `AoT.clear` and bulk `del aot[:]` no longer scale quadratically.

## [1.3.1] - 2026-05-03

### Fixed

- Adding a dotted key to a section now respects the section's indent and
  blank-line policy, and inserts a separating newline when needed.
- Reject float literals with a misplaced underscore between the exponent sign
  and its digits (e.g.
  `1e+_1`).

## [1.3.0] - 2026-05-02

### Added

- `Table.entry` / `Table.get_entry` for untyped path access.

### Changed

- The typed accessors (`table`, `array`, `aot`, and their `get_*` variants) now
  accept a `Sequence[str]` as well as a dotted string.

## [1.2.0] - 2026-05-02

### Changed

- `Table.install` and `Table.ensure_table` now accept any `Sequence[str]`
  for the key path, not just `tuple[str, ...]`.

### Deprecated

- `tomlrt.parse()`; use `tomlrt.loads()` instead.

## [1.1.0] - 2026-04-30

### Added

- `Document(mapping=None)` public constructor.

### Deprecated

- `tomlrt.document(...)`; use `tomlrt.Document(...)` instead.

## [1.0.3] - 2026-04-30

### Changed

- Faster parsing, especially on large documents with many arrays-of-tables.

## [1.0.2] - 2026-04-27

### Changed

- Documentation pass and small typing tightenings across the public API.
  The `Document.cst` escape hatch is gone.

## [1.0.1] - 2026-04-26

### Fixed

- Synthesising parent headers and installing sections no longer mix blank-line
  and compact styles.
- Replacing an AoT entry no longer injects a stray blank between the new entry
  and its surviving sibling in compact documents.
- Newly appended/inserted AoT entries now adopt the indent style of their
  siblings' KV lines instead of rendering flush-left.

## [1.0.0] - 2026-04-26

### Added

- Public `TomlInput` type alias for typing helpers that build or mutate document
  fragments.

### Changed

- `AoT.sort()` now requires the `key=` argument, since `Table` entries are not
  orderable.

### Removed

- The legacy `SectionSpec` tag type.

### Fixed

- `Table.inline()` now renders with spaced braces (`{ k = v }`), matching the
  style produced when assigning a plain `dict`.

## [0.6.0] - 2026-04-26

- Many bug fixes

## [0.5.0] - 2026-04-25

### Fixed

- A wide range of round-trip, comment-preservation, and structural mutation bugs
  across tables, arrays-of-tables, and multi-line arrays.

## [0.4.0] - 2026-04-23

### Fixed

- A broad sweep of correctness fixes in the mutation API, covering every flavour
  of structural change: assignment into and through array-of-tables,
  append/insert/pop on multi-line arrays with comments, attached-AoT
  installation, comment-trivia preservation across promotion and shifts, CRLF
  line-ending preservation, and copy/deepcopy of `Array` and `AoT` subviews.
  Several silent corruptions (CST and dict-side state diverging after a mutation)
  are gone, and a number of error messages are now more specific about which value
  was rejected and why.

## [0.3.0] - 2026-04-21

### Changed

- **Structural assignment is now driven by the value, not the method name.** The
  parallel `set_table` / `set_aot` / `set_array` methods have been removed in
  favour of a single assignment path:

  ```python
  doc[k] = Table.section({...})          # [k] standard section
  doc[k] = {...}                         # k = { ... } inline table
  doc[k] = AoT([{...}, {...}])           # [[k]] array of tables
  doc[k] = Array([...], multiline=True)  # multi-line array value
  ```

  `Table.section` is a classmethod factory returning the public tag type
  :class:`SectionSpec`.
  :class:`AoT` and :class:`Array` can now be constructed standalone and then
  assigned.

- **New `Table.install(path, value)`** accepts either a dotted `str` path or a
  `tuple[str, ...]` of literal segments.
  Tuples provide an escape for keys that legitimately contain a `.`::

        doc.install(("foo.bar",), 1)   # "foo.bar" = 1  (single segment)
        doc.install("foo.bar", 1)      # [foo]\nbar = 1 (dotted path)

  `ensure_table` also accepts both forms.

- `__setitem__` no longer splits `str` keys on `.`; a plain `str` is always
  treated as a single literal segment, matching the standard `dict` contract.
  Use `install()` for dotted-path placement.

### Removed

- `Table.set_table`, `Table.set_aot`, `Table.set_array`.
  Use the value-driven equivalents above, or `Table.install` for dotted paths /
  tuple keys.

### Fixed

- `AoT.insert(0, …)` now adds a blank-line separator between the newly inserted
  `[[..]]` entry and the existing one that follows it (matching sibling spacing,
  defaulting to blank-separated).
  The policy previously only looked at _preceding_ content, so inserting before
  existing entries glued two `[[..]]` headers together.
- The dict-style view of a parsed :class:`Document` no longer goes stale
  relative to :func:`dumps` after structural mutations.
  Assigning over an array-of-tables, deleting then re-binding a key, and `pop()`
  followed by re-assignment all kept showing the pre-mutation value while the
  rendered TOML reflected the new state.
  The cached per-table section scope that drove this has been replaced with
  on-demand derivation from the surrounding AoT entry (when there is one), so dict
  reads and `dumps` output are always consistent.
- Mutations on a sub-table reached via a dotted key from an ancestor section now
  work correctly.
  Given `poetry.name = "x"` written inside `[tool]`,
  `doc["tool"]["poetry"].pop("name")` and `doc["tool"]["poetry"]["name"] = "y"`
  previously raised `KeyError` or duplicated the key in a new section; both now
  edit the original entry in place.
- Setting :attr:`Document.preamble` on an empty document and then adding content
  now renders the preamble at the top of the file.
  It was previously parked in the document's trailing trivia and emitted _after_
  the new content (so `dumps` produced `x = 1\n# c\n` instead of `# c\n\nx =
1\n`); the comment also became invisible to the getter once content arrived.
  Migration now happens at the insertion site for any of `doc[k] = …`,
  :meth:`Table.install`, :meth:`AoT.insert`, or AoT assignment.
- :meth:`Table.promote_array` now carries the source inline-table KV's leading
  comments / blank lines onto the first new `[[..]]` header, and any trailing
  EOL comment onto the last new entry.
  The trivia was previously discarded outright, so promoting an inline array
  silently dropped any authoring comments around it.
- Import of `assert_never` no longer breaks on Python 3.10.
  The symbol is now sourced from `typing_extensions` on interpreters older than
  3.11, mirroring the existing `override` import.

## [0.2.0] - 2026-04-20

### Changed

- **`Table` is now a real `dict` subclass.** `isinstance(t, dict)` returns
  `True`, `**table` unpacking works, and any third-party API typed against
  `dict[str, Any]` / `isinstance(x, dict)` now accepts a `Table` directly.
  Reads go through `dict`'s native `__iter__` / `__getitem__` / `__len__` /
  `__contains__`; the CST is still the single source of truth for _layout_
  (whitespace, comments, key order, table-shape choices) and is kept in lock-step
  with the dict storage on every mutation.
  Held references behave like ordinary Python dict references: `del doc['foo']`
  orphans the held `Table` (data preserved, mutations no longer reach the
  document) and re-binding the path installs a fresh `Table` rather than
  re-attaching the old one.
  Identity is stable: `doc['foo'] is doc['foo']` and the same goes for nested
  children.
- `Table.pop` now returns the actual stored value (an orphaned `Table` / `AoT` /
  `Array` for container values) rather than a deep plain-Python snapshot.
  Use `Table.to_dict()` / `Array.to_list()` first if you need a snapshot.
- Detached tables and AoTs are now isolated from the original document.
  Structural mutations on a held container after its parent removed it
  (`set_table`, `set_aot`, `promote_inline`, `promote_array`, `AoT.add`,
  `AoT.append`, `AoT.insert` …) no longer leak back into the document by
  re-creating the removed sections.
- `AoT.pop` now returns the live entry object that was at the given index (then
  orphans it), mirroring `Table.pop` and preserving identity with whatever the
  caller previously read out of the AoT.
- `Table` now subclasses `MutableMapping[str, Any]` (was `MutableMapping[str,
TomlValue]`), and `Table.__getitem__` returns `Any` (was the strict `Scalar |
Array | AoT | Table` union).
  Symmetrically, `Array` now subclasses `list[Any]` and `Array.__getitem__` /
  `Array.pop` return `Any`.
  This matches what `tomllib.loads` returns (`dict[str, Any]`) and what `tomlkit`
  does, and lets chained subscripts like `doc["tool"]["poetry"]["name"]`
  type-check without `cast`.
  Consumers typed against `MutableMapping[str, Any]` or `list[Any]` (which is most
  of the ecosystem) now compose with `Table` / `Array` directly.
  The strict return type is still available through the `.table()` / `.array()` /
  `.aot()` accessors and their `get_*` counterparts when you want it.
- `Array.append` / `extend` / `insert` / `__setitem__` now type their input
  parameter as `object` instead of the narrower `TomlValue` alias, matching
  `Table.__setitem__` and the underlying `value_to_node` converter.
  At runtime they always accepted arbitrary Python values (plain `dict` -> inline
  table, plain `list` -> inline array); the annotations were lying.
- Synthesised inline arrays no longer carry padding spaces inside the brackets.
  `[1, 2, 3]` instead of `[ 1, 2, 3 ]`, and `[1]` instead of `[ 1 ]`.
  Inter-element spaces are unchanged.
  Inline tables (`{ a = 1, b = 2 }`) still keep their conventional inner spacing.
  Parsed arrays round-trip with their original spacing.
- Modest parse speedup: cache `Key.path` so the dotted-key tuple is built once
  per key, and pass the parent's already-scoped section list through to child
  `_StdTable` constructors so each child's initial population walks only its own
  subtree instead of the whole document.

### Added

- `Table.get_table(key, default=None)`, `Table.get_array(...)`,
  `Table.get_aot(...)` and the analogous `Array.get_table(index, ...)` /
  `Array.get_array(index, ...)` are typed-but-optional accessors.
  They mirror the strict `.table()` / `.array()` / `.aot()` accessors but return
  `default` (or `None`) when the key/index is missing, rather than raising.
  A wrong-type entry still raises :class:`TypeError`: missing is "no answer",
  wrong shape is a bug.
  Overloads preserve the type of a user-supplied default.
- `Table.to_dict()` / `Array.to_list()` / `AoT.to_list()` return a deep,
  plain-Python copy of the view, walking nested tomlrt views into real `dict` /
  `list` containers.
  Intended for the interop boundary with consumers that expect actual `dict`
  objects (`fastjsonschema`, `pydantic`, JSON encoders, code that does
  `isinstance(x, dict)`).
  Scalars are returned as-is; the result shares no mutable state with the
  document.
- `AoT.add(entry={})` appends `entry` and returns the new :class:`Table` view,
  sparing users the `aot.append(...); aot[-1]` two-step when they need a handle
  to the freshly-added entry for further population.
- `tomlrt.document(data=None)` returns a fresh :class:`Document`, optionally
  populated from a mapping.
  Without arguments, equivalent to `tomlrt.parse("")` but more discoverable for
  the "build a TOML file from scratch" use case.
  With a mapping, recursively walks the data: nested mappings become `[section]`
  blocks, lists of mappings become `[[array.of.tables]]` blocks, and leaf values
  use ordinary key-value assignment.
  The resulting document shares no mutable state with the input.
- `Array.set_multiline(*, multiline, indent="    ")` and the read/write
  `Array.multiline` property toggle an inline array between single-line and
  multi-line layout.
- `Table.set_aot(key, entries=())` creates an array-of-tables at `key`
  (overwriting any existing value) and returns the live view, so users can build
  `[[ ...
]]` sections without going through the inline-array path.
- `Table.set_table(key, value=())` creates a standard-table section at `key`,
  replacing any existing value.
  Accepts dotted paths (e.g.
  `"tool.poetry"`); intermediate tables are kept implicit so no empty `[tool]`
  super-table headers are emitted.
- `Table.ensure_table(key)` returns the table at `key`, creating an empty
  section if absent.
  Accepts dotted paths and walks through implicit super-tables.
- `Table.set_array(key, items=(), *, multiline=False, indent="    ")` creates an
  inline array at `key` (replacing any existing value), optionally laid out one
  item per line.
  Accepts dotted paths so a multiline array deep in the tree can be created in a
  single call.
- `Document.preamble` and `Document.epilogue` properties expose the comment
  block at the top and bottom of the document.
  They are blank-line-separated from any structural content (and from any
  "attached" leading comment of the first key), so writing one will not clobber
  the other or any per-key comment block.
- `Table.set_aot` now accepts dotted paths, mirroring `set_table`.
- `Table.table`, `Table.array` and `Table.aot` typed accessors now accept dotted
  paths for navigation through nested structures.
- `Table.promote_array(key)` converts an existing inline array of inline tables
  into an array-of-tables, mirroring the existing `Table.promote_inline` for
  tables.

### Fixed

- An empty array whose source contains a newline inside the brackets (`a =
[\n]`) now round-trips and accepts subsequent `append` calls while preserving
  its multi-line shape.
- `Table.set_aot` and `Table.promote_array` now lay their `[[ ...
]]` blocks out with blank-line separators between entries, and with a blank line
  between the block and any preceding content.
- Programmatically appending to an `AoT` (or appending the second entry into a
  freshly-built one) now blank-line-separates the new `[[ ...
]]` header from whatever precedes it in the document, matching round-trip output
  of equivalent parsed input.
  Previously, fresh AoTs and AoTs whose new entries followed an unrelated
  sub-section were rendered with the headers visually glued together.
  When existing entries clearly establish a no-blank-line style (≥ 2 sibling gaps
  to learn from), that style is still respected.

## [0.1.0] - 2026-04-20

Initial release.
