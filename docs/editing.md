# Editing documents

## Dotted paths

`doc["a.b"] = 1` always treats `"a.b"` as a _single literal key_.
To descend into nested tables, pass a dotted string (split on `.`) or any
sequence of literal segments to `install` or `ensure_table`:

```python
doc.install("tool.poetry.version", "0.1.0")  # [tool.poetry] version = "..."
doc.install(("tool", "weird.key"), 1)        # [tool] "weird.key" = 1

ruff = doc.ensure_table("tool.ruff")         # creates [tool.ruff] if absent
ruff["line-length"] = 88
```

Both create any missing intermediate tables on the way down.
They differ at the leaf:

- `install(path, value)` writes `value` at `path`, replacing whatever was there.
  It returns the freshly-installed live view (or the leaf value).
- `ensure_table(path)` is idempotent: it returns the existing table at `path` if
  there is one, or creates an empty one if not.
  It never overwrites an existing value.

## Structural assignment

A plain `dict` value installs as an inline table; a plain `list` installs as an
inline array.
To pick a different shape, assign a flavoured value:

```python
from tomlrt import AoT, Array, Table

doc["tool"] = Table.section({"version": 1})      # [tool] section
doc["xy"]   = Table.inline({"x": 1, "y": 2})     # xy = { x = 1, y = 2 }
doc["pkgs"] = AoT([{"a": 1}, {"b": 2}])          # [[pkgs]] … [[pkgs]]
doc["tags"] = Array(["a", "b"], multiline=True)  # multi-line array
```

## Live vs snapshot

Assigning a fresh `Table.section(...)`, `Table.inline(...)`, `Array(...)`, or
`AoT(...)` _attaches it live_: your reference becomes the live view at the
destination, and later mutations through that reference show up in the document.

```python
xs = Array([1, 2])
doc["xs"] = xs
xs.append(3)             # doc["xs"] is now [1, 2, 3]
assert doc["xs"] is xs

t = Table.section()
doc["a"] = t
t["x"] = 1               # doc["a"] is now {"x": 1}
assert doc["a"] is t
```

Plain `dict` / `list` values are _snapshot_ on assignment — mutating the
original after assignment does _not_ affect the document.
Reach for `Table.section`, `Table.inline`, or `Array` when you want live
semantics.

A container that is already attached somewhere is deep-cloned on assignment, so
two slots never share state.
This applies whether the source and destination are in the same document
(`doc["b"] = doc["a"]`) or different ones (`d2["x"] = d1["x"]`).
The clone is byte-faithful: comments, whitespace, and string / number style on
the bytes you didn't touch survive the move.

Assigning a whole parsed `Document` as a value lifts its body into a section,
preserving the comments and layout that were in the source:

```python
template = tomlrt.loads(template_text)   # a standalone file, no [header]
doc["tool"]["bumpversion"] = template    # becomes [tool.bumpversion], trivia intact
```

The document's own file-level preamble / epilogue (comment blocks separated from
the body by a blank line) belong to no key and are not carried across.

## Removal and orphaning

Removing a `Table`, `Array`, or `AoT` — via `del`, `pop`, `clear`, or overwrite
— detaches the view.
It keeps its data, but further mutations no longer reach the document:

```python
old = doc.pop("tool")        # detached Table view
old["debug"] = True          # does NOT affect doc
```

## Growing an array-of-tables

`AoT.add()` appends a fresh entry and returns the new `Table` view, so you can
keep mutating it:

```python
pkgs = doc.aot("packages")
entry = pkgs.add({"name": "foo"})
entry["version"] = "1.0"
```

## Sorting child keys

`Document.sort()` and `Table.sort()` reorder direct child keys in
place. The signature mirrors `list.sort`: keyword-only `key` and
`reverse`. Comments and blank lines travel with their keys.

```python
doc = tomlrt.loads("""
    # the name
    name = "tomlrt"
    # the version
    version = "0.1"
    # the author
    author = "me"
""")
doc.sort()
```

When a document mixes bare keys and `[section]` / `[[aot]]` headers,
`.sort()` keeps the headers after the bare keys — any bare key
emitted after a section header would otherwise be re-parsed as a
member of that section.

```python
doc = tomlrt.loads("""
    [a]
    x = 1
    [b]
    y = 2
""")
doc["nickname"] = "hi"
doc.sort()
# nickname, [a], [b]
```

- `key` and `reverse` apply within each partition (bare keys, then
  sections); they do not interleave the two.
- Dotted keys (`a.x = 1`) are bare keys for sorting purposes.

## Empty arrays-of-tables

TOML has no syntax for an array-of-tables with zero entries, so an empty `AoT`
does not appear in `dumps` output — the key is silently absent.
The in-memory `AoT` remains usable: append entries to it and they will reappear
in the next dump.

## Inline-array layout

`Array.multiline` flips between single- and multi-line layout in place.
For multi-line layout with a custom indent, call `set_multiline`:

```python
arr = doc.array("tags")
arr.set_multiline(multiline=True, indent="  ")
```

Collapsing a multi-line array to single-line is rejected if any item carries a
comment; clear them first (see [Comments](comments.md)).

## Promoting inline → section

If a value started life as an inline table or inline array of inline tables, you
can promote it in place:

```python
doc = tomlrt.loads('[tool]\nruff = { line-length = 88 }\n')
doc.table("tool").promote_inline("ruff")       # → [tool.ruff]

doc = tomlrt.loads('pkgs = [{a = 1}, {b = 2}]\n')
doc.promote_array("pkgs")                      # → [[pkgs]] … [[pkgs]]
```

Promotion is rejected if it would lose inner comments; clear them first (see
[Comments](comments.md)).

## Canonical formatting

Whenever you want to drop format preservation and snap a section, document, or
array to a canonical layout, call `format()`:

```python
doc.format()           # whole document
doc.table("tool").format()
doc.array("pkgs").format()
```

See [Formatting](formatting.md) for what `format()` rewrites and what it
leaves alone.
