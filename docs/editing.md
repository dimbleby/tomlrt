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

## Arrays-of-tables

`AoT.add()` appends a fresh entry and returns the new `Table` view, so you can
keep mutating it:

```python
pkgs = doc.aot("packages")
entry = pkgs.add({"name": "foo"})
entry["version"] = "1.0"
```

An array-of-tables with no entries has no `[[key]]` syntax, so it serialises as
`key = []` — which re-parses as an empty inline `Array`, not an `AoT`.

## Reshaping the layout

Editing changes a document's _data_. To reshape its _layout_ — sort
keys, switch inline values between single- and multi-line, promote an
inline value to a section, or snap a subtree to a canonical format — see
[Layout](layout.md).
