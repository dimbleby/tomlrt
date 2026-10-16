# Editing documents

Use ordinary dictionary and list operations to change values:

```python
doc["project"]["version"] = "0.2.0"
doc["project"]["dependencies"].append("requests>=2")
del doc["project"]["description"]
```

## Dotted paths

`doc["a.b"] = 1` always treats `"a.b"` as a _single literal key_.
To descend into nested tables, use `install` or `ensure_table`:

```python
doc.install("tool.poetry.version", "0.1.0")  # [tool.poetry] version = "..."
doc.install(("tool", "weird.key"), 1)        # [tool] "weird.key" = 1

ruff = doc.ensure_table("tool.ruff")         # creates [tool.ruff] if absent
ruff["line-length"] = 88
```

Both accept dotted paths or sequences of key names and create missing parent
tables.

- `install(path, value)` sets or replaces the value and returns it.
- `ensure_table(path)` returns an existing table or creates one. It does not
  overwrite a non-table value.

For intermediate tables along the path, `ensure_table` keeps their existing
form. So does `install` when setting a scalar or inline value. New intermediate
tables are inline inside inline tables, and sections elsewhere:

```python
doc = tomlrt.loads("tool = {}\n")
doc.install("tool.ruff.line-length", 88)
# tool = { ruff = { line-length = 88 } }
```

## Choosing tables and arrays

A plain `dict` value installs as an inline table; a plain `list` installs as an
inline array.
Use tomlrt's classes to choose how a value is written:

```python
from tomlrt import AoT, Array, Table

doc["tool"] = Table.section({"version": 1})      # [tool] section
doc["xy"]   = Table.inline({"x": 1, "y": 2})     # xy = { x = 1, y = 2 }
doc["pkgs"] = AoT([{"a": 1}, {"b": 2}])          # [[pkgs]] … [[pkgs]]
doc["tags"] = Array(["a", "b"], multiline=True)  # multi-line array
```

## Live vs snapshot

Assigning a newly created tomlrt table or array adds that object to the document.
Changes through your original reference then update the document:

```python
xs = Array([1, 2])
doc["xs"] = xs
xs.append(3)             # doc["xs"] is now [1, 2, 3]
assert doc["xs"] is xs
```

Plain dictionaries and lists are copied on assignment. Changing the original
afterward does not affect the document.

A tomlrt table or array that already belongs to a document is also copied,
retaining its comments and formatting. For example, `doc["b"] = doc["a"]`
creates an independent copy of `a`.

Assigning a whole `Document` adds its contents as a section:

```python
template = tomlrt.loads(template_text)   # a standalone file, no [header]
doc["tool"]["bumpversion"] = template    # becomes [tool.bumpversion]
```

The source document's [preamble and epilogue](comments.md#document-preamble-and-epilogue)
are not copied.

## Removing values

Removing a `Table`, `Array`, or `AoT` — via `del`, `pop`, `clear`, or overwrite
— leaves any references you kept usable, but no longer connected to the document:

```python
old = doc.pop("tool")
old["debug"] = True          # does NOT affect doc
```

Assign it elsewhere to add it to a document again.

## Arrays-of-tables

An `AoT` represents repeated `[[table]]` blocks. Use list operations to insert,
replace, remove, or reorder its tables. `add()` appends a table and returns it:

```python
pkgs = doc.aot("packages")
entry = pkgs.add({"name": "foo"})
entry["version"] = "1.0"
```

An array-of-tables with no entries has no `[[key]]` syntax, so it serialises as
`key = []` — which re-parses as an empty inline `Array`, not an `AoT`.

## Reshaping the layout

See [Layout](layout.md) to sort keys, change table and array layout, or format
spacing.
