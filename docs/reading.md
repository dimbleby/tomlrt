# Reading documents

Plain `doc["key"]` returns `Any`. This page covers the richer read API:
shape-checked typed accessors, dotted-path navigation, untyped path lookups,
and conversion back to plain Python.

## Typed accessors

The typed accessors give you a shape-checked `Table`, `Array`, or `AoT`
directly.

### Strict accessors

Each raises `KeyError` if the key is missing and `TypeError` if the value at the
key has the wrong shape:

```python
project = doc.table("project")            # -> Table
deps    = project.array("dependencies")   # -> Array
pkgs    = doc.aot("packages")             # -> AoT
```

The same accessors exist on `Array` for indexed lookup:

```python
first_pkg = pkgs.table(0)             # -> Table
nested    = some_array.array(2)       # -> Array
```

### Lenient accessors

`get_table` / `get_array` / `get_aot` mirror `dict.get`: return the value when
the shape matches, otherwise the `default` (default `None`).
They raise `TypeError` only if the key exists but has the _wrong_ shape:

```python
ruff = doc.get_table("tool", default={}).get("ruff")
```

## Dotted paths and sequences of segments

The `Table`-side accessors (`table`, `array`, `aot`, `entry`, and their `get_*`
variants) accept either a dotted path string or a sequence of literal segments.
Use the sequence form when a segment must contain a literal `.`:

```python
doc.table("tool.poetry")            # tool -> poetry
doc.table(("tool", "weird.key"))    # tool -> "weird.key"
```

## Untyped path access

Sometimes you just want the value at a path without asserting a shape — e.g.
when dispatching on the result yourself, or when the leaf is a scalar.
`entry` and `get_entry` walk the same path the typed accessors do but return
`Any`:

```python
value = doc.entry("tool.poetry.name")           # raises if missing
maybe = doc.get_entry("tool.poetry.licence")    # None if missing
```

## Copying views

Use `copy.copy(view)` or `copy.deepcopy(view)` for an independent copy that
retains comments and available formatting. Both operations are deep for tomlrt
views. Inherited methods such as `table.copy()` and `array.copy()` still make
ordinary shallow `dict` and `list` copies.

## Back to plain Python

`Table.to_dict()`, `Array.to_list()`, and `AoT.to_list()` return a deep copy
made of plain `dict` / `list` / scalar values, sharing no state with the
document. Use one when you want to mutate the result without affecting the
document, or to hand data off to code that will outlive the document:

```python
plain = doc.to_dict()
plain["project"]["name"] = "renamed"   # does not touch the document
```

Plain exports deliberately discard formatting and are usually cheaper than
copying an existing layout.
