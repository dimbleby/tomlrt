# Quickstart

## Parse, edit, dump

```python
import tomlrt

with open("pyproject.toml", "rb") as f:
    doc = tomlrt.load(f)

doc["project"]["version"] = "0.2.0"
doc["project"]["dependencies"].append("requests>=2")

with open("pyproject.toml", "wb") as f:
    tomlrt.dump(doc, f)
```

For in-memory round-trips, use `loads` / `dumps`:

```python
doc = tomlrt.loads(text)
text_again = tomlrt.dumps(doc)
assert text == text_again
```

`tomlrt.dumps(doc)` and `doc.render()` are equivalent.

Both `dumps` and `dump` accept a plain mapping, wrapping it in a `Document`
for you, so `tomlrt.dumps({"a": 1})` works.

## Reading values

A `Document` behaves like a `dict`; nested tables are `Table` (also a `dict`
subclass), inline arrays are `Array` (a `list` subclass), and arrays-of-tables
are `AoT` (a `list` of `Table`).
Plain reads with `doc["key"]` work as you'd expect — see [Reading
documents](reading.md) when you want typechecker-friendly traversal.

## Writing values

| Assigning                             | Becomes                      |
| ------------------------------------- | ---------------------------- |
| `str`/`int`/`bool`/`float`/`datetime` | a TOML scalar                |
| `dict`                                | an inline table (snapshot)   |
| `list`                                | an inline array (snapshot)   |
| `Table.section({})`                   | a live `[section]` block     |
| `Table.inline({})`                    | a live inline table          |
| `AoT([...])`                          | `[[array.of.tables]]` blocks |
| `Array([...], multiline=True)`        | a multi-line inline array    |

For the difference between snapshot and live containers, see [Editing
documents](editing.md#live-vs-snapshot).
