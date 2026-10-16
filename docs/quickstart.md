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

Open files in binary mode (`rb` / `wb`) so that line endings are preserved.
For strings, use `loads` / `dumps`:

```python
doc = tomlrt.loads(text)
text_again = tomlrt.dumps(doc)
assert text == text_again
```

Both `dumps` and `dump` also accept plain dictionaries.

## Reading values

A document and its tables behave like dictionaries; arrays behave like lists.
Use ordinary indexing to read values:

```python
name = doc["project"]["name"]
dependencies = doc["project"]["dependencies"]
```

See [Reading documents](reading.md) for nested lookups and typed accessors.

## Writing values

Assign Python values to set or replace entries. A dictionary becomes an inline
table (`key = { ... }`); a list becomes an array (`key = [...]`).

For `[section]` and `[[array.of.tables]]` blocks, see [Building
documents](building.md) and [Editing documents](editing.md).
