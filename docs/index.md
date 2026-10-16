# tomlrt

A format-preserving TOML reader and writer for Python.

Parse a document, edit it, dump it, and the bytes you didn't touch round-trip
exactly — comments, whitespace, string style, and number formatting all intact.

Supports TOML 1.1.

```python
import tomlrt

with open("pyproject.toml", "rb") as f:
    doc = tomlrt.load(f)

doc["project"]["version"] = "0.2.0"
doc["project"]["dependencies"].append("requests>=2")

print(tomlrt.dumps(doc))   # comments and layout are preserved
```

## Where next

- [Quickstart](quickstart.md) — parse, edit, dump.
- [Reading documents](reading.md) — look up values and convert to plain Python.
- [Building documents](building.md) — construct a `Document` from scratch or
  from a `dict`, and choose table and array forms.
- [Editing documents](editing.md) — set, replace, remove, and copy values.
- [Comments](comments.md) — read and write comments.
- [Layout](layout.md) — sort keys, arrange tables and arrays, and format spacing.
- [Errors](errors.md) — exception types.
- [API reference](api.md) — classes, methods, and options.
