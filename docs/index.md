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
- [Building documents](building.md) — construct a `Document` from scratch or
  from a `dict`.
- [Reading documents](reading.md) — typed accessors, dotted paths, and
  conversion back to plain Python.
- [Editing documents](editing.md) — dotted-path mutation, structural
  assignment, live views, layout control.
- [Comments](comments.md) — the comment API.
- [Formatting](formatting.md) — opt-in canonical formatting for values,
  sections, or whole documents.
- [Errors](errors.md) — exception types.
- [API reference](api.md) — full public surface.
