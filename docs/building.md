# Building documents

## From scratch

```python exec="on" source="above" result="toml"
from tomlrt import Document, dumps

doc = Document()
doc["title"] = "example"
print(dumps(doc))
```

## From a `dict`

Pass a mapping to [`Document`][tomlrt.Document] to populate it recursively:

```python exec="on" source="above" result="toml"
from tomlrt import Document, dumps

data = {
    "project": {
        "name": "demo",
        "version": "0.1.0",
        "dependencies": ["requests>=2"],
    },
    "tool": {"ruff": {"line-length": 88}},
}

doc = Document(data)
print(dumps(doc))
```

Nested mappings become `[section]` blocks (not inline tables); lists of mappings
become `[[array.of.tables]]` blocks; everything else is an ordinary key-value
assignment.

`Document(data)` copies its input; changing `data` afterward does not affect
the document.

## Choosing tables and arrays

When assigning individual values, a plain `dict` installs as an inline table;
a plain `list` installs as an inline array.
Use [`Table.section()`][tomlrt.Table.section] or
[`Table.inline()`][tomlrt.Table.inline] to choose the table form, and
[`AoT`][tomlrt.AoT] or [`Array`][tomlrt.Array] to choose the array form:

```python exec="on" source="above" result="toml"
from tomlrt import AoT, Array, Document, Table, dumps

doc = Document()
doc["tool"] = Table.section({"version": 1})
doc["xy"] = Table.inline({"x": 1, "y": 2})
doc["pkgs"] = AoT([{"a": 1}, {"b": 2}])
doc["tags"] = Array(["a", "b"], multiline=True)
print(dumps(doc))
```
