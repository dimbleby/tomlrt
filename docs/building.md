# Building documents

## From scratch

```python
from tomlrt import Document, dumps

doc = Document()
doc["title"] = "example"
print(dumps(doc))
# title = "example"
```

## From a `dict`

Pass a mapping to populate the document recursively:

```python
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

```toml
[project]
name = "demo"
version = "0.1.0"
dependencies = ["requests>=2"]

[tool.ruff]
line-length = 88
```

Nested mappings become `[section]` blocks (not inline tables); lists of mappings
become `[[array.of.tables]]` blocks; everything else is an ordinary key-value
assignment.

`Document(data)` copies its input; changing `data` afterward does not affect
the document.

## Choosing tables and arrays

When assigning individual values, a plain `dict` installs as an inline table;
a plain `list` installs as an inline array.
Use tomlrt's classes to choose how a value is written:

```python
from tomlrt import AoT, Array, Table

doc["tool"] = Table.section({"version": 1})      # [tool] section
doc["xy"]   = Table.inline({"x": 1, "y": 2})     # xy = { x = 1, y = 2 }
doc["pkgs"] = AoT([{"a": 1}, {"b": 2}])          # [[pkgs]] … [[pkgs]]
doc["tags"] = Array(["a", "b"], multiline=True)  # multi-line array
```
