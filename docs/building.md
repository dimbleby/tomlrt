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

To choose between sections and inline tables when adding values, see
[Editing documents](editing.md).
