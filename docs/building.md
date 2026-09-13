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

Existing table and array views contribute their available comments and
formatting, including annotated tables inside lists or standalone `AoT`
values. Construction copies those views without taking ownership of the
originals; assignment attaches free views live instead.

Once you have a `Document` — whether constructed here or returned by
`tomlrt.loads` / `tomlrt.load` — see [Editing documents](editing.md) for how
to mutate it.
