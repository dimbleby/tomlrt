# Errors

tomlrt raises a small exception hierarchy.

## `TOMLError`

Base class for tomlrt's own errors — operations that are invalid for a
document's structure, such as promoting a non-inline table.

```python
try:
    doc.table("project").promote_inline("authors")
except tomlrt.TOMLError as exc:
    log.warning("could not promote: %s", exc)
```

## `TOMLParseError`

Raised by `loads` / `load` when the input isn't valid TOML.
Carries useful position information:

```python
try:
    tomlrt.loads("a = ?")
except tomlrt.TOMLParseError as exc:
    print(exc.line, exc.col, exc.offset)
```

| Attribute | Meaning                                         |
| --------- | ----------------------------------------------- |
| `line`    | 1-based line number                             |
| `col`     | 1-based column number                           |
| `offset`  | 0-based character offset into the source string |

The human-readable description of the problem is the exception's
string form — `str(exc)` (equivalently `exc.args[0]`) — which has
the shape `"{message} (line L, column C)"`:

```python
try:
    tomlrt.loads("a = ?")
except tomlrt.TOMLParseError as exc:
    print(exc)  # invalid integer '?' (line 1, column 5)
```
