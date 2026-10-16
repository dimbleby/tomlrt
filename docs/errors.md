# Errors

tomlrt raises `TOMLParseError` for invalid input and `TOMLError` for operations
that a document's structure does not allow. Ordinary lookups also raise
`KeyError`, `IndexError`, and `TypeError` as appropriate.

## `TOMLError`

Base class for tomlrt's own errors. For example, converting a non-inline table
to a section raises `TOMLError`.

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

Printing the exception includes the problem and its location:

```python
try:
    tomlrt.loads("a = ?")
except tomlrt.TOMLParseError as exc:
    print(exc)  # invalid integer '?' (line 1, column 5)
```
