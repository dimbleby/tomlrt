# Reading documents

Read tables like dictionaries and arrays like lists:

```python
name = doc["project"]["name"]
first_dependency = doc["project"]["dependencies"][0]
```

## Typed accessors

Use `table`, `array`, or `aot` to check the kind of value you are reading.
They return a `Table`, `Array`, or `AoT` (array of tables), which also helps
static type checkers:

```python
project = doc.table("project")
deps = project.array("dependencies")
pkgs = doc.aot("packages")
```

Arrays support `table` and `array` for indexed lookup:

```python
first_table = some_array.table(0)
nested = some_array.array(2)
```

Missing keys raise `KeyError`; a value of the wrong kind raises `TypeError`.
Use `get_table`, `get_array`, or `get_aot` to return a default for a missing
key instead. The default is `None`; a value of the wrong kind still raises
`TypeError`.

```python
ruff = doc.get_table("tool.ruff")
```

## Nested lookups

On documents and tables, these methods accept a dotted path or a sequence of
key names. Use a sequence when a name contains a literal `.`:

```python
doc.table("tool.poetry")            # tool -> poetry
doc.table(("tool", "weird.key"))    # tool -> "weird.key"
```

Use `entry` or `get_entry` to read any kind of value at a path:

```python
value = doc.entry("tool.poetry.name")           # raises if missing
maybe = doc.get_entry("tool.poetry.licence")    # None if missing
```

## Converting to plain Python

Use `to_dict()` and `to_list()` to get ordinary Python dictionaries and lists:

```python
plain = doc.to_dict()
plain["project"]["name"] = "renamed"   # does not touch the document
```
