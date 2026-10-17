# Layout

These operations change how a document is written without changing its values.
Comments stay with their entries.

## Sorting child keys

[`doc.sort()`][tomlrt.Document.sort] and [`table.sort()`][tomlrt.Table.sort]
reorder their direct keys in place, keeping comments and formatting.
They accept `key` and `reverse`, like `list.sort()`.

```python exec="on" source="above" result="toml"
import tomlrt

doc = tomlrt.loads("""\
# the name
name = "tomlrt"
# the version
version = "0.1"
# the author
author = "me"
""")
doc.sort()
print(tomlrt.dumps(doc))
```

## Single-line and multi-line values

Use [`multiline`][tomlrt.Array.multiline] on arrays and inline tables to change
their layout. For a custom indent, use [`set_multiline`][tomlrt.Array.set_multiline]:

```python
arr = doc.array("tags")
arr.multiline = True
arr.set_multiline(multiline=True, indent=2)

tbl = doc.table("pkg")
tbl.set_multiline(multiline=True, indent=2)
```

Setting `multiline=False` makes the value single-line. This raises
[`TOMLError`][tomlrt.TOMLError] if it would lose a comment;
[remove the comments](comments.md) first.

## Converting inline tables to sections

Use [`promote_inline`][tomlrt.Table.promote_inline] to turn an inline table
into a section:

```python exec="on" source="above" result="toml"
import tomlrt

doc = tomlrt.loads("""\
[tool]
ruff = { line-length = 88 }
""")
doc.table("tool").promote_inline("ruff")
print(tomlrt.dumps(doc))
```

Use [`promote_array`][tomlrt.Document.promote_array] to turn an array of inline
tables into repeated `[[table]]` blocks:

```python exec="on" source="above" result="toml"
import tomlrt

doc = tomlrt.loads('pkgs = [{a = 1}, {b = 2}]\n')
doc.promote_array("pkgs")
print(tomlrt.dumps(doc))
```

Conversion raises `TOMLError` if it would lose comments inside the value.

## Canonical formatting

Call [`format()`][tomlrt.Document.format] to replace the original spacing with
consistent formatting. It works in place on a document, table, or array:

```python exec="on" source="above" result="toml"
import tomlrt

doc = tomlrt.loads("""
a   =1
b=2
arr=[1,2 ,3 ]


[bar]
x={a=1,b=2}
""")

doc.format()
print(tomlrt.dumps(doc))
```

Formatting normalizes whitespace, indentation, and blank lines. Comments stay
with their entries, and multi-line values stay multi-line. Formatting one
section leaves other sections alone.

For a newly created table, add it to a document before calling `format()`.

Pass [`FormatOptions`][tomlrt.FormatOptions] to customize the result:

```python
doc.format(options=tomlrt.FormatOptions(indent=4, normalize_comments=False))
```

### Quote style

By default, formatting preserves string and quoted-key spelling, including
escapes. Set [`quote_style`][tomlrt.FormatOptions.quote_style] to prefer single
or double quotes:

```python
doc.format(options=tomlrt.FormatOptions(quote_style=tomlrt.QuoteStyle.DOUBLE))
```

The formatter avoids adding escapes, even when that means using the other
quote style:

| Before | After |
|--------|-------|
| `plain = 'hello'` | `plain = "hello"` |
| `greeting = "say \"hi\""` | `greeting = 'say "hi"'` |
| `path = 'C:\Users'` | `path = 'C:\Users'` |

Bare keys stay bare, and multi-line strings stay multi-line.

### Comment normalization

Formatting adds one space after `#` and removes trailing whitespace from comment
text. Empty comments remain `#`.

Pass `FormatOptions(normalize_comments=False)` to leave comment text untouched.
