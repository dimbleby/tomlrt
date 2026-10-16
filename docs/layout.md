# Layout

These operations change how a document is written without changing its values.
Comments stay with their entries.

## Sorting child keys

`doc.sort()` and `table.sort()` reorder their direct keys in place, keeping
comments and formatting. They accept `key` and `reverse`, like `list.sort()`.

```python
doc = tomlrt.loads("""
    # the name
    name = "tomlrt"
    # the version
    version = "0.1"
    # the author
    author = "me"
""")
doc.sort()
```

## Single-line and multi-line values

Use `.multiline` on arrays and inline tables to change their layout.
For a custom indent, use `set_multiline`:

```python
arr = doc.array("tags")
arr.multiline = True
arr.set_multiline(multiline=True, indent=2)

tbl = doc.table("pkg")
tbl.set_multiline(multiline=True, indent=2)
```

Setting `multiline=False` makes the value single-line. This raises `TOMLError`
if it would lose a comment; [remove the comments](comments.md) first.

## Converting inline tables to sections

Use `promote_inline` to turn an inline table into a section, or `promote_array`
to turn an array of inline tables into repeated `[[table]]` blocks:

```python
doc = tomlrt.loads('[tool]\nruff = { line-length = 88 }\n')
doc.table("tool").promote_inline("ruff")       # → [tool.ruff]

doc = tomlrt.loads('pkgs = [{a = 1}, {b = 2}]\n')
doc.promote_array("pkgs")                      # → [[pkgs]] … [[pkgs]]
```

Conversion raises `TOMLError` if it would lose comments inside the value.

## Canonical formatting

Call `format()` to replace the original spacing with consistent formatting.
It works in place on a document, table, or array:

```python
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

prints

```toml
a = 1
b = 2
arr = [1, 2, 3]

[bar]
x = { a = 1, b = 2 }
```

Formatting normalizes whitespace, indentation, and blank lines. Comments stay
with their entries, and multi-line values stay multi-line. Formatting one
section leaves other sections alone.

For a newly created table, add it to a document before calling `format()`.

Pass `FormatOptions` to customize the result:

```python
doc.format(options=tomlrt.FormatOptions(indent=4, normalize_comments=False))
```

See the [API reference](api.md#tomlrt.FormatOptions) for all options.

### Quote style

By default, formatting preserves string and quoted-key spelling, including
escapes. Set `quote_style` to prefer single or double quotes:

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
