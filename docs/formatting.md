# Formatting documents

`tomlrt` is format-preserving by design: parsing and dumping a document
returns exactly the same bytes. When you want to *opt in* to canonical
formatting — for a single value, a section, or the whole document —
call `format()`.

## `Container.format(*, comments=True)` / `Array.format(*, comments=True)`

Both methods mutate in place and return `None`, mirroring `list.sort()`.

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

## What gets normalised

- **Whitespace around `=`**: collapsed to one space on each side.
- **Dotted keys**: separators become bare `.` (no surrounding whitespace).
  The user's bare-vs-quoted spelling for each key segment is preserved.
- **Header brackets**: `[  a . b  ]` → `[a.b]`.
- **Inline arrays and inline tables**: spacing collapses to the
  canonical form (`[1, 2, 3]`, `{ x = 1, y = 2 }`). The array's
  overall *shape* is preserved — a multi-line array stays multi-line,
  and a single-line array stays single-line.
- **Blank lines between sibling KVs** collapse to none.
- **Blank lines before structural section / array-of-tables headers**
  collapse to exactly one.
- **All newlines** are retargeted to the owning document's newline
  style (LF or CRLF), including newlines inside multi-line inline
  values.

## What is preserved

- The document **preamble** (any leading comments above the first
  section).
- **Orphan comment blocks** — comment runs separated from a key or
  header by at least one blank line — are kept verbatim in place.
- Each key's **attached comment block** (the comments immediately
  above the key, with no intervening blank line) stays attached.
- Multi-line inline values keep their multi-line shape.
- Slots outside the receiver's subtree are not touched: calling
  `format()` on a single section leaves sibling sections alone.

## The `comments` flag

When `comments=True` (the default), every comment reached by the walk
is rewritten so that there is exactly one space between `#` and the
body, and any trailing whitespace inside the comment is stripped:

| Before | After |
|--------|-------|
| `#foo` | `# foo` |
| `#   foo` | `# foo` |
| `# foo   ` | `# foo` |
| `#` (empty) | `#` |

Pass `comments=False` to leave comment text untouched. (Trailing
whitespace on otherwise-blank lines is still cleaned up — that is a
formatting fix, not a comment fix.)

## Detached views

`format()` works on attached containers and arrays — those reachable
from a parsed or built `Document`. Calling it on a detached
`Table.section()` or `Table.inline()` factory raises `TOMLError`:
there is no document to canonicalise against.

A nested inline `Table` value reached from inside a parsed document
*is* attached and can be formatted.
