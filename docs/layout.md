# Layout

`tomlrt` is format-preserving by design: parsing and dumping a document
returns exactly the same bytes. Separately from editing a document's
_data_, you can reshape its _layout_ — the order of keys, whether an
inline value spans one line or many, whether a value is inline or a
section, and the canonical spacing of the whole thing.

None of these operations change a document's _data_ — `to_dict()`
compares equal before and after. All of them are format-preserving,
keeping comments and trivia attached to their data, except
[`format()`](#canonical-formatting), which deliberately _drops_ format
preservation to snap a subtree to a canonical layout.

## Sorting child keys

`Document.sort()` and `Table.sort()` reorder direct child keys in
place. The signature mirrors `list.sort`: keyword-only `key` and
`reverse`. Comments and blank lines travel with their keys. Unlike
[`format()`](#canonical-formatting), sorting is format-preserving — only
the order changes.

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

When a document mixes bare keys and `[section]` / `[[aot]]` headers,
`.sort()` keeps the headers after the bare keys — any bare key
emitted after a section header would otherwise be re-parsed as a
member of that section.

```python
doc = tomlrt.loads("""
    [a]
    x = 1
    [b]
    y = 2
""")
doc["nickname"] = "hi"
doc.sort()
# nickname, [a], [b]
```

- `key` and `reverse` apply within each partition (bare keys, then
  sections); they do not interleave the two.
- Dotted keys (`a.x = 1`) are bare keys for sorting purposes.

## Inline-array layout

`Array.multiline` flips between single- and multi-line layout in place.
For multi-line layout with a custom indent, call `set_multiline`:

```python
arr = doc.array("tags")
arr.set_multiline(multiline=True, indent=2)
```

Collapsing a multi-line array to single-line is rejected if any item carries a
comment; clear them first (see [Comments](comments.md)).

## Inline-table layout

Inline tables expose the same controls. `Table.multiline` and
`Table.set_multiline` flip an inline table between single- and multi-line
layout (TOML 1.1 allows multi-line inline tables):

```python
tbl = doc.table("pkg")
tbl.set_multiline(multiline=True, indent=2)
```

As with arrays, collapsing back to a single line is rejected when an entry
carries a comment. `multiline` / `set_multiline` apply to the inline table as
a whole, so call them on the table itself, not on a dotted-key view of it.

## Promoting inline → section

If a value started life as an inline table or inline array of inline tables, you
can promote it in place:

```python
doc = tomlrt.loads('[tool]\nruff = { line-length = 88 }\n')
doc.table("tool").promote_inline("ruff")       # → [tool.ruff]

doc = tomlrt.loads('pkgs = [{a = 1}, {b = 2}]\n')
doc.promote_array("pkgs")                      # → [[pkgs]] … [[pkgs]]
```

Promotion is rejected if it would lose inner comments; clear them first (see
[Comments](comments.md)).

## Canonical formatting

The operations above are format-preserving. When you instead want to
_opt in_ to canonical formatting — to drop the original spacing of a
single value, a section, or the whole document — call `format()`.

`Container.format()` and `Array.format()` both mutate in place and return
`None`, mirroring `list.sort()`. Pass a `FormatOptions` object to configure
canonical formatting consistently at any scope.

| Option | Default | Effect |
|--------|---------|--------|
| `normalize_comments` | `True` | Normalize comment text. |
| `indent` | `2` | Spaces added per nested multiline inline value. |
| `eol_comment_spaces` | `1` | Spaces before supported EOL comments. |
| `multiline_trailing_comma` | `True` | Emit a final comma in multiline arrays and inline tables. |

For example, `doc.format(options=tomlrt.FormatOptions(indent=4))` uses a
four-space step at every nested multiline level.

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

### What gets normalised

- **Whitespace around `=`**: collapsed to one space on each side.
- **Dotted keys**: separators become bare `.` (no surrounding whitespace).
  The user's bare-vs-quoted spelling for each key segment is preserved.
- **Header brackets**: `[  a . b  ]` → `[a.b]`.
- **Inline arrays and inline tables**: spacing collapses to the
  canonical form (`[1, 2, 3]`, `{ x = 1, y = 2 }`). The overall
  _shape_ is preserved — a multi-line value stays multi-line, and a
  single-line value stays single-line.
- **Blank lines between sibling KVs** collapse to none.
- **Blank lines before structural section / array-of-tables headers**
  collapse to exactly one.
- Other runs of blank lines reached by the formatter collapse to one.
- **All newlines** are retargeted to the owning document's newline
  style (LF or CRLF), including newlines inside multi-line inline
  values.

### What is preserved

- The document **preamble** (any leading comments above the first
  section).
- **Orphan comment blocks** — comment runs separated from a key or
  header by at least one blank line — are kept in place. Multiple
  separating blank lines collapse to one.
- Each key's **attached comment block** (the comments immediately
  above the key, with no intervening blank line) stays attached.
- Multi-line inline values keep their multi-line shape.
- Slots outside the receiver's subtree are not touched: calling
  `format()` on a single section leaves sibling sections alone.

### Comment normalization

By default, every comment reached by the walk is rewritten so that there is
exactly one space between `#` and the body, and any trailing whitespace inside
the comment is stripped:

| Before | After |
|--------|-------|
| <code style="white-space: pre">#foo</code> | <code style="white-space: pre"># foo</code> |
| <code style="white-space: pre">#   foo</code> | <code style="white-space: pre"># foo</code> |
| <code style="white-space: pre"># foo   </code> | <code style="white-space: pre"># foo</code> |
| <code style="white-space: pre">#</code> (empty) | <code style="white-space: pre">#</code> |

Pass `FormatOptions(normalize_comments=False)` to leave comment text untouched.
The former `comments=` keyword remains available for compatibility but is
deprecated; do not pass it together with `options=`.

`eol_comment_spaces` applies to key/value, section-header, array-item, and
inline-table-entry comments. Opening-bracket comment spacing remains authored.

### Detached views

`format()` works on attached containers and arrays — those reachable
from a parsed or built `Document`. Calling it on a detached
`Table.section()` or `Table.inline()` factory raises `TOMLError`:
there is no document to canonicalise against.
