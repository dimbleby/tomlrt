# API reference

The complete public surface, generated from the docstrings in the source.

## Stability

Anything imported from the top-level `tomlrt` namespace is part of the public,
semver-stable API:

| Symbol                              | Kind       |
| ----------------------------------- | ---------- |
| `loads`, `load`                     | function   |
| `dumps`, `dump`                     | function   |
| `Document`, `Table`, `Array`, `AoT` | class      |
| `FormatOptions`                     | class      |
| `TomlInput`                         | type alias |
| `TOMLError`, `TOMLParseError`       | exception  |

Anything not re-exported from `tomlrt/__init__.py` (modules prefixed with `_`,
internal helpers) may change without notice and should not be imported by user
code.

## Top-level functions

::: tomlrt.loads
::: tomlrt.load
::: tomlrt.dumps
::: tomlrt.dump

## Formatting

::: tomlrt.FormatOptions
    options:
      members:
        - normalize_comments
        - indent
        - eol_comment_spaces
        - multiline_trailing_comma

## Containers

::: tomlrt.Document
    options:
      members:
        - __init__
        - render
        - table
        - array
        - aot
        - entry
        - get_table
        - get_array
        - get_aot
        - get_entry
        - install
        - ensure_table
        - promote_inline
        - promote_array
        - preamble
        - epilogue
        - comments
        - leading_comments
        - leading_block
        - sort
        - format
        - to_dict

::: tomlrt.Table
    options:
      members:
        - section
        - inline
        - is_inline
        - multiline
        - set_multiline
        - table
        - array
        - aot
        - entry
        - get_table
        - get_array
        - get_aot
        - get_entry
        - install
        - ensure_table
        - promote_inline
        - promote_array
        - header_comment
        - header_leading_comments
        - header_leading_block
        - comments
        - leading_comments
        - leading_block
        - has_header
        - sort
        - format
        - to_dict

::: tomlrt.Array
    options:
      members:
        - __init__
        - multiline
        - set_multiline
        - table
        - array
        - get_table
        - get_array
        - comments
        - leading_comments
        - leading_block
        - format
        - to_list

::: tomlrt.AoT
    options:
      members:
        - __init__
        - add
        - to_list

## Type aliases

::: tomlrt.TomlInput

## Errors

::: tomlrt.TOMLError
    options:
      members: false
::: tomlrt.TOMLParseError
    options:
      members:
        - line
        - col
        - offset
