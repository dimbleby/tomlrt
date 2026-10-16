# API reference

Details of the public classes, methods, and options.

## Stability

Import public names from `tomlrt`. They follow semantic versioning:

| Symbol                              | Kind       |
| ----------------------------------- | ---------- |
| `loads`, `load`                     | function   |
| `dumps`, `dump`                     | function   |
| `Document`, `Table`, `Array`, `AoT` | class      |
| `FormatOptions`                     | class      |
| `QuoteStyle`                        | enum       |
| `TomlInput`                         | type alias |
| `TOMLError`, `TOMLParseError`       | exception  |

Modules prefixed with `_` are internal and may change without notice.

## Top-level functions

::: tomlrt.loads
::: tomlrt.load
::: tomlrt.dumps
::: tomlrt.dump

## Formatting

::: tomlrt.QuoteStyle

::: tomlrt.FormatOptions
    options:
      members:
        - normalize_comments
        - indent
        - eol_comment_spaces
        - multiline_trailing_comma
        - quote_style

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
