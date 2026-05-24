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
        - header_comment
        - header_leading_comments
        - comments
        - leading_comments
        - to_dict

::: tomlrt.Table
    options:
      members:
        - section
        - inline
        - is_inline
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
        - comments
        - leading_comments
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
