"""tomlrt: a format-preserving TOML reader and writer."""

from __future__ import annotations

from tomlrt._container import AoT, Array, Document, Table, TomlInput
from tomlrt._errors import TOMLError, TOMLParseError
from tomlrt._public import dump, dumps, load, loads

__all__ = [
    "AoT",
    "Array",
    "Document",
    "TOMLError",
    "TOMLParseError",
    "Table",
    "TomlInput",
    "dump",
    "dumps",
    "load",
    "loads",
]
