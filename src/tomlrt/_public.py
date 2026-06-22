"""Public top-level API for tomlrt."""

from __future__ import annotations

from typing import IO, TYPE_CHECKING, Any

from tomlrt._build import build_from_parse
from tomlrt._container import Document
from tomlrt._parser import _Parser

if TYPE_CHECKING:
    from collections.abc import Mapping


def loads(text: str) -> Document:
    """Parse a TOML document string into a [`Document`][tomlrt.Document]."""
    parser = _Parser(text)
    result = parser.parse()
    return build_from_parse(result)


def load(fp: IO[bytes]) -> Document:
    """Parse a TOML document from a *binary* file-like object.

    The file must be opened in binary mode (``open(path, "rb")``).
    """
    data = fp.read()
    if not isinstance(data, (bytes, bytearray)):
        msg = (  # type: ignore[unreachable]
            "tomlrt.load expects a binary file (open with mode='rb'); "
            f"got a text stream returning {type(data).__name__}"
        )
        raise TypeError(msg)
    return loads(bytes(data).decode("utf-8"))


def dumps(data: Mapping[str, Any]) -> str:
    """Serialize a [`Document`][tomlrt.Document] back to a TOML string.

    A mapping that is not already a [`Document`][tomlrt.Document] is
    wrapped in one first, so ``tomlrt.dumps({"a": 1})`` works.
    """
    doc = data if isinstance(data, Document) else Document(data)
    return doc.render()


def dump(data: Mapping[str, Any], fp: IO[bytes]) -> None:
    """Serialize a [`Document`][tomlrt.Document] and write it to a *binary* stream.

    The file must be opened in binary mode (``open(path, "wb")``).

    Accepts a plain mapping as well as a [`Document`][tomlrt.Document]
    (see [`dumps`][tomlrt.dumps]).
    """
    fp.write(dumps(data).encode("utf-8"))


__all__ = ["dump", "dumps", "load", "loads"]
