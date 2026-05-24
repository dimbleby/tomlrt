"""Public top-level API for tomlrt."""

from __future__ import annotations

from typing import IO, TYPE_CHECKING

from tomlrt._build import build_from_parse
from tomlrt._parser import _Parser

if TYPE_CHECKING:
    from tomlrt._container import Document


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


def dumps(doc: Document) -> str:
    """Serialize a [`Document`][tomlrt.Document] back to a TOML string."""
    return doc.render()


def dump(doc: Document, fp: IO[bytes]) -> None:
    """Serialize a [`Document`][tomlrt.Document] and write it to a *binary* stream.

    The file must be opened in binary mode (``open(path, "wb")``).
    """
    fp.write(dumps(doc).encode("utf-8"))


__all__ = ["dump", "dumps", "load", "loads"]
