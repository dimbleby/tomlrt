"""Public top-level API for tomlrt."""

from __future__ import annotations

from typing import IO, TYPE_CHECKING, Any

from tomlrt._build import build_from_parse
from tomlrt._container import Document
from tomlrt._errors import TOMLParseError
from tomlrt._parser import _Parser
from tomlrt._synth import render_mapping

if TYPE_CHECKING:
    from collections.abc import Mapping


def loads(text: str) -> Document:
    """Parse a TOML document string into a [`Document`][tomlrt.Document]."""
    # Avoid retaining parser-only state while building logical views.
    result = _Parser(text).parse()
    return build_from_parse(result)


def load(fp: IO[bytes]) -> Document:
    """Parse a TOML document from a *binary* file-like object.

    The file must be opened in binary mode (``open(path, "rb")``).
    """
    data: object = fp.read()
    if not isinstance(data, (bytes, bytearray)):
        msg = (
            "tomlrt.load expects a binary file (open with mode='rb'); "
            f"got a text stream returning {type(data).__name__}"
        )
        raise TypeError(msg)
    encoded = bytes(data)
    try:
        text = encoded.decode("utf-8")
    except UnicodeDecodeError as exc:
        prefix = encoded[: exc.start].decode("utf-8")
        offset = len(prefix)
        line = prefix.count("\n") + 1
        col = offset - prefix.rfind("\n")
        msg = "invalid UTF-8"
        raise TOMLParseError(
            msg,
            line=line,
            col=col,
            offset=offset,
        ) from exc
    return loads(text)


def dumps(data: Mapping[str, Any]) -> str:
    """Serialize a [`Document`][tomlrt.Document] back to a TOML string.

    A mapping that is not already a [`Document`][tomlrt.Document] is
    synthesised as one, so ``tomlrt.dumps({"a": 1})`` works.
    """
    if isinstance(data, Document):
        return data.render()
    return render_mapping(data)


def dump(data: Mapping[str, Any], fp: IO[bytes]) -> None:
    """Serialize a [`Document`][tomlrt.Document] and write it to a *binary* stream.

    The file must be opened in binary mode (``open(path, "wb")``).

    Accepts a plain mapping as well as a [`Document`][tomlrt.Document]
    (see [`dumps`][tomlrt.dumps]).
    """
    fp.write(dumps(data).encode("utf-8"))


__all__ = ["dump", "dumps", "load", "loads"]
