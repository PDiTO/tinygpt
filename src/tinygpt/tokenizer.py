"""Character-level and byte-level tokenizers.

Both are deliberately simple. The point of this project is the model and the
decoding loop, so the tokenizer only needs to be reversible and serialisable.
"""

from __future__ import annotations

import codecs
import json
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any, ClassVar


class Tokenizer(ABC):
    """Maps text to a list of integer token ids and back."""

    kind: ClassVar[str]

    @property
    @abstractmethod
    def vocab_size(self) -> int: ...

    @abstractmethod
    def encode(self, text: str) -> list[int]: ...

    @abstractmethod
    def decode(self, ids: Iterable[int]) -> str: ...

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable description that :func:`tokenizer_from_dict` can rebuild."""

    def decode_stream(self, ids: Iterable[int]) -> Iterator[str]:
        """Decode tokens one at a time, yielding text as soon as it is printable."""
        for i in ids:
            yield self.decode([i])

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8")


class CharTokenizer(Tokenizer):
    """One token per distinct character seen in the training text."""

    kind = "char"

    def __init__(self, chars: Sequence[str]) -> None:
        if not chars:
            raise ValueError("vocabulary must not be empty")
        if any(len(c) != 1 for c in chars):
            raise ValueError("every vocabulary entry must be a single character")
        if len(set(chars)) != len(chars):
            raise ValueError("vocabulary contains duplicate characters")
        self._itos = list(chars)
        self._stoi = {c: i for i, c in enumerate(self._itos)}

    @classmethod
    def from_text(cls, text: str) -> CharTokenizer:
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self._itos)

    @property
    def chars(self) -> list[str]:
        return list(self._itos)

    def encode(self, text: str) -> list[int]:
        try:
            return [self._stoi[c] for c in text]
        except KeyError:
            unknown = sorted({c for c in text if c not in self._stoi})
            raise ValueError(f"characters not in vocabulary: {unknown!r}") from None

    def decode(self, ids: Iterable[int]) -> str:
        out = []
        for i in ids:
            if not 0 <= i < len(self._itos):
                raise ValueError(f"token id {i} out of range for vocab of size {self.vocab_size}")
            out.append(self._itos[i])
        return "".join(out)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "chars": self._itos}


class ByteTokenizer(Tokenizer):
    """UTF-8 bytes as tokens. Fixed vocabulary of 256, never sees an unknown symbol."""

    kind = "byte"

    @property
    def vocab_size(self) -> int:
        return 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: Iterable[int]) -> str:
        return bytes(ids).decode("utf-8", errors="replace")

    def decode_stream(self, ids: Iterable[int]) -> Iterator[str]:
        # A multi-byte character arrives over several tokens, so buffer until it is complete.
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        for i in ids:
            text = decoder.decode(bytes([i]))
            if text:
                yield text
        tail = decoder.decode(b"", final=True)
        if tail:
            yield tail

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind}


def tokenizer_from_dict(data: dict[str, Any]) -> Tokenizer:
    kind = data.get("kind")
    if kind == CharTokenizer.kind:
        return CharTokenizer(data["chars"])
    if kind == ByteTokenizer.kind:
        return ByteTokenizer()
    raise ValueError(f"unknown tokenizer kind: {kind!r}")


def load_tokenizer(path: str | Path) -> Tokenizer:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a tokenizer description")
    return tokenizer_from_dict(data)
