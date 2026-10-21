"""Token counting with the embedding model's own tokenizer (no special tokens)."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from tokenizers import Tokenizer


class TokenCounter(Protocol):
    def count_many(self, texts: list[str]) -> list[int]: ...


class HFTokenCounter:
    """Counts WordPiece tokens exactly as the embedding model sees them."""

    def __init__(self, tokenizer_json: Path) -> None:
        self._tokenizer = Tokenizer.from_file(str(tokenizer_json))
        self._tokenizer.no_truncation()
        self._tokenizer.no_padding()

    def count_many(self, texts: list[str]) -> list[int]:
        if not texts:
            return []
        encodings = self._tokenizer.encode_batch(texts, add_special_tokens=False)
        return [len(e.ids) for e in encodings]
