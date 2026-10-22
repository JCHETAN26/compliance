"""Model input construction for the rule-check cross-encoder (training == serving).

Two input shapes go through the same backbone:

- **evidence input** ``[CLS] hypothesis [SEP] span [SEP]`` (at most
  ``max_evidence_tokens``), scored by the evidence head;
- **label input** ``[CLS] hypothesis [SEP] span_1 [SEP] ... span_k [SEP]`` (at most
  ``max_label_tokens``) over the top-k evidence spans *in document order*, classified by
  the 3-way label head.

Truncation is deterministic and explicit: the hypothesis is capped at
``max_hypothesis_tokens``; a span is cut at its end. For label inputs the span budget
is shared by water-filling (short spans keep all their tokens, the remainder is split
evenly among longer ones). Every returned input says whether anything was cut.

Only the ``tokenizers`` library is used (``tokenizer.json``), so the service needs no
``transformers`` tokenizer and training uses byte-identical ids. ``normalization``
(optional, e.g. ``"NFKC"``) is applied to every text before tokenization: DeBERTa-v3's
original SentencePiece model normalises with NFKC, which its converted ``tokenizer.json``
does not reproduce on its own.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from tokenizers import Tokenizer


class EncoderError(ValueError):
    pass


@dataclass(frozen=True)
class EncoderConfig:
    max_hypothesis_tokens: int = 64
    max_evidence_tokens: int = 256
    max_label_tokens: int = 512
    cls_token: str = "[CLS]"  # noqa: S105 - tokenizer special token, not a secret
    sep_token: str = "[SEP]"  # noqa: S105
    pad_token: str = "[PAD]"  # noqa: S105
    normalization: Literal["NFKC"] | None = None

    def __post_init__(self) -> None:
        if self.max_hypothesis_tokens < 8:
            raise EncoderError("max_hypothesis_tokens must be at least 8")
        if self.max_evidence_tokens < self.max_hypothesis_tokens + 8:
            raise EncoderError("max_evidence_tokens leaves no room for the span")
        if self.max_label_tokens < self.max_evidence_tokens:
            raise EncoderError("max_label_tokens must be >= max_evidence_tokens")
        if self.normalization not in (None, "NFKC"):
            raise EncoderError(f"unsupported normalization {self.normalization!r}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EncodedInput:
    ids: list[int]
    truncated: bool


def water_fill(lengths: Sequence[int], budget: int) -> list[int]:
    """Allocate ``budget`` tokens across spans: each gets min(length, fair share)."""
    if budget < 0:
        raise EncoderError("negative token budget")
    alloc = [0] * len(lengths)
    remaining = budget
    order = sorted(range(len(lengths)), key=lambda i: (lengths[i], i))
    for position, index in enumerate(order):
        share = remaining // (len(order) - position)
        alloc[index] = min(lengths[index], share)
        remaining -= alloc[index]
    return alloc


class InputEncoder:
    def __init__(self, tokenizer: Tokenizer, config: EncoderConfig) -> None:
        tokenizer.no_truncation()
        tokenizer.no_padding()
        self.tokenizer = tokenizer
        self.config = config
        self.cls_id = self._token_id(config.cls_token)
        self.sep_id = self._token_id(config.sep_token)
        self.pad_id = self._token_id(config.pad_token)

    @classmethod
    def from_file(cls, path: Path, config: EncoderConfig) -> InputEncoder:
        return cls(Tokenizer.from_file(str(path)), config)

    def _token_id(self, token: str) -> int:
        token_id = self.tokenizer.token_to_id(token)
        if token_id is None:
            raise EncoderError(f"tokenizer has no special token {token!r}")
        return int(token_id)

    def tokenize(self, texts: Sequence[str]) -> list[list[int]]:
        """Token ids without special tokens, one list per text."""
        if not texts:
            return []
        form = self.config.normalization
        prepared = [unicodedata.normalize(form, t) for t in texts] if form else list(texts)
        return [e.ids for e in self.tokenizer.encode_batch(prepared, add_special_tokens=False)]

    def _hypothesis(self, hypothesis_ids: Sequence[int]) -> tuple[list[int], bool]:
        cap = self.config.max_hypothesis_tokens
        return list(hypothesis_ids[:cap]), len(hypothesis_ids) > cap

    def evidence_input(
        self, hypothesis_ids: Sequence[int], span_ids: Sequence[int]
    ) -> EncodedInput:
        hyp, hyp_cut = self._hypothesis(hypothesis_ids)
        budget = self.config.max_evidence_tokens - len(hyp) - 3
        span = list(span_ids[:budget])
        ids = [self.cls_id, *hyp, self.sep_id, *span, self.sep_id]
        return EncodedInput(ids, hyp_cut or len(span_ids) > budget)

    def label_input(
        self, hypothesis_ids: Sequence[int], spans_ids: Sequence[Sequence[int]]
    ) -> EncodedInput:
        """``spans_ids`` must already be in document order (see ``top_k_in_order``)."""
        hyp, hyp_cut = self._hypothesis(hypothesis_ids)
        budget = self.config.max_label_tokens - len(hyp) - 2 - len(spans_ids)
        if budget < len(spans_ids):
            raise EncoderError("too many spans for max_label_tokens")
        alloc = water_fill([len(s) for s in spans_ids], budget)
        ids = [self.cls_id, *hyp, self.sep_id]
        cut = hyp_cut
        for span, keep in zip(spans_ids, alloc, strict=True):
            ids.extend(span[:keep])
            ids.append(self.sep_id)
            cut = cut or keep < len(span)
        return EncodedInput(ids, cut)

    def pad(self, rows: Sequence[Sequence[int]]) -> tuple[np.ndarray, np.ndarray]:
        """Right-pad rows into int64 ``input_ids`` and ``attention_mask`` arrays."""
        if not rows:
            raise EncoderError("cannot pad an empty batch")
        width = max(len(r) for r in rows)
        ids = np.full((len(rows), width), self.pad_id, dtype=np.int64)
        mask = np.zeros((len(rows), width), dtype=np.int64)
        for i, row in enumerate(rows):
            ids[i, : len(row)] = row
            mask[i, : len(row)] = 1
        return ids, mask


def top_k_in_order(scores: Sequence[float], k: int) -> list[int]:
    """Indices of the ``k`` highest scores (ties -> lower index), returned in index order.

    Index order is document order, which is the order the label head was trained on.
    """
    if k < 1:
        raise EncoderError("k must be >= 1")
    ranked = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    return sorted(ranked[:k])


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if temperature <= 0:
        raise EncoderError("temperature must be positive")
    z = np.asarray(logits, dtype=np.float64) / temperature
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    out: np.ndarray = e / e.sum(axis=-1, keepdims=True)
    return out


def sigmoid(x: np.ndarray) -> np.ndarray:
    out: np.ndarray = 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))
    return out
