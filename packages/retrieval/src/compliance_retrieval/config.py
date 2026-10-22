"""Retrieval configuration (defaults come from Settings, frozen from the DEV evaluation)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from compliance_core.config import Settings

KeywordMode = Literal["or", "websearch"]
MAX_RERANK_CANDIDATES = 100  # the model service accepts at most 100 passages per call


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int = 8
    candidate_k: int = 50  # per-method candidates fed into fusion
    ef_search: int = 100  # pgvector hnsw.ef_search (>= candidate_k is recommended)
    rrf_k: int = 5  # RRF constant: score = sum_m w_m / (rrf_k + rank_m)
    vector_weight: float = 1.0
    keyword_weight: float = 1.5
    keyword_mode: KeywordMode = "or"
    rerank_candidates: int = 30
    statement_timeout_ms: int = 5000

    def __post_init__(self) -> None:
        if not 1 <= self.top_k <= self.candidate_k:
            raise ValueError("top_k must be between 1 and candidate_k")
        if self.ef_search < 1 or self.rrf_k < 1:
            raise ValueError("ef_search and rrf_k must be positive")
        if self.vector_weight < 0 or self.keyword_weight < 0:
            raise ValueError("fusion weights must be non-negative")
        if self.vector_weight == 0 and self.keyword_weight == 0:
            raise ValueError("at least one fusion weight must be positive")
        if not 1 <= self.rerank_candidates <= MAX_RERANK_CANDIDATES:
            raise ValueError(f"rerank_candidates must be between 1 and {MAX_RERANK_CANDIDATES}")
        if self.keyword_mode not in ("or", "websearch"):
            raise ValueError("keyword_mode must be 'or' or 'websearch'")
        if self.statement_timeout_ms < 1:
            raise ValueError("statement_timeout_ms must be positive")

    @property
    def rerank_pool(self) -> int:
        """How many fused hits are reranked (never fewer than top_k)."""
        return max(self.rerank_candidates, self.top_k)

    @classmethod
    def from_settings(cls, settings: Settings) -> RetrievalConfig:
        return cls(
            top_k=settings.search_top_k,
            candidate_k=max(settings.search_candidate_k, settings.search_top_k),
            ef_search=settings.search_ef_search,
            rrf_k=settings.search_rrf_k,
            vector_weight=settings.search_vector_weight,
            keyword_weight=settings.search_keyword_weight,
            keyword_mode=settings.search_keyword_mode,
            rerank_candidates=settings.search_rerank_candidates,
            statement_timeout_ms=settings.search_statement_timeout_ms,
        )

    def with_top_k(self, top_k: int) -> RetrievalConfig:
        return replace(self, top_k=top_k, candidate_k=max(self.candidate_k, top_k))
