"""In-process model gateway for the eval (implements ``compliance_retrieval.ModelGateway``).

It calls exactly the functions the model service runs for ``/v1/embed`` (kind=query)
and ``/v1/rerank`` - ``Embedder.embed_queries`` and ``CrossEncoderReranker.score`` on the
pinned, SHA-256-verified models - without the HTTP hop. Results are memoised per input
(the 17 ContractNLI hypotheses repeat across documents; a rerank score depends only on
the query and the passage), which changes timing but never a value.
"""

from __future__ import annotations

import asyncio

from compliance_core.config import Settings
from compliance_core.model_client import QueryEmbedding, ServiceInfo
from compliance_ingestion.embeddings import Embedder
from compliance_model_service.rerank import CrossEncoderReranker


class LocalModels:
    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        reranker: CrossEncoderReranker | None,
        instruction: str,
    ) -> None:
        self.settings = settings
        self.embedder = embedder
        self.reranker = reranker
        self.instruction = instruction
        self._queries: dict[tuple[str, str], list[float]] = {}
        self._rerank: dict[tuple[str, str], float] = {}

    def with_instruction(self, instruction: str) -> LocalModels:
        clone = LocalModels(self.settings, self.embedder, self.reranker, instruction)
        clone._queries = self._queries
        clone._rerank = self._rerank
        return clone

    async def embed_query(self, query: str) -> QueryEmbedding:
        key = (self.instruction, query)
        if key not in self._queries:
            vectors = await asyncio.to_thread(
                self.embedder.embed_queries, [query], self.instruction
            )
            self._queries[key] = vectors[0]
        return QueryEmbedding(
            vector=self._queries[key],
            model=self.settings.embedding_model_id,
            revision=self.settings.embedding_model_revision,
            query_instruction=self.instruction,
        )

    async def rerank(self, query: str, passages: list[tuple[str, str]]) -> dict[str, float]:
        if self.reranker is None:
            raise RuntimeError("reranker not loaded")
        reranker = self.reranker
        missing = [(pid, text) for pid, text in passages if (query, text) not in self._rerank]
        if missing:
            scores = await asyncio.to_thread(reranker.score, query, [t for _, t in missing])
            for (_, passage), score in zip(missing, scores, strict=True):
                self._rerank[(query, passage)] = score
        return {pid: self._rerank[(query, passage)] for pid, passage in passages}

    async def info(self) -> ServiceInfo:
        s = self.settings
        return ServiceInfo(
            embedding_model=s.embedding_model_id,
            embedding_revision=s.embedding_model_revision,
            query_instruction=self.instruction,
            rerank_available=self.reranker is not None,
            rerank_model=s.rerank_model_id if self.reranker else None,
            rerank_revision=s.rerank_model_revision if self.reranker else None,
        )
