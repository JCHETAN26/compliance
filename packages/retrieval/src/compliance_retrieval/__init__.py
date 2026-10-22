"""Hybrid retrieval (Phase 2): pgvector HNSW + Postgres full-text (ts_rank_cd), fused
with weighted reciprocal rank fusion, optional cross-encoder reranking.

Lightweight by design (no ML dependencies): query embeddings and reranking come from the
model service through ``compliance_core.model_client``.
"""

IMPLEMENTED = True
