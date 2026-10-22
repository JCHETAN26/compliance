"""Optional cross-encoder reranker (CPU), pinned like the embedding model.

``cross-encoder/ms-marco-MiniLM-L6-v2`` (Apache-2.0, ~91 MB) scores (query, passage)
pairs. It is only loaded when ``RERANK_ENABLED=true``; an enabled reranker that cannot
be verified or loaded fails service startup (no silent fallback).

The cross-encoder reads at most 512 tokens per (query, passage) pair; longer passages
are truncated *for scoring only*. Stored chunk text and offsets are never changed.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

from compliance_core.config import Settings
from compliance_ingestion.embeddings import ModelSpec

RERANK_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
)


def rerank_spec(settings: Settings) -> ModelSpec:
    return ModelSpec(
        repo_id=settings.rerank_model_id,
        revision=settings.rerank_model_revision,
        safetensors_sha256=settings.rerank_model_sha256,
        files=RERANK_FILES,
    )


class CrossEncoderReranker:
    def __init__(self, path: Path, *, threads: int = 4, batch_size: int = 32) -> None:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        from sentence_transformers import CrossEncoder

        torch.set_num_threads(threads)
        self._model = CrossEncoder(str(path), device="cpu", local_files_only=True, max_length=512)
        self.batch_size = batch_size

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        scores = self._model.predict(
            [(query, passage) for passage in passages],
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [float(s) for s in scores]


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    from compliance_core.config import get_settings
    from compliance_ingestion.embeddings import ModelUnavailableError, ensure_model

    parser = argparse.ArgumentParser(description="Fetch/verify the pinned reranker model")
    parser.add_argument("command", choices=["fetch", "verify"])
    args = parser.parse_args(argv)
    settings = get_settings()
    spec = rerank_spec(settings)
    try:
        path = ensure_model(settings.model_cache_dir, spec, allow_download=args.command == "fetch")
    except ModelUnavailableError as exc:
        print(f"error: {exc}", file=sys.stderr)  # noqa: T201
        return 1
    print(f"{spec.repo_id}@{spec.revision} verified at {path}")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
