"""Local sentence-transformers embeddings (BAAI/bge-small-en-v1.5, 384-d, CPU).

Model files are fetched once at a pinned Hugging Face revision into the git-ignored
artifact cache. ``model.safetensors`` is checked against its pinned SHA-256 at download
time and again at every load. Loading is strictly offline; there is no fallback to any
other embedding (no hash vectors, no random vectors).

Long inputs and the model context
---------------------------------
The encoder accepts at most ``max_seq_length`` (512) tokens including ``[CLS]`` and
``[SEP]``; sentence-transformers would silently truncate anything longer. Chunks are
never cut (whole sentences only), so a chunk may exceed that. For such chunks
``Embedder`` explicitly:

1. tokenises the full chunk text (no special tokens) and cuts it into windows of at most
   ``max_seq_length - 2`` tokens, overlapping by ``window_overlap`` tokens, with every
   window boundary on a word start (never inside a WordPiece word);
2. encodes each window (each fits, verified by re-tokenising);
3. averages the L2-normalised window vectors weighted by window token count, then
   re-normalises.

The stored vector belongs to the unchanged source chunk; ``EmbeddedText.windows``
records how many windows were used (1 means the text was encoded directly).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tokenizers import Tokenizer

from compliance_core.downloads import sha256_file

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 384
_REQUIRED_FILES = (
    "config.json",
    "config_sentence_transformers.json",
    "modules.json",
    "sentence_bert_config.json",
    "1_Pooling/config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
)
_MARKER = ".verified.json"


class ModelUnavailableError(RuntimeError):
    """The pinned embedding model is not present (and could not be fetched)."""


@dataclass(frozen=True)
class ModelSpec:
    repo_id: str
    revision: str
    safetensors_sha256: str

    @property
    def dirname(self) -> str:
        return f"{self.repo_id.replace('/', '--')}--{self.revision[:12]}"


def model_dir(cache_dir: Path, spec: ModelSpec) -> Path:
    return cache_dir / spec.dirname


def _verify(path: Path, spec: ModelSpec) -> None:
    missing = [f for f in _REQUIRED_FILES if not (path / f).is_file()]
    if missing:
        raise ModelUnavailableError(f"embedding model at {path} is incomplete: missing {missing}")
    actual = sha256_file(path / "model.safetensors")
    if actual != spec.safetensors_sha256:
        raise ModelUnavailableError(
            f"embedding weights at {path} have SHA-256 {actual}, expected {spec.safetensors_sha256}"
        )


def ensure_model(cache_dir: Path, spec: ModelSpec, *, allow_download: bool) -> Path:
    """Return the verified local model directory, downloading it first if allowed."""
    target = model_dir(cache_dir, spec)
    if target.is_dir():
        _verify(target, spec)
        return target
    if not allow_download:
        raise ModelUnavailableError(
            f"embedding model {spec.repo_id}@{spec.revision} is not in {cache_dir}. "
            "Run `make fetch-artifacts` (or allow ARTIFACT_AUTO_DOWNLOAD)."
        )
    from huggingface_hub import snapshot_download  # imported lazily: network path only

    cache_dir.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".model-", dir=cache_dir))
    try:
        snapshot_download(
            repo_id=spec.repo_id,
            revision=spec.revision,
            local_dir=staging,
            allow_patterns=list(_REQUIRED_FILES),
        )
        shutil.rmtree(staging / ".cache", ignore_errors=True)
        _verify(staging, spec)
        (staging / _MARKER).write_text(
            json.dumps(
                {
                    "repo_id": spec.repo_id,
                    "revision": spec.revision,
                    "sha256": spec.safetensors_sha256,
                }
            )
        )
        staging.chmod(0o755)  # mkdtemp creates 0700; containers read it as another uid
        try:
            staging.rename(target)
        except OSError:
            if not target.is_dir():  # a concurrent fetch may have won the race
                raise
    except Exception as exc:
        raise ModelUnavailableError(
            f"could not download {spec.repo_id}@{spec.revision}: {exc}"
        ) from exc
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    _verify(target, spec)
    logger.info("embedding model ready", extra={"model": spec.repo_id, "revision": spec.revision})
    return target


@dataclass(frozen=True)
class EmbeddedText:
    vector: list[float]
    windows: int
    tokens: int


class Embedder:
    def __init__(
        self,
        path: Path,
        *,
        threads: int = 4,
        batch_size: int = 32,
        max_seq_length: int | None = None,
        window_overlap: int = 64,
    ) -> None:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(threads)
        self._model = SentenceTransformer(str(path), device="cpu", local_files_only=True)
        if max_seq_length is not None:
            self._model.max_seq_length = max_seq_length
        if self._model.max_seq_length is None:
            raise ModelUnavailableError("the embedding model does not declare max_seq_length")
        self.max_seq_length = int(self._model.max_seq_length)
        dim = self._model.get_embedding_dimension()
        if dim is None or dim != EMBEDDING_DIM:
            raise ModelUnavailableError(f"expected a {EMBEDDING_DIM}-d model, got {dim}")
        self._tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        self._tokenizer.no_truncation()
        self._tokenizer.no_padding()
        self.batch_size = batch_size
        self.window_tokens = self.max_seq_length - 2  # room for [CLS] and [SEP]
        if not 0 <= window_overlap < self.window_tokens // 2:
            raise ValueError("window_overlap must be smaller than half the window")
        self.window_overlap = window_overlap

    def _count(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False).ids)

    def windows(self, text: str) -> list[tuple[str, int]]:
        """Split ``text`` into encoder-sized windows: [(window_text, token_count)]."""
        encoding = self._tokenizer.encode(text, add_special_tokens=False)
        total = len(encoding.ids)
        if total <= self.window_tokens:
            return [(text, total)]
        tokens, offsets = encoding.tokens, encoding.offsets
        word_starts = [i for i, tok in enumerate(tokens) if not tok.startswith("##")]
        result: list[tuple[str, int]] = []
        start = 0
        while start < total:
            limit = start + self.window_tokens
            if limit >= total:
                end = total
            else:
                candidates = [w for w in word_starts if start < w <= limit]
                end = candidates[-1] if candidates else limit
            window_text = text[offsets[start][0] : offsets[end - 1][1]]
            count = self._count(window_text)
            while count > self.window_tokens and end - start > 1:
                # Re-tokenising a substring can differ at the edges; shrink until it fits.
                end -= 1
                window_text = text[offsets[start][0] : offsets[end - 1][1]]
                count = self._count(window_text)
            result.append((window_text, count))
            if end >= total:
                break
            next_start = max(end - self.window_overlap, start + 1)
            starts = [w for w in word_starts if next_start <= w < end]
            start = starts[0] if starts else end
        return result

    def _encode(self, texts: Sequence[str]) -> Any:
        return self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    def embed(self, texts: Sequence[str]) -> list[EmbeddedText]:
        plans = [self.windows(text) for text in texts]
        flat = [window for plan in plans for window, _ in plan]
        vectors = self._encode(flat) if flat else np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        results: list[EmbeddedText] = []
        offset = 0
        for text, plan in zip(texts, plans, strict=True):
            block = vectors[offset : offset + len(plan)]
            offset += len(plan)
            if len(plan) == 1:
                vector = block[0]
            else:
                weights = np.array([count for _, count in plan], dtype=np.float32)
                vector = (block * weights[:, None]).sum(axis=0)
                vector = vector / np.linalg.norm(vector)
            results.append(
                EmbeddedText(
                    vector=[float(v) for v in vector],
                    windows=len(plan),
                    tokens=plan[0][1] if len(plan) == 1 else self._count(text),
                )
            )
        return results


def spec_from_settings() -> ModelSpec:
    from compliance_core.config import get_settings

    settings = get_settings()
    return ModelSpec(
        repo_id=settings.embedding_model_id,
        revision=settings.embedding_model_revision,
        safetensors_sha256=settings.embedding_model_sha256,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch/verify the pinned embedding model")
    parser.add_argument("command", choices=["fetch", "verify"])
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    from compliance_core.config import get_settings

    cache = args.cache_dir or get_settings().model_cache_dir
    spec = spec_from_settings()
    try:
        path = ensure_model(cache, spec, allow_download=args.command == "fetch")
    except ModelUnavailableError as exc:
        print(f"error: {exc}", file=sys.stderr)  # noqa: T201
        return 1
    print(f"{spec.repo_id}@{spec.revision} verified at {path}")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
