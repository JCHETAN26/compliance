"""ContractNLI evidence-retrieval evaluation: tune on DEV, freeze, measure TEST once.

    uv run python -m evals.retrieval.run dev    # grid on DEV -> evals/retrieval/frozen_config.json
    uv run python -m evals.retrieval.run test   # frozen config only -> TEST numbers
    uv run python -m evals.retrieval.run all    # dev then test (make eval-retrieval)

Methods (all through production code: ``compliance_retrieval`` SQL, fusion and the
``run_search`` service on a migrated pgvector database, with the pinned encoder and
reranker): keyword (Postgres FTS ``ts_rank_cd``), vector (pgvector HNSW cosine),
hybrid (weighted RRF of both) and hybrid + cross-encoder rerank. Every search is
document-scoped: the task is "find the evidence for this hypothesis in this contract".

DEV tuning scores the production candidate SQL once per (pair, query instruction,
keyword mode) and applies the production ``reciprocal_rank_fusion`` for each grid
point. The selected configuration is then re-run end to end through ``run_search`` on
DEV (and checked to give identical rankings) before it is frozen. The TEST stage reads
only the frozen file and never changes it. See ``metrics.py`` for metric definitions.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import statistics
import sys
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from compliance_core.config import Settings
from compliance_core.model_client import embedding_label
from compliance_ingestion.chunking import ChunkOptions
from compliance_ingestion.embeddings import Embedder, ensure_model, spec_from_settings
from compliance_ingestion.tokens import HFTokenCounter
from compliance_model_service.rerank import CrossEncoderReranker, rerank_spec
from compliance_retrieval.config import RetrievalConfig
from compliance_retrieval.fusion import Candidate, reciprocal_rank_fusion, single_method
from compliance_retrieval.schemas import SearchRequest
from compliance_retrieval.service import SearchDependencies, run_search
from compliance_retrieval.sql import (
    Scope,
    explain,
    keyword_candidates,
    vector_candidates,
    vector_statement,
)
from compliance_seed.job import ensure_dataset
from evals.retrieval import corpus as corpus_db
from evals.retrieval import metrics, plots, provenance
from evals.retrieval.data import SPLITS, EvalPair, SplitData, build_split
from evals.retrieval.models import LocalModels

REPO = provenance.REPO
RESULTS = REPO / "evals" / "results"
FROZEN = REPO / "evals" / "retrieval" / "frozen_config.json"
SEED = 20261021
TOP_K = 10  # ranks scored (recall@10 / MRR@10)
CANDIDATE_K = 50
EF_SEARCH = 100
TIMEOUT_MS = 30_000
BOOTSTRAP = 2000

GRID_RRF_K = (1, 5, 10, 20, 40, 60, 100)
GRID_KEYWORD_WEIGHT = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0)
GRID_KEYWORD_MODES = ("or", "websearch")


@dataclass
class Context:
    settings: Settings
    url: str
    splits: dict[str, SplitData]
    stored: corpus_db.LoadedCorpus
    session_factory: async_sessionmaker[AsyncSession]
    models: LocalModels
    ranges: dict[str, tuple[int, int]]  # chunk id -> (char_start, char_end)
    texts: dict[str, str]
    doc_chunks: dict[uuid.UUID, list[tuple[int, int]]]
    encoder: str
    corpus_info: dict[str, Any]


def log(message: str) -> None:
    print(message, flush=True)  # noqa: T201


async def prepare(*, with_reranker: bool) -> Context:
    settings = Settings(artifact_cache_dir=REPO / ".cache", artifact_auto_download=False)
    archive = ensure_dataset(settings)
    model_path = ensure_model(
        settings.model_cache_dir, spec_from_settings(settings), allow_download=False
    )
    counter = HFTokenCounter(model_path / "tokenizer.json")
    options = ChunkOptions(
        settings.chunk_min_tokens, settings.chunk_max_tokens, settings.chunk_overlap_tokens
    )
    log("segmenting ContractNLI (official splits) with the production chunker ...")
    splits = {s: build_split(archive, s, counter, options) for s in SPLITS}
    encoder = embedding_label(settings.embedding_model_id, settings.embedding_model_revision)
    eval_dir = REPO / "evals" / "retrieval"
    fingerprint = corpus_db.corpus_fingerprint(
        {
            "dataset_sha256": _sha256(archive),
            "encoder": encoder,
            "embedding_sha256": settings.embedding_model_sha256,
            "chunk_options": asdict(options),
            # Only code that shapes the stored corpus (segmentation, embedding, loading).
            "code": corpus_db.code_hash(
                eval_dir / "data.py",
                eval_dir / "corpus.py",
                *sorted((REPO / "packages" / "ingestion" / "src").rglob("*.py")),
            ),
        }
    )
    url = corpus_db.eval_database_url()
    embedder = Embedder(model_path, threads=settings.embedding_threads, batch_size=32)
    if await corpus_db.stored_fingerprint(url) == fingerprint:
        log(f"reusing isolated eval database (fingerprint {fingerprint[:12]})")
    else:
        log("building isolated eval database (migrate + embed all splits) ...")
        started = time.perf_counter()
        await corpus_db.recreate_database(url)
        await corpus_db.load_corpus(url, list(splits.values()), embedder, encoder, fingerprint)
        log(f"  corpus built in {time.perf_counter() - started:.0f}s")
    stored = await corpus_db.corpus_index(url)
    for split, data in splits.items():
        expected = sum(len(d.chunks) for d in data.documents.values())
        if stored.chunk_counts.get(split) != expected:
            raise RuntimeError(f"stored corpus for {split} does not match the segmentation")
        missing = [d for d in data.documents if (split, d) not in stored.documents]
        if missing:  # every official document (duplicates included) must be searchable
            raise RuntimeError(f"{split}: documents missing from the eval database: {missing}")
    engine = corpus_db.make_engine(url)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    ranges: dict[str, tuple[int, int]] = {}
    texts: dict[str, str] = {}
    doc_chunks: dict[uuid.UUID, list[tuple[int, int]]] = {}
    async with factory() as session:
        rows = await session.execute(
            text(
                "SELECT id, document_id, char_start, char_end, text FROM chunks ORDER BY chunk_index"
            )
        )
        for cid, did, start, end, body in rows.all():
            ranges[str(cid)] = (start, end)
            texts[str(cid)] = body
            doc_chunks.setdefault(did, []).append((start, end))
    reranker = None
    if with_reranker:
        rpath = ensure_model(settings.model_cache_dir, rerank_spec(settings), allow_download=False)
        reranker = CrossEncoderReranker(rpath, threads=settings.embedding_threads)
    models = LocalModels(settings, embedder, reranker, settings.embedding_query_instruction)
    corpus_info = {
        "database": url.rsplit("@", 1)[-1],
        "fingerprint": fingerprint,
        "dataset_sha256": _sha256(archive),
        "chunk_options": asdict(options),
        "splits": {
            s: {
                "documents": len(d.documents),
                "chunks": stored.chunk_counts[s],
                "chunks_per_document": _dist([len(x.chunks) for x in d.documents.values()]),
                "label_counts": d.label_counts,
                "evaluated_pairs": len(d.pairs),
                "unique_queries": len({p.hypothesis for p in d.pairs}),
                "documents_with_evaluated_pairs": len({p.doc_id for p in d.pairs}),
                "share_of_pairs_whose_document_has_at_most_k_chunks": {
                    str(k): round(
                        sum(1 for p in d.pairs if len(d.documents[p.doc_id].chunks) <= k)
                        / max(len(d.pairs), 1),
                        4,
                    )
                    for k in metrics.KS
                },
                "documents_in_overflow_workspaces": sum(
                    1
                    for x in d.documents
                    if stored.documents[(s, x)].workspace_id
                    != stored.documents[(s, next(iter(d.documents)))].workspace_id
                ),
                "evaluated_gold_spans": sum(len(p.gold) for p in d.pairs),
                "excluded_not_mentioned_pairs": d.excluded_no_evidence,
                "excluded_evidence_label_without_spans": d.excluded_empty_evidence,
                "unreachable_gold_spans": _unreachable(d),
            }
            for s, d in splits.items()
        },
    }
    return Context(
        settings,
        url,
        splits,
        stored,
        factory,
        models,
        ranges,
        texts,
        doc_chunks,
        encoder,
        corpus_info,
    )


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _dist(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "mean": round(statistics.mean(ordered), 2),
        "median": float(statistics.median(ordered)),
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
    }


def _pct(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(q * len(ordered)))], 2)


def _unreachable(data: SplitData) -> int:
    count = 0
    for pair in data.pairs:
        chunks = data.documents[pair.doc_id].chunks
        for span in pair.gold:
            if not any(metrics.covers((c.char_start, c.char_end), span) for c in chunks):
                count += 1
    return count


def scope_of(ctx: Context, pair: EvalPair) -> Scope:
    stored = ctx.stored.documents[(pair.split, pair.doc_id)]
    return Scope(stored.workspace_id, document_ids=(stored.document_id,))


def score_ids(ctx: Context, ids: Sequence[str], pair: EvalPair) -> metrics.PairScore:
    return metrics.score_ranking([ctx.ranges[i] for i in ids[:TOP_K]], pair.gold)


# ----------------------------------------------------------------------------- DEV


async def candidate_lists(
    ctx: Context, pairs: Sequence[EvalPair], instruction: str, mode: str
) -> tuple[list[list[Candidate]], list[list[Candidate]]]:
    models = ctx.models.with_instruction(instruction)
    vec: list[list[Candidate]] = []
    kw: list[list[Candidate]] = []
    async with ctx.session_factory() as session:
        for pair in pairs:
            scope = scope_of(ctx, pair)
            emb = await models.embed_query(pair.hypothesis)
            vec.append(
                await vector_candidates(
                    session,
                    scope,
                    emb.vector,
                    encoder=ctx.encoder,
                    k=CANDIDATE_K,
                    ef_search=EF_SEARCH,
                    statement_timeout_ms=TIMEOUT_MS,
                )
            )
            kw.append(
                await keyword_candidates(
                    session,
                    scope,
                    pair.hypothesis,
                    k=CANDIDATE_K,
                    mode=mode,  # type: ignore[arg-type]
                    statement_timeout_ms=TIMEOUT_MS,
                )
            )
    return vec, kw


def evaluate_lists(
    ctx: Context, pairs: Sequence[EvalPair], rankings: Sequence[Sequence[str]]
) -> dict[str, float]:
    return metrics.aggregate(
        [score_ids(ctx, ids, pair) for pair, ids in zip(pairs, rankings, strict=True)]
    )


async def run_dev(ctx: Context, command: list[str]) -> dict[str, Any]:
    data = ctx.splits["dev"]
    pairs = data.pairs
    log(f"DEV: {len(pairs)} (document, hypothesis) pairs with evidence")
    configured = ctx.settings.embedding_query_instruction
    instructions = {"configured": configured, "none": ""}
    singles: dict[str, dict[str, float]] = {}
    grid: list[dict[str, Any]] = []
    cache: dict[tuple[str, str], tuple[list[list[Candidate]], list[list[Candidate]]]] = {}
    for ikey, instruction in instructions.items():
        for mode in GRID_KEYWORD_MODES:
            vec, kw = await candidate_lists(ctx, pairs, instruction, mode)
            cache[(ikey, mode)] = (vec, kw)
            singles[f"vector[instruction={ikey}]"] = evaluate_lists(
                ctx, pairs, [[h.chunk_id for h in single_method("vector", v)] for v in vec]
            )
            singles[f"keyword[mode={mode}]"] = evaluate_lists(
                ctx, pairs, [[h.chunk_id for h in single_method("keyword", k)] for k in kw]
            )
            for rrf_k in GRID_RRF_K:
                for weight in GRID_KEYWORD_WEIGHT:
                    rankings = [
                        [
                            h.chunk_id
                            for h in reciprocal_rank_fusion(
                                {"vector": v, "keyword": k},
                                {"vector": 1.0, "keyword": weight},
                                rrf_k,
                            )
                        ]
                        for v, k in zip(vec, kw, strict=True)
                    ]
                    grid.append(
                        {
                            "instruction": ikey,
                            "keyword_mode": mode,
                            "rrf_k": rrf_k,
                            "keyword_weight": weight,
                            "metrics": evaluate_lists(ctx, pairs, rankings),
                        }
                    )
    # Deterministic selection: best primary metric, then MRR, then grid order.
    best = max(
        enumerate(grid), key=lambda item: (metrics.selection_key(item[1]["metrics"]), -item[0])
    )[1]
    log(f"DEV selected: {best}")
    frozen_config = {
        "query_instruction": instructions[best["instruction"]],
        "keyword_mode": best["keyword_mode"],
        "rrf_k": best["rrf_k"],
        "vector_weight": 1.0,
        "keyword_weight": best["keyword_weight"],
        "candidate_k": CANDIDATE_K,
        "ef_search": EF_SEARCH,
        "rerank_candidates": 30,
    }
    # End-to-end DEV run of the frozen configuration through run_search.
    e2e = await end_to_end(ctx, pairs, frozen_config)
    vec, kw = cache[(best["instruction"], best["keyword_mode"])]
    offline = [
        [
            h.chunk_id
            for h in reciprocal_rank_fusion(
                {"vector": v, "keyword": k},
                {"vector": 1.0, "keyword": best["keyword_weight"]},
                best["rrf_k"],
            )
        ][:TOP_K]
        for v, k in zip(vec, kw, strict=True)
    ]
    if e2e["rankings"]["hybrid"] != offline:
        raise RuntimeError("end-to-end hybrid rankings differ from the offline fusion")
    random_scores = metrics.aggregate(
        [
            metrics.random_baseline(ctx.doc_chunks[scope_of(ctx, p).document_ids[0]], p.gold)
            for p in pairs
        ]
    )
    result: dict[str, Any] = {
        "stage": "dev",
        "split": "dev",
        "task": "document-scoped evidence retrieval (query = hypothesis, gold = evidence spans)",
        "metric_definitions": metrics.__doc__,
        "primary_metric": metrics.PRIMARY,
        "grid": {
            "query_instruction": instructions,
            "keyword_mode": list(GRID_KEYWORD_MODES),
            "rrf_k": list(GRID_RRF_K),
            "keyword_weight (vector_weight=1)": list(GRID_KEYWORD_WEIGHT),
            "points": len(grid),
        },
        "single_methods": singles,
        "hybrid_grid": grid,
        "selected": best,
        "frozen_config": frozen_config,
        "end_to_end_frozen": {k: v for k, v in e2e.items() if k not in ("rankings", "pair_scores")},
        "random_baseline": random_scores,
        "consistency_check": "end-to-end run_search hybrid rankings == offline fusion rankings",
        "corpus": ctx.corpus_info,
        "provenance": provenance.collect(command, SEED, _model_info(ctx)),
    }
    result["provenance"]["config_hash"] = provenance.config_hash(frozen_config)
    RESULTS.mkdir(parents=True, exist_ok=True)
    _write(RESULTS / "retrieval_dev.json", result)
    _write(
        FROZEN,
        {
            "selected_on": "ContractNLI dev split (official)",
            "selection_metric": f"{metrics.PRIMARY} (tie-break mrr@{metrics.MRR_CUTOFF})",
            "dev_metrics": best["metrics"],
            "config": frozen_config,
            "config_hash": provenance.config_hash(frozen_config),
            "frozen_at_utc": result["provenance"]["timestamp_utc"],
            "dev_results": "evals/results/retrieval_dev.json",
            "code_fingerprint": result["provenance"]["code_fingerprint"]["sha256"],
        },
    )
    sel = (best["keyword_weight"], best["rrf_k"])
    plots.tuning_heatmap(
        {
            (g["keyword_weight"], g["rrf_k"]): g["metrics"][metrics.PRIMARY]
            for g in grid
            if g["instruction"] == best["instruction"] and g["keyword_mode"] == best["keyword_mode"]
        },
        sel,
        f"DEV recall@3 by RRF k and keyword weight "
        f"(instruction={best['instruction']}, keyword mode={best['keyword_mode']})",
        RESULTS / "retrieval_dev_grid.png",
    )
    return result


# ----------------------------------------------------------------------------- shared


def retrieval_config(frozen: dict[str, Any]) -> RetrievalConfig:
    return RetrievalConfig(
        top_k=TOP_K,
        candidate_k=frozen["candidate_k"],
        ef_search=frozen["ef_search"],
        rrf_k=frozen["rrf_k"],
        vector_weight=frozen["vector_weight"],
        keyword_weight=frozen["keyword_weight"],
        keyword_mode=frozen["keyword_mode"],
        rerank_candidates=frozen["rerank_candidates"],
        statement_timeout_ms=TIMEOUT_MS,
    )


METHODS: dict[str, tuple[str, bool]] = {
    "keyword": ("keyword", False),
    "vector": ("vector", False),
    "hybrid": ("hybrid", False),
    "hybrid+rerank": ("hybrid", True),
}


async def end_to_end(
    ctx: Context,
    pairs: Sequence[EvalPair],
    frozen: dict[str, Any],
    methods: dict[str, tuple[str, bool]] = METHODS,
) -> dict[str, Any]:
    deps = SearchDependencies(
        session_factory=ctx.session_factory,
        models=ctx.models.with_instruction(frozen["query_instruction"]),
        config=retrieval_config(frozen),
    )
    out: dict[str, dict[str, Any]] = {
        "metrics": {},
        "per_label": {},
        "timings_ms": {},
        "rankings": {},
        "pair_scores": {},
    }
    for name, (mode, rerank) in methods.items():
        if rerank and ctx.models.reranker is None:
            continue
        scores: list[metrics.PairScore] = []
        rankings: list[list[str]] = []
        timings: dict[str, list[float]] = {}
        for pair in pairs:
            response = await run_search(
                deps,
                scope_of(ctx, pair),
                SearchRequest(query=pair.hypothesis, mode=mode, top_k=TOP_K, rerank=rerank),  # type: ignore[arg-type]
            )
            if response.excluded_from_vector:
                raise RuntimeError("encoder mismatch inside the eval corpus")
            ids = [str(hit.chunk_id) for hit in response.results]
            rankings.append(ids)
            scores.append(score_ids(ctx, ids, pair))
            for key, value in response.timings.model_dump().items():
                if value is not None:
                    timings.setdefault(key, []).append(value)
        out["metrics"][name] = metrics.aggregate(scores)
        out["per_label"][name] = {
            label: metrics.aggregate(
                [s for s, p in zip(scores, pairs, strict=True) if p.label == label]
            )
            for label in sorted({p.label for p in pairs})
        }
        out["timings_ms"][name] = {
            key: {"p50": _pct(v, 0.5), "p95": _pct(v, 0.95), "n": len(v)}
            for key, v in timings.items()
        }
        out["rankings"][name] = rankings
        out["pair_scores"][name] = scores
        log(f"  {name:<14} {out['metrics'][name]}")
    return out


def bootstrap_diff(
    pairs: Sequence[EvalPair],
    a: Sequence[metrics.PairScore],
    b: Sequence[metrics.PairScore],
    value: Callable[[metrics.PairScore], float],
) -> dict[str, float]:
    """Document-clustered paired bootstrap of mean(a) - mean(b) (95 % percentile CI)."""
    by_doc: dict[str, list[int]] = {}
    for i, pair in enumerate(pairs):
        by_doc.setdefault(pair.doc_id, []).append(i)
    docs = sorted(by_doc)
    rng = random.Random(SEED)  # noqa: S311 - statistical resampling, not security
    diffs: list[float] = []
    for _ in range(BOOTSTRAP):
        idx = [i for d in rng.choices(docs, k=len(docs)) for i in by_doc[d]]
        diffs.append(sum(value(a[i]) - value(b[i]) for i in idx) / len(idx))
    diffs.sort()
    point = sum(value(x) - value(y) for x, y in zip(a, b, strict=True)) / len(a)
    return {
        "difference": round(point, 4),
        "ci95_low": round(diffs[int(0.025 * BOOTSTRAP)], 4),
        "ci95_high": round(diffs[int(0.975 * BOOTSTRAP) - 1], 4),
    }


def _model_info(ctx: Context) -> dict[str, Any]:
    s = ctx.settings
    return {
        "models": {
            "embedding": {
                "id": s.embedding_model_id,
                "revision": s.embedding_model_revision,
                "sha256": s.embedding_model_sha256,
            },
            "reranker": {
                "id": s.rerank_model_id,
                "revision": s.rerank_model_revision,
                "sha256": s.rerank_model_sha256,
            },
        }
    }


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=False, default=str) + "\n")
    log(f"wrote {path.relative_to(REPO)}")


# ----------------------------------------------------------------------------- TEST


async def run_test(ctx: Context, command: list[str]) -> dict[str, Any]:
    if not FROZEN.is_file():
        raise SystemExit("no frozen configuration; run the DEV stage first")
    frozen_file = json.loads(FROZEN.read_text())
    frozen = frozen_file["config"]
    if provenance.config_hash(frozen) != frozen_file["config_hash"]:
        raise SystemExit("frozen_config.json was edited after freezing (hash mismatch)")
    pairs = ctx.splits["test"].pairs
    log(f"TEST: {len(pairs)} pairs, frozen config {frozen_file['config_hash'][:12]}")
    e2e = await end_to_end(ctx, pairs, frozen)
    default = {
        "query_instruction": ctx.settings.embedding_query_instruction,
        "keyword_mode": "or",
        "rrf_k": 60,
        "vector_weight": 1.0,
        "keyword_weight": 1.0,
        "candidate_k": CANDIDATE_K,
        "ef_search": EF_SEARCH,
        "rerank_candidates": 30,
    }
    log("TEST reference: untuned default hybrid (k=60, equal weights)")
    reference = await end_to_end(
        ctx, pairs, default, {"hybrid (untuned k=60, 1:1)": ("hybrid", False)}
    )
    random_scores = [
        metrics.random_baseline(ctx.doc_chunks[scope_of(ctx, p).document_ids[0]], p.gold)
        for p in pairs
    ]
    ps = e2e["pair_scores"]
    comparisons: dict[str, Any] = {}
    for a, b in (("hybrid", "vector"), ("hybrid", "keyword"), ("hybrid+rerank", "hybrid")):
        if a in ps and b in ps:
            comparisons[f"{a} - {b}"] = {
                metrics.PRIMARY: bootstrap_diff(pairs, ps[a], ps[b], lambda s: s.recall[3]),
                "mrr@10": bootstrap_diff(pairs, ps[a], ps[b], lambda s: s.rr),
            }
    hybrid = e2e["metrics"]["hybrid"][metrics.PRIMARY]
    plan = await _doc_scoped_plan(ctx, pairs[0], frozen["query_instruction"])
    result: dict[str, Any] = {
        "stage": "test",
        "split": "test",
        "task": "document-scoped evidence retrieval (query = hypothesis, gold = evidence spans)",
        "scope_note": (
            "Isolated text-retrieval evaluation over ContractNLI canonical text with the "
            "production chunker/encoder/SQL; it does not exercise PDF rendering, OCR, the "
            "HTTP API or report generation."
        ),
        "metric_definitions": metrics.__doc__,
        "frozen_config_file": "evals/retrieval/frozen_config.json",
        "frozen_config_hash": frozen_file["config_hash"],
        "frozen_config": frozen,
        "metrics": {
            **e2e["metrics"],
            **reference["metrics"],
            "random (expected)": metrics.aggregate(random_scores),
        },
        "per_label": e2e["per_label"],
        "paired_bootstrap_vs": comparisons,
        "hybrid_beats": {
            m: hybrid > e2e["metrics"][m][metrics.PRIMARY] for m in ("vector", "keyword")
        },
        "timings_ms": {**e2e["timings_ms"], **reference["timings_ms"]},
        "latency_scope": (
            "SearchTimings from run_search, in-process: localhost Postgres 16 + pgvector "
            "(tmpfs), sequential queries, query embeddings memoised after first use (17 "
            "distinct hypotheses) so embed_ms is mostly cache hits; rerank_ms is CPU "
            "cross-encoder time for uncached (query, passage) pairs."
        ),
        "document_scoped_vector_plan": plan,
        "corpus": ctx.corpus_info,
        "provenance": provenance.collect(command, SEED, _model_info(ctx)),
    }
    _write(RESULTS / "retrieval_test.json", result)
    order = ["random (expected)", "keyword", "vector", "hybrid", "hybrid+rerank"]
    plots.method_bars(
        {name: result["metrics"][name] for name in order if name in result["metrics"]},
        f"ContractNLI TEST evidence retrieval ({len(pairs)} pairs, document-scoped)",
        RESULTS / "retrieval_test_methods.png",
    )
    return result


async def _doc_scoped_plan(ctx: Context, pair: EvalPair, instruction: str) -> dict[str, Any]:
    emb = await ctx.models.with_instruction(instruction).embed_query(pair.hypothesis)
    statement = vector_statement(scope_of(ctx, pair), emb.vector, encoder=ctx.encoder, k=TOP_K)
    async with ctx.session_factory() as session, session.begin():
        await session.execute(
            text("SELECT set_config('hnsw.ef_search', :v, true)"), {"v": str(EF_SEARCH)}
        )
        await session.execute(
            text("SELECT set_config('hnsw.iterative_scan', 'strict_order', true)")
        )
        plan = await explain(session, statement)
    redacted = re.sub(r"'\[[^\]]*\]'::vector", "'<384-d query vector>'::vector", plan)
    return {
        "uses_hnsw_index": "ix_chunks_embedding_hnsw" in plan,
        "plan": redacted.splitlines(),
        "note": (
            "Planner's own choice for a production document-scoped vector query; with a "
            "per-document filter over a few thousand rows Postgres may prefer exact "
            "distance ordering. The ANN index itself is measured in hnsw_sweep.json."
        ),
    }


async def main_async(stage: str, command: list[str]) -> int:
    async with corpus_db.EvalLock(corpus_db.eval_database_url()):
        ctx = await prepare(with_reranker=True)
        if stage in ("dev", "all"):
            await run_dev(ctx, command)
        if stage in ("test", "all"):
            await run_test(ctx, command)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("stage", choices=["dev", "test", "all"])
    args = parser.parse_args(argv)
    command = ["python", "-m", "evals.retrieval.run", *sys.argv[1:]]
    return asyncio.run(main_async(args.stage, command))


if __name__ == "__main__":
    raise SystemExit(main())
