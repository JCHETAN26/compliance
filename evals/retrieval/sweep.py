"""HNSW parameter sweep: ANN recall@10 against exact neighbours vs query latency.

    uv run python -m evals.retrieval.sweep

Corpus: every stored chunk vector of the official train + dev splits in the isolated
eval database (test is never touched), copied into a sweep-only table
``sweep_vectors`` in that database. Queries: the 17 ContractNLI hypotheses plus a seeded
sample of DEV gold evidence-span texts, embedded as queries with the frozen query
instruction. Exact ground truth: brute-force cosine top-10 in numpy (float64).

For each (m, ef_construction) the index is rebuilt (build time and size recorded);
for each ef_search the plan is checked with EXPLAIN to be an HNSW index scan, 20
warm-up queries are run, then every query is timed once. Latency scope: client-side
wall time of one ``SELECT id ... ORDER BY embedding <=> $1 LIMIT 10`` over asyncpg
on localhost (tmpfs Postgres), binary vector parameter, excluding query embedding.

This measures ANN quality (agreement with exact search), *not* evidence recall; the
latter is in ``retrieval_*.json``. With only a few thousand vectors the graph is small,
so ANN recall saturates quickly and latencies are sub-millisecond-scale - the numbers
describe this corpus size, not a large deployment.
"""

from __future__ import annotations

import asyncio
import json
import random
import statistics
import sys
import time
from typing import Any

import asyncpg
import numpy as np
from pgvector.asyncpg import register_vector
from sqlalchemy.engine import make_url

from evals.retrieval import corpus as corpus_db
from evals.retrieval import plots, provenance
from evals.retrieval.run import FROZEN, RESULTS, SEED, prepare

GRID_M = (8, 16, 32)
GRID_EF_CONSTRUCTION = (32, 64, 128, 256)
GRID_EF_SEARCH = (10, 20, 40, 80, 100, 160)
K = 10
WARMUP = 20
SAMPLED_SPANS = 283  # + 17 hypotheses = 300 queries
PRODUCTION = {"m": 16, "ef_construction": 64}


async def _connect(url: str) -> asyncpg.Connection:
    parsed = make_url(url)
    conn = await asyncpg.connect(
        user=parsed.username,
        password=parsed.password,
        host=parsed.host,
        port=parsed.port,
        database=parsed.database,
    )
    await register_vector(conn)
    return conn


def _latency(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "p50": round(statistics.median(ordered), 3),
        "p95": round(ordered[int(0.95 * (len(ordered) - 1))], 3),
        "mean": round(statistics.mean(ordered), 3),
    }


async def run(command: list[str]) -> dict[str, Any]:
    ctx = await prepare(with_reranker=False)
    instruction = (
        json.loads(FROZEN.read_text())["config"]["query_instruction"]
        if FROZEN.is_file()
        else ctx.settings.embedding_query_instruction
    )
    rng = random.Random(SEED)  # noqa: S311 - reproducible sampling, not security
    dev = ctx.splits["dev"]
    spans = sorted(
        {dev.documents[p.doc_id].text[s:e] for p in dev.pairs for s, e in p.gold if e - s > 20}
    )
    hypotheses = sorted({p.hypothesis for p in dev.pairs})
    queries = hypotheses + rng.sample(spans, SAMPLED_SPANS)
    query_vectors = np.array(
        ctx.models.embedder.embed_queries(queries, instruction), dtype=np.float64
    )

    conn = await _connect(ctx.url)
    try:
        await conn.execute("DROP TABLE IF EXISTS sweep_vectors")
        await conn.execute(
            "CREATE TABLE sweep_vectors AS SELECT row_number() OVER (ORDER BY c.id)::int AS id,"
            " c.embedding FROM chunks c JOIN documents d ON d.id = c.document_id"
            " WHERE d.filename LIKE 'contractnli-train-%' OR d.filename LIKE 'contractnli-dev-%'"
        )
        await conn.execute("ALTER TABLE sweep_vectors ADD PRIMARY KEY (id)")
        rows = await conn.fetch("SELECT id, embedding FROM sweep_vectors ORDER BY id")
        ids = np.array([r["id"] for r in rows])
        matrix = np.array([r["embedding"].to_numpy() for r in rows], dtype=np.float64)
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
        sims = query_vectors @ matrix.T
        exact = [set(ids[np.argsort(-row, kind="stable")[:K]].tolist()) for row in sims]
        sql = f"SELECT id FROM sweep_vectors ORDER BY embedding <=> $1 LIMIT {K}"  # noqa: S608

        # Exact (sequential scan) reference: correctness of SQL vs numpy and baseline latency.
        exact_rows: list[float] = []
        agreement = 0
        async with conn.transaction():
            await conn.execute("SET LOCAL enable_indexscan = off")
            plan = "\n".join(r[0] for r in await conn.fetch(f"EXPLAIN {sql}", query_vectors[0]))
            if "sweep_hnsw" in plan:
                raise RuntimeError("exact baseline unexpectedly used the index")
            for q in query_vectors[:WARMUP]:
                await conn.fetch(sql, q)
            for q, truth in zip(query_vectors, exact, strict=True):
                started = time.perf_counter()
                got = await conn.fetch(sql, q)
                exact_rows.append((time.perf_counter() - started) * 1000)
                agreement += len({r["id"] for r in got} & truth)
        exact_result = {
            "plan": plan.splitlines(),
            "recall@10_vs_numpy": round(agreement / (K * len(queries)), 4),
            "latency_ms": _latency(exact_rows),
        }
        print(f"exact scan: {exact_result['recall@10_vs_numpy']} {exact_result['latency_ms']}")  # noqa: T201

        results: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        for m in GRID_M:
            for efc in GRID_EF_CONSTRUCTION:
                if efc < 2 * m:
                    skipped.append(
                        {
                            "m": m,
                            "ef_construction": efc,
                            "reason": "pgvector requires ef_construction >= 2 * m",
                        }
                    )
                    continue
                await conn.execute("DROP INDEX IF EXISTS sweep_hnsw")
                started = time.perf_counter()
                await conn.execute(
                    f"CREATE INDEX sweep_hnsw ON sweep_vectors USING hnsw "
                    f"(embedding vector_cosine_ops) WITH (m = {m}, ef_construction = {efc})"
                )
                build_s = time.perf_counter() - started
                size = await conn.fetchval("SELECT pg_relation_size('sweep_hnsw')")
                await conn.execute("ANALYZE sweep_vectors")
                for ef in GRID_EF_SEARCH:
                    latencies: list[float] = []
                    hits = 0
                    async with conn.transaction():
                        await conn.execute("SELECT set_config('hnsw.ef_search', $1, true)", str(ef))
                        await conn.execute("SET LOCAL hnsw.iterative_scan = off")
                        await conn.execute("SET LOCAL enable_seqscan = off")
                        plan = "\n".join(
                            r[0] for r in await conn.fetch(f"EXPLAIN {sql}", query_vectors[0])
                        )
                        if "Index Scan using sweep_hnsw" not in plan:
                            raise RuntimeError(f"HNSW index not used:\n{plan}")
                        for q in query_vectors[:WARMUP]:
                            await conn.fetch(sql, q)
                        for q, truth in zip(query_vectors, exact, strict=True):
                            started = time.perf_counter()
                            got = await conn.fetch(sql, q)
                            latencies.append((time.perf_counter() - started) * 1000)
                            hits += len({r["id"] for r in got} & truth)
                    row: dict[str, Any] = {
                        "m": m,
                        "ef_construction": efc,
                        "ef_search": ef,
                        "ann_recall@10": round(hits / (K * len(queries)), 4),
                        "latency_ms": _latency(latencies),
                        "index_build_s": round(build_s, 3),
                        "index_bytes": int(size),
                        "plan_uses_hnsw": True,
                        "production_setting": m == PRODUCTION["m"]
                        and efc == PRODUCTION["ef_construction"],
                    }
                    results.append(row)
                    print(  # noqa: T201
                        f"m={m:<3} efc={efc:<4} ef={ef:<4} recall@10={row['ann_recall@10']:.4f} "
                        f"p50={row['latency_ms']['p50']}ms"
                    )
        await conn.execute("DROP TABLE sweep_vectors")
    finally:
        await conn.close()

    payload = {
        "stage": "hnsw_sweep",
        "description": __doc__,
        "corpus": {
            "vectors": len(ids),
            "splits": ["train", "dev"],
            "dimension": 384,
            "distance": "cosine (vector_cosine_ops)",
        },
        "queries": {
            "total": len(queries),
            "hypotheses": len(hypotheses),
            "sampled_dev_evidence_spans": SAMPLED_SPANS,
            "query_instruction": instruction,
            "seed": SEED,
        },
        "k": K,
        "warmup_queries_per_setting": WARMUP,
        "latency_scope": (
            "client wall time per SELECT over asyncpg on localhost, after warm-up; "
            "excludes query embedding; single connection, sequential"
        ),
        "exact_reference": exact_result,
        "results": results,
        "skipped_combinations": skipped,
        "provenance": provenance.collect(command, SEED),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "hnsw_sweep.json").write_text(json.dumps(payload, indent=2) + "\n")
    plots.sweep_curves(results, RESULTS / "hnsw_sweep.png")
    print("wrote evals/results/hnsw_sweep.json and hnsw_sweep.png")  # noqa: T201
    return payload


async def locked(command: list[str]) -> dict[str, Any]:
    async with corpus_db.EvalLock(corpus_db.eval_database_url()):
        return await run(command)


def main() -> int:
    asyncio.run(locked(["python", "-m", "evals.retrieval.sweep", *sys.argv[1:]]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
