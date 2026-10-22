# evals/ — retrieval evaluation (Phase 2)

The end-to-end report evaluation, thresholds and dashboard are Phase 6 and do **not**
exist yet. This directory holds the Phase 2 retrieval evaluation only.

```bash
make test-infra-up       # throwaway Postgres/pgvector on 127.0.0.1:55432 (tmpfs)
make fetch-artifacts     # pinned encoder + reranker + ContractNLI into ./.cache
make eval-retrieval      # DEV tuning -> freeze -> TEST -> HNSW sweep
# stages: make eval-retrieval-dev | eval-retrieval-test | eval-hnsw-sweep
```

Outputs:
- `evals/results/retrieval_dev.json`: grid, single methods, the selection and an
  end-to-end check.
- `evals/retrieval/frozen_config.json`: the selected configuration and its hash.
- `evals/results/retrieval_test.json`: methods, bootstrap CIs, per-label scores,
  timings and the query plan.
- `evals/results/hnsw_sweep.json`.
- Plots `retrieval_dev_grid.png`, `retrieval_test_methods.png` and `hnsw_sweep.png`.
  The JSON files are the table view.

Every JSON file records:
- the command, UTC timestamp and seed (`20261021`);
- the actual `git rev-parse HEAD` with the dirty state and number of dirty paths
  (results are produced before commit);
- a SHA-256 fingerprint of the content of every file that can change the numbers
  (`packages/{core,ingestion,retrieval}/src`, `apps/model-service/src`,
  `evals/retrieval`, `pyproject.toml`, `uv.lock`);
- the model ids, revisions and SHA-256s, and the package versions.

## What is measured

**Task: document-scoped evidence retrieval.** For each ContractNLI (contract,
hypothesis) pair labelled Entailment or Contradiction, the hypothesis is the query, the
search is restricted to that one contract, and the gold set is the pair's annotated
evidence spans. This is ContractNLI's own setting ("find the evidence in this
contract"). It is **not** workspace-global search quality.

- **Data:** the official ContractNLI archive (SHA-256 pinned) and its official
  train/dev/test splits, unchanged.
  - NotMentioned pairs have no evidence and are excluded and counted: 423 in DEV, 903
    in TEST. No evidence-labelled pair lacks spans.
  - There are 17 unique hypothesis texts, so 17 unique queries.
- **Corpus:**
  - Each document's canonical `text`, unchanged, is indexed in an **isolated database**
    (`compliance_eval` on the test Postgres) with the production migrations, chunker,
    `Embedder` and SQL.
  - Wrapped lines are grouped into paragraph blocks (a line ends a paragraph only if it
    ends in `.;:!?`), so a sentence is never split at a line break.
  - Chunk offsets satisfy `text[start:end] == chunk`, and gold offsets are untouched.
  - Every gold span is reachable by some chunk under the hit rule (0 unreachable in
    DEV and TEST).
  - One exact duplicate text in TEST is stored in an overflow workspace, and both
    official documents are evaluated.
  - This is isolated text retrieval: no PDF rendering, OCR, HTTP API or report
    generation is involved. Those paths are covered by the integration, smoke and
    browser checks instead.
- **Methods:** all run through production code. Candidates come from
  `compliance_retrieval.sql`, fusion from `reciprocal_rank_fusion`, and end-to-end
  results from `run_search`. Embeddings and rerank scores come from the pinned models,
  in-process: the same functions the model service runs, without the HTTP hop.
  - keyword: FTS `ts_rank_cd`;
  - vector: pgvector cosine;
  - hybrid: weighted RRF;
  - hybrid+rerank: the cross-encoder over the top 30 fused hits.
- **Metrics:** defined before results; see `evals/retrieval/metrics.py`.
  - **Hit rule:** a chunk covers a gold span if it contains ≥ 50 % of the span's
    characters.
  - **Evidence recall@k:** the share of a pair's gold spans covered by the top-k
    chunks, averaged over pairs. **Primary metric: recall@3** (tie-break MRR@10).
  - **hit@k:** whether any gold span is covered within the top k.
  - **MRR@10:** the reciprocal rank of the first covering chunk.
  - **Random:** the exact expected score under a random ordering of the document's
    chunks.
- **Interpretability caveat:** documents are short.
  - TEST has a median of 7 chunks per document, and 72 % of pairs have ≤ 10 chunks in
    their document, so recall@10 is near-trivial (random gets 0.92).
  - Use @1/@3 and MRR.
- **Protocol:**
  - Grid on DEV only: query instruction on/off × keyword mode × RRF k ∈
    {1,5,10,20,40,60,100} × keyword weight ∈ {0.25…4} (vector weight 1), 196 points.
  - Selection by DEV recall@3. The selected config is re-run end to end through
    `run_search` on DEV, and its rankings must equal the offline fusion. It is then
    written to `frozen_config.json` with a hash.
  - The TEST stage reads only that file and refuses to run if its hash doesn't match.
  - **Freeze history** (no parameter was ever revised after TEST):
    1. The DEV selection was frozen at 2026-10-21T23:42:44Z with hash
       `6cb1378fdec77f44581e184ff11c5072d7feead6fbb63019191b488c5f491516` (code
       fingerprint `37318357…`). A copy of that file is at
       `/tmp/cw-eval/frozen_config.run2.json`.
    2. The first TEST run read that file (23:53:32Z).
    3. The canonical results come from the later full `make eval-retrieval`. It re-ran
       DEV, which reselected the same configuration with the same hash (the file's
       `frozen_at_utc` therefore reads 2026-10-22T00:03:45Z). It then repeated the
       same TEST measurement unchanged to verify reproducibility. Every TEST metric was
       identical; only latencies differ.
    4. HNSW sweep numbers come from graph construction and timing, and are not exactly
       reproducible run to run.
  - **Component comparison, not separately optimised baselines:** instruction,
    keyword mode and fusion were selected jointly for hybrid, and vector/keyword are
    reported with that shared configuration. On DEV these are also the best standalone
    choices: vector with the instruction scores recall@3 0.774 vs 0.767 without it, and
    keyword `or` mode 0.766 vs 0.121 for `websearch`. So no single method is
    disadvantaged by the shared choice.
  - An untuned reference (k=60, equal weights) is also reported on TEST. It plays no
    part in selection.
  - Differences carry 95 % CIs from a document-clustered paired bootstrap (2000
    resamples).

## Results (TEST, frozen DEV configuration; reproduced unchanged by the final run)

The DEV selection was: BGE query instruction on, keyword mode `or`, RRF k = 5, keyword
weight 1.5, vector weight 1. Config hash `6cb1378f…`.

| Method (1188 pairs, 123 docs) | recall@1 | recall@3 | recall@5 | recall@10 | MRR@10 |
|---|---:|---:|---:|---:|---:|
| random (expected) | 0.148 | 0.433 | 0.671 | 0.919 | 0.387 |
| keyword (FTS `ts_rank_cd`) | 0.416 | 0.789 | 0.918 | 0.989 | 0.658 |
| vector (pgvector, bge-small) | **0.455** | 0.811 | 0.934 | 0.991 | **0.692** |
| hybrid (RRF, DEV-tuned) | 0.432 | **0.816** | 0.935 | 0.992 | 0.679 |
| hybrid, untuned k=60 1:1 (reference) | 0.452 | 0.816 | 0.937 | 0.991 | 0.695 |
| **hybrid + cross-encoder rerank** | **0.549** | **0.860** | **0.955** | **0.996** | **0.763** |

Honest reading (from `paired_bootstrap_vs` in the JSON):
- **Hybrid beats keyword** with confidence intervals excluding 0: recall@3 +0.028
  [+0.014, +0.042] and MRR +0.021 [+0.012, +0.029].
- **Hybrid does not significantly beat vector.**
  - recall@3: +0.005 [−0.014, +0.021].
  - MRR: −0.013 [−0.032, +0.007].
  - Hybrid is also lower at recall@1 (0.432 vs 0.455).
  - The handoff's target ("hybrid beats at least one single method") is met only
    against keyword.
- **DEV tuning did not transfer.**
  - On DEV the grid is flat: selected 0.819 vs 0.813 for k=60/1:1.
  - On TEST the tuned and untuned hybrids are equal on recall@3 (0.816), and the
    untuned one is slightly better on recall@1 and MRR.
  - The frozen benchmark configuration is k = 5, keyword weight 1.5.
  - **App defaults:** the application's defaults (`search_rrf_k` /
    `search_keyword_weight` in `compliance_core.config` and `.env.example`) were set
    to these same frozen DEV values before the canonical run. They are part of the
    recorded code fingerprint.
  - That choice rests on DEV alone; TEST shows no advantage over standard RRF.
    `SEARCH_RRF_K=60 SEARCH_KEYWORD_WEIGHT=1.0` restores k = 60 with equal weights
    without code changes. Nothing was revised from TEST outcomes.
- **Reranking is the clear gain:**
  - recall@3 +0.044 [+0.021, +0.069] and MRR +0.084 [+0.065, +0.103];
  - at a CPU cost of p50 379 ms / p95 1259 ms of rerank time per query;
  - it stays **off by default** (`RERANK_ENABLED`).
- **Other findings:**
  - On DEV, `websearch` keyword mode (all terms required) scores recall@3 0.12 for
    hypothesis-style queries, which is why `or` is the default.
  - The BGE query instruction helps vector search slightly on DEV (0.774 vs 0.767).

**Latency** (`timings_ms`, p50/p95, 1188 sequential queries):
- total: keyword 2.4/2.9 ms, vector 4.3/5.8 ms, hybrid 4.4/6.0 ms;
- the scope is in-process on localhost Postgres on tmpfs, with query embeddings
  memoised (17 distinct queries), so `embed_ms` is a cache hit and not live inference.

**Query plan:**
- A document-scoped vector query is planned as an exact sort over the document's few
  rows, not the HNSW index (`document_scoped_vector_plan`). That is the planner's
  correct choice at this selectivity; the ANN index is measured separately below.

## HNSW sweep (ANN quality vs latency; not evidence recall)

**Setup:**
- **Corpus:** 4047 chunk vectors from train + dev (test untouched).
- **Queries:** 300 = the 17 hypotheses + 283 seeded DEV evidence-span texts, embedded
  as queries.
- **Ground truth:** exact cosine top-10 in numpy. An SQL exact scan agrees 100 %
  (p50 1.14 ms).
- **Grid:** m ∈ {8,16,32} × ef_construction ∈ {32,64,128,256} × ef_search ∈
  {10,20,40,80,100,160}. m=32/ef_construction=32 is skipped, because pgvector requires
  ef_construction ≥ 2m.
- **Verification:** every setting's plan was checked to be `Index Scan using
  sweep_hnsw`, with 20 warm-up queries per setting.
- **Latency:** client wall time per `SELECT … ORDER BY embedding <=> $1 LIMIT 10` over
  asyncpg on localhost.

**Results:**
- **Production index (m=16, ef_construction=64):** recall@10 is 0.861 / 0.934 /
  0.965 / 0.987 / **0.990** / 0.996 at ef_search 10 / 20 / 40 / 80 / **100** / 160.
  The p50 is 0.20 → 0.64 ms (**0.47 ms at the default ef_search = 100**). The index is
  8.2 MB and builds in 0.7 s.
- **Cheapest setting with ANN recall ≥ 0.99:** m=16, ef_construction=128, ef_search=80
  (p50 0.42 ms).
- **Caveat:** with only ~4k vectors, every setting is sub-millisecond and recall
  saturates. These numbers describe this corpus size, not a large deployment.

## Limitations

- Chunk-level retrieval within one contract only. Recall@k counts covered evidence
  spans, not whether the chunk alone suffices for an NLI decision.
- The 17 queries repeat across documents. The bootstrap resamples documents, but the
  query set is small and fixed.
- The reranker was trained on MS MARCO web passages, not contracts. Its score is
  truncated to 512 tokens per (query, chunk).
- Results come from canonical text. Real uploads add PDF/OCR extraction noise, which
  this eval does not measure.
