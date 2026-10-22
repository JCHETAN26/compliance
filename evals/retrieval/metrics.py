"""Evidence-retrieval metrics (document-scoped, chunk-level).

Definitions (fixed before any results were looked at):

- **Coverage criterion.** A retrieved chunk *covers* a gold evidence span when their
  character overlap is at least ``COVERAGE`` (50 %) of the span's length. Chunks are
  ~200-400 tokens and spans are sentences/clauses (median ~23 tokens), so a covering
  chunk normally contains the whole span; the 50 % rule only matters for spans that
  straddle a chunk boundary. A span that no chunk of its document covers is
  *unreachable* (counted and reported; it still counts as a miss).
- **Evidence recall@k** (primary family). For one (document, hypothesis) pair:
  |gold spans covered by any of the top-k chunks| / |gold spans|. Reported as the
  mean over pairs (macro).
- **Hit@k.** 1 if any top-k chunk covers any gold span, else 0; mean over pairs.
- **MRR@10.** Reciprocal rank of the first chunk (within the top 10) that covers any
  gold span, else 0; mean over pairs.
- **Random baseline.** The exact expectation of each metric under a uniformly random
  ordering of the document's chunks (closed form, no sampling), to show how much
  ranking skill a number reflects given only ~9 chunks per document.

The primary selection metric for DEV tuning is mean evidence recall@3, with MRR@10 as
the tie-breaker.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import comb

COVERAGE = 0.5
KS = (1, 3, 5, 10)
MRR_CUTOFF = 10
PRIMARY = "recall@3"

Span = tuple[int, int]


def covers(chunk: Span, span: Span, threshold: float = COVERAGE) -> bool:
    start, end = span
    length = end - start
    if length <= 0:
        return False
    overlap = min(chunk[1], end) - max(chunk[0], start)
    return overlap >= threshold * length


@dataclass(frozen=True)
class PairScore:
    recall: dict[int, float]
    hit: dict[int, float]
    rr: float


def score_ranking(ranked: Sequence[Span], gold: Sequence[Span]) -> PairScore:
    """Score one ranked list of chunk ranges against a pair's gold spans."""
    if not gold:
        raise ValueError("pairs without gold evidence are excluded, not scored")
    first_hit: int | None = None
    covered_at: list[int | None] = []
    for span in gold:
        rank = next((i for i, chunk in enumerate(ranked, 1) if covers(chunk, span)), None)
        covered_at.append(rank)
        if rank is not None and (first_hit is None or rank < first_hit):
            first_hit = rank
    recall = {k: sum(1 for r in covered_at if r is not None and r <= k) / len(gold) for k in KS}
    hit = {k: float(first_hit is not None and first_hit <= k) for k in KS}
    rr = 1.0 / first_hit if first_hit is not None and first_hit <= MRR_CUTOFF else 0.0
    return PairScore(recall, hit, rr)


def _p_none_in_top_k(n: int, relevant: int, k: int) -> float:
    """P(no relevant item in the first k positions of a uniform permutation)."""
    k = min(k, n)
    return comb(n - relevant, k) / comb(n, k) if n - relevant >= k else 0.0


def random_baseline(chunks: Sequence[Span], gold: Sequence[Span]) -> PairScore:
    """Exact expected scores under a uniformly random ordering of ``chunks``."""
    n = len(chunks)
    per_span = [sum(1 for c in chunks if covers(c, s)) for s in gold]
    any_relevant = sum(1 for c in chunks if any(covers(c, s) for s in gold))
    recall = {
        k: sum(1 - _p_none_in_top_k(n, r, k) if r else 0.0 for r in per_span) / len(gold)
        for k in KS
    }
    hit = {k: (1 - _p_none_in_top_k(n, any_relevant, k)) if any_relevant else 0.0 for k in KS}
    # E[1/R] where R = rank of the first relevant item: P(R = i) = P(none in first i-1)
    # - P(none in first i).
    rr = 0.0
    if any_relevant:
        for i in range(1, min(n, MRR_CUTOFF) + 1):
            p = _p_none_in_top_k(n, any_relevant, i - 1) - _p_none_in_top_k(n, any_relevant, i)
            rr += p / i
    return PairScore(recall, hit, rr)


def aggregate(scores: Sequence[PairScore]) -> dict[str, float]:
    if not scores:
        return {}
    out: dict[str, float] = {}
    for k in KS:
        out[f"recall@{k}"] = sum(s.recall[k] for s in scores) / len(scores)
        out[f"hit@{k}"] = sum(s.hit[k] for s in scores) / len(scores)
    out[f"mrr@{MRR_CUTOFF}"] = sum(s.rr for s in scores) / len(scores)
    return {key: round(value, 4) for key, value in out.items()}


def selection_key(metrics: dict[str, float]) -> tuple[float, float]:
    return (metrics[PRIMARY], metrics[f"mrr@{MRR_CUTOFF}"])
