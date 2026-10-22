"""Weighted reciprocal rank fusion with deterministic tie handling.

For each method m with ranked candidates (rank 1 = best), a chunk's fused score is
``sum_m w_m / (k + rank_m)``; methods that did not return the chunk contribute 0, and a
method with weight 0 is ignored entirely (it neither scores nor adds candidates).
Duplicate chunk ids within one method's list keep only their best-scored occurrence.

Ties are broken deterministically by: higher fused score, then better (smaller) best
rank across methods, then more methods agreeing, then chunk id (lexicographic). Raw
per-method scores and ranks are kept as provenance.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Candidate:
    chunk_id: str
    score: float  # method-native score (similarity or ts_rank_cd); higher is better


@dataclass
class FusedHit:
    chunk_id: str
    fused_score: float
    ranks: dict[str, int] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)

    @property
    def best_rank(self) -> int:
        return min(self.ranks.values()) if self.ranks else 10**9


def order_candidates(candidates: Sequence[Candidate]) -> list[Candidate]:
    """Deterministic order for one method: score desc, then chunk id."""
    return sorted(candidates, key=lambda c: (-c.score, c.chunk_id))


def reciprocal_rank_fusion(
    ranked: Mapping[str, Sequence[Candidate]],
    weights: Mapping[str, float],
    k: int,
) -> list[FusedHit]:
    if k < 1:
        raise ValueError("RRF constant must be >= 1")
    hits: dict[str, FusedHit] = {}
    for method, candidates in ranked.items():
        weight = weights.get(method, 0.0)
        if weight < 0:
            raise ValueError("weights must be non-negative")
        if weight == 0:
            continue
        seen: set[str] = set()
        rank = 0
        for candidate in order_candidates(candidates):
            if candidate.chunk_id in seen:
                continue  # a method contributes at most once per chunk
            seen.add(candidate.chunk_id)
            rank += 1
            hit = hits.setdefault(candidate.chunk_id, FusedHit(candidate.chunk_id, 0.0))
            hit.fused_score += weight / (k + rank)
            hit.ranks[method] = rank
            hit.scores[method] = candidate.score
    return sorted(
        hits.values(),
        # Rounded so that mathematically equal sums (e.g. 1/3+1/3 vs 1/2+1/6) tie exactly.
        key=lambda h: (-round(h.fused_score, 12), h.best_rank, -len(h.ranks), h.chunk_id),
    )


def single_method(method: str, candidates: Sequence[Candidate]) -> list[FusedHit]:
    """Rank one method's candidates with the same provenance shape (no fusion)."""
    out: list[FusedHit] = []
    seen: set[str] = set()
    for candidate in order_candidates(candidates):
        if candidate.chunk_id in seen:
            continue
        seen.add(candidate.chunk_id)
        rank = len(out) + 1
        out.append(
            FusedHit(candidate.chunk_id, candidate.score, {method: rank}, {method: candidate.score})
        )
    return out
