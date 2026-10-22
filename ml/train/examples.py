"""Per-epoch TRAIN example construction (seeded; TRAIN only; never used at inference).

Evidence examples (``[CLS] h [SEP] span [SEP]`` -> 0/1):
- every gold evidence segment of the pair (target 1);
- up to ``hard_negatives`` non-gold segments sampled from the pair's top
  ``hard_negative_depth`` dense-ranked segments (target 0);
- up to ``random_negatives`` other non-gold segments sampled uniformly (target 0).
NotMentioned pairs contribute negatives only.

Label examples (``[CLS] h [SEP] s1 [SEP] ... sk [SEP]`` -> label), one per pair, with
``k`` drawn from ``label_k_choices``:
- evidence pairs keep their gold segments (a random subset if more than k), and with
  probability ``gold_drop_prob`` one of them is dropped while at least one remains;
- the remaining slots are filled from the dense ranking (non-gold, best first);
- NotMentioned pairs read the top-k dense-ranked segments.
Spans are put in document order, as at inference.

At inference the label head instead reads the *evidence head's own* top-k. Gold is
never available there.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from ml.data.contractnli import Pair
from ml.data.store import Dataset
from ml.train.config import SamplingSettings


@dataclass(frozen=True)
class EvidenceExample:
    doc_id: str
    key: str
    segment: int
    target: float


@dataclass(frozen=True)
class LabelExample:
    doc_id: str
    key: str
    segments: tuple[int, ...]  # document order
    target: int  # index into LABELS


def evidence_examples(
    dataset: Dataset, pair: Pair, sampling: SamplingSettings, rng: random.Random
) -> list[EvidenceExample]:
    doc = dataset.document(pair)
    gold = set(pair.gold_segments)
    out = [EvidenceExample(pair.doc_id, pair.key, s, 1.0) for s in sorted(gold)]
    hard_pool = [s for s in dataset.retrieved(pair, sampling.hard_negative_depth) if s not in gold]
    hard = rng.sample(hard_pool, min(sampling.hard_negatives, len(hard_pool)))
    used = gold | set(hard)
    rest = [s for s in range(len(doc.segments)) if s not in used]
    rand = rng.sample(rest, min(sampling.random_negatives, len(rest)))
    out.extend(EvidenceExample(pair.doc_id, pair.key, s, 0.0) for s in sorted([*hard, *rand]))
    return out


def label_example(
    dataset: Dataset,
    pair: Pair,
    sampling: SamplingSettings,
    rng: random.Random,
    label_index: dict[str, int],
) -> LabelExample:
    doc = dataset.document(pair)
    k = min(rng.choice(sampling.label_k_choices), len(doc.segments))
    gold = list(pair.gold_segments)
    keep = sorted(rng.sample(gold, k)) if len(gold) > k else gold
    if len(keep) > 1 and rng.random() < sampling.gold_drop_prob:
        keep.remove(rng.choice(keep))
    chosen = set(keep)
    for seg in dataset.retrieved(pair, len(doc.segments)):
        if len(chosen) >= k:
            break
        if seg not in pair.gold_segments:
            chosen.add(seg)
    return LabelExample(pair.doc_id, pair.key, tuple(sorted(chosen)), label_index[pair.label])


def build_epoch(
    dataset: Dataset,
    pairs: list[Pair],
    sampling: SamplingSettings,
    *,
    seed: int,
    epoch: int,
    label_index: dict[str, int],
) -> tuple[list[EvidenceExample], list[LabelExample]]:
    rng = random.Random(f"{seed}:{epoch}")  # noqa: S311 - reproducible sampling, not security
    evidence: list[EvidenceExample] = []
    labels: list[LabelExample] = []
    for pair in pairs:
        evidence.extend(evidence_examples(dataset, pair, sampling, rng))
        labels.append(label_example(dataset, pair, sampling, rng, label_index))
    return evidence, labels
