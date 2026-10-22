"""ContractNLI for the rule-check model: official splits, ingestion segmentation, gold.

**Splits.** The official ``train`` / ``dev`` / ``test`` files of the SHA-256-pinned
archive, unchanged. Every (document, hypothesis) annotation becomes one labelled pair
(Entailment / Contradiction / NotMentioned); nothing is excluded.

**Spans (model inputs).** Each contract is segmented exactly like the application's
ingestion: wrapped lines are grouped into paragraph/heading blocks
(``evals.retrieval.data.line_blocks``, Phase 2) and every block is split into sentences
by ``compliance_ingestion.sentences.sentence_spans`` (the same splitter the chunker
uses). Segments are ``[start, end)`` offsets into the unchanged canonical text. The
segmentation never looks at labels or evidence.

**Gold evidence (targets only).** ContractNLI annotates evidence as indices into its own
official span list. Official spans and ingestion segments are different units: an
official span is often finer than a sentence (list items, headings, clause parts), so
one segment can contain several official spans; occasionally the sentence splitter cuts
inside an official span, so one official span can overlap two or more segments. The
fixed rule (chosen by definition, not tuned on any split) is overlap:

- a segment and an official span *overlap* iff they share at least one non-whitespace
  character;
- a segment is a *gold evidence segment* for a pair iff it overlaps one of the pair's
  gold official spans.

``prepare`` reports how often each direction of the many-to-many relation occurs. Gold
is used for training targets and evaluation only, never as inference input.

Evaluation units: evidence metrics are reported (a) on segments (what the application
cites) and (b) projected onto the official spans: an official span's score is the
maximum score of the segments it overlaps, which is comparable with the ContractNLI
paper's span-level evaluation.
"""

from __future__ import annotations

import bisect
import json
import zipfile
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from compliance_ingestion.sentences import sentence_spans
from compliance_model_service.rule_check.labels import DATASET_LABELS, LABELS
from evals.retrieval.data import line_blocks

SPLITS = ("train", "dev", "test")


@dataclass(frozen=True)
class Pair:
    split: str
    doc_id: str
    key: str  # ContractNLI hypothesis key, e.g. "nda-7"
    label: str  # one of LABELS
    gold_official: tuple[int, ...]  # indices into Document.official_spans
    gold_segments: tuple[int, ...]  # indices into Document.segments

    @property
    def pair_id(self) -> str:
        return f"{self.split}:{self.doc_id}:{self.key}"


@dataclass
class Document:
    split: str
    doc_id: str
    file_name: str
    text: str
    segments: list[tuple[int, int]]
    official_spans: list[tuple[int, int]]
    # official span index -> indices of the segments it overlaps (empty: whitespace-only)
    official_to_segments: list[tuple[int, ...]] = field(default_factory=list)

    def segment_text(self, index: int) -> str:
        start, end = self.segments[index]
        return self.text[start:end]


@dataclass
class SplitData:
    split: str
    documents: dict[str, Document]
    pairs: list[Pair]
    hypotheses: dict[str, str]


def segment(text: str) -> list[tuple[int, int]]:
    """Ingestion segmentation of a canonical text (paragraph blocks -> sentences)."""
    spans: list[tuple[int, int]] = []
    for block in line_blocks(text):
        spans.extend(sentence_spans(text, block.char_start, block.char_end))
    return spans


def strip_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def overlaps(text: str, a: tuple[int, int], b: tuple[int, int]) -> bool:
    """True iff the ranges share at least one non-whitespace character."""
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    return any(not text[i].isspace() for i in range(lo, hi))


def map_official_to_segments(
    text: str, segments: Sequence[tuple[int, int]], official: Sequence[tuple[int, int]]
) -> list[tuple[int, ...]]:
    """For each official span, the (sorted) indices of the segments it overlaps."""
    starts = [s for s, _ in segments]
    result: list[tuple[int, ...]] = []
    for raw_start, raw_end in official:
        start, end = strip_span(text, raw_start, raw_end)
        hits: list[int] = []
        if start < end:
            j = max(bisect.bisect_right(starts, start) - 1, 0)  # sorted, disjoint segments
            while j < len(segments) and segments[j][0] < end:
                if overlaps(text, segments[j], (start, end)):
                    hits.append(j)
                j += 1
            if not hits:
                raise ValueError(f"official span [{start}, {end}) overlaps no segment")
        result.append(tuple(hits))
    return result


def load_raw(archive: Path, split: str) -> dict[str, Any]:
    with zipfile.ZipFile(archive) as zf:
        raw: dict[str, Any] = json.loads(zf.read(f"contract-nli/{split}.json"))
    return raw


def build_split(archive: Path, split: str, doc_ids: Sequence[str] | None = None) -> SplitData:
    raw = load_raw(archive, split)
    hypotheses = {k: str(v["hypothesis"]) for k, v in raw["labels"].items()}
    wanted = set(doc_ids) if doc_ids is not None else None
    documents: dict[str, Document] = {}
    pairs: list[Pair] = []
    for item in raw["documents"]:
        doc_id = str(item["id"])
        if wanted is not None and doc_id not in wanted:
            continue
        text = str(item["text"])
        official = [(int(s), int(e)) for s, e in item["spans"]]
        segments = segment(text)
        doc = Document(split, doc_id, str(item["file_name"]), text, segments, official)
        doc.official_to_segments = map_official_to_segments(text, segments, official)
        documents[doc_id] = doc
        annotations = item["annotation_sets"][0]["annotations"]
        for key in sorted(annotations, key=_key_order):
            annotation = annotations[key]
            label = DATASET_LABELS[str(annotation["choice"])]
            gold_official = tuple(int(i) for i in annotation["spans"])
            gold_segments = sorted({j for i in gold_official for j in doc.official_to_segments[i]})
            pairs.append(Pair(split, doc_id, key, label, gold_official, tuple(gold_segments)))
    if wanted is not None and set(documents) != wanted:
        raise ValueError(f"{split}: unknown document ids {sorted(wanted - set(documents))}")
    return SplitData(split, documents, pairs, hypotheses)


def _key_order(key: str) -> tuple[int, str]:
    tail = key.rsplit("-", 1)[-1]
    return (int(tail) if tail.isdigit() else 10**6, key)


def label_counts(pairs: Sequence[Pair]) -> dict[str, int]:
    counts = Counter(p.label for p in pairs)
    return {label: counts.get(label, 0) for label in LABELS}


def hypothesis_keys(hypotheses: dict[str, str]) -> list[str]:
    return sorted(hypotheses, key=_key_order)
