"""ContractNLI as a retrieval benchmark: documents, chunks and gold evidence.

Data: the official ContractNLI archive (SHA-256 pinned in ``seed/.../manifest.yaml``),
official ``train`` / ``dev`` / ``test`` splits, unchanged.

Corpus construction (isolated *text* retrieval, no PDF rendering or OCR):

- A document's canonical text is ``doc["text"]`` exactly as distributed; gold evidence
  spans are ``doc["spans"][i] = [start, end)`` character offsets into it, unchanged.
- Blocks group the text's wrapped lines into paragraphs and headings (see
  ``line_blocks``); offsets stay document-level and the text is never modified.
- Chunks come from the production chunker (``compliance_ingestion.chunking`` with the
  production ``ChunkOptions`` and the pinned encoder's tokenizer), and are embedded by
  the production ``Embedder`` (same windowing) - exactly what ingestion stores.

Queries and gold: for every (document, hypothesis) pair whose annotation is
``Entailment`` or ``Contradiction``, the query is the hypothesis text and the gold set is
that pair's annotated evidence spans. ``NotMentioned`` pairs have no evidence and are
excluded (and counted); so are evidence-labelled pairs with zero spans, if any.
"""

from __future__ import annotations

import json
import re
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from compliance_ingestion.chunking import ChunkOptions, build_chunks
from compliance_ingestion.layout import RawBlock, assign_sections, looks_like_heading
from compliance_ingestion.tokens import HFTokenCounter

SPLITS = ("train", "dev", "test")
EVIDENCE_LABELS = ("Entailment", "Contradiction")


@dataclass(frozen=True)
class EvalChunk:
    index: int
    char_start: int
    char_end: int
    text: str
    token_count: int
    section_path: str | None


@dataclass
class EvalDocument:
    split: str
    doc_id: str
    file_name: str
    text: str
    spans: list[tuple[int, int]]
    chunks: list[EvalChunk] = field(default_factory=list)


@dataclass(frozen=True)
class EvalPair:
    split: str
    doc_id: str
    hypothesis_key: str
    hypothesis: str
    label: str
    gold: tuple[tuple[int, int], ...]  # evidence spans (char ranges in the canonical text)


@dataclass
class SplitData:
    split: str
    documents: dict[str, EvalDocument]
    pairs: list[EvalPair]
    label_counts: dict[str, int]
    excluded_no_evidence: int  # NotMentioned pairs
    excluded_empty_evidence: int  # Entailment/Contradiction pairs without spans


def load_split(archive: Path, split: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with zipfile.ZipFile(archive) as zf:
        raw = json.loads(zf.read(f"contract-nli/{split}.json"))
    return list(raw["documents"]), dict(raw["labels"])


# A line ending like this closes its paragraph (optionally followed by closing quotes or
# brackets). Anything else is a wrapped line that the next line continues.
_PARAGRAPH_END = re.compile(r"[.;:!?][\"'\u201d\u2019)\]]*$")


def line_blocks(text: str) -> list[RawBlock]:
    """Group the canonical text's non-blank lines into paragraph and heading blocks.

    ContractNLI keeps the source PDFs' line wrapping as ``\n`` and has no blank-line
    paragraph breaks, so a line break is not a sentence or paragraph boundary. As in
    the production PDF parser (lines of one block joined, sentences split across them),
    consecutive lines form one paragraph block until a line ends with sentence or
    clause punctuation; a heading-like line (production ``looks_like_heading``) is its
    own block only when no paragraph is open. Block offsets are document-level ranges of
    the unchanged text (inner newlines included), and nothing depends on gold labels.
    """
    blocks: list[RawBlock] = []
    open_start: int | None = None
    open_end = 0
    for match in re.finditer(r"[^\n]+", text):
        line = match.group(0)
        lead = len(line) - len(line.lstrip())
        end = len(line.rstrip())
        if end <= lead:
            continue
        start, stop = match.start() + lead, match.start() + end
        stripped = text[start:stop]
        if open_start is None and looks_like_heading(stripped, emphasised=False):
            blocks.append(RawBlock(char_start=start, char_end=stop, kind="heading"))
            continue
        if open_start is None:
            open_start = start
        open_end = stop
        if _PARAGRAPH_END.search(stripped):
            blocks.append(RawBlock(char_start=open_start, char_end=open_end, kind="paragraph"))
            open_start = None
    if open_start is not None:
        blocks.append(RawBlock(char_start=open_start, char_end=open_end, kind="paragraph"))
    return blocks


def chunk_document(text: str, counter: HFTokenCounter, options: ChunkOptions) -> list[EvalChunk]:
    blocks = assign_sections(line_blocks(text), text)
    return [
        EvalChunk(
            index=c.chunk_index,
            char_start=c.char_start,
            char_end=c.char_end,
            text=c.text,
            token_count=c.token_count,
            section_path=c.section_path,
        )
        for c in build_chunks(text, blocks, counter, options)
    ]


def build_split(
    archive: Path, split: str, counter: HFTokenCounter, options: ChunkOptions
) -> SplitData:
    raw_docs, labels = load_split(archive, split)
    documents: dict[str, EvalDocument] = {}
    pairs: list[EvalPair] = []
    label_counts: Counter[str] = Counter()
    no_evidence = empty = 0
    for raw in raw_docs:
        doc_id = str(raw["id"])
        text = str(raw["text"])
        spans = [(int(s), int(e)) for s, e in raw["spans"]]
        document = EvalDocument(split, doc_id, str(raw["file_name"]), text, spans)
        document.chunks = chunk_document(text, counter, options)
        for chunk in document.chunks:  # offsets are exact slices of the canonical text
            if text[chunk.char_start : chunk.char_end] != chunk.text:
                raise ValueError(f"chunk offsets do not match text in document {doc_id}")
        documents[doc_id] = document
        annotations = raw["annotation_sets"][0]["annotations"]
        for key in sorted(annotations):
            annotation = annotations[key]
            label = str(annotation["choice"])
            label_counts[label] += 1
            if label not in EVIDENCE_LABELS:
                no_evidence += 1
                continue
            gold = tuple(spans[i] for i in annotation["spans"])
            if not gold:
                empty += 1
                continue
            pairs.append(
                EvalPair(
                    split=split,
                    doc_id=doc_id,
                    hypothesis_key=key,
                    hypothesis=str(labels[key]["hypothesis"]),
                    label=label,
                    gold=gold,
                )
            )
    return SplitData(split, documents, pairs, dict(label_counts), no_evidence, empty)
