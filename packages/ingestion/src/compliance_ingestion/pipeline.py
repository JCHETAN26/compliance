"""Parse a file, assign sections, chunk it and return one validated ``ExtractionResult``."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from compliance_ingestion.chunking import ChunkOptions, build_chunks
from compliance_ingestion.docx_parser import extract_docx
from compliance_ingestion.layout import assign_sections
from compliance_ingestion.pdf import PdfOptions, extract_pdf
from compliance_ingestion.spreadsheet import SpreadsheetExtraction, extract_csv, extract_xlsx
from compliance_ingestion.tokens import TokenCounter
from compliance_ingestion.types import (
    ChunkInfo,
    DocType,
    ExtractionError,
    ExtractionResult,
    ExtractionStats,
)

Format = Literal["pdf", "docx", "xlsx", "csv"]

_CONTRACT_TERMS = (
    "agreement",
    "non-disclosure",
    "nondisclosure",
    "confidentiality",
    "contract",
    "the parties",
    "hereinafter",
    "in witness whereof",
    "whereas",
    "terms and conditions",
)


@dataclass(frozen=True)
class PipelineOptions:
    pdf: PdfOptions
    chunks: ChunkOptions
    max_sheet_rows: int


def infer_doc_type(fmt: Format, text: str) -> DocType:
    """Keyword heuristic over the opening text (documented as such; not a classifier)."""
    if fmt in ("xlsx", "csv"):
        return "spreadsheet"
    head = text[:4000].lower()
    if sum(1 for term in _CONTRACT_TERMS if term in head) >= 2:
        return "contract"
    if re.search(r"\bpolicy\b", head[:400]) or head.count("policy") >= 3:
        return "policy"
    return "other"


def _spreadsheet_result(
    fmt: Format, extraction: SpreadsheetExtraction, counter: TokenCounter
) -> ExtractionResult:
    text = extraction.builder.text()
    names = {s.sheet_index: s.sheet_name for s in extraction.sheets}
    counts = counter.count_many([text[r.char_start : r.char_end] for r in extraction.rows])
    chunks = [
        ChunkInfo(
            chunk_index=i,
            char_start=row.char_start,
            char_end=row.char_end,
            text=text[row.char_start : row.char_end],
            token_count=n,
            section_path=f"{names[row.sheet_index]} > Row {row.row_number}",
            sheet_index=row.sheet_index,
            row_number=row.row_number,
        )
        for i, (row, n) in enumerate(zip(extraction.rows, counts, strict=True))
    ]
    return ExtractionResult(
        format=fmt,
        doc_type="spreadsheet",
        text=text,
        sheets=extraction.sheets,
        rows=extraction.rows,
        chunks=chunks,
        warnings=extraction.warnings,
        stats=ExtractionStats(rows=len(extraction.rows), chunks=len(chunks), chars=len(text)),
    )


def extract_document(
    path: Path, fmt: Format, options: PipelineOptions, counter: TokenCounter
) -> ExtractionResult:
    if fmt == "xlsx":
        return _spreadsheet_result(fmt, extract_xlsx(path, options.max_sheet_rows), counter)
    if fmt == "csv":
        return _spreadsheet_result(fmt, extract_csv(path, options.max_sheet_rows), counter)
    if fmt == "pdf":
        pdf = extract_pdf(path, options.pdf)
        builder, raw_blocks, pages, warnings = pdf.builder, pdf.blocks, pdf.pages, pdf.warnings
        char_boxes = builder.boxes
    elif fmt == "docx":
        docx = extract_docx(path)
        builder, raw_blocks, pages, warnings = docx.builder, docx.blocks, [], docx.warnings
        char_boxes = None
    else:  # pragma: no cover - Format is exhaustive
        raise ExtractionError(f"Unsupported format {fmt!r}")
    text = builder.text()
    if not text.strip():
        raise ExtractionError("No text could be extracted from this document.")
    blocks = assign_sections(raw_blocks, text)
    chunks = build_chunks(
        text, blocks, counter, options.chunks, char_boxes=char_boxes, pages=pages or None
    )
    return ExtractionResult(
        format=fmt,
        doc_type=infer_doc_type(fmt, text),
        text=text,
        pages=pages,
        blocks=blocks,
        chunks=chunks,
        warnings=warnings,
        stats=ExtractionStats(
            pages=len(pages),
            ocr_pages=sum(p.is_ocr for p in pages),
            blocks=len(blocks),
            chunks=len(chunks),
            chars=len(text),
        ),
    )
