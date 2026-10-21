"""PDF extraction with PyMuPDF: per-character boxes, layout blocks and OCR fallback.

For every page with a usable text layer the characters are read from ``rawdict`` so
each character of the canonical text keeps its exact box. Pages with almost no text
(< ``ocr_min_text_chars`` non-whitespace characters) are rasterised and OCR'd. OCR word
boxes are mapped back to page points and spread evenly over the word's characters.
"""

from __future__ import annotations

import statistics
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from compliance_ingestion.layout import RawBlock, is_list_item, looks_like_heading
from compliance_ingestion.ocr import OcrPageTooLargeError, ocr_page
from compliance_ingestion.textbuilder import (
    BLOCK_SEPARATOR,
    LINE_SEPARATOR,
    Box,
    CharBox,
    TextBuilder,
)
from compliance_ingestion.types import BBox, BlockKind, ExtractionError, PageInfo

# Expand ligatures (no TEXT_PRESERVE_LIGATURES) so "ﬁ" becomes "fi" for search; keep
# whitespace; clip to the media box so off-page text is ignored.
_TEXT_FLAGS = pymupdf.TEXT_PRESERVE_WHITESPACE | pymupdf.TEXT_MEDIABOX_CLIP
_BOLD_FLAG = 16
_DROP = {"­", "​", "‌", "‍", "﻿"}


@dataclass(frozen=True)
class PdfOptions:
    max_pages: int
    ocr_language: str
    ocr_dpi: int
    ocr_min_text_chars: int
    ocr_timeout: int


@dataclass
class PdfExtraction:
    builder: TextBuilder
    pages: list[PageInfo]
    blocks: list[RawBlock]
    warnings: list[str]


def _clean_char(char: str) -> str | None:
    if char in _DROP:
        return None
    if char in (" ", "\t"):  # noqa: RUF001 - intentional Unicode
        return " "
    if unicodedata.category(char) in ("Cc", "Cs", "Co") and char not in "\n":
        return None
    return char


def _box(
    page_no: int, bbox: tuple[float, float, float, float], matrix: pymupdf.Matrix | None
) -> Box:
    rect = pymupdf.Rect(bbox)
    if matrix is not None:
        rect = rect * matrix
    return (page_no, rect.x0, rect.y0, rect.x1, rect.y1)


def _body_font_size(doc: pymupdf.Document, max_pages: int) -> float:
    sizes: list[float] = []
    for page in doc.pages(0, min(doc.page_count, max_pages)):
        for block in page.get_text("dict", flags=_TEXT_FLAGS)["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    sizes.extend([round(span["size"], 1)] * len(span["text"].strip()))
    return statistics.median(sizes) if sizes else 10.0


def _page_text_layer(
    page: pymupdf.Page, page_no: int, builder: TextBuilder, body_size: float
) -> list[RawBlock]:
    matrix = page.rotation_matrix if page.rotation else None
    raw = page.get_text("rawdict", flags=_TEXT_FLAGS, sort=True)
    blocks: list[RawBlock] = []
    for block in raw["blocks"]:
        if block.get("type") != 0:
            continue
        lines: list[list[tuple[str, CharBox, float, bool]]] = []
        for line in block["lines"]:
            chars: list[tuple[str, CharBox, float, bool]] = []
            for span in line["spans"]:
                bold = bool(span["flags"] & _BOLD_FLAG) or "bold" in span["font"].lower()
                for ch in span["chars"]:
                    cleaned = _clean_char(ch["c"])
                    if cleaned is None:
                        continue
                    chars.append((cleaned, _box(page_no, ch["bbox"], matrix), span["size"], bold))
            # Trim surrounding whitespace on each line; keep inner spacing exactly.
            while chars and chars[0][0].isspace():
                chars.pop(0)
            while chars and chars[-1][0].isspace():
                chars.pop()
            if chars:
                lines.append(chars)
        if not lines:
            continue
        builder.separate(BLOCK_SEPARATOR)
        start = builder.length
        sizes: list[float] = []
        bold_chars = 0
        for i, line_chars in enumerate(lines):
            if i:
                builder.append(LINE_SEPARATOR)
            for char, box, size, bold in line_chars:
                builder.append_char(char, None if char.isspace() else box)
                if not char.isspace():
                    sizes.append(size)
                    bold_chars += bold
        end = builder.length
        text = builder.text()[start:end] if end - start < 400 else ""
        emphasised = bool(sizes) and (
            statistics.mean(sizes) >= body_size * 1.15 or bold_chars >= 0.8 * len(sizes)
        )
        kind = _classify(text, emphasised) if text else "paragraph"
        heading_level = None
        if kind == "heading" and sizes:
            heading_level = 1 if statistics.mean(sizes) >= body_size * 1.4 else 2
        bbox = _box(page_no, tuple(block["bbox"]), matrix)
        blocks.append(
            RawBlock(
                char_start=start,
                char_end=end,
                kind=kind,
                page_number=page_no,
                bbox=BBox(page=page_no, x0=bbox[1], y0=bbox[2], x1=bbox[3], y1=bbox[4]),
                heading_level=heading_level,
            )
        )
    return blocks


def _classify(text: str, emphasised: bool) -> BlockKind:
    if is_list_item(text):
        return "list_item"
    if looks_like_heading(text, emphasised=emphasised):
        return "heading"
    return "paragraph"


def _page_ocr(
    page: pymupdf.Page, page_no: int, builder: TextBuilder, options: PdfOptions
) -> tuple[list[RawBlock], float | None]:
    result = ocr_page(
        page, dpi=options.ocr_dpi, language=options.ocr_language, timeout=options.ocr_timeout
    )
    blocks: list[RawBlock] = []
    for paragraph in result.paragraphs:
        builder.separate(BLOCK_SEPARATOR)
        start = builder.length
        for li, line in enumerate(paragraph):
            if li:
                builder.append(LINE_SEPARATOR)
            for wi, word in enumerate(line):
                if wi:
                    builder.append(" ")
                width = (word.x1 - word.x0) / len(word.text)
                for ci, char in enumerate(word.text):
                    builder.append_char(
                        char,
                        (
                            page_no,
                            word.x0 + ci * width,
                            word.y0,
                            word.x0 + (ci + 1) * width,
                            word.y1,
                        ),
                    )
        end = builder.length
        xs0 = min(w.x0 for line in paragraph for w in line)
        ys0 = min(w.y0 for line in paragraph for w in line)
        xs1 = max(w.x1 for line in paragraph for w in line)
        ys1 = max(w.y1 for line in paragraph for w in line)
        text = builder.text()[start:end] if end - start < 400 else ""
        blocks.append(
            RawBlock(
                char_start=start,
                char_end=end,
                kind=_classify(text, emphasised=False) if text else "paragraph",
                page_number=page_no,
                bbox=BBox(page=page_no, x0=xs0, y0=ys0, x1=xs1, y1=ys1),
            )
        )
    return blocks, result.mean_confidence


def extract_pdf(path: Path, options: PdfOptions) -> PdfExtraction:
    try:
        doc = pymupdf.open(path, filetype="pdf")
    except (pymupdf.FileDataError, RuntimeError) as exc:
        raise ExtractionError("The PDF is damaged or not a valid PDF file.") from exc
    with doc:
        if doc.needs_pass or doc.is_encrypted:
            raise ExtractionError("Password-protected PDFs are not supported.")
        if doc.page_count == 0:
            raise ExtractionError("The PDF has no pages.")
        if doc.page_count > options.max_pages:
            raise ExtractionError(
                f"The PDF has {doc.page_count} pages; the limit is {options.max_pages}."
            )
        body_size = _body_font_size(doc, options.max_pages)
        builder = TextBuilder()
        pages: list[PageInfo] = []
        blocks: list[RawBlock] = []
        warnings: list[str] = []
        for page_index in range(doc.page_count):
            page = doc.load_page(page_index)
            page_no = page_index + 1
            text_chars = sum(1 for c in page.get_text("text", flags=_TEXT_FLAGS) if not c.isspace())
            builder.separate(BLOCK_SEPARATOR)
            page_start = builder.length
            ocr_confidence: float | None = None
            is_ocr = text_chars < options.ocr_min_text_chars
            if is_ocr:
                try:
                    page_blocks, ocr_confidence = _page_ocr(page, page_no, builder, options)
                except OcrPageTooLargeError as exc:
                    # One unrenderable page should not discard the rest of the document;
                    # the page stays empty and the reason is surfaced to the user.
                    page_blocks = []
                    warnings.append(f"Page {page_no} was not OCR'd: {exc}")
                else:
                    if not page_blocks:
                        warnings.append(f"Page {page_no} has no text layer and OCR found no text.")
            else:
                page_blocks = _page_text_layer(page, page_no, builder, body_size)
            blocks.extend(page_blocks)
            page_end = builder.length
            pages.append(
                PageInfo(
                    page_number=page_no,
                    width=round(page.rect.width, 2),
                    height=round(page.rect.height, 2),
                    is_ocr=is_ocr,
                    ocr_confidence=ocr_confidence,
                    char_start=page_start,
                    char_end=page_end,
                )
            )
        return PdfExtraction(builder=builder, pages=pages, blocks=blocks, warnings=warnings)
