"""DOCX extraction: body paragraphs and tables in document order, with styles and offsets.

Headers, footers, footnotes, comments and text boxes are not extracted (documented
limitation). Word has no fixed pagination, so DOCX documents have no page records and
chunk citations use character offsets only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import docx
from docx.document import Document as DocxDocument
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

from compliance_ingestion.layout import RawBlock, is_list_item, looks_like_heading
from compliance_ingestion.textbuilder import BLOCK_SEPARATOR, TextBuilder
from compliance_ingestion.types import BlockKind, ExtractionError

_HEADING_STYLE = re.compile(r"^heading\s*(\d)$", re.IGNORECASE)


@dataclass
class DocxExtraction:
    builder: TextBuilder
    blocks: list[RawBlock]
    warnings: list[str]


def _clean(text: str) -> str:
    text = text.replace(" ", " ").replace("\t", " ").replace("\r", "")  # noqa: RUF001 - intentional Unicode
    text = "".join(c for c in text if c == "\n" or c >= " ")
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line)


def _paragraph_kind(paragraph: Paragraph, text: str) -> tuple[BlockKind, str | None, int | None]:
    style = paragraph.style.name if paragraph.style is not None else None
    if style:
        lowered = style.lower()
        if lowered == "title":
            return "heading", style, 0
        match = _HEADING_STYLE.match(style)
        if match:
            return "heading", style, int(match.group(1))
        if "list" in lowered:
            return "list_item", style, None
    ppr = paragraph._p.pPr
    if ppr is not None and ppr.numPr is not None:
        return "list_item", style, None
    if is_list_item(text):
        return "list_item", style, None
    emphasised = bool(paragraph.runs) and all(
        bool(run.bold) for run in paragraph.runs if run.text.strip()
    )
    if looks_like_heading(text, emphasised=emphasised):
        return "heading", style, 2
    return "paragraph", style, None


def _iter_body(document: DocxDocument) -> list[Paragraph | Table]:
    items: list[Paragraph | Table] = []
    for child in document.element.body.iterchildren():
        if child.tag == qn("w:p"):
            items.append(Paragraph(child, document))
        elif child.tag == qn("w:tbl"):
            items.append(Table(child, document))
    return items


def extract_docx(path: Path) -> DocxExtraction:
    try:
        document = docx.Document(str(path))
    except Exception as exc:  # python-docx/lxml raise many types for corrupt packages
        raise ExtractionError("The DOCX file is damaged or not a valid Word document.") from exc
    builder = TextBuilder()
    blocks: list[RawBlock] = []
    for item in _iter_body(document):
        if isinstance(item, Paragraph):
            text = _clean(item.text)
            if not text:
                continue
            kind, style, level = _paragraph_kind(item, text)
            builder.separate(BLOCK_SEPARATOR)
            start = builder.length
            builder.append(text)
            blocks.append(
                RawBlock(
                    char_start=start,
                    char_end=builder.length,
                    kind=kind,
                    style=style,
                    heading_level=level,
                )
            )
            continue
        for row in item.rows:
            cells: list[str] = []
            seen: set[int] = set()
            for cell in row.cells:
                # Merged cells are repeated by python-docx; keep each underlying cell once.
                if id(cell._tc) in seen:
                    continue
                seen.add(id(cell._tc))
                cell_text = " ".join(_clean(cell.text).split("\n"))
                if cell_text:
                    cells.append(cell_text)
            if not cells:
                continue
            builder.separate(BLOCK_SEPARATOR)
            start = builder.length
            builder.append(" | ".join(cells))
            blocks.append(
                RawBlock(char_start=start, char_end=builder.length, kind="table_row", style="Table")
            )
    warnings = [] if blocks else ["The document body contains no text."]
    return DocxExtraction(builder=builder, blocks=blocks, warnings=warnings)
