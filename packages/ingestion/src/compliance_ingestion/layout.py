"""Format-independent layout helpers: block classification, section paths, bbox merging."""

from __future__ import annotations

import re
from dataclasses import dataclass

from compliance_ingestion.textbuilder import CharBox
from compliance_ingestion.types import BBox, BlockInfo, BlockKind

_NUMBERED = re.compile(
    r"""^\s*(?:
        (?P<kw>article|section|clause)\s+(?P<kwnum>[0-9]+[a-z]?|[ivxlc]+)\b
        |(?P<num>\d{1,2}(?:\.\d{1,2}){0,3})[.)]?(?=\s)
    )""",
    re.IGNORECASE | re.VERBOSE,
)
_LIST = re.compile(r"^\s*(?:[•●◦▪\-–—*]\s|\(?[a-z]\)\s|\(?[ivx]{1,5}\)\s)", re.IGNORECASE)  # noqa: RUF001 - intentional Unicode
_TRAILING_PUNCT = (",", ";", ":")
MAX_LABEL = 80


@dataclass
class RawBlock:
    """A block before section assignment; offsets are document-level."""

    char_start: int
    char_end: int
    kind: BlockKind
    style: str | None = None
    page_number: int | None = None
    bbox: BBox | None = None
    heading_level: int | None = None  # for unnumbered headings (style/font based)


def numbering_level(text: str) -> int | None:
    """Depth of a leading clause number: ``3.`` -> 1, ``3.2`` -> 2, ``Article IV`` -> 1."""
    match = _NUMBERED.match(text)
    if not match:
        return None
    if match.group("kw"):
        return 1
    return match.group("num").count(".") + 1


def is_list_item(text: str) -> bool:
    return bool(_LIST.match(text))


def looks_like_heading(text: str, *, emphasised: bool, max_chars: int = 100) -> bool:
    """Short, unterminated lines that are visually emphasised or fully upper-case."""
    stripped = " ".join(text.split())
    if not stripped or len(stripped) > max_chars or len(text.strip().splitlines()) > 2:
        return False
    if stripped.endswith(_TRAILING_PUNCT) or (
        stripped.endswith(".") and numbering_level(stripped) is None
    ):
        return False
    letters = [c for c in stripped if c.isalpha()]
    if len(letters) < 3:
        return False
    upper = sum(c.isupper() for c in letters) / len(letters) >= 0.9
    return emphasised or upper


def _label(text: str) -> str:
    first = " ".join(text.split())
    match = _NUMBERED.match(first)
    if match:
        # "3. Confidential Information. The Recipient ..." -> "3. Confidential Information"
        rest = first[match.end() :].strip()
        title = re.split(r"(?<=[.:])\s", rest, maxsplit=1)[0].rstrip(".:")
        label = f"{first[: match.end()].strip()} {title}".strip()
        if len(title) > MAX_LABEL // 2:
            label = first[: match.end()].strip()
        return label[:MAX_LABEL]
    return first[:MAX_LABEL]


def assign_sections(raw: list[RawBlock], text: str) -> list[BlockInfo]:
    """Turn raw blocks into ``BlockInfo`` with heading levels and section paths."""
    stack: list[tuple[int, str]] = []
    blocks: list[BlockInfo] = []
    for index, block in enumerate(raw):
        block_text = text[block.char_start : block.char_end]
        level: int | None = None
        starts_section = False
        if block.kind == "heading":
            level = numbering_level(block_text) or block.heading_level or 1
            starts_section = True
        elif block.kind == "paragraph":
            level = numbering_level(block_text)
            starts_section = level is not None
        if starts_section and level is not None:
            if level > 0:
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, _label(block_text)))
            else:
                stack.clear()
        path = " > ".join(label for _, label in stack) or None
        blocks.append(
            BlockInfo(
                block_index=index,
                kind=block.kind,
                style=block.style,
                level=level,
                starts_section=starts_section,
                char_start=block.char_start,
                char_end=block.char_end,
                section_path=path,
                page_number=block.page_number,
                bbox=block.bbox,
            )
        )
    return blocks


def line_runs(boxes: list[CharBox], tolerance: float = 1.5) -> list[list[float]]:
    """Visual line runs over the whole text, with their character ranges:
    ``[page, char_start, char_end, x0, y0, x1, y1]`` (same grouping rule as
    :func:`merge_char_boxes`). Citations use them to highlight an exact range by line."""
    runs: list[list[float]] = []
    for index, box in enumerate(boxes):
        if box is None:
            continue
        page, x0, y0, x1, y1 = box
        if runs:
            cur = runs[-1]
            overlap = min(cur[6], y1) - max(cur[4], y0)
            min_height = max(min(cur[6] - cur[4], y1 - y0), 0.01)
            rightwards = x0 >= cur[3] - tolerance and x0 <= cur[5] + 40
            if cur[0] == page and overlap >= 0.5 * min_height and rightwards:
                cur[2] = index + 1
                cur[3], cur[4] = min(cur[3], x0), min(cur[4], y0)
                cur[5], cur[6] = max(cur[5], x1), max(cur[6], y1)
                continue
        runs.append([page, index, index + 1, x0, y0, x1, y1])
    return [[int(r[0]), int(r[1]), int(r[2]), *(round(v, 2) for v in r[3:])] for r in runs]


def merge_char_boxes(
    boxes: list[CharBox], start: int, end: int, tolerance: float = 1.5
) -> list[BBox]:
    """Merge character boxes in ``[start, end)`` into one rectangle per visual line run.

    Characters join the current rectangle while they are on the same page, vertically
    overlap it by at least half of the smaller height, and keep moving rightwards.
    """
    rects: list[list[float]] = []
    pages: list[int] = []
    for box in boxes[start:end]:
        if box is None:
            continue
        page, x0, y0, x1, y1 = box
        if rects:
            cur = rects[-1]
            same_page = pages[-1] == page
            overlap = min(cur[3], y1) - max(cur[1], y0)
            min_height = max(min(cur[3] - cur[1], y1 - y0), 0.01)
            rightwards = x0 >= cur[0] - tolerance and x0 <= cur[2] + 40
            if same_page and overlap >= 0.5 * min_height and rightwards:
                cur[0] = min(cur[0], x0)
                cur[1] = min(cur[1], y0)
                cur[2] = max(cur[2], x1)
                cur[3] = max(cur[3], y1)
                continue
        rects.append([x0, y0, x1, y1])
        pages.append(page)
    return [
        BBox(page=p, x0=round(r[0], 2), y0=round(r[1], 2), x1=round(r[2], 2), y1=round(r[3], 2))
        for p, r in zip(pages, rects, strict=True)
    ]
