"""Tesseract OCR for image-only PDF pages, returning words with page-point boxes.

Memory safety: the raster size is computed from the page dimensions **before**
``get_pixmap`` is called. The requested DPI is lowered until the raster fits
``MAX_OCR_PIXELS``. If even ``MIN_OCR_DPI`` would exceed the cap (an absurdly large
page box), the page is refused with ``OcrPageTooLargeError`` and nothing is rasterised.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import pymupdf
import pytesseract
from PIL import Image

from compliance_ingestion.types import ExtractionError

MAX_OCR_PIXELS = 40_000_000
# Below ~100 DPI Tesseract accuracy on body text collapses; refuse rather than guess.
MIN_OCR_DPI = 100


class OcrPageTooLargeError(ExtractionError):
    """The page cannot be rasterised at a usable DPI within the pixel budget."""


@dataclass(frozen=True)
class OcrWord:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    confidence: float


@dataclass(frozen=True)
class OcrResult:
    # paragraphs -> lines -> words, in Tesseract reading order
    paragraphs: list[list[list[OcrWord]]]
    mean_confidence: float | None
    dpi: int


class _RectLike(Protocol):
    @property
    def width(self) -> float: ...

    @property
    def height(self) -> float: ...


def raster_size(rect: _RectLike, dpi: int) -> tuple[int, int]:
    """Pixel dimensions PyMuPDF produces for ``rect`` at ``dpi`` (rounded up)."""
    scale = dpi / 72
    return math.ceil(rect.width * scale), math.ceil(rect.height * scale)


def choose_ocr_dpi(rect: _RectLike, requested_dpi: int, max_pixels: int = MAX_OCR_PIXELS) -> int:
    """Largest DPI <= requested whose raster fits ``max_pixels``; raises if < MIN_OCR_DPI."""
    if rect.width <= 0 or rect.height <= 0:
        raise OcrPageTooLargeError("The page has an empty or invalid size and cannot be OCR'd.")
    area_sq_in = (rect.width / 72) * (rect.height / 72)
    dpi = min(requested_dpi, math.floor(math.sqrt(max_pixels / area_sq_in)))
    while dpi >= MIN_OCR_DPI:
        width, height = raster_size(rect, dpi)
        if width * height <= max_pixels:
            return dpi
        dpi -= 1
    raise OcrPageTooLargeError(
        f"The page is {rect.width:.0f}x{rect.height:.0f} pt; rasterising it at "
        f"{MIN_OCR_DPI} DPI would exceed the {max_pixels:,}-pixel OCR limit."
    )


def ocr_page(
    page: pymupdf.Page,
    *,
    dpi: int,
    language: str,
    timeout: int,
    max_pixels: int = MAX_OCR_PIXELS,
) -> OcrResult:
    rect = page.rect
    effective_dpi = choose_ocr_dpi(rect, dpi, max_pixels)
    pix = page.get_pixmap(dpi=effective_dpi, colorspace=pymupdf.csGRAY, alpha=False)
    if pix.width * pix.height > max_pixels:  # defensive: PyMuPDF rounding differs
        raise OcrPageTooLargeError("Rendered OCR raster exceeded the pixel limit.")
    image = Image.frombytes("L", (pix.width, pix.height), pix.samples)
    page_no = page.number + 1 if page.number is not None else 0
    try:
        data = pytesseract.image_to_data(
            image,
            lang=language,
            config="--psm 3",
            output_type=pytesseract.Output.DICT,
            timeout=timeout,
        )
    # Order matters: TesseractError subclasses RuntimeError, and pytesseract signals a
    # timeout with a plain RuntimeError, so the specific classes are handled first.
    except pytesseract.TesseractNotFoundError as exc:
        raise ExtractionError("OCR is unavailable: the tesseract binary is not installed.") from exc
    except pytesseract.TesseractError as exc:
        raise ExtractionError(f"OCR failed on page {page_no} (tesseract error).") from exc
    except RuntimeError as exc:
        if "timeout" in str(exc).lower():
            raise ExtractionError(f"OCR timed out on page {page_no} after {timeout}s.") from exc
        raise ExtractionError(f"OCR failed on page {page_no}.") from exc

    # Pixmaps are rendered in the page's displayed orientation, so pixels map linearly
    # onto page.rect (which already reflects /Rotate).
    sx = rect.width / pix.width
    sy = rect.height / pix.height
    grouped: dict[tuple[int, int], dict[int, list[OcrWord]]] = {}
    confidences: list[float] = []
    for i, raw_text in enumerate(data["text"]):
        text = (raw_text or "").strip()
        conf = float(data["conf"][i])
        if not text or conf < 0:
            continue
        left, top = float(data["left"][i]), float(data["top"][i])
        width, height = float(data["width"][i]), float(data["height"][i])
        word = OcrWord(
            text=text,
            x0=rect.x0 + left * sx,
            y0=rect.y0 + top * sy,
            x1=rect.x0 + (left + width) * sx,
            y1=rect.y0 + (top + height) * sy,
            confidence=conf,
        )
        paragraph = (int(data["block_num"][i]), int(data["par_num"][i]))
        grouped.setdefault(paragraph, {}).setdefault(int(data["line_num"][i]), []).append(word)
        confidences.append(conf)
    paragraphs = [[lines[line] for line in sorted(lines)] for _, lines in sorted(grouped.items())]
    mean = round(sum(confidences) / len(confidences), 2) if confidences else None
    return OcrResult(paragraphs=paragraphs, mean_confidence=mean, dpi=effective_dpi)
