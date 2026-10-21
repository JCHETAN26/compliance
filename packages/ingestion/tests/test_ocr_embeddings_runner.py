# mypy: disallow-untyped-calls=False
# (fixtures are built with PyMuPDF/python-docx, which ship partial annotations)
"""OCR raster safety, OCR error mapping, real embeddings with long-input windows,
model verification, and the sandboxed extraction runner."""

from __future__ import annotations

import asyncio
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pymupdf
import pytesseract
import pytest

from compliance_ingestion import ocr as ocr_module
from compliance_ingestion.embeddings import (
    EMBEDDING_DIM,
    Embedder,
    ModelSpec,
    ModelUnavailableError,
    ensure_model,
)
from compliance_ingestion.ocr import (
    MAX_OCR_PIXELS,
    MIN_OCR_DPI,
    OcrPageTooLargeError,
    choose_ocr_dpi,
    raster_size,
)
from compliance_ingestion.pdf import PdfOptions, extract_pdf
from compliance_ingestion.runner import RunnerLimits, run_extraction
from compliance_ingestion.types import ExtractionError

# --------------------------------------------------------------------------- OCR safety


@dataclass
class _Rect:
    width: float
    height: float
    x0: float = 0.0
    y0: float = 0.0


class _ExplodingPage:
    """Fake page whose rasteriser must never be reached."""

    number = 0

    def __init__(self, width: float, height: float) -> None:
        self.rect = _Rect(width, height)

    def get_pixmap(self, **_: Any) -> Any:
        raise AssertionError("get_pixmap must not be called for an oversized page")


def test_dpi_is_lowered_to_fit_the_pixel_budget() -> None:
    a4 = _Rect(595, 842)
    assert choose_ocr_dpi(a4, 300) == 300
    poster = _Rect(72 * 40, 72 * 30)  # 40 x 30 inch page
    dpi = choose_ocr_dpi(poster, 300)
    assert MIN_OCR_DPI <= dpi < 300
    w, h = raster_size(poster, dpi)
    assert w * h <= MAX_OCR_PIXELS
    w1, h1 = raster_size(poster, dpi + 1)
    assert w1 * h1 > MAX_OCR_PIXELS, "the chosen DPI is the largest that fits"


@pytest.mark.parametrize("size", [(14400, 14400), (72 * 200, 72 * 30), (1e7, 1e7)])
def test_oversized_page_is_refused_before_rasterising(size: tuple[float, float]) -> None:
    page = _ExplodingPage(*size)
    with pytest.raises(OcrPageTooLargeError, match="pixel OCR limit"):
        ocr_module.ocr_page(page, dpi=300, language="eng", timeout=5)  # type: ignore[arg-type]


def test_invalid_page_size_refused() -> None:
    with pytest.raises(OcrPageTooLargeError):
        choose_ocr_dpi(_Rect(0, 100), 300)


def test_oversized_real_page_becomes_a_warning_not_a_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    doc = pymupdf.open()
    doc.new_page(width=600, height=800).insert_text(
        (72, 72), "Normal page with enough text to skip OCR."
    )
    doc.new_page(width=14400, height=14400)  # 200 x 200 inch, no text -> OCR candidate
    doc.save(tmp_path / "huge.pdf")
    calls: list[int] = []
    original = pymupdf.Page.get_pixmap

    def guarded(self: pymupdf.Page, *args: Any, **kwargs: Any) -> Any:
        calls.append(self.number)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pymupdf.Page, "get_pixmap", guarded)
    result = extract_pdf(tmp_path / "huge.pdf", PdfOptions(10, "eng", 300, 25, 60))
    assert calls == [], "the oversized page was never rasterised"
    assert any("Page 2 was not OCR'd" in w for w in result.warnings)
    assert "Normal page" in result.builder.text()


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (pytesseract.TesseractError(1, "bad image"), "tesseract error"),
        (RuntimeError("Tesseract process timeout"), "timed out"),
        (RuntimeError("something else"), "OCR failed on page 1."),
        (pytesseract.TesseractNotFoundError(), "not installed"),
    ],
)
def test_ocr_errors_are_classified(
    monkeypatch: pytest.MonkeyPatch, error: Exception, message: str
) -> None:
    def boom(*_: Any, **__: Any) -> Any:
        raise error

    monkeypatch.setattr(pytesseract, "image_to_data", boom)
    doc = pymupdf.open()
    page = doc.new_page(width=200, height=200)
    with pytest.raises(ExtractionError, match=message):
        ocr_module.ocr_page(page, dpi=100, language="eng", timeout=5)
    if "timeout" not in str(error).lower():
        with pytest.raises(ExtractionError) as info:
            ocr_module.ocr_page(page, dpi=100, language="eng", timeout=5)
        assert "timed out" not in str(info.value)


# --------------------------------------------------------------------------- embeddings


def _norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def test_real_embeddings_are_384d_normalised_and_semantic(embedder: Embedder) -> None:
    out = embedder.embed(
        [
            "The receiving party shall keep the information confidential.",
            "Recipient must not disclose confidential information to third parties.",
            "The cafeteria serves lunch at noon.",
        ]
    )
    assert all(len(e.vector) == EMBEDDING_DIM for e in out)
    assert all(abs(_norm(e.vector) - 1.0) < 1e-4 for e in out)
    assert all(e.windows == 1 for e in out)
    sim = np.array(out[0].vector) @ np.array(out[1].vector)
    unrelated = np.array(out[0].vector) @ np.array(out[2].vector)
    assert sim > unrelated + 0.1


def test_short_text_matches_direct_model_encoding(embedder: Embedder) -> None:
    text = "Confidential information must be returned or destroyed on request."
    direct = embedder._encode([text])[0]
    assert np.allclose(embedder.embed([text])[0].vector, direct, atol=1e-6)


def test_long_text_uses_bounded_windows_without_truncation(model_path: Path) -> None:
    # A small context makes windowing observable with a moderately long sentence.
    small = Embedder(model_path, threads=2, max_seq_length=64, window_overlap=8)
    sentence = (
        "The Recipient shall hold all Confidential Information in strict confidence and shall "
        "not disclose, publish, copy or otherwise make available any part of it to any person "
        "other than its employees, officers, advisers and contractors who need to know it for the "
        "Purpose and who are bound by obligations of confidentiality no less protective than "
        "those set out in this Agreement, and shall return or destroy it on written request."
    )
    windows = small.windows(sentence)
    assert len(windows) > 1
    assert all(count <= small.window_tokens for _, count in windows)
    # Windows are verbatim substrings in order, cover the start and the end of the text.
    assert sentence.startswith(windows[0][0])
    assert sentence.endswith(windows[-1][0])
    cursor = 0
    for window_text, _ in windows:
        position = sentence.index(window_text, max(cursor - len(window_text), 0))
        assert position >= cursor - len(window_text)
        cursor = position + len(window_text)
    total = len(small._tokenizer.encode(sentence, add_special_tokens=False).ids)
    assert total > small.window_tokens
    result = small.embed([sentence])[0]
    assert result.windows == len(windows)
    assert result.tokens == total
    assert abs(_norm(result.vector) - 1.0) < 1e-4
    first_only = small._encode([windows[0][0]])[0]
    assert not np.allclose(result.vector, first_only, atol=1e-3), "the tail is not dropped"


def test_full_size_model_windows_a_chunk_over_512_tokens(embedder: Embedder) -> None:
    long_sentence = (
        "The Recipient shall protect the information " + "and its copies " * 220 + "carefully."
    )
    windows = embedder.windows(long_sentence)
    assert embedder.window_tokens == 510
    assert len(windows) >= 2
    assert all(count <= 510 for _, count in windows)
    assert embedder.embed([long_sentence])[0].windows == len(windows)


def test_model_verification_rejects_missing_or_tampered_files(
    tmp_path: Path, model_path: Path
) -> None:
    spec = ModelSpec("BAAI/bge-small-en-v1.5", "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a", "0" * 64)
    with pytest.raises(ModelUnavailableError, match="not in"):
        ensure_model(tmp_path / "empty", spec, allow_download=False)
    fake = tmp_path / "cache" / spec.dirname
    fake.mkdir(parents=True)
    for name in ("config.json", "tokenizer.json"):
        shutil.copy(model_path / name, fake / name)
    with pytest.raises(ModelUnavailableError, match="incomplete"):
        ensure_model(tmp_path / "cache", spec, allow_download=False)


# --------------------------------------------------------------------------- runner

OPTIONS: dict[str, int | str] = {
    "max_pdf_pages": 50,
    "ocr_language": "eng",
    "ocr_dpi": 300,
    "ocr_min_text_chars": 25,
    "ocr_timeout_seconds": 60,
    "chunk_min_tokens": 200,
    "chunk_max_tokens": 400,
    "chunk_overlap_tokens": 40,
    "max_sheet_rows": 1000,
}


def _pdf(path: Path, text: str) -> Path:
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), text)
    doc.save(path)
    return path


def test_runner_success_rejection_and_timeout(tmp_path: Path, model_path: Path) -> None:
    tokenizer = model_path / "tokenizer.json"
    ok = asyncio.run(
        run_extraction(
            _pdf(tmp_path / "ok.pdf", "A short agreement between the parties."),
            "pdf",
            tokenizer,
            OPTIONS,
            RunnerLimits(120, 1024),
        )
    )
    assert "agreement" in ok.text
    (tmp_path / "bad.pdf").write_bytes(b"%PDF-1.7 broken")
    with pytest.raises(ExtractionError):
        asyncio.run(
            run_extraction(tmp_path / "bad.pdf", "pdf", tokenizer, OPTIONS, RunnerLimits(120, 1024))
        )
    with pytest.raises(ExtractionError, match="longer than the 0s limit"):
        asyncio.run(
            run_extraction(tmp_path / "ok.pdf", "pdf", tokenizer, OPTIONS, RunnerLimits(0, 1024))
        )
