"""Isolated extraction entry point, run by the worker as a child process.

``python -m compliance_ingestion.extract_cli --input F --format pdf --tokenizer T --options O --output R``

Untrusted documents are parsed here rather than in the long-lived worker process, so
that a parser crash, runaway memory use or a hang (PyMuPDF/Tesseract are native code)
only kills this child. The child applies an address-space limit to itself (inherited
by Tesseract), and the parent enforces a wall-clock timeout. Because PyMuPDF reserves
several GB of *virtual* address space at import (with ~150 MB resident), the limit is
set after imports and the tokenizer are loaded, as ``current VmSize + --memory-mb``. It
therefore bounds what the parsing of this document may additionally allocate.

Exit codes: 0 = result JSON written; 2 = document rejected (``ExtractionError``; the
message is safe to show users and is written to the output as ``{"error": ...}``);
anything else = unexpected failure (no document content is printed).
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
from pathlib import Path

from compliance_ingestion.chunking import ChunkOptions
from compliance_ingestion.pdf import PdfOptions
from compliance_ingestion.pipeline import PipelineOptions, extract_document
from compliance_ingestion.tokens import HFTokenCounter
from compliance_ingestion.types import ExtractionError

EXIT_REJECTED = 2


def options_from_json(raw: dict[str, int | str]) -> PipelineOptions:
    return PipelineOptions(
        pdf=PdfOptions(
            max_pages=int(raw["max_pdf_pages"]),
            ocr_language=str(raw["ocr_language"]),
            ocr_dpi=int(raw["ocr_dpi"]),
            ocr_min_text_chars=int(raw["ocr_min_text_chars"]),
            ocr_timeout=int(raw["ocr_timeout_seconds"]),
        ),
        chunks=ChunkOptions(
            min_tokens=int(raw["chunk_min_tokens"]),
            max_tokens=int(raw["chunk_max_tokens"]),
            overlap_tokens=int(raw["chunk_overlap_tokens"]),
        ),
        max_sheet_rows=int(raw["max_sheet_rows"]),
    )


def _current_vm_bytes() -> int:
    with open("/proc/self/status", encoding="ascii") as status:
        for line in status:
            if line.startswith("VmSize:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("cannot read VmSize")  # pragma: no cover - Linux only


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--format", choices=["pdf", "docx", "xlsx", "csv"], required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--options", required=True, help="JSON object of pipeline limits")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--memory-mb", type=int, default=0)
    args = parser.parse_args(argv)

    options = options_from_json(json.loads(args.options))
    counter = HFTokenCounter(args.tokenizer)
    counter.count_many(["warm up"])  # initialise native thread pools before limiting
    if args.memory_mb > 0:
        limit = _current_vm_bytes() + args.memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    try:
        result = extract_document(args.input, args.format, options, counter)
    except ExtractionError as exc:
        args.output.write_text(json.dumps({"error": str(exc)}), encoding="utf-8")
        return EXIT_REJECTED
    except MemoryError:
        args.output.write_text(
            json.dumps({"error": "The document needs more memory than the parsing limit allows."}),
            encoding="utf-8",
        )
        return EXIT_REJECTED
    args.output.write_text(result.model_dump_json(), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
