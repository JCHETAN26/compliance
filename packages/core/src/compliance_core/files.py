"""Upload filename sanitising and content-based file type detection.

The declared extension must agree with the sniffed content (a ``.pdf`` that is really
a zip, or a ``.csv`` that is really a PDF, is rejected). Zip containers (DOCX/XLSX) are
inspected without extracting them: entry count, total and per-entry uncompressed sizes,
compression ratios, encryption, path traversal names and macro parts are all checked,
so decompression bombs are refused before any parser opens them.
"""

from __future__ import annotations

import re
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Literal

FileFormat = Literal["pdf", "docx", "xlsx", "csv"]

MIME_TYPES: dict[FileFormat, str] = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "csv": "text/csv",
}
EXTENSIONS: dict[str, FileFormat] = {".pdf": "pdf", ".docx": "docx", ".xlsx": "xlsx", ".csv": "csv"}
MAX_FILENAME = 200
_UNSAFE = re.compile(r'[\x00-\x1f\x7f<>:"/\\|?*]')


class UnsupportedFileError(ValueError):
    """The upload is not an acceptable document; the message is safe for users."""


@dataclass(frozen=True)
class ZipLimits:
    max_entries: int
    max_uncompressed_bytes: int
    max_ratio: int


def sanitize_filename(raw: str | None) -> str:
    """Display-safe base name: no directories, control or reserved characters."""
    name = unicodedata.normalize("NFKC", raw or "")
    name = PureWindowsPath(PurePosixPath(name).name).name  # strip both separator styles
    name = _UNSAFE.sub("_", name)
    name = " ".join(name.split()).strip(" .")
    if not name:
        name = "document"
    if len(name) > MAX_FILENAME:
        stem, dot, ext = name.rpartition(".")
        if dot and 0 < len(ext) <= 10:
            name = stem[: MAX_FILENAME - len(ext) - 1].rstrip() + "." + ext
        else:
            name = name[:MAX_FILENAME]
    return name


def _inspect_zip(path: Path, limits: ZipLimits) -> set[str]:
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
    except (zipfile.BadZipFile, OSError) as exc:
        raise UnsupportedFileError(
            "The file is not a valid Office document (corrupt zip)."
        ) from exc
    if len(infos) > limits.max_entries:
        raise UnsupportedFileError("The document contains too many internal parts.")
    total = 0
    names: set[str] = set()
    for info in infos:
        if info.flag_bits & 0x1:
            raise UnsupportedFileError(
                "Encrypted or password-protected documents are not supported."
            )
        name = info.filename
        if name.startswith(("/", "\\")) or ".." in PurePosixPath(name.replace("\\", "/")).parts:
            raise UnsupportedFileError("The document contains unsafe internal paths.")
        total += info.file_size
        if total > limits.max_uncompressed_bytes:
            raise UnsupportedFileError(
                "The document expands beyond the allowed size (possible zip bomb)."
            )
        if info.compress_size and info.file_size / info.compress_size > limits.max_ratio:
            raise UnsupportedFileError(
                "The document has a suspicious compression ratio (possible zip bomb)."
            )
        names.add(name.lower())
    if any(n.endswith("vbaproject.bin") for n in names):
        raise UnsupportedFileError("Macro-enabled documents are not supported.")
    if "[content_types].xml" not in names:
        raise UnsupportedFileError("The file is not a valid Office Open XML document.")
    return names


def _looks_like_text(sample: bytes) -> bool:
    if not sample:
        return False
    if b"\x00" in sample:
        return False
    for encoding in ("utf-8", "cp1252"):
        try:
            text = sample.decode(encoding)
        except UnicodeDecodeError:
            continue
        printable = sum(1 for c in text if c.isprintable() or c in "\r\n\t")
        return printable / len(text) >= 0.97
    return False


def detect_format(path: Path, filename: str, limits: ZipLimits) -> FileFormat:
    """Return the verified format, or raise ``UnsupportedFileError``."""
    suffix = Path(filename).suffix.lower()
    declared = EXTENSIONS.get(suffix)
    if declared is None:
        raise UnsupportedFileError(
            f"Unsupported file type {suffix or '(none)'}. Upload PDF, DOCX, XLSX or CSV files."
        )
    if path.stat().st_size == 0:
        raise UnsupportedFileError("The file is empty.")
    with path.open("rb") as handle:
        head = handle.read(65536)
    is_pdf = b"%PDF-" in head[:1024]
    is_zip = head.startswith(b"PK\x03\x04")
    if declared == "pdf":
        if not is_pdf:
            raise UnsupportedFileError(
                "The file content is not a PDF (extension does not match content)."
            )
        return "pdf"
    if declared in ("docx", "xlsx"):
        if not is_zip:
            raise UnsupportedFileError(f"The file content is not a .{declared} document.")
        names = _inspect_zip(path, limits)
        required = "word/document.xml" if declared == "docx" else "xl/workbook.xml"
        if required not in names:
            raise UnsupportedFileError(f"The file content is not a .{declared} document.")
        return declared
    # CSV: plain text, and not some other known binary format in disguise.
    if is_pdf or is_zip or not _looks_like_text(head):
        raise UnsupportedFileError("The file content is not plain-text CSV.")
    return "csv"
