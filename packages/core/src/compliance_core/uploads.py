"""Streaming ``multipart/form-data`` parsing with limits enforced as bytes arrive.

The request body is never buffered whole. Each file part is hashed (SHA-256) and
written to a staging file chunk by chunk:

- a part larger than ``max_file_bytes`` stops being written, its staging file is
  deleted, and it is reported as rejected; the rest of the body is still read but counts
  towards the request limit;
- a body larger than ``max_request_bytes`` aborts parsing (``UploadTooLargeError``);
- more than ``max_files`` file parts aborts parsing (``TooManyFilesError``);
- non-file form fields are ignored but capped at 1 KiB each;
- part headers are capped while they stream in: ``MAX_HEADER_LINE_BYTES`` per header
  (name + value) and ``MAX_PART_HEADER_BYTES`` per part, ``MAX_PART_HEADERS`` headers per
  part. Exceeding any cap aborts with ``MalformedUploadError`` (HTTP 400) before the
  header bytes can accumulate.

On any error every staging file created so far is deleted. On success the caller owns
the returned staging files.
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

from python_multipart.multipart import MultipartParser, parse_options_header

from compliance_core.files import sanitize_filename

MAX_FIELD_BYTES = 1024
MAX_HEADER_LINE_BYTES = 4096
MAX_PART_HEADER_BYTES = 8192
MAX_PART_HEADERS = 8
_MB = 1024 * 1024


class UploadError(Exception):
    status_code = 400
    code = "bad_upload"


class UploadTooLargeError(UploadError):
    status_code = 413
    code = "payload_too_large"


class TooManyFilesError(UploadError):
    status_code = 413
    code = "too_many_files"


class MalformedUploadError(UploadError):
    status_code = 400
    code = "malformed_upload"


@dataclass(frozen=True)
class UploadLimits:
    max_file_bytes: int
    max_request_bytes: int
    max_files: int


@dataclass
class StagedFile:
    filename: str  # sanitised display name
    path: Path | None  # None when the part was rejected while streaming
    size: int = 0
    sha256: str = ""
    error: str | None = None


@dataclass
class _Part:
    headers: dict[bytes, bytes] = field(default_factory=dict)
    is_file: bool = False
    filename: str = "document"
    handle: BinaryIO | None = None
    path: Path | None = None
    digest: Any = field(default_factory=hashlib.sha256)
    size: int = 0
    rejected: str | None = None


def boundary_from_content_type(content_type: str | None) -> bytes:
    ctype, params = parse_options_header(content_type or "")
    if ctype != b"multipart/form-data":
        raise MalformedUploadError("Content-Type must be multipart/form-data.")
    boundary = params.get(b"boundary")
    if not boundary or len(boundary) > 200:
        raise MalformedUploadError("Missing or invalid multipart boundary.")
    return boundary


class _StreamingStager:
    def __init__(self, limits: UploadLimits, new_staging_file: Callable[[], Path]) -> None:
        self.limits = limits
        self.new_staging_file = new_staging_file
        self.results: list[StagedFile] = []
        self.created: list[Path] = []
        self.part: _Part | None = None
        self._field = b""
        self._value = b""
        self._part_header_bytes = 0

    # -- python-multipart callbacks ------------------------------------------------
    def on_part_begin(self) -> None:
        self.part = _Part()
        self._part_header_bytes = 0

    def _grow_header(self, n: int) -> None:
        self._part_header_bytes += n
        if len(self._field) + len(self._value) + n > MAX_HEADER_LINE_BYTES:
            raise MalformedUploadError("A multipart header line is too large.")
        if self._part_header_bytes > MAX_PART_HEADER_BYTES:
            raise MalformedUploadError("Multipart part headers are too large.")

    def on_header_field(self, data: bytes, start: int, end: int) -> None:
        self._grow_header(end - start)
        self._field += data[start:end]

    def on_header_value(self, data: bytes, start: int, end: int) -> None:
        self._grow_header(end - start)
        self._value += data[start:end]

    def on_header_end(self) -> None:
        part = self._require_part()
        if len(part.headers) >= MAX_PART_HEADERS:
            raise MalformedUploadError("Too many multipart part headers.")
        part.headers[self._field.strip().lower()] = self._value.strip()
        self._field = self._value = b""

    def on_headers_finished(self) -> None:
        part = self._require_part()
        disposition = part.headers.get(b"content-disposition")
        if disposition is None:
            raise MalformedUploadError("Multipart part without Content-Disposition.")
        _, params = parse_options_header(disposition)
        if b"filename" not in params:
            return
        if len(self.results) >= self.limits.max_files:
            raise TooManyFilesError(
                f"At most {self.limits.max_files} files can be uploaded at once."
            )
        part.is_file = True
        part.filename = sanitize_filename(params[b"filename"].decode("utf-8", "replace"))
        part.path = self.new_staging_file()
        self.created.append(part.path)
        part.handle = part.path.open("wb")

    def on_part_data(self, data: bytes, start: int, end: int) -> None:
        part = self._require_part()
        part.size += end - start
        if not part.is_file:
            if part.size > MAX_FIELD_BYTES:
                raise MalformedUploadError("Form field too large.")
            return
        if part.rejected:
            return
        if part.size > self.limits.max_file_bytes:
            part.rejected = f"The file exceeds the {self.limits.max_file_bytes // _MB} MB limit."
            self._discard(part)
            return
        chunk = data[start:end]
        assert part.handle is not None  # noqa: S101 - opened in on_headers_finished
        part.handle.write(chunk)
        part.digest.update(chunk)

    def on_part_end(self) -> None:
        part = self._require_part()
        self.part = None
        if not part.is_file:
            return
        if part.handle is not None:
            part.handle.close()
            part.handle = None
        self.results.append(
            StagedFile(
                filename=part.filename,
                path=None if part.rejected else part.path,
                size=part.size,
                sha256="" if part.rejected else part.digest.hexdigest(),
                error=part.rejected,
            )
        )

    # -- helpers ---------------------------------------------------------------------
    def _require_part(self) -> _Part:
        if self.part is None:
            raise MalformedUploadError("The multipart body could not be parsed.")
        return self.part

    @staticmethod
    def _discard(part: _Part) -> None:
        if part.handle is not None:
            part.handle.close()
            part.handle = None
        if part.path is not None:
            part.path.unlink(missing_ok=True)
            part.path = None

    def cleanup(self) -> None:
        if self.part is not None and self.part.handle is not None:
            self.part.handle.close()
        for path in self.created:
            path.unlink(missing_ok=True)


async def stage_multipart(
    body: AsyncIterator[bytes],
    content_type: str | None,
    limits: UploadLimits,
    new_staging_file: Callable[[], Path],
) -> list[StagedFile]:
    """Parse ``body`` and return one ``StagedFile`` per file part, in order."""
    boundary = boundary_from_content_type(content_type)
    stager = _StreamingStager(limits, new_staging_file)
    callbacks = {
        name: getattr(stager, name)
        for name in (
            "on_part_begin",
            "on_header_field",
            "on_header_value",
            "on_header_end",
            "on_headers_finished",
            "on_part_data",
            "on_part_end",
        )
    }
    parser = MultipartParser(
        boundary,
        callbacks,  # type: ignore[arg-type]
        max_header_count=MAX_PART_HEADERS,
        max_header_size=MAX_HEADER_LINE_BYTES,
    )
    received = 0
    try:
        async for chunk in body:
            received += len(chunk)
            if received > limits.max_request_bytes:
                raise UploadTooLargeError(
                    f"The upload exceeds the {limits.max_request_bytes // _MB} MB request limit."
                )
            try:
                parser.write(chunk)
            except UploadError:
                raise
            except Exception as exc:  # python-multipart's own parse errors
                if "header" in str(exc).lower():
                    raise MalformedUploadError("A multipart header line is too large.") from exc
                raise MalformedUploadError("The multipart body could not be parsed.") from exc
        try:
            parser.finalize()
        except Exception as exc:
            raise MalformedUploadError("The multipart body is incomplete.") from exc
        if stager.part is not None:
            raise MalformedUploadError("The multipart body is incomplete.")
    except BaseException:
        stager.cleanup()
        raise
    return stager.results
