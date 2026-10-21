"""Upload security primitives: filenames, content sniffing, zip bombs, streaming
multipart limits (incl. incremental oversized headers) and storage key safety."""

from __future__ import annotations

import asyncio
import hashlib
import io
import uuid
import zipfile
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from compliance_core.files import UnsupportedFileError, ZipLimits, detect_format, sanitize_filename
from compliance_core.storage import LocalStorage, StorageError, document_key
from compliance_core.uploads import (
    MAX_HEADER_LINE_BYTES,
    MalformedUploadError,
    TooManyFilesError,
    UploadLimits,
    UploadTooLargeError,
    stage_multipart,
)

LIMITS = ZipLimits(max_entries=50, max_uncompressed_bytes=5_000_000, max_ratio=100)
BOUNDARY = "testboundary123"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("contract.pdf", "contract.pdf"),
        ("../../etc/passwd.pdf", "passwd.pdf"),
        ("C:\\Users\\x\\evil.docx", "evil.docx"),
        ("na\x00me\x07.csv", "na_me_.csv"),
        ('a<b>:"c|d?*.xlsx', "a_b___c_d__.xlsx"),
        ("   ", "document"),
        ("..", "document"),
        ("\uff46\uff55\uff4c\uff4c\uff57\uff49\uff44\uff54\uff48.pdf", "fullwidth.pdf"),  # NFKC
    ],
)
def test_sanitize_filename(raw: str, expected: str) -> None:
    assert sanitize_filename(raw) == expected


def test_long_filename_keeps_extension() -> None:
    name = sanitize_filename("x" * 500 + ".pdf")
    assert len(name) <= 200
    assert name.endswith(".pdf")


def _zip(path: Path, entries: dict[str, bytes], *, compress: bool = True) -> Path:
    mode = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    with zipfile.ZipFile(path, "w", mode) as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return path


def _docx_entries(extra: dict[str, bytes] | None = None) -> dict[str, bytes]:
    return {
        "[Content_Types].xml": b"<Types/>",
        "word/document.xml": b"<w:document/>",
        **(extra or {}),
    }


def test_detects_valid_formats(tmp_path: Path) -> None:
    (tmp_path / "a.pdf").write_bytes(b"%PDF-1.7\n%...")
    assert detect_format(tmp_path / "a.pdf", "a.pdf", LIMITS) == "pdf"
    _zip(tmp_path / "a.docx", _docx_entries())
    assert detect_format(tmp_path / "a.docx", "a.docx", LIMITS) == "docx"
    _zip(tmp_path / "a.xlsx", {"[Content_Types].xml": b"x", "xl/workbook.xml": b"x"})
    assert detect_format(tmp_path / "a.xlsx", "A.XLSX", LIMITS) == "xlsx"
    (tmp_path / "a.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    assert detect_format(tmp_path / "a.csv", "a.csv", LIMITS) == "csv"


@pytest.mark.parametrize(
    ("name", "content", "message"),
    [
        ("a.exe", b"MZ", "Unsupported file type"),
        ("a.doc", b"\xd0\xcf\x11\xe0", "Unsupported file type"),
        ("a.pdf", b"PK\x03\x04zip", "not a PDF"),
        ("a.csv", b"%PDF-1.4 disguised", "not plain-text CSV"),
        ("a.csv", b"\x00\x01\x02binary", "not plain-text CSV"),
        ("a.docx", b"%PDF-1.4", "not a .docx"),
        ("empty.pdf", b"", "empty"),
    ],
)
def test_rejects_spoofed_and_unsupported(
    tmp_path: Path, name: str, content: bytes, message: str
) -> None:
    (tmp_path / name).write_bytes(content)
    with pytest.raises(UnsupportedFileError, match=message):
        detect_format(tmp_path / name, name, LIMITS)


def test_xlsx_extension_on_docx_content_rejected(tmp_path: Path) -> None:
    _zip(tmp_path / "a.xlsx", _docx_entries())
    with pytest.raises(UnsupportedFileError, match=r"not a \.xlsx"):
        detect_format(tmp_path / "a.xlsx", "a.xlsx", LIMITS)


def test_zip_bomb_ratio_and_total_size(tmp_path: Path) -> None:
    _zip(tmp_path / "ratio.docx", _docx_entries({"word/bomb.xml": b"\0" * 2_000_000}))
    with pytest.raises(UnsupportedFileError, match="zip bomb"):
        detect_format(tmp_path / "ratio.docx", "ratio.docx", LIMITS)
    big = {f"word/p{i}.xml": bytes(range(256)) * 4000 for i in range(6)}  # ~6 MB stored
    _zip(tmp_path / "total.docx", _docx_entries(big), compress=False)
    with pytest.raises(UnsupportedFileError, match="expands beyond"):
        detect_format(tmp_path / "total.docx", "total.docx", LIMITS)


def test_zip_entry_count_traversal_macro_and_encryption(tmp_path: Path) -> None:
    many = {f"word/x{i}.xml": b"x" for i in range(60)}
    _zip(tmp_path / "many.docx", _docx_entries(many))
    with pytest.raises(UnsupportedFileError, match="too many"):
        detect_format(tmp_path / "many.docx", "many.docx", LIMITS)
    _zip(tmp_path / "trav.docx", _docx_entries({"../evil.xml": b"x"}))
    with pytest.raises(UnsupportedFileError, match="unsafe internal paths"):
        detect_format(tmp_path / "trav.docx", "trav.docx", LIMITS)
    _zip(tmp_path / "macro.docx", _docx_entries({"word/vbaProject.bin": b"x"}))
    with pytest.raises(UnsupportedFileError, match="Macro"):
        detect_format(tmp_path / "macro.docx", "macro.docx", LIMITS)
    path = _zip(tmp_path / "enc.docx", _docx_entries())
    data = bytearray(path.read_bytes())
    # Set the "encrypted" general-purpose flag bit in every local and central header.
    for signature, offset in ((b"PK\x03\x04", 6), (b"PK\x01\x02", 8)):
        start = 0
        while (index := data.find(signature, start)) != -1:
            data[index + offset] |= 0x1
            start = index + 4
    path.write_bytes(bytes(data))
    with pytest.raises(UnsupportedFileError, match="Encrypted"):
        detect_format(path, "enc.docx", LIMITS)


# --------------------------------------------------------------------------- multipart


def _part(name: str, filename: str | None, data: bytes) -> bytes:
    disposition = f'form-data; name="{name}"' + (f'; filename="{filename}"' if filename else "")
    return (
        (
            f"--{BOUNDARY}\r\nContent-Disposition: {disposition}\r\n"
            "Content-Type: application/octet-stream\r\n\r\n"
        ).encode()
        + data
        + b"\r\n"
    )


def _body(parts: list[bytes]) -> bytes:
    return b"".join(parts) + f"--{BOUNDARY}--\r\n".encode()


async def _chunks(data: bytes, size: int = 1000) -> AsyncIterator[bytes]:
    for i in range(0, len(data), size):
        yield data[i : i + size]


def _stage(tmp_path: Path, body: bytes, limits: UploadLimits, chunk: int = 1000) -> list:  # type: ignore[type-arg]
    staging = tmp_path / "staging"
    staging.mkdir(exist_ok=True)
    counter = iter(range(10_000))

    def new_file() -> Path:
        path = staging / f"f{next(counter)}"
        path.touch()
        return path

    return asyncio.run(
        stage_multipart(
            _chunks(body, chunk), f"multipart/form-data; boundary={BOUNDARY}", limits, new_file
        )
    )


LIMITS_UP = UploadLimits(max_file_bytes=10_000, max_request_bytes=100_000, max_files=3)


def test_streams_files_with_hashes(tmp_path: Path) -> None:
    a, b = b"A" * 5000, b"%PDF-1.4 " + b"B" * 3000
    staged = _stage(
        tmp_path,
        _body(
            [
                _part("note", None, b"ignored"),
                _part("files", "../a.csv", a),
                _part("files", "b.pdf", b),
            ]
        ),
        LIMITS_UP,
        chunk=333,
    )
    assert [s.filename for s in staged] == ["a.csv", "b.pdf"]
    assert staged[0].path.read_bytes() == a
    assert staged[0].sha256 == hashlib.sha256(a).hexdigest()
    assert staged[1].size == len(b)


def test_oversized_file_is_rejected_and_deleted_while_streaming(tmp_path: Path) -> None:
    staged = _stage(
        tmp_path,
        _body([_part("files", "big.pdf", b"x" * 20_000), _part("files", "ok.csv", b"a,b")]),
        LIMITS_UP,
    )
    assert staged[0].path is None
    assert "exceeds" in (staged[0].error or "")
    assert staged[1].path is not None
    assert staged[1].error is None
    assert sorted(p.name for p in (tmp_path / "staging").iterdir()) == [staged[1].path.name]


def test_request_limit_aborts_and_cleans_up(tmp_path: Path) -> None:
    body = _body([_part("files", f"f{i}.csv", b"y" * 9000) for i in range(3)])
    with pytest.raises(UploadTooLargeError):
        _stage(tmp_path, body, UploadLimits(10_000, 20_000, 5))
    assert list((tmp_path / "staging").iterdir()) == []


def test_too_many_files(tmp_path: Path) -> None:
    body = _body([_part("files", f"f{i}.csv", b"a") for i in range(4)])
    with pytest.raises(TooManyFilesError):
        _stage(tmp_path, body, LIMITS_UP)
    assert list((tmp_path / "staging").iterdir()) == []


@pytest.mark.parametrize("chunk", [1, 7, 4096])
def test_oversized_header_rejected_incrementally(tmp_path: Path, chunk: int) -> None:
    huge = "x" * (MAX_HEADER_LINE_BYTES * 3)
    body = (
        f'--{BOUNDARY}\r\nContent-Disposition: form-data; name="files"; filename="a.csv"\r\n'
        f"X-Padding: {huge}\r\n\r\ndata\r\n--{BOUNDARY}--\r\n"
    ).encode()
    consumed = 0

    async def counting(data: bytes) -> AsyncIterator[bytes]:
        nonlocal consumed
        for i in range(0, len(data), chunk):
            consumed += chunk
            yield data[i : i + chunk]

    staging = tmp_path / "staging"
    staging.mkdir()
    with pytest.raises(MalformedUploadError, match="header"):
        asyncio.run(
            stage_multipart(
                counting(body),
                f"multipart/form-data; boundary={BOUNDARY}",
                LIMITS_UP,
                lambda: staging / "x",
            )
        )
    # Parsing stopped soon after the cap, not after buffering the whole header.
    assert consumed < MAX_HEADER_LINE_BYTES + 512 + chunk


def test_many_part_headers_rejected(tmp_path: Path) -> None:
    headers = "".join(f"X-H{i}: v\r\n" for i in range(20))
    body = (
        f'--{BOUNDARY}\r\nContent-Disposition: form-data; name="files"; filename="a.csv"\r\n'
        f"{headers}\r\ndata\r\n--{BOUNDARY}--\r\n"
    ).encode()
    with pytest.raises(MalformedUploadError):
        _stage(tmp_path, body, LIMITS_UP)


@pytest.mark.parametrize(
    "content_type", [None, "application/json", "multipart/form-data", "text/plain; boundary=x"]
)
def test_content_type_validation(tmp_path: Path, content_type: str | None) -> None:
    async def run() -> None:
        await stage_multipart(_chunks(b""), content_type, LIMITS_UP, lambda: tmp_path / "x")

    with pytest.raises(MalformedUploadError):
        asyncio.run(run())


def test_truncated_body_rejected_and_cleaned(tmp_path: Path) -> None:
    body = _body([_part("files", "a.csv", b"abc" * 100)])[:-30]
    with pytest.raises(MalformedUploadError):
        _stage(tmp_path, body, LIMITS_UP)
    assert list((tmp_path / "staging").iterdir()) == []


# --------------------------------------------------------------------------- storage


def test_storage_roundtrip_and_key_safety(tmp_path: Path) -> None:
    storage = LocalStorage(tmp_path / "root")
    staged = storage.new_staging_file()
    staged.write_bytes(b"hello")
    key = document_key(uuid.uuid4(), uuid.uuid4())
    storage.put(key, staged)
    assert not staged.exists()
    assert storage.size(key) == 5
    assert storage.exists(key)
    with storage.open(key) as handle:
        assert handle.read() == b"hello"
    storage.delete(key)
    assert not storage.exists(key)
    assert [p.name for p in (tmp_path / "root").iterdir()] == [".staging"], "empty dirs pruned"
    for bad in ("../x", "/etc/passwd", "a/../../b", "A/B", "a//b", ".staging/x", ""):
        with pytest.raises(StorageError):
            storage.exists(bad)


def test_storage_put_from_outside_staging_copies(tmp_path: Path) -> None:
    storage = LocalStorage(tmp_path / "root")
    source = tmp_path / "seed.pdf"
    source.write_bytes(io.BytesIO(b"%PDF").getvalue())
    storage.put("documents/abc/def/original", source)
    assert source.exists()
    assert storage.size("documents/abc/def/original") == 4


@pytest.mark.integration
async def test_rate_limiter_reservations_are_atomic(redis_client: Any) -> None:
    from compliance_core.services.errors import RateLimitedError
    from compliance_core.services.session_store import UploadRateLimiter

    await redis_client.flushdb()
    limiter = UploadRateLimiter(redis_client, files_per_minute=5, requests_per_minute=3)
    results = await asyncio.gather(
        *(limiter.reserve_request("u1") for _ in range(10)), return_exceptions=True
    )
    assert sum(r is None for r in results) == 3
    assert all(isinstance(r, RateLimitedError) for r in results if r is not None)
    await limiter.reserve_request("u2")  # other users are independent
    await limiter.reserve_files("u1", 5)
    with pytest.raises(RateLimitedError):
        await limiter.reserve_files("u1", 1)
