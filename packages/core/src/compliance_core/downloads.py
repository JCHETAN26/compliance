"""Size-capped, checksum-verified downloads written atomically into a cache directory."""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)
_CHUNK = 1024 * 1024


class DownloadError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(_CHUNK):
            digest.update(block)
    return digest.hexdigest()


def download_verified(
    url: str, dest: Path, *, sha256: str, max_bytes: int, timeout: float = 60.0
) -> Path:
    """Download ``url`` to ``dest`` unless an identical file is already there.

    The payload is streamed to a temporary file in the destination directory, rejected
    if it exceeds ``max_bytes`` or its SHA-256 differs from ``sha256``, then renamed into
    place, so ``dest`` never holds a partial or unverified file.
    """
    if not url.startswith("https://"):
        raise DownloadError(f"refusing non-HTTPS download URL {url!r}")
    if dest.exists():
        if sha256_file(dest) == sha256:
            return dest
        raise DownloadError(f"{dest} exists but its SHA-256 does not match the pinned value")
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".download-", dir=dest.parent)
    tmp = Path(tmp_name)
    digest = hashlib.sha256()
    received = 0
    try:
        with os.fdopen(fd, "wb") as out:
            request = urllib.request.Request(url, headers={"User-Agent": "compliance-workspace"})  # noqa: S310 - https enforced above
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
                while block := response.read(_CHUNK):
                    received += len(block)
                    if received > max_bytes:
                        raise DownloadError(f"{url} exceeds the {max_bytes}-byte limit")
                    digest.update(block)
                    out.write(block)
        actual = digest.hexdigest()
        if actual != sha256:
            raise DownloadError(f"SHA-256 mismatch for {url}: expected {sha256}, got {actual}")
        os.replace(tmp, dest)
        logger.info("downloaded artifact", extra={"url": url, "bytes": received})
        return dest
    finally:
        tmp.unlink(missing_ok=True)
