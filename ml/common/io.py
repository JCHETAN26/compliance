"""Small, atomic file helpers for ML artifacts (JSON, gzipped JSON lines)."""

from __future__ import annotations

import gzip
import json
import os
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from ml.common.provenance import REPO

CACHE = REPO / ".cache" / "ml"
RESULTS = REPO / "ml" / "eval" / "results"


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        # mkstemp creates 0600; artifacts are read by the service container's own uid.
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def write_json(path: Path, value: Any) -> None:
    atomic_write_bytes(path, (json.dumps(value, indent=2, sort_keys=False) + "\n").encode())


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl_gz(path: Path, rows: Iterable[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    try:
        # Deterministic header: mtime=0 and an empty FNAME (otherwise GzipFile embeds the
        # random temporary file name), so equal rows give byte-identical files.
        with (
            open(tmp, "wb") as raw,
            gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as gz,
        ):
            for row in rows:
                gz.write((json.dumps(row, separators=(",", ":")) + "\n").encode())
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def read_jsonl_gz(path: Path) -> Iterator[Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)
