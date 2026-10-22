"""Run provenance: what code, data, models and environment produced a result file.

The git commit is read from the repository (``git rev-parse HEAD``) together with the
dirty state; it is never invented. Because results are usually produced before the
code is committed, ``code_fingerprint`` hashes the actual content of every file that
can influence the numbers (tracked or not), so a later commit can be matched to it.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
FINGERPRINT_ROOTS = (
    "packages/core/src",
    "packages/ingestion/src",
    "packages/retrieval/src",
    "apps/model-service/src",
    "evals/retrieval",
)
FINGERPRINT_FILES = ("pyproject.toml", "uv.lock")


def _git(*args: str) -> str:
    return subprocess.run(  # noqa: S603 - fixed git arguments
        ["git", *args],  # noqa: S607
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def code_fingerprint() -> dict[str, Any]:
    files: list[Path] = [REPO / f for f in FINGERPRINT_FILES]
    for root in FINGERPRINT_ROOTS:
        files.extend(
            p
            for p in (REPO / root).rglob("*")
            if p.is_file()
            and p.suffix in {".py", ".yaml", ".json", ".toml", ".typed"}
            and "__pycache__" not in p.parts
            and "tests" not in p.relative_to(REPO / root).parts
            and p.name != "frozen_config.json"
        )
    digest = hashlib.sha256()
    for path in sorted(files):
        rel = path.relative_to(REPO).as_posix()
        digest.update(rel.encode() + b"\0" + hashlib.sha256(path.read_bytes()).digest())
    return {"sha256": digest.hexdigest(), "files": len(files), "roots": list(FINGERPRINT_ROOTS)}


def config_hash(config: Any) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def collect(command: list[str], seed: int, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    import numpy
    import sentence_transformers
    import torch

    status = _git("status", "--porcelain")
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "command": " ".join(command),
        "git_head": _git("rev-parse", "HEAD"),
        "git_dirty": bool(status),
        "git_dirty_paths": len(status.splitlines()),
        "code_fingerprint": code_fingerprint(),
        "seed": seed,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {
            "torch": torch.__version__,
            "sentence_transformers": sentence_transformers.__version__,
            "numpy": numpy.__version__,
        },
        **(extra or {}),
    }
