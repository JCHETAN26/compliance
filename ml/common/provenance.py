"""Provenance for every ML result file: code identity, config, data, environment.

The git commit and dirty state are read from the repository, never invented. Results
are normally produced before the code is committed, so ``code_fingerprint`` hashes the
content of every file that can influence ML numbers (tracked or not); a later commit
can be matched against it. The list of dirty paths is recorded as well.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
FINGERPRINT_ROOTS = (
    "ml",
    "apps/model-service/src/compliance_model_service/rule_check",
    "packages/ingestion/src/compliance_ingestion",
    "rule_packs",
)
FINGERPRINT_FILES = (
    "evals/retrieval/data.py",
    "pyproject.toml",
    "uv.lock",
)
_SUFFIXES = {".py", ".yaml", ".yml", ".toml", ".txt", ".json"}
_SKIP_PARTS = {"__pycache__", "tests", "results", "notebooks"}


def _git(*args: str) -> str:
    return subprocess.run(  # noqa: S603 - fixed git arguments
        ["git", *args],  # noqa: S607
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def git_state() -> dict[str, Any]:
    try:
        status = _git("status", "--porcelain")
        head = _git("rev-parse", "HEAD")
    except (OSError, subprocess.CalledProcessError) as exc:  # e.g. a source bundle
        return {"git_head": None, "git_error": str(exc)}
    paths = [line[3:] for line in status.splitlines()]
    return {
        "git_head": head,
        "git_dirty": bool(paths),
        "git_dirty_paths": paths[:200],
        "git_dirty_count": len(paths),
    }


def _fingerprint_files() -> list[Path]:
    files = [REPO / f for f in FINGERPRINT_FILES]
    for root in FINGERPRINT_ROOTS:
        base = REPO / root
        files.extend(
            p
            for p in base.rglob("*")
            if p.is_file()
            and p.suffix in _SUFFIXES
            and not _SKIP_PARTS.intersection(p.relative_to(base).parts)
        )
    return sorted(set(files))


def code_fingerprint() -> dict[str, Any]:
    digest = hashlib.sha256()
    files = _fingerprint_files()
    for path in files:
        rel = path.relative_to(REPO).as_posix()
        digest.update(rel.encode() + b"\0" + hashlib.sha256(path.read_bytes()).digest())
    return {"sha256": digest.hexdigest(), "files": len(files), "roots": list(FINGERPRINT_ROOTS)}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def config_hash(config: Any) -> str:
    return hashlib.sha256(canonical_json(config).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def environment() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "packages": {},
    }
    for name in ("numpy", "torch", "transformers", "tokenizers", "onnxruntime", "sklearn"):
        try:
            module = __import__(name)
            info["packages"][name] = getattr(module, "__version__", "unknown")
        except ImportError:
            continue
    try:
        import torch

        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda
    except ImportError:
        pass
    return info


def collect(
    command: Iterable[str],
    *,
    seed: int | None,
    config: Any = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "command": " ".join(command),
        **git_state(),
        "code_fingerprint": code_fingerprint(),
        "seed": seed,
        "environment": environment(),
    }
    if config is not None:
        record["config"] = config
        record["config_hash"] = config_hash(config)
    record.update(extra or {})
    return record


TRAINING_CODE = (
    "ml/train",
    "ml/data/contractnli.py",
    "ml/data/store.py",
    "ml/eval/infer.py",
    "ml/eval/metrics.py",
    "apps/model-service/src/compliance_model_service/rule_check/encoding.py",
    "apps/model-service/src/compliance_model_service/rule_check/labels.py",
    "apps/model-service/src/compliance_model_service/rule_check/onnx_backend.py",
)


def training_code_fingerprint() -> str:
    """SHA-256 over the code that defines training inputs, the model and the loop.

    A run may only be resumed by the same code (``ml.train.train`` checks it).
    """
    files: list[Path] = []
    for entry in TRAINING_CODE:
        path = REPO / entry
        files.extend(sorted(path.rglob("*.py")) if path.is_dir() else [path])
    digest = hashlib.sha256()
    for path in files:
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(REPO).as_posix()
        digest.update(rel.encode() + b"\0" + hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()
