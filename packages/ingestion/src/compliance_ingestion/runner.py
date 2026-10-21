"""Run ``extract_cli`` in a child process with a timeout and memory limit."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from compliance_ingestion.types import ExtractionError, ExtractionResult


class ExtractionCrashError(RuntimeError):
    """The extraction child died unexpectedly (bug or resource exhaustion)."""


@dataclass(frozen=True)
class RunnerLimits:
    timeout_seconds: int
    memory_mb: int


def _child_env() -> dict[str, str]:
    """Single-threaded native libraries: thread pools reserve address space that the
    child's RLIMIT_AS would otherwise exhaust, and parsing is not CPU-parallel anyway."""
    env = dict(os.environ)
    env.update(
        {
            "TOKENIZERS_PARALLELISM": "false",
            "RAYON_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "OMP_THREAD_LIMIT": "1",  # Tesseract
            "OPENBLAS_NUM_THREADS": "1",
        }
    )
    return env


async def run_extraction(
    input_path: Path,
    fmt: str,
    tokenizer_path: Path,
    options: dict[str, int | str],
    limits: RunnerLimits,
) -> ExtractionResult:
    with tempfile.TemporaryDirectory(prefix="extract-") as tmp:
        output = Path(tmp) / "result.json"
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "compliance_ingestion.extract_cli",
            "--input",
            str(input_path),
            "--format",
            fmt,
            "--tokenizer",
            str(tokenizer_path),
            "--options",
            json.dumps(options),
            "--output",
            str(output),
            "--memory-mb",
            str(limits.memory_mb),
            env=_child_env(),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, stderr = await asyncio.wait_for(
                process.communicate(), timeout=limits.timeout_seconds
            )
        except TimeoutError:
            process.kill()
            await process.wait()
            raise ExtractionError(
                f"Processing took longer than the {limits.timeout_seconds}s limit."
            ) from None
        if process.returncode == 0:
            return ExtractionResult.model_validate_json(output.read_text(encoding="utf-8"))
        if process.returncode == 2 and output.exists():
            raise ExtractionError(str(json.loads(output.read_text(encoding="utf-8"))["error"]))
        # Only the last stderr line (an exception type/message) is kept, never content.
        tail = stderr.decode(errors="replace").strip().splitlines()[-1:] if stderr else []
        raise ExtractionCrashError(
            f"extraction process exited with {process.returncode}: {tail[0][:300] if tail else 'no output'}"
        )
