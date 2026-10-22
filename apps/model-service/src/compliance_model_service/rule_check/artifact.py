"""Rule-check model artifacts: a manifest plus SHA-256-verified ONNX and tokenizer files.

Layout of an artifact directory (produced by ``ml/export``)::

    manifest.json
    tokenizer.json
    model.int8.onnx        # dynamic int8 quantization (served by default)
    model.fp32.onnx        # optional; served only with RULE_CHECK_VARIANT=fp32

The service refuses an artifact whose manifest is malformed, whose label order differs
from ``LABELS``, whose files are missing or whose SHA-256 does not match. ``kind`` says
whether the weights come from a real training run (``production``) or the CI smoke
pipeline (``smoke``: a tiny randomly initialised backbone trained for seconds on a data
subset); smoke artifacts are refused unless explicitly allowed (tests only).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from compliance_core.downloads import sha256_file
from compliance_model_service.rule_check.encoding import EncoderConfig
from compliance_model_service.rule_check.labels import LABELS

MANIFEST = "manifest.json"
FORMAT_VERSION = 1
Variant = Literal["int8", "fp32"]


class ArtifactError(RuntimeError):
    """The artifact is absent, malformed or fails verification."""


class _Frozen(BaseModel):
    # Finite numbers only: an infinite temperature or NaN threshold is a broken artifact.
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class ArtifactFile(_Frozen):
    path: str = Field(pattern=r"^[A-Za-z0-9._-]+$")  # a plain file name inside the dir
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes: int = Field(ge=1)


class EncoderSpec(_Frozen):
    max_hypothesis_tokens: int
    max_evidence_tokens: int
    max_label_tokens: int
    cls_token: str
    sep_token: str
    pad_token: str
    normalization: Literal["NFKC"] | None = None

    def to_config(self) -> EncoderConfig:
        return EncoderConfig(**self.model_dump())


class InferenceSpec(_Frozen):
    top_k: int = Field(ge=1, le=16)
    temperature: float = Field(gt=0)
    evidence_threshold: float = Field(ge=0, le=1)
    # Cascade threshold on calibrated confidence, selected on DEV (None: not selected).
    tau: float | None = Field(default=None, ge=0, le=1)


class ArtifactFiles(_Frozen):
    tokenizer: ArtifactFile
    onnx_int8: ArtifactFile | None = None
    onnx_fp32: ArtifactFile | None = None


class Manifest(_Frozen):
    format_version: Literal[1]
    # Also used as a directory name when installing: a plain, safe file name only.
    model_version: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$")
    kind: Literal["production", "smoke"]
    created_utc: str
    backbone: dict[str, Any]
    labels: tuple[str, ...]
    encoder: EncoderSpec
    inference: InferenceSpec
    files: ArtifactFiles
    # Informational: copied from the evaluation result files; never computed here.
    metrics: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)

    @field_validator("labels")
    @classmethod
    def _labels_match(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if tuple(value) != LABELS:
            raise ValueError(f"label order {value} does not match the service's {LABELS}")
        return value


class VerifiedArtifact(_Frozen):
    root: Path
    manifest: Manifest
    variant: Variant
    onnx_path: Path
    tokenizer_path: Path


def read_manifest(root: Path) -> Manifest:
    path = root / MANIFEST
    if not path.is_file():
        raise ArtifactError(f"no rule-check artifact manifest at {path}")
    try:
        return Manifest.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (ValueError, ValidationError) as exc:
        raise ArtifactError(f"invalid rule-check manifest {path}: {exc}") from exc


def _verified_file(root: Path, entry: ArtifactFile) -> Path:
    path = (root / entry.path).resolve()
    if path.parent != root.resolve():
        raise ArtifactError(f"artifact file {entry.path!r} escapes {root}")
    if not path.is_file():
        raise ArtifactError(f"artifact file {path} is missing")
    size = path.stat().st_size
    if size != entry.bytes:
        raise ArtifactError(f"{path} has {size} bytes, manifest says {entry.bytes}")
    actual = sha256_file(path)
    if actual != entry.sha256:
        raise ArtifactError(f"{path} SHA-256 {actual} does not match manifest {entry.sha256}")
    return path


def load_artifact(root: Path, *, variant: Variant, allow_smoke: bool) -> VerifiedArtifact:
    manifest = read_manifest(root)
    if manifest.kind == "smoke" and not allow_smoke:
        raise ArtifactError(
            f"{root} holds a CI smoke artifact (tiny random backbone), not a trained model; "
            "refusing to serve it (RULE_CHECK_ALLOW_SMOKE is false)"
        )
    entry = manifest.files.onnx_int8 if variant == "int8" else manifest.files.onnx_fp32
    if entry is None:
        raise ArtifactError(f"artifact {manifest.model_version} has no {variant} ONNX file")
    return VerifiedArtifact(
        root=root,
        manifest=manifest,
        variant=variant,
        onnx_path=_verified_file(root, entry),
        tokenizer_path=_verified_file(root, manifest.files.tokenizer),
    )


def describe_file(path: Path) -> ArtifactFile:
    return ArtifactFile(path=path.name, sha256=sha256_file(path), bytes=path.stat().st_size)
