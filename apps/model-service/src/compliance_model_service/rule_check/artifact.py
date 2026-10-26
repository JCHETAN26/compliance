"""Rule-check model artifacts: a manifest plus SHA-256-verified ONNX and tokenizer files.

Layout of an artifact directory (produced by ``ml/export``)::

    manifest.json
    tokenizer.json
    model.int8.onnx        # weight-only int8 quantization (served by default)
    model.fp32.onnx        # optional; served only with RULE_CHECK_VARIANT=fp32

The service refuses an artifact whose manifest is malformed, whose label order differs
from ``LABELS``, whose files are missing or whose SHA-256 does not match. ``kind`` says
whether the weights come from a real training run (``production``) or the CI smoke
pipeline (``smoke``: a tiny randomly initialised backbone trained for seconds on a data
subset); smoke artifacts are refused unless explicitly allowed (tests only).

Quality gate: a production int8 file is served only if the export's DEV parity check
recorded in ``provenance.parity.int8`` shows it keeps the fp32 model's decisions
(``int8_quality_problem``). Dynamic quantization can destroy a backbone; such an int8
file is refused with the reason, and fp32 (``RULE_CHECK_VARIANT=fp32``) is named when the
artifact has it. ``verify_artifact`` checks file integrity only (for tools that measure
or move artifacts, not serve them).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from compliance_core.downloads import sha256_file
from compliance_model_service.rule_check.encoding import EncoderConfig
from compliance_model_service.rule_check.labels import LABELS

MANIFEST = "manifest.json"
FORMAT_VERSION = 1
Variant = Literal["int8", "fp32"]
# Minimum int8-vs-PyTorch agreement on the export's seeded DEV parity pairs. These catch
# a collapsed quantization (the first production export measured 0.083 / 0.0), not small
# drift; the accuracy effect itself is measured separately (ml.export.eval_onnx).
INT8_MIN_LABEL_AGREEMENT = 0.9
INT8_MIN_TOP_K_AGREEMENT = 0.75


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


def _agreement(value: object) -> float | None:
    """A recorded agreement fraction in [0, 1], else None (bool, NaN, inf, out of range)."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if 0 <= value <= 1 else None  # NaN fails the comparison too


def int8_quality_problem(parity: object) -> str | None:
    """Why the recorded int8 parity does not allow serving int8 (None: it does).

    Fails closed: missing, non-numeric, non-finite or out-of-range evidence is a problem.
    """
    int8 = parity.get("int8") if isinstance(parity, Mapping) else None
    if not isinstance(int8, Mapping):
        return "the manifest records no int8 parity check"
    raw_label, raw_top_k = int8.get("label_argmax_agreement"), int8.get("top_k_agreement")
    label, top_k = _agreement(raw_label), _agreement(raw_top_k)
    if label is None or top_k is None:
        return (
            "the manifest's int8 parity check is incomplete or malformed: label agreement "
            f"{raw_label!r}, top-k agreement {raw_top_k!r} (each must be a number in [0, 1])"
        )
    if label < INT8_MIN_LABEL_AGREEMENT or top_k < INT8_MIN_TOP_K_AGREEMENT:
        return (
            f"int8 parity on DEV failed: label agreement {label:.3f} "
            f"(min {INT8_MIN_LABEL_AGREEMENT}), top-k agreement {top_k:.3f} "
            f"(min {INT8_MIN_TOP_K_AGREEMENT})"
        )
    return None


def verify_artifact(root: Path) -> tuple[Manifest, tuple[Variant, ...]]:
    """Manifest, tokenizer and every present listed ONNX file verified (size, SHA-256).

    Returns the variants whose files are present. A listed file may be absent (results
    bundles leave the large fp32 file behind); at least one ONNX file must be present.
    No serving or quality decision is made here (see ``load_artifact``).
    """
    manifest = read_manifest(root)
    _verified_file(root, manifest.files.tokenizer)
    present: list[Variant] = []
    listed: tuple[tuple[Variant, ArtifactFile | None], ...] = (
        ("int8", manifest.files.onnx_int8),
        ("fp32", manifest.files.onnx_fp32),
    )
    for variant, entry in listed:
        if entry is not None and (root / entry.path).exists():
            _verified_file(root, entry)
            present.append(variant)
    if not present:
        raise ArtifactError(f"artifact {manifest.model_version} has no ONNX file present")
    return manifest, tuple(present)


def load_artifact(root: Path, *, variant: Variant, allow_smoke: bool) -> VerifiedArtifact:
    """The verified files to serve ``variant`` (refuses smoke and failed-quality int8)."""
    manifest = read_manifest(root)
    if manifest.kind == "smoke" and not allow_smoke:
        raise ArtifactError(
            f"{root} holds a CI smoke artifact (tiny random backbone), not a trained model; "
            "refusing to serve it (RULE_CHECK_ALLOW_SMOKE is false)"
        )
    fp32_hint = (
        "; the artifact also has an fp32 file: set RULE_CHECK_VARIANT=fp32"
        if manifest.files.onnx_fp32 is not None
        else ""
    )
    entry = manifest.files.onnx_int8 if variant == "int8" else manifest.files.onnx_fp32
    if entry is None:
        hint = fp32_hint if variant == "int8" else ""
        raise ArtifactError(f"artifact {manifest.model_version} has no {variant} ONNX file{hint}")
    if variant == "int8" and manifest.kind == "production":
        problem = int8_quality_problem(manifest.provenance.get("parity"))
        if problem is not None:
            raise ArtifactError(
                f"refusing the int8 file of {manifest.model_version}: {problem}{fp32_hint}"
            )
    return VerifiedArtifact(
        root=root,
        manifest=manifest,
        variant=variant,
        onnx_path=_verified_file(root, entry),
        tokenizer_path=_verified_file(root, manifest.files.tokenizer),
    )


def describe_file(path: Path) -> ArtifactFile:
    return ArtifactFile(path=path.name, sha256=sha256_file(path), bytes=path.stat().st_size)
