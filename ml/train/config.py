"""Training configuration (YAML -> validated model). Every run stores it and its hash."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from compliance_model_service.rule_check.encoding import EncoderConfig


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class PinnedFile(_Strict):
    name: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes: int | None = None


class TokenizerSpec(_Strict):
    repo_id: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    # "spm": convert spm.model to tokenizer.json with transformers (needs sentencepiece);
    # "tokenizer_json": the repository ships tokenizer.json.
    source: Literal["spm", "tokenizer_json"]
    files: list[PinnedFile]


class BackboneSpec(_Strict):
    kind: Literal["pretrained", "random_tiny"]
    repo_id: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    weights: PinnedFile | None = None  # required for pretrained
    config_file: PinnedFile | None = None
    # random_tiny only: overrides applied to the repository's config (seeded random init).
    tiny_overrides: dict[str, Any] | None = None
    tokenizer: TokenizerSpec

    @model_validator(mode="after")
    def _consistent(self) -> BackboneSpec:
        if self.kind == "pretrained" and self.weights is None:
            raise ValueError("pretrained backbones must pin their weights file")
        if self.kind == "random_tiny" and not self.tiny_overrides:
            raise ValueError("random_tiny backbones need tiny_overrides")
        return self


class EncoderSettings(_Strict):
    max_hypothesis_tokens: int = 64
    max_evidence_tokens: int = 256
    max_label_tokens: int = 512
    cls_token: str = "[CLS]"  # noqa: S105 - tokenizer special token
    sep_token: str = "[SEP]"  # noqa: S105
    pad_token: str = "[PAD]"  # noqa: S105
    # Applied to every text before tokenization (training and serving alike).
    normalization: Literal["NFKC"] | None = None

    def to_config(self) -> EncoderConfig:
        return EncoderConfig(**self.model_dump())


class DataSettings(_Strict):
    data_dir: str = ".cache/ml/data/full"
    # Unseen-hypothesis experiment: these keys are removed from TRAIN entirely and from
    # the DEV pairs used for any selection; they are evaluated on TEST only.
    held_out_hypotheses: list[str] = Field(default_factory=list)


class SamplingSettings(_Strict):
    """Per-epoch TRAIN example construction (seeded by seed + epoch)."""

    # Evidence examples per pair: every gold segment, plus negatives drawn from the
    # pair's dense ranking (hard) and uniformly from the rest of the document (random).
    hard_negatives: int = Field(default=4, ge=0)
    random_negatives: int = Field(default=4, ge=0)
    hard_negative_depth: int = Field(default=16, ge=1)
    # Label examples: k spans per pair, k drawn from label_k_choices. Evidence pairs
    # keep their gold segments (each dropped with gold_drop_prob while one remains),
    # the remaining slots are filled from the dense ranking. Inference instead reads the
    # evidence head's own top-k (gold is never available there).
    label_k_choices: list[int] = Field(default_factory=lambda: [3, 4, 5])
    gold_drop_prob: float = Field(default=0.15, ge=0, le=1)


class TrainingSettings(_Strict):
    epochs: int = Field(default=3, ge=1)
    evidence_batch_size: int = Field(default=32, ge=1)
    label_batch_size: int = Field(default=8, ge=1)
    grad_accum_steps: int = Field(default=1, ge=1)
    lr: float = Field(default=2e-5, gt=0)
    head_lr: float = Field(default=1e-4, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    warmup_ratio: float = Field(default=0.06, ge=0, lt=1)
    max_grad_norm: float = Field(default=1.0, gt=0)
    evidence_loss_weight: float = Field(default=1.0, ge=0)
    label_loss_weight: float = Field(default=1.0, ge=0)
    class_weighting: Literal["inverse_frequency", "none"] = "inverse_frequency"
    dropout: float = Field(default=0.1, ge=0, lt=1)
    fp16: bool = True  # CUDA only
    max_steps: int | None = Field(default=None, ge=1)  # smoke cap
    checkpoint_every_steps: int = Field(default=500, ge=1)


class SelectionSettings(_Strict):
    """All model selection uses DEV (seen hypotheses only); TEST is never read here."""

    metric: Literal["macro_f1"] = "macro_f1"
    top_k_candidates: list[int] = Field(default_factory=lambda: [3, 4, 5])


class InferenceSettings(_Strict):
    # Rows handed to one forward call; it sorts them by length and runs sub-batches of at
    # most max_tokens_per_forward padded tokens.
    chunk_rows: int = Field(default=8192, ge=1)
    max_tokens_per_forward: int = Field(default=32_768, ge=512)


class TrainConfig(_Strict):
    run_name: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]{0,80}$")
    kind: Literal["production", "smoke"]
    seed: int = 20261021
    requires_cuda: bool = True
    backbone: BackboneSpec
    encoder: EncoderSettings = Field(default_factory=EncoderSettings)
    data: DataSettings = Field(default_factory=DataSettings)
    sampling: SamplingSettings = Field(default_factory=SamplingSettings)
    training: TrainingSettings = Field(default_factory=TrainingSettings)
    selection: SelectionSettings = Field(default_factory=SelectionSettings)
    inference: InferenceSettings = Field(default_factory=InferenceSettings)

    @model_validator(mode="after")
    def _consistent(self) -> TrainConfig:
        if self.kind == "production" and self.backbone.kind != "pretrained":
            raise ValueError("production runs need a pretrained backbone")
        if self.kind == "production" and self.training.max_steps is not None:
            raise ValueError("production runs must not cap training steps")
        for k in [*self.sampling.label_k_choices, *self.selection.top_k_candidates]:
            if k < 1:
                raise ValueError("k values must be >= 1")
        return self


def load_config(path: Path) -> TrainConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return TrainConfig.model_validate(raw)
