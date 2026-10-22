"""Regression for the Colab failure "Attempting to unscale FP16 gradients" (source 4cd76d8).

transformers 5 loads ``from_pretrained`` with ``dtype="auto"`` (the checkpoint's dtype when the
config names none) and ``from_config`` follows a ``dtype`` in the config. DeBERTa-v3-base ships
an fp16 ``pytorch_model.bin``, so the backbone came up with fp16 parameters and GradScaler
refused to unscale. These tests rebuild that situation with a tiny checkpoint saved in fp16 and
run a real, enabled GradScaler (CPU) through the production ``optimizer_update``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from ml.train.backbone import (
    BackboneFiles,
    ParameterDtypeError,
    build_architecture,
    build_backbone,
    require_float32,
)
from ml.train.config import BackboneSpec
from ml.train.model import RuleCheckModel
from ml.train.train import optimizer_update

REVISION = "0" * 40
TINY = {
    "hidden_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "intermediate_size": 64,
}


def _fp16_checkpoint(root: Path) -> BackboneFiles:
    from transformers import AutoModel, DebertaV2Config

    settings: dict[str, Any] = {
        "vocab_size": 1000,
        "max_position_embeddings": 128,
        "relative_attention": True,
        "position_buckets": 64,
        "norm_rel_ebd": "layer_norm",
        "share_att_key": True,
        "pos_att_type": ["p2c", "c2p"],
        "position_biased_input": False,
        "type_vocab_size": 0,
        **TINY,
    }
    config = DebertaV2Config(**settings)
    model = AutoModel.from_config(config).half()  # type: ignore[no-untyped-call]
    model.save_pretrained(str(root), safe_serialization=False)  # a .bin like deberta-v3-base
    saved = json.loads((root / "config.json").read_text())
    assert saved.get("dtype") == "float16"  # the config also declares fp16
    return BackboneFiles(root=root, tokenizer_json=root / "unused-tokenizer.json")


def _spec(kind: str) -> BackboneSpec:
    tokenizer = {"repo_id": "t/t", "revision": REVISION, "source": "tokenizer_json", "files": []}
    data = {"kind": kind, "repo_id": "t/t", "revision": REVISION, "tokenizer": tokenizer}
    if kind == "pretrained":
        data["weights"] = {"name": "pytorch_model.bin", "sha256": "0" * 64}
    else:
        data["tiny_overrides"] = TINY
    return BackboneSpec.model_validate(data)


def test_fp16_checkpoint_and_config_load_as_float32(tmp_path: Path) -> None:
    from transformers import AutoModel

    files = _fp16_checkpoint(tmp_path)
    default = AutoModel.from_pretrained(str(tmp_path), local_files_only=True)
    assert next(default.parameters()).dtype == torch.float16  # the hazard is real
    with pytest.raises(ParameterDtypeError, match="not float32"):
        require_float32(default)

    for model in (
        build_backbone(_spec("pretrained"), files, seed=1),  # pinned pretrained weights path
        build_backbone(_spec("random_tiny"), files, seed=1),  # smoke path, fp16 config
        build_architecture(_spec("random_tiny"), files),  # export path, fp16 config
    ):
        require_float32(model)
    # the fp16 weights were loaded (upcast), not replaced by a fresh initialisation
    loaded = build_backbone(_spec("pretrained"), files, seed=1)
    reference = dict(default.named_parameters())
    for name, param in loaded.named_parameters():
        assert torch.equal(param.detach(), reference[name].detach().float()), name


def _scaler() -> torch.amp.GradScaler:
    scaler = torch.amp.GradScaler("cpu", init_scale=2.0**10, enabled=True)
    assert scaler.is_enabled()
    return scaler


def test_real_grad_scaler_rejects_fp16_parameters() -> None:
    """The exact Colab failure, reproduced on CPU with the production update function."""
    layer = torch.nn.Linear(4, 1).half()
    optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    scaler = _scaler()
    scaled = scaler.scale(layer(torch.ones(2, 4, dtype=torch.float16)).float().sum())
    assert isinstance(scaled, torch.Tensor)
    torch.autograd.backward(scaled)
    with pytest.raises(ValueError, match="Attempting to unscale FP16 gradients"):
        optimizer_update(layer, optimizer, scheduler, scaler, 1.0)


def test_real_grad_scaler_trains_float32_master_weights_with_autocast(tmp_path: Path) -> None:
    files = _fp16_checkpoint(tmp_path)
    model = RuleCheckModel(build_backbone(_spec("pretrained"), files, seed=1), dropout=0.0)
    require_float32(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    scaler = _scaler()
    before = [p.detach().clone() for p in model.parameters()]
    ids = torch.randint(5, 1000, (3, 12))
    mask = torch.ones_like(ids)
    with torch.autocast("cpu", dtype=torch.bfloat16):  # low-precision activations only
        evidence, labels = model(ids, mask)
    assert labels.dtype == torch.bfloat16
    loss = torch.nn.functional.cross_entropy(labels.float(), torch.tensor([0, 1, 2]))
    loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(
        evidence.float(), torch.ones(3)
    )
    scaled = scaler.scale(loss)
    assert isinstance(scaled, torch.Tensor)
    torch.autograd.backward(scaled)
    assert all(p.grad is None or p.grad.dtype == torch.float32 for p in model.parameters())
    assert optimizer_update(model, optimizer, scheduler, scaler, 1.0) is True
    assert scheduler.last_epoch == 1
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters(), strict=True))
    require_float32(model)  # still float32 after the update
