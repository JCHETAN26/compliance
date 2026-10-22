"""The rule-check cross-encoder: one backbone, an evidence head and a 3-way label head.

Both heads read the first-token ([CLS]) representation of the same backbone pass:

- evidence head -> one logit per ``[CLS] hypothesis [SEP] span [SEP]`` input;
- label head -> three logits (``LABELS`` order) per label input.

``forward`` always returns both outputs (the heads are tiny), which is also the exported
ONNX signature: ``input_ids``, ``attention_mask`` -> ``evidence_logits [B]``,
``label_logits [B, 3]``. Callers use the head that matches the input they built.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from compliance_model_service.rule_check.labels import NUM_LABELS


def _head(hidden: int, out: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(hidden, hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, out),
    )


class RuleCheckModel(nn.Module):
    def __init__(self, backbone: Any, *, dropout: float = 0.1) -> None:
        super().__init__()
        self.backbone = backbone
        hidden = int(backbone.config.hidden_size)
        self.evidence_head = _head(hidden, 1, dropout)
        self.label_head = _head(hidden, NUM_LABELS, dropout)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0]
        evidence = self.evidence_head(pooled).squeeze(-1)
        labels = self.label_head(pooled)
        return evidence, labels

    def head_parameters(self) -> list[nn.Parameter]:
        return [*self.evidence_head.parameters(), *self.label_head.parameters()]
