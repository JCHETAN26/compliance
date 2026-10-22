"""Label order of the rule-check model's 3-way head (fixed; part of the artifact contract)."""

from __future__ import annotations

from typing import Final

# Index i of the label head's logits is LABELS[i].
LABELS: Final = ("entailment", "contradiction", "not_mentioned")
NUM_LABELS: Final = len(LABELS)

# ContractNLI annotation "choice" values -> label.
DATASET_LABELS: Final = {
    "Entailment": "entailment",
    "Contradiction": "contradiction",
    "NotMentioned": "not_mentioned",
}

# Model label -> application finding status (docs/BUILD_HANDOFF.md section 7.1).
APP_STATUS: Final = {
    "entailment": "satisfied",
    "contradiction": "violated",
    "not_mentioned": "not_mentioned",
}


def label_index(label: str) -> int:
    return LABELS.index(label)
