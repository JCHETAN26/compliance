"""Rule-check model: shared input encoding, artifact verification and inference.

This package is the single source of truth for how a (hypothesis, spans) request
becomes model inputs. ``ml/`` (training, evaluation, export) imports the same encoder,
label order and top-k logic, so served predictions are computed exactly as evaluated.
"""

from compliance_model_service.rule_check.labels import (
    APP_STATUS,
    DATASET_LABELS,
    LABELS,
    NUM_LABELS,
)

__all__ = ["APP_STATUS", "DATASET_LABELS", "LABELS", "NUM_LABELS"]
