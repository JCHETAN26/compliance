"""ONNX Runtime (CPU) execution of the exported rule-check graph.

The exported graph takes ``input_ids`` and ``attention_mask`` (int64, ``[batch, seq]``)
and returns ``evidence_logits`` ``[batch]`` and ``label_logits`` ``[batch, 3]``. Rows of
very different lengths (evidence spans vs label inputs) are sorted by length and run in
sub-batches capped by ``max_tokens_per_forward`` padded tokens, then restored to order.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from compliance_model_service.rule_check.encoding import InputEncoder
from compliance_model_service.rule_check.labels import NUM_LABELS

INPUTS = ("input_ids", "attention_mask")
OUTPUTS = ("evidence_logits", "label_logits")


class OnnxGraphError(RuntimeError):
    pass


def length_groups(lengths: Sequence[int], max_tokens: int) -> list[list[int]]:
    """Row indices grouped (shortest first) so that rows * longest <= max_tokens."""
    order = sorted(range(len(lengths)), key=lambda i: (lengths[i], i))
    groups: list[list[int]] = []
    current: list[int] = []
    for index in order:
        widest = lengths[index]  # ascending order: the newest row is the longest
        if current and (len(current) + 1) * widest > max_tokens:
            groups.append(current)
            current = []
        current.append(index)
    if current:
        groups.append(current)
    return groups


class OnnxBackend:
    def __init__(
        self,
        path: Path,
        encoder: InputEncoder,
        *,
        threads: int = 4,
        max_tokens_per_forward: int = 16_384,
    ) -> None:
        import onnxruntime as ort  # type: ignore[import-untyped]

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        inputs = tuple(i.name for i in self.session.get_inputs())
        outputs = tuple(o.name for o in self.session.get_outputs())
        if sorted(inputs) != sorted(INPUTS) or sorted(outputs) != sorted(OUTPUTS):
            raise OnnxGraphError(f"unexpected ONNX signature: inputs {inputs}, outputs {outputs}")
        self.encoder = encoder
        self.max_tokens = max_tokens_per_forward

    def __call__(self, rows: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
        evidence = np.empty(len(rows), dtype=np.float32)
        labels = np.empty((len(rows), NUM_LABELS), dtype=np.float32)
        for group in length_groups([len(r) for r in rows], self.max_tokens):
            ids, mask = self.encoder.pad([rows[i] for i in group])
            ev, lab = self.session.run(list(OUTPUTS), {"input_ids": ids, "attention_mask": mask})
            ev = np.asarray(ev, dtype=np.float32).reshape(-1)
            lab = np.asarray(lab, dtype=np.float32)
            if ev.shape != (len(group),) or lab.shape != (len(group), NUM_LABELS):
                raise OnnxGraphError(f"unexpected output shapes {ev.shape} / {lab.shape}")
            evidence[group] = ev
            labels[group] = lab
        return evidence, labels
