"""Two-pass rule-check inference over a micro-batched backend.

1. Every span is scored as ``[CLS] hypothesis [SEP] span [SEP]`` by the evidence head
   (probability = sigmoid of the evidence logit).
2. The ``top_k`` highest-scoring spans, in document (request) order, form one label
   input; the label head's logits are divided by the DEV-fitted temperature and
   softmaxed. The label is the argmax and its probability is the confidence.

``ml/eval`` computes offline predictions with the same encoder, ``top_k_in_order`` and
temperature, so served outputs match evaluated outputs up to numeric precision.

Admission: tokenization runs in a one-thread executor *before* rows reach the bounded
micro-batcher, so it has its own bound. At most ``max_pending_tokenize`` requests may
have tokenization work queued or running; a slot is released only when that executor
work has finished (or was cancelled before it started), never merely because the
caller went away. Beyond the bound a request is rejected (``OverloadedError`` -> 503).
Together with the per-request character budget enforced by the endpoint this bounds the
memory held by queued tokenization.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from compliance_model_service.rule_check.artifact import InferenceSpec
from compliance_model_service.rule_check.batching import MicroBatcher, OverloadedError
from compliance_model_service.rule_check.encoding import (
    InputEncoder,
    sigmoid,
    softmax,
    top_k_in_order,
)
from compliance_model_service.rule_check.labels import APP_STATUS, LABELS

MAX_EVIDENCE_RETURNED = 10


@dataclass(frozen=True)
class SpanText:
    span_id: str
    text: str


@dataclass(frozen=True)
class ScoredSpan:
    span_id: str
    score: float


@dataclass(frozen=True)
class CheckOutcome:
    label: str
    status: str
    confidence: float
    probabilities: dict[str, float]
    evidence: list[ScoredSpan]
    considered: list[ScoredSpan]
    truncated_inputs: int
    escalate: bool | None  # confidence < tau; None when no tau was selected


class RuleCheckEngine:
    def __init__(
        self,
        encoder: InputEncoder,
        batcher: MicroBatcher,
        inference: InferenceSpec,
        tokenize_pool: ThreadPoolExecutor,
        *,
        max_pending_tokenize: int = 8,
    ) -> None:
        if max_pending_tokenize < 1:
            raise ValueError("max_pending_tokenize must be >= 1")
        self.encoder = encoder
        self.batcher = batcher
        self.inference = inference
        self._tokenize_pool = tokenize_pool
        self.max_pending_tokenize = max_pending_tokenize
        self.pending_tokenize = 0
        self.rejected_tokenize = 0

    def _release_tokenize_slot(self) -> None:
        self.pending_tokenize -= 1

    async def _tokenize(self, texts: list[str]) -> list[list[int]]:
        if self.pending_tokenize >= self.max_pending_tokenize:
            self.rejected_tokenize += 1
            raise OverloadedError
        loop = asyncio.get_running_loop()
        self.pending_tokenize += 1
        work = self._tokenize_pool.submit(self.encoder.tokenize, texts)
        # Released when the executor work is really over (done, failed, or cancelled
        # before it started); the callback may run in the worker thread.
        work.add_done_callback(lambda _: loop.call_soon_threadsafe(self._release_tokenize_slot))
        return await asyncio.wrap_future(work)

    async def check(self, hypothesis: str, spans: Sequence[SpanText]) -> CheckOutcome:
        tokens = await self._tokenize([hypothesis, *(s.text for s in spans)])
        hyp, span_tokens = tokens[0], tokens[1:]
        rows = [self.encoder.evidence_input(hyp, s) for s in span_tokens]
        evidence_logits, _ = await self.batcher.infer([r.ids for r in rows])
        scores = [float(x) for x in sigmoid(evidence_logits)]

        chosen = top_k_in_order(scores, min(self.inference.top_k, len(spans)))
        label_row = self.encoder.label_input(hyp, [span_tokens[i] for i in chosen])
        _, label_logits = await self.batcher.infer([label_row.ids])
        probs = softmax(label_logits[0], self.inference.temperature)
        best = int(probs.argmax())
        confidence = float(probs[best])

        ranked = sorted(range(len(spans)), key=lambda i: (-scores[i], i))
        evidence = [
            ScoredSpan(spans[i].span_id, scores[i])
            for i in ranked
            if scores[i] >= self.inference.evidence_threshold
        ][:MAX_EVIDENCE_RETURNED]
        tau = self.inference.tau
        return CheckOutcome(
            label=LABELS[best],
            status=APP_STATUS[LABELS[best]],
            confidence=confidence,
            probabilities={label: float(p) for label, p in zip(LABELS, probs, strict=True)},
            evidence=evidence,
            considered=[ScoredSpan(spans[i].span_id, scores[i]) for i in chosen],
            truncated_inputs=sum(r.truncated for r in rows) + int(label_row.truncated),
            escalate=None if tau is None else confidence < tau,
        )
