"""The trained rule-check model as seen by the agents.

``ServiceRuleCheckModel`` calls the model service (``POST /v1/check``) with a bounded
timeout and retries for transient failures, and validates every answer. Its tau is the
DEV-selected threshold from the served artifact's metadata (0.3 for the installed fp32
model); a workspace may override it. ``MockRuleCheckModel`` is the deterministic
lexical stand-in for tests and offline demos; it identifies itself as a mock everywhere.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from typing import Annotated, Literal, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from compliance_llm import lexical_judgement

MAX_SPANS = 512
LABEL_STATUS = {
    "entailment": "satisfied",
    "contradiction": "violated",
    "not_mentioned": "not_mentioned",
}


class ModelUnavailableError(Exception):
    def __init__(self, message: str, code: str = "model_unavailable") -> None:
        super().__init__(message)
        self.message = message
        self.code = code


@dataclass(frozen=True)
class ModelInfo:
    available: bool
    backend: Literal["service", "mock"]
    model_version: str | None
    kind: str | None
    variant: str | None
    tau: float | None
    reason: str | None = None


@dataclass(frozen=True)
class ModelVerdict:
    status: str
    label: str
    confidence: float
    probabilities: dict[str, float]
    evidence: list[tuple[str, float]]
    considered: list[tuple[str, float]]
    model_version: str


class RuleCheckModel(Protocol):
    backend: Literal["service", "mock"]

    async def info(self) -> ModelInfo: ...

    async def check(self, hypothesis: str, spans: list[tuple[str, str]]) -> ModelVerdict: ...

    async def aclose(self) -> None: ...


Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
PROBABILITY_TOLERANCE = 1e-3


class _Evidence(BaseModel):
    model_config = ConfigDict(extra="ignore")
    span_id: str
    score: Probability


class _CheckBody(BaseModel):
    """The model service's answer, validated for internal consistency: a malformed
    answer must never become a (high-confidence) finding."""

    model_config = ConfigDict(extra="ignore")
    label: Literal["entailment", "contradiction", "not_mentioned"]
    status: Literal["satisfied", "violated", "not_mentioned"]
    confidence: Probability
    probabilities: dict[str, Probability]
    evidence: list[_Evidence]
    considered: list[_Evidence]
    model_version: Annotated[str, Field(min_length=1)]
    model_kind: Literal["production", "smoke"]

    @model_validator(mode="after")
    def _consistent(self) -> _CheckBody:
        if set(self.probabilities) != set(LABEL_STATUS):
            raise ValueError("probabilities must cover exactly the three labels")
        if abs(sum(self.probabilities.values()) - 1.0) > PROBABILITY_TOLERANCE:
            raise ValueError("probabilities are not normalised")
        if LABEL_STATUS[self.label] != self.status:
            raise ValueError("label and status disagree")
        if abs(self.probabilities[self.label] - self.confidence) > PROBABILITY_TOLERANCE:
            raise ValueError("confidence differs from the label's probability")
        if self.probabilities[self.label] < max(self.probabilities.values()) - 1e-9:
            raise ValueError("label is not the most probable class")
        for name, items in (("evidence", self.evidence), ("considered", self.considered)):
            ids = [e.span_id for e in items]
            if len(ids) != len(set(ids)):
                raise ValueError(f"duplicate {name} span ids")
        return self


class ServiceRuleCheckModel:
    backend: Literal["service", "mock"] = "service"

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float,
        retries: int = 3,
        transport: httpx.AsyncBaseTransport | None = None,
        backoff: float = 1.0,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout, connect=min(timeout, 5.0)),
            transport=transport,
        )
        self._retries = retries
        self._backoff = backoff

    async def aclose(self) -> None:
        await self._client.aclose()

    async def info(self) -> ModelInfo:
        try:
            response = await self._client.get("/metadata")
            body = response.json() if response.status_code == 200 else None
        except (httpx.HTTPError, ValueError):
            body = None
        rc = body.get("rule_check") if isinstance(body, dict) else None
        if not isinstance(rc, dict):
            return ModelInfo(False, "service", None, None, None, None, "model service unreachable")
        if not rc.get("available"):
            return ModelInfo(False, "service", None, None, None, None, "no trained model loaded")
        if rc.get("kind") != "production":
            return ModelInfo(
                False,
                "service",
                rc.get("model_version"),
                rc.get("kind"),
                rc.get("variant"),
                None,
                "the loaded artifact is a smoke-test model, not a trained model",
            )
        inference = rc.get("inference") if isinstance(rc.get("inference"), dict) else {}
        assert isinstance(inference, dict)  # noqa: S101
        tau = inference.get("tau")
        version = rc.get("model_version")
        if not isinstance(version, str) or not version:
            return ModelInfo(False, "service", None, None, None, None, "no model version")
        if (
            isinstance(tau, bool)
            or not isinstance(tau, int | float)
            or not math.isfinite(tau)
            or not 0 <= tau <= 1
        ):
            # Without the DEV-selected threshold the cascade cannot be applied.
            return ModelInfo(
                False,
                "service",
                version,
                str(rc.get("kind")),
                None,
                None,
                "the model artifact has no valid DEV-selected tau",
            )
        return ModelInfo(
            True,
            "service",
            version,
            str(rc.get("kind")),
            str(rc.get("variant")) if rc.get("variant") else None,
            float(tau),
        )

    async def check(self, hypothesis: str, spans: list[tuple[str, str]]) -> ModelVerdict:
        if not spans:
            raise ModelUnavailableError("no spans to check", "no_spans")
        if len(spans) > MAX_SPANS:
            raise ModelUnavailableError(
                f"document has {len(spans)} sentence spans; the model accepts {MAX_SPANS}",
                "too_many_spans",
            )
        payload = {
            "hypothesis": hypothesis,
            "spans": [{"span_id": sid, "text": text} for sid, text in spans],
        }
        last = "no attempt"
        for attempt in range(self._retries + 1):
            if attempt:
                await asyncio.sleep(self._backoff * (2 ** (attempt - 1)))
            try:
                response = await self._client.post("/v1/check", json=payload)
            except httpx.TimeoutException:
                last = "timeout"
                continue
            except httpx.TransportError as exc:
                last = type(exc).__name__
                continue
            if response.status_code in (502, 503, 504):
                last = f"HTTP {response.status_code}"
                continue
            if response.status_code in (413, 422):
                raise ModelUnavailableError(
                    f"model service rejected the request (HTTP {response.status_code})",
                    "model_request_rejected",
                )
            if response.status_code != 200:
                raise ModelUnavailableError(
                    f"model service failed with HTTP {response.status_code}", "model_failed"
                )
            try:
                body = _CheckBody.model_validate_json(response.content)
            except ValidationError:
                raise ModelUnavailableError(
                    "model service answer violates its contract", "model_bad_response"
                ) from None
            ids = {sid for sid, _ in spans}
            if body.model_kind != "production" or any(
                e.span_id not in ids for e in [*body.evidence, *body.considered]
            ):
                raise ModelUnavailableError(
                    "model service answer is not usable (smoke model or unknown span ids)",
                    "model_bad_response",
                )
            return ModelVerdict(
                status=body.status,
                label=body.label,
                confidence=body.confidence,
                probabilities=body.probabilities,
                evidence=[(e.span_id, e.score) for e in body.evidence],
                considered=[(e.span_id, e.score) for e in body.considered],
                model_version=body.model_version,
            )
        raise ModelUnavailableError(f"model service unavailable after retries: {last}")


class MockRuleCheckModel:
    """Deterministic lexical stand-in (tests and offline demos only)."""

    backend: Literal["service", "mock"] = "mock"
    MODEL_VERSION = "mock-rule-check-v1"
    TAU = 0.3

    async def aclose(self) -> None:
        return None

    async def info(self) -> ModelInfo:
        return ModelInfo(True, "mock", self.MODEL_VERSION, "mock", None, self.TAU)

    async def check(self, hypothesis: str, spans: list[tuple[str, str]]) -> ModelVerdict:
        await asyncio.sleep(0)
        if not spans:
            raise ModelUnavailableError("no spans to check", "no_spans")
        judgement = lexical_judgement(hypothesis, spans)
        # Deterministic stand-in confidences in [0.34, 0.99] (never certainty).
        if judgement.status == "not_mentioned":
            confidence = round(min(max(1.0 - judgement.score, 0.34), 0.99), 4)
        else:
            confidence = round(0.5 + 0.49 * judgement.score, 4)
        rest = round((1.0 - confidence) / 2, 4)
        label = {v: k for k, v in LABEL_STATUS.items()}[judgement.status]
        probabilities = dict.fromkeys(LABEL_STATUS, rest)
        probabilities[label] = confidence
        evidence = [(sid, round(judgement.score, 4)) for sid in judgement.evidence_ids]
        return ModelVerdict(
            status=judgement.status,
            label=label,
            confidence=confidence,
            probabilities=probabilities,
            evidence=evidence,
            considered=evidence,
            model_version=self.MODEL_VERSION,
        )
