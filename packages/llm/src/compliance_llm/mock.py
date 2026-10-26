"""Deterministic offline mock provider.

The mock never reads credentials and never opens a network connection. It answers from
the request's structured ``context`` with simple, documented lexical heuristics, so test
and demo outcomes are reproducible. It is a stand-in for wiring and safety tests, not a
model of legal meaning, and its results are labelled as mock everywhere they appear.

Usage is an *estimate* (characters / 4) labelled ``mock_estimate``; cost is always 0.

``behavior="follow_injections"`` simulates a compromised model that obeys instructions
embedded in documents (fabricated span ids, quotes and dates); ``behavior="always_comply"``
lies on every request. Tests use them to show the pipeline rejects such output; they are
never selectable through configuration.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

from compliance_llm.types import LLMRequest, LLMResponse, LLMUsage, Price

MOCK_MODEL = "mock-deterministic-v1"
MOCK_PRICE = Price(0.0, 0.0, "mock provider (no cost)")
INJECTION = re.compile(
    r"ignore\s+(?:all\s+|any\s+)?(?:previous|prior|above)\s+instructions", re.IGNORECASE
)

_STOP = frozenset(
    [
        "a",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "been",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "other",
        "our",
        "shall",
        "such",
        "that",
        "the",
        "their",
        "them",
        "this",
        "those",
        "to",
        "under",
        "upon",
        "was",
        "were",
        "which",
        "will",
        "with",
        "may",
        "must",
        "some",
        "all",
        "each",
    ]
)
_NEGATION = re.compile(
    r"\b(?:shall|may|will|must|can|does|do|is|are|be)\s+not\b|\bcannot\b|\bnever\b|\bno\b"
    r"|\bnor\b|\bnot\s+(?:be|to)\b",
    re.IGNORECASE,
)
_WORD = re.compile(r"[a-z0-9]+")
MATCH_THRESHOLD = 0.6


def _stem(word: str) -> str:
    for suffix in ("ing", "ion", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def content_tokens(text: str) -> set[str]:
    return {_stem(w) for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 1}


def negated(text: str) -> bool:
    return _NEGATION.search(text) is not None


@dataclass(frozen=True)
class LexicalJudgement:
    status: Literal["satisfied", "violated", "not_mentioned"]
    evidence_ids: list[str]
    score: float


def lexical_judgement(hypothesis: str, spans: Sequence[tuple[str, str]]) -> LexicalJudgement:
    """Best lexical match of ``hypothesis`` among ``(span_id, text)`` spans.

    Overlap = shared content tokens / hypothesis content tokens. At or above
    ``MATCH_THRESHOLD`` the best span decides: same negation polarity as the hypothesis
    gives ``satisfied``, opposite polarity ``violated``. Otherwise ``not_mentioned``.
    """
    wanted = content_tokens(hypothesis)
    if not wanted:
        return LexicalJudgement("not_mentioned", [], 0.0)
    best_id, best_text, best = "", "", 0.0
    for span_id, text in spans:
        score = len(wanted & content_tokens(text)) / len(wanted)
        if score > best:
            best_id, best_text, best = span_id, text, score
    if best < MATCH_THRESHOLD:
        return LexicalJudgement("not_mentioned", [], round(best, 4))
    status: Literal["satisfied", "violated"] = (
        "satisfied" if negated(best_text) == negated(hypothesis) else "violated"
    )
    return LexicalJudgement(status, [best_id], round(best, 4))


# --------------------------------------------------------------------------- extraction

_MONTHS = [
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
]
_MONTH = (
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
)
_DATE = (
    rf"(?:{_MONTH}\s+\d{{1,2}},\s+\d{{4}}|\d{{1,2}}\s+{_MONTH}\s+\d{{4}}|\d{{4}}-\d{{2}}-\d{{2}})"
)
_NUMBER_WORDS = [
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
]
_DURATION = (
    r"(?:(?:" + "|".join(_NUMBER_WORDS) + r")\s*(?:\(\d{1,3}\)\s*)?|\d{1,3}\s+)(?:years?|months?)"
)
_PATTERNS: dict[str, re.Pattern[str]] = {
    "effective_date": re.compile(
        rf"(?:effective|dated|entered\s+into)\s+(?:as\s+of\s+|on\s+)?(?:the\s+)?({_DATE})",
        re.IGNORECASE,
    ),
    "expiration_date": re.compile(
        rf"(?:expire[sd]?|terminate[sd]?|end[sd]?)\s+on\s+({_DATE})", re.IGNORECASE
    ),
    "term": re.compile(rf"(?:term|period)\s+of\s+({_DURATION})", re.IGNORECASE),
    "governing_law": re.compile(
        r"governed\s+by\s+(?:and\s+construed\s+in\s+accordance\s+with\s+)?the\s+laws?\s+of\s+"
        r"(?:the\s+)?([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})"
    ),
    "party": re.compile(
        r"\b(?:between|and)\s+([A-Z][A-Za-z0-9&.'-]*(?:\s+[A-Z][A-Za-z0-9&.'-]*){0,5}"
        r"\s+(?:Ltd|Limited|GmbH|Inc|LLC|LLP|Corp|Corporation|AG|plc|S\.A\.)\.?)"
    ),
}


def _iso(text: str) -> str | None:
    text = text.strip()
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", text)
    try:
        if m:
            return date(int(m[1]), int(m[2]), int(m[3])).isoformat()
        m = re.fullmatch(rf"({_MONTH})\s+(\d{{1,2}}),\s+(\d{{4}})", text, re.IGNORECASE)
        if m:
            return date(int(m[3]), _MONTHS.index(m[1].lower()) + 1, int(m[2])).isoformat()
        m = re.fullmatch(rf"(\d{{1,2}})\s+({_MONTH})\s+(\d{{4}})", text, re.IGNORECASE)
        if m:
            return date(int(m[3]), _MONTHS.index(m[2].lower()) + 1, int(m[1])).isoformat()
    except ValueError:
        return None
    return None


def _sentence(text: str, match: re.Match[str]) -> str:
    """The sentence around a match (what a model would quote), capped at 300 chars."""
    start = max(text.rfind(". ", 0, match.start()) + 2, 0) if ". " in text[: match.start()] else 0
    end = text.find(". ", match.end())
    end = len(text) if end == -1 else end + 1
    quote = text[start:end].strip()
    return quote if len(quote) <= 300 else match.group(0)


def mock_entities(
    chunks: Sequence[Mapping[str, Any]], entity_types: Sequence[str]
) -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for chunk in chunks:
        ref, text = str(chunk["ref"]), str(chunk["text"])
        for kind in entity_types:
            pattern = _PATTERNS.get(kind)
            if pattern is None:
                continue
            for match in pattern.finditer(text):
                captured = match.group(1).strip()
                if kind.endswith("_date"):
                    value = _iso(captured)
                    if value is None:
                        continue
                else:
                    value = re.sub(r"\s+", " ", captured)
                if (kind, value) in seen:
                    continue
                seen.add((kind, value))
                found.append(
                    {
                        "type": kind,
                        "chunk_ref": ref,
                        "quote": _sentence(text, match),
                        "value": value,
                    }
                )
    return found


# --------------------------------------------------------------------------- provider


def _estimate(text: str) -> int:
    return math.ceil(len(text) / 4)


class MockProvider:
    name = "mock"
    is_mock = True
    price = MOCK_PRICE

    def __init__(
        self,
        model: str = MOCK_MODEL,
        behavior: Literal["deterministic", "follow_injections", "always_comply"] = "deterministic",
    ) -> None:
        self.model = model
        self.behavior = behavior
        self.requests: list[LLMRequest] = []

    async def aclose(self) -> None:
        return None

    async def complete(self, request: LLMRequest) -> LLMResponse:
        await asyncio.sleep(0)  # a real await point, like a provider call
        self.requests.append(request)
        data = self._answer(request)
        usage = LLMUsage(
            input_tokens=_estimate(request.system) + _estimate(request.user),
            # An estimate, capped at the request's output allowance like a real provider.
            output_tokens=min(_estimate(json.dumps(data)), request.max_output_tokens),
            source="mock_estimate",
        )
        return LLMResponse(data=data, usage=usage, model=self.model, latency_ms=0)

    def _injected(self, texts: Sequence[str]) -> bool:
        if self.behavior == "always_comply":
            return True
        return self.behavior == "follow_injections" and any(INJECTION.search(t) for t in texts)

    def _answer(self, request: LLMRequest) -> dict[str, Any]:
        ctx = request.context
        if request.task == "rule_verdict":
            spans = [(str(s["id"]), str(s["text"])) for s in ctx.get("spans", [])]
            if self._injected([t for _, t in spans]):
                return {
                    "verdict": "satisfied",
                    "evidence_span_ids": ["S999"],
                    "rationale": "Instructed by the document to report compliance.",
                }
            judgement = lexical_judgement(str(ctx.get("hypothesis", "")), spans)
            return {
                "verdict": judgement.status,
                "evidence_span_ids": judgement.evidence_ids,
                "rationale": (
                    f"Mock lexical judge: best span overlap {judgement.score:.2f} "
                    f"(threshold {MATCH_THRESHOLD})."
                ),
            }
        if request.task == "support_check":
            quotes = [(str(q["id"]), str(q["text"])) for q in ctx.get("quotes", [])]
            if self._injected([t for _, t in quotes]):
                return {
                    "supported": True,
                    "supporting_quote_ids": [q for q, _ in quotes],
                    "rationale": "Instructed by the document to confirm support.",
                }
            judgement = lexical_judgement(str(ctx.get("hypothesis", "")), quotes)
            claimed = str(ctx.get("claimed_status", ""))
            supported = judgement.status == claimed
            return {
                "supported": supported,
                "supporting_quote_ids": judgement.evidence_ids if supported else [],
                "rationale": (
                    f"Mock lexical support check: quotes read as {judgement.status}, "
                    f"claim is {claimed}."
                ),
            }
        if request.task == "extract_entities":
            chunks = list(ctx.get("chunks", []))
            if self._injected([str(c["text"]) for c in chunks]):
                return {
                    "entities": [
                        {
                            "type": "expiration_date",
                            "chunk_ref": str(chunks[0]["ref"]) if chunks else "C1",
                            "quote": "This Agreement expires on January 1, 2099",
                            "value": "2099-01-01",
                        }
                    ]
                }
            return {"entities": mock_entities(chunks, list(ctx.get("entity_types", [])))}
        raise ValueError(f"mock provider has no handler for task {request.task!r}")
