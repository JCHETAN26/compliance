"""Strict JSON schemas sent to providers, and the Pydantic models that validate answers.

Schemas list every property as required and forbid additional properties, which is what
OpenAI strict mode requires and what Anthropic/Ollama accept. Validation is repeated on
our side for every provider: an answer that does not validate is an invalid output.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

ENTITY_TYPES = ("party", "effective_date", "expiration_date", "term", "governing_law")


def _obj(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(properties),
        "properties": properties,
    }


RULE_VERDICT_SCHEMA = _obj(
    {
        "verdict": {"type": "string", "enum": ["satisfied", "violated", "not_mentioned"]},
        "evidence_span_ids": {"type": "array", "items": {"type": "string"}},
        "rationale": {"type": "string"},
    }
)
SUPPORT_SCHEMA = _obj(
    {
        "supported": {"type": "boolean"},
        "supporting_quote_ids": {"type": "array", "items": {"type": "string"}},
        "rationale": {"type": "string"},
    }
)
EXTRACT_SCHEMA = _obj(
    {
        "entities": {
            "type": "array",
            "items": _obj(
                {
                    "type": {"type": "string", "enum": list(ENTITY_TYPES)},
                    "chunk_ref": {"type": "string"},
                    "quote": {"type": "string"},
                    "value": {"type": "string"},
                }
            ),
        }
    }
)

Short = Annotated[str, StringConstraints(max_length=2000)]
Ref = Annotated[str, StringConstraints(min_length=1, max_length=16)]


class _Out(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RuleVerdictOut(_Out):
    verdict: Literal["satisfied", "violated", "not_mentioned"]
    evidence_span_ids: Annotated[list[Ref], Field(max_length=30)]
    rationale: Short


class SupportOut(_Out):
    supported: bool
    supporting_quote_ids: Annotated[list[Ref], Field(max_length=30)]
    rationale: Short


class EntityOut(_Out):
    type: Literal["party", "effective_date", "expiration_date", "term", "governing_law"]
    chunk_ref: Ref
    quote: Annotated[str, StringConstraints(min_length=1, max_length=1000)]
    value: Annotated[str, StringConstraints(min_length=1, max_length=300)]


class ExtractOut(_Out):
    entities: Annotated[list[EntityOut], Field(max_length=60)]
