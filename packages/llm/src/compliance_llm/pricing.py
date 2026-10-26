"""Token prices and conservative token estimates for budget reservations."""

from __future__ import annotations

import json
from decimal import ROUND_CEILING, Decimal

from compliance_core.money import token_cost_micro
from compliance_llm.types import LLMConfigurationError, LLMRequest, LLMUsage, Price

# USD per million tokens, standard routing, from
# https://platform.claude.com/docs/en/about-claude/pricing (retrieved 2026-10-21; the same
# table as ml/baselines/anthropic_pricing.yaml). Re-check before real runs and override with
# LLM_INPUT_USD_PER_MTOK / LLM_OUTPUT_USD_PER_MTOK if prices change.
ANTHROPIC_PRICES: dict[str, tuple[float, float, float, float]] = {
    # input, output, cache write (5 min), cache read
    "claude-haiku-5-5": (0.10, 0.50, 0.125, 0.01),
    "claude-sonnet-5-5": (2.00, 10.00, 2.50, 0.10),
}
ANTHROPIC_PRICE_SOURCE = "built-in table (platform.claude.com pricing, retrieved 2026-10-21)"

# Reservation bound, the same rule the Phase 3 budget runner was verified with: every byte
# of the UTF-8 JSON request body counts as one input token (a token covers at least one
# byte of text), plus a fixed allowance for hidden structured-output/framing tokens,
# priced at the highest input-side rate, plus the full output allowance.
OVERHEAD_TOKENS = 4096
# Requests whose input bound exceeds this are refused (prices above hold up to 100k tokens).
MAX_INPUT_TOKEN_BOUND = 100_000


class RequestTooLargeError(LLMConfigurationError):
    """The request's provable input bound exceeds MAX_INPUT_TOKEN_BOUND."""


def request_body_bytes(request: LLMRequest, model: str) -> int:
    body = {
        "model": model,
        "max_tokens": request.max_output_tokens,
        "system": request.system,
        "user": request.user,
        "schema": request.schema,
        "schema_name": request.schema_name,
    }
    return len(json.dumps(body, ensure_ascii=False, sort_keys=True).encode("utf-8"))


def input_token_bound(request: LLMRequest, model: str) -> int:
    bound = request_body_bytes(request, model) + OVERHEAD_TOKENS
    if bound > MAX_INPUT_TOKEN_BOUND:
        raise RequestTooLargeError(
            f"request input bound {bound} tokens exceeds {MAX_INPUT_TOKEN_BOUND}"
        )
    return bound


def reservation(request: LLMRequest, model: str, price: Price) -> tuple[int, int]:
    """(tokens, micro-USD) that bound everything one attempt can consume."""
    bound = input_token_bound(request, model)
    tokens = bound + request.max_output_tokens
    micro = token_cost_micro(
        bound, request.max_output_tokens, price.top_input, price.output_per_mtok
    )
    return tokens, micro


def usage_cost_micro(usage: LLMUsage, price: Price) -> int:
    """Exact cost of reported usage in micro-USD, rounded up."""
    exact = (
        usage.input_tokens * Decimal(str(price.input_per_mtok))
        + usage.output_tokens * Decimal(str(price.output_per_mtok))
        + usage.cache_write_tokens * Decimal(str(price.cache_write))
        + usage.cache_read_tokens * Decimal(str(price.cache_read))
    )
    return int(exact.to_integral_value(rounding=ROUND_CEILING))


def estimate_input_tokens(request: LLMRequest) -> int:
    """Upper bound on input tokens (body bytes + overhead); see :func:`input_token_bound`."""
    return input_token_bound(request, "")


def resolve_price(
    provider: str,
    model: str,
    input_override: float | None,
    output_override: float | None,
) -> Price:
    if (input_override is None) != (output_override is None):
        raise LLMConfigurationError(
            "Set both LLM_INPUT_USD_PER_MTOK and LLM_OUTPUT_USD_PER_MTOK, or neither."
        )
    if provider == "anthropic" and model in ANTHROPIC_PRICES:
        inp, out, cache_write, cache_read = ANTHROPIC_PRICES[model]
        if input_override is not None and output_override is not None:
            # Overrides apply to input/output; cache rates keep their published ratios.
            scale = input_override / inp if inp else 1.0
            return Price(
                input_override,
                output_override,
                "environment override",
                cache_write * scale,
                cache_read * scale,
            )
        return Price(inp, out, ANTHROPIC_PRICE_SOURCE, cache_write, cache_read)
    if input_override is not None and output_override is not None:
        return Price(input_override, output_override, "environment override")
    if provider == "mock":
        return Price(0.0, 0.0, "mock provider (no cost)")
    if provider == "ollama":
        return Price(0.0, 0.0, "local Ollama model (no per-token cost)")
    raise LLMConfigurationError(
        f"No price is known for {provider} model {model!r}; set LLM_INPUT_USD_PER_MTOK and "
        "LLM_OUTPUT_USD_PER_MTOK so the run budget can be enforced."
    )
