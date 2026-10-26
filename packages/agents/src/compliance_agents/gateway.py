"""Budgeted, memoized, retried LLM calls.

For every step (deterministic ``step_key``):
1. a memoized result is returned without any provider request (resume never re-bills);
2. otherwise each attempt is reserved in the ledger first with a provable bound (UTF-8
   body bytes + 4096 framing tokens at the highest input-side rate, plus the full output
   allowance, rounded up), then sent;
3. the attempt is settled with the provider's measured usage; measured usage above the
   reservation stops the run. A request that may have been processed without (trustworthy)
   reported usage keeps its full reservation;
   one that certainly was not processed is ``released``. Cancellation during a request
   records ``unknown`` before propagating;
4. answers are validated against the step's Pydantic model; invalid answers are billed
   (``failed``) and retried up to ``max_attempts``.
"""

from __future__ import annotations

import asyncio
import random
import uuid
from dataclasses import dataclass
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from compliance_agents.store import RunStore
from compliance_llm import (
    LLMError,
    LLMProvider,
    LLMRequest,
    LLMUsage,
    RequestTooLargeError,
    reservation,
    usage_cost_micro,
)

T = TypeVar("T", bound=BaseModel)


class LLMStepError(Exception):
    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.message = message
        self.code = code


class BudgetInvariantError(Exception):
    """Measured usage exceeded the attempt's reservation: the bound is wrong, so the run
    stops instead of sending more requests (the Phase 3 runner's rule)."""


@dataclass(frozen=True)
class LLMMeta:
    cached: bool
    attempts: int
    model: str
    input_tokens: int
    output_tokens: int
    cost_microusd: int
    usage_source: str


class LLMGateway:
    def __init__(
        self,
        store: RunStore,
        provider: LLMProvider,
        *,
        max_attempts: int,
        backoff_seconds: float = 0.5,
    ) -> None:
        self.store = store
        self.provider = provider
        self.max_attempts = max_attempts
        self.backoff_seconds = backoff_seconds

    async def _settle_measured(
        self,
        call_id: uuid.UUID,
        usage: LLMUsage,
        *,
        status: str,
        reserved: tuple[int, int],
        latency_ms: int | None = None,
        error_code: str | None = None,
        memo: tuple[str, str, dict[str, Any]] | None = None,
        agent: str,
        context: dict[str, Any],
    ) -> int:
        cost = usage_cost_micro(usage, self.provider.price)
        await self.store.settle(
            call_id,
            status=status,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_write_tokens=usage.cache_write_tokens,
            cache_read_tokens=usage.cache_read_tokens,
            cost_microusd=cost,
            usage_source=usage.source,
            latency_ms=latency_ms,
            error_code=error_code,
            memo=memo,
        )
        if usage.total > reserved[0] or cost > reserved[1]:
            await self.store.emit(
                agent,
                "error",
                {
                    **context,
                    "code": "budget_invariant",
                    "message": "measured usage exceeded the reservation",
                    "reserved_tokens": reserved[0],
                    "measured_tokens": usage.total,
                },
            )
            raise BudgetInvariantError(
                f"measured usage ({usage.total} tokens, {cost} micro-USD) exceeded the "
                f"reservation ({reserved[0]} tokens, {reserved[1]} micro-USD)"
            )
        return cost

    async def run(
        self,
        *,
        step_key: str,
        agent: str,
        prompt_version: str,
        request: LLMRequest,
        output_model: type[T],
        event_context: dict[str, Any] | None = None,
    ) -> tuple[T, LLMMeta]:
        memo = await self.store.memo_get(step_key)
        if memo is not None:
            meta = memo.get("meta", {})
            return output_model.model_validate(memo["data"]), LLMMeta(
                cached=True,
                attempts=0,
                model=str(meta.get("model", self.provider.model)),
                input_tokens=0,
                output_tokens=0,
                cost_microusd=0,
                usage_source="none",
            )
        context = dict(event_context or {})
        try:
            reserve_tokens, reserve_micro = reservation(
                request, self.provider.model, self.provider.price
            )
        except RequestTooLargeError as exc:
            raise LLMStepError(str(exc), "request_too_large") from None
        reserved = (reserve_tokens, reserve_micro)
        last_code, last_message = "llm_failed", "no attempt was made"
        for _ in range(self.max_attempts):
            call_id, attempt = await self.store.reserve(
                step_key=step_key,
                agent=agent,
                provider=self.provider.name,
                model=self.provider.model,
                is_mock=self.provider.is_mock,
                prompt_version=prompt_version,
                tokens=reserve_tokens,
                microusd=reserve_micro,
            )
            await self.store.emit(
                agent,
                "tool_call",
                {
                    **context,
                    "tool": f"llm.{request.task}",
                    "provider": self.provider.name,
                    "model": self.provider.model,
                    "mock": self.provider.is_mock,
                    "attempt": attempt,
                    "reserved_tokens": reserve_tokens,
                    "reserved_usd": reserve_micro / 1e6,
                },
            )
            try:
                response = await self.provider.complete(request)
            except asyncio.CancelledError:
                await asyncio.shield(
                    self.store.settle(
                        call_id, status="unknown", error_code="cancelled", fenced=False
                    )
                )
                raise
            except LLMError as exc:
                if exc.usage is not None:
                    await self._settle_measured(
                        call_id,
                        exc.usage,
                        status="failed",
                        reserved=reserved,
                        error_code=exc.code,
                        agent=agent,
                        context=context,
                    )
                else:
                    await self.store.settle(
                        call_id,
                        status="unknown" if exc.request_sent else "released",
                        error_code=exc.code,
                    )
                last_code, last_message = exc.code, exc.message
                await self.store.emit(
                    agent,
                    "error",
                    {
                        **context,
                        "tool": f"llm.{request.task}",
                        "attempt": attempt,
                        "code": exc.code,
                        "message": exc.message,
                        "retryable": exc.retryable,
                    },
                )
                if not exc.retryable:
                    break
                await self._backoff(attempt)
                continue
            except Exception:
                # An unexpected failure inside the provider: it may have sent the request.
                await asyncio.shield(
                    self.store.settle(
                        call_id, status="unknown", error_code="provider_exception", fenced=False
                    )
                )
                raise
            usage = response.usage
            try:
                parsed = output_model.model_validate(response.data)
            except ValidationError as exc:
                last_code = "llm_invalid_output"
                last_message = f"answer failed schema validation ({exc.error_count()} errors)"
                if usage is not None:
                    await self._settle_measured(
                        call_id,
                        usage,
                        status="failed",
                        reserved=reserved,
                        latency_ms=response.latency_ms,
                        error_code=last_code,
                        agent=agent,
                        context=context,
                    )
                else:
                    await self.store.settle(
                        call_id,
                        status="failed",
                        usage_source="none",
                        latency_ms=response.latency_ms,
                        error_code=last_code,
                    )
                await self.store.emit(
                    agent,
                    "error",
                    {
                        **context,
                        "tool": f"llm.{request.task}",
                        "attempt": attempt,
                        "code": last_code,
                        "message": last_message,
                        "retryable": True,
                    },
                )
                await self._backoff(attempt)
                continue
            memo_value = (
                step_key,
                agent,
                {
                    "data": parsed.model_dump(mode="json"),
                    "meta": {"model": response.model, "prompt_version": prompt_version},
                },
            )
            if usage is None:
                # Answered without trustworthy usage: keep the full reservation charged.
                await self.store.settle(
                    call_id,
                    status="succeeded",
                    usage_source="none",
                    latency_ms=response.latency_ms,
                    error_code="usage_missing",
                    memo=memo_value,
                )
                cost, source, in_tok, out_tok = reserve_micro, "none", 0, 0
            else:
                cost = await self._settle_measured(
                    call_id,
                    usage,
                    status="succeeded",
                    reserved=reserved,
                    latency_ms=response.latency_ms,
                    memo=memo_value,
                    agent=agent,
                    context=context,
                )
                source, in_tok, out_tok = usage.source, usage.input_tokens, usage.output_tokens
            await self.store.emit(
                agent,
                "tokens",
                {
                    **context,
                    "tool": f"llm.{request.task}",
                    "attempt": attempt,
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "usage_source": source,
                    "cost_usd": cost / 1e6,
                    "latency_ms": response.latency_ms,
                },
            )
            return parsed, LLMMeta(
                cached=False,
                attempts=attempt,
                model=response.model,
                input_tokens=in_tok,
                output_tokens=out_tok,
                cost_microusd=cost,
                usage_source=source,
            )
        raise LLMStepError(last_message, last_code)

    async def _backoff(self, attempt: int) -> None:
        delay = self.backoff_seconds * (2 ** (attempt - 1))
        await asyncio.sleep(delay + random.uniform(0, delay / 2))  # noqa: S311 - jitter
