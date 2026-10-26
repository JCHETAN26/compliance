"""Anthropic, OpenAI and Ollama providers over plain HTTP (no SDKs).

Each ``complete`` call is exactly one HTTP request with a bounded timeout; retries and
budget accounting are the caller's job (every attempt must be reserved first). Errors
say whether the request may have reached the provider (``request_sent``) and carry the
reported usage when a billed response was unusable.

Credentials are passed in by the factory, sent only as request headers, and never put in
exception messages, logs or return values. Document text appears only in request bodies.
"""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from compliance_llm.types import (
    LLMInvalidOutputError,
    LLMRejectedError,
    LLMRequest,
    LLMResponse,
    LLMTransientError,
    LLMUsage,
    Price,
)

# Rate limiting is answered before any processing, so nothing is billed.
_RATE_LIMITED = {429}
# Possibly processed (and billed) before failing: count as unknown outcome.
_SERVER_ERRORS = {408, 500, 502, 503, 504, 529}


def _parse_object(text: str, usage: LLMUsage | None) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except (TypeError, ValueError):
        raise LLMInvalidOutputError(
            "the provider's answer is not valid JSON", request_sent=True, usage=usage
        ) from None
    if not isinstance(value, dict):
        raise LLMInvalidOutputError(
            "the provider's answer is not a JSON object", request_sent=True, usage=usage
        )
    return value


class _HttpProvider:
    name = "http"
    is_mock = False

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        price: Price,
        timeout: float,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.model = model
        self.price = price
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout, connect=min(timeout, 10.0)),
            transport=transport,
            follow_redirects=False,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _post(
        self, path: str, headers: dict[str, str], body: dict[str, Any]
    ) -> tuple[dict[str, Any], int]:
        started = time.perf_counter()
        try:
            response = await self._client.post(path, headers=headers, json=body)
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout) as exc:
            raise LLMTransientError(
                f"could not connect to {self.name}: {type(exc).__name__}", request_sent=False
            ) from None
        except httpx.TimeoutException as exc:
            raise LLMTransientError(
                f"{self.name} request timed out: {type(exc).__name__}",
                request_sent=True,
                code="llm_timeout",
            ) from None
        except httpx.TransportError as exc:
            raise LLMTransientError(
                f"{self.name} transport error: {type(exc).__name__}", request_sent=True
            ) from None
        latency_ms = round((time.perf_counter() - started) * 1000)
        status = response.status_code
        if status in _RATE_LIMITED:
            raise LLMTransientError(
                f"{self.name} rate limited the request (HTTP {status})",
                request_sent=False,
                code="llm_rate_limited",
            )
        if status in _SERVER_ERRORS or status >= 500:
            raise LLMTransientError(
                f"{self.name} failed with HTTP {status}", request_sent=True, code="llm_server_error"
            )
        if 400 <= status < 500:
            raise LLMRejectedError(
                f"{self.name} rejected the request with HTTP {status} ({_error_type(response)})",
                request_sent=False,
                code=f"llm_http_{status}",
            )
        if status != 200:
            raise LLMInvalidOutputError(
                f"{self.name} answered with unexpected HTTP {status}", request_sent=True
            )
        try:
            payload = response.json()
        except ValueError:
            raise LLMInvalidOutputError(
                f"{self.name} returned a non-JSON body", request_sent=True
            ) from None
        if not isinstance(payload, dict):
            raise LLMInvalidOutputError(
                f"{self.name} returned an unexpected body", request_sent=True
            )
        return payload, latency_ms


def _error_type(response: httpx.Response) -> str:
    """The provider's error *type* only (bodies may echo request content)."""
    try:
        body = response.json()
    except ValueError:
        return "no error body"
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict):
        kind = error.get("type") or error.get("code")
        return str(kind)[:80] if kind else "error"
    if isinstance(error, str):
        return "error"
    return "error"


def _count(value: object) -> int | None:
    """A trustworthy token count: a non-negative int (booleans are rejected)."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _usage(
    raw: object,
    input_key: str,
    output_key: str,
    *,
    cache_write_key: str | None = None,
    cache_read_key: str | None = None,
) -> LLMUsage | None:
    """Parse provider usage strictly. Missing or malformed figures give None, so the
    caller keeps the attempt's full reservation instead of counting it as free."""
    if not isinstance(raw, dict):
        return None
    inp, out = _count(raw.get(input_key)), _count(raw.get(output_key))
    if inp is None or out is None:
        return None
    extra: dict[str, int] = {}
    for name, key in (
        ("cache_write_tokens", cache_write_key),
        ("cache_read_tokens", cache_read_key),
    ):
        if key is None or key not in raw or raw[key] is None:
            continue
        value = _count(raw[key])
        if value is None:
            return None
        extra[name] = value
    details = raw.get("cache_creation")
    if isinstance(details, dict) and (_count(details.get("ephemeral_1h_input_tokens")) or 0) > 0:
        return None  # 1-hour cache writes have a different price than the table knows
    return LLMUsage(input_tokens=inp, output_tokens=out, source="provider", **extra)


class AnthropicProvider(_HttpProvider):
    """Messages API with JSON-schema structured output (``output_config.format``), the
    same request shape the Phase 3 Haiku baseline used successfully."""

    name = "anthropic"

    def __init__(self, *, api_key: str, effort: str | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._api_key = api_key
        self._effort = effort

    async def complete(self, request: LLMRequest) -> LLMResponse:
        output_config: dict[str, Any] = {
            "format": {"type": "json_schema", "schema": dict(request.schema)}
        }
        if self._effort:
            output_config["effort"] = self._effort
        body = {
            "model": self.model,
            "max_tokens": request.max_output_tokens,
            "system": request.system,
            "messages": [{"role": "user", "content": request.user}],
            "output_config": output_config,
        }
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload, latency_ms = await self._post("/v1/messages", headers, body)
        usage = _usage(
            payload.get("usage"),
            "input_tokens",
            "output_tokens",
            cache_write_key="cache_creation_input_tokens",
            cache_read_key="cache_read_input_tokens",
        )
        stop = payload.get("stop_reason")
        if stop in ("max_tokens", "refusal"):
            raise LLMInvalidOutputError(
                f"anthropic stopped with {stop!r}", request_sent=True, usage=usage
            )
        content = payload.get("content") or []
        text = "".join(
            str(block.get("text", ""))
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
        return LLMResponse(
            data=_parse_object(text, usage),
            usage=usage,
            model=str(payload.get("model") or self.model),
            latency_ms=latency_ms,
            stop_reason=str(stop) if stop else None,
        )


class OpenAIProvider(_HttpProvider):
    """Chat Completions with ``response_format`` strict JSON schema."""

    name = "openai"

    def __init__(self, *, api_key: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._api_key = api_key

    async def complete(self, request: LLMRequest) -> LLMResponse:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.user},
            ],
            "max_completion_tokens": request.max_output_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": request.schema_name,
                    "schema": dict(request.schema),
                    "strict": True,
                },
            },
        }
        headers = {"authorization": f"Bearer {self._api_key}", "content-type": "application/json"}
        payload, latency_ms = await self._post("/v1/chat/completions", headers, body)
        # Cached prompt tokens are cheaper on OpenAI; counting them at the full input
        # price over-states cost, which is the safe direction.
        usage = _usage(payload.get("usage"), "prompt_tokens", "completion_tokens")
        choices = payload.get("choices") or []
        first = choices[0] if choices and isinstance(choices[0], dict) else {}
        message = first.get("message") if isinstance(first.get("message"), dict) else {}
        assert isinstance(message, dict)  # noqa: S101
        finish = first.get("finish_reason")
        if message.get("refusal") or finish == "length":
            raise LLMInvalidOutputError(
                f"openai finished with {finish or 'a refusal'}", request_sent=True, usage=usage
            )
        return LLMResponse(
            data=_parse_object(str(message.get("content") or ""), usage),
            usage=usage,
            model=str(payload.get("model") or self.model),
            latency_ms=latency_ms,
            stop_reason=str(finish) if finish else None,
        )


class OllamaProvider(_HttpProvider):
    """Local Ollama ``/api/chat`` with a JSON-schema ``format`` (no credentials)."""

    name = "ollama"

    async def complete(self, request: LLMRequest) -> LLMResponse:
        body = {
            "model": self.model,
            "stream": False,
            "format": dict(request.schema),
            "options": {"temperature": 0, "num_predict": request.max_output_tokens},
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.user},
            ],
        }
        payload, latency_ms = await self._post(
            "/api/chat", {"content-type": "application/json"}, body
        )
        usage = _usage(payload, "prompt_eval_count", "eval_count")
        message = payload.get("message") if isinstance(payload.get("message"), dict) else {}
        assert isinstance(message, dict)  # noqa: S101
        if payload.get("done_reason") == "length":
            raise LLMInvalidOutputError(
                "ollama stopped at the token limit", request_sent=True, usage=usage
            )
        return LLMResponse(
            data=_parse_object(str(message.get("content") or ""), usage),
            usage=usage,
            model=str(payload.get("model") or self.model),
            latency_ms=latency_ms,
            stop_reason=str(payload.get("done_reason") or "") or None,
        )
