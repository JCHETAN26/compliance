"""Provider adapters over plain HTTP (no network: httpx.MockTransport), the offline mock,
pricing and exact money arithmetic."""

from __future__ import annotations

import json
import os
import socket
from collections.abc import Callable
from decimal import Decimal
from typing import Any

import httpx
import pytest

from compliance_core.config import AppEnv, Settings
from compliance_core.money import token_cost_micro, usd_to_micro_ceil
from compliance_llm import (
    MOCK_MODEL,
    LLMConfigurationError,
    LLMInvalidOutputError,
    LLMRejectedError,
    LLMRequest,
    LLMTransientError,
    MockProvider,
    RequestTooLargeError,
    build_provider,
    estimate_input_tokens,
    reservation,
    resolve_choice,
    usage_cost_micro,
)
from compliance_llm.http_providers import AnthropicProvider, OllamaProvider, OpenAIProvider
from compliance_llm.pricing import OVERHEAD_TOKENS, request_body_bytes
from compliance_llm.types import Price

SECRET = "sk-test-SECRET-value-0123456789"
SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["verdict"],
    "properties": {"verdict": {"type": "string"}},
}
REQUEST = LLMRequest(
    task="rule_verdict",
    system="system text",
    user="user text",
    schema_name="rule_verdict",
    schema=SCHEMA,
    max_output_tokens=100,
    context={"hypothesis": "h", "spans": []},
)
PRICE = Price(0.10, 0.50, "test")


def _transport(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


def _anthropic(handler: Callable[[httpx.Request], httpx.Response]) -> AnthropicProvider:
    return AnthropicProvider(
        api_key=SECRET,
        model="claude-haiku-5-5",
        base_url="https://anthropic.test",
        price=PRICE,
        timeout=5,
        transport=_transport(handler),
    )


async def test_anthropic_request_shape_and_usage() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["headers"] = dict(request.headers)
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "msg_1",
                "model": "claude-haiku-5-5",
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": '{"verdict": "satisfied"}'}],
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "cache_creation_input_tokens": 2,
                    "cache_read_input_tokens": 3,
                },
            },
        )

    provider = _anthropic(handler)
    response = await provider.complete(REQUEST)
    await provider.aclose()
    assert seen["url"] == "https://anthropic.test/v1/messages"
    assert seen["headers"]["x-api-key"] == SECRET
    assert seen["headers"]["anthropic-version"] == "2023-06-01"
    body = seen["body"]
    assert body["output_config"]["format"] == {"type": "json_schema", "schema": SCHEMA}
    assert body["system"] == "system text"
    assert body["messages"] == [{"role": "user", "content": "user text"}]
    assert "tools" not in body  # the model is never given tools
    assert response.data == {"verdict": "satisfied"}
    usage = response.usage
    assert usage is not None
    assert (usage.input_tokens, usage.output_tokens) == (10, 5)
    assert (usage.cache_write_tokens, usage.cache_read_tokens) == (2, 3)
    assert usage.source == "provider"


@pytest.mark.parametrize(
    ("status", "error", "sent"),
    [
        (429, LLMTransientError, False),
        (500, LLMTransientError, True),
        (529, LLMTransientError, True),
        (400, LLMRejectedError, False),
        (401, LLMRejectedError, False),
    ],
)
async def test_http_errors_are_classified_without_leaking_the_key(
    status: int, error: type[Exception], sent: bool
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status, json={"type": "error", "error": {"type": "x", "message": f"echo {SECRET}"}}
        )

    provider = _anthropic(handler)
    with pytest.raises(error) as info:
        await provider.complete(REQUEST)
    assert info.value.request_sent is sent  # type: ignore[attr-defined]
    assert SECRET not in str(info.value)
    await provider.aclose()


async def test_transport_failures_distinguish_sent_from_not_sent() -> None:
    def connect_fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    def read_timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    for handler, sent in ((connect_fail, False), (read_timeout, True)):
        provider = _anthropic(handler)
        with pytest.raises(LLMTransientError) as info:
            await provider.complete(REQUEST)
        assert info.value.request_sent is sent
        await provider.aclose()


async def test_truncated_or_non_json_answers_are_invalid_but_billed() -> None:
    def truncated(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "stop_reason": "max_tokens",
                "content": [{"type": "text", "text": '{"verd'}],
                "usage": {"input_tokens": 7, "output_tokens": 100},
            },
        )

    def prose(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "I think it is satisfied."}],
                "usage": {"input_tokens": 7, "output_tokens": 9},
            },
        )

    for handler, out in ((truncated, 100), (prose, 9)):
        provider = _anthropic(handler)
        with pytest.raises(LLMInvalidOutputError) as info:
            await provider.complete(REQUEST)
        assert info.value.usage is not None
        assert info.value.usage.output_tokens == out
        await provider.aclose()


async def test_openai_and_ollama_request_shapes() -> None:
    bodies: list[dict[str, Any]] = []

    def openai(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        assert request.headers["authorization"] == f"Bearer {SECRET}"
        return httpx.Response(
            200,
            json={
                "model": "gpt-test",
                "choices": [
                    {"finish_reason": "stop", "message": {"content": '{"verdict": "violated"}'}}
                ],
                "usage": {"prompt_tokens": 11, "completion_tokens": 4},
            },
        )

    def ollama(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        assert "authorization" not in request.headers
        return httpx.Response(
            200,
            json={
                "model": "llama-test",
                "message": {"content": '{"verdict": "not_mentioned"}'},
                "prompt_eval_count": 12,
                "eval_count": 3,
                "done_reason": "stop",
            },
        )

    oa = OpenAIProvider(
        api_key=SECRET,
        model="gpt-test",
        base_url="https://oa.test",
        price=PRICE,
        timeout=5,
        transport=_transport(openai),
    )
    ol = OllamaProvider(
        model="llama-test",
        base_url="http://ollama.test",
        price=PRICE,
        timeout=5,
        transport=_transport(ollama),
    )
    r1 = await oa.complete(REQUEST)
    r2 = await ol.complete(REQUEST)
    await oa.aclose()
    await ol.aclose()
    assert bodies[0]["response_format"]["json_schema"] == {
        "name": "rule_verdict",
        "schema": SCHEMA,
        "strict": True,
    }
    assert bodies[1]["format"] == SCHEMA
    assert bodies[1]["stream"] is False
    assert r1.data["verdict"] == "violated"
    assert r1.usage is not None
    assert r1.usage.input_tokens == 11
    assert r2.data["verdict"] == "not_mentioned"
    assert r2.usage is not None
    assert r2.usage.output_tokens == 3


class _GuardedEnviron(dict[str, str]):
    """Records every key read (the mock provider must not read credentials)."""

    def __init__(self, base: dict[str, str]) -> None:
        super().__init__(base)
        self.reads: list[str] = []

    def get(self, key: str, default: Any = None) -> Any:
        self.reads.append(key)
        return super().get(key, default)

    def __getitem__(self, key: str) -> str:
        self.reads.append(key)
        return super().__getitem__(key)

    def __contains__(self, key: object) -> bool:
        self.reads.append(str(key))
        return super().__contains__(key)


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    return


async def test_plain_mock_reads_no_key_and_opens_no_connection(
    monkeypatch: pytest.MonkeyPatch, no_network: None
) -> None:
    guarded = _GuardedEnviron({**os.environ, "ANTHROPIC_API_KEY": SECRET, "OPENAI_API_KEY": SECRET})
    monkeypatch.setattr("compliance_llm.factory.os.environ", guarded)
    settings = Settings(app_env=AppEnv.TEST, llm_provider="anthropic")
    choice = resolve_choice(settings, "mock")
    provider = build_provider(settings, choice)
    assert isinstance(provider, MockProvider)
    assert provider.is_mock
    response = await provider.complete(
        LLMRequest(
            task="rule_verdict",
            system="s",
            user="u",
            schema_name="rule_verdict",
            schema=SCHEMA,
            max_output_tokens=50,
            context={
                "hypothesis": "Receiving Party shall return Confidential Information.",
                "spans": [
                    {
                        "id": "S1",
                        "text": "The Receiving Party shall return all Confidential Information.",
                    }
                ],
            },
        )
    )
    assert response.data["verdict"] == "satisfied"
    assert response.data["evidence_span_ids"] == ["S1"]
    assert response.usage is not None
    assert response.usage.source == "mock_estimate"
    assert response.model == MOCK_MODEL
    assert not [k for k in guarded.reads if k.endswith("_API_KEY")]


def test_provider_choice_validation() -> None:
    mock_only = Settings(app_env=AppEnv.TEST)
    with pytest.raises(LLMConfigurationError, match="not configured"):
        resolve_choice(mock_only, "anthropic")
    with pytest.raises(LLMConfigurationError, match="only offers"):
        resolve_choice(mock_only, "mock", "gpt-4")
    anthropic = Settings(app_env=AppEnv.TEST, llm_provider="anthropic")
    choice = resolve_choice(anthropic, "anthropic")
    assert choice.model == "claude-haiku-5-5"
    assert choice.price.input_per_mtok == 0.10
    with pytest.raises(LLMConfigurationError, match="No price"):
        resolve_choice(anthropic, "anthropic", "claude-unknown")
    openai = Settings(app_env=AppEnv.TEST, llm_provider="openai", llm_model="gpt-x")
    with pytest.raises(LLMConfigurationError, match="No price"):
        resolve_choice(openai, "openai")
    priced = Settings(
        app_env=AppEnv.TEST,
        llm_provider="openai",
        llm_model="gpt-x",
        llm_input_usd_per_mtok=1.0,
        llm_output_usd_per_mtok=2.0,
    )
    assert resolve_choice(priced, "openai").price.output_per_mtok == 2.0
    half = Settings(app_env=AppEnv.TEST, llm_provider="openai", llm_input_usd_per_mtok=1.0)
    with pytest.raises(LLMConfigurationError, match="both"):
        resolve_choice(half, "openai", "gpt-x")


def test_real_provider_without_key_is_a_configuration_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    settings = Settings(app_env=AppEnv.TEST, llm_provider="anthropic")
    with pytest.raises(LLMConfigurationError, match="ANTHROPIC_API_KEY"):
        build_provider(settings, resolve_choice(settings, "anthropic"))


def _usage_response(usage: Any) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": '{"verdict": "satisfied"}'}],
                "usage": usage,
            },
        )

    return handler


@pytest.mark.parametrize(
    "usage",
    [
        None,
        {},
        {"input_tokens": 10},
        {"input_tokens": True, "output_tokens": 5},
        {"input_tokens": -1, "output_tokens": 5},
        {"input_tokens": 10, "output_tokens": "5"},
        {"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": -3},
        {
            "input_tokens": 10,
            "output_tokens": 5,
            "cache_creation": {"ephemeral_1h_input_tokens": 7},
        },
    ],
)
async def test_untrustworthy_usage_is_reported_as_missing(usage: Any) -> None:
    provider = _anthropic(_usage_response(usage))
    response = await provider.complete(REQUEST)
    await provider.aclose()
    assert response.data == {"verdict": "satisfied"}
    assert response.usage is None  # the caller keeps the full reservation


async def test_cache_components_are_priced_at_cache_rates() -> None:
    provider = _anthropic(
        _usage_response(
            {
                "input_tokens": 1_000,
                "output_tokens": 2_000,
                "cache_creation_input_tokens": 3_000,
                "cache_read_input_tokens": 4_000,
            }
        )
    )
    response = await provider.complete(REQUEST)
    await provider.aclose()
    usage = response.usage
    assert usage is not None
    assert (usage.input_tokens, usage.cache_write_tokens, usage.cache_read_tokens) == (
        1000,
        3000,
        4000,
    )
    price = resolve_choice(
        Settings(app_env=AppEnv.TEST, llm_provider="anthropic"), "anthropic"
    ).price
    # 1000*0.10 + 2000*0.50 + 3000*0.125 + 4000*0.01 = 1515 micro-USD
    assert usage_cost_micro(usage, price) == 1515


def test_reservation_is_a_provable_upper_bound() -> None:
    price = Price(0.10, 0.50, "t", 0.125, 0.01)
    unicode_text = "\u00e9\u4e2d\U0001f600 " * 2000  # multi-byte characters
    request = LLMRequest(
        task="t",
        system="s",
        user=unicode_text,
        schema_name="s",
        schema=SCHEMA,
        max_output_tokens=777,
    )
    tokens, micro = reservation(request, "claude-haiku-5-5", price)
    body = request_body_bytes(request, "claude-haiku-5-5")
    assert body >= len(unicode_text.encode("utf-8"))
    assert tokens == body + OVERHEAD_TOKENS + 777
    # input priced at the highest input-side rate (cache write), output at output rate
    assert micro == token_cost_micro(body + OVERHEAD_TOKENS, 777, 0.125, 0.50)
    big_schema = {**SCHEMA, "description": "x" * 50_000}
    bigger = LLMRequest(
        task="t", system="s", user="u", schema_name="s", schema=big_schema, max_output_tokens=10
    )
    assert reservation(bigger, "m", price)[0] > 50_000
    huge = LLMRequest(
        task="t", system="s", user="x" * 200_000, schema_name="s", schema={}, max_output_tokens=10
    )
    with pytest.raises(RequestTooLargeError):
        reservation(huge, "m", price)
    assert estimate_input_tokens(request) >= body


def test_money_rounds_up_and_is_exact() -> None:
    assert usd_to_micro_ceil(1.0) == 1_000_000
    assert usd_to_micro_ceil("0.0000001") == 1  # a tenth of a micro-dollar rounds up
    assert usd_to_micro_ceil(0.1) == 100_000  # no float drift
    # 3 tokens at $0.37/Mtok = 1.11 micro-USD -> 2
    assert token_cost_micro(3, 0, 0.37, 0) == 2
    assert token_cost_micro(1_000_000, 1_000_000, Decimal("0.10"), Decimal("0.50")) == 600_000
    assert token_cost_micro(0, 0, 5, 5) == 0
    with pytest.raises(ValueError, match="invalid"):
        usd_to_micro_ceil(-1)
