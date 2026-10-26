"""Provider-neutral request/response types and errors for structured LLM calls."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

UsageSource = Literal["provider", "mock_estimate"]


@dataclass(frozen=True)
class LLMRequest:
    """One structured-output request.

    ``system`` and ``user`` are the exact texts a real provider receives. ``schema`` is a
    strict JSON schema (every property required, no additional properties) so the same
    schema works for Anthropic, OpenAI strict mode and Ollama. ``context`` carries the
    structured inputs the prompt was rendered from; only the offline mock provider reads
    it (it never parses prompt text).
    """

    task: str
    system: str
    user: str
    schema_name: str
    schema: Mapping[str, Any]
    max_output_tokens: int
    context: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LLMUsage:
    """Provider-reported usage. ``input_tokens`` excludes cache writes/reads, which are
    priced separately (Anthropic reports all three)."""

    input_tokens: int
    output_tokens: int
    source: UsageSource
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0

    @property
    def total(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_write_tokens
            + self.cache_read_tokens
        )


@dataclass(frozen=True)
class LLMResponse:
    """``usage`` is None when the provider answered without trustworthy usage figures;
    the caller must then keep the attempt's full reservation charged."""

    data: dict[str, Any]
    usage: LLMUsage | None
    model: str
    latency_ms: int
    stop_reason: str | None = None


@dataclass(frozen=True)
class Price:
    """USD per million tokens. Cache prices default to the input price (providers other
    than Anthropic report no cache components)."""

    input_per_mtok: float
    output_per_mtok: float
    source: str
    cache_write_per_mtok: float | None = None
    cache_read_per_mtok: float | None = None

    @property
    def cache_write(self) -> float:
        return (
            self.input_per_mtok if self.cache_write_per_mtok is None else self.cache_write_per_mtok
        )

    @property
    def cache_read(self) -> float:
        return self.input_per_mtok if self.cache_read_per_mtok is None else self.cache_read_per_mtok

    @property
    def top_input(self) -> float:
        return max(self.input_per_mtok, self.cache_write, self.cache_read)


class LLMError(Exception):
    """Base error. ``request_sent`` says whether the request may have reached the
    provider (and may therefore be billed); ``usage`` is set when the provider reported
    usage even though the call failed (e.g. a billed response that violates the schema)."""

    code = "llm_error"
    retryable = False

    def __init__(
        self,
        message: str,
        *,
        request_sent: bool,
        usage: LLMUsage | None = None,
        code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.request_sent = request_sent
        self.usage = usage
        if code:
            self.code = code


class LLMTransientError(LLMError):
    """Timeouts, transport failures, overload and server errors: worth retrying."""

    code = "llm_transient"
    retryable = True


class LLMRejectedError(LLMError):
    """The provider rejected the request (4xx other than rate limiting): not retryable."""

    code = "llm_rejected"


class LLMInvalidOutputError(LLMError):
    """The provider answered, but not with JSON that satisfies the schema."""

    code = "llm_invalid_output"
    retryable = True


class LLMConfigurationError(Exception):
    """The selected provider cannot be used (missing key, model or price)."""


class LLMProvider(Protocol):
    name: str
    model: str
    is_mock: bool
    price: Price

    async def complete(self, request: LLMRequest) -> LLMResponse: ...

    async def aclose(self) -> None: ...
