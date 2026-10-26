"""Build the configured provider. Credentials are read from the process environment only
here, only for the provider actually selected, and never stored or returned."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field

import httpx

from compliance_core.config import Settings
from compliance_core.services.errors import InvalidOperationError
from compliance_core.services.reviews import ProviderSelection
from compliance_llm.http_providers import AnthropicProvider, OllamaProvider, OpenAIProvider
from compliance_llm.mock import MOCK_MODEL, MockProvider
from compliance_llm.pricing import resolve_price
from compliance_llm.types import LLMConfigurationError, LLMProvider, Price

PROVIDERS = ("mock", "anthropic", "openai", "ollama")
DEFAULT_MODELS = {"mock": MOCK_MODEL, "anthropic": "claude-haiku-5-5"}
KEY_ENV = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}


@dataclass(frozen=True)
class ProviderChoice:
    """A provider selection without credentials (safe to store and display)."""

    provider: str
    model: str
    is_mock: bool
    price: Price


def default_model(settings: Settings, provider: str) -> str:
    if provider == settings.llm_provider and settings.llm_model:
        return settings.llm_model
    return DEFAULT_MODELS.get(provider, "")


def available_providers(settings: Settings) -> list[str]:
    """``mock`` is always available; a real provider only if it is the deployment's
    configured ``LLM_PROVIDER``."""
    return ["mock"] if settings.llm_provider == "mock" else ["mock", settings.llm_provider]


def resolve_choice(settings: Settings, provider: str, model: str | None = None) -> ProviderChoice:
    """Validate a provider/model choice *without* reading any credential."""
    if provider not in available_providers(settings):
        raise LLMConfigurationError(
            f"Provider {provider!r} is not configured on this server (LLM_PROVIDER)."
        )
    chosen = (model or "").strip() or default_model(settings, provider)
    if not chosen:
        raise LLMConfigurationError(f"Provider {provider!r} needs an explicit model (LLM_MODEL).")
    if provider == "mock" and chosen != MOCK_MODEL:
        raise LLMConfigurationError(f"The mock provider only offers {MOCK_MODEL!r}.")
    overrides = (
        (settings.llm_input_usd_per_mtok, settings.llm_output_usd_per_mtok)
        if provider == settings.llm_provider
        else (None, None)
    )
    price = resolve_price(provider, chosen, *overrides)
    return ProviderChoice(provider=provider, model=chosen, is_mock=provider == "mock", price=price)


def credential_present(provider: str) -> bool:
    """Whether the provider's key variable is set (its value is not read)."""
    env = KEY_ENV.get(provider)
    return env is None or bool(os.environ.get(env))


def configured_providers(declared: str) -> Callable[[str], bool]:
    """A credential check from a non-secret comma-separated list of provider names whose
    keys the *worker* holds (used where the key itself must not be present, e.g. the MCP
    server). Mock and Ollama need no key."""
    names = frozenset(p.strip().lower() for p in declared.split(",") if p.strip())
    return lambda provider: provider not in KEY_ENV or provider in names


@dataclass(frozen=True)
class ProviderResolver:
    """Validates review provider choices for ``compliance_core.services.review_start``.

    ``credential_check`` says whether the provider's credential is available to the worker
    that will execute the run; it never reads or returns the credential itself.
    """

    settings: Settings
    credential_check: Callable[[str], bool] = field(default=credential_present)

    def default_model(self, provider: str) -> str:
        return default_model(self.settings, provider)

    def resolve(self, provider: str, model: str) -> ProviderSelection:
        try:
            choice = resolve_choice(self.settings, provider, model)
        except LLMConfigurationError as exc:
            raise InvalidOperationError(str(exc)) from exc
        if not choice.is_mock and not self.credential_check(choice.provider):
            raise InvalidOperationError(
                f"The {choice.provider} provider has no API key configured on this server."
            )
        return ProviderSelection(
            provider=choice.provider,
            model=choice.model,
            is_mock=choice.is_mock,
            price_source=choice.price.source,
            input_usd_per_mtok=choice.price.input_per_mtok,
            output_usd_per_mtok=choice.price.output_per_mtok,
        )


def build_provider(
    settings: Settings,
    choice: ProviderChoice,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> LLMProvider:
    if choice.provider == "mock":
        return MockProvider(choice.model)
    timeout = settings.llm_timeout_seconds
    if choice.provider == "ollama":
        return OllamaProvider(
            model=choice.model,
            base_url=settings.ollama_base_url,
            price=choice.price,
            timeout=timeout,
            transport=transport,
        )
    env = KEY_ENV[choice.provider]
    key = os.environ.get(env, "")
    if not key:
        raise LLMConfigurationError(f"{env} is not set for the {choice.provider} provider.")
    if choice.provider == "anthropic":
        return AnthropicProvider(
            api_key=key,
            effort=settings.anthropic_effort,
            model=choice.model,
            base_url=settings.anthropic_base_url,
            price=choice.price,
            timeout=timeout,
            transport=transport,
        )
    return OpenAIProvider(
        api_key=key,
        model=choice.model,
        base_url=settings.openai_base_url,
        price=choice.price,
        timeout=timeout,
        transport=transport,
    )
