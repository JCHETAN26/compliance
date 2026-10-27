"""LLM provider abstraction: Anthropic, OpenAI and Ollama over HTTP, and an offline mock.

Every call is a single structured-output request; budgeting, retries and grounding are
done by the caller (``compliance_agents``).
"""

from compliance_llm.factory import (
    PROVIDERS,
    ProviderChoice,
    ProviderResolver,
    available_providers,
    build_provider,
    configured_providers,
    credential_present,
    default_model,
    resolve_choice,
)
from compliance_llm.mock import MOCK_MODEL, MockProvider, lexical_judgement
from compliance_llm.pricing import (
    RequestTooLargeError,
    estimate_input_tokens,
    reservation,
    resolve_price,
    usage_cost_micro,
)
from compliance_llm.types import (
    LLMConfigurationError,
    LLMError,
    LLMInvalidOutputError,
    LLMProvider,
    LLMRejectedError,
    LLMRequest,
    LLMResponse,
    LLMTransientError,
    LLMUsage,
    Price,
)

__all__ = [
    "MOCK_MODEL",
    "PROVIDERS",
    "LLMConfigurationError",
    "LLMError",
    "LLMInvalidOutputError",
    "LLMProvider",
    "LLMRejectedError",
    "LLMRequest",
    "LLMResponse",
    "LLMTransientError",
    "LLMUsage",
    "MockProvider",
    "Price",
    "ProviderChoice",
    "ProviderResolver",
    "RequestTooLargeError",
    "available_providers",
    "build_provider",
    "configured_providers",
    "credential_present",
    "default_model",
    "estimate_input_tokens",
    "lexical_judgement",
    "reservation",
    "resolve_choice",
    "resolve_price",
    "usage_cost_micro",
]
