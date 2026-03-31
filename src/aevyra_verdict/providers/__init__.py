# Copyright 2026 Aevyra AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LLM provider registry."""

from __future__ import annotations

from typing import Any

from aevyra_verdict.providers.base import CompletionResult, Provider

_PROVIDER_REGISTRY: dict[str, type[Provider]] = {}


def _register_builtins():
    """Lazy-register built-in providers to avoid import errors if SDKs aren't installed."""
    from aevyra_verdict.providers.openai_provider import OpenAIProvider
    from aevyra_verdict.providers.anthropic_provider import AnthropicProvider
    from aevyra_verdict.providers.google_provider import GoogleProvider
    from aevyra_verdict.providers.mistral_provider import MistralProvider
    from aevyra_verdict.providers.cohere_provider import CohereProvider
    from aevyra_verdict.providers.openrouter_provider import OpenRouterProvider

    for cls in [OpenAIProvider, AnthropicProvider, GoogleProvider, MistralProvider, CohereProvider, OpenRouterProvider]:
        _PROVIDER_REGISTRY[cls.name] = cls

    # 'local' is an alias for OpenAI-compatible local endpoints (Ollama, vLLM).
    # No API key required — base_url must be supplied via --base-url or config.
    class LocalProvider(OpenAIProvider):
        name = "local"

        def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None, **kwargs):
            # Ollama/vLLM don't need a real key; use a placeholder so openai client doesn't complain
            super().__init__(
                model=model,
                api_key=api_key or "local",
                base_url=base_url or "http://localhost:11434/v1",
                **kwargs,
            )

        def complete(self, messages, temperature=0.0, max_tokens=1024, **kwargs):
            result = super().complete(messages, temperature=temperature, max_tokens=max_tokens, **kwargs)
            result.provider = "local"
            return result

    _PROVIDER_REGISTRY["local"] = LocalProvider


def register_provider(name: str, cls: type[Provider]) -> None:
    """Register a custom provider class."""
    _PROVIDER_REGISTRY[name] = cls


def get_provider(name: str, model: str, **kwargs: Any) -> Provider:
    """Instantiate a provider by name.

    Args:
        name: Provider name — "openai", "anthropic", "google", "mistral", "cohere",
              or a custom-registered name.
        model: Model identifier to use.
        **kwargs: Passed to the provider constructor (api_key, base_url, etc.).

    Returns:
        A configured Provider instance ready to call .complete().

    Example:
        provider = get_provider("anthropic", "claude-sonnet-4-20250514")
        result = provider.complete([{"role": "user", "content": "Hello"}])
    """
    if not _PROVIDER_REGISTRY:
        _register_builtins()

    if name not in _PROVIDER_REGISTRY:
        available = ", ".join(sorted(_PROVIDER_REGISTRY.keys()))
        raise ValueError(
            f"Unknown provider {name!r}. Available: {available}. "
            f"Use register_provider() to add custom providers."
        )
    return _PROVIDER_REGISTRY[name](model=model, **kwargs)


def list_providers() -> list[str]:
    """Return names of all registered providers."""
    if not _PROVIDER_REGISTRY:
        _register_builtins()
    return sorted(_PROVIDER_REGISTRY.keys())


__all__ = [
    "Provider",
    "CompletionResult",
    "get_provider",
    "register_provider",
    "list_providers",
]
