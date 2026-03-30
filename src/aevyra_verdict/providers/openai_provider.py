"""OpenAI provider — works with any OpenAI-compatible API."""

from __future__ import annotations

import os
from typing import Any

from aevyra_verdict.providers.base import CompletionResult, Provider


class OpenAIProvider(Provider):
    """Provider for OpenAI and OpenAI-compatible APIs (Azure, Together, etc.)."""

    name = "openai"

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(model=model, api_key=api_key, base_url=base_url, **kwargs)
        from openai import OpenAI

        self._client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url,
        )

    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> CompletionResult:
        merged_kwargs = {**self.extra_kwargs, **kwargs}

        def _call():
            return self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **merged_kwargs,
            )

        response, latency_ms = self._timed_complete(_call)
        choice = response.choices[0]

        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            }

        return CompletionResult(
            text=choice.message.content or "",
            model=response.model,
            provider=self.name,
            usage=usage,
            latency_ms=latency_ms,
            raw_response=response,
        )
