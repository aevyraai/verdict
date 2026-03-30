"""Cohere provider — translates OpenAI message format to Cohere's chat API."""

from __future__ import annotations

import os
from typing import Any

from aevyra_verdict.providers.base import CompletionResult, Provider


class CohereProvider(Provider):
    """Provider for Cohere models."""

    name = "cohere"

    _ROLE_MAP = {"system": "system", "user": "user", "assistant": "assistant"}

    def __init__(
        self,
        model: str = "command-r-plus",
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(model=model, api_key=api_key, base_url=base_url, **kwargs)
        import cohere

        client_kwargs = {"api_key": api_key or os.environ.get("COHERE_API_KEY")}
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = cohere.ClientV2(**client_kwargs)

    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> CompletionResult:
        cohere_messages = [
            {"role": self._ROLE_MAP.get(m["role"], "user"), "content": m["content"]}
            for m in messages
        ]
        merged_kwargs = {**self.extra_kwargs, **kwargs}

        def _call():
            return self._client.chat(
                model=self.model,
                messages=cohere_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **merged_kwargs,
            )

        response, latency_ms = self._timed_complete(_call)

        text = ""
        if response.message and response.message.content:
            for block in response.message.content:
                if hasattr(block, "text"):
                    text += block.text

        usage = {}
        if response.usage and response.usage.tokens:
            usage = {
                "prompt_tokens": response.usage.tokens.input_tokens or 0,
                "completion_tokens": response.usage.tokens.output_tokens or 0,
            }

        return CompletionResult(
            text=text,
            model=self.model,
            provider=self.name,
            usage=usage,
            latency_ms=latency_ms,
            raw_response=response,
        )
