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

"""OpenRouter provider — access 200+ models through a single OpenAI-compatible API."""

from __future__ import annotations

import os
from typing import Any

from aevyra_verdict.providers.base import CompletionResult, Provider

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterProvider(Provider):
    """Provider for OpenRouter (openrouter.ai).

    OpenRouter exposes a unified OpenAI-compatible API across 200+ models from
    OpenAI, Anthropic, Google, Meta, Mistral, and others. Use model names in
    'provider/model' format as listed on openrouter.ai/models, e.g.:
        - anthropic/claude-sonnet-4-6
        - openai/gpt-5.4-mini
        - meta-llama/llama-3.1-8b-instruct
        - google/gemini-2.0-flash

    Requires the OPENROUTER_API_KEY environment variable.
    """

    name = "openrouter"

    def __init__(
        self,
        model: str = "anthropic/claude-sonnet-4-6",
        api_key: str | None = None,
        base_url: str | None = None,
        site_url: str | None = None,
        app_name: str | None = None,
        **kwargs: Any,
    ):
        """Args:
            model: Model identifier in 'provider/model' format (e.g. 'openai/gpt-5.4-mini').
            api_key: OpenRouter API key. Defaults to OPENROUTER_API_KEY env var.
            base_url: Override the OpenRouter base URL. Rarely needed.
            site_url: Your site URL, sent as HTTP-Referer. Recommended by OpenRouter
                      for attribution in their analytics.
            app_name: Your app name, sent as X-Title.
        """
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url or _OPENROUTER_BASE_URL,
            **kwargs,
        )
        from openai import OpenAI

        resolved_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not resolved_key:
            raise ValueError(
                "OpenRouter API key not found. Set the OPENROUTER_API_KEY environment variable "
                "or pass api_key= directly."
            )

        extra_headers = {}
        if site_url:
            extra_headers["HTTP-Referer"] = site_url
        if app_name:
            extra_headers["X-Title"] = app_name

        self._client = OpenAI(
            api_key=resolved_key,
            base_url=self.base_url,
            default_headers=extra_headers or None,
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
