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

"""Mistral provider — translates OpenAI message format to Mistral's API."""

from __future__ import annotations

import os
from typing import Any

from aevyra_verdict.providers.base import CompletionResult, Provider


class MistralProvider(Provider):
    """Provider for Mistral AI models."""

    name = "mistral"

    def __init__(
        self,
        model: str = "mistral-large-latest",
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(model=model, api_key=api_key, base_url=base_url, **kwargs)
        try:
            from mistralai import Mistral
        except ImportError:
            raise ImportError(
                "The Mistral provider requires the mistralai package. "
                "Install it with: pip install aevyra-verdict[mistral]"
            )

        self._client = Mistral(
            api_key=api_key or os.environ.get("MISTRAL_API_KEY"),
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
            return self._client.chat.complete(
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
