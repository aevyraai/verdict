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

"""Anthropic provider — translates OpenAI message format to Anthropic's API."""

from __future__ import annotations

import os
from typing import Any

from aevyra_verdict.providers.base import CompletionResult, Provider


class AnthropicProvider(Provider):
    """Provider for Anthropic's Claude models."""

    name = "anthropic"

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(model=model, api_key=api_key, base_url=base_url, **kwargs)
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "The Anthropic provider requires the anthropic package. "
                "Install it with: pip install aevyra-verdict[anthropic]"
            )

        client_kwargs = {"api_key": api_key or os.environ.get("ANTHROPIC_API_KEY")}
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = anthropic.Anthropic(**client_kwargs)

    def _split_system(
        self, messages: list[dict[str, str]]
    ) -> tuple[str | None, list[dict[str, str]]]:
        """Anthropic takes system as a separate parameter, not in messages."""
        system = None
        filtered = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                filtered.append(m)
        return system, filtered

    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> CompletionResult:
        system, conv_messages = self._split_system(messages)
        merged_kwargs = {**self.extra_kwargs, **kwargs}

        def _call():
            call_kwargs = {
                "model": self.model,
                "messages": conv_messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                **merged_kwargs,
            }
            if system:
                call_kwargs["system"] = system
            return self._client.messages.create(**call_kwargs)

        response, latency_ms = self._timed_complete(_call)

        text = ""
        for block in response.content:
            if hasattr(block, "text"):
                text += block.text

        usage = {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
        }

        return CompletionResult(
            text=text,
            model=response.model,
            provider=self.name,
            usage=usage,
            latency_ms=latency_ms,
            raw_response=response,
        )
