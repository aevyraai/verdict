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

"""Google Gemini provider — translates OpenAI message format to the google-genai SDK."""

from __future__ import annotations

import os
from typing import Any

from aevyra_verdict.providers.base import CompletionResult, Provider


class GoogleProvider(Provider):
    """Provider for Google Gemini models via the google-genai SDK."""

    name = "google"

    _ROLE_MAP = {"user": "user", "assistant": "model", "system": "user"}

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(model=model, api_key=api_key, base_url=base_url, **kwargs)
        from google import genai

        self._client = genai.Client(
            api_key=api_key or os.environ.get("GOOGLE_API_KEY"),
        )

    def _convert_messages(
        self, messages: list[dict[str, str]]
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Convert OpenAI-format messages to Gemini format.

        Returns (system_instruction, contents).
        """
        system = None
        contents = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                role = self._ROLE_MAP.get(m["role"], "user")
                contents.append({"role": role, "parts": [{"text": m["content"]}]})
        return system, contents

    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> CompletionResult:
        from google.genai import types

        system, contents = self._convert_messages(messages)

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        if system:
            config.system_instruction = system

        def _call():
            return self._client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )

        response, latency_ms = self._timed_complete(_call)

        text = response.text or ""
        usage = {}
        if response.usage_metadata:
            usage = {
                "prompt_tokens": response.usage_metadata.prompt_token_count or 0,
                "completion_tokens": response.usage_metadata.candidates_token_count or 0,
            }

        return CompletionResult(
            text=text,
            model=self.model,
            provider=self.name,
            usage=usage,
            latency_ms=latency_ms,
            raw_response=response,
        )
