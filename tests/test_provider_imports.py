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

"""Tests that each provider raises a clear ImportError when its SDK is missing."""

from __future__ import annotations

import sys
import unittest
from unittest.mock import patch

from aevyra_verdict.providers.openai_provider import OpenAIProvider
from aevyra_verdict.providers.openrouter_provider import OpenRouterProvider
from aevyra_verdict.providers.anthropic_provider import AnthropicProvider
from aevyra_verdict.providers.google_provider import GoogleProvider
from aevyra_verdict.providers.mistral_provider import MistralProvider
from aevyra_verdict.providers.cohere_provider import CohereProvider


def _block(package: str):
    """Return a context manager that makes `package` appear uninstalled."""
    return patch.dict(sys.modules, {package: None})


class TestProviderImportErrors(unittest.TestCase):
    """Each provider should raise ImportError with a helpful install message
    when its underlying SDK is not available. The lazy import happens in
    __init__, so the class can be imported fine — only instantiation fails."""

    def _assert_helpful_error(self, exc: ImportError, extra: str) -> None:
        msg = str(exc)
        self.assertIn("pip install", msg)
        self.assertIn(extra, msg)

    def test_openai_provider_missing_sdk(self):
        with _block("openai"):
            with self.assertRaises(ImportError) as ctx:
                OpenAIProvider(model="gpt-5.4-nano", api_key="test")
        self._assert_helpful_error(ctx.exception, "openai")

    def test_openrouter_provider_missing_sdk(self):
        with _block("openai"):
            with self.assertRaises(ImportError) as ctx:
                OpenRouterProvider(model="openai/gpt-5.4-nano", api_key="test")
        self._assert_helpful_error(ctx.exception, "openai")

    def test_anthropic_provider_missing_sdk(self):
        with _block("anthropic"):
            with self.assertRaises(ImportError) as ctx:
                AnthropicProvider(model="claude-haiku", api_key="test")
        self._assert_helpful_error(ctx.exception, "anthropic")

    def test_google_provider_missing_sdk(self):
        with _block("google"):
            with self.assertRaises(ImportError) as ctx:
                GoogleProvider(model="gemini-2.0-flash", api_key="test")
        self._assert_helpful_error(ctx.exception, "google")

    def test_mistral_provider_missing_sdk(self):
        with _block("mistralai"):
            with self.assertRaises(ImportError) as ctx:
                MistralProvider(model="mistral-small", api_key="test")
        self._assert_helpful_error(ctx.exception, "mistral")

    def test_cohere_provider_missing_sdk(self):
        with _block("cohere"):
            with self.assertRaises(ImportError) as ctx:
                CohereProvider(model="command-r", api_key="test")
        self._assert_helpful_error(ctx.exception, "cohere")


if __name__ == "__main__":
    unittest.main()
