"""Integration tests — each group requires its own service or API key.

Tests are automatically skipped if the service is not available.

Run selectively with pytest markers:
  pytest -m ollama
  pytest -m vllm
  pytest -m openrouter
  pytest -m openai
  pytest -m "not (ollama or vllm or openrouter or openai)"   # unit tests only
"""

from __future__ import annotations

import pytest

from conftest import (
    skip_if_no_ollama,
    skip_if_no_openai,
    skip_if_no_openrouter,
    skip_if_no_vllm,
)
from aevyra_verdict.dataset import Dataset
from aevyra_verdict.providers import get_provider
from aevyra_verdict.runner import EvalRunner, RunConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TINY_PROMPT = [{"role": "user", "content": "Reply with just the word 'ok'."}]


def _run_one(provider, model, base_url=None, api_key=None):
    """Instantiate a provider and run a single completion."""
    p = get_provider(provider, model, base_url=base_url, api_key=api_key)
    result = p.complete(TINY_PROMPT, max_tokens=16, temperature=0.0)
    return result


def _run_eval(provider, model, dataset_path, base_url=None, api_key=None):
    """Run a full eval with EvalRunner against a tiny dataset."""
    from aevyra_verdict.metrics import RougeScore

    ds = Dataset.from_jsonl(dataset_path)
    config = RunConfig(max_tokens=32, temperature=0.0, max_workers=2)
    runner = EvalRunner(config=config)
    runner.add_provider(provider, model, base_url=base_url, api_key=api_key)
    runner.add_metric(RougeScore())
    return runner.run(ds)


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------

OLLAMA_MODEL = "llama3.2:1b"
OLLAMA_BASE_URL = "http://localhost:11434/v1"


@pytest.mark.ollama
class TestOllama:
    @skip_if_no_ollama
    def test_single_completion(self):
        result = _run_one("local", OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
        assert isinstance(result.text, str)
        assert len(result.text) > 0
        assert result.provider == "local"

    @skip_if_no_ollama
    def test_latency_recorded(self):
        result = _run_one("local", OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
        assert result.latency_ms > 0

    @skip_if_no_ollama
    def test_full_eval(self, tiny_dataset):
        results = _run_eval("local", OLLAMA_MODEL, tiny_dataset, base_url=OLLAMA_BASE_URL)
        assert results is not None
        assert len(results.metric_names) > 0

    @skip_if_no_ollama
    def test_default_base_url(self):
        """local provider should default to Ollama's port without explicit base_url."""
        result = _run_one("local", OLLAMA_MODEL)
        assert len(result.text) > 0


# ---------------------------------------------------------------------------
# vLLM
# ---------------------------------------------------------------------------

VLLM_MODEL = "meta-llama/Llama-3.2-1B"
VLLM_BASE_URL = "http://localhost:8000/v1"


@pytest.mark.vllm
class TestVLLM:
    @skip_if_no_vllm
    def test_single_completion(self):
        result = _run_one("local", VLLM_MODEL, base_url=VLLM_BASE_URL)
        assert isinstance(result.text, str)
        assert len(result.text) > 0

    @skip_if_no_vllm
    def test_latency_recorded(self):
        result = _run_one("local", VLLM_MODEL, base_url=VLLM_BASE_URL)
        assert result.latency_ms > 0

    @skip_if_no_vllm
    def test_full_eval(self, tiny_dataset):
        results = _run_eval("local", VLLM_MODEL, tiny_dataset, base_url=VLLM_BASE_URL)
        assert results is not None
        assert len(results.metric_names) > 0


# ---------------------------------------------------------------------------
# OpenRouter
# ---------------------------------------------------------------------------

OPENROUTER_MODEL = "meta-llama/llama-3.2-3b-instruct:free"  # free tier


@pytest.mark.openrouter
class TestOpenRouter:
    @skip_if_no_openrouter
    def test_single_completion(self):
        result = _run_one("openrouter", OPENROUTER_MODEL)
        assert isinstance(result.text, str)
        assert len(result.text) > 0

    @skip_if_no_openrouter
    def test_latency_recorded(self):
        result = _run_one("openrouter", OPENROUTER_MODEL)
        assert result.latency_ms > 0

    @skip_if_no_openrouter
    def test_full_eval(self, tiny_dataset):
        results = _run_eval("openrouter", OPENROUTER_MODEL, tiny_dataset)
        assert results is not None
        assert len(results.metric_names) > 0

    @skip_if_no_openrouter
    def test_usage_returned(self):
        result = _run_one("openrouter", OPENROUTER_MODEL)
        # OpenRouter may or may not return token counts depending on the model
        assert isinstance(result.usage, dict)


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

OPENAI_MODEL = "gpt-4o-mini"  # cheapest model


@pytest.mark.openai
class TestOpenAI:
    @skip_if_no_openai
    def test_single_completion(self):
        result = _run_one("openai", OPENAI_MODEL)
        assert isinstance(result.text, str)
        assert len(result.text) > 0
        assert result.provider == "openai"

    @skip_if_no_openai
    def test_latency_recorded(self):
        result = _run_one("openai", OPENAI_MODEL)
        assert result.latency_ms > 0

    @skip_if_no_openai
    def test_usage_returned(self):
        result = _run_one("openai", OPENAI_MODEL)
        assert "prompt_tokens" in result.usage
        assert "completion_tokens" in result.usage

    @skip_if_no_openai
    def test_full_eval(self, tiny_dataset):
        results = _run_eval("openai", OPENAI_MODEL, tiny_dataset)
        assert results is not None
        assert len(results.metric_names) > 0

    @skip_if_no_openai
    def test_deterministic_at_zero_temperature(self):
        """Two completions at temp=0 should return identical text."""
        p = get_provider("openai", OPENAI_MODEL)
        r1 = p.complete(TINY_PROMPT, max_tokens=16, temperature=0.0)
        r2 = p.complete(TINY_PROMPT, max_tokens=16, temperature=0.0)
        assert r1.text == r2.text
