"""Ollama integration tests — requires Ollama running at localhost:11434."""

from __future__ import annotations

import pytest

from conftest import skip_if_no_ollama
from aevyra_verdict.dataset import Dataset
from aevyra_verdict.providers import get_provider
from aevyra_verdict.runner import EvalRunner, RunConfig

OLLAMA_MODEL = "llama3.2:1b"
OLLAMA_BASE_URL = "http://localhost:11434/v1"
TINY_PROMPT = [{"role": "user", "content": "Reply with just the word 'ok'."}]


def _run_one(base_url=OLLAMA_BASE_URL):
    p = get_provider("local", OLLAMA_MODEL, base_url=base_url)
    return p.complete(TINY_PROMPT, max_tokens=16, temperature=0.0)


def _run_eval(dataset_path):
    from aevyra_verdict.metrics import RougeScore
    ds = Dataset.from_jsonl(dataset_path)
    config = RunConfig(max_tokens=32, temperature=0.0, max_workers=2)
    runner = EvalRunner(config=config)
    runner.add_provider("local", OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
    runner.add_metric(RougeScore())
    return runner.run(ds)


@pytest.mark.ollama
class TestOllama:
    @skip_if_no_ollama
    def test_single_completion(self):
        result = _run_one()
        assert isinstance(result.text, str)
        assert len(result.text) > 0
        assert result.provider == "local"

    @skip_if_no_ollama
    def test_latency_recorded(self):
        result = _run_one()
        assert result.latency_ms > 0

    @skip_if_no_ollama
    def test_full_eval(self, tiny_dataset):
        results = _run_eval(tiny_dataset)
        assert results is not None
        assert len(results.metric_names) > 0

    @skip_if_no_ollama
    def test_default_base_url(self):
        """local provider should default to Ollama's port without explicit base_url."""
        result = _run_one(base_url=None)
        assert len(result.text) > 0
