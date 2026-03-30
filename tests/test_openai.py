"""OpenAI integration tests — requires OPENAI_API_KEY."""

from __future__ import annotations

import pytest

from conftest import skip_if_no_openai
from aevyra_verdict.dataset import Dataset
from aevyra_verdict.providers import get_provider
from aevyra_verdict.runner import EvalRunner, RunConfig

OPENAI_MODEL = "gpt-4o-mini"
TINY_PROMPT = [{"role": "user", "content": "Reply with just the word 'ok'."}]


def _run_one():
    p = get_provider("openai", OPENAI_MODEL)
    return p.complete(TINY_PROMPT, max_tokens=16, temperature=0.0)


def _run_eval(dataset_path):
    from aevyra_verdict.metrics import RougeScore
    ds = Dataset.from_jsonl(dataset_path)
    config = RunConfig(max_tokens=32, temperature=0.0, max_workers=2)
    runner = EvalRunner(config=config)
    runner.add_provider("openai", OPENAI_MODEL)
    runner.add_metric(RougeScore())
    return runner.run(ds)


@pytest.mark.openai
class TestOpenAI:
    @skip_if_no_openai
    def test_single_completion(self):
        result = _run_one()
        assert isinstance(result.text, str)
        assert len(result.text) > 0
        assert result.provider == "openai"

    @skip_if_no_openai
    def test_latency_recorded(self):
        result = _run_one()
        assert result.latency_ms > 0

    @skip_if_no_openai
    def test_usage_returned(self):
        result = _run_one()
        assert "prompt_tokens" in result.usage
        assert "completion_tokens" in result.usage

    @skip_if_no_openai
    def test_full_eval(self, tiny_dataset):
        results = _run_eval(tiny_dataset)
        assert results is not None
        assert len(results.metric_names) > 0

    @skip_if_no_openai
    def test_deterministic_at_zero_temperature(self):
        """Two completions at temp=0 should return identical text."""
        p = get_provider("openai", OPENAI_MODEL)
        r1 = p.complete(TINY_PROMPT, max_tokens=16, temperature=0.0)
        r2 = p.complete(TINY_PROMPT, max_tokens=16, temperature=0.0)
        assert r1.text == r2.text
