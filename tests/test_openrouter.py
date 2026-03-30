"""OpenRouter integration tests — requires OPENROUTER_API_KEY."""

from __future__ import annotations

import pytest

from conftest import skip_if_no_openrouter
from aevyra_verdict.dataset import Dataset
from aevyra_verdict.providers import get_provider
from aevyra_verdict.runner import EvalRunner, RunConfig

OPENROUTER_MODEL = "google/gemini-2.0-flash-exp:free"
TINY_PROMPT = [{"role": "user", "content": "Reply with just the word 'ok'."}]


def _run_one():
    p = get_provider("openrouter", OPENROUTER_MODEL)
    return p.complete(TINY_PROMPT, max_tokens=16, temperature=0.0)


def _run_eval(dataset_path):
    from aevyra_verdict.metrics import RougeScore
    ds = Dataset.from_jsonl(dataset_path)
    config = RunConfig(max_tokens=32, temperature=0.0, max_workers=2)
    runner = EvalRunner(config=config)
    runner.add_provider("openrouter", OPENROUTER_MODEL)
    runner.add_metric(RougeScore())
    return runner.run(ds)


@pytest.mark.openrouter
class TestOpenRouter:
    @skip_if_no_openrouter
    def test_single_completion(self):
        result = _run_one()
        assert isinstance(result.text, str)
        assert len(result.text) > 0

    @skip_if_no_openrouter
    def test_latency_recorded(self):
        result = _run_one()
        assert result.latency_ms > 0

    @skip_if_no_openrouter
    def test_full_eval(self, tiny_dataset):
        results = _run_eval(tiny_dataset)
        assert results is not None
        assert len(results.metric_names) > 0

    @skip_if_no_openrouter
    def test_usage_returned(self):
        result = _run_one()
        assert isinstance(result.usage, dict)
