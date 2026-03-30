"""vLLM integration tests — requires vLLM running at localhost:8000.

Override model/URL via env vars:
  VLLM_MODEL=facebook/opt-125m VLLM_BASE_URL=http://localhost:8000/v1 pytest tests/test_vllm.py
"""

from __future__ import annotations

import os

import pytest

from conftest import skip_if_no_vllm
from aevyra_verdict.dataset import Dataset
from aevyra_verdict.providers import get_provider
from aevyra_verdict.runner import EvalRunner, RunConfig

VLLM_MODEL = os.environ.get("VLLM_MODEL", "meta-llama/Llama-3.2-1B")
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
TINY_PROMPT = [{"role": "user", "content": "Reply with just the word 'ok'."}]


def _run_one():
    p = get_provider("local", VLLM_MODEL, base_url=VLLM_BASE_URL)
    return p.complete(TINY_PROMPT, max_tokens=16, temperature=0.0)


def _run_eval(dataset_path):
    from aevyra_verdict.metrics import RougeScore
    ds = Dataset.from_jsonl(dataset_path)
    config = RunConfig(max_tokens=32, temperature=0.0, max_workers=2)
    runner = EvalRunner(config=config)
    runner.add_provider("local", VLLM_MODEL, base_url=VLLM_BASE_URL)
    runner.add_metric(RougeScore())
    return runner.run(ds)


@pytest.mark.vllm
class TestVLLM:
    @skip_if_no_vllm
    def test_single_completion(self):
        result = _run_one()
        assert isinstance(result.text, str)
        assert len(result.text) > 0

    @skip_if_no_vllm
    def test_latency_recorded(self):
        result = _run_one()
        assert result.latency_ms > 0

    @skip_if_no_vllm
    def test_full_eval(self, tiny_dataset):
        results = _run_eval(tiny_dataset)
        assert results is not None
        assert len(results.metric_names) > 0
