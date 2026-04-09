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

"""OpenAI integration tests — requires OPENAI_API_KEY."""

from __future__ import annotations

import pytest

from conftest import skip_if_no_openai
from aevyra_verdict.dataset import Dataset
from aevyra_verdict.providers import get_provider
from aevyra_verdict.runner import EvalRunner, RunConfig

OPENAI_MODEL = "gpt-5.4-nano"
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
    def test_full_eval(self, gsm8k_dataset):
        results = _run_eval(gsm8k_dataset)
        assert results is not None
        assert len(results.metric_names) > 0

    @skip_if_no_openai
    def test_deterministic_at_zero_temperature(self):
        """Two completions at temp=0 should return identical text."""
        p = get_provider("openai", OPENAI_MODEL)
        r1 = p.complete(TINY_PROMPT, max_tokens=16, temperature=0.0)
        r2 = p.complete(TINY_PROMPT, max_tokens=16, temperature=0.0)
        assert r1.text == r2.text
