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

"""OpenRouter integration tests — requires OPENROUTER_API_KEY."""

from __future__ import annotations

import pytest

from conftest import skip_if_no_openrouter
from aevyra_verdict.dataset import Dataset
from aevyra_verdict.providers import get_provider
from aevyra_verdict.runner import EvalRunner, RunConfig

OPENROUTER_MODEL = "openrouter/free"  # auto-selects from available free models
TINY_PROMPT = [{"role": "user", "content": "Reply with just the word 'ok'."}]


def _run_one():
    p = get_provider("openrouter", OPENROUTER_MODEL)
    return p.complete(TINY_PROMPT, max_tokens=256, temperature=0.0)


def _run_eval(dataset_path):
    from aevyra_verdict.metrics import RougeScore
    ds = Dataset.from_jsonl(dataset_path)
    config = RunConfig(max_tokens=256, temperature=0.0, max_workers=2)
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
    def test_full_eval(self, gsm8k_dataset):
        results = _run_eval(gsm8k_dataset)
        assert results is not None
        assert len(results.metric_names) > 0

    @skip_if_no_openrouter
    def test_usage_returned(self):
        result = _run_one()
        assert isinstance(result.usage, dict)
