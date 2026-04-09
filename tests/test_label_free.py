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

"""Unit tests for label-free evaluation support.

All tests are fully offline — no API calls, no external services.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from aevyra_verdict.dataset import Dataset
from aevyra_verdict.metrics.base import Metric, ScoreResult
from aevyra_verdict.metrics.custom import CustomMetric
from aevyra_verdict.metrics.judge import LLMJudge
from aevyra_verdict.metrics.reference import BleuScore, ExactMatch, RougeScore
from aevyra_verdict.runner import EvalRunner, RunConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _label_free_dataset(tmp_path, n: int = 2) -> Dataset:
    """Return a Dataset with no ideal answers."""
    samples = [{"messages": [{"role": "user", "content": f"Question {i}"}]} for i in range(n)]
    path = tmp_path / "label_free.jsonl"
    path.write_text("\n".join(json.dumps(s) for s in samples))
    return Dataset.from_jsonl(path)


def _labelled_dataset(tmp_path) -> Dataset:
    """Return a Dataset with ideal answers."""
    samples = [
        {"messages": [{"role": "user", "content": "Say hello"}], "ideal": "hello"},
        {"messages": [{"role": "user", "content": "What is 1+1?"}], "ideal": "2"},
    ]
    path = tmp_path / "labelled.jsonl"
    path.write_text("\n".join(json.dumps(s) for s in samples))
    return Dataset.from_jsonl(path)


def _mock_provider(response_text: str = "test response"):
    """Return a mock Provider that always returns the given text."""
    from aevyra_verdict.providers.base import CompletionResult

    provider = MagicMock()
    provider.name = "mock"
    provider.model = "mock-model"
    provider.complete.return_value = CompletionResult(
        text=response_text,
        input_tokens=10,
        output_tokens=5,
        model="mock-model",
    )
    return provider


def _mock_judge_provider(score_json: str = '{"score": 4, "reasoning": "Good response"}'):
    """Return a mock Provider suitable for use as an LLMJudge judge."""
    from aevyra_verdict.providers.base import CompletionResult

    provider = MagicMock()
    provider.name = "mock_judge"
    provider.model = "mock-judge-model"
    provider.complete.return_value = CompletionResult(
        text=score_json,
        input_tokens=20,
        output_tokens=10,
        model="mock-judge-model",
    )
    return provider


# ---------------------------------------------------------------------------
# requires_ideal flag
# ---------------------------------------------------------------------------


class TestRequiresIdealFlag:
    def test_exact_match_requires_ideal(self):
        assert ExactMatch.requires_ideal is True

    def test_bleu_requires_ideal(self):
        assert BleuScore.requires_ideal is True

    def test_rouge_requires_ideal(self):
        assert RougeScore.requires_ideal is True

    def test_llm_judge_does_not_require_ideal(self):
        assert LLMJudge.requires_ideal is False

    def test_custom_metric_does_not_require_ideal_by_default(self):
        metric = CustomMetric("my_metric", fn=lambda response, **kw: 1.0)
        assert metric.requires_ideal is False

    def test_base_metric_defaults_to_false(self):
        class MyMetric(Metric):
            name = "my_metric"

            def score(self, response, ideal=None, messages=None, **kwargs):
                return ScoreResult(score=1.0, metric_name=self.name)

        assert MyMetric.requires_ideal is False


# ---------------------------------------------------------------------------
# Dataset label-free helpers
# ---------------------------------------------------------------------------


class TestDatasetLabelFree:
    def test_has_ideals_false_when_no_labels(self, tmp_path):
        ds = _label_free_dataset(tmp_path)
        assert ds.has_ideals() is False

    def test_has_ideals_true_when_all_labelled(self, tmp_path):
        ds = _labelled_dataset(tmp_path)
        assert ds.has_ideals() is True

    def test_ideal_is_none_for_label_free_samples(self, tmp_path):
        ds = _label_free_dataset(tmp_path)
        for sample in ds:
            assert sample.ideal is None

    def test_label_free_jsonl_loads_successfully(self):
        """The bundled label_free_data.jsonl example should load without errors."""
        from pathlib import Path

        path = Path(__file__).parent.parent / "examples" / "label_free_data.jsonl"
        if not path.exists():
            pytest.skip("label_free_data.jsonl not present")
        ds = Dataset.from_jsonl(path)
        assert len(ds) > 0
        assert not ds.has_ideals()


# ---------------------------------------------------------------------------
# EvalRunner validation
# ---------------------------------------------------------------------------


class TestEvalRunnerLabelFreeValidation:
    def _runner_with_provider(self, provider):
        runner = EvalRunner(config=RunConfig(max_workers=1, max_model_workers=1))
        runner.add_provider_instance("mock", provider)
        return runner

    def test_exact_match_on_label_free_raises(self, tmp_path):
        ds = _label_free_dataset(tmp_path)
        runner = self._runner_with_provider(_mock_provider())
        runner.add_metric(ExactMatch())

        with pytest.raises(ValueError, match="exact_match"):
            runner.run(ds)

    def test_rouge_on_label_free_raises(self, tmp_path):
        ds = _label_free_dataset(tmp_path)
        runner = self._runner_with_provider(_mock_provider())
        runner.add_metric(RougeScore())

        with pytest.raises(ValueError, match="rouge"):
            runner.run(ds)

    def test_bleu_on_label_free_raises(self, tmp_path):
        ds = _label_free_dataset(tmp_path)
        runner = self._runner_with_provider(_mock_provider())
        runner.add_metric(BleuScore())

        with pytest.raises(ValueError, match="bleu"):
            runner.run(ds)

    def test_error_message_mentions_llm_judge(self, tmp_path):
        ds = _label_free_dataset(tmp_path)
        runner = self._runner_with_provider(_mock_provider())
        runner.add_metric(ExactMatch())

        with pytest.raises(ValueError, match="LLMJudge"):
            runner.run(ds)

    def test_multiple_reference_metrics_all_listed_in_error(self, tmp_path):
        ds = _label_free_dataset(tmp_path)
        runner = self._runner_with_provider(_mock_provider())
        runner.add_metric(ExactMatch())
        runner.add_metric(RougeScore())

        with pytest.raises(ValueError) as exc_info:
            runner.run(ds)

        msg = str(exc_info.value)
        assert "exact_match" in msg
        assert "rouge" in msg

    def test_labelled_dataset_with_reference_metrics_does_not_raise(self, tmp_path):
        """Reference metrics should not raise when the dataset has ideals."""
        ds = _labelled_dataset(tmp_path)
        runner = self._runner_with_provider(_mock_provider("hello"))
        runner.add_metric(ExactMatch())
        # Should not raise
        results = runner.run(ds, show_progress=False)
        assert results is not None


# ---------------------------------------------------------------------------
# Label-free eval with LLMJudge (mocked)
# ---------------------------------------------------------------------------


class TestLabelFreeWithJudge:
    def test_llm_judge_runs_on_label_free_dataset(self, tmp_path):
        ds = _label_free_dataset(tmp_path)
        judge_provider = _mock_judge_provider('{"score": 4, "reasoning": "Good"}')

        runner = EvalRunner(config=RunConfig(max_workers=1, max_model_workers=1))
        runner.add_provider_instance("mock", _mock_provider())
        runner.add_metric(LLMJudge(judge_provider=judge_provider))

        results = runner.run(ds, show_progress=False)
        assert results is not None
        model_result = list(results.model_results.values())[0]
        for sample_scores in model_result.scores:
            assert "llm_judge" in sample_scores
            assert 0.0 <= sample_scores["llm_judge"].score <= 1.0

    def test_custom_metric_runs_on_label_free_dataset(self, tmp_path):
        ds = _label_free_dataset(tmp_path)

        def brevity(response: str, ideal: str | None = None, **kwargs) -> float:
            return 1.0 if len(response.split()) < 50 else 0.5

        runner = EvalRunner(config=RunConfig(max_workers=1, max_model_workers=1))
        runner.add_provider_instance("mock", _mock_provider("short answer"))
        runner.add_metric(CustomMetric("brevity", fn=brevity))

        results = runner.run(ds, show_progress=False)
        model_result = list(results.model_results.values())[0]
        for sample_scores in model_result.scores:
            assert "brevity" in sample_scores
            assert sample_scores["brevity"].score == 1.0
