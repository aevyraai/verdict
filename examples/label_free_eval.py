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

"""Label-free evaluation example.

Demonstrates how to evaluate model responses when you have no reference answers
(ideals). LLMJudge scores responses autonomously based on criteria you define —
no ground-truth labels required.

Run:
    export OPENAI_API_KEY=...
    python examples/label_free_eval.py
"""

from aevyra_verdict import Dataset, EvalRunner, LLMJudge
from aevyra_verdict.providers import get_provider

# ---------------------------------------------------------------------------
# Load a dataset with no ideal answers
# ---------------------------------------------------------------------------
dataset = Dataset.from_jsonl("examples/label_free_data.jsonl")

print(f"Dataset: {len(dataset)} samples, has_ideals={dataset.has_ideals()}")

# ---------------------------------------------------------------------------
# Set up the judge
# Using the same provider for both generation and judging is fine for demos,
# but in production you may want a stronger model as the judge.
# ---------------------------------------------------------------------------
judge_provider = get_provider("openai", "gpt-5.4")

# Single-score judge with default criteria (accuracy, helpfulness, clarity, completeness)
judge = LLMJudge(judge_provider=judge_provider)

# Or score across specific dimensions that matter for your task:
# judge = LLMJudge(
#     judge_provider=judge_provider,
#     dimensions=["clarity", "accuracy", "conciseness"],
# )

# Or provide a fully custom evaluation prompt:
# judge = LLMJudge(
#     judge_provider=judge_provider,
#     criteria="Focus on whether the response is accurate and easy to understand.",
# )

# ---------------------------------------------------------------------------
# Run the eval — no ideals needed
# ---------------------------------------------------------------------------
runner = EvalRunner()
runner.add_provider("openai", "gpt-5.4-nano")
runner.add_metric(judge)

results = runner.run(dataset)

# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------
print("\n=== Results ===")
summary = results.summary()
for model_label, model_summary in summary.items():
    print(f"\n{model_label}")
    for metric_name, stats in model_summary.items():
        print(f"  {metric_name}: mean={stats['mean']:.3f}, std={stats['std']:.3f}")

# Per-sample reasoning from the judge
print("\n=== Judge reasoning (sample 0) ===")
model_result = list(results.model_results.values())[0]
first_scores = model_result.scores[0]
for metric_name, score_result in first_scores.items():
    print(f"Score: {score_result.score:.2f}")
    if score_result.reasoning:
        print(f"Reasoning: {score_result.reasoning}")
    if score_result.sub_scores:
        print(f"Per-dimension: {score_result.sub_scores}")
