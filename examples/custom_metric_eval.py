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

"""Example: using custom scoring functions and filtering by metadata."""

from aevyra_verdict import Dataset, EvalRunner, CustomMetric, RougeScore


def brevity_score(response: str, ideal: str | None = None, **kwargs) -> float:
    """Prefer concise responses. Score drops linearly after 200 words."""
    word_count = len(response.split())
    if word_count <= 200:
        return 1.0
    return max(0.0, 1.0 - (word_count - 200) / 300)


def contains_code(response: str, ideal: str | None = None, **kwargs) -> dict:
    """Check if the response contains a code block (for coding questions)."""
    has_code = "```" in response or "def " in response or "function " in response
    return {
        "score": 1.0 if has_code else 0.0,
        "reasoning": "Contains code block" if has_code else "No code found",
    }


dataset = Dataset.from_jsonl("examples/sample_data.jsonl")

# Filter to only coding questions
coding_dataset = dataset.filter(category="coding")
print(f"Coding questions: {len(coding_dataset)}")

runner = EvalRunner()
runner.add_provider("openai", "gpt-5.4-mini")
runner.add_provider("anthropic", "claude-sonnet-4-6")

runner.add_metric(RougeScore())
runner.add_metric(CustomMetric("brevity", brevity_score))
runner.add_metric(CustomMetric("contains_code", contains_code))

results = runner.run(coding_dataset)

print(results.compare("contains_code"))
print(results.compare("brevity"))
