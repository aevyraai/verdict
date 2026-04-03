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

"""Example custom metric functions for use with --custom-metric.

Usage:
  aevyra-verdict run examples/sample_data.jsonl -m openai/gpt-5.4-nano --custom-metric custom_metrics.py:brevity_score
  aevyra-verdict run examples/sample_data.jsonl -m openai/gpt-5.4-nano --custom-metric custom_metrics.py:contains_code
"""


def brevity_score(response: str, ideal: str | None = None, **kwargs) -> float:
    """Prefer concise responses. Score drops linearly after 150 words."""
    word_count = len(response.split())
    if word_count <= 150:
        return 1.0
    return max(0.0, 1.0 - (word_count - 150) / 200)


def contains_code(response: str, ideal: str | None = None, **kwargs) -> dict:
    """Check whether the response includes a code block."""
    has_code = "```" in response or response.count("    ") >= 2
    return {
        "score": 1.0 if has_code else 0.0,
        "reasoning": "Contains code block" if has_code else "No code found",
    }


def keyword_coverage(response: str, ideal: str | None = None, **kwargs) -> float:
    """Fraction of keywords from the ideal answer that appear in the response."""
    if not ideal:
        return 0.0
    ideal_words = {w.lower() for w in ideal.split() if len(w) > 4}
    if not ideal_words:
        return 1.0
    response_words = {w.lower() for w in response.split()}
    return len(ideal_words & response_words) / len(ideal_words)
