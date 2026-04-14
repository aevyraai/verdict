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

"""LLM-as-judge metric — uses a model to evaluate another model's response."""

from __future__ import annotations

import json
import re
from typing import Any

from aevyra_verdict.metrics.base import Metric, ScoreResult
from aevyra_verdict.providers.base import Provider


_DEFAULT_JUDGE_PROMPT = """\
You are an expert evaluator. Score the following AI response on a scale from 1 to 5.

## Criteria
{criteria}

## Conversation
{conversation}

## Response to evaluate
{response}

{ideal_section}

## Instructions
Provide your evaluation as JSON with two fields:
- "score": an integer from 1 to 5
- "reasoning": a brief explanation of your score

Respond ONLY with the JSON object, no other text.
"""

_MULTI_JUDGE_PROMPT = """\
You are an expert evaluator. Score the following AI response on each dimension below, \
using a scale from 1 to 5.

## Dimensions
{dimensions_list}

## Conversation
{conversation}

## Response to evaluate
{response}

{ideal_section}

## Instructions
Provide your evaluation as JSON with:
- "scores": an object mapping each dimension name to an integer from 1 to 5
- "reasoning": a brief explanation of your scores

Respond ONLY with the JSON object, no other text.
"""

_DEFAULT_CRITERIA = """\
- Accuracy: Is the response factually correct?
- Helpfulness: Does it address the user's question?
- Clarity: Is it well-written and easy to understand?
- Completeness: Does it cover the key points?"""


class LLMJudge(Metric):
    """Uses an LLM to evaluate responses.

    Single-score mode (default): the judge scores on configurable criteria
    returning a single 1–5 rating, normalized to 0.0–1.0.

    Multi-dimensional mode: pass ``dimensions`` to score each dimension
    independently in one API call. Each dimension is exposed as a separate
    entry in ``ScoreResult.sub_scores`` (e.g. ``"llm_judge_accuracy": 0.75``).
    The composite ``score`` is the mean across dimensions and is used for
    threshold comparison during optimization. All dimensions are tracked
    individually throughout the run.

    Example::

        LLMJudge(
            judge_provider=provider,
            dimensions=["accuracy", "helpfulness", "tone"],
        )
    """

    name = "llm_judge"

    @property
    def judge_tokens_used(self) -> int:
        """Total tokens consumed by the judge provider across all score() calls."""
        return getattr(self.judge, "tokens_used", 0)

    def __init__(
        self,
        judge_provider: Provider,
        criteria: str | None = None,
        prompt_template: str | None = None,
        dimensions: list[str] | None = None,
        temperature: float = 0.0,
    ):
        """Args:
        judge_provider: A configured Provider instance to use as the judge.
        criteria: Evaluation criteria text (single-score mode only).
            Defaults to accuracy/helpfulness/clarity/completeness.
        prompt_template: Fully custom prompt template. Single-score mode
            uses ``{criteria}``, ``{conversation}``, ``{response}``,
            ``{ideal_section}`` placeholders. Multi-dimensional mode uses
            ``{dimensions_list}``, ``{conversation}``, ``{response}``,
            ``{ideal_section}``.
        dimensions: List of dimension names for multi-dimensional scoring
            (e.g. ``["accuracy", "helpfulness", "tone"]``). When set,
            each dimension is scored separately in one API call and
            exposed in ``ScoreResult.sub_scores``.
        temperature: Temperature for judge calls.
        """
        self.judge = judge_provider
        self.dimensions = dimensions  # None = single-score mode
        self.criteria = criteria or _DEFAULT_CRITERIA
        self.prompt_template = prompt_template
        self.temperature = temperature

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #

    def score(
        self,
        response: str,
        ideal: str | None = None,
        messages: list[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> ScoreResult:
        if self.dimensions:
            return self._score_multi(response, ideal=ideal, messages=messages)
        return self._score_single(response, ideal=ideal, messages=messages)

    # ------------------------------------------------------------------ #
    # Single-score mode
    # ------------------------------------------------------------------ #

    def _score_single(
        self,
        response: str,
        ideal: str | None = None,
        messages: list[dict[str, str]] | None = None,
    ) -> ScoreResult:
        template = self.prompt_template or _DEFAULT_JUDGE_PROMPT
        judge_prompt = template.format(
            criteria=self.criteria,
            conversation=self._fmt_messages(messages),
            response=response,
            ideal_section=self._fmt_ideal(ideal),
        )
        judge_result = self.judge.complete(
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=self.temperature,
            max_tokens=512,
        )
        raw_score, reasoning = self._parse_single(judge_result.text)
        normalized = (raw_score - 1) / 4.0
        return ScoreResult(
            score=normalized,
            metric_name=self.name,
            details={"raw_score": raw_score, "judge_model": judge_result.model},
            reasoning=reasoning,
        )

    # ------------------------------------------------------------------ #
    # Multi-dimensional mode
    # ------------------------------------------------------------------ #

    def _score_multi(
        self,
        response: str,
        ideal: str | None = None,
        messages: list[dict[str, str]] | None = None,
    ) -> ScoreResult:
        assert self.dimensions  # guarded by caller
        dims_list = "\n".join(f"- {d}" for d in self.dimensions)
        template = self.prompt_template or _MULTI_JUDGE_PROMPT
        judge_prompt = template.format(
            dimensions_list=dims_list,
            conversation=self._fmt_messages(messages),
            response=response,
            ideal_section=self._fmt_ideal(ideal),
        )
        judge_result = self.judge.complete(
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=self.temperature,
            max_tokens=512,
        )
        dim_scores, reasoning = self._parse_multi(judge_result.text, self.dimensions)

        # Normalize each dimension (1–5 → 0–1) and store as sub_scores
        sub_scores = {f"{self.name}_{dim}": (raw - 1) / 4.0 for dim, raw in dim_scores.items()}
        composite = sum(sub_scores.values()) / len(sub_scores) if sub_scores else 0.0

        return ScoreResult(
            score=composite,
            metric_name=self.name,
            details={"raw_scores": dim_scores, "judge_model": judge_result.model},
            reasoning=reasoning,
            sub_scores=sub_scores,
        )

    # ------------------------------------------------------------------ #
    # Parsing helpers
    # ------------------------------------------------------------------ #

    def _parse_single(self, text: str) -> tuple[int, str]:
        """Parse a single-score judge response. Returns (score 1–5, reasoning)."""
        text = self._strip_code_fence(text)
        try:
            data = json.loads(text)
            score = int(data.get("score", 3))
            reasoning = data.get("reasoning", "")
        except (json.JSONDecodeError, ValueError):
            numbers = re.findall(r"\b([1-5])\b", text)
            score = int(numbers[0]) if numbers else 3
            reasoning = text
        return max(1, min(5, score)), reasoning

    def _parse_multi(self, text: str, dimensions: list[str]) -> tuple[dict[str, int], str]:
        """Parse a multi-dimensional judge response. Returns ({dim: score}, reasoning)."""
        text = self._strip_code_fence(text)
        reasoning = ""
        dim_scores: dict[str, int] = {}
        try:
            data = json.loads(text)
            reasoning = data.get("reasoning", "")
            raw = data.get("scores", {})
            for dim in dimensions:
                # Accept exact match or case-insensitive match
                val = raw.get(dim) or raw.get(dim.lower()) or raw.get(dim.capitalize())
                try:
                    dim_scores[dim] = max(1, min(5, int(val))) if val is not None else 3
                except (TypeError, ValueError):
                    dim_scores[dim] = 3
        except (json.JSONDecodeError, ValueError):
            # Fallback: assign 3 to all dimensions
            for dim in dimensions:
                dim_scores[dim] = 3
            reasoning = text
        # Fill any missing dimensions
        for dim in dimensions:
            dim_scores.setdefault(dim, 3)
        return dim_scores, reasoning

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        text = text.strip()
        if "```" in text:
            m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
            if m:
                return m.group(1)
        return text

    @staticmethod
    def _fmt_messages(messages: list[dict[str, str]] | None) -> str:
        if not messages:
            return "(no conversation provided)"
        return "".join(f"[{m['role']}]: {m['content']}\n" for m in messages)

    @staticmethod
    def _fmt_ideal(ideal: str | None) -> str:
        if not ideal:
            return ""
        return f"## Reference answer (for comparison)\n{ideal}"
