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

_DEFAULT_CRITERIA = """\
- Accuracy: Is the response factually correct?
- Helpfulness: Does it address the user's question?
- Clarity: Is it well-written and easy to understand?
- Completeness: Does it cover the key points?"""


class LLMJudge(Metric):
    """Uses an LLM to evaluate responses.

    The judge model scores responses on configurable criteria (1-5 scale,
    normalized to 0.0-1.0). You can provide a custom prompt template or
    customize the criteria.
    """

    name = "llm_judge"

    def __init__(
        self,
        judge_provider: Provider,
        criteria: str | None = None,
        prompt_template: str | None = None,
        temperature: float = 0.0,
    ):
        """Args:
            judge_provider: A configured Provider instance to use as the judge.
            criteria: Evaluation criteria text. Defaults to accuracy/helpfulness/clarity.
            prompt_template: Custom prompt template with {criteria}, {conversation},
                           {response}, and {ideal_section} placeholders.
            temperature: Temperature for judge calls.
        """
        self.judge = judge_provider
        self.criteria = criteria or _DEFAULT_CRITERIA
        self.prompt_template = prompt_template or _DEFAULT_JUDGE_PROMPT
        self.temperature = temperature

    def score(
        self,
        response: str,
        ideal: str | None = None,
        messages: list[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> ScoreResult:
        conversation_text = ""
        if messages:
            for m in messages:
                conversation_text += f"[{m['role']}]: {m['content']}\n"

        ideal_section = ""
        if ideal:
            ideal_section = f"## Reference answer (for comparison)\n{ideal}"

        judge_prompt = self.prompt_template.format(
            criteria=self.criteria,
            conversation=conversation_text or "(no conversation provided)",
            response=response,
            ideal_section=ideal_section,
        )

        judge_result = self.judge.complete(
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=self.temperature,
            max_tokens=512,
        )

        raw_score, reasoning = self._parse_judge_response(judge_result.text)

        # Normalize 1-5 scale to 0.0-1.0
        normalized = (raw_score - 1) / 4.0

        return ScoreResult(
            score=normalized,
            metric_name=self.name,
            details={"raw_score": raw_score, "judge_model": judge_result.model},
            reasoning=reasoning,
        )

    def _parse_judge_response(self, text: str) -> tuple[int, str]:
        """Parse the judge's JSON response. Returns (score, reasoning)."""
        # Try to extract JSON from the response
        text = text.strip()

        # Handle markdown code blocks
        if "```" in text:
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
            if match:
                text = match.group(1)

        try:
            data = json.loads(text)
            score = int(data.get("score", 3))
            reasoning = data.get("reasoning", "")
        except (json.JSONDecodeError, ValueError):
            # Fallback: try to find a number in the text
            numbers = re.findall(r"\b([1-5])\b", text)
            score = int(numbers[0]) if numbers else 3
            reasoning = text

        score = max(1, min(5, score))
        return score, reasoning
