"""Custom metric — lets users define their own scoring function."""

from __future__ import annotations

from typing import Any, Callable

from aevyra_verdict.metrics.base import Metric, ScoreResult


class CustomMetric(Metric):
    """A metric defined by a user-supplied scoring function.

    The function receives (response, ideal, messages) and should return
    either a float (0.0–1.0) or a dict with at least a "score" key.

    Example:
        def word_count_score(response, ideal=None, messages=None):
            # Penalize responses that are too short
            count = len(response.split())
            return min(count / 100, 1.0)

        metric = CustomMetric("word_count", word_count_score)
    """

    def __init__(
        self,
        name: str,
        fn: Callable[..., float | dict[str, Any]],
    ):
        self.name = name
        self._fn = fn

    def score(
        self,
        response: str,
        ideal: str | None = None,
        messages: list[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> ScoreResult:
        result = self._fn(response, ideal=ideal, messages=messages, **kwargs)

        if isinstance(result, (int, float)):
            return ScoreResult(score=float(result), metric_name=self.name)

        if isinstance(result, dict):
            return ScoreResult(
                score=float(result["score"]),
                metric_name=self.name,
                details={k: v for k, v in result.items() if k != "score"},
                reasoning=result.get("reasoning"),
            )

        raise TypeError(
            f"Custom metric function must return a float or dict, got {type(result)}"
        )
