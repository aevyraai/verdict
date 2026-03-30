"""Base metric interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ScoreResult:
    """The result of scoring a single response."""

    score: float  # Normalized to 0.0–1.0 where possible
    metric_name: str
    details: dict[str, Any] | None = None  # Metric-specific breakdown
    reasoning: str | None = None  # For LLM-as-judge, the judge's reasoning


class Metric(ABC):
    """Abstract base class for eval metrics."""

    name: str

    @abstractmethod
    def score(
        self,
        response: str,
        ideal: str | None = None,
        messages: list[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> ScoreResult:
        """Score a model response.

        Args:
            response: The model's output text.
            ideal: The reference/expected output (if available).
            messages: The original prompt messages (useful for LLM-as-judge).
            **kwargs: Additional metric-specific arguments.

        Returns:
            A ScoreResult with a normalized score and optional details.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
