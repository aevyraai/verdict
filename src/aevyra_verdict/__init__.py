"""Aevyra Evals — a framework for running LLM evals across models and providers."""

from aevyra_verdict.dataset import Dataset, Conversation
from aevyra_verdict.runner import EvalRunner
from aevyra_verdict.results import EvalResults
from aevyra_verdict.providers import get_provider
from aevyra_verdict.metrics import (
    ExactMatch,
    RougeScore,
    BleuScore,
    LLMJudge,
    CustomMetric,
)

__version__ = "0.1.0"

__all__ = [
    "Dataset",
    "Conversation",
    "EvalRunner",
    "EvalResults",
    "get_provider",
    "ExactMatch",
    "RougeScore",
    "BleuScore",
    "LLMJudge",
    "CustomMetric",
]
