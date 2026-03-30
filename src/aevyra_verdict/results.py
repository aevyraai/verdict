"""Eval results — aggregation, comparison, and export."""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aevyra_verdict.metrics.base import ScoreResult
from aevyra_verdict.providers.base import CompletionResult


@dataclass
class ModelResult:
    """Results for a single model across all eval samples."""

    label: str
    provider_name: str
    model: str
    completions: list[CompletionResult | None]
    scores: list[dict[str, ScoreResult]]  # per-sample, keyed by metric name
    errors: list[str | None]

    @property
    def num_samples(self) -> int:
        return len(self.completions)

    @property
    def num_errors(self) -> int:
        return sum(1 for e in self.errors if e is not None)

    @property
    def success_rate(self) -> float:
        if not self.completions:
            return 0.0
        return (self.num_samples - self.num_errors) / self.num_samples

    def mean_score(self, metric_name: str) -> float | None:
        """Average score for a specific metric across all successful samples."""
        values = []
        for sample_scores in self.scores:
            if metric_name in sample_scores:
                values.append(sample_scores[metric_name].score)
        return statistics.mean(values) if values else None

    def median_score(self, metric_name: str) -> float | None:
        values = [
            s[metric_name].score for s in self.scores if metric_name in s
        ]
        return statistics.median(values) if values else None

    def stdev_score(self, metric_name: str) -> float | None:
        values = [
            s[metric_name].score for s in self.scores if metric_name in s
        ]
        return statistics.stdev(values) if len(values) > 1 else None

    def mean_latency_ms(self) -> float | None:
        latencies = [c.latency_ms for c in self.completions if c is not None]
        return statistics.mean(latencies) if latencies else None

    def total_tokens(self) -> int:
        return sum(c.total_tokens for c in self.completions if c is not None)


class EvalResults:
    """Aggregated results from an eval run across multiple models."""

    def __init__(
        self,
        dataset_name: str,
        model_results: dict[str, ModelResult],
        metric_names: list[str],
    ):
        self.dataset_name = dataset_name
        self.model_results = model_results
        self.metric_names = metric_names

    @property
    def models(self) -> list[str]:
        return list(self.model_results.keys())

    def summary(self) -> dict[str, Any]:
        """Return a summary dict comparing all models across all metrics."""
        rows = {}
        for label, mr in self.model_results.items():
            row: dict[str, Any] = {
                "provider": mr.provider_name,
                "model": mr.model,
                "success_rate": round(mr.success_rate, 3),
                "mean_latency_ms": round(mr.mean_latency_ms() or 0, 1),
                "total_tokens": mr.total_tokens(),
            }
            for metric in self.metric_names:
                mean = mr.mean_score(metric)
                row[f"{metric}_mean"] = round(mean, 4) if mean is not None else None
                stdev = mr.stdev_score(metric)
                row[f"{metric}_stdev"] = round(stdev, 4) if stdev is not None else None
            rows[label] = row
        return rows

    def compare(self, metric_name: str | None = None) -> str:
        """Return a formatted comparison table as a string.

        If metric_name is None, uses the first metric.
        """
        metric = metric_name or (self.metric_names[0] if self.metric_names else None)
        if metric is None:
            return "No metrics to compare."

        lines = []
        lines.append(f"Eval: {self.dataset_name} | Metric: {metric}")
        lines.append("-" * 72)

        header = f"{'Model':<35} {'Mean':>8} {'Stdev':>8} {'Latency':>10} {'Errors':>7}"
        lines.append(header)
        lines.append("-" * 72)

        # Sort by mean score descending
        sorted_models = sorted(
            self.model_results.items(),
            key=lambda x: x[1].mean_score(metric) or 0,
            reverse=True,
        )

        for label, mr in sorted_models:
            mean = mr.mean_score(metric)
            stdev = mr.stdev_score(metric)
            latency = mr.mean_latency_ms()

            mean_str = f"{mean:>8.4f}" if mean is not None else f"{'N/A':>8}"
            stdev_str = f"{stdev:>8.4f}" if stdev is not None else f"{'N/A':>8}"
            latency_str = f"{latency:>8.1f}ms" if latency is not None else f"{'N/A':>10}"
            lines.append(
                f"{label:<35} {mean_str} {stdev_str} {latency_str} {mr.num_errors:>7}"
            )

        lines.append("-" * 72)
        return "\n".join(lines)

    def to_dataframe(self):
        """Convert summary to a pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame.from_dict(self.summary(), orient="index")

    def to_json(self, path: str | Path | None = None) -> str:
        """Export results as JSON."""
        data = {
            "dataset": self.dataset_name,
            "metrics": self.metric_names,
            "models": self.summary(),
            "per_sample": self._per_sample_data(),
        }
        json_str = json.dumps(data, indent=2, default=str)
        if path:
            Path(path).write_text(json_str)
        return json_str

    def _per_sample_data(self) -> list[dict[str, Any]]:
        """Build per-sample comparison data."""
        if not self.model_results:
            return []

        first_model = next(iter(self.model_results.values()))
        num_samples = first_model.num_samples
        samples = []

        for i in range(num_samples):
            sample: dict[str, Any] = {"index": i}
            for label, mr in self.model_results.items():
                entry: dict[str, Any] = {}
                comp = mr.completions[i]
                if comp:
                    entry["response"] = comp.text[:500]  # Truncate for export
                    entry["latency_ms"] = round(comp.latency_ms, 1)
                entry["error"] = mr.errors[i]
                for metric_name, score_result in mr.scores[i].items():
                    entry[f"{metric_name}_score"] = round(score_result.score, 4)
                    if score_result.reasoning:
                        entry[f"{metric_name}_reasoning"] = score_result.reasoning
                sample[label] = entry
            samples.append(sample)

        return samples

    def __repr__(self) -> str:
        return (
            f"EvalResults(dataset={self.dataset_name!r}, "
            f"models={self.models}, metrics={self.metric_names})"
        )
