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

"""Eval runner — orchestrates running completions and scoring across models."""

from __future__ import annotations

import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from aevyra_verdict.dataset import Conversation, Dataset
from aevyra_verdict.metrics.base import Metric, ScoreResult
from aevyra_verdict.providers.base import CompletionResult, Provider
from aevyra_verdict.results import EvalResults, ModelResult


@dataclass
class RunConfig:
    """Configuration for an eval run."""

    temperature: float = 0.0
    max_tokens: int = 1024

    # Concurrency
    max_workers: int = 10
    """Number of concurrent requests per model. Tune down if you hit rate limits."""
    max_model_workers: int = 4
    """Number of models to evaluate concurrently."""

    # Retry / rate-limit handling
    num_retries: int = 4
    """Total retry attempts after the first failure."""
    retry_base_delay: float = 1.0
    """Initial delay in seconds before the first retry."""
    retry_max_delay: float = 60.0
    """Cap on retry delay — exponential backoff won't exceed this."""
    retry_jitter: float = 0.25
    """Random ± fraction added to each delay to avoid thundering-herd retries."""

    extra_kwargs: dict[str, Any] = field(default_factory=dict)


def _is_rate_limit_error(exc: Exception) -> bool:
    """Return True if the exception looks like an API rate-limit or quota error.

    Checks by exception class name (works across SDK versions without hard imports)
    and falls back to inspecting the error message for 429 / rate-limit signals.
    """
    class_name = type(exc).__name__
    if class_name in ("RateLimitError", "TooManyRequestsError", "QuotaExceededError"):
        return True

    # Some SDKs wrap HTTP errors — check the status code if present
    for attr in ("status_code", "code", "http_status"):
        if getattr(exc, attr, None) == 429:
            return True

    # Last resort: string match on the message
    msg = str(exc).lower()
    return any(
        token in msg for token in ("rate limit", "rate_limit", "quota", "429", "too many requests")
    )


def _backoff_delay(attempt: int, config: RunConfig) -> float:
    """Exponential backoff with full jitter.

    On attempt 0 the base delay applies; each subsequent attempt doubles it,
    capped at retry_max_delay. Jitter is a random ± fraction of the computed delay.
    """
    delay = min(config.retry_base_delay * (2**attempt), config.retry_max_delay)
    jitter = delay * config.retry_jitter * (2 * random.random() - 1)
    return max(0.0, delay + jitter)


class EvalRunner:
    """Runs evals: sends prompts to multiple models and scores the responses.

    Requests are dispatched concurrently (controlled by RunConfig.max_workers
    per model, and max_model_workers across models). Rate-limit errors trigger
    exponential backoff with jitter before retrying.

    Basic usage:
        runner = EvalRunner()
        runner.add_provider("openai", "gpt-5.4-nano")
        runner.add_provider("qwen", "qwen3.5-9b")
        runner.add_metric(ExactMatch())
        results = runner.run(dataset)
    """

    def __init__(self, config: RunConfig | None = None):
        self.config = config or RunConfig()
        self.providers: dict[str, Provider] = {}
        self.metrics: list[Metric] = []

    def add_provider(
        self,
        provider_name: str,
        model: str,
        *,
        label: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> EvalRunner:
        """Add a model to evaluate.

        Args:
            provider_name: "openai", "anthropic", "google", "mistral", "cohere".
            model: Model identifier (e.g., "gpt-5.4-mini", "claude-sonnet-4-6").
            label: Display label for this model in results. Defaults to
                   "{provider_name}/{model}".
            api_key: API key override (otherwise uses env var).
            base_url: Custom base URL for the API.

        Returns:
            self, for chaining.
        """
        from aevyra_verdict.providers import get_provider

        provider = get_provider(provider_name, model, api_key=api_key, base_url=base_url, **kwargs)
        key = label or f"{provider_name}/{model}"
        self.providers[key] = provider
        return self

    def add_provider_instance(self, label: str, provider: Provider) -> EvalRunner:
        """Add a pre-configured provider instance."""
        self.providers[label] = provider
        return self

    def add_metric(self, metric: Metric) -> EvalRunner:
        """Add a scoring metric."""
        self.metrics.append(metric)
        return self

    def run(
        self,
        dataset: Dataset,
        show_progress: bool = True,
    ) -> EvalResults:
        """Run the eval across all providers and metrics concurrently.

        Models are evaluated concurrently (up to max_model_workers). Within each
        model, samples are dispatched concurrently (up to max_workers). Rate-limit
        errors are retried with exponential backoff automatically.

        Returns:
            EvalResults with per-model, per-sample scores and aggregates.
        """
        if not self.providers:
            raise ValueError("No providers added. Call add_provider() first.")
        if not self.metrics:
            raise ValueError("No metrics added. Call add_metric() first.")

        # Fail fast if reference-based metrics are paired with a label-free dataset.
        if not dataset.has_ideals():
            needs_ideal = [m.name for m in self.metrics if getattr(m, "requires_ideal", False)]
            if needs_ideal:
                raise ValueError(
                    f"The following metrics require reference answers (ideal), but the dataset "
                    f"has none: {needs_ideal}. Either supply a labelled dataset or replace these "
                    f"metrics with LLMJudge for label-free evaluation."
                )

        all_model_results: dict[str, ModelResult] = {}

        with ThreadPoolExecutor(max_workers=self.config.max_model_workers) as model_pool:
            future_to_label = {
                model_pool.submit(
                    self._run_single_model,
                    label=label,
                    provider=provider,
                    dataset=dataset,
                    show_progress=show_progress,
                ): label
                for label, provider in self.providers.items()
            }
            for future in as_completed(future_to_label):
                label = future_to_label[future]
                all_model_results[label] = future.result()

        # Preserve insertion order
        ordered = {label: all_model_results[label] for label in self.providers}

        return EvalResults(
            dataset_name=dataset.name,
            model_results=ordered,
            metric_names=[m.name for m in self.metrics],
        )

    def _run_single_model(
        self,
        label: str,
        provider: Provider,
        dataset: Dataset,
        show_progress: bool,
    ) -> ModelResult:
        """Run all conversations against a single model, concurrently."""
        n = len(dataset)
        completions: list[CompletionResult | None] = [None] * n
        scores: list[dict[str, ScoreResult]] = [{} for _ in range(n)]
        errors: list[str | None] = [None] * n

        progress = None
        if show_progress and tqdm is not None:
            progress = tqdm(total=n, desc=f"  {label}", file=sys.stderr)

        def process_sample(idx: int, conversation: Conversation):
            completion, error = self._get_completion(provider, conversation)
            sample_scores: dict[str, ScoreResult] = {}

            if completion and not error:
                for metric in self.metrics:
                    try:
                        sample_scores[metric.name] = metric.score(
                            response=completion.text,
                            ideal=conversation.ideal,
                            messages=conversation.prompt_messages,
                        )
                    except Exception as e:
                        sample_scores[metric.name] = ScoreResult(
                            score=0.0,
                            metric_name=metric.name,
                            details={"error": str(e)},
                        )

            return idx, completion, sample_scores, error

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as pool:
            futures = [pool.submit(process_sample, idx, conv) for idx, conv in enumerate(dataset)]
            for future in as_completed(futures):
                idx, completion, sample_scores, error = future.result()
                completions[idx] = completion
                scores[idx] = sample_scores
                errors[idx] = error
                if progress is not None:
                    progress.update(1)

        if progress is not None:
            progress.close()

        return ModelResult(
            label=label,
            provider_name=provider.name,
            model=provider.model,
            completions=completions,
            scores=scores,
            errors=errors,
        )

    def _get_completion(
        self,
        provider: Provider,
        conversation: Conversation,
    ) -> tuple[CompletionResult | None, str | None]:
        """Get a completion with retry logic and exponential backoff.

        Rate-limit errors always trigger a backoff sleep before retrying.
        Other transient errors (timeouts, 5xx) are retried immediately on the
        first attempt, then with backoff on subsequent attempts.
        Non-retryable errors (bad request, auth failure) are surfaced immediately
        without burning retry budget.
        """
        last_error: str | None = None

        for attempt in range(self.config.num_retries + 1):
            try:
                result = provider.complete(
                    messages=conversation.prompt_messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    **self.config.extra_kwargs,
                )
                return result, None

            except Exception as exc:
                is_rate_limit = _is_rate_limit_error(exc)
                last_error = (
                    f"[attempt {attempt + 1}/{self.config.num_retries + 1}] "
                    f"{'RateLimit' if is_rate_limit else type(exc).__name__}: {exc}"
                )

                # Don't retry on non-transient errors (auth, bad request)
                if _is_fatal_error(exc):
                    return None, last_error

                if attempt < self.config.num_retries:
                    delay = _backoff_delay(attempt, self.config)
                    # Rate limits always get the full delay; other errors skip the
                    # initial delay on attempt 0 to fast-retry transient blips.
                    if is_rate_limit or attempt > 0:
                        time.sleep(delay)

        return None, last_error


def _is_fatal_error(exc: Exception) -> bool:
    """Return True for errors that won't be fixed by retrying (auth, bad request)."""
    class_name = type(exc).__name__
    if class_name in ("AuthenticationError", "PermissionDeniedError", "InvalidRequestError"):
        return True

    for attr in ("status_code", "code", "http_status"):
        status = getattr(exc, attr, None)
        if status in (400, 401, 403):
            return True

    return False
