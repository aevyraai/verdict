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

"""Reference-based metrics: exact match, BLEU, ROUGE."""

from __future__ import annotations

from typing import Any

from aevyra_verdict.metrics.base import Metric, ScoreResult


class ExactMatch(Metric):
    """Exact string match between response and ideal (after normalization)."""

    name = "exact_match"
    requires_ideal = True

    def __init__(self, case_sensitive: bool = False, strip: bool = True):
        self.case_sensitive = case_sensitive
        self.strip = strip

    def score(
        self,
        response: str,
        ideal: str | None = None,
        messages: list[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> ScoreResult:
        if ideal is None:
            raise ValueError("ExactMatch requires an ideal (reference) response")

        a, b = response, ideal
        if self.strip:
            a, b = a.strip(), b.strip()
        if not self.case_sensitive:
            a, b = a.lower(), b.lower()

        match = a == b
        return ScoreResult(
            score=1.0 if match else 0.0,
            metric_name=self.name,
            details={"match": match},
        )


class BleuScore(Metric):
    """BLEU score using NLTK's sentence_bleu."""

    name = "bleu"
    requires_ideal = True

    def __init__(self, max_ngram: int = 4):
        self.max_ngram = max_ngram

    def score(
        self,
        response: str,
        ideal: str | None = None,
        messages: list[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> ScoreResult:
        if ideal is None:
            raise ValueError("BLEU requires an ideal (reference) response")

        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

        reference_tokens = ideal.split()
        hypothesis_tokens = response.split()

        weights = tuple(1.0 / self.max_ngram for _ in range(self.max_ngram))
        smoothing = SmoothingFunction().method1

        bleu = sentence_bleu(
            [reference_tokens],
            hypothesis_tokens,
            weights=weights,
            smoothing_function=smoothing,
        )

        return ScoreResult(
            score=float(bleu),
            metric_name=self.name,
            details={"max_ngram": self.max_ngram},
        )


class RougeScore(Metric):
    """ROUGE score using the rouge-score library."""

    name = "rouge"
    requires_ideal = True

    def __init__(self, variant: str = "rougeL"):
        """Args:
        variant: Which ROUGE variant to use. One of "rouge1", "rouge2", "rougeL".
        """
        self.variant = variant
        # Instantiate once to avoid "Using default tokenizer." printing on every call
        import io
        import contextlib
        from rouge_score import rouge_scorer
        with contextlib.redirect_stdout(io.StringIO()):
            self._scorer = rouge_scorer.RougeScorer([self.variant], use_stemmer=True)

    def score(
        self,
        response: str,
        ideal: str | None = None,
        messages: list[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> ScoreResult:
        if ideal is None:
            raise ValueError("ROUGE requires an ideal (reference) response")

        scores = self._scorer.score(ideal, response)

        result = scores[self.variant]
        return ScoreResult(
            score=float(result.fmeasure),
            metric_name=f"{self.name}_{self.variant}",
            details={
                "precision": float(result.precision),
                "recall": float(result.recall),
                "fmeasure": float(result.fmeasure),
            },
        )
