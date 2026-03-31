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

"""Aevyra Verdict — a framework for evaluating and comparing LLM outputs across models and providers."""

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
