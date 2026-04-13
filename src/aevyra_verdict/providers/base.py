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

"""Base provider interface for LLM API providers."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_RETRY_DELAYS = [5, 10, 20, 40, 60]  # seconds between attempts


def _is_retryable(exc: BaseException) -> bool:
    """Return True if the exception is a transient rate-limit or overload error."""
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status in (429, 529):
        return True
    # anthropic.OverloadedError has no status_code attr on older SDK versions
    if type(exc).__name__ in ("OverloadedError", "RateLimitError"):
        return True
    return False


@dataclass
class CompletionResult:
    """The result of a single completion request."""

    text: str
    model: str
    provider: str
    usage: dict[str, int] = field(default_factory=dict)  # prompt_tokens, completion_tokens
    latency_ms: float = 0.0
    raw_response: Any = None

    @property
    def total_tokens(self) -> int:
        return self.usage.get("prompt_tokens", 0) + self.usage.get("completion_tokens", 0)


class Provider(ABC):
    """Abstract base class for LLM providers.

    Each provider translates the OpenAI message format into whatever
    the underlying API expects, then normalizes the response back into
    a CompletionResult.
    """

    name: str

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.extra_kwargs = kwargs

    @abstractmethod
    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> CompletionResult:
        """Send a completion request and return the result."""
        ...

    def _timed_complete(
        self,
        fn,
        *args,
        **kwargs,
    ) -> tuple[Any, float]:
        """Helper to time a completion call. Returns (result, latency_ms).

        Automatically retries on transient rate-limit (429) and overload (529)
        errors with exponential backoff. Raises on the final attempt.
        """
        for attempt, delay in enumerate(_RETRY_DELAYS + [None]):
            start = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
                latency_ms = (time.perf_counter() - start) * 1000
                return result, latency_ms
            except Exception as exc:
                if not _is_retryable(exc) or delay is None:
                    raise
                logger.warning(
                    "%s %s (attempt %d/%d): %s — retrying in %ds",
                    self.__class__.__name__, self.model,
                    attempt + 1, len(_RETRY_DELAYS),
                    type(exc).__name__, delay,
                )
                time.sleep(delay)

        raise RuntimeError("unreachable")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model!r})"
