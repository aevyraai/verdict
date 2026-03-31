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

"""Dataset loading and representation for eval datasets.

Supported input formats:
  - openai    : {"messages": [...], "ideal": "...", "metadata": {...}}
  - sharegpt  : {"conversations": [{"from": "human", "value": "..."}]}
  - alpaca    : {"instruction": "...", "input": "...", "output": "..."}
  - auto      : detect format from the first record (default)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

Format = Literal["auto", "openai", "sharegpt", "alpaca"]


@dataclass
class Message:
    """A single message in a conversation."""

    role: str  # "system", "user", "assistant"
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> Message:
        return cls(role=data["role"], content=data["content"])


@dataclass
class Conversation:
    """A single eval sample: a conversation with an optional reference output."""

    messages: list[Message]
    ideal: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def prompt_messages(self) -> list[dict[str, str]]:
        """Messages as a list of dicts, ready to send to a provider."""
        return [m.to_dict() for m in self.messages]

    @property
    def last_user_message(self) -> str | None:
        """The last user message in the conversation."""
        for m in reversed(self.messages):
            if m.role == "user":
                return m.content
        return None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Conversation:
        messages = [Message.from_dict(m) for m in data["messages"]]
        return cls(
            messages=messages,
            ideal=data.get("ideal"),
            metadata=data.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# Format converters
# ---------------------------------------------------------------------------

_SHAREGPT_ROLE_MAP = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "system": "system",
    "chatgpt": "assistant",
    "bing": "assistant",
    "bard": "assistant",
}


def _detect_format(record: dict[str, Any]) -> Format:
    """Infer the format from a single record."""
    if "messages" in record:
        return "openai"
    if "conversations" in record:
        return "sharegpt"
    if "instruction" in record or "prompt" in record:
        return "alpaca"
    raise ValueError(
        "Could not detect dataset format. Expected one of: "
        "'messages' (OpenAI), 'conversations' (ShareGPT), or 'instruction' (Alpaca). "
        "Pass format= explicitly if your dataset uses a non-standard structure."
    )


def _convert_sharegpt(record: dict[str, Any]) -> dict[str, Any]:
    """Convert a ShareGPT record to OpenAI format.

    ShareGPT structure:
        {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}

    The last assistant turn becomes the `ideal` reference answer and is excluded
    from the prompt messages sent to the model.
    """
    turns = record.get("conversations", [])
    messages = []
    ideal = None

    for i, turn in enumerate(turns):
        role_raw = turn.get("from", turn.get("role", "")).lower()
        content = turn.get("value", turn.get("content", ""))
        role = _SHAREGPT_ROLE_MAP.get(role_raw)

        if role is None:
            continue  # Skip unknown roles rather than erroring

        # The final assistant turn is the reference answer, not a prompt message
        is_last = i == len(turns) - 1
        if role == "assistant" and is_last:
            ideal = content
        else:
            messages.append({"role": role, "content": content})

    return {
        "messages": messages,
        "ideal": ideal,
        "metadata": {k: v for k, v in record.items() if k != "conversations"},
    }


def _convert_alpaca(record: dict[str, Any]) -> dict[str, Any]:
    """Convert an Alpaca record to OpenAI format.

    Alpaca structure:
        {"instruction": "...", "input": "...", "output": "..."}

    `instruction` becomes the user message. `input` is appended to the instruction
    when present (the standard Alpaca template). `output` becomes the ideal answer.
    """
    instruction = record.get("instruction") or record.get("prompt", "")
    context = record.get("input", "").strip()
    output = record.get("output") or record.get("response", "")

    # Alpaca's standard template combines instruction + input
    if context:
        user_content = f"{instruction}\n\n{context}"
    else:
        user_content = instruction

    messages = [{"role": "user", "content": user_content}]

    # System prompt is occasionally included
    system = record.get("system", "").strip()
    if system:
        messages.insert(0, {"role": "system", "content": system})

    return {
        "messages": messages,
        "ideal": output or None,
        "metadata": {
            k: v for k, v in record.items()
            if k not in ("instruction", "input", "output", "prompt", "response", "system")
        },
    }


def _normalize(record: dict[str, Any], fmt: Format) -> dict[str, Any]:
    """Convert a raw record to OpenAI format based on the given format."""
    if fmt == "openai":
        return record
    if fmt == "sharegpt":
        return _convert_sharegpt(record)
    if fmt == "alpaca":
        return _convert_alpaca(record)
    raise ValueError(f"Unknown format: {fmt!r}")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class Dataset:
    """A collection of eval conversations.

    Supports OpenAI message format (native), ShareGPT, and Alpaca. Pass
    ``format="auto"`` (default) to detect the format automatically from the
    first record.
    """

    def __init__(self, conversations: list[Conversation], name: str = "unnamed"):
        self.conversations = conversations
        self.name = name

    def __len__(self) -> int:
        return len(self.conversations)

    def __getitem__(self, idx: int) -> Conversation:
        return self.conversations[idx]

    def __iter__(self):
        return iter(self.conversations)

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        name: str | None = None,
        format: Format = "auto",
    ) -> Dataset:
        """Load a dataset from a JSONL file.

        Args:
            path: Path to the JSONL file.
            name: Display name for the dataset. Defaults to the filename stem.
            format: Input format — "auto" (detect), "openai", "sharegpt", or "alpaca".
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        conversations = []
        detected_format: Format | None = None

        with open(path) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON on line {line_num}: {e}") from e

                # Detect format once from the first record
                if detected_format is None:
                    detected_format = (
                        _detect_format(data) if format == "auto" else format
                    )

                try:
                    normalized = _normalize(data, detected_format)
                except Exception as e:
                    raise ValueError(f"Failed to parse line {line_num}: {e}") from e

                if not normalized.get("messages"):
                    continue  # Skip empty conversations

                conversations.append(Conversation.from_dict(normalized))

        dataset_name = name or path.stem
        return cls(conversations=conversations, name=dataset_name)

    @classmethod
    def from_list(
        cls,
        items: list[dict[str, Any]],
        name: str = "inline",
        format: Format = "auto",
    ) -> Dataset:
        """Create a dataset from a list of dicts.

        Args:
            items: List of records in OpenAI, ShareGPT, or Alpaca format.
            name: Display name for the dataset.
            format: Input format — "auto" (detect), "openai", "sharegpt", or "alpaca".
        """
        detected_format: Format | None = None
        conversations = []
        for item in items:
            if detected_format is None:
                detected_format = (
                    _detect_format(item) if format == "auto" else format
                )
            normalized = _normalize(item, detected_format)
            if normalized.get("messages"):
                conversations.append(Conversation.from_dict(normalized))
        return cls(conversations=conversations, name=name)

    def filter(self, **kwargs: Any) -> Dataset:
        """Return a new dataset filtered by metadata fields.

        Example: dataset.filter(category="reasoning", difficulty="hard")
        """
        filtered = [
            c for c in self.conversations
            if all(c.metadata.get(k) == v for k, v in kwargs.items())
        ]
        return Dataset(conversations=filtered, name=f"{self.name}_filtered")

    def has_ideals(self) -> bool:
        """Return True if all conversations have a reference (ideal) answer."""
        return all(c.ideal is not None for c in self.conversations)

    def summary(self) -> dict[str, Any]:
        """Return a summary of the dataset."""
        return {
            "name": self.name,
            "num_conversations": len(self.conversations),
            "has_ideals": self.has_ideals(),
            "metadata_keys": sorted(
                {k for c in self.conversations for k in c.metadata}
            ),
        }
