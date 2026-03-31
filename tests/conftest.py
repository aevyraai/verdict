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

"""Shared fixtures and pytest configuration for aevyra-verdict tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Custom markers
# ---------------------------------------------------------------------------

def pytest_configure(config):
    config.addinivalue_line("markers", "ollama: requires a running Ollama server")
    config.addinivalue_line("markers", "vllm: requires a running vLLM server")
    config.addinivalue_line("markers", "openrouter: requires OPENROUTER_API_KEY")
    config.addinivalue_line("markers", "openai: requires OPENAI_API_KEY")


# ---------------------------------------------------------------------------
# Skip helpers — tests are skipped automatically if the service/key is absent
# ---------------------------------------------------------------------------

def _ollama_available() -> bool:
    import urllib.request
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        return True
    except Exception:
        return False


def _vllm_available() -> bool:
    import urllib.request
    try:
        urllib.request.urlopen("http://localhost:8000/health", timeout=2)
        return True
    except Exception:
        return False


skip_if_no_ollama = pytest.mark.skipif(
    not _ollama_available(),
    reason="Ollama not running at localhost:11434",
)

skip_if_no_vllm = pytest.mark.skipif(
    not _vllm_available(),
    reason="vLLM not running at localhost:8000",
)

skip_if_no_openrouter = pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set",
)

skip_if_no_openai = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

GSM8K_PATH = Path(__file__).parent.parent / "examples" / "gsm8k_sample.jsonl"


@pytest.fixture
def gsm8k_dataset(tmp_path):
    """First 5 samples from GSM8K — used for integration tests."""
    import json
    if not GSM8K_PATH.exists():
        pytest.skip("gsm8k_sample.jsonl not present — run: python examples/fetch_gsm8k.py")
    lines = GSM8K_PATH.read_text().splitlines()[:5]
    path = tmp_path / "gsm8k_5.jsonl"
    path.write_text("\n".join(lines))
    return path


@pytest.fixture
def tiny_dataset(tmp_path):
    """A minimal 2-sample dataset written to a temp JSONL file."""
    import json

    samples = [
        {
            "messages": [{"role": "user", "content": "Say the word 'hello' and nothing else."}],
            "ideal": "hello",
        },
        {
            "messages": [{"role": "user", "content": "What is 1 + 1? Reply with just the number."}],
            "ideal": "2",
        },
    ]
    path = tmp_path / "tiny.jsonl"
    path.write_text("\n".join(json.dumps(s) for s in samples))
    return path


@pytest.fixture
def tiny_dataset_sharegpt(tmp_path):
    """A minimal ShareGPT-format dataset."""
    import json

    samples = [
        {
            "conversations": [
                {"from": "human", "value": "Say the word 'hello' and nothing else."},
                {"from": "gpt", "value": "hello"},
            ]
        }
    ]
    path = tmp_path / "tiny_sharegpt.jsonl"
    path.write_text("\n".join(json.dumps(s) for s in samples))
    return path


@pytest.fixture
def tiny_dataset_alpaca(tmp_path):
    """A minimal Alpaca-format dataset."""
    import json

    samples = [
        {
            "instruction": "Reply with just the number.",
            "input": "What is 1 + 1?",
            "output": "2",
        }
    ]
    path = tmp_path / "tiny_alpaca.jsonl"
    path.write_text("\n".join(json.dumps(s) for s in samples))
    return path
