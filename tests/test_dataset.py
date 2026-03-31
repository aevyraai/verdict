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

"""Unit tests for Dataset — no API calls, runs fully offline."""

from __future__ import annotations

import json

import pytest

from pathlib import Path

from aevyra_verdict.dataset import Dataset

GSM8K_PATH = Path(__file__).parent.parent / "examples" / "gsm8k_sample.jsonl"


# ---------------------------------------------------------------------------
# OpenAI format
# ---------------------------------------------------------------------------

class TestOpenAIFormat:
    def test_from_jsonl(self, tiny_dataset):
        ds = Dataset.from_jsonl(tiny_dataset)
        assert len(ds) == 2

    def test_has_ideals(self, tiny_dataset):
        ds = Dataset.from_jsonl(tiny_dataset)
        assert ds.has_ideals()

    def test_messages_structure(self, tiny_dataset):
        ds = Dataset.from_jsonl(tiny_dataset)
        sample = ds[0]
        assert sample.messages[0].role == "user"
        assert "hello" in sample.messages[0].content.lower()

    def test_ideal_preserved(self, tiny_dataset):
        ds = Dataset.from_jsonl(tiny_dataset)
        assert ds[0].ideal == "hello"
        assert ds[1].ideal == "2"

    def test_from_list(self):
        items = [
            {"messages": [{"role": "user", "content": "Hi"}], "ideal": "Hello"},
        ]
        ds = Dataset.from_list(items)
        assert len(ds) == 1
        assert ds[0].ideal == "Hello"

    def test_filter_by_metadata(self, tmp_path):
        samples = [
            {"messages": [{"role": "user", "content": "Q1"}], "metadata": {"category": "math"}},
            {"messages": [{"role": "user", "content": "Q2"}], "metadata": {"category": "science"}},
            {"messages": [{"role": "user", "content": "Q3"}], "metadata": {"category": "math"}},
        ]
        path = tmp_path / "meta.jsonl"
        path.write_text("\n".join(json.dumps(s) for s in samples))

        ds = Dataset.from_jsonl(path)
        math_ds = ds.filter(category="math")
        assert len(math_ds) == 2

    def test_summary(self, tiny_dataset):
        ds = Dataset.from_jsonl(tiny_dataset)
        summary = ds.summary()
        assert summary["num_conversations"] == 2
        assert summary["has_ideals"] is True


# ---------------------------------------------------------------------------
# ShareGPT format
# ---------------------------------------------------------------------------

class TestShareGPTFormat:
    def test_auto_detect(self, tiny_dataset_sharegpt):
        ds = Dataset.from_jsonl(tiny_dataset_sharegpt)
        assert len(ds) == 1

    def test_explicit_format(self, tiny_dataset_sharegpt):
        ds = Dataset.from_jsonl(tiny_dataset_sharegpt, format="sharegpt")
        assert len(ds) == 1

    def test_last_assistant_turn_becomes_ideal(self, tiny_dataset_sharegpt):
        ds = Dataset.from_jsonl(tiny_dataset_sharegpt)
        assert ds[0].ideal == "hello"

    def test_human_turn_in_messages(self, tiny_dataset_sharegpt):
        ds = Dataset.from_jsonl(tiny_dataset_sharegpt)
        sample = ds[0]
        assert any(m.role == "user" for m in sample.messages)

    def test_assistant_turn_excluded_from_messages(self, tiny_dataset_sharegpt):
        ds = Dataset.from_jsonl(tiny_dataset_sharegpt)
        sample = ds[0]
        # The last gpt turn should be ideal, not in messages
        assert not any(m.role == "assistant" for m in sample.messages)

    def test_from_list_sharegpt(self):
        items = [
            {
                "conversations": [
                    {"from": "human", "value": "Hi"},
                    {"from": "gpt", "value": "Hello!"},
                ]
            }
        ]
        ds = Dataset.from_list(items, format="sharegpt")
        assert len(ds) == 1
        assert ds[0].ideal == "Hello!"


# ---------------------------------------------------------------------------
# GSM8K sample dataset
# ---------------------------------------------------------------------------

class TestGSM8KSample:
    def test_file_exists(self):
        assert GSM8K_PATH.exists(), (
            f"GSM8K sample not found at {GSM8K_PATH}. "
            "Run: python examples/fetch_gsm8k.py"
        )

    def test_loads_50_samples(self):
        if not GSM8K_PATH.exists():
            pytest.skip("gsm8k_sample.jsonl not present")
        ds = Dataset.from_jsonl(GSM8K_PATH)
        assert len(ds) == 50

    def test_all_have_ideals(self):
        if not GSM8K_PATH.exists():
            pytest.skip("gsm8k_sample.jsonl not present")
        ds = Dataset.from_jsonl(GSM8K_PATH)
        assert ds.has_ideals()
        assert all(s.ideal for s in ds)

    def test_all_have_system_prompt(self):
        if not GSM8K_PATH.exists():
            pytest.skip("gsm8k_sample.jsonl not present")
        ds = Dataset.from_jsonl(GSM8K_PATH)
        for sample in ds:
            assert sample.messages[0].role == "system"

    def test_metadata_source_is_gsm8k(self):
        if not GSM8K_PATH.exists():
            pytest.skip("gsm8k_sample.jsonl not present")
        ds = Dataset.from_jsonl(GSM8K_PATH)
        assert all(s.metadata.get("source") == "gsm8k" for s in ds)

    def test_ideals_are_numbers(self):
        if not GSM8K_PATH.exists():
            pytest.skip("gsm8k_sample.jsonl not present")
        ds = Dataset.from_jsonl(GSM8K_PATH)
        for sample in ds:
            # GSM8K ideals should be numeric strings
            assert sample.ideal.replace(",", "").replace(".", "").lstrip("-").isdigit(), \
                f"Non-numeric ideal: {sample.ideal!r}"


# ---------------------------------------------------------------------------
# Alpaca format
# ---------------------------------------------------------------------------

class TestAlpacaFormat:
    def test_auto_detect(self, tiny_dataset_alpaca):
        ds = Dataset.from_jsonl(tiny_dataset_alpaca)
        assert len(ds) == 1

    def test_explicit_format(self, tiny_dataset_alpaca):
        ds = Dataset.from_jsonl(tiny_dataset_alpaca, format="alpaca")
        assert len(ds) == 1

    def test_output_becomes_ideal(self, tiny_dataset_alpaca):
        ds = Dataset.from_jsonl(tiny_dataset_alpaca)
        assert ds[0].ideal == "2"

    def test_instruction_and_input_combined(self, tiny_dataset_alpaca):
        ds = Dataset.from_jsonl(tiny_dataset_alpaca)
        sample = ds[0]
        content = sample.messages[-1].content
        assert "Reply with just the number" in content
        assert "What is 1 + 1?" in content

    def test_no_input_field(self, tmp_path):
        samples = [{"instruction": "Translate to French: Hello", "output": "Bonjour"}]
        path = tmp_path / "alpaca_no_input.jsonl"
        path.write_text(json.dumps(samples[0]))

        ds = Dataset.from_jsonl(path, format="alpaca")
        assert ds[0].ideal == "Bonjour"
        assert "Translate to French" in ds[0].messages[-1].content

    def test_system_field(self, tmp_path):
        samples = [
            {
                "instruction": "Say hi",
                "input": "",
                "output": "Hi!",
                "system": "You are a friendly bot.",
            }
        ]
        path = tmp_path / "alpaca_sys.jsonl"
        path.write_text(json.dumps(samples[0]))

        ds = Dataset.from_jsonl(path, format="alpaca")
        sample = ds[0]
        assert sample.messages[0].role == "system"
        assert "friendly bot" in sample.messages[0].content

    def test_from_list_alpaca(self):
        items = [{"instruction": "Say hello", "output": "Hello!"}]
        ds = Dataset.from_list(items, format="alpaca")
        assert len(ds) == 1
        assert ds[0].ideal == "Hello!"
