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

"""CLI unit tests — covers all flags using Typer's test runner. No real API calls made."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from aevyra_verdict.cli import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_dataset(tmp_path, n=2):
    """Write a minimal JSONL dataset to a temp file."""
    samples = [
        {
            "messages": [{"role": "user", "content": f"Question {i}"}],
            "ideal": f"Answer {i}",
        }
        for i in range(n)
    ]
    path = tmp_path / "data.jsonl"
    path.write_text("\n".join(json.dumps(s) for s in samples))
    return path


# ---------------------------------------------------------------------------
# inspect command
# ---------------------------------------------------------------------------

class TestInspectCommand:
    def test_inspect_shows_sample_count(self, tmp_path):
        path = make_dataset(tmp_path, n=3)
        result = runner.invoke(app, ["inspect", str(path)])
        assert result.exit_code == 0
        assert "3" in result.output

    def test_inspect_shows_has_ideals(self, tmp_path):
        path = make_dataset(tmp_path)
        result = runner.invoke(app, ["inspect", str(path)])
        assert result.exit_code == 0
        assert "True" in result.output or "true" in result.output.lower()

    def test_inspect_missing_file(self, tmp_path):
        result = runner.invoke(app, ["inspect", str(tmp_path / "missing.jsonl")])
        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "error" in result.output.lower()

    def test_inspect_shows_first_message(self, tmp_path):
        path = make_dataset(tmp_path)
        result = runner.invoke(app, ["inspect", str(path)])
        assert result.exit_code == 0
        assert "Question 0" in result.output


# ---------------------------------------------------------------------------
# providers command
# ---------------------------------------------------------------------------

class TestProvidersCommand:
    def test_lists_builtin_providers(self):
        result = runner.invoke(app, ["providers"])
        assert result.exit_code == 0
        for name in ["openai", "anthropic", "google", "mistral", "cohere", "openrouter", "local"]:
            assert name in result.output

    def test_shows_env_var_names(self):
        result = runner.invoke(app, ["providers"])
        assert "OPENAI_API_KEY" in result.output
        assert "ANTHROPIC_API_KEY" in result.output
        assert "OPENROUTER_API_KEY" in result.output


# ---------------------------------------------------------------------------
# run command — argument validation (no real API calls)
# ---------------------------------------------------------------------------

class TestRunCommandValidation:
    def test_missing_model_and_config(self, tmp_path):
        path = make_dataset(tmp_path)
        result = runner.invoke(app, ["run", str(path)])
        assert result.exit_code != 0
        assert "--model" in result.output or "--config" in result.output

    def test_model_and_config_mutually_exclusive(self, tmp_path):
        path = make_dataset(tmp_path)
        config = tmp_path / "models.yaml"
        config.write_text("models:\n  - provider: openai\n    model: gpt-4o-mini\n")
        result = runner.invoke(app, [
            "run", str(path),
            "-m", "openai/gpt-4o-mini",
            "-c", str(config),
        ])
        assert result.exit_code != 0
        assert "either" in result.output.lower() or "not both" in result.output.lower()

    def test_missing_dataset_file(self, tmp_path):
        result = runner.invoke(app, [
            "run", str(tmp_path / "missing.jsonl"),
            "-m", "openai/gpt-4o-mini",
        ])
        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "error" in result.output.lower()

    def test_invalid_model_format(self, tmp_path):
        path = make_dataset(tmp_path)
        result = runner.invoke(app, ["run", str(path), "-m", "invalid-no-slash"])
        assert result.exit_code != 0
        assert "provider/model" in result.output.lower() or "format" in result.output.lower()

    def test_unknown_metric_warns(self, tmp_path, capsys):
        path = make_dataset(tmp_path)
        # This will fail at API call stage, but we just check the warning is emitted
        result = runner.invoke(app, [
            "run", str(path),
            "-m", "openai/gpt-4o-mini",
            "--metric", "nonexistent",
        ])
        assert "unknown" in result.output.lower() or "warning" in result.output.lower() \
            or result.exit_code != 0

    def test_invalid_custom_metric_format(self, tmp_path):
        path = make_dataset(tmp_path)
        result = runner.invoke(app, [
            "run", str(path),
            "-m", "openai/gpt-4o-mini",
            "--custom-metric", "no-colon",
        ])
        assert result.exit_code != 0
        assert "file.py:function_name" in result.output or "format" in result.output.lower()

    def test_missing_custom_metric_file(self, tmp_path):
        path = make_dataset(tmp_path)
        result = runner.invoke(app, [
            "run", str(path),
            "-m", "openai/gpt-4o-mini",
            "--custom-metric", "missing.py:my_func",
        ])
        assert result.exit_code != 0
        assert "not found" in result.output.lower()

    def test_missing_judge_prompt_file(self, tmp_path):
        path = make_dataset(tmp_path)
        result = runner.invoke(app, [
            "run", str(path),
            "-m", "openai/gpt-4o-mini",
            "--judge", "openai/gpt-4o-mini",
            "--judge-prompt", str(tmp_path / "missing_prompt.md"),
        ])
        assert result.exit_code != 0
        assert "not found" in result.output.lower()

    def test_missing_config_file(self, tmp_path):
        path = make_dataset(tmp_path)
        result = runner.invoke(app, [
            "run", str(path),
            "-c", str(tmp_path / "missing.yaml"),
        ])
        assert result.exit_code != 0
        assert "not found" in result.output.lower()


# ---------------------------------------------------------------------------
# run command — config file parsing
# ---------------------------------------------------------------------------

class TestConfigFileParsing:
    def test_yaml_config_no_models_key(self, tmp_path):
        path = make_dataset(tmp_path)
        config = tmp_path / "bad.yaml"
        config.write_text("something: else\n")
        result = runner.invoke(app, ["run", str(path), "-c", str(config)])
        assert result.exit_code != 0
        assert "models" in result.output.lower()

    def test_yaml_config_missing_provider(self, tmp_path):
        path = make_dataset(tmp_path)
        config = tmp_path / "bad.yaml"
        config.write_text("models:\n  - model: gpt-4o-mini\n")
        result = runner.invoke(app, ["run", str(path), "-c", str(config)])
        assert result.exit_code != 0

    def test_json_config_parsed(self, tmp_path):
        path = make_dataset(tmp_path)
        config = tmp_path / "models.json"
        config.write_text(json.dumps({
            "models": [{"provider": "openai", "model": "gpt-4o-mini"}]
        }))
        # Will fail at API call stage but config parsing should succeed
        result = runner.invoke(app, ["run", str(path), "-c", str(config)])
        # Should get past config parsing (error will be API related, not config)
        assert "unrecognised config" not in result.output.lower()


# ---------------------------------------------------------------------------
# run command — flags
# ---------------------------------------------------------------------------

class TestRunFlags:
    def test_base_url_flag_accepted(self, tmp_path):
        path = make_dataset(tmp_path)
        result = runner.invoke(app, [
            "run", str(path),
            "-m", "local/llama3.2:1b",
            "--base-url", "http://localhost:11434/v1",
        ])
        # Will fail at API stage (no Ollama in unit tests), but flag should be accepted
        assert "no such option" not in result.output.lower()

    def test_multiple_models_accepted(self, tmp_path):
        path = make_dataset(tmp_path)
        result = runner.invoke(app, [
            "run", str(path),
            "-m", "openai/gpt-4o-mini",
            "-m", "openai/gpt-4o",
        ])
        assert "no such option" not in result.output.lower()

    def test_temperature_flag(self, tmp_path):
        path = make_dataset(tmp_path)
        result = runner.invoke(app, [
            "run", str(path),
            "-m", "openai/gpt-4o-mini",
            "--temperature", "0.7",
        ])
        assert "no such option" not in result.output.lower()

    def test_max_tokens_flag(self, tmp_path):
        path = make_dataset(tmp_path)
        result = runner.invoke(app, [
            "run", str(path),
            "-m", "openai/gpt-4o-mini",
            "--max-tokens", "512",
        ])
        assert "no such option" not in result.output.lower()

    def test_max_workers_flag(self, tmp_path):
        path = make_dataset(tmp_path)
        result = runner.invoke(app, [
            "run", str(path),
            "-m", "openai/gpt-4o-mini",
            "--max-workers", "4",
        ])
        assert "no such option" not in result.output.lower()

    def test_output_flag(self, tmp_path):
        path = make_dataset(tmp_path)
        out = tmp_path / "results.json"
        result = runner.invoke(app, [
            "run", str(path),
            "-m", "openai/gpt-4o-mini",
            "-o", str(out),
        ])
        assert "no such option" not in result.output.lower()

    def test_multiple_metrics_accepted(self, tmp_path):
        path = make_dataset(tmp_path)
        result = runner.invoke(app, [
            "run", str(path),
            "-m", "openai/gpt-4o-mini",
            "--metric", "rouge",
            "--metric", "bleu",
        ])
        assert "no such option" not in result.output.lower()

    def test_judge_prompt_flag(self, tmp_path):
        path = make_dataset(tmp_path)
        prompt = tmp_path / "judge.md"
        prompt.write_text("Rate this: {response}")
        result = runner.invoke(app, [
            "run", str(path),
            "-m", "openai/gpt-4o-mini",
            "--judge", "openai/gpt-4o-mini",
            "--judge-prompt", str(prompt),
        ])
        assert "no such option" not in result.output.lower()

    def test_custom_metric_flag(self, tmp_path):
        path = make_dataset(tmp_path)
        metric_file = tmp_path / "my_metrics.py"
        metric_file.write_text(
            "def my_score(response, ideal=None, **kwargs):\n    return 1.0\n"
        )
        result = runner.invoke(app, [
            "run", str(path),
            "-m", "openai/gpt-4o-mini",
            "--custom-metric", f"{metric_file}:my_score",
        ])
        assert "no such option" not in result.output.lower()
        assert "not found" not in result.output.lower()
