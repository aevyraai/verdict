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

"""aevyra-verdict CLI — run and compare LLM evals from the terminal."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Annotated, Any, Optional

try:
    import typer
except ImportError:
    print("typer is required for the CLI. Install it with: pip install typer")
    sys.exit(1)

app = typer.Typer(
    name="aevyra-verdict",
    help="Compare LLM outputs across models and providers.",
    no_args_is_help=True,
)


# ---------------------------------------------------------------------------
# Config file loading
# ---------------------------------------------------------------------------

def _load_config(path: Path) -> dict[str, Any]:
    """Load a models config file. Supports JSON, YAML, and TOML."""
    suffix = path.suffix.lower()

    if suffix == ".json":
        return json.loads(path.read_text())

    if suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError:
            typer.echo(
                "[error] PyYAML is needed to read .yaml config files.\n"
                "  Fix: pip install pyyaml",
                err=True,
            )
            raise typer.Exit(code=1)
        return yaml.safe_load(path.read_text())

    if suffix == ".toml":
        try:
            import tomllib  # Python 3.11+
        except ImportError:
            try:
                import tomli as tomllib  # backport for 3.10
            except ImportError:
                typer.echo(
                    "[error] tomli is needed to read .toml config files on Python < 3.11.\n"
                    "  Fix: pip install tomli",
                    err=True,
                )
                raise typer.Exit(code=1)
        return tomllib.loads(path.read_text())

    typer.echo(
        f"[error] Unrecognised config format {suffix!r}. Use .json, .yaml, or .toml.",
        err=True,
    )
    raise typer.Exit(code=1)


def _models_from_config(path: Path) -> list[dict[str, Any]]:
    """Parse a config file and return a list of model dicts.

    Each dict must have 'provider' and 'model'. Optional keys:
    'label', 'api_key', 'base_url'.
    """
    raw = _load_config(path)
    models = raw.get("models", [])
    if not models:
        typer.echo(f"[error] Config file {path} has no 'models' list.", err=True)
        raise typer.Exit(code=1)
    for i, m in enumerate(models):
        if "provider" not in m or "model" not in m:
            typer.echo(
                f"[error] Model entry {i} in {path} is missing 'provider' or 'model'.",
                err=True,
            )
            raise typer.Exit(code=1)
    return models


# ---------------------------------------------------------------------------
# Model flag parsing
# ---------------------------------------------------------------------------

def _parse_model_flag(spec: str) -> dict[str, str]:
    """Parse 'provider/model' flag into a model dict."""
    if "/" not in spec:
        typer.echo(
            f"[error] Model spec {spec!r} must be in 'provider/model' format.\n"
            f"  Examples: openai/gpt-5.4-nano   anthropic/claude-sonnet-4-6   google/gemini-2.0-flash",
            err=True,
        )
        raise typer.Exit(code=1)
    provider, _, model_id = spec.partition("/")
    return {"provider": provider.strip(), "model": model_id.strip()}


def _check_api_key(provider: str, base_url: str | None = None) -> None:
    """Warn if the expected API key env var is missing (skip for local endpoints)."""
    if base_url and ("localhost" in base_url or "127.0.0.1" in base_url):
        return  # Local vLLM / Ollama — no key needed
    key_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "cohere": "COHERE_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }
    env_var = key_map.get(provider.lower())
    if env_var and not os.environ.get(env_var):
        typer.echo(
            f"[warning] {env_var} is not set. Calls to {provider!r} will fail.\n"
            f"  Fix: export {env_var}=your-key",
            err=True,
        )


# ---------------------------------------------------------------------------
# Custom metric loading
# ---------------------------------------------------------------------------

def _load_custom_metric(spec: str):
    """Load a custom scoring function from 'path/to/file.py:function_name'.

    The function must accept (response, ideal=None, messages=None, **kwargs)
    and return a float (0.0–1.0) or a dict with a 'score' key.
    """
    if ":" not in spec:
        typer.echo(
            f"[error] --custom-metric must be in 'file.py:function_name' format.\n"
            f"  Example: --custom-metric my_metrics.py:brevity_score",
            err=True,
        )
        raise typer.Exit(code=1)

    file_part, _, fn_name = spec.rpartition(":")
    file_path = Path(file_part)

    if not file_path.exists():
        typer.echo(f"[error] Custom metric file not found: {file_path}", err=True)
        raise typer.Exit(code=1)

    module_spec = importlib.util.spec_from_file_location("_custom_metrics", file_path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    if not hasattr(module, fn_name):
        typer.echo(
            f"[error] Function {fn_name!r} not found in {file_path}.\n"
            f"  Available: {[n for n in dir(module) if not n.startswith('_')]}",
            err=True,
        )
        raise typer.Exit(code=1)

    return fn_name, getattr(module, fn_name)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.command()
def run(
    dataset: Annotated[Path, typer.Argument(help="Path to a JSONL dataset file.")],
    model: Annotated[
        list[str],
        typer.Option(
            "-m", "--model",
            help="Model in 'provider/model' format. Repeat for multiple. "
                 "Cannot be combined with --config.",
        ),
    ] = [],
    config: Annotated[
        Optional[Path],
        typer.Option(
            "-c", "--config",
            help="Path to a models config file (.yaml, .json, or .toml). "
                 "Cannot be combined with --model.",
        ),
    ] = None,
    metric: Annotated[
        list[str],
        typer.Option(
            "--metric",
            help="Built-in metric to use: rouge, bleu, exact. Repeat for multiple.",
        ),
    ] = ["rouge"],
    judge: Annotated[
        Optional[str],
        typer.Option(
            "--judge",
            help="Use LLM-as-judge with this model spec (e.g. openai/gpt-5.4).",
        ),
    ] = None,
    judge_prompt: Annotated[
        Optional[Path],
        typer.Option(
            "--judge-prompt",
            help="Path to a prompt template file (.md or .txt). "
                 "Use {criteria}, {conversation}, {response}, {ideal_section} as placeholders.",
        ),
    ] = None,
    custom_metric: Annotated[
        list[str],
        typer.Option(
            "--custom-metric",
            help="Custom scoring function in 'file.py:function_name' format. "
                 "Repeat for multiple.",
        ),
    ] = [],
    output: Annotated[
        Optional[Path],
        typer.Option("-o", "--output", help="Save results as JSON to this path."),
    ] = None,
    max_workers: Annotated[
        int,
        typer.Option(help="Concurrent requests per model. Lower if hitting rate limits."),
    ] = 10,
    temperature: Annotated[
        float,
        typer.Option(help="Sampling temperature (0.0 = deterministic)."),
    ] = 0.0,
    max_tokens: Annotated[
        int,
        typer.Option(help="Max tokens per completion."),
    ] = 1024,
    base_url: Annotated[
        Optional[str],
        typer.Option(
            "--base-url",
            help="Custom base URL for all --model entries (e.g. http://localhost:11434/v1 for Ollama).",
        ),
    ] = None,
) -> None:
    """Run evals on a JSONL dataset across one or more models and print a comparison.

    \b
    Examples:
      # Inline models
      aevyra-verdict run data.jsonl -m openai/gpt-5.4-nano -m qwen/qwen3.5-9b

      # Local Ollama instance
      aevyra-verdict run data.jsonl -m local/llama3.2:1b --base-url http://localhost:11434/v1

      # From a config file (supports local vLLM, multiple providers)
      aevyra-verdict run data.jsonl --config models.yaml

      # Custom judge prompt
      aevyra-verdict run data.jsonl -m openai/gpt-5.4-nano --judge openai/gpt-5.4 --judge-prompt prompt.txt

      # Custom scoring function
      aevyra-verdict run data.jsonl -m openai/gpt-5.4-nano --custom-metric my_metrics.py:brevity_score

      # Save results
      aevyra-verdict run data.jsonl --config models.yaml --metric rouge --judge openai/gpt-5.4 -o results.json
    """
    from aevyra_verdict.dataset import Dataset
    from aevyra_verdict.runner import EvalRunner, RunConfig
    from aevyra_verdict.metrics import RougeScore, BleuScore, ExactMatch, LLMJudge, CustomMetric
    from aevyra_verdict.providers import get_provider

    # --- Validate all inputs upfront before any provider instantiation ---
    if not dataset.exists():
        typer.echo(f"[error] Dataset not found: {dataset}", err=True)
        raise typer.Exit(code=1)

    if config and model:
        typer.echo("[error] Use either --config or --model, not both.", err=True)
        raise typer.Exit(code=1)

    if not config and not model:
        typer.echo(
            "[error] Specify models via --model or --config.\n"
            "  Examples:\n"
            "    --model openai/gpt-5.4-nano\n"
            "    --config models.yaml",
            err=True,
        )
        raise typer.Exit(code=1)

    if judge_prompt and not judge_prompt.exists():
        typer.echo(f"[error] Judge prompt file not found: {judge_prompt}", err=True)
        raise typer.Exit(code=1)

    for cm_spec in custom_metric:
        if ":" not in cm_spec:
            typer.echo(
                f"[error] --custom-metric must be in 'file.py:function_name' format.\n"
                f"  Example: --custom-metric my_metrics.py:brevity_score",
                err=True,
            )
            raise typer.Exit(code=1)
        file_part = cm_spec.rpartition(":")[0]
        if not Path(file_part).exists():
            typer.echo(f"[error] Custom metric file not found: {file_part}", err=True)
            raise typer.Exit(code=1)

    # --- Resolve model list ---
    model_defs: list[dict[str, Any]] = []
    if config:
        if not config.exists():
            typer.echo(f"[error] Config file not found: {config}", err=True)
            raise typer.Exit(code=1)
        model_defs = _models_from_config(config)
    else:
        model_defs = [_parse_model_flag(m) for m in model]
        if base_url:
            for m in model_defs:
                m.setdefault("base_url", base_url)

    for m in model_defs:
        _check_api_key(m["provider"], m.get("base_url"))

    # --- Load dataset ---
    typer.echo(f"Loading dataset: {dataset}")
    ds = Dataset.from_jsonl(dataset)
    typer.echo(f"  {len(ds)} samples | has_ideals={ds.has_ideals()}")

    # --- Set up runner ---
    config_obj = RunConfig(
        temperature=temperature,
        max_tokens=max_tokens,
        max_workers=max_workers,
    )
    runner = EvalRunner(config=config_obj)

    for m in model_defs:
        label = m.get("label") or f"{m['provider']}/{m['model']}"
        runner.add_provider(
            m["provider"],
            m["model"],
            label=label,
            api_key=m.get("api_key"),
            base_url=m.get("base_url"),
        )

    # --- Add built-in metrics ---
    valid_metrics = {"rouge", "bleu", "exact"}
    for m_name in metric:
        m_lower = m_name.lower()
        if m_lower == "rouge":
            runner.add_metric(RougeScore())
        elif m_lower == "bleu":
            runner.add_metric(BleuScore())
        elif m_lower == "exact":
            runner.add_metric(ExactMatch())
        else:
            typer.echo(
                f"[warning] Unknown metric {m_name!r}. Valid choices: {', '.join(sorted(valid_metrics))}",
                err=True,
            )

    # --- Add LLM-as-judge ---
    if judge:
        judge_def = _parse_model_flag(judge)
        _check_api_key(judge_def["provider"])
        judge_provider = get_provider(judge_def["provider"], judge_def["model"])

        prompt_template = None
        if judge_prompt:
            prompt_template = judge_prompt.read_text()
            typer.echo(f"Using custom judge prompt from: {judge_prompt}")

        runner.add_metric(LLMJudge(
            judge_provider=judge_provider,
            prompt_template=prompt_template,
        ))

    # --- Add custom Python metrics ---
    for cm_spec in custom_metric:
        fn_name, fn = _load_custom_metric(cm_spec)
        runner.add_metric(CustomMetric(fn_name, fn))
        typer.echo(f"Loaded custom metric: {fn_name}")

    if not runner.metrics:
        typer.echo("[error] No valid metrics configured.", err=True)
        raise typer.Exit(code=1)

    # --- Run ---
    model_labels = [m.get("label") or f"{m['provider']}/{m['model']}" for m in model_defs]
    typer.echo(
        f"\nRunning {len(model_defs)} model(s) × {len(ds)} samples "
        f"({max_workers} concurrent requests per model)...\n"
        f"  Models: {', '.join(model_labels)}\n"
    )
    results = runner.run(ds, show_progress=True)

    # --- Print comparison ---
    typer.echo("")
    for metric_name in results.metric_names:
        typer.echo(results.compare(metric_name))
        typer.echo("")

    # --- Save ---
    if output:
        results.to_json(output)
        typer.echo(f"Results saved to {output}")


@app.command()
def inspect(
    dataset: Annotated[Path, typer.Argument(help="Path to a JSONL dataset file.")],
) -> None:
    """Show a summary of a dataset without running any models."""
    from aevyra_verdict.dataset import Dataset

    if not dataset.exists():
        typer.echo(f"[error] Dataset not found: {dataset}", err=True)
        raise typer.Exit(code=1)

    ds = Dataset.from_jsonl(dataset)
    summary = ds.summary()

    typer.echo(f"\nDataset: {summary['name']}")
    typer.echo(f"  Samples:    {summary['num_conversations']}")
    typer.echo(f"  Has ideals: {summary['has_ideals']}")
    if summary["metadata_keys"]:
        typer.echo(f"  Metadata:   {', '.join(summary['metadata_keys'])}")

    typer.echo("\nFirst sample:")
    first = ds[0]
    for msg in first.messages:
        preview = msg.content[:120].replace("\n", " ")
        typer.echo(f"  [{msg.role}] {preview}{'...' if len(msg.content) > 120 else ''}")
    if first.ideal:
        preview = first.ideal[:120].replace("\n", " ")
        typer.echo(f"  [ideal] {preview}{'...' if len(first.ideal) > 120 else ''}")


@app.command()
def providers() -> None:
    """List all available providers and the env vars needed for each."""
    from aevyra_verdict.providers import list_providers

    key_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "cohere": "COHERE_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "local": "(none — local endpoint)",
    }

    typer.echo("\nAvailable providers:\n")
    for name in list_providers():
        env_var = key_map.get(name, "—")
        if name == "local":
            typer.echo(f"  {name:<12} {env_var}")
        else:
            is_set = "✓ set" if os.environ.get(env_var) else "✗ not set"
            typer.echo(f"  {name:<12} {env_var:<22} {is_set}")
    typer.echo(
        "\nTip: use 'local' provider with --base-url for Ollama or vLLM:\n"
        "  # Ollama (default port)\n"
        "  aevyra-verdict run data.jsonl -m local/llama3.2:1b --base-url http://localhost:11434/v1\n"
        "\n"
        "  # vLLM\n"
        "  aevyra-verdict run data.jsonl -m local/meta-llama/Llama-3.2-1B --base-url http://localhost:8000/v1\n"
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
