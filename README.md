# aevyra-verdict

[![CI](https://github.com/aevyraai/verdict/actions/workflows/ci.yml/badge.svg)](https://github.com/aevyraai/verdict/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/aevyra-verdict.svg)](https://pypi.org/project/aevyra-verdict/)

Benchmark any LLM against your data. Pick the best model, then make it better.

verdict runs your prompts across any combination of models, scores the responses with
pluggable metrics, and gives you a side-by-side comparison — so you can choose the right
model for your task, then track whether your prompt engineering or fine-tuning is actually
moving the needle.

## Use cases

**Choosing the right model.** Instead of guessing, run your actual prompts across GPT-5.4-mini,
Claude Sonnet, Gemini, Llama — and pick the one that scores highest on your specific task.

**Measuring improvement.** Establish a baseline score, tweak your system prompt or
fine-tune your model, re-run verdict. If the number goes up, your change helped. If it
doesn't, you know to try something else.

**Benchmarking open-source vs closed models.** Measure how a local model stacks up
against SOTA closed models on your workload — and identify exactly where the gap is.

## Install

```bash
pip install aevyra-verdict
```

This pulls in the SDKs for OpenAI, Anthropic, Google (Gemini), Mistral, and Cohere.
You only need API keys for the providers you actually use.

## Quick start

```bash
# 1. Check which API keys are configured
aevyra-verdict providers

# 2. Compare models on a dataset and save results
aevyra-verdict run dataset.jsonl \
  -m openai/gpt-5.4-mini \
  -m anthropic/claude-sonnet-4-6 \
  -o results.json
```

Or use the Python API directly:

```python
from aevyra_verdict import Dataset, EvalRunner, RougeScore, LLMJudge
from aevyra_verdict.providers import get_provider

dataset = Dataset.from_jsonl("examples/sample_data.jsonl")

runner = EvalRunner()
runner.add_provider("openai", "gpt-5.4-mini")
runner.add_provider("anthropic", "claude-sonnet-4-6")
runner.add_metric(RougeScore())
runner.add_metric(LLMJudge(judge_provider=get_provider("openai", "gpt-5.4-nano")))

results = runner.run(dataset)
print(results.compare())
```

Set your API keys as environment variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
`GOOGLE_API_KEY`, `MISTRAL_API_KEY`, `COHERE_API_KEY`) or pass them directly when
adding providers.

## How it works

The framework has four layers that compose together:

**Dataset** reads JSONL files where each line has a `messages` array (OpenAI chat
format), an optional `ideal` reference answer, and optional `metadata` for filtering.

**Providers** wrap each LLM API behind a common interface. The OpenAI message format
is the canonical input — each provider translates it to whatever the underlying SDK
expects (Anthropic's separate system parameter, Gemini's `contents` format, etc.) and
normalizes the response back into a `CompletionResult` with text, usage stats, and
latency.

**Metrics** score each response. Three families are supported:
- Reference-based (exact match, BLEU, ROUGE) — compare output against a known-good answer
- LLM-as-judge — use a separate model to evaluate quality on configurable criteria
- Custom — pass any Python function that returns a score

**Runner** ties it together: models and samples are dispatched concurrently via
thread pools. Rate-limit errors (HTTP 429) trigger exponential backoff with jitter
before retrying; fatal errors (auth failures, bad requests) are surfaced immediately
without burning retry budget. Results land in `EvalResults`.

```mermaid
flowchart LR
    DS[Dataset]:::data
    R[EvalRunner]:::model
    M[Metrics]:::metric
    OUT[Results]:::output

    DS --> R --> M --> OUT

    classDef data    fill:#6E3FF3,color:#fff,stroke:none
    classDef model   fill:#9B6BFF,color:#fff,stroke:none
    classDef metric  fill:#3FBFFF,color:#fff,stroke:none
    classDef output  fill:#2ECC71,color:#fff,stroke:none
```

## Usage

### Dataset format

Three formats are supported. The format is auto-detected from the first record.

**OpenAI** (native):

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"}
  ],
  "ideal": "The capital of France is Paris.",
  "metadata": {"category": "factual", "difficulty": "easy"}
}
```

**ShareGPT** (common HuggingFace fine-tuning format):

```json
{
  "conversations": [
    {"from": "human", "value": "What is the capital of France?"},
    {"from": "gpt", "value": "The capital of France is Paris."}
  ]
}
```

**Alpaca** (instruction-following datasets):

```json
{
  "instruction": "Translate to French.",
  "input": "Hello, how are you?",
  "output": "Bonjour, comment allez-vous?"
}
```

`messages` / `conversations` / `instruction` is required. `ideal` and `metadata` are
optional (or extracted automatically for ShareGPT and Alpaca). Pass `format=` explicitly
to override auto-detection:

```python
dataset = Dataset.from_jsonl("sharegpt_data.jsonl", format="sharegpt")
dataset = Dataset.from_jsonl("alpaca_data.jsonl", format="alpaca")
```

You can also create datasets inline:

```python
dataset = Dataset.from_list([
    {"messages": [{"role": "user", "content": "Hello"}], "ideal": "Hi there"},
])
```

Filter by metadata fields:

```python
hard_questions = dataset.filter(difficulty="hard", category="reasoning")
```

### Providers

Five providers are built in:

```python
from aevyra_verdict.providers import get_provider, list_providers

print(list_providers())
# ['anthropic', 'cohere', 'google', 'mistral', 'openai']

# Each provider takes a model name and optional api_key / base_url
provider = get_provider("openai", "gpt-5.4-mini", api_key="sk-...")
result = provider.complete([{"role": "user", "content": "Hello"}])
print(result.text, result.latency_ms, result.usage)
```

The OpenAI provider works with any OpenAI-compatible API (Azure, Together, vLLM,
etc.) by passing a `base_url`.

To add a custom provider, subclass `Provider` and register it:

```python
from aevyra_verdict.providers import Provider, register_provider

class MyProvider(Provider):
    name = "my_provider"
    def complete(self, messages, temperature=0.0, max_tokens=1024, **kwargs):
        # your implementation
        ...

register_provider("my_provider", MyProvider)
```

### Metrics

**Reference-based** (requires `ideal` in the dataset):

```python
from aevyra_verdict import ExactMatch, BleuScore, RougeScore

ExactMatch()                        # case-insensitive by default
ExactMatch(case_sensitive=True)
BleuScore(max_ngram=4)
RougeScore(variant="rougeL")        # also "rouge1", "rouge2"
```

**LLM-as-judge** (works with or without `ideal`):

```python
from aevyra_verdict import LLMJudge
from aevyra_verdict.providers import get_provider

judge = get_provider("anthropic", "claude-sonnet-4-6")
LLMJudge(judge_provider=judge)
LLMJudge(judge_provider=judge, criteria="Focus only on factual accuracy.")
```

The judge scores on a 1–5 scale (normalized to 0.0–1.0) and returns its reasoning.

**Custom metrics:**

```python
from aevyra_verdict import CustomMetric

def word_count_score(response, ideal=None, **kwargs):
    return min(len(response.split()) / 100, 1.0)

CustomMetric("word_count", word_count_score)
```

Custom functions return either a `float` or a `dict` with at least a `"score"` key
(optionally `"reasoning"` and any other details).

## CLI

After `pip install -e .`, the `aevyra-verdict` command is available.

### Inspect a dataset

Preview a dataset before running — shows sample count, whether ideals are present, and the first sample. No API calls made.

```bash
aevyra-verdict inspect dataset.jsonl
```

### Check configured providers

List all available providers and whether their API keys are set:

```bash
aevyra-verdict providers
```

### Specifying models

Pass `--model` (or `-m`) once per model, in `provider/model` format:

```bash
aevyra-verdict run dataset.jsonl \
  -m openai/gpt-5.4-mini \
  -m anthropic/claude-sonnet-4-6 \
  -m google/gemini-2.0-flash
```

For more than a couple of models, or when you want to reuse a configuration, use a config file instead:

```bash
aevyra-verdict run dataset.jsonl --config models.yaml
```

The config file supports JSON, YAML, and TOML. Each model entry takes `provider` and `model`, with optional `label`, `api_key`, and `base_url`:

```yaml
# models.yaml
models:
  - provider: openai
    model: gpt-5.4-mini
    label: gpt-5.4-mini

  - provider: anthropic
    model: claude-sonnet-4-6
    label: claude-sonnet

  # Local vLLM instance — uses the OpenAI-compatible API
  - provider: openai
    model: meta-llama/Llama-3.1-8B-Instruct
    base_url: http://localhost:8000/v1
    api_key: "none"
    label: llama-local
```

Start a local vLLM server with: `vllm serve meta-llama/Llama-3.1-8B-Instruct`

### Specifying metrics

Use `--metric` for built-in options (`rouge`, `bleu`, `exact`) and repeat for multiple:

```bash
aevyra-verdict run dataset.jsonl -m openai/gpt-5.4-mini --metric rouge --metric bleu
```

Add an LLM-as-judge with `--judge`:

```bash
aevyra-verdict run dataset.jsonl -m openai/gpt-5.4-mini --judge openai/gpt-5.4-nano
```

To customise the judge's evaluation criteria, pass a prompt template file. The recommended format is `.md` since judge prompts tend to have structure. Use `{criteria}`, `{conversation}`, `{response}`, and `{ideal_section}` as placeholders:

```bash
aevyra-verdict run dataset.jsonl -m openai/gpt-5.4-mini \
  --judge openai/gpt-5.4-nano \
  --judge-prompt examples/judge_prompt.md
```

`examples/judge_prompt.md` is a copy of the default template — a good starting point.

To use a custom Python scoring function, point at a file and name the function:

```bash
aevyra-verdict run dataset.jsonl -m openai/gpt-5.4-mini \
  --custom-metric examples/custom_metrics.py:brevity_score \
  --custom-metric examples/custom_metrics.py:contains_code
```

The function receives `(response, ideal=None, messages=None)` and returns either a `float` (0.0–1.0) or a dict with a `"score"` key and optional `"reasoning"`. See `examples/custom_metrics.py` for three working examples.

Save results to JSON with `-o`:

```bash
aevyra-verdict run dataset.jsonl --config models.yaml -o results.json
```

### Results

```python
results = runner.run(dataset)

# Formatted comparison table
print(results.compare("rouge_rougeL"))

# Summary dict
results.summary()

# Pandas DataFrame
df = results.to_dataframe()

# Export to JSON
results.to_json("eval_results.json")
```

### Configuration

```python
from aevyra_verdict.runner import RunConfig

config = RunConfig(
    temperature=0.0,       # deterministic by default
    max_tokens=1024,

    # Concurrency
    max_workers=10,        # concurrent requests per model
    max_model_workers=4,   # models evaluated concurrently

    # Retries and rate-limit handling
    num_retries=4,         # attempts after the first failure
    retry_base_delay=1.0,  # seconds before the first retry (doubles each attempt)
    retry_max_delay=60.0,  # backoff cap in seconds
    retry_jitter=0.25,     # ±25% random jitter to avoid thundering-herd retries
)
runner = EvalRunner(config=config)
```

Rate-limit errors (HTTP 429 / `RateLimitError`) always sleep through the backoff
before retrying. Auth and bad-request errors are surfaced immediately — no point
retrying a 401. If you're consistently hitting rate limits, the first thing to try
is lowering `max_workers`.

## Contributing

Bug reports and PRs are welcome. Open an issue first for anything larger than a bug fix.

**Adding a provider** — subclass `Provider` in `src/aevyra_verdict/providers/`, implement
`complete()`, and register it with `register_provider()`. See `openai_provider.py` as the
reference implementation.

**Adding a metric** — subclass `Metric` in `src/aevyra_verdict/metrics/`, implement
`score()`, and add it to the exports in `metrics/__init__.py`. See `reference.py` for
reference-based metrics and `judge.py` for LLM-as-judge.

## License

Apache 2.0
