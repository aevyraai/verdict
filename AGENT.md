# aevyra-verdict — Agent Guide

This file is written for AI agents that need to use aevyra-verdict programmatically.
It covers canonical usage patterns, common mistakes, decision rules, and output
interpretation. For human-facing documentation see README.md.

---

## Install

```bash
pip install aevyra-verdict
```

Or in development:

```bash
pip install -e .
```

---

## Minimal working example

```python
from aevyra_verdict import Dataset, EvalRunner, RougeScore

ds = Dataset.from_jsonl("data.jsonl")

runner = EvalRunner()
runner.add_provider("openai", "gpt-5.4-nano")
runner.add_metric(RougeScore())

results = runner.run(ds)
print(results.compare())
```

**Required order:** add providers → add metrics → call run(). Calling run() before
adding a metric raises an error. Calling run() before adding a provider raises an error.

---

## Dataset

### Loading

```python
from aevyra_verdict import Dataset

# Auto-detects format from first record
ds = Dataset.from_jsonl("data.jsonl")

# Explicit format
ds = Dataset.from_jsonl("sharegpt.jsonl", format="sharegpt")
ds = Dataset.from_jsonl("alpaca.jsonl", format="alpaca")

# From a list in memory
ds = Dataset.from_list([
    {"messages": [{"role": "user", "content": "Hi"}], "ideal": "Hello"},
])
```

### Supported formats

| Format | Key field | Ideal answer |
|--------|-----------|--------------|
| `openai` | `messages` (array of `{role, content}`) | `ideal` field |
| `sharegpt` | `conversations` (array of `{from, value}`) | Last `gpt`/`assistant` turn |
| `alpaca` | `instruction` + optional `input` | `output` field |

### Filtering

```python
# Filter by metadata fields
hard = ds.filter(difficulty="hard")
math = ds.filter(category="math")
```

### Key properties

```python
len(ds)               # number of samples
ds[0]                 # Conversation object
ds[0].messages        # list of Message(role, content)
ds[0].ideal           # reference answer string or None
ds[0].metadata        # dict of extra fields
ds.has_ideals()       # True if all samples have ideal answers
ds.summary()          # dict with num_conversations, has_ideals, metadata_keys
```

---

## Providers

### Built-in providers

| Name | Env var | Notes |
|------|---------|-------|
| `openai` | `OPENAI_API_KEY` | GPT models |
| `anthropic` | `ANTHROPIC_API_KEY` | Claude models |
| `google` | `GOOGLE_API_KEY` | Gemini models |
| `mistral` | `MISTRAL_API_KEY` | Mistral models |
| `cohere` | `COHERE_API_KEY` | Command models |
| `openrouter` | `OPENROUTER_API_KEY` | Any model via OpenRouter |
| `local` | none | Ollama / vLLM via OpenAI-compatible API |

### Adding providers to a runner

```python
runner.add_provider("openai", "gpt-5.4-nano")
runner.add_provider("anthropic", "claude-haiku-4-5")

# With custom label
runner.add_provider("openai", "gpt-5.4-mini", label="gpt-5.4-mini")

# Local Ollama
runner.add_provider("local", "llama3.2:1b", base_url="http://localhost:11434/v1")

# Local vLLM
runner.add_provider("local", "meta-llama/Llama-3.2-1B", base_url="http://localhost:8000/v1")

# OpenRouter
runner.add_provider("openrouter", "meta-llama/llama-3.2-3b-instruct")
```

### Direct provider use (single completion, no runner)

```python
from aevyra_verdict.providers import get_provider

p = get_provider("openai", "gpt-5.4-nano")
result = p.complete(
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=256,
    temperature=0.0,
)
print(result.text)         # response string
print(result.latency_ms)   # float
print(result.usage)        # {"prompt_tokens": N, "completion_tokens": N}
```

---

## Metrics

### Built-in metrics

```python
from aevyra_verdict import RougeScore, BleuScore, ExactMatch, LLMJudge, CustomMetric

runner.add_metric(RougeScore())   # requires ideal answers
runner.add_metric(BleuScore())    # requires ideal answers
runner.add_metric(ExactMatch())   # requires ideal answers

# LLM-as-judge (no ideal answer required)
judge = get_provider("openai", "gpt-5.4-nano")
runner.add_metric(LLMJudge(judge_provider=judge))

# Custom judge prompt
runner.add_metric(LLMJudge(
    judge_provider=judge,
    prompt_template=Path("judge_prompt.md").read_text(),
))
```

### Custom Python metric

```python
def my_metric(response: str, ideal: str = None, messages: list = None, **kwargs):
    # Return a float 0.0–1.0
    return 1.0 if "keyword" in response.lower() else 0.0

runner.add_metric(CustomMetric("keyword_check", my_metric))
```

### Decision rules for metric selection

- Dataset **has ideal answers** → use `RougeScore` as baseline; add `LLMJudge` for quality
- Dataset **has no ideal answers** → use `LLMJudge` only
- Task is **exact / factual** (math, code, classification) → prefer `ExactMatch` or `RougeScore`
- Task is **generative / creative** → prefer `LLMJudge`
- Need **task-specific scoring** → use `CustomMetric`

---

## Runner configuration

```python
from aevyra_verdict.runner import EvalRunner, RunConfig

config = RunConfig(
    temperature=0.0,        # use 0.0 for reproducibility
    max_tokens=1024,
    max_workers=10,         # concurrent requests per model
    max_model_workers=4,    # models evaluated in parallel
    num_retries=4,
    retry_base_delay=1.0,
    retry_max_delay=60.0,
    retry_jitter=0.25,
)
runner = EvalRunner(config=config)
```

**Rate limit guidance:** If hitting 429 errors, lower `max_workers` first. The runner
retries automatically with exponential backoff — do not add your own retry loop around
`runner.run()`.

---

## Reading results

```python
results = runner.run(ds)

# Comparison table (string, ready to print)
print(results.compare())                  # uses first metric
print(results.compare("rouge"))           # specific metric

# Structured summary dict
summary = results.summary()
# Keys per model: provider, model, success_rate, mean_latency_ms, total_tokens,
#                 {metric}_mean, {metric}_stdev

# Access a specific model's score
mean_rouge = results.model_results["openai/gpt-5.4-nano"].mean_score("rouge")

# Export
results.to_json("results.json")
df = results.to_dataframe()   # pandas DataFrame

# Available models and metrics
results.models        # ["openai/gpt-5.4-nano", "anthropic/claude-haiku"]
results.metric_names  # ["rouge", "llm_judge"]
```

### Interpreting scores

| Metric | Range | What "good" means |
|--------|-------|-------------------|
| ROUGE | 0.0–1.0 | >0.4 is reasonable; >0.7 is strong |
| BLEU | 0.0–1.0 | >0.3 is reasonable for generation tasks |
| ExactMatch | 0 or 1 | 1 = exact match |
| LLMJudge | 0.0–1.0 | Depends on prompt; typically >0.7 is good |
| Custom | user-defined | Depends on implementation |

---

## Common mistakes

**Don't call run() with no metrics:**
```python
runner = EvalRunner()
runner.add_provider("openai", "gpt-5.4-nano")
# runner.run(ds)  ← raises error: no metrics configured
runner.add_metric(RougeScore())
results = runner.run(ds)  # correct
```

**Don't use reference-based metrics without ideal answers:**
```python
ds = Dataset.from_list([{"messages": [...]}])  # no ideal field
runner.add_metric(RougeScore())  # scores will all be 0.0 or None
# Use LLMJudge instead when there are no reference answers
```

**Don't add your own retry logic around run():**
```python
# Wrong — the runner already retries internally
for attempt in range(3):
    try:
        results = runner.run(ds)
        break
    except Exception:
        pass

# Correct — just call run(), configure retries via RunConfig
results = runner.run(ds)
```

**Don't pass `api_key=None` explicitly to local providers:**
```python
# Wrong
runner.add_provider("local", "llama3.2:1b", api_key=None)

# Correct — local provider uses a placeholder key automatically
runner.add_provider("local", "llama3.2:1b", base_url="http://localhost:11434/v1")
```

