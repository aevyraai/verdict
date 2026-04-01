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

"""Basic eval example: compare two models on a small dataset."""

from aevyra_verdict import Dataset, EvalRunner, ExactMatch, RougeScore, LLMJudge
from aevyra_verdict.providers import get_provider

# Load the dataset
dataset = Dataset.from_jsonl("examples/sample_data.jsonl")
print(f"Loaded {len(dataset)} samples from '{dataset.name}'")
print(f"Dataset summary: {dataset.summary()}\n")

# Set up the runner with models to compare
runner = EvalRunner()
runner.add_provider("openai", "gpt-5.4-mini")
runner.add_provider("anthropic", "claude-sonnet-4-6")

# Add metrics
runner.add_metric(RougeScore(variant="rougeL"))
runner.add_metric(ExactMatch())

# Optionally add LLM-as-judge (uses a separate model to evaluate)
judge_provider = get_provider("openai", "gpt-5.4-nano")
runner.add_metric(LLMJudge(judge_provider=judge_provider))

# Run the eval
print("Running eval...")
results = runner.run(dataset)

# Print comparison table
print("\n" + results.compare("rouge_rougeL"))
print("\n" + results.compare("llm_judge"))

# Get summary as dict
summary = results.summary()
for model, stats in summary.items():
    print(f"\n{model}: {stats}")

# Export to JSON
results.to_json("eval_results.json")
print("\nResults saved to eval_results.json")

# Convert to pandas DataFrame for further analysis
# df = results.to_dataframe()
# print(df)
