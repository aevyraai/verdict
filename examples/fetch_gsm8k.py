"""Download 50 samples from GSM8K and convert to aevyra-verdict format.

Usage:
    pip install datasets
    python examples/fetch_gsm8k.py
"""

import json
from pathlib import Path


def fetch_gsm8k(n=50, split="test", output="examples/gsm8k_sample.jsonl"):
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("Run: pip install datasets")

    print(f"Downloading GSM8K ({split} split)...")
    ds = load_dataset("openai/gsm8k", "main", split=split)

    samples = []
    for row in list(ds)[:n]:
        question = row["question"]
        answer = row["answer"]

        # GSM8K answers end with "#### <number>" — extract the final number as ideal
        ideal = answer.split("####")[-1].strip() if "####" in answer else answer.strip()

        samples.append({
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a math tutor. Solve the problem step by step, "
                        "then end your response with '#### <answer>' where <answer> "
                        "is just the final number."
                    ),
                },
                {"role": "user", "content": question},
            ],
            "ideal": ideal,
            "metadata": {"source": "gsm8k", "split": split},
        })

    out = Path(output)
    out.write_text("\n".join(json.dumps(s) for s in samples))
    print(f"Saved {len(samples)} samples to {out}")


if __name__ == "__main__":
    fetch_gsm8k()
