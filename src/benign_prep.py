"""
benign_prep.py - sensitive-but-benign prompts for the over-refusal fix.

Pulls OR-Bench-80k (prompts that sound risky but are benign - the exact distribution where the
CAI run started over-refusing) and EXCLUDES the OR-Bench-hard-1k eval set, so training and eval
stay disjoint. XSTest is never trained on, so it remains a clean out-of-distribution check.

Pure data wrangling, runs local. Run:  uv run --with datasets --with huggingface_hub python src/benign_prep.py
"""
import json
import os

from datasets import load_dataset
from huggingface_hub import HfApi

N = int(os.environ.get("N_BENIGN", "1500"))
OUT = os.environ.get("BENIGN_DATASET", "yavuz-ai/cai-benign-prompts")


def main():
    hard = load_dataset("bench-llm/or-bench", "or-bench-hard-1k", split="train")
    hard_set = {(r.get("prompt") or "").strip() for r in hard}
    full = load_dataset("bench-llm/or-bench", "or-bench-80k", split="train").shuffle(seed=42)

    rows, seen = [], set()
    for r in full:
        p = (r.get("prompt") or "").strip()
        if p and p not in hard_set and p not in seen:
            seen.add(p)
            rows.append({"prompt": p})
            if len(rows) >= N:
                break
    print(f"benign prompts: {len(rows)}  (excluded {len(hard_set)} hard-1k eval prompts)")

    os.makedirs("out", exist_ok=True)
    path = "out/benign_prompts.jsonl"
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    api = HfApi()
    api.create_repo(OUT, exist_ok=True, repo_type="dataset")
    api.upload_file(path_or_fileobj=path, path_in_repo="prompts.jsonl", repo_id=OUT, repo_type="dataset")
    print(f"pushed -> https://huggingface.co/datasets/{OUT}")


if __name__ == "__main__":
    main()
