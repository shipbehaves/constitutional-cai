"""
eval_prep.py - normalize the four public eval benchmarks into one eval_prompts.jsonl.

Two axes:
  - safety   (HARMFUL prompts; we want the model NOT to comply / to handle safely):
      strongreject (313), hh-rlhf harmless-base TEST (300)
  - refusal  (BENIGN prompts; we want the model to comply, NOT over-refuse):
      xstest safe (~250), or-bench-hard-1k (300)

All sources are ungated (StrongREJECT/XSTest via GitHub CSV; hh raw gzip; OR-Bench via datasets).
Pure data wrangling (CPU/IO), no GPU. Pushes the merged set to a Hub dataset for the GPU jobs.

Run:  uv run --with datasets --with huggingface_hub python src/eval_prep.py
"""
import csv
import gzip
import io
import json
import os
import re
import urllib.request

from huggingface_hub import HfApi, hf_hub_download

OUT_DATASET = os.environ.get("EVAL_PROMPTS_DATASET", "yavuz-ai/cai-eval-prompts")
N_HH = int(os.environ.get("N_HH", "300"))
N_ORBENCH = int(os.environ.get("N_ORBENCH", "300"))
SR_URL = "https://raw.githubusercontent.com/alexandrasouly/strongreject/main/strongreject_dataset/strongreject_dataset.csv"
XS_URL = "https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv"
_FH = re.compile(r"\n\nHuman:\s*(.*?)\n\nAssistant:", re.DOTALL)


def fetch_csv(url):
    return list(csv.DictReader(io.StringIO(urllib.request.urlopen(url).read().decode())))


def main():
    rows = []

    # 1) StrongREJECT - harmful -> safety axis
    for r in fetch_csv(SR_URL):
        p = (r.get("forbidden_prompt") or "").strip()
        if p:
            rows.append({"set": "strongreject", "axis": "safety", "prompt": p})

    # 2) hh-rlhf harmless-base TEST - harmful -> safety axis (held out from generation, which used train)
    path = hf_hub_download("Anthropic/hh-rlhf", "harmless-base/test.jsonl.gz", repo_type="dataset")
    seen, hh = set(), 0
    with gzip.open(path, "rt") as f:
        for line in f:
            m = _FH.search(json.loads(line)["chosen"])
            if not m:
                continue
            p = m.group(1).strip()
            if p and "Assistant:" not in p and "Human:" not in p and p not in seen:
                seen.add(p)
                rows.append({"set": "hh_test", "axis": "safety", "prompt": p})
                hh += 1
                if hh >= N_HH:
                    break

    # 3) XSTest SAFE - benign -> over-refusal axis
    for r in fetch_csv(XS_URL):
        if (r.get("label") or "").strip() == "safe":
            p = (r.get("prompt") or "").strip()
            if p:
                rows.append({"set": "xstest_safe", "axis": "refusal", "prompt": p})

    # 4) OR-Bench hard - benign-but-often-refused -> over-refusal axis
    from datasets import load_dataset
    ob = load_dataset("bench-llm/or-bench", "or-bench-hard-1k", split="train")
    for i, r in enumerate(ob):
        if i >= N_ORBENCH:
            break
        p = (r.get("prompt") or "").strip()
        if p:
            rows.append({"set": "orbench_hard", "axis": "refusal", "prompt": p})

    from collections import Counter
    by_set = Counter(r["set"] for r in rows)
    by_axis = Counter(r["axis"] for r in rows)
    print(f"total prompts: {len(rows)}")
    print(f"by set : {dict(by_set)}")
    print(f"by axis: {dict(by_axis)}")

    os.makedirs("out", exist_ok=True)
    path = "out/eval_prompts.jsonl"
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    if OUT_DATASET:
        api = HfApi()
        api.create_repo(OUT_DATASET, exist_ok=True, repo_type="dataset")
        api.upload_file(path_or_fileobj=path, path_in_repo="eval_prompts.jsonl", repo_id=OUT_DATASET, repo_type="dataset")
        print(f"pushed -> https://huggingface.co/datasets/{OUT_DATASET}")


if __name__ == "__main__":
    main()
