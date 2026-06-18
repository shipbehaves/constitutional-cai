"""
cai_filter.py - turn raw CAI generations into clean SFT + DPO datasets, with survival accounting.

Runs locally / on CPU (pure Python, no GPU). Reads the raw sl_cai.jsonl + rl_cai.jsonl from a Hub
dataset, applies the locked filters, prints the survival table (the failure-first headline), and
emits two datasets in the EXACT conversational schemas the trainers consume:
  - SFT  : {"messages": [user, assistant(revised)]}            (train_sft.py)
  - prefs: {"chosen": [user, assistant], "rejected": [...]}     (train_dpo.py conversational format)

Filters: SL = drop critique-echo (Jaccard>0.6), empty (<5 words), dup. RL = both-orders agreement
(keep only pairs whose winner is consistent across A/B and B/A), drop unparseable verdicts,
chosen==rejected, near-dups (Jaccard>0.8), length-imbalance (word ratio outside [0.5,2.0]), empty.

Run (dry, on pilot):  RAW_DATASET=yavuz-ai/cai-raw-pilot uv run python src/cai_filter.py
Run (real + push):    RAW_DATASET=yavuz-ai/cai-raw SFT_DATASET=yavuz-ai/cai-sft \
                      PREFS_DATASET=yavuz-ai/cai-prefs uv run python src/cai_filter.py
"""
import json
import os
import re

from huggingface_hub import HfApi, hf_hub_download

RAW_DATASET = os.environ.get("RAW_DATASET", "yavuz-ai/cai-raw")
SFT_DATASET = os.environ.get("SFT_DATASET", "")       # push target; empty = dry run
PREFS_DATASET = os.environ.get("PREFS_DATASET", "")
OUT_DIR = os.environ.get("OUT_DIR", "out")

_LETTER = re.compile(r"ANSWER:\s*([AB])", re.IGNORECASE)


def words(s):
    return s.split()


def jaccard(a, b):
    sa, sb = set(a.lower().split()), set(b.lower().split())
    return len(sa & sb) / len(sa | sb) if (sa or sb) else 0.0


def parse_letter(v):
    m = _LETTER.search(v or "")
    return m.group(1).upper() if m else None


def read_jsonl(name):
    p = hf_hub_download(RAW_DATASET, name, repo_type="dataset")
    return [json.loads(line) for line in open(p)]


def filter_sl(sl):
    clean, drop, seen = [], {"echo": 0, "empty": 0, "dup": 0}, set()
    for r in sl:
        rev = r["revised"].strip()
        if jaccard(r["critique"], rev) > 0.6:
            drop["echo"] += 1; continue
        if len(words(rev)) < 5:
            drop["empty"] += 1; continue
        if rev[:200] in seen:
            drop["dup"] += 1; continue
        seen.add(rev[:200])
        # prompt-completion conversational format -> TRL SFTTrainer completion_only_loss masks the prompt
        clean.append({"prompt": [{"role": "user", "content": r["prompt"]}],
                      "completion": [{"role": "assistant", "content": rev}]})
    return clean, drop


def filter_rl(rl):
    clean, drop = [], {"unparsed": 0, "disagree": 0, "empty": 0, "identical": 0, "neardup": 0, "lenbal": 0}
    for r in rl:
        la, lb = parse_letter(r["verdict_ab"]), parse_letter(r["verdict_ba"])
        if not (la and lb):
            drop["unparsed"] += 1; continue
        win_ab = r["ans_a"] if la == "A" else r["ans_b"]   # A/B order: slot A = ans_a
        win_ba = r["ans_b"] if lb == "A" else r["ans_a"]   # B/A order: slot A = ans_b
        if win_ab != win_ba:
            drop["disagree"] += 1; continue                # position-biased / inconsistent -> drop
        chosen = win_ab.strip()
        rejected = (r["ans_b"] if win_ab == r["ans_a"] else r["ans_a"]).strip()
        if len(words(chosen)) < 5 or len(words(rejected)) < 5:
            drop["empty"] += 1; continue
        if chosen == rejected:
            drop["identical"] += 1; continue
        if jaccard(chosen, rejected) > 0.8:
            drop["neardup"] += 1; continue
        ratio = len(words(chosen)) / max(len(words(rejected)), 1)
        if ratio < 0.5 or ratio > 2.0:
            drop["lenbal"] += 1; continue
        clean.append({"chosen": [{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": chosen}],
                      "rejected": [{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": rejected}]})
    return clean, drop


def main():
    sl, rl = read_jsonl("sl_cai.jsonl"), read_jsonl("rl_cai.jsonl")
    sl_clean, sl_drop = filter_sl(sl)
    rl_clean, rl_drop = filter_rl(rl)

    print(f"\n=== SURVIVAL ACCOUNTING (raw={RAW_DATASET}) ===")
    print(f"SL-CAI: {len(sl)} raw -> {len(sl_clean)} clean ({len(sl_clean)/max(len(sl),1):.1%})   drops={sl_drop}")
    print(f"RL-CAI: {len(rl)} raw -> {len(rl_clean)} clean ({len(rl_clean)/max(len(rl),1):.1%})   drops={rl_drop}")

    os.makedirs(OUT_DIR, exist_ok=True)
    sft_path = os.path.join(OUT_DIR, "sft_train.jsonl")
    prefs_path = os.path.join(OUT_DIR, "prefs_train.jsonl")
    with open(sft_path, "w") as f:
        for r in sl_clean:
            f.write(json.dumps(r) + "\n")
    with open(prefs_path, "w") as f:
        for r in rl_clean:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {sft_path} ({len(sl_clean)}) and {prefs_path} ({len(rl_clean)})")

    api = HfApi()
    if SFT_DATASET:
        api.create_repo(SFT_DATASET, exist_ok=True, repo_type="dataset")
        api.upload_file(path_or_fileobj=sft_path, path_in_repo="train.jsonl", repo_id=SFT_DATASET, repo_type="dataset")
        print(f"pushed SFT -> https://huggingface.co/datasets/{SFT_DATASET}")
    if PREFS_DATASET:
        api.create_repo(PREFS_DATASET, exist_ok=True, repo_type="dataset")
        api.upload_file(path_or_fileobj=prefs_path, path_in_repo="train.jsonl", repo_id=PREFS_DATASET, repo_type="dataset")
        print(f"pushed prefs -> https://huggingface.co/datasets/{PREFS_DATASET}")
    if not (SFT_DATASET or PREFS_DATASET):
        print("(dry run - set SFT_DATASET / PREFS_DATASET to push)")


if __name__ == "__main__":
    main()
