# constitutional-cai

A reproduction of Constitutional AI (Bai et al. 2022) on small open models. The model learns from
its own feedback against a written constitution, with no human preference labels.

## Method (two stages)

1. **SL-CAI (self-revision):** for a red-team prompt, the model answers, critiques its own answer
   against a sampled constitution principle, and revises it. Supervised fine-tune on the revised
   answers.
2. **RL-CAI (RLAIF):** the model generates two answers per prompt and picks the more constitutional
   one. Those AI-labeled preferences train the policy with DPO.

## Plan (small steps)

1. **smoke** (free, single 8 GB GPU, e.g. RTX 4070): `uv run python src/cai_smoke.py` proves the critique-and-revise loop
   end to end on a 0.5B model.
2. **generation at scale**: produce critique/revision and preference data on a GPU (vLLM).
3. **train**: SL-CAI SFT, then RL-CAI DPO, on a small open model.
4. **eval**: harmlessness on a held-out red-team set plus an over-refusal check, so the model gets
   safer without becoming evasive.

## Result

Qwen3-4B-Instruct, two-axis eval: safety on harmful prompts (IBM Granite Guardian) and over-refusal
on benign prompts (refusal detection). Held-out benchmarks: StrongREJECT + hh-rlhf harmless-base test
(safety), XSTest-safe + OR-Bench-hard (over-refusal).

| model | safe on harmful (up) | over-refusal on benign (down) |
|---|---|---|
| [base](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | 0.989 | 0.327 |
| [SL-CAI](https://huggingface.co/yavuz-ai/qwen3-4b-cai-sft) | 1.000 | 0.431 |
| [RL-CAI](https://huggingface.co/yavuz-ai/qwen3-4b-cai-rlaif) | 1.000 | 0.484 |
| [RL-CAI + benign mix](https://huggingface.co/yavuz-ai/qwen3-4b-cai-rlaif-v2) | 1.000 | 0.455 |

### The failure analysis (the substance)

This is not a "it got safer" result. The base model is already ~99% safe, so CAI added no safety
headroom, and over-refusal on benign prompts **regressed** from 0.327 to 0.484 - the model got
"safer" mainly by refusing more, benign requests included. That is the harmless-vs-overrefusal
tension, reproduced and measured.

Two findings make it useful:

1. **Most naive AI preferences were position artifacts.** Judging each answer-pair in both orders and
   keeping only consistent winners showed 51.5% of raw pairs flipped when the order swapped. A
   single-pass judge would have trained on noise. Only the agreement filter removes it;
   counterbalancing alone does not.
2. **The over-refusal is mostly the SFT stage, and a targeted fix partially recovers it.** Stage
   decomposition: SFT added +10.4 points of over-refusal, DPO +5.3. Mixing ~25% sensitive-but-benign
   preference pairs (helpful over refusal) into DPO, at beta 0.1, pulled over-refusal back to 0.455
   (recovering ~55% of the DPO stage's regression) with safety held at 1.000. The residual is the
   SFT stage; the evidenced next step is benign revisions in SFT too.

Full teardown in [`NOTES.md`](./NOTES.md).

## Reproducibility

`uv` + `uv.lock` pin every dependency. Seeds fixed. Artifacts pushed to the Hub. Each step is a
single documented command:

```
uv run --with datasets --with huggingface_hub python src/eval_prep.py   # build the eval set
hf jobs uv run src/cai_generate.py  --flavor a100-large --secrets HF_TOKEN --env N_PROMPTS=5000 --env OUTPUT_DATASET=<raw>
uv run --with huggingface_hub python src/cai_filter.py                  # both-orders filter -> clean SFT/DPO sets
hf jobs uv run src/train_sft.py     --flavor a100-large --secrets HF_TOKEN --env SFT_DATASET=<sft> --env OUTPUT_REPO=<m_sft>
hf jobs uv run src/train_dpo.py     --flavor a100-large --secrets HF_TOKEN --env BASE_MODEL=<m_sft> --env DATASET=<prefs> --env OUTPUT_REPO=<rl>
hf jobs uv run src/eval_generate.py --flavor a100-large --secrets HF_TOKEN --env MODEL=<model> --env TAG=<tag> --env OUTPUT_DATASET=<resp>
hf jobs uv run src/eval_score.py    --flavor a100-large --secrets HF_TOKEN --env TAGS=base,sft,rlaif,rlaif_v2
```

## Status

Complete: trained SL-CAI + RL-CAI (+ benign-mix variant), two-axis eval, findings written up.
Models and datasets on the Hub under `yavuz-ai/`.
