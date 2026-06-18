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

1. **smoke** (free, laptop): `uv run python src/cai_smoke.py` proves the critique-and-revise loop
   end to end on a 0.5B model.
2. **generation at scale**: produce critique/revision and preference data on a GPU (vLLM).
3. **train**: SL-CAI SFT, then RL-CAI DPO, on a small open model.
4. **eval**: harmlessness on a held-out red-team set plus an over-refusal check, so the model gets
   safer without becoming evasive.

## Reproducibility

`uv` + `uv.lock` pin every dependency. Seeds fixed. Artifacts pushed to the Hub. Each step is a
single documented command.

## Status

Scaffolded. Running the smoke.
