---
license: apache-2.0
base_model: yavuz-ai/qwen3-4b-cai-sft
library_name: peft
datasets:
  - Anthropic/hh-rlhf
  - bench-llm/or-bench
tags:
  - constitutional-ai
  - rlaif
  - dpo
  - safety
pipeline_tag: text-generation
---

# qwen3-4b-cai-rlaif-v2 (RL-CAI, benign-mixed)

The RL-CAI stage with the over-refusal fix: a LoRA DPO adapter on the SL-CAI model
([qwen3-4b-cai-sft](https://huggingface.co/yavuz-ai/qwen3-4b-cai-sft)), trained on the harmful
preference pairs **plus ~25% sensitive-but-benign pairs** (chosen = helpful, rejected = refusal) so
DPO is pulled off over-refusal. Code + writeup: https://github.com/shipbehaves/constitutional-cai

## result

Over-refusal on benign prompts drops 0.484 -> 0.455 versus the harmful-only RL-CAI model, recovering
~55% of the DPO stage's regression, with safety held at 1.0. It is a partial fix: the larger
over-refusal contribution comes from the SFT stage, which a DPO-only change cannot undo. Still a
research artifact reproducing the harmless-vs-overrefusal tension, not a general-purpose assistant.

## training

LoRA DPO (TRL), r=16/alpha=32, beta 0.1 (kept low on purpose; raising it would anchor harder to the
over-refusing SFT reference), 1 epoch, ~2,369 pairs (25% benign from OR-Bench-80k held out from the
OR-Bench-hard eval set). Reference = the adapter-disabled SL-CAI model.
