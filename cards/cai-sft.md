---
license: apache-2.0
base_model: Qwen/Qwen3-4B-Instruct-2507
library_name: transformers
datasets:
  - Anthropic/hh-rlhf
tags:
  - constitutional-ai
  - rlaif
  - safety
pipeline_tag: text-generation
---

# qwen3-4b-cai-sft (SL-CAI stage)

The supervised stage of a Constitutional AI reproduction ([Bai et al. 2022](https://arxiv.org/abs/2212.08073)):
Qwen3-4B-Instruct fine-tuned on its own self-critiqued, revised answers to red-team prompts, against
a 5-principle constitution adapted from public sources (the CAI paper and Anthropic's published
constitution). LoRA SFT, merged. Code + full writeup: https://github.com/shipbehaves/constitutional-cai

## research artifact, not a better assistant

On a two-axis eval this stage raises over-refusal on benign prompts from 0.327 to 0.431 while safety
stays ~1.0. It is published for reproducibility and failure analysis, not as a general-purpose model.
See the repo for the stage-by-stage teardown of the harmless-vs-overrefusal tension.

## training

LoRA SFT (r=16, alpha=32, all-linear), completion-only loss on the revised answers, 2 epochs,
lr 1e-4, bf16, on 1x A100. Data: ~4,800 self-revisions generated from hh-rlhf harmless-base prompts.
