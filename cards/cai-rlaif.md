---
license: apache-2.0
base_model: yavuz-ai/qwen3-4b-cai-sft
library_name: peft
datasets:
  - Anthropic/hh-rlhf
tags:
  - constitutional-ai
  - rlaif
  - dpo
  - safety
pipeline_tag: text-generation
---

# qwen3-4b-cai-rlaif (RL-CAI stage)

The RLAIF stage of a Constitutional AI reproduction ([Bai et al. 2022](https://arxiv.org/abs/2212.08073)):
a LoRA DPO adapter on top of the SL-CAI model ([qwen3-4b-cai-sft](https://huggingface.co/yavuz-ai/qwen3-4b-cai-sft)),
trained on AI-labeled preference pairs. The model judged which of two answers better followed a
constitution principle; only pairs whose verdict was consistent across both answer orders were kept
(the rest were position-bias artifacts). Code + writeup: https://github.com/shipbehaves/constitutional-cai

## research artifact, not a better assistant

This stage pushes over-refusal on benign prompts to 0.484 (safety ~1.0). It reproduces the
harmless-vs-overrefusal regression; see the repo and the benign-mix variant
([qwen3-4b-cai-rlaif-v2](https://huggingface.co/yavuz-ai/qwen3-4b-cai-rlaif-v2)) that partially
recovers it.

## training

LoRA DPO (TRL), r=16/alpha=32, beta 0.1, 1 epoch, on ~1,777 both-orders-agreed preference pairs.
Reference = the adapter-disabled SL-CAI model (RL-CAI starts from the SL-CAI model).

## use

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
base = AutoModelForCausalLM.from_pretrained("yavuz-ai/qwen3-4b-cai-sft")
model = PeftModel.from_pretrained(base, "yavuz-ai/qwen3-4b-cai-rlaif")
tok = AutoTokenizer.from_pretrained("yavuz-ai/qwen3-4b-cai-sft")
```
