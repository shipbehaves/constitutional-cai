# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch",
#   "transformers==5.12.1",
#   "peft==0.19.1",
#   "huggingface_hub",
#   "hf_transfer",
# ]
# ///
"""
merge_adapter.py - merge a LoRA adapter into its base and push the standalone model.

Used to turn the RL-CAI adapter (on top of M_sft) into a single model so vLLM can load it directly
for eval, like the base and the already-merged SL-CAI model.

Run:  hf jobs uv run src/merge_adapter.py --flavor a100-large --secrets HF_TOKEN \
        --env BASE_MODEL=yavuz-ai/qwen3-4b-cai-sft --env ADAPTER_REPO=yavuz-ai/qwen3-4b-cai-rlaif \
        --env OUTPUT_REPO=yavuz-ai/qwen3-4b-cai-rlaif-merged
"""
import os
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = os.environ["BASE_MODEL"]
ADAPTER = os.environ["ADAPTER_REPO"]
OUTPUT_REPO = os.environ["OUTPUT_REPO"]
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).to(DEVICE)
    model = PeftModel.from_pretrained(model, ADAPTER)
    merged = model.merge_and_unload()
    merged.save_pretrained("out/merged")
    tok.save_pretrained("out/merged")
    merged.push_to_hub(OUTPUT_REPO)
    tok.push_to_hub(OUTPUT_REPO)
    print(f"MERGE OK -> https://huggingface.co/{OUTPUT_REPO}")


if __name__ == "__main__":
    main()
