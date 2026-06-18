# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch",
#   "transformers==5.12.1",
#   "trl==1.6.0",
#   "datasets",
#   "peft==0.19.1",
#   "accelerate",
#   "huggingface_hub",
#   "hf_transfer",
# ]
# ///
"""
train_sft.py - SL-CAI stage: LoRA SFT on the filtered self-revisions, then MERGE the adapter.

Trains the base model to produce the constitution-aligned revised answers. Uses prompt-completion
data + completion_only_loss=True (TRL 1.6.0), so loss is on the assistant revision only and the
red-team prompt is masked (verified against the pinned trl source). After training, merge_and_unload
and save a clean bf16 model = M_sft, which becomes BOTH the policy init and the frozen reference for
the RL-CAI DPO stage (Bai et al.: "RL-CAI starts from the SL-CAI model").

SFT_DATASET may be a Hub repo (real run) or a local .jsonl (the free smoke).

Run (GPU):  hf jobs uv run src/train_sft.py --flavor a100-large --secrets HF_TOKEN \
              --env BASE_MODEL=Qwen/Qwen3-4B-Instruct-2507 --env SFT_DATASET=yavuz-ai/cai-sft \
              --env OUTPUT_REPO=yavuz-ai/qwen3-4b-cai-sft
Run (smoke): MODEL/0.5B + a local jsonl, see __main__.
"""
import os
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
SFT_DATASET = os.environ.get("SFT_DATASET", "")     # Hub repo or local .jsonl path
OUTPUT_REPO = os.environ.get("OUTPUT_REPO", "")
EPOCHS = float(os.environ.get("EPOCHS", "2"))
LR = float(os.environ.get("LR", "1e-4"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "-1"))  # smoke sets a small cap
SEED = int(os.environ.get("SEED", "42"))
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_sft(path):
    if path.endswith(".jsonl"):
        return load_dataset("json", data_files=path, split="train")
    return load_dataset(path, split="train")


def main():
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16 if DEVICE == "cuda" else None)
    model.to(DEVICE)
    ds = load_sft(SFT_DATASET)
    print(f"base={BASE_MODEL} sft_examples={len(ds)} device={DEVICE}")

    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM", target_modules="all-linear")
    args = SFTConfig(
        output_dir="out/sft", num_train_epochs=EPOCHS, max_steps=MAX_STEPS,
        per_device_train_batch_size=4, gradient_accumulation_steps=4,   # effective batch 16
        learning_rate=LR, warmup_ratio=0.05, lr_scheduler_type="cosine",
        completion_only_loss=True, packing=False, max_length=1024,
        bf16=(DEVICE == "cuda"), gradient_checkpointing=(DEVICE == "cuda"),
        logging_steps=10, save_strategy="no", report_to="none", seed=SEED,
    )
    trainer = SFTTrainer(model=model, args=args, train_dataset=ds, processing_class=tok, peft_config=lora)
    trainer.train()

    # merge the adapter into the base -> a clean standalone model = M_sft
    merged = trainer.model.merge_and_unload()
    os.makedirs("out/sft_merged", exist_ok=True)
    merged.save_pretrained("out/sft_merged")
    tok.save_pretrained("out/sft_merged")
    print("merged and saved M_sft -> out/sft_merged")

    if OUTPUT_REPO:
        merged.push_to_hub(OUTPUT_REPO)
        tok.push_to_hub(OUTPUT_REPO)
        print(f"pushed M_sft -> https://huggingface.co/{OUTPUT_REPO}")
    print("SFT OK")


if __name__ == "__main__":
    main()
