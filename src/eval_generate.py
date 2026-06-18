# /// script
# requires-python = ">=3.11"
# dependencies = ["vllm", "huggingface_hub", "hf_transfer"]
# ///
"""
eval_generate.py - generate one model's responses to the eval prompt set (vLLM, batched).

Run once per model condition (base / SL-CAI / RL-CAI). Greedy decoding, no system prompt, fixed
max_new, so the three conditions are compared on identical inputs. Pushes {set, axis, prompt,
response} as <TAG>.jsonl to the responses dataset; eval_score.py judges them.

Run (GPU):  hf jobs uv run src/eval_generate.py --flavor a100-large --secrets HF_TOKEN \
              --env MODEL=Qwen/Qwen3-4B-Instruct-2507 --env TAG=base \
              --env OUTPUT_DATASET=yavuz-ai/cai-eval-responses
"""
import json
import os

os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")   # HF Jobs image has no nvcc -> avoid flashinfer JIT
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

from vllm import LLM, SamplingParams
from huggingface_hub import HfApi, hf_hub_download

MODEL = os.environ["MODEL"]
TAG = os.environ.get("TAG", "model")
PROMPTS_DATASET = os.environ.get("EVAL_PROMPTS_DATASET", "yavuz-ai/cai-eval-prompts")
OUTPUT_DATASET = os.environ.get("OUTPUT_DATASET", "")
N = int(os.environ.get("N_PROMPTS", "0"))      # 0 = all
MAXTOK = int(os.environ.get("MAX_TOKENS", "512"))


def main():
    path = hf_hub_download(PROMPTS_DATASET, "eval_prompts.jsonl", repo_type="dataset")
    prompts = [json.loads(line) for line in open(path)]
    if N:
        prompts = prompts[:N]
    print(f"model={MODEL} tag={TAG} prompts={len(prompts)}")

    llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.90, max_model_len=2048)
    sp = SamplingParams(temperature=0.0, max_tokens=MAXTOK)   # greedy, deterministic, no system prompt
    outs = llm.chat([[{"role": "user", "content": p["prompt"]}] for p in prompts], sp, use_tqdm=False)
    responses = [o.outputs[0].text.strip() for o in outs]

    os.makedirs("out", exist_ok=True)
    out_path = f"out/{TAG}.jsonl"
    with open(out_path, "w") as f:
        for p, r in zip(prompts, responses):
            f.write(json.dumps({"set": p["set"], "axis": p["axis"], "prompt": p["prompt"], "response": r}) + "\n")
    print(f"wrote {len(responses)} responses -> {out_path}")

    if OUTPUT_DATASET:
        api = HfApi()
        api.create_repo(OUTPUT_DATASET, exist_ok=True, repo_type="dataset")
        api.upload_file(path_or_fileobj=out_path, path_in_repo=f"{TAG}.jsonl", repo_id=OUTPUT_DATASET, repo_type="dataset")
        print(f"pushed -> https://huggingface.co/datasets/{OUTPUT_DATASET} ({TAG}.jsonl)")
    print("EVAL GENERATE OK")


if __name__ == "__main__":
    main()
