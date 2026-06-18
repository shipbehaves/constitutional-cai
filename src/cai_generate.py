# /// script
# requires-python = ">=3.11"
# dependencies = ["vllm", "huggingface_hub", "hf_transfer"]
# ///
"""
cai_generate.py - generate Constitutional AI data with vLLM (batched), on a GPU.

SL-CAI: answer -> self-critique vs a sampled principle -> REVISE (a fresh answer turn, so the model
        cannot echo its critique). Emit {prompt, principle, initial, critique, revised}.
RL-CAI: two SAMPLED answers -> judge BOTH orders (A/B and B/A). Emit both raw verdicts; the filter
        keeps only pairs whose winner agrees across orders (the real position-bias fix).

Generation is batched PER STAGE via vllm .chat (full batches; the 3-turn dependency is between
stages, not within a batch). Raw assistant text is stored; the trainer re-applies the chat template
(template parity is asserted separately, $0, before any training).

Run on GPU:  hf jobs uv run src/cai_generate.py --flavor a100-large --secrets HF_TOKEN \
               --env N_PROMPTS=200 --env MODEL=Qwen/Qwen3-4B-Instruct-2507 \
               --env OUTPUT_DATASET=yavuz-ai/cai-raw-pilot
"""
import json
import os
import re

# The HF Jobs uv image has the CUDA runtime but no toolkit (nvcc), so flashinfer's JIT build
# fails. Use vLLM's prebuilt flash-attn backend and disable the flashinfer sampler, set BEFORE
# importing vllm.
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from vllm import LLM, SamplingParams

# Inlined from constitution.py + redteam.py so the script is self-contained for `hf jobs uv run`
# (which ships only the named file). Keep in sync with those modules.
PRINCIPLES = [
    "Choose the response that is most honest: it admits uncertainty, never fabricates facts, "
    "sources, numbers, or capabilities, and declines to predict things that cannot be known.",
    "Choose the response that is genuinely helpful and addresses the user's legitimate need "
    "directly, without unnecessary hedging.",
    "Choose the response that refuses or safely redirects clearly harmful, dangerous, or illegal "
    "requests, while NOT over-refusing benign ones.",
    "Choose the response that avoids revealing private or identifying information about real "
    "individuals.",
    "Choose the response that is respectful and non-judgmental and does not lecture or moralize "
    "at the user.",
]

_FIRST_HUMAN = re.compile(r"\n\nHuman:\s*(.*?)\n\nAssistant:", re.DOTALL)


def _extract_first_human_turn(transcript):
    m = _FIRST_HUMAN.search(transcript)
    if not m:
        return None
    p = m.group(1).strip()
    if not p or "Assistant:" in p or "Human:" in p:
        return None
    return p


def load_red_team_prompts(n):
    # read the raw gzipped JSONL directly (avoids the datasets<->pyarrow<->vllm dependency clash)
    import gzip
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("Anthropic/hh-rlhf", "harmless-base/train.jsonl.gz", repo_type="dataset")
    seen, out = set(), []
    with gzip.open(path, "rt") as f:
        for line in f:
            p = _extract_first_human_turn(json.loads(line)["chosen"])
            if p and p not in seen:
                seen.add(p); out.append(p)
                if len(out) >= n:
                    break
    return out

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-4B-Instruct-2507")
N_PROMPTS = int(os.environ.get("N_PROMPTS", "200"))
OUT_DIR = os.environ.get("OUT_DIR", "out")
OUTPUT_DATASET = os.environ.get("OUTPUT_DATASET", "")   # HF dataset repo to push raw data to
SEED = int(os.environ.get("SEED", "42"))
MAXTOK = int(os.environ.get("MAX_TOKENS", "320"))
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

_LETTER = re.compile(r"ANSWER:\s*([AB])", re.IGNORECASE)


def jaccard(a, b):
    sa, sb = set(a.lower().split()), set(b.lower().split())
    return len(sa & sb) / len(sa | sb) if (sa or sb) else 0.0


def parse_letter(verdict):
    m = _LETTER.search(verdict or "")
    if m:
        return m.group(1).upper()
    return None


def judge_msg(prompt, first, second, principle):
    return [{"role": "user", "content":
             f"A user sent this prompt:\n{prompt}\n\n"
             f"Response A:\n{first}\n\nResponse B:\n{second}\n\n"
             f"Which response better follows this principle: {principle}\n"
             f"Reason in one sentence, then end with a final line exactly 'ANSWER: A' or 'ANSWER: B'."}]


def main():
    prompts = load_red_team_prompts(N_PROMPTS)
    principles = [PRINCIPLES[i % len(PRINCIPLES)] for i in range(len(prompts))]
    print(f"model={MODEL} prompts={len(prompts)}")

    llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.90, max_model_len=2048)

    def chat(batch, sp):
        return [o.outputs[0].text.strip() for o in llm.chat(batch, sp, use_tqdm=False)]

    greedy = SamplingParams(temperature=0.0, max_tokens=MAXTOK)
    samp_a = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=MAXTOK, seed=SEED)
    samp_b = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=MAXTOK, seed=SEED + 1)
    judge_sp = SamplingParams(temperature=0.0, max_tokens=128)

    # ---- SL-CAI: initial -> critique -> revise (each a full batch) -------------------------------
    initial = chat([[{"role": "user", "content": p}] for p in prompts], greedy)
    critique = chat([[{"role": "user", "content": prompts[i]},
                      {"role": "assistant", "content": initial[i]},
                      {"role": "user", "content": f"Critique your answer against this principle: {principles[i]} "
                                                  f"Say where it falls short, briefly."}]
                     for i in range(len(prompts))], greedy)
    revised = chat([[{"role": "user", "content": prompts[i]},
                     {"role": "assistant", "content": initial[i]},
                     {"role": "user", "content": f"Now answer the original question again, fully following "
                                                 f"this principle: {principles[i]} Reply with only the improved "
                                                 f"answer, not a critique."}]
                    for i in range(len(prompts))], greedy)

    # ---- RL-CAI: two sampled answers -> judge BOTH orders ----------------------------------------
    ans_a = chat([[{"role": "user", "content": p}] for p in prompts], samp_a)
    ans_b = chat([[{"role": "user", "content": p}] for p in prompts], samp_b)
    v_ab = chat([judge_msg(prompts[i], ans_a[i], ans_b[i], principles[i]) for i in range(len(prompts))], judge_sp)
    v_ba = chat([judge_msg(prompts[i], ans_b[i], ans_a[i], principles[i]) for i in range(len(prompts))], judge_sp)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "sl_cai.jsonl"), "w") as f:
        for i in range(len(prompts)):
            f.write(json.dumps({"prompt": prompts[i], "principle": principles[i], "initial": initial[i],
                                "critique": critique[i], "revised": revised[i]}) + "\n")
    with open(os.path.join(OUT_DIR, "rl_cai.jsonl"), "w") as f:
        for i in range(len(prompts)):
            f.write(json.dumps({"prompt": prompts[i], "principle": principles[i], "ans_a": ans_a[i],
                                "ans_b": ans_b[i], "verdict_ab": v_ab[i], "verdict_ba": v_ba[i]}) + "\n")

    # ---- PILOT STATS (the gate): echo rate + both-orders agreement -------------------------------
    echoes = sum(jaccard(critique[i], revised[i]) > 0.6 for i in range(len(prompts)))
    agree = parsed = 0
    for i in range(len(prompts)):
        la, lb = parse_letter(v_ab[i]), parse_letter(v_ba[i])
        if la and lb:
            parsed += 1
            win_ab = ans_a[i] if la == "A" else ans_b[i]      # in A/B order, A is ans_a
            win_ba = ans_b[i] if lb == "A" else ans_a[i]      # in B/A order, A is ans_b
            agree += (win_ab == win_ba)
    n = len(prompts)
    print(f"\nPILOT STATS (n={n})")
    print(f"  revise-echo rate   : {echoes}/{n} = {echoes/n:.1%}  (gate: want < 30%)")
    print(f"  judge parse rate   : {parsed}/{n} = {parsed/n:.1%}")
    print(f"  both-orders agree  : {agree}/{parsed if parsed else 1} = {agree/(parsed or 1):.1%}  (gate: want > 45%)")
    print(f"  -> est clean RL pairs from {n} = ~{agree}")

    if OUTPUT_DATASET:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(OUTPUT_DATASET, exist_ok=True, repo_type="dataset")
        api.upload_folder(folder_path=OUT_DIR, repo_id=OUTPUT_DATASET, repo_type="dataset")
        print(f"pushed raw data to https://huggingface.co/datasets/{OUTPUT_DATASET}")
    print("CAI GENERATE OK")


if __name__ == "__main__":
    main()
