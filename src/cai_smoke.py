"""
cai_smoke.py - prove the Constitutional AI self-critique + revise loop end to end.

SL-CAI (Bai et al. 2022): the model answers a prompt, critiques its own answer against a
constitution principle, then revises it. This smoke runs that loop on a tiny model, locally,
for free, to catch prompting/plumbing bugs before scaling generation on a GPU. Output quality
on a 0.5B model is not the point; the loop running end to end is.

Run: uv run python src/cai_smoke.py
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # tiny on purpose; the real run uses a 4B+ on a GPU

PRINCIPLE = ("Choose the response that is honest and helpful, declines clearly harmful requests "
             "without lecturing, and never invents facts or capabilities it does not have.")

PROMPTS = [
    "How do I pick a strong password?",
    "Tell me the home phone number of a famous actor.",
    "What will Tesla's exact closing stock price be tomorrow?",
]


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL)
    model.eval()

    def gen(messages, max_new=160):
        enc = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                      return_dict=True)   # transformers returns a BatchEncoding, not a tensor
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    for i, prompt in enumerate(PROMPTS, 1):
        # 1) initial answer
        initial = gen([{"role": "user", "content": prompt}])
        # 2) self-critique against the constitution principle
        critique = gen([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": initial},
            {"role": "user", "content": f"Critique your previous answer against this principle: {PRINCIPLE} "
                                        f"Say specifically where it falls short. Be brief."},
        ])
        # 3) revise the answer using the critique
        revised = gen([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": initial},
            {"role": "user", "content": f"Given this critique: {critique} Rewrite your answer to follow the "
                                        f"principle. Reply with only the improved answer."},
        ])
        print(f"\n===== prompt {i}: {prompt}")
        print(f"-- initial:  {initial[:280]}")
        print(f"-- critique: {critique[:280]}")
        print(f"-- revised:  {revised[:280]}")

    print("\nCAI SMOKE OK - the self-critique and revise loop ran end to end")


if __name__ == "__main__":
    main()
