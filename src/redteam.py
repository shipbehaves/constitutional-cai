"""
redteam.py - load red-team prompts from Anthropic/hh-rlhf (harmless-base).

The dataset has NO "prompt" column: each row is a full transcript string in the
"\\n\\nHuman: ... \\n\\nAssistant: ..." format. We extract the FIRST human turn. The extraction
is a leakage vector (a stray "Assistant:" would smuggle an answer into the prompt), so it is
unit-tested and asserts no turn markers survive in the returned prompt.
"""
import re

_FIRST_HUMAN = re.compile(r"\n\nHuman:\s*(.*?)\n\nAssistant:", re.DOTALL)


def extract_first_human_turn(transcript: str):
    """Return the first human turn, or None if it can't be cleanly isolated."""
    m = _FIRST_HUMAN.search(transcript)
    if not m:
        return None
    prompt = m.group(1).strip()
    # leakage guards: non-empty, and no further turn markers smuggled in
    if not prompt or "Assistant:" in prompt or "Human:" in prompt:
        return None
    return prompt


def load_red_team_prompts(n, seed=42):
    """Load up to n unique first-human-turn prompts from hh-rlhf harmless-base (train split)."""
    from datasets import load_dataset
    ds = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base", split="train")
    seen, out = set(), []
    for row in ds:
        p = extract_first_human_turn(row["chosen"])
        if p and p not in seen:
            seen.add(p)
            out.append(p)
            if len(out) >= n:
                break
    return out
