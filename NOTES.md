# Lab notebook - constitutional-cai

Running log of findings. The writeup leads with these, not with a happy path.

---

## Finding 1 - most naive AI preferences were position-bias artifacts

In RL-CAI the model judges which of two answers better follows a sampled principle. That judgment is
strongly biased toward whichever answer is shown first. Judging each pair in BOTH orders (A/B and
B/A) and keeping only pairs whose winner is consistent across orders showed that **51.5% of the raw
preference pairs (2,573 / 5,000) flipped when the order was swapped** - they were position artifacts,
not real preferences. A naive single-pass judge would have trained on all of them.

Counterbalancing order alone does not fix this (it just spreads the bias across the dataset); only
the agreement filter removes it. After agreement + pair hygiene (identical, near-duplicate,
length-imbalanced), 35.5% of pairs survived (1,777 clean). The survival accounting is the deliverable
as much as the final model.

---

## Finding 2 - CAI on an already-aligned model: a safety ceiling and an over-refusal regression

Setup: Qwen3-4B-Instruct; SL-CAI (self-critique then revise, supervised) followed by RL-CAI
(AI-preference DPO); 5 public principles. Eval is two-axis: safety on harmful prompts (IBM Granite
Guardian) and over-refusal on benign prompts (refusal detection), held-out benchmarks.

| model | safe on harmful (up) | over-refusal on benign (down) |
|---|---|---|
| base | 0.989 | 0.327 |
| SL-CAI | 1.000 | 0.431 |
| RL-CAI | 1.000 | 0.484 |
| RL-CAI + benign mix | 1.000 | 0.455 |

**Symptom:** the base model is already ~99% safe, so CAI added essentially no safety headroom - but
over-refusal on benign prompts climbed from 0.327 to 0.484. The model became "safer" mostly by
refusing more, benign requests included. This is the harmless-vs-overrefusal tension, reproduced.

**Stage decomposition:** the SFT stage added +10.4 points of over-refusal; the DPO stage added +5.3
more. The SFT stage - trained only on revised answers to red-team prompts - is the dominant cause.

**Fix + proof:** mixing ~25% sensitive-but-benign preference pairs (chosen = helpful, rejected =
refusal) into the DPO set, at beta 0.1, pulled over-refusal back 0.484 -> 0.455, recovering ~55% of
the DPO stage's regression while safety stayed at 1.000. Note on beta: it was kept at 0.1, NOT
raised; raising beta would anchor the policy harder to the over-refusing SFT reference, the opposite
of what is needed here. The benign prompts came from OR-Bench-80k held out from the OR-Bench-hard
eval set; XSTest was never trained on, so it stays a clean out-of-distribution check.

**Residual + next step:** a DPO-only fix cannot undo the larger SFT-stage over-refusal. The evidenced
next step is to put benign helpful revisions into the SFT stage as well, then repeat the two-axis
eval.
