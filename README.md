# On-Policy Distillation + RLVR — a minimal, single-H200 reproduction

A learning project that reproduces the core ideas of Thinking Machines' *On-Policy
Distillation* (Lu, 2025) on **one H200, ≤ 80 GB disk, ≤ 8 h per Slurm job**, using
**TRL + vLLM**. The blog uses the Tinker API and a 32B→8B reasoning run; this repo
strips that down to the parts that actually teach the mechanics.

> Blog: https://thinkingmachines.ai/blog/on-policy-distillation/
> Cookbook (reference): https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/distillation

---

## 0. Goals (and what we deliberately drop)

1. **Learn on-policy distillation.** Build the loop: sample from the *student*,
   score every token with a *teacher*, train on the per-token reverse-KL signal.
2. **Learn the RL pipeline (RLVR).** Build the same loop with a *verifiable* reward
   (right/wrong answer) instead of a teacher — i.e. GRPO. Feel why sparse reward is
   harder than dense reward.

**Dropped:** the 400k-prompt SFT mid-training from the blog. SFT is not part of
on-policy distillation; its only job is to move a *base* student into the teacher's
support (it adds support via forward-KL; reverse-KL distillation then mode-seeks
*within* that support). We avoid needing it by using an **instruct** student that
already produces math CoT and the right answer format. No SFT required.

---

## 1. The benchmark decision (model-card homework)

**Chosen benchmark: AIME (2024 + 2025), pass@1, with a short token budget.**
Reported together (60 problems) to damp the variance of a 30-problem test.
Rationale: it is the blog's own domain, it is **verifiable** (single integer
answer → trivial exact-match reward, perfect for RLVR), and a 4B model sits around
**~47%** on AIME'25 → real headroom to move, unlike MATH-500 where a strong 4B is
near-ceiling and you'd see almost nothing.

**Honest finding from the cards — read this before you expect a big teacher gap:**

| Model (mode) | AIME'25 | GPQA-D | source |
|---|---|---|---|
| Qwen3-4B-Instruct-2507 (non-thinking) | **47.4** | **62.0** | official model card |
| Qwen3-8B (non-thinking) | ~lower than the 4B-2507 | ~lower | April release |
| Qwen3-8B (thinking) | ~mid-60s | ~62 | April release |
| Qwen3-4B (thinking, original April) | ~mid-60s | ~56 | April release |

Takeaways that drive the whole design:

- The **July "2507" 4B is unusually strong** — in matched (non-thinking) mode it
  **ties or beats the April 8B**. So a naive *Qwen3-8B → Qwen3-4B-Instruct-2507*
  distillation has **almost no gap to teach**.
- The 8B only clearly leads in **thinking** mode, and even then the AIME gap is a
  few points — i.e. within the noise of a 30-question eval and bought with long,
  expensive reasoning traces.
- **Do not trust reported numbers across eval harnesses** (they disagree by 5–15
  pts depending on prompt/parser/length). Phase 0 below measures *your own*
  baseline so the later deltas are apples-to-apples.

**Consequence:** instead of leaning on the 8B/4B size gap, we **create** a clean,
attributable gap with RLVR (Phase 1), then distill that gain back (Phase 2). The 8B
(and optionally a 14B) is used in an *optional* Phase 3 to test the very thing you
asked about — **does distillation still help when the teacher is barely better than
the student?** Expected answer, and a good lesson: **no, not much.** Distillation
transfers the teacher's distribution; if the teacher isn't better, there's nothing
to transfer. Reverse-KL is "unhackable" only *with respect to the teacher* — a
mediocre teacher faithfully hands the student its own ceiling (and its mistakes).

---

## 2. Models

All Qwen3 dense + MoE checkpoints **share one tokenizer / 151,936-token vocab**.
That matters: TRL's GKD does *logit-level* KL, which is only well-defined when
teacher and student share a vocabulary. Staying inside the Qwen3 family guarantees
that. Everything runs **non-thinking** (short outputs → cheap → fits one job).

| Role | Default checkpoint | bf16 size | FP8 size | used in |
|---|---|---|---|---|
| Student | `Qwen/Qwen3-4B-Instruct-2507` | ~8 GB | ~4 GB | all phases |
| Self-distill teacher | *your Phase-1 RLVR checkpoint* (a 4B) | ~8 GB | — | Phase 2 |
| External teacher (optional) | `Qwen/Qwen3-8B` (thinking) | ~16 GB | ~8 GB | Phase 3 |
| Bigger teacher (optional) | `Qwen/Qwen3-14B` (thinking) | ~29 GB | ~15 GB | Phase 3 |

Student trains with **LoRA** (rank 32–64) throughout — keeps VRAM low and makes
checkpoints a few hundred MB, which is what makes the 8-hour / 80 GB limits
survivable (see §3, §6).

---

## 3. Hardware & storage budget (why it fits)

Per Slurm job: 1× H200 (141 GB HBM3e, Hopper → native FP8), ≤ 80 GB disk, ≤ 8 h.

**Disk (worst case, Phase 3 with 14B teacher):**

```
Qwen3-14B (FP8)              ~15 GB
Qwen3-4B-Instruct-2507       ~ 8 GB
math train + AIME eval sets  ~ 1 GB
LoRA checkpoints (all)       ~ 2 GB
HF cache slack               ~ 5 GB
------------------------------------
total                        ~31 GB   << 80 GB  ✓
```

(Phases 1–2 are far smaller, ~12 GB.)

**VRAM (Phase 2, the tightest co-resident case — student trained + teacher scored):**

```
student 4B base (bf16, frozen)     ~ 8 GB
LoRA adapters + grads + optim      < 1 GB
teacher 8B (bf16, forward only)    ~16 GB
vLLM KV cache + activations        ~10–25 GB (scales with batch × seq)
------------------------------------------
total                              ~35–50 GB  << 141 GB  ✓
```

**Time (rough, single H200, estimates — measure yours):** each phase is sized to
land inside **one** 8-hour job so you never have to chain across the queue.

| Phase | steps | ~min/step | wall-clock |
|---|---|---|---|
| 0 baseline eval | — | — | < 1 h |
| 1 RLVR (GRPO) | 100–150 | 2–3 | ~4–6 h |
| 2 on-policy distill | 50–100 | 1–2 | ~2–3 h |
| 3 optional distill | 50–100 | 1–2 | ~2–3 h |

> Caveat that affects timing: stock **TRL GKD samples on-policy with HF `generate`**,
> which is much slower than vLLM. Keep Phase-2 steps modest, or use the
> vLLM-accelerated hand-rolled loop in Appendix A. RLVR (GRPO) *does* support vLLM.

**Partition tip (Explorer):** the H200 queue is contended (often 15+ jobs pending).
For **short jobs ≤ 1 h** — Phase 0 eval, data prep, plotting — don't wait on an H200;
use `--partition=gpu-short`, or grab a less-queued GPU from the `sharing` partition,
e.g. `--partition=sharing --gres=gpu:h100:1` (also `gpu:a100:1`, `gpu:l40s:1`). An H100
on `sharing` often *starts* sooner and finishes a short job faster than waiting for an
H200 slot. Keep the H200 (`--partition=gpu --gres=gpu:h200:1`) for the long training
phases (1–3, up to 8 h) where the 141 GB HBM and native FP8 actually matter. Check live
availability with the GPU Monitor portal or `sinfo -s`.

---

## 4. Local vs cluster — the iron rule

Your VS Code remote is flaky, so treat the laptop and the cluster as two checkouts
of the same git repo and **sync by hand through the cluster portal/terminal.**

- **Local (Windows, this folder):** edit code, write configs, `git commit`,
  `git push`. **Never** run training/eval/vLLM locally — there is no GPU here and
  the scripts assume one.
- **Cluster (Linux, Slurm):** `git pull`, `sbatch`, read logs, `git push` any
  result artifacts (metrics JSON, small plots — *not* checkpoints).

```
  Windows (edit + git)                Cluster login node (git + slurm)
  ┌───────────────────┐   git push    ┌───────────────────────────┐
  │ README, *.py,      │ ───────────▶  │ git pull                  │
  │ *.sbatch, configs  │               │ sbatch slurm/phaseX.sbatch│
  │                    │ ◀───────────  │ → compute node (1×H200)   │
  └───────────────────┘   git pull     │ logs/, results/*.json     │
        (results only)                 └───────────────────────────┘
```

Anything that needs the GPU lives behind `sbatch`. Nothing heavy runs interactively
(login nodes usually forbid it anyway).

---

## 5. Repo layout

```
on_policy_distillation/
├── README.md                  ← this file
├── requirements.txt
├── env/
│   └── setup_env.sbatch        # one-off: build the conda/venv on the cluster
├── data/
│   └── prepare_data.py         # download + cache math train + AIME eval (run via sbatch)
├── src/
│   ├── eval_aime.py            # vLLM batch-gen + math-verify scoring (Phase 0 & re-eval)
│   ├── reward_math.py          # verifiable reward fn (used by GRPO)
│   ├── train_grpo.py           # Phase 1 — RLVR via trl.GRPOTrainer
│   ├── train_gkd.py            # Phase 2/3 — on-policy distillation via trl GKDTrainer
│   └── handrolled_opd.py       # Appendix A — vLLM rollout + reverse-KL, no TRL (bonus)
├── configs/
│   ├── grpo.yaml
│   └── gkd.yaml
├── slurm/
│   ├── phase0_eval.sbatch
│   ├── phase1_grpo.sbatch
│   ├── phase2_distill.sbatch
│   └── phase3_distill_external.sbatch
├── logs/                       # slurm stdout/err (gitignored)
└── results/                    # metrics JSON + plots (commit these)
```

`.gitignore` must exclude `logs/`, `*.safetensors`, `checkpoints/`, and the HF
cache. Commit only `results/*.json` and small `*.png`.

---

## 6. Environment (built once, on the cluster)

`requirements.txt` (pin versions when it works — these are floors):

```
torch>=2.4
transformers>=4.51        # Qwen3 needs >=4.51
trl>=0.12                 # GKD lives in trl.experimental.gkd
peft>=0.13
accelerate>=1.0
datasets>=3.0
vllm>=0.8.5               # Qwen3 + FP8 on Hopper
math-verify              # answer checking for the RLVR reward
flash-attn --no-build-isolation
```

`env/setup_env.sbatch` (run with `sbatch` once; flash-attn would need a GPU node to
build, but it's optional — see below). **Use conda, not venv:** the cluster's system
Python is 3.9, but `math-verify` requires ≥3.10, so we build a 3.11 conda env.

```bash
#!/bin/bash
#SBATCH --job-name=opd-setup
#SBATCH --partition=gpu              # <-- your H200 partition
#SBATCH --gres=gpu:h200:1            # <-- your gres syntax
#SBATCH --time=03:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/setup_%j.out

module load cuda/12.8.0          # matches torch 2.8 cu128 build
module load miniconda3/25.9.1    # 3.11 env clears the math-verify >=3.10 floor
eval "$(conda shell.bash hook)"
conda create -y -p "$HOME/.conda/envs/opd" -c conda-forge --override-channels python=3.11  # conda-forge avoids Anaconda ToS gate
conda activate "$HOME/.conda/envs/opd"
pip install --upgrade pip
pip install -r requirements.txt
pip install flash-attn --no-build-isolation || echo "flash-attn skipped (optional) — sdpa"
python -c "import torch, vllm, trl, transformers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

Set the HF cache on **scratch**, not home — model weights (~24 GB for 4B+8B, +~29 GB
if you add the 14B) will blow a home quota. On Explorer, `/scratch/$USER` has the room:

```bash
export HF_HOME=/scratch/$USER/hf_cache   # put this in every sbatch + ~/.bashrc
```

Note: scratch is auto-purged after a period of inactivity — fine for an active project
(weights re-touch each run), but re-download if you return after a long gap.

---

## 7. Data (run via sbatch, not locally)

`data/prepare_data.py` should:

- **RLVR/distillation training prompts:** a verifiable-answer math set. Good picks:
  `HuggingFaceH4/MATH` (train split) or a competition-math set such as
  `agentica-org/DeepScaleR-Preview-Dataset` / `open-r1/OpenR1-Math-220k`.
  Downsample to **~3–5k prompts** — distillation reuses prompts happily (the blog
  shows it can even match the teacher from a *single* prompt across many steps), and
  small keeps you inside one job.
- **Eval (held out):** AIME'24 + AIME'25, e.g. `HuggingFaceH4/aime_2024` and an
  AIME'25 set (`opencompass/AIME2025` or `yentinglin/aime_2025`). Keep these
  strictly out of training.
- Each example needs `{"problem": ..., "answer": <ground truth>}`. Cache to `data/`.

---

## 8. Phase 0 — baseline (establish the gap honestly)

Measure your own numbers for student + every candidate teacher on the *same* harness.
`src/eval_aime.py`: load with vLLM, batch-generate (non-thinking,
`max_new_tokens=8192`, the boxed-answer prompt), score with `math-verify`, write
`results/phase0_<model>.json`.

> **Measured lesson (don't skip):** the token budget *is* the eval. At
> `max_new_tokens=2048` the 4B scored **16.7%** on AIME'25 with ~37% of solutions
> truncated before `\boxed{}`; at **8192** it scored **43.3%** (≈ the official ~47).
> The 8B truncated 53–67% at 2048 — its 2048 number is pure noise. So the fixed
> ruler is **8192**, and the same truncation logic forces the *training* completion
> caps up too (see §9/§10) — a 1024 cap would zero out the reward signal.

```bash
# slurm/phase0_eval.sbatch  (sketch — same #SBATCH header as setup)
module load miniconda3/25.9.1; eval "$(conda shell.bash hook)"; conda activate "$HOME/.conda/envs/opd"; export HF_HOME=/scratch/$USER/hf_cache
for M in Qwen/Qwen3-4B-Instruct-2507 Qwen/Qwen3-8B ; do
  python src/eval_aime.py --model "$M" --bench aime24,aime25 \
         --max-new-tokens 8192 --out results/phase0
done
```

You now know the real student baseline and whether any teacher actually beats it.
**This number is your ruler for Phases 1–3.** Measured 4B baseline (this harness,
8192, greedy): AIME'25 **43.3%**.

---

## 9. Phase 1 — RLVR (GRPO) → goal #2

Same loop shape as distillation, but the reward is **1.0 if the final boxed answer
matches ground truth, else 0.0** — sparse and verifiable. This independently lifts
AIME and is where your clean, attributable improvement comes from.

`src/reward_math.py`:

```python
from math_verify import parse, verify
def math_reward(completions, answer, **kwargs):
    out = []
    for c, gt in zip(completions, answer):
        try:
            out.append(1.0 if verify(parse(gt), parse(c)) else 0.0)
        except Exception:
            out.append(0.0)
    return out
```

`src/train_grpo.py` (TRL, vLLM-accelerated):

```python
from trl import GRPOConfig, GRPOTrainer
from reward_math import math_reward
# load Qwen3-4B-Instruct-2507 + LoRA (peft), dataset of {"prompt","answer"}
cfg = GRPOConfig(
    output_dir="checkpoints/phase1_grpo",
    use_vllm=True, vllm_mode="colocate",     # fast on-policy sampling on the same GPU
    num_generations=8,                       # group size G
    per_device_train_batch_size=8,
    max_prompt_length=1024, max_completion_length=3072,  # 1024 truncates -> reward~0
    learning_rate=1e-6, max_steps=150, save_steps=50,
    bf16=True, gradient_checkpointing=True,
)
GRPOTrainer(model=model, reward_funcs=math_reward, args=cfg,
            train_dataset=ds, peft_config=lora).train()
```

```bash
# slurm/phase1_grpo.sbatch  (#SBATCH: gpu:h200:1, time=08:00:00, mem=96G)
module load miniconda3/25.9.1; eval "$(conda shell.bash hook)"; conda activate "$HOME/.conda/envs/opd"; export HF_HOME=/scratch/$USER/hf_cache
python src/train_grpo.py --config configs/grpo.yaml
python src/eval_aime.py --model checkpoints/phase1_grpo --bench aime24,aime25 --out results/phase1
```

**Watch:** mean reward should climb; AIME should rise a few points over the Phase-0
baseline. You'll also *feel* the sparsity — reward is noisy, needs the group
baseline (GRPO) and many rollouts to get signal. That contrast is the lesson.

**Result (measured, 2026-06-08) — a null result, and why it's the *expected* one.**
Two runs: (a) MATH train set → reward saturated at 1.0, `frac_reward_zero_std≈1.0`,
~zero gradient (the prompts were too easy → no GRPO advantage); (b) DeepScaleR train
set → real reward variance and gradient, but **no AIME gain**. avg@4 (k=4, temp 0.7,
8192 ruler), 60 questions:

| | aime24 | aime25 | combined |
|---|---|---|---|
| Baseline 4B | 58.33 | 41.67 | **50.00** |
| Phase-1 GRPO | 53.33 | 43.33 | **48.33** |

The −1.67 combined is ~1 question out of 60 — well inside eval noise — and the two
subsets move in *opposite* directions (the fingerprint of noise, not a training
effect). The earlier greedy single-shot −5 was mostly measurement noise; avg@4
collapses it. **Conclusion: `Qwen3-4B-Instruct-2507` is already fully RL-post-trained
on math; naive single-reward LoRA-GRPO on the same verifiable distribution has
nothing to add, and can mildly *shift the policy away* from its optimized state
(slightly higher truncation, no accuracy gain).** This is exactly the §1 prediction —
the strong 2507 student leaves almost no headroom for a crude RLVR pass. We accept it
as the Phase-1 finding rather than burn slots chasing a within-noise gain.

---

## 10. Phase 2 — on-policy distillation → goal #1

> **Plan change (2026-06-08), forced by the Phase-1 null result.** The original Phase 2
> was *self-distill* from the Phase-1 GRPO checkpoint, to reproduce the blog's "recover
> the RL gains in far fewer steps" headline. But Phase 1 produced **no** gain to recover
> (§9), so self-distillation from it would just reproduce the baseline. Storage is also no
> longer the binding constraint (HF cache on scratch, §6) — VRAM on one H200 is — so the
> teacher choice is now free to be much bigger. We therefore distill from a real,
> already-post-trained teacher:
>
> - **Phase 2 = distill from `Qwen/Qwen3-30B-A3B-Instruct-2507`.** This is now where most
>   of the reproduction credit lands. It is the cleanest teacher for our constraints:
>   same Qwen3 family (shared 151,936 vocab → logit KL valid); **non-thinking** (matches
>   the 8192 ruler, no mode mismatch); **AIME25 61.3 vs the student's 47.4 → a genuine
>   +14 gap to teach**; and **no SFT** (it already plays the blog's "teacher" role that
>   their 32B+SFT filled). It's MoE — 30.5B total but only **3.3B active/token** — so the
>   teacher forward pass is cheap (~3B-dense cost), and bf16 weights (~61 GB) sit
>   co-resident with the 4B student inside 141 GB (FP8 ~31 GB if you want margin).
>   Student = a **fresh** `Qwen3-4B-Instruct-2507`.
> - **Phase 3 = frozen for now.** Phase 2's 30B teacher is expected to carry the headline,
>   so the optional bigger-teacher stress-test is parked. (The 235B-A22B-Instruct-2507
>   scores higher still but won't fit one H200 even in FP8; a Phase-3 revisit would mean
>   thinking-mode or multi-GPU.)

(Historical self-distill recipe, kept for reference.) Teacher = your **Phase-1 GRPO
checkpoint**; student = a **fresh** `Qwen3-4B-Instruct-2507`. Same family → shared
vocab → logit KL is valid. Pure on-policy + reverse KL ≈ the blog's setup:

- `lmbda=1.0` → 100% student-generated (on-policy) sequences
- `beta=1.0` → reverse KL

`src/train_gkd.py`:

```python
from trl.experimental.gkd import GKDConfig, GKDTrainer   # NOTE: experimental path
# student = fresh Qwen3-4B-Instruct-2507 (+LoRA);  dataset = chat-formatted math prompts
cfg = GKDConfig(
    output_dir="checkpoints/phase2_distill",
    teacher_model_name_or_path="checkpoints/phase1_grpo",  # the RL'd 4B
    lmbda=1.0, beta=1.0, temperature=1.0,
    max_new_tokens=2048,   # 1024 truncates most solutions (see §8); 2048 balances HF-generate speed
    per_device_train_batch_size=8, learning_rate=1e-6,
    max_steps=100, save_steps=25, bf16=True, gradient_checkpointing=True,
)
GKDTrainer(model=student, args=cfg, train_dataset=ds,
           processing_class=tok, peft_config=lora).train()
```

```bash
# slurm/phase2_distill.sbatch  (#SBATCH: gpu:h200:1, time=08:00:00, mem=96G)
module load miniconda3/25.9.1; eval "$(conda shell.bash hook)"; conda activate "$HOME/.conda/envs/opd"; export HF_HOME=/scratch/$USER/hf_cache
python src/train_gkd.py --config configs/gkd.yaml
python src/eval_aime.py --model checkpoints/phase2_distill --bench aime24,aime25 --out results/phase2
```

**Success looks like:** Phase 2 reaches ~Phase-1's AIME in many fewer steps than
Phase 1 needed (the blog reports ~7–10× fewer gradient steps for self-distillation).
That step-count ratio — not the absolute score — is the result you're reproducing.
Plot **AIME vs. gradient step** for Phase 1 and Phase 2 on the same axes.

**Result (measured, 2026-06-21) — flat, a second null.** The run finished 300 steps
(~0.6 epoch, LoRA r32, FP8 teacher — the bf16 teacher OOM'd at step 138 on the dense
full-vocab KL tensor; HF-generate decode was so slow the job auto-resumed across ~8 Slurm
slots, with per-step time ballooning 1.6 s → 600+ s as rollouts lengthened). avg@4 (k=4,
temp 0.7, 8192 ruler), 60 questions:

| | aime24 | aime25 | combined |
|---|---|---|---|
| Baseline 4B | 58.33 | 41.67 | **50.00** |
| Phase-2 OPD | 53.33 | 44.17 | **48.75** |

The −1.25 combined is within eval noise — the teacher's real +14 AIME25 gap did **not**
transfer. (The committed *greedy* `phase2_*` JSON, 43.33, is a stale 100-step intermediate
checkpoint with the bf16 teacher — not the final model; compare only avg@4.) Two suspects,
both consistent with §1's thesis: (a) the **2048 training cap vs the 8192 eval ruler** —
on-policy rollouts were truncated during distillation, re-introducing the short-bias the
eval punishes; (b) **reverse-KL into a rank-32 LoRA on an already-strong student** can only
mode-seek *within* the student's existing support, so a teacher whose edge lives in
reasoning the 4B can't represent has little it can actually hand over. Net: like Phase 1,
no attributable gain.

---

## 11. Phase 3 (optional) — distill from an external teacher → your worry, tested

Now test "*does on-policy distillation work if the teacher itself is limited?*"
Student = `Qwen3-4B-Instruct-2507`; teacher = `Qwen/Qwen3-8B` (or `Qwen/Qwen3-14B`
for a real gap). Same `train_gkd.py`, just point `teacher_model_name_or_path` at the
external model.

```bash
# slurm/phase3_distill_external.sbatch
python src/train_gkd.py --config configs/gkd.yaml \
       --teacher Qwen/Qwen3-8B           # swap to Qwen/Qwen3-14B for a stronger teacher
python src/eval_aime.py --model checkpoints/phase3_distill --bench aime24,aime25 --out results/phase3
```

**Expected, and the point:** with the **8B** teacher the student **won't improve** —
and our Phase-0 measurement makes this even sharper than the README originally guessed:
the non-thinking 8B (21.67% combined) is actually **worse** than the 4B student (51.67%),
so reverse-KL would drag the student *down*, not up. For a genuine positive gap use the
**14B** teacher (affordable now the cache is on scratch), **or go larger still (>20B)** —
disk is no longer the limit, so a teacher big enough to clearly beat the 4B in
non-thinking mode is the cleanest way to *demonstrate* distillation working. (Alternative:
run the 8B in *thinking* mode, but then the student must also think — Appx B — which
breaks the fixed non-thinking 8192 ruler, so prefer a bigger non-thinking teacher.)
Conclusion to write up: *what bounds the student is the teacher–student gap, not the act
of distilling.*

---

## 12. Reading results / deliverables

> **Progress** (2026-06-21): ✅ env · ✅ data · ✅ **Phase 0 done** (ruler = 8192 tok;
> 4B = 51.67% combined, 8B = 21.67%) · ✅ **Phase 1 done — null result** (RLVR/GRPO:
> avg@4 50.00 baseline → 48.33, within noise; the 2507 student is already RL-saturated,
> §9) · ✅ **Phase 2 done — flat/null result** (on-policy distill from
> `Qwen3-30B-A3B-Instruct-2507-FP8`, 300 steps: avg@4 50.00 baseline → **48.75**, within
> noise — the +14 teacher gap did *not* transfer; suspects = 2048 train-cap vs 8192
> eval-ruler mismatch + LoRA-r32 capacity, §10) · ❄️ Phase 3 frozen.
>
> **Project status:** all three phases land at ~48–52% combined AIME — no attributable
> movement off baseline. The honest write-up is therefore a **boundary-conditions** study
> ("when do GRPO / on-policy distillation fail to help?"), not a blog reproduction. See the
> three measured failure modes in §9 (RL on a saturated student), §10 (reverse-KL into a
> small LoRA + train/eval length mismatch), and §1/§11 (transfer is bounded by the
> teacher–student gap *that the student can represent*, not by the act of distilling).

Commit to `results/`:

- `phase{0..3}_*.json` — AIME'24/'25 pass@1 per checkpoint.
- A plot of **AIME vs gradient step** for Phase 1 (RL) and Phase 2 (distill) →
  shows the efficiency gap.
- A short table: baseline → RLVR → self-distill → external-teacher distill.

If your AIME numbers are jumpy (30 Q each), report 24+25 combined and/or
avg@k (sample k=4 and average) — cheaper than chasing a bigger eval set.

---

## 13. Post-hoc diagnostic — why Phase 2 didn't transfer (overlap ratio)

After the flat Phase-2 result (§10), a recent paper — Li et al., *Rethinking On-Policy
Distillation* (THUNLP, **arXiv:2604.13016**) — gave a clean way to test *why*. It shows OPD
succeeds only when **(i)** student and teacher share compatible **thinking patterns** (high
overlap of their top-k token sets on student-visited states) and **(ii)** the teacher carries
genuinely **new knowledge** beyond what the student has — a higher *benchmark score* from a
same-family, same-recipe teacher is **not** enough. Failed runs show high overlap but a
per-token distribution the student already matches, so reverse-KL has nothing to push.

`src/diag_overlap.py` (run via `slurm/diag_overlap.sbatch`; light enough for 1× H100, < 1 h)
reproduces the paper's metrics on our pair: it samples student rollouts, runs one
teacher-forcing forward pass of both the 4B student and the 30B-A3B teacher over the student's
own tokens, and averages the alignment metrics over student-visited states.

**Measured** (fresh `Qwen3-4B-Instruct-2507` vs `Qwen3-30B-A3B-Instruct-2507-FP8`; 64 rollouts /
~57k positions, k=16):

| metric | value | reading |
|---|---|---|
| `overlap_ratio` | **0.67** | HIGH (paper: mismatch ≈0.28, healthy 0.55–0.91) → condition (i) satisfied |
| `entropy_gap` | **0.09** nats | tiny — near-identical, very peaked confidence (both entropies ≈0.20) |
| mass on shared tokens | **≈99.8%** | on the student's own states, ~all of both models' mass sits on the *same* tokens |
| `overlap_advantage` | **−0.15** | residual reverse-KL on the shared set — *not* zero |

**Reading.** Condition (i) is clearly **satisfied** (high overlap, matched confidence) — so the
missing SFT cold-start (the paper's fix for *pattern mismatch*) was never our problem, and
dropping SFT (§0) was not the mistake. And on the states the student actually visits, the two
models put ~99.8% of their mass on the **same tokens**: the teacher's +14 AIME edge barely
surfaces as a token-level distribution the student could absorb on-policy — exactly the paper's
condition-(ii) failure (a same-family scaled sibling carries little *new* transferable signal).
**Honest caveats:** `overlap_advantage` is **−0.15, not zero** — there *is* a small residual
disagreement (comparable to the ~0.20-nat entropy), just diluted across the ~99.8% agreement;
and this is a single snapshot, not the train-dynamics curve the paper uses to separate
"converges" from "stagnates." So treat it as strong **corroboration** of the §10 read, not
proof. It sharpens §1/§11: *what bounds the student is whether the teacher has new knowledge the
student can represent — not its benchmark score, and not the act of distilling.*

---

## Appendix A — hand-rolled loop (bonus, no TRL)

This is the version that most closely mirrors the blog's pseudocode and is **faster**
than TRL GKD because sampling uses vLLM. Skip unless you want the deeper understanding
(it's the bonus you flagged). Single H200, all of this behind `sbatch`.

Per step:

1. **Sample** N student rollouts with vLLM (`SamplingParams(..., logprobs=...)`) →
   keep the student token logprobs `logp_student`.
2. **Score with teacher.** Feed the *exact* student token sequences to the teacher
   and read `prompt_logprobs` (vLLM) — the OSS equivalent of Tinker's
   `compute_logprobs`. This is one teacher forward pass, no backprop. → `logp_teacher`.
3. **Reverse-KL advantage:** `advantage_t = -(logp_student_t - logp_teacher_t)`
   per token. (Discount 0 → optimize each token independently, as the blog does.)
4. **Update** the student with an importance-sampling policy-gradient step
   (HF forward/backward on the student + LoRA), using `advantage_t`.

Two practical notes: keep one vLLM engine for the student (refresh its weights from
the LoRA-merged student every K steps) and one for the teacher; and cap sequence
length short — distillation learns fine at shorter context since there's no hard
end-of-sequence reward cliff like RL has. Swapping step 3's reward for the §9
verifiable reward turns this exact scaffold into your GRPO loop — which is the whole
"distillation is RL with a dense reward" point.

---

## Appendix B — gotchas

- **`trl.experimental.gkd` is experimental** — the import path and arg names move
  between TRL versions. Pin your TRL version once it runs; re-check the GKD doc if
  an upgrade breaks it.
- **GKD needs a shared vocab** (teacher/student same tokenizer). Stay in the Qwen3
  family. Don't mix in a non-Qwen teacher.
- **GKD on-policy sampling uses HF `generate`** (slow). That's the main reason
  Phase 2/3 are capped at ~100 steps; use Appendix A if you want vLLM-speed.
- **Thinking vs non-thinking:** keep both models non-thinking for matching formats
  and short, cheap outputs. If you ever enable thinking for a stronger teacher,
  the student must also think or the per-token KL on the missing `<think>` tokens
  goes haywire.
- **FP8 teachers** load fine on H200 (Hopper, w8a8) and ~halve disk/VRAM with
  negligible logprob error — a good lever if you add the 14B teacher.
- **8-hour wall + LoRA checkpoints:** because adapters are small, `save_steps`
  checkpoints fit easily and a job that times out can resume from the last adapter.
  Never try to persist full-weight checkpoints — they'll blow the 80 GB quota.
- **Don't run anything in this list on your laptop.** No GPU locally; every command
  above is a cluster `sbatch`.

---

## References

- Lu, K. & Thinking Machines Lab. *On-Policy Distillation*, 2025.
- Agarwal et al. *On-Policy Distillation of Language Models* (GKD), 2023 — the method
  TRL's GKDTrainer implements.
- Qwen3 Technical Report, 2025; Qwen3-4B-Instruct-2507 / Qwen3-8B model cards.
- TRL docs: GKDTrainer, GRPOTrainer.
- Li et al. *Rethinking On-Policy Distillation of Large Language Models: Phenomenology,
  Mechanism, and Recipe*, 2026 (arXiv:2604.13016; code: github.com/thunlp/OPD) — the two
  OPD success conditions and the overlap diagnostic reproduced in §13.
