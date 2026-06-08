---
name: project-progress
description: on_policy_distillation — phase-by-phase progress tracker (where to pick up)
metadata:
  type: project
---

Progress on the on_policy_distillation project (as of 2026-06-08):

- ✅ **Env setup** — conda 3.11 env, pinned stack, scratch HF cache. See [[cluster-env-setup]].
- ✅ **Data prep** — `data/train_math.json` re-pulled from `agentica-org/DeepScaleR-Preview-Dataset` (~4k competition-math prompts; the original 746 HuggingFaceH4/MATH set was too easy → GRPO reward saturated at 1.0 → zero advantage), `data/aime24.json`, `data/aime25.json` (30 each).
- ✅ **Phase 0 baseline** — Ruler = 8192 tokens. 4B = **51.67% combined**, 8B = 21.67%. See [[token-budget-and-baseline]] (8B non-thinking is a *worse* teacher than the student).
- ✅ **Phase 1 — RLVR/GRPO — DONE, NULL RESULT.** DeepScaleR gave real reward variance/gradient (vs MATH which gave none), but no AIME gain. avg@4 (k=4, temp 0.7, 8192): baseline **50.00** combined → Phase-1 **48.33** (−1.67 ≈ 1 question/60, within noise; subsets move opposite directions = noise signature). **Finding: `Qwen3-4B-Instruct-2507` is already fully RL-post-trained on math; naive single-reward LoRA-GRPO adds nothing and can mildly shift the policy *away* from its optimized state.** Accepted as the finding, not chasing a within-noise gain. Adapter at `checkpoints/phase1_grpo` (no longer needed as a teacher — self-distill plan dropped).
- 🔄 **Phase 2 — distill from `Qwen/Qwen3-30B-A3B-Instruct-2507` (GKD).** Teacher = 30B MoE (3.3B active), non-thinking, AIME25 61.3 vs student 47.4 (+14 gap), no SFT, vocab matched (logit KL valid — confirmed, no error). Pipeline works (loss dropped 0.78→0.49). **First attempt (100 steps) was badly UNDER-TRAINED**: epoch 0.2, ~16 min of an 8h slot, loss still falling → student perturbed but hadn't absorbed the teacher; greedy combined fell to 43.33 (vs baseline 51.67 greedy / 50.00 avg@4). This is NOT evidence SFT is needed — the instruct student is already in-support (SFT's role is base→support only; that test would need Qwen3-4B-Base). Re-run config (in `configs/gkd.yaml`): max_steps 100→600, lr 1e-6→5e-6, max_new_tokens 2048→4096 (the 2048 cap repeated Phase-1's short-bias vs the 8192 eval), save_steps 100; auto-eval switched to avg@4 → `results/phase2_avg4`. `sbatch slurm/phase2_distill.sbatch`. If avg@4 still ≤ baseline after proper training, then question whether LoRA r32 / 4B capacity caps the transfer.
- ❄️ **Phase 3 — FROZEN.** Parked while Phase 2's 30B teacher carries the reproduction. (235B-A22B-2507 scores higher but won't fit one H200 even FP8; a revisit means thinking-mode or multi-GPU.) Discarded options: 8B (worse than student in non-thinking), 14B/32B April hybrids (no non-thinking math edge).

Watch-items: GKD import path is `trl.experimental.gkd` (train_gkd.py has a top-level fallback). Eval auto-merges LoRA adapters into a `<ckpt>_merged` full-model dir before vLLM (eval_aime.py); these ~8GB merged dirs now write under `$HF_HOME/opd_merged` on **scratch** (storage rule), not the home node. Still reclaim Phase-1's `checkpoints/phase1_grpo_merged` (~8GB) + intermediate `checkpoint-*/` on the home node; the small final adapter is the keep-worthy artifact.
