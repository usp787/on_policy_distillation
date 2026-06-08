---
name: project-progress
description: on_policy_distillation — phase-by-phase progress tracker (where to pick up)
metadata:
  type: project
---

Progress on the on_policy_distillation project (as of 2026-06-07):

- ✅ **Env setup** — conda 3.11 env, pinned stack, scratch HF cache. See [[cluster-env-setup]].
- ✅ **Data prep** — `data/train_math.json` (746 prompts, HuggingFaceH4/MATH — fewer than the 3-5k target; can re-pull from DeepScaleR/OpenR1 for more), `data/aime24.json`, `data/aime25.json` (30 each).
- ✅ **Phase 0 baseline** — done, committed to `results/`. Ruler = 8192 tokens. 4B = **51.67% combined**, 8B = 21.67%. See [[token-budget-and-baseline]] for numbers + the strategic finding (8B is a worse teacher than the student).
- ⬜ **Phase 1 — RLVR/GRPO** — NEXT. `sbatch slurm/phase1_grpo.sbatch` (8h H200 job, max_steps=150, completion cap 3072). Watch reward climb off ~0; auto-re-evals to `results/phase1`. Teacher checkpoint for Phase 2 comes from here (`checkpoints/phase1_grpo`).
- ⬜ **Phase 2 — self-distill (GKD)** — teacher = Phase-1 RL'd 4B, fresh 4B student. The headline result (recover RL gains in far fewer steps). GKD completion cap = 2048 (HF-generate is slow).
- ⬜ **Phase 3 (optional)** — external teacher. NOTE: 8B is worse than student → won't help; use 14B for a real gap (affordable now cache is on scratch).

Open watch-items for Phase 1+: confirm trl 0.24 GRPO reward wiring + LoRA; if reward stays 0, completion cap still too short or reward parsing issue. GKD import path is `trl.experimental.gkd` (may have moved in 0.24 — train_gkd.py has a fallback).
