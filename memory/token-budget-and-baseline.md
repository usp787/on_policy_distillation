---
name: token-budget-and-baseline
description: Phase-0 finding — token budget dominates AIME score; fixed ruler is 8192; measured 4B baseline
metadata:
  type: project
---

**Phase-0 finding (2026-06-07):** on this AIME harness the generation token budget *is* the score. Qwen3-4B-Instruct-2507 (greedy, non-thinking) on AIME'25: **16.7% at max_new_tokens=2048 (~37% truncated) → 43.3% at 8192**. The official card says ~47.4, so 8192 ≈ honest. Qwen3-8B at 2048 truncated 53–67% (its 2048 number is meaningless noise — must re-eval at 8192).

**Decisions locked in:**
- **Fixed eval ruler = `max_new_tokens=8192`**, constant across all phases (eval_aime.py default + phase0_eval.sbatch). Don't compare scores across different budgets.
- Training completion caps raised off the old 1024 (which would truncate→reward≈0→no learning): GRPO `max_completion_length=3072` (vLLM-fast), GKD `max_new_tokens=2048` (HF-generate is slow, so compromise). Tunable: raise if reward stays ~0, lower if 8h wall is blown.
- eval_aime.py gained `truncated_pct` (per-bench) and `--debug-samples N` (dumps raw completions to `<out>.debug.json`) — use these to diagnose low scores.

Measured 4B baseline ruler: **AIME'25 43.3%** (8192, greedy, n=30). Relates to [[cluster-env-setup]].
