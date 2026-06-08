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

**Final Phase-0 baselines (8192, greedy, n=30/bench):**
- Qwen3-4B-Instruct-2507: AIME'24 **56.67**, AIME'25 **46.67**, combined **51.67** (beats official ~47.4 → honest ruler). Truncation still 27–40%.
- Qwen3-8B (non-thinking): AIME'24 23.33, AIME'25 20.0, combined **21.67**, truncation 33–50%.

**Key strategic finding:** the non-thinking 8B is *worse* than the 4B student (21.67 vs 51.67) — confirms README §1 thesis. So Phase 3 "external teacher = 8B" won't help (teacher worse than student → reverse-KL drags student down). For a positive Phase-3 gap use the **14B** (now affordable with cache on scratch) or 8B in thinking mode (student must also think, Appx B). Phase 2 self-distill (teacher = RL'd 4B) is unaffected.

Budget is bounded by GPU KV cache + wall-time, NOT host RAM — at 8192 there was ~105 GiB KV cache free and 18x concurrency headroom, so 16384/32768 are feasible if ever needed. Kept 8192 for consistency. Relates to [[cluster-env-setup]].
