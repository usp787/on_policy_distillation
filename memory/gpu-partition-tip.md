---
name: gpu-partition-tip
description: Explorer GPU partition selection — use gpu-short/sharing for short jobs to dodge the H200 queue
metadata:
  type: feedback
---

For Explorer jobs **≤ 1 hour** (e.g. Phase 0 eval, data prep, plotting), don't default to the contended H200 queue — request a faster-to-schedule GPU instead.

**Why:** the H200 is heavily queued (saw 17 H200 jobs pending at once). H200 lives in partitions `{gpu, gpu-interactive, sharing, gpu-short}`. The `sharing` partition also exposes H100, A100 (up to :8), L40S, A6000, etc. — often less backed up. An H100 on `sharing` can start sooner and run a short job faster than waiting for an H200 slot.

**How to apply:**
- Short job, want quick turnaround: `--partition=gpu-short` (keep `--time` short).
- Grab a specific non-H200 GPU: `--partition=sharing --gres=gpu:h100:1` (or `gpu:a100:1`, `gpu:l40s:1`).
- H100 node is `d4041` (`gpu:h100:4`), only in `sharing` — occupied as of 2026-06-07 but may free up; re-check the GPU Monitor / `sinfo`.
- Keep the H200 (`--partition=gpu --gres=gpu:h200:1`) for the long training phases (1/2/3, up to 8 h) where the 141 GB HBM and FP8 matter.

Check live availability: GPU Monitor portal, or `sinfo -s`, or `sinfo -n <node> --Format="Gres:30,GresUsed:30"`. Relates to [[cluster-env-setup]].
