---
name: cluster-env-setup
description: Working cluster env for the on_policy_distillation project (Northeastern Explorer HPC)
metadata:
  type: project
---

The on_policy_distillation project runs on **Northeastern Explorer HPC** (login: explorer.northeastern.edu, user zha.j). Local Windows checkout at c:\Users\usp78\Desktop\on_policy_distillation is edit-only; all GPU work is `sbatch` on the cluster. Repo: github.com/usp787/on_policy_distillation. Synced manually by hand (git push local / git pull cluster).

**Working environment (established 2026-06):**
- Conda env at `$HOME/.conda/envs/opd`, **Python 3.11** (system python is 3.9, but `math-verify` requires >=3.10 — this forced conda over venv).
- Modules: `cuda/12.8.0` (matches torch cu128 build) + `miniconda3/25.9.1`.
- Conda env MUST be created with `-c conda-forge --override-channels` to dodge the Anaconda default-channel ToS gate (`CondaToSNonInteractiveError`).
- Activate in sbatch with: `module load miniconda3/25.9.1; eval "$(conda shell.bash hook)"; conda activate $HOME/.conda/envs/opd`.
- **Pinned stack (requirements.txt):** torch==2.8.0 (cu128), transformers==4.57.6, trl==0.24.0, peft==0.17.1, accelerate==1.10.1, datasets==4.5.0, vllm==0.11.0, math-verify. Unpinned floors on py3.11 pull bleeding-edge (torch 2.11/cu130, trl 1.5, transformers 5.x) that breaks vLLM and the trl 0.x GKD/GRPO APIs — keep these pinned.
- **flash-attn is skipped** (build fails; not worth it). transformers falls back to sdpa, vLLM uses its own kernels — negligible perf cost since seqs are short and vLLM is unaffected.
- Set `export HF_HOME=/scratch/$USER/hf_cache` in every sbatch (NOT home — home quota is ~40GB and model weights blow it; scratch `/scratch/zha.j` has room but auto-purges after inactivity). Conda env stays in home (~16GB); hf_cache (~23GB of weights) lives on scratch.
- CUDA modules available: 12.1.1, 12.3.0, 12.8.0, 13.2.0. Cluster uses old Environment Modules (no `module spider`; use `module avail`).
