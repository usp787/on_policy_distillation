#!/bin/bash
#SBATCH --job-name=opd-phase2
#SBATCH --partition=gpu              # <-- your H200 partition
#SBATCH --gres=gpu:h200:1            # <-- your gres syntax
#SBATCH --time=08:00:00
#SBATCH --mem=96G
#SBATCH --output=logs/phase2_%j.out
#SBATCH --error=logs/phase2_%j.err

# Phase 2 — on-policy distillation from the Qwen3-30B-A3B-Instruct-2507 teacher. README §10.
# Teacher (30B MoE, non-thinking, AIME25 61.3) is set in configs/gkd.yaml; it gives a real
# +14 AIME gap over the 4B student, so this is where most of the reproduction credit lands.
# (Self-distill from the Phase-1 4B was dropped — Phase 1 was a null result, nothing to
#  recover.) Storage-sensitive items go to scratch: the ~61 GB teacher downloads under
#  HF_HOME, and eval_aime.py writes its ~8 GB merged student model under HF_HOME too.
set -euo pipefail

module load cuda/12.8.0
module load miniconda3/25.9.1
eval "$(conda shell.bash hook)"
conda activate "$HOME/.conda/envs/opd"
export HF_HOME=/scratch/$USER/hf_cache   # off the home quota (teacher + merged weights live here)

python src/train_gkd.py --config configs/gkd.yaml
# avg@4 (k=4, temp 0.7) so the number is directly comparable to the avg@4 baseline
# (50.00 combined), not the noisy greedy pass@1.
python src/eval_aime.py --model checkpoints/phase2_distill \
       --bench aime24,aime25 --k 4 --temperature 0.7 --out results/phase2_avg4
