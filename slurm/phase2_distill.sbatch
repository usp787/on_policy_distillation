#!/bin/bash
#SBATCH --job-name=opd-phase2
#SBATCH --partition=gpu              # <-- your H200 partition
#SBATCH --gres=gpu:h200:1            # <-- your gres syntax
#SBATCH --time=08:00:00
#SBATCH --mem=96G
#SBATCH --output=logs/phase2_%j.out
#SBATCH --error=logs/phase2_%j.err

# Phase 2 — on-policy distillation (self-distill from the Phase-1 GRPO 4B). README §10.
set -euo pipefail

module load cuda/12.8.0
module load miniconda3/25.9.1
eval "$(conda shell.bash hook)"
conda activate "$HOME/.conda/envs/opd"
export HF_HOME=/scratch/$USER/hf_cache   # off the home quota (model weights live here)

python src/train_gkd.py --config configs/gkd.yaml
python src/eval_aime.py --model checkpoints/phase2_distill \
       --bench aime24,aime25 --out results/phase2
