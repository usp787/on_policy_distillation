#!/bin/bash
#SBATCH --job-name=opd-phase1
#SBATCH --partition=gpu              # <-- your H200 partition
#SBATCH --gres=gpu:h200:1            # <-- your gres syntax
#SBATCH --time=08:00:00
#SBATCH --mem=96G
#SBATCH --output=logs/phase1_%j.out
#SBATCH --error=logs/phase1_%j.err

# Phase 1 — RLVR (GRPO), then re-eval. README §9.
set -euo pipefail

module load cuda/12.8.0
module load miniconda3/25.9.1
eval "$(conda shell.bash hook)"
conda activate "$HOME/.conda/envs/opd"
export HF_HOME=/scratch/$USER/hf_cache   # off the home quota (model weights live here)

python src/train_grpo.py --config configs/grpo.yaml
python src/eval_aime.py --model checkpoints/phase1_grpo \
       --bench aime24,aime25 --out results/phase1
