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
source "$HOME/.venvs/opd/bin/activate"
export HF_HOME=$HOME/hf_cache

python src/train_grpo.py --config configs/grpo.yaml
python src/eval_aime.py --model checkpoints/phase1_grpo \
       --bench aime24,aime25 --out results/phase1
