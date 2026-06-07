#!/bin/bash
#SBATCH --job-name=opd-phase3
#SBATCH --partition=gpu              # <-- your H200 partition
#SBATCH --gres=gpu:h200:1            # <-- your gres syntax
#SBATCH --time=08:00:00
#SBATCH --mem=96G
#SBATCH --output=logs/phase3_%j.out
#SBATCH --error=logs/phase3_%j.err

# Phase 3 (optional) — distill from an external teacher. README §11.
# Swap Qwen/Qwen3-8B -> Qwen/Qwen3-14B for a genuine teacher-student gap.
set -euo pipefail

module load cuda/12.8.0
module load miniconda3/25.9.1
eval "$(conda shell.bash hook)"
conda activate "$HOME/.conda/envs/opd"
export HF_HOME=/scratch/$USER/hf_cache   # off the home quota (model weights live here)

python src/train_gkd.py --config configs/gkd.yaml \
       --teacher Qwen/Qwen3-8B \
       --output-dir checkpoints/phase3_distill
python src/eval_aime.py --model checkpoints/phase3_distill \
       --bench aime24,aime25 --out results/phase3
