#!/bin/bash
#SBATCH --job-name=opd-phase0
# This is a SHORT (<1h) job — the H200 queue is contended, so consider a faster-to-
# schedule GPU instead of waiting: e.g. `--partition=gpu-short`, or grab one from the
# sharing pool: `--partition=sharing --gres=gpu:h100:1` (also gpu:a100:1 / gpu:l40s:1).
#SBATCH --partition=gpu              # <-- H200 partition; or gpu-short / sharing (see above)
#SBATCH --gres=gpu:h200:1            # <-- or gpu:h100:1 / gpu:a100:1 on sharing
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/phase0_%j.out
#SBATCH --error=logs/phase0_%j.err

# Phase 0 — baseline eval (README §8). Establishes your own ruler.
set -euo pipefail

module load cuda/12.8.0
module load miniconda3/25.9.1
eval "$(conda shell.bash hook)"
conda activate "$HOME/.conda/envs/opd"
export HF_HOME=$HOME/hf_cache

for M in Qwen/Qwen3-4B-Instruct-2507 Qwen/Qwen3-8B ; do
  python src/eval_aime.py --model "$M" --bench aime24,aime25 \
         --max-new-tokens 2048 --out results/phase0
done
