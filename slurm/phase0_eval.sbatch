#!/bin/bash
#SBATCH --job-name=opd-phase0
#SBATCH --partition=gpu              # <-- your H200 partition
#SBATCH --gres=gpu:h200:1            # <-- your gres syntax
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/phase0_%j.out
#SBATCH --error=logs/phase0_%j.err

# Phase 0 — baseline eval (README §8). Establishes your own ruler.
set -euo pipefail

module load cuda/12.8.0
source "$HOME/.venvs/opd/bin/activate"
export HF_HOME=$HOME/hf_cache

for M in Qwen/Qwen3-4B-Instruct-2507 Qwen/Qwen3-8B ; do
  python src/eval_aime.py --model "$M" --bench aime24,aime25 \
         --max-new-tokens 2048 --out results/phase0
done
