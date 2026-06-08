#!/bin/bash
#SBATCH --job-name=opd-evalavg4
# SHORT (<1h) job — same scheduling tip as phase0: the H200 queue is contended, so
# consider `--partition=gpu-short`, or grab one from the sharing pool:
# `--partition=sharing --gres=gpu:h100:1` (also gpu:a100:1 / gpu:l40s:1).
#SBATCH --partition=gpu              # <-- H200 partition; or gpu-short / sharing (see above)
#SBATCH --gres=gpu:h200:1            # <-- or gpu:h100:1 / gpu:a100:1 on sharing
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/evalavg4_%j.out
#SBATCH --error=logs/evalavg4_%j.err

# Step A — avg@4 re-eval of baseline + Phase-1 checkpoint on the SAME ruler (8192),
# to damp the 30-question greedy-pass@1 noise before judging the RLVR delta.
# k=4 @ temp 0.7. Compare truncated_pct between the two to check the short-cap bias.
set -euo pipefail

module load cuda/12.8.0
module load miniconda3/25.9.1
eval "$(conda shell.bash hook)"
conda activate "$HOME/.conda/envs/opd"
export HF_HOME=/scratch/$USER/hf_cache   # off the home quota (model weights live here)

python src/eval_aime.py --model Qwen/Qwen3-4B-Instruct-2507 \
       --bench aime24,aime25 --k 4 --temperature 0.7 --out results/phase0_avg4
python src/eval_aime.py --model checkpoints/phase1_grpo \
       --bench aime24,aime25 --k 4 --temperature 0.7 --out results/phase1_avg4
