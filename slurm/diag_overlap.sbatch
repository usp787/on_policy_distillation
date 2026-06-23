#!/bin/bash
#SBATCH --job-name=opd-diag-overlap
# OPD overlap-ratio diagnostic (src/diag_overlap.py; arXiv:2604.13016). This is LIGHT:
# load the 4B (bf16) + 30B-A3B (FP8, ~31 GB) co-resident, sample 64 short rollouts, then one
# teacher-forcing forward of each model per rollout. No training. Fits 1× H100 (80 GB) in <1 h,
# so default to the less-contended `sharing` pool instead of waiting for an H200.
#SBATCH --partition=sharing
#SBATCH --gres=gpu:h100:1            # 80 GB: 31 (FP8 teacher) + 8 (4B student) + logits, comfortable
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/diag_%j.out
#SBATCH --error=logs/diag_%j.err
#
# FALLBACK — if the H100/sharing pool is unavailable or the FP8 teacher won't fit (e.g. you
# bump --max-new-tokens to 2048 and want margin), comment the two #SBATCH lines above and use
# an H200 on the gpu partition instead (141 GB; trivially fits, but it's the contended queue):
#   #SBATCH --partition=gpu
#   #SBATCH --gres=gpu:h200:1
#   #SBATCH --time=02:00:00
set -euo pipefail

module load cuda/12.8.0          # Hopper FP8 kernels (H100 and H200 are both sm_90)
module load miniconda3/25.9.1
eval "$(conda shell.bash hook)"
conda activate "$HOME/.conda/envs/opd"
export HF_HOME=/scratch/$USER/hf_cache   # teacher weights already cached here from Phase 2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # tame the full-vocab logits allocations

# Default: FRESH 4B student vs the 30B teacher = the INITIAL overlap, the paper's predictor of
# whether OPD could ever transfer. Same family/mode/2507 → prediction: HIGH overlap (so condition
# (i) is met) but near-zero overlap-advantage (condition (ii) violated) ⇒ explains the flat Phase 2.
python src/diag_overlap.py \
  --student Qwen/Qwen3-4B-Instruct-2507 \
  --teacher Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
  --data data/train_math.json \
  --num-prompts 64 --max-new-tokens 1024 --topk 16 \
  --out results/diag_overlap

# OPTIONAL second pass — probe the POST-distillation student (did OPD move overlap at all?).
# Needs the merged full model eval_aime.py already wrote under $HF_HOME; uncomment to run:
# python src/diag_overlap.py \
#   --student "$HF_HOME/opd_merged/phase2_distill_merged" \
#   --teacher Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
#   --data data/train_math.json --num-prompts 64 --max-new-tokens 1024 --topk 16 \
#   --out results/diag_overlap_post
