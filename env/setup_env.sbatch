#!/bin/bash
#SBATCH --job-name=opd-setup
#SBATCH --partition=gpu              # <-- your H200 partition
#SBATCH --gres=gpu:h200:1            # <-- your gres syntax
#SBATCH --time=03:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/setup_%j.out
#SBATCH --error=logs/setup_%j.err

# One-off: build a conda env on the cluster. We use conda (not venv) because the
# system Python is 3.9, but math-verify requires >=3.10 — so we create a 3.11 env.
# flash-attn needs a GPU node to build, which is why this runs under sbatch. See README §6.
set -euo pipefail

module load cuda/12.8.0          # matches torch 2.8 cu128 build (Explorer module name)
module load miniconda3/25.9.1    # provides conda; 3.11 env fixes the math-verify >=3.10 floor

eval "$(conda shell.bash hook)"  # enable `conda activate` in a non-interactive shell
conda create -y -p "$HOME/.conda/envs/opd" python=3.11
conda activate "$HOME/.conda/envs/opd"

pip install --upgrade pip
pip install -r requirements.txt
# flash-attn is OPTIONAL: not in requirements.txt, slow to compile, and easily
# blows the wall-clock limit. If it fails, HF transformers falls back to `sdpa`
# and vLLM uses its own kernels — so we never let it block setup.
pip install flash-attn --no-build-isolation || echo "flash-attn skipped (optional) — using sdpa"

python -c "import torch, vllm, trl, transformers; \
print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); \
print('trl', trl.__version__, 'transformers', transformers.__version__)"

echo "Setup complete. Remember to: export HF_HOME=\$HOME/hf_cache"
