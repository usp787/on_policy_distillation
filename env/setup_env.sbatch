#!/bin/bash
#SBATCH --job-name=opd-setup
#SBATCH --partition=gpu              # <-- your H200 partition
#SBATCH --gres=gpu:h200:1            # <-- your gres syntax
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/setup_%j.out
#SBATCH --error=logs/setup_%j.err

# One-off: build the venv on the cluster. flash-attn needs a GPU node to build,
# which is why this runs under sbatch rather than on a login node. See README §6.
set -euo pipefail

module load cuda/12.4 2>/dev/null || true     # adjust to your cluster

python -m venv "$HOME/.venvs/opd"
source "$HOME/.venvs/opd/bin/activate"

pip install --upgrade pip
pip install -r requirements.txt
# flash-attn is intentionally not in requirements.txt (needs --no-build-isolation
# and a CUDA toolchain present at build time).
pip install flash-attn --no-build-isolation

python -c "import torch, vllm, trl, transformers; \
print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); \
print('trl', trl.__version__, 'transformers', transformers.__version__)"

echo "Setup complete. Remember to: export HF_HOME=\$HOME/hf_cache"
