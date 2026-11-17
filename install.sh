#!/usr/bin/env bash
# Conda environment for this repository: MuJoCo / MJX 3.6 with the Warp backend, JAX 0.7.1 on CUDA 12, RL-X.
# Creates the env (default name trirl_warp36), clones RL-X next to this repository and the TRIRL dataset into it.
#   ./install.sh [ENV_NAME]
# For the original paper setup (MuJoCo 3.2, LocoMujoco), see the icml26 tag of this repository.
set -e

ENV_NAME="${1:-trirl_warp36}"
cd "$(dirname "$0")"
THIS_REPO="$(pwd)"
TOP_DIR="$(dirname "$THIS_REPO")"

# Make conda activate work inside the script.
eval "$(conda shell.bash hook)"
conda create -n "$ENV_NAME" python=3.11.4 -y
conda activate "$ENV_NAME"

# Pinned simulation and training stack (see requirements.txt), CPU-only torch for TensorBoard logging
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu

# RL-X from source. Its own dependency list would pull other MuJoCo and JAX versions, so only the code is installed.
cd "$TOP_DIR"
[ -d RL-X ] || git clone https://github.com/nico-bohlinger/RL-X.git
cd RL-X
git checkout 14379ff # clone a specific commit to ensure reproducibility
pip install -e . --no-deps --config-settings editable_mode=compat

# Install this package
cd "$THIS_REPO"
pip install -e . --no-deps

# Install git-xet for HuggingFace dataset support
curl --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/huggingface/xet-core/refs/heads/main/git_xet/install.sh | sh

# Clone dataset into this repo. The pizza and ball-balancing demonstrations are generated with scripts/pizza and
# scripts/balance into trirl_dataset/rl_expert (see docs/usage.md).
cd "$THIS_REPO"
[ -d trirl_dataset ] || git clone https://huggingface.co/datasets/anishdiwan/trirl_dataset

# Quick check: MuJoCo, JAX and Warp import, and a GPU is visible to JAX
python -c "import mujoco, jax, warp; print('mujoco', mujoco.__version__, '| jax', jax.__version__, jax.devices(), '| warp', warp.config.version)"

echo "Installation complete: conda activate $ENV_NAME"
