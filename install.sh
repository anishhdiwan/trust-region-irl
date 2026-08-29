#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"
THIS_REPO="$(pwd)"
TOP_DIR="$(dirname "$THIS_REPO")"

# Make conda activate work inside the script.
eval "$(conda shell.bash hook)"

# Install RL-X
conda create -n trirl python=3.11.4 -y
conda activate trirl

cd "$TOP_DIR"
git clone https://github.com/nico-bohlinger/RL-X.git
cd RL-X
git checkout 14379ff # clone a specific commit to ensure reproducibility
pip install -e ".[all]" --config-settings editable_mode=compat
pip uninstall $(pip freeze | grep -i '-cu12' | cut -d '=' -f 1) -y || true
pip install -U "jax[cuda12]"

# Install this package
cd "$THIS_REPO"
pip install -e .

# Install LocoMujoco
cd "$TOP_DIR"
git clone https://github.com/robfiras/loco-mujoco.git
cd loco-mujoco
git checkout 131c1e7 # clone a specific commit to ensure reproducibility
pip install -e .

# Install git-xet for HuggingFace dataset support
curl --proto '=https' --tlsv1.2 -sSf 
https://raw.githubusercontent.com/huggingface/xet-core/refs/heads/main/git_xet/install.sh | sh

# Clone dataset into this repo
cd "$THIS_REPO"
git clone https://huggingface.co/datasets/anishdiwan/trirl_dataset

echo "Installation complete"
