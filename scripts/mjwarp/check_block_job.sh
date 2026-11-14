#!/bin/bash
#SBATCH -p main
#SBATCH --gres=gpu:1
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 01:00:00
# Backend check of this repo's PushT env on MuJoCo 3.6 (conda env trirl_warp36, CUDA 12)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate trirl_warp36
export XLA_PYTHON_CLIENT_PREALLOCATE=false
cd ~/sant/trirl_workspace/mohammed_trirl
nvidia-smi --query-gpu=name --format=csv,noheader
PYTHONPATH=$PWD:$HOME/sant/trirl_workspace/RL-X python -u scripts/mjwarp/check_block.py 2>&1 | grep -v "^Warp \|Module \|Kernel cache\|GLFW\|DISPLAY"
