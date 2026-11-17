#!/bin/bash
#SBATCH -p main
#SBATCH --exclude=dgx-station
#SBATCH --gres=gpu:1
#SBATCH -c 1
#SBATCH --mem=16G
#SBATCH -t 02:00:00
# Run any script of this repo on one GPU (data generation, evaluation, rendering). Submit from the repo root; the
# arguments are the python command line, relative to the repo root. MUJOCO_GL=egl renders headless.
#
#   sbatch scripts/slurm/run_job.sh scripts/balance/eval_policies.py results/balance_eval experiments/runs/.../latest.model.zip
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/slurm/common.sh"
export MUJOCO_GL=${MUJOCO_GL:-egl} PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
echo "$(describe) | python $*"
cd "$REPO"
set -o pipefail
python -u "$@" 2>&1 | quiet
