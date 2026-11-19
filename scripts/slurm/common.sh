# Shared setup for the Slurm jobs in this folder; sourced by them, not run on its own.
# Submit the jobs from the repository root (sbatch scripts/slurm/<job>.sh): the repo is found via SLURM_SUBMIT_DIR.
# Overridable: REPO (this repository), RLX (RL-X checkout, default: next to the repo, as install.sh puts it),
# CONDA_ENV (default trirl_warp36, made by install.sh), CONDA_BASE (conda install), LOG_DIR (default REPO/logs).
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
RLX=${RLX:-$(dirname "$REPO")/RL-X}
CONDA_ENV=${CONDA_ENV:-trirl_warp36}
CONDA_BASE=${CONDA_BASE:-$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")}
LOG_DIR=${LOG_DIR:-$REPO/logs}
mkdir -p "$LOG_DIR"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export PYTHONPATH="$REPO:$RLX${PYTHONPATH:+:$PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false   # several jobs may share a GPU's memory with other processes
# Warp prints a line per compiled kernel module; hide those (and headless-display warnings) from the job output
quiet() { grep -v --line-buffered "^Warp \|^   \|Module \|Kernel cache\|GLFW\|DISPLAY\|abstract_arrays\|return np.asarray"; }
describe() { echo "$(git -C "$REPO" log --oneline -1) | branch $(git -C "$REPO" branch --show-current) | env $CONDA_ENV | mujoco $(python -c 'import mujoco; print(mujoco.__version__)') | gpu $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"; }
# Warp drops contacts or constraints silently except for an "overflow - please increase ..." warning: stop the run on it
watch_overflow() {   # watch_overflow PID LOG
    while kill -0 "$1" 2>/dev/null; do
        if grep -q "overflow - please increase" "$2" 2>/dev/null; then
            echo "OVERFLOW DETECTED, stopping run:"; grep -m3 "overflow - please increase" "$2"; kill "$1"; break
        fi; sleep 30
    done
}
