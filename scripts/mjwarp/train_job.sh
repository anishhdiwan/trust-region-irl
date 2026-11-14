#!/bin/bash
#SBATCH -p main
#SBATCH --gres=gpu:1
#SBATCH -c 1
#SBATCH --mem=8G
# TRIRL-SPR on PushT with a chosen MJX backend. IMPL=warp needs CONDA_ENV=trirl_warp36 (MuJoCo 3.6, CUDA 12).
# Hard check: a watchdog cancels the job as soon as Warp prints any buffer-overflow warning (dropped contacts/constraints).
# Usage: IMPL=warp SEED=0 RUN_NAME=... ALGO_ARGS="--algorithm.x=..." ENV_ARGS="--environment.y=..." sbatch -t 24:00:00 train_job.sh
IMPL=${IMPL:-warp}; CONDA_ENV=${CONDA_ENV:-trirl_warp36}; SEED=${SEED:-0}; NR_ENVS=${NR_ENVS:-4096}
RUN_NAME=${RUN_NAME:-pusht_spr_${IMPL}_s${SEED}}; PROJECT=${PROJECT:-pusht_spr_mjwarp}; WANDB=${WANDB:-True}; SAVE=${SAVE:-True}
source ~/miniconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV
export XLA_PYTHON_CLIENT_PREALLOCATE=false
R=$HOME/sant/trirl_workspace/mohammed_trirl
cd $R/experiments
LOG=$HOME/sant/trirl_workspace/logs/${RUN_NAME}_${SLURM_JOB_ID}.console
echo "impl=$IMPL env=$CONDA_ENV seed=$SEED $(git -C $R log --oneline -1) branch=$(git -C $R branch --show-current) mujoco=$(python -c 'import mujoco;print(mujoco.__version__)') gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) console=$LOG"
PYTHONPATH=$R:$HOME/sant/trirl_workspace/RL-X python -u experiment.py \
  --algorithm.name=trirl_ppo_spr.flax_full_jit ${ALGO_ARGS} \
  --environment.name=pusht_mjx --environment.nr_envs=$NR_ENVS --environment.seed=$SEED --environment.impl=$IMPL ${ENV_ARGS} \
  --runner.mode=train --runner.track_tb=True --runner.track_console=True --runner.track_wandb=$WANDB --runner.save_model=$SAVE \
  --runner.wandb_entity=trirl --runner.project_name=$PROJECT --runner.exp_name=pusht_spr --runner.run_name=$RUN_NAME > $LOG 2>&1 &
PID=$!
# watchdog: any "overflow - please increase ..." line from warp means contacts or constraints were dropped
( while kill -0 $PID 2>/dev/null; do
    if grep -q "overflow - please increase" $LOG 2>/dev/null; then
      echo "OVERFLOW DETECTED, stopping run:"; grep -m3 "overflow - please increase" $LOG; kill $PID; break
    fi; sleep 30; done ) &
wait $PID; RC=$?
grep -q "overflow - please increase" $LOG && { echo "run stopped: buffer overflow"; exit 3; }
echo "exit code $RC"; exit $RC
