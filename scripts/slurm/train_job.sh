#!/bin/bash
#SBATCH -p main
#SBATCH --exclude=dgx-station
#SBATCH --gres=gpu:1
#SBATCH -c 1
#SBATCH --mem=16G
# Train an algorithm on an environment (default: TRIRL-SPR on PushT with the Warp backend). Submit from the repo root.
# A watchdog stops the run as soon as Warp reports a buffer overflow (dropped contacts or constraints).
#
#   ENV_NAME=balance_fr3_mjx SEED=0 RUN_NAME=balance_s0 PROJECT=balance EXP_NAME=spr \
#   ALGO_ARGS="--algorithm.total_timesteps=500e6 --algorithm.reward_type=byol-embodiment-free" \
#   ENV_ARGS="--environment.horizon=200" sbatch -t 24:00:00 scripts/slurm/train_job.sh
#
# Variables: ALGO (trirl_ppo_spr.flax_full_jit), ENV_NAME (pusht_mjx), IMPL (warp), SEED (0), NR_ENVS (4096), RUN_NAME,
# PROJECT, EXP_NAME, ALGO_ARGS, ENV_ARGS, WANDB (True), WANDB_ENTITY (trirl), SAVE (True); see scripts/slurm/common.sh.
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/slurm/common.sh"
ALGO=${ALGO:-trirl_ppo_spr.flax_full_jit}; ENV_NAME=${ENV_NAME:-pusht_mjx}; IMPL=${IMPL:-warp}; SEED=${SEED:-0}; NR_ENVS=${NR_ENVS:-4096}
RUN_NAME=${RUN_NAME:-${ENV_NAME}_${IMPL}_s${SEED}}; PROJECT=${PROJECT:-$ENV_NAME}; EXP_NAME=${EXP_NAME:-${ALGO%%.*}}
WANDB=${WANDB:-True}; WANDB_ENTITY=${WANDB_ENTITY:-trirl}; SAVE=${SAVE:-True}
LOG=$LOG_DIR/${RUN_NAME}_${SLURM_JOB_ID}.console
echo "$(describe) | algo $ALGO | env $ENV_NAME ($IMPL) | seed $SEED | console $LOG"
cd "$REPO/experiments"
python -u experiment.py --algorithm.name=$ALGO ${ALGO_ARGS} \
  --environment.name=$ENV_NAME --environment.nr_envs=$NR_ENVS --environment.seed=$SEED --environment.impl=$IMPL ${ENV_ARGS} \
  --runner.mode=train --runner.track_tb=True --runner.track_console=True --runner.track_wandb=$WANDB --runner.save_model=$SAVE \
  --runner.wandb_entity=$WANDB_ENTITY --runner.project_name=$PROJECT --runner.exp_name=$EXP_NAME --runner.run_name=$RUN_NAME > "$LOG" 2>&1 &
PID=$!
watch_overflow $PID "$LOG" &
wait $PID; RC=$?
grep -q "overflow - please increase" "$LOG" && { echo "run stopped: buffer overflow"; exit 3; }
echo "exit code $RC"; exit $RC
