#!/bin/bash
#SBATCH -p main
#SBATCH --exclude=dgx-station
#SBATCH --gres=gpu:1
#SBATCH -c 1
#SBATCH --mem=16G
# Reference: PPO on the environment's own task reward (ppo_best.flax_full_jit: RL-X PPO that also keeps best.model), with
# the PPO settings of the TRIRL-SPR runs (override with ALGO_ARGS). Submit from the repo root.
#
#   ENV_NAME=balance_fr3_mjx SEED=0 RUN_NAME=ppo_true_mass4 ENV_ARGS="--environment.ball_mass_scale=4" sbatch -t 08:00:00 scripts/slurm/ppo_job.sh
#
# Variables: ENV_NAME (balance_fr3_mjx), IMPL (warp), SEED (0), NR_ENVS (4096), STEPS (500e6), RUN_NAME, PROJECT (reward_transfer),
# ALGO_ARGS, ENV_ARGS, WANDB_ENTITY (trirl).
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/slurm/common.sh"
ENV_NAME=${ENV_NAME:-balance_fr3_mjx}; IMPL=${IMPL:-warp}; SEED=${SEED:-0}; NR_ENVS=${NR_ENVS:-4096}; STEPS=${STEPS:-500e6}
RUN_NAME=${RUN_NAME:-ppo_true_reward_s${SEED}}; PROJECT=${PROJECT:-reward_transfer}; WANDB_ENTITY=${WANDB_ENTITY:-trirl}
PPO_ARGS="--algorithm.learning_rate=4e-4 --algorithm.nr_steps=128 --algorithm.nr_epochs=10 --algorithm.minibatch_size=512 --algorithm.gamma=0.99
  --algorithm.gae_lambda=0.95 --algorithm.clip_range=0.2 --algorithm.entropy_coef=0.001 --algorithm.std_dev=0.4 --algorithm.evaluation_and_save_frequency=2097152"
LOG=$LOG_DIR/${RUN_NAME}_${SLURM_JOB_ID}.console
echo "$(describe) | PPO on the true reward | env $ENV_NAME ($IMPL) [$ENV_ARGS] | seed $SEED | steps $STEPS | console $LOG"
cd "$REPO/experiments"
python -u experiment.py --algorithm.name=ppo_best.flax_full_jit --algorithm.total_timesteps=$STEPS $PPO_ARGS ${ALGO_ARGS} \
  --environment.name=$ENV_NAME --environment.nr_envs=$NR_ENVS --environment.seed=$SEED --environment.impl=$IMPL ${ENV_ARGS} \
  --runner.mode=train --runner.track_tb=True --runner.track_console=True --runner.track_wandb=True --runner.save_model=True \
  --runner.wandb_entity=$WANDB_ENTITY --runner.project_name=$PROJECT --runner.exp_name=ppo_true_reward --runner.run_name=$RUN_NAME > "$LOG" 2>&1 &
PID=$!
watch_overflow $PID "$LOG" &
wait $PID; RC=$?
grep -q "overflow - please increase" "$LOG" && { echo "run stopped: buffer overflow"; exit 3; }
echo "exit code $RC"; exit $RC
