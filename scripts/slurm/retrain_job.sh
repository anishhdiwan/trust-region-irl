#!/bin/bash
#SBATCH -p main
#SBATCH --exclude=dgx-station
#SBATCH --gres=gpu:1
#SBATCH -c 1
#SBATCH --mem=16G
# PPO from scratch on a frozen learned reward: the SPR encoder and corrected reward of a TRIRL-SPR checkpoint (retraining
# mode). The environment settings do not come from the checkpoint, so ENV_ARGS can change the dynamics (reward transfer).
# Keeps best.model (highest deterministic eval return, true task reward) and latest.model. Submit from the repo root.
#
#   SRC_CKPT=experiments/runs/<project>/<exp>/<run>/models/latest.model.zip ENV_NAME=balance_fr3_mjx SEED=0 RUN_NAME=ppo_rw_s0 \
#   ENV_ARGS="--environment.ball_mass_scale=4" sbatch -t 08:00:00 scripts/slurm/retrain_job.sh
#
# Variables: SRC_CKPT (required), ENV_NAME (balance_fr3_mjx), IMPL (warp), SEED (0), NR_ENVS (4096), STEPS (500e6), RUN_NAME,
# PROJECT (reward_transfer), ALGO_ARGS, ENV_ARGS, WANDB_ENTITY (trirl). The run is saved as <RUN_NAME>_ppo_retraining.
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/slurm/common.sh"
: "${SRC_CKPT:?set SRC_CKPT to a TRIRL-SPR model archive}"
ENV_NAME=${ENV_NAME:-balance_fr3_mjx}; IMPL=${IMPL:-warp}; SEED=${SEED:-0}; NR_ENVS=${NR_ENVS:-4096}; STEPS=${STEPS:-500e6}
RUN_NAME=${RUN_NAME:-ppo_on_reward_s${SEED}}; PROJECT=${PROJECT:-reward_transfer}; WANDB_ENTITY=${WANDB_ENTITY:-trirl}
LOG=$LOG_DIR/${RUN_NAME}_${SLURM_JOB_ID}.console
# loading unpacks the archive next to itself, so work on a private copy (parallel jobs on one checkpoint would collide)
CK=$LOG_DIR/ckpt_${RUN_NAME}_${SLURM_JOB_ID}; mkdir -p "$CK"; cp "$REPO/$SRC_CKPT" "$CK/reward.model.zip" 2>/dev/null || cp "$SRC_CKPT" "$CK/reward.model.zip"
echo "$(describe) | reward from $SRC_CKPT | env $ENV_NAME ($IMPL) [$ENV_ARGS] | seed $SEED | steps $STEPS | console $LOG"
cd "$REPO/experiments"
python -u experiment.py --algorithm.name=trirl_ppo_spr.flax_full_jit --algorithm.retraining=True --algorithm.total_timesteps=$STEPS ${ALGO_ARGS} \
  --environment.name=$ENV_NAME --environment.nr_envs=$NR_ENVS --environment.seed=$SEED --environment.impl=$IMPL ${ENV_ARGS} \
  --runner.mode=test --runner.load_model="$CK/reward.model.zip" --runner.track_tb=False --runner.track_console=False --runner.track_wandb=False \
  --runner.save_model=False --runner.wandb_entity=$WANDB_ENTITY --runner.project_name=$PROJECT --runner.exp_name=ppo_retrain --runner.run_name=$RUN_NAME > "$LOG" 2>&1 &
PID=$!
watch_overflow $PID "$LOG" &
wait $PID; RC=$?; rm -rf "$CK"
grep -q "overflow - please increase" "$LOG" && { echo "run stopped: buffer overflow"; exit 3; }
grep -q "Uncaught exception" "$LOG" && { echo "run failed (RL-X test mode logs exceptions instead of raising):"; grep -A20 "Uncaught exception" "$LOG" | tail -20; exit 4; }
echo "exit code $RC"; exit $RC
