# Using this codebase

This document covers the environments, algorithms and scripts added on top of the paper release: the robot tasks
PushT, pizza spreading and ball balancing on the FR3, TRIRL with SPR reward features (`trirl_ppo_spr`), the MuJoCo
Warp backend, PPO on learned rewards, and the evaluation tools. It explains what every parameter means and how to run
each part. Installation is in [install.md](install.md).

- [How a run is put together](#how-a-run-is-put-together)
- [Environments](#environments): [PushT](#pusht-pusht_mjx), [pizza](#pizza-spreading-pizza_dr_mjx), [ball balancing](#ball-balancing-balance_fr3_mjx)
- [Algorithms](#algorithms): [TRIRL-SPR](#trirl-with-spr-reward-features-trirl_ppo_sprflax_full_jit), [PPO on a learned reward](#ppo-on-a-learned-reward-retraining), [PPO baseline](#ppo-baseline-ppo_bestflax_full_jit)
- [Runner options](#runner-options)
- [Running on a Slurm cluster](#running-on-a-slurm-cluster)
- [Expert data](#expert-data)
- [Evaluation and videos](#evaluation-and-videos)
- [Reward transfer](#reward-transfer)
- [Repository layout](#repository-layout)

## How a run is put together

Every run is `experiments/experiment.py` (RL-X's runner) with three groups of options: `--algorithm.*`,
`--environment.*` and `--runner.*`. Defaults live in `trust_region_irl/algorithms/<algorithm>/<implementation>/default_config.py`
and `trust_region_irl/environments/<environment>/default_config.py`; anything given on the command line overrides them.
Run from the `experiments` folder, since dataset paths are relative to it.

```bash
cd experiments
python experiment.py \
  --algorithm.name=trirl_ppo_spr.flax_full_jit --algorithm.data_path=../trirl_dataset/rl_expert/balance_fr3_varied_300.npz \
  --algorithm.reward_type=byol-embodiment-free --algorithm.total_timesteps=500e6 \
  --environment.name=balance_fr3_mjx --environment.impl=warp --environment.nr_envs=4096 --environment.seed=0 \
  --runner.mode=train --runner.save_model=True --runner.track_console=True
```

Models are saved under `experiments/runs/<project_name>/<exp_name>/<run_name>/models`.

**Observations and embodiment-free rewards.** In all three robot tasks the observation has a task part (objects, tool,
goal) followed by the arm's 7 joint positions and 7 joint velocities. The policy and critic see everything. With
`--algorithm.reward_type=byol-embodiment-free` the reward network sees only the task part, listed by the environment's
`encoder_input_indices`, so the learned reward does not depend on the robot.

**Backends.** `--environment.impl=jax` uses MJX's JAX physics; `--environment.impl=warp` uses MuJoCo Warp, which is
several times faster on large batches. Warp keeps one contact pool for all parallel environments: `contacts_per_env`
sizes it, and Warp prints `overflow - please increase ...` when it is too small and contacts get dropped. The Slurm jobs
stop a run when that happens. Both backends give the same physics up to floating-point rounding (see
`scripts/mjwarp/check_backends.py`).

## Environments

All three use the FR3 arm with velocity actuators and a differential IK that turns a 2-D task command into joint
velocities. They run at 20 Hz control.

### PushT (`pusht_mjx`)

The arm pushes a T-shaped block to a goal pose. The action is the planar tool velocity. Observation (24): block position
(3) and orientation quaternion (4) relative to the goal, tool position relative to the goal (3), arm joint positions (7)
and velocities (7). The reward input of `byol-embodiment-free` is the first 10 entries.

| Parameter | Default | Meaning |
|---|---|---|
| `nr_envs` | 64 | parallel environments (use 4096 for training) |
| `seed` | 1 | random seed of the run |
| `impl` | jax | physics backend: `jax` or `warp` |
| `contacts_per_env` | 64 | Warp only: contact slots per environment in the shared pool |
| `ls_iterations` | 0 | if > 0, overrides the solver's line-search iterations |
| `feature_fn` | base | hand-designed reward features for the feature-based algorithms (`base`, `base_rbf`, `base_without_ctrl`, `pos_ort`, `state_action`); not used by the BYOL reward types |
| `random_TPos` | False | random start pose of the T instead of the fixed presets |
| `success_threshold` | 0.05 | success when both position and orientation error are below this (without the curriculum) |
| `curriculum_success` | False | lower the success threshold during training (trirl_ppo_spr) |
| `success_threshold_values` | 0.5,0.25,0.1,0.05 | the thresholds of the curriculum |
| `success_threshold_fractions` | 0.2,0.2,0.2,0.4 | the share of training each threshold lasts (must sum to 1) |
| `render`, `device`, `copy_train_env_for_eval` | False, gpu, True | viewer on/off, JAX device, use the training env for evaluation |

PushT runs with SPR rewards should use `--environment.curriculum_success=True`.

### Pizza spreading (`pizza_dr_mjx`)

The arm spreads sauce on a pizza with the pusher tool. Each episode draws a pizza radius between 10 and 20 cm and a
centre in the band around the robot base where the IK can spiral out to the rim without singularities (within 45° of
straight ahead). The tool starts above the pizza at travel height. The action is the planar tool velocity (×0.354 m/s).

An episode goes through four phases:
1. **Travel** 5 cm above the pizza until the tool is within 1 cm of the centre.
2. **Touchdown:** the environment lowers the tool, holding x-y still.
3. **Spreading:** coverage, the tool-swept area of the sauce disc (radius R minus a 1.5 cm crust), counts only now.
4. **Lift-off:** after 0.5 s of standing still on the pizza (action norm below 0.05), the environment lifts the tool back
   to travel height. It hovers there until the fixed horizon.

Observation (212): tool position (3, world), pizza centre (2) and radius (1), arm joint positions (7) and velocities (7),
and a binary swept-area grid of 8 rings × 24 sectors (192, 1 = swept) anchored at the pizza centre and scaled to the
sauce disc. The reward input of `byol-embodiment-free` is everything except the arm joints (indices 0-5 and 20-211).
Coverage, the mean of the grid, is the success measure; the environment reward is for logging only.

| Parameter | Default | Meaning |
|---|---|---|
| `horizon` | 800 | steps per episode (40 s): travel, the spiral of the largest pizza and the lift-off |
| `success_threshold` | 0.90 | coverage counted as success |
| `end_on_success` | False | end the episode at success (default: fixed horizon, success is only a metric) |
| `wander_margin` | 0 (= 0.025 m) | after touchdown, the episode is cut when the tool stays more than this beyond the rim; 10 effectively disables the cut |
| `wander_hold_s` | 0 (= 0.5 s) | how long the tool must stay beyond the margin before the cut |
| `impl`, `contacts_per_env`, `ls_iterations`, `nr_envs`, `seed` | | as for PushT |

Logged metrics include `rollout/coverage`, `rollout/is_success`, `rollout/wandered_off`, `rollout/lifted`,
`rollout/clean_finish` (lifted off with the coverage target), `rollout/success_clean` (success without wandering off),
`rollout/touchdown_step`, `rollout/lift_step` and `rollout/off_pizza_frac`.

The pizza position band comes from `scripts/pizza/workspace_sweep.py`, and the start poses in `data/start_table.npz`
from `scripts/pizza/build_start_table.py`. Rerun both if the robot or the IK changes.

### Ball balancing (`balance_fr3_mjx`)

A port of BaMPC's Balance-FR3: the arm holds a plate and tilts it to roll a ball (radius 4 cm) to a goal that moves on
the plate. The model, solver options and task-space IK are BaMPC's.

- **Action (2):** plate tilt rate about the world x and y axes, in [-1, 1] × 1 rad/s, held for the five 10 ms physics
  sub-steps of each 50 ms step. The IK holds the plate's position and yaw by itself.
- **Goal paths:** each episode draws one of five families with random parameters, the same as the BaMPC demos: still;
  line (amplitude 3-8 cm, period 12-25 s); ellipse (radii 3-8 cm, period 12-25 s, either direction); Lissajous (radii
  3-6 cm, frequency ratio 1:2 or 2:1, period 16-30 s); ramped waypoints (the goal glides at 2-4 cm/s to a random point,
  then picks a new one once the ball is within 3 cm). Moving paths faster than 6 cm/s are redrawn. The ball starts at a
  random point of the level plate.
- **Observation (39):** ball position, linear and angular velocity (9); plate position, quaternion and angular velocity
  (10); goal position and velocity (6); arm joint positions (7) and velocities (7). The reward input of
  `byol-embodiment-free` is the first 25 entries.
- No early termination; the logged metrics are `rollout/goal_dist`, `rollout/goal_dist_2nd_half`, `rollout/ball_lost`
  and `rollout/goal_family`.

| Parameter | Default | Meaning |
|---|---|---|
| `horizon` | 200 | steps per episode (10 s, as the BaMPC demos) |
| `ball_mass_scale` | 1.0 | ball mass and inertia (a solid ball rolls the same at any mass; only large changes matter) |
| `tilt_gain_scale` | 1.0 | plate tilt rate per unit action, inside the IK |
| `ik_k_pos_scale` | 1.0 | IK gain holding the plate position |
| `ik_k_yaw_scale` | 1.0 | IK gain holding the plate yaw |
| `ik_damping_scale` | 1.0 | IK damped least-squares regulariser |
| `impl`, `contacts_per_env`, `nr_envs`, `seed` | | as for PushT |

The five dynamics options default to BaMPC's task and are meant for testing how well a learned reward transfers.

## Algorithms

### TRIRL with SPR reward features (`trirl_ppo_spr.flax_full_jit`)

TRIRL with a PPO learner, where the reward is linear in features from a self-predictive (BYOL-style) encoder trained
together with the discriminator. Before TRIRL starts, the encoder is pretrained on its own for `pretrain_env_steps`.
After that, every iteration collects a rollout, updates the encoder and discriminator, corrects the reward (the
trust-region step) and updates the policy with PPO under a trust-region term.

**PPO**

| Parameter | Default | Meaning |
|---|---|---|
| `total_timesteps` | 2e9 | environment steps of TRIRL training (the pretraining comes on top) |
| `learning_rate`, `anneal_learning_rate` | 4e-4, True | policy and critic learning rate, linearly annealed to 0 |
| `nr_steps` | 128 | rollout length per environment and iteration (one batch = nr_envs × nr_steps) |
| `nr_epochs`, `minibatch_size` | 10, 32768 | PPO epochs and minibatch size per iteration |
| `gamma`, `gae_lambda` | 0.99, 0.9 | discount and GAE λ |
| `clip_range` | 0.1 | PPO ratio clip |
| `entropy_coef` | 0.001 | entropy weight; also the reward scale (β = 1/entropy_coef) |
| `critic_coef`, `max_grad_norm` | 1.0, 5.0 | critic loss weight, gradient clipping |
| `std_dev` | 1.0 | initial policy standard deviation |
| `action_clipping_and_rescaling` | False | clip actions to [-1, 1] and rescale them to the env's action bounds |
| `evaluation_and_save_frequency` | -1 | environment steps between deterministic evaluations and saves (-1: only at the end) |
| `evaluation_active` | True | run the evaluation (it also picks `best.model`) |

**IRL**

| Parameter | Default | Meaning |
|---|---|---|
| `data_path` | | expert dataset (`.npz`, relative to `experiments/`) |
| `reward_type` | byol-observation-based | reward input: `byol-observation-based` (full observation), `byol-embodiment-free` (the env's `encoder_input_indices`), `byol-task-based-observation` (PushT's T pose, 0-6) |
| `epsilon` | 0.69 | step size of the reward correction |
| `init_eta`, `const_eta` | 100, False | weight of the policy trust-region term; linearly annealed to 0 unless constant |
| `env_reward_frac` | 0.0 | share of the environment reward mixed in (0 = pure IRL) |
| `learning_rate_disc`, `nr_epochs_disc` | 2e-4, 30 | encoder and discriminator learning rate and epochs per iteration |
| `gp_lambda`, `gp_alpha` | 0.033, 0.5 | gradient-penalty weight on the discriminator, and where between expert and agent sample it is taken |
| `handle_absorbing_states` | True | treat episode ends as absorbing states with their own reward |
| `nr_hidden_units_disc` | 256 | not used by the SPR encoder (fixed sizes in `byol.py`) |

**SPR / BYOL encoder**

| Parameter | Default | Meaning |
|---|---|---|
| `pretrain_env_steps` | 50e6 | environment steps of encoder-only pretraining before TRIRL (not counted in `total_timesteps`) |
| `prediction_steps` | 1 | k: the encoder predicts the embeddings of the next k states from the current one and the k actions |
| `target_momentum` | 0.99 | EMA rate of the target encoder |
| `warmup_env_steps` | 0 | TRIRL steps during which the discriminator loss does not train the encoder |
| `w_disc_loss` | 1.0 | weight of the discriminator loss next to the BYOL loss |

**Other**

| Parameter | Default | Meaning |
|---|---|---|
| `retraining` | False | with `--runner.mode=test` and a loaded model: train a new PPO policy on the frozen learned reward (below) |
| `curriculum_learning` | False | in retraining, follow the reward of every TRIRL iteration (from `theta_matrix.npy`) instead of only the final one |

Settings used for the 500M pizza and ball-balancing runs:
```
--algorithm.total_timesteps=500e6 --algorithm.pretrain_env_steps=25e6 --algorithm.prediction_steps=5
--algorithm.reward_type=byol-embodiment-free --algorithm.clip_range=0.2 --algorithm.epsilon=0.2 --algorithm.init_eta=30
--algorithm.gae_lambda=0.95 --algorithm.gp_lambda=0.5 --algorithm.learning_rate_disc=1e-4 --algorithm.nr_epochs_disc=10
--algorithm.minibatch_size=512 --algorithm.std_dev=0.4 --algorithm.evaluation_and_save_frequency=2097152
--environment.nr_envs=4096 --environment.impl=warp
```

Each save writes `latest.model.zip` and, when the deterministic evaluation return reaches a new high, `best.model.zip`,
plus `theta_matrix.npy` (the corrected reward weights of every iteration).

### PPO on a learned reward (retraining)

Trains a new PPO policy from scratch on the frozen reward of a TRIRL-SPR model: its encoder and corrected reward
weights. Nothing about the reward is updated. The environment options are not taken from the model, so the same reward
can be used on changed dynamics.

```bash
cd experiments
python experiment.py --algorithm.name=trirl_ppo_spr.flax_full_jit --algorithm.retraining=True --algorithm.total_timesteps=500e6 \
  --environment.name=balance_fr3_mjx --environment.impl=warp --environment.nr_envs=4096 --environment.tilt_gain_scale=0.5 \
  --runner.mode=test --runner.load_model=runs/<project>/<exp>/<run>/models/latest.model.zip \
  --runner.project_name=<project> --runner.exp_name=ppo_retrain --runner.run_name=<run>
```

PPO settings come from the loaded model unless given on the command line. The run is saved as `<run_name>_ppo_retraining`
with `latest.model` and `best.model`. Loading unpacks the archive next to itself, so give parallel runs their own copy
(`scripts/slurm/retrain_job.sh` does).

### PPO baseline (`ppo_best.flax_full_jit`)

RL-X's PPO, unchanged except that it also keeps `best.model`. It trains on the environment's own reward (for ball
balancing: minus the ball-goal distance) and is the reference for what PPO can reach with the true reward. Its options
are the PPO options above.

## Runner options

| Option | Meaning |
|---|---|
| `mode` | `train`, `test` (also used for retraining) or `show_config` |
| `track_console`, `track_tb`, `track_wandb` | log to the console, TensorBoard, Weights & Biases |
| `wandb_entity`, `project_name`, `exp_name`, `run_name` | W&B entity; the run is stored in `runs/<project_name>/<exp_name>/<run_name>` |
| `save_model`, `load_model` | save models; path of a model to load |
| `jax_cache_dir` | compilation cache (default `/tmp/jax_cache`) |

`--runner.mode=show_config` prints every option of a given algorithm and environment.

## Running on a Slurm cluster

`scripts/slurm` has four jobs. Submit them from the repository root; they find the repo through `SLURM_SUBMIT_DIR` and
take their settings from environment variables.

| Job | What it does |
|---|---|
| `train_job.sh` | train an algorithm on an environment (default TRIRL-SPR on PushT with Warp) |
| `retrain_job.sh` | PPO from scratch on a frozen TRIRL-SPR reward (`SRC_CKPT`), optionally with changed dynamics |
| `ppo_job.sh` | the PPO baseline on the environment's true reward |
| `run_job.sh` | any script of the repo on one GPU (data generation, checks, evaluation, rendering) |

Shared variables (`scripts/slurm/common.sh`): `CONDA_ENV` (default `trirl_warp36`), `RLX` (default `../RL-X`), `LOG_DIR`
(default `logs/`, where each run's console output goes), `CONDA_BASE`. Each training job also takes `ENV_NAME`, `SEED`,
`NR_ENVS`, `IMPL`, `RUN_NAME`, `PROJECT`, `EXP_NAME`, `ALGO_ARGS` and `ENV_ARGS`; see the header of each script.

```bash
# TRIRL-SPR on ball balancing, seed 0
ENV_NAME=balance_fr3_mjx SEED=0 RUN_NAME=balance_spr_s0 PROJECT=balance EXP_NAME=spr \
ALGO_ARGS="--algorithm.data_path=../trirl_dataset/rl_expert/balance_fr3_varied_300.npz --algorithm.reward_type=byol-embodiment-free ..." \
sbatch -t 24:00:00 scripts/slurm/train_job.sh

# any script on a GPU
sbatch scripts/slurm/run_job.sh scripts/balance/eval_policies.py results/balance_eval experiments/runs/.../latest.model.zip
```

The `#SBATCH` lines (partition `main`, excluded node) are for the IAS cluster; change them, or override them on the
`sbatch` command line, for other clusters.

## Expert data

Expert datasets are `.npz` files with `states`, `actions`, `next_states`, `rewards` and `absorbing` (1 on the last step of
each demo), stored demo after demo, plus `episode_lengths` and `demo_meta`. The PushT and paper datasets come from the
HuggingFace dataset; the pizza and ball-balancing ones are generated:

**Pizza** (synthetic spirals, played through the environment itself):
```bash
python scripts/pizza/generate_demos.py trirl_dataset/rl_expert/pizza_dr_demos.npz 512 4      # about 2,000 demos
python scripts/pizza/select_demos.py trirl_dataset/rl_expert/pizza_dr_demos.npz trirl_dataset/rl_expert/pizza_dr_demos_100.npz results/pizza_demos 100
```
Each demo travels to the centre, spirals clockwise from the centre to the sauce edge (pitch 2.2-3.0 cm, speed
0.18-0.30 m/s, random start angle), closes one loop, stops and is lifted off. Only full-horizon demos that lift off with
at least 90% coverage are kept. `select_demos.py` keeps a diverse subset (k-means over the randomisation).

**Ball balancing** (BaMPC's MPPI planner, in a [bampc-code](https://github.com/bampc-framework/bampc-code) checkout and its
own environment with MuJoCo 3.10):
```bash
# in bampc-code, conda env of BaMPC
python <this repo>/scripts/balance/bampc/collect_demos.py raw_part1.npz 50 101      # 50 demos, seed 101 (run several parts in parallel)
PER_FAMILY=10 EPISODE_S=30 python <this repo>/scripts/balance/bampc/collect_demos.py raw_30s_part1.npz 0 101   # 10 per goal family, 30 s each
python <this repo>/scripts/balance/bampc/merge_parts.py raw.npz raw_part*.npz
# in this repo's environment
python scripts/balance/make_expert_dataset.py raw.npz trirl_dataset/rl_expert/balance_fr3_varied_300.npz
```
`collect_demos.py` also takes `HOLD=1` (one command per 50 ms step, as the policy acts; compare with
`bampc/compare_hold.py`). `bampc/export_model.py` and `bampc/dump_task_params.py` regenerate the model files in
`balance_fr3_mjx/data`, and `scripts/balance/check_replay.py` checks the port against recorded demos.

Training on 30 s demos needs `--environment.horizon=600`.

## Evaluation and videos

All evaluation scripts run the deterministic policy (the mean action) on seeded random episodes, so different
checkpoints see the same episodes. They take any saved model: TRIRL `*.model.zip`, retraining and PPO `*.model`.

| Script | Output |
|---|---|
| `scripts/pizza/plot_policy_paths.py OUT CKPT...` | tool paths on 8 pizzas per checkpoint, coverage over time, metrics; marks where a wander cut would end the episode |
| `scripts/balance/eval_policies.py OUT CKPT... [--expert DATA] [--dynamics k=v ...]` | ball-goal distance and ball loss per goal family, next to zero action and the expert; plots |
| `scripts/balance/render_policy.py CKPT OUT [--seconds 50] [--expert-raw RAW]` | one video per goal family plus all joined, with the goal path and the ball trail drawn on the plate |
| `scripts/balance/render_expert.py RAW OUT` | the same videos for BaMPC demos |
| `scripts/balance/dynamics_sweep.py OUT.json CKPT... [--variants JSON]` | how much each dynamics variant hurts a trained policy, without retraining |

Rendering needs `MUJOCO_GL=egl` on a headless GPU node (`run_job.sh` sets it).

## Reward transfer

The question: does a reward learned with TRIRL on one robot setup still produce good behaviour when the dynamics change?

1. Train TRIRL-SPR with `byol-embodiment-free` on the nominal task.
2. Retrain PPO from scratch on its frozen reward on the nominal task (`retrain_job.sh`), to check that the reward alone
   reproduces the behaviour.
3. Pick dynamics variants that hurt the trained policy (`dynamics_sweep.py`; for ball balancing, a 4× heavier ball, half
   the tilt gain and 1000× IK damping all do).
4. Retrain PPO on the frozen reward under each variant (`retrain_job.sh` with `ENV_ARGS`), and train the PPO baseline on
   the true reward under the same variant (`ppo_job.sh`) as the reference.
5. Compare with `eval_policies.py --dynamics ...`: the reward transfers if PPO on it gets about as close to the
   true-reward reference under the variant as on the nominal task.

## Repository layout

```
trust_region_irl/
  algorithms/
    trirl_ppo_spr/        TRIRL with SPR (BYOL multi-step) reward features
    ppo_retraining/       PPO on a frozen learned reward (used by retraining mode)
    ppo_best/             RL-X PPO that keeps best.model (true-reward baseline)
    ...                   the paper's algorithms and baselines
  environments/
    pusht_mjx/            PushT (FR3, T block); its data/assets hold the FR3 meshes the other tasks link to
    pizza_dr_mjx/         pizza spreading with domain randomisation
    balance_fr3_mjx/      BaMPC ball balancing
    ...                   the paper's environments
  policy_checkpoint.py    load the policy of any saved model
scripts/
  slurm/                  cluster jobs (train, retrain, PPO baseline, run any script)
  mjwarp/                 Warp vs JAX backend checks for PushT
  pizza/                  workspace sweep, start table, demos, path plots
  balance/                BaMPC tools (bampc/), dataset, checks, evaluation, videos, dynamics sweep
experiments/              experiment.py (entry point); runs/ is created here
```
