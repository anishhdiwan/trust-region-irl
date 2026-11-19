# Trust Region IRL for robot manipulation

This branch builds on the official code of **Trust Region Inverse Reinforcement Learning** (TRIRL, ICML 2026;
[paper](https://arxiv.org/pdf/2605.11020), [webpage](https://anishhdiwan.github.io/trust-region-irl/)) and applies it to
manipulation tasks on a Franka FR3. It adds three robot environments, a reward learned with self-predictive
representations, a much faster physics backend, and tools to test whether a learned reward can be re-optimised and
transferred. The paper's own setup is kept at the `icml26` tag.

## What is new

**Environments** (MJX, 20 Hz control, the arm driven by a differential IK from a 2-D task-space action)
- `pusht_mjx`: push a T-shaped block to a goal pose.
- `pizza_dr_mjx`: spread sauce on a pizza of random size (10-20 cm) and position with a pusher tool; the policy
  commands the planar tool velocity, touchdown and lift-off are handled by the environment.
- `balance_fr3_mjx`: tilt a plate to roll a ball to a goal moving on it, a port of BaMPC's Balance-FR3 with its exact
  model and IK; optional dynamics changes (ball mass, IK gains) for transfer tests.

**Learning**
- `trirl_ppo_spr`: TRIRL with a PPO learner whose reward is linear in features from a self-predictive (SPR / BYOL,
  multi-step) encoder trained alongside the discriminator. `reward_type=byol-embodiment-free` keeps the arm joints out
  of the reward, so it describes the task rather than the robot.
- PPO on a frozen learned reward (`--algorithm.retraining=True`) and a PPO baseline on the true reward (`ppo_best`),
  both keeping the best and the latest model.
- The expert batch is sampled with replacement in every IRL algorithm.

**Infrastructure**
- MuJoCo Warp backend (`--environment.impl=warp`, MuJoCo 3.6), several times faster than MJX's JAX backend.
- Scripts for expert data (synthetic pizza spirals, BaMPC planner demos), evaluation, path plots and videos, and
  Slurm jobs for training and retraining.

## Install

```bash
./install.sh        # conda env trirl_warp36: MuJoCo 3.6 + Warp, JAX 0.7.1 (CUDA 12), RL-X, this package, the dataset
```
Manual steps are in [docs/install.md](docs/install.md). The pizza and ball-balancing demonstrations are generated with
the scripts in `scripts/` (see [docs/usage.md](docs/usage.md#expert-data)).

## Quick start

```bash
cd experiments
python experiment.py --algorithm.name=trirl_ppo_spr.flax_full_jit \
  --algorithm.data_path=../trirl_dataset/rl_expert/balance_fr3_varied_300.npz \
  --algorithm.reward_type=byol-embodiment-free --algorithm.total_timesteps=500e6 \
  --environment.name=balance_fr3_mjx --environment.impl=warp --environment.nr_envs=4096 \
  --runner.mode=train --runner.save_model=True --runner.track_console=True
```
On a Slurm cluster, submit from the repository root:
```bash
ENV_NAME=balance_fr3_mjx SEED=0 RUN_NAME=balance_s0 ALGO_ARGS="..." sbatch scripts/slurm/train_job.sh
```

## Documentation

[docs/usage.md](docs/usage.md) explains every environment and algorithm parameter, the Slurm jobs, how to generate
the expert data, the evaluation and video scripts, and the reward-transfer procedure.
