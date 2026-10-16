"""Evaluate the best checkpoint of every seed in a TRIRL run directory.

For each seed found under the given experiment directory (e.g.
'runs/<project>/<exp_name>', the same directory plot_reward.py points at),
this deterministically rolls the policy out from each of the 6 preset
initial T-block positions (PushT.preset_block_poses) and records the
final-step position/orientation error, writing one row per (seed, position)
to a CSV.

Only the policy is restored from each checkpoint (the discriminator/critic
are skipped, since their architecture can vary per experiment and isn't
needed to roll the policy out), so this doesn't need to know which
--environment.feature_fn the run was trained with.

Run with the same environment used for training (conda env "trirl"), from
the experiments/ directory:

    conda activate trirl
    cd experiments
    python eval_final_error.py runs/ICRA_TRIRL_10_2F/pusht_ppo_fb
    python eval_final_error.py runs/ICRA_TRIRL_10_2F/pusht_ppo_fb --out results.csv
"""
import argparse
import csv
import os
import shutil
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # rollout is tiny (6 envs x horizon); avoid GPU contention

import numpy as np
import jax
import jax.numpy as jnp
from mujoco import mjx
import orbax.checkpoint as ocp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trust_region_irl.environments.pusht_mjx.environment import PushT
from trust_region_irl.environments.pusht_mjx.state import State
from trust_region_irl.algorithms.trirl_ppo_fb.flax_full_jit.policy import Policy, get_processed_action_function

NR_PRESET_POSITIONS = 6


def load_policy_params(model_zip):
    """Restore only the policy sub-tree of a checkpoint, shape-inferred from disk."""
    tmp = tempfile.mkdtemp()
    try:
        shutil.unpack_archive(model_zip, tmp, "zip")
        checkpointer = ocp.PyTreeCheckpointer()
        metadata = checkpointer.metadata(tmp)
        restore_args = jax.tree_util.tree_map(
            lambda m: ocp.RestoreArgs(restore_type=np.ndarray),
            metadata.item_metadata.tree,
            is_leaf=lambda x: hasattr(x, "shape"),
        )
        restored = checkpointer.restore(tmp, restore_args=restore_args)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return restored["policy"]["params"]


def quat_to_rotvec_np(q):
    w = np.clip(q[..., 0], -1.0, 1.0)
    angle = 2.0 * np.arccos(w)
    s = np.sqrt(np.clip(1.0 - w * w, 0.0, None))
    s_safe = np.where(s < 1e-8, 1.0, s)
    axis = np.where(s[..., None] < 1e-8, np.array([1.0, 0.0, 0.0]), q[..., 1:] / s_safe[..., None])
    return angle[..., None] * axis


def goal_errors_np(observation, err_clip):
    block_pos = observation[:, 0:3]
    block_quat = observation[:, 3:7]
    block_quat = np.where(block_quat[:, 0:1] < 0.0, -block_quat, block_quat)
    pos_err = np.linalg.norm(block_pos, axis=-1)
    orn_err = np.linalg.norm(quat_to_rotvec_np(block_quat), axis=-1)
    pos_err = np.clip(np.nan_to_num(pos_err, nan=err_clip, posinf=err_clip), 0.0, err_clip)
    orn_err = np.clip(np.nan_to_num(orn_err, nan=np.pi, posinf=np.pi), 0.0, np.pi)
    return pos_err, orn_err


def make_initial_state_for_position(env, idx):
    """Reproduce PushT._reset's random_TPos branch, but for a fixed preset index."""
    block_xy = env.preset_block_poses[idx][0:2]
    block_angle = env.preset_block_poses[idx][2]
    half_angle = block_angle * 0.5
    block_quat = jnp.array([jnp.cos(half_angle), 0.0, 0.0, jnp.sin(half_angle)])
    block_z = 0.045
    block_qpos = jnp.concatenate([block_xy, jnp.array([block_z]), block_quat])
    qpos = env.initial_qpos.at[0:7].set(block_qpos)

    data = env.mjx_data
    data = data.replace(qpos=qpos, qvel=env.initial_qvel,
                         ctrl=jnp.zeros(env.mjx_model.nu),
                         mocap_pos=env.mocap_pos, mocap_quat=env.mocap_quat)
    data = mjx.forward(env.mjx_model, data)

    next_observation = env.get_observation(data)
    info = {
        "rollout/episode_return": 0.0, "rollout/episode_length": 0,
        "env_info/pos_err": 0.0, "env_info/orient_err": 0.0, "env_info/is_success": 0.0,
        "env_info/reward_dist": 0.0, "env_info/reward_ctrl": 0.0, "env_info/diverged": 0.0,
    }
    info_episode_store = {"episode_return": 0.0, "episode_length": 0}
    key = jax.random.PRNGKey(idx)
    return State(data, next_observation, next_observation, 0.0, False, False, info, info_episode_store, key)


def make_initial_batch(env):
    """One State batched over all 6 preset positions (batch axis 0)."""
    states = [make_initial_state_for_position(env, idx) for idx in range(NR_PRESET_POSITIONS)]
    return jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves, axis=0), *states)


def evaluate_seed(model_zip, env, std_dev, action_clipping_and_rescaling):
    params = load_policy_params(model_zip)

    policy = Policy(
        as_shape=env.single_action_space.shape,
        std_dev=std_dev,
        policy_observation_indices=jnp.arange(env.single_observation_space.shape[0]),
    )
    get_processed_action = get_processed_action_function(
        action_clipping_and_rescaling,
        jnp.array(env.single_action_space.low),
        jnp.array(env.single_action_space.high),
    )

    @jax.jit
    def step_fn(state):
        action_mean, _ = policy.apply(params, state.next_observation)
        action = get_processed_action(action_mean)
        return env.step(state, action)

    state = make_initial_batch(env)
    for _ in range(env.horizon):
        state = step_fn(state)

    final_obs = np.asarray(state.actual_next_observation)
    pos_err, orn_err = goal_errors_np(final_obs, env.ERR_CLIP)
    is_success = (pos_err + orn_err) < env.success_threshold
    return pos_err, orn_err, is_success


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reward_dir", type=str, help="Experiment dir, e.g. runs/<project>/<exp_name>, containing one subfolder per seed")
    parser.add_argument("--std-dev", type=float, default=0.4, help="Policy std_dev used at training time (only affects logstd param shape, not the deterministic rollout)")
    parser.add_argument("--no-action-clipping", action="store_true", help="Set if the run used --algorithm.action_clipping_and_rescaling=False")
    parser.add_argument("--out", type=str, default=None, help="Output CSV path (default: <reward_dir>/final_error_table.csv)")
    args = parser.parse_args()

    reward_dir = Path(args.reward_dir)
    seed_dirs = sorted(d for d in reward_dir.iterdir() if d.is_dir() and (d / "models" / "best.model.zip").exists())
    if not seed_dirs:
        sys.exit(f"No seed subfolders with models/best.model.zip found under {reward_dir}")

    # random_TPos doesn't matter here: we override the initial block pose ourselves per position.
    env = PushT(render=False, random_TPos=False)

    rows = []
    for seed_dir in seed_dirs:
        model_zip = seed_dir / "models" / "best.model.zip"
        try:
            pos_err, orn_err, is_success = evaluate_seed(model_zip, env, args.std_dev, not args.no_action_clipping)
        except Exception as e:
            print(f"Skipping {seed_dir.name}, error: {e}")
            continue

        total_err = pos_err + orn_err
        rows.append({"seed": seed_dir.name, **{str(idx): total_err[idx] for idx in range(NR_PRESET_POSITIONS)}})
        print(f"{seed_dir.name}: final_total_err={total_err.round(4).tolist()}")

    if not rows:
        sys.exit("No seed evaluated successfully, nothing to write.")

    out_path = Path(args.out) if args.out else reward_dir / "final_error_table.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(seed_dirs)}x{NR_PRESET_POSITIONS} final-error table to {out_path}")


if __name__ == "__main__":
    main()
