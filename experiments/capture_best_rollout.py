"""Roll out the best-performing (seed, preset-position) pair from
final_error_table.csv and save evenly-spaced offscreen snapshots for use in
the paper.

Reuses the exact deterministic-rollout logic from eval_final_error.py
(same policy restore, same env/step function, same preset-position reset),
then renders `--num-captures` frames spaced evenly across the episode with
mujoco's offscreen Renderer (no display/GLFW window needed).

Run with the same environment used for training (conda env "trirl"), from
the experiments/ directory:

    conda activate trirl
    cd experiments
    python capture_best_rollout.py runs/ICRA_TRIRL_10_2F/pusht_ppo_fb
"""
import argparse
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import imageio.v2 as imageio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trust_region_irl.environments.pusht_mjx.environment import PushT
from trust_region_irl.algorithms.trirl_ppo_fb.flax_full_jit.policy import Policy, get_processed_action_function
from eval_final_error import load_policy_params, make_initial_state_for_position


def find_best_seed_position(error_csv):
    with open(error_csv) as f:
        rows = list(csv.DictReader(f))
    best = min(
        ((row["seed"], idx, float(row[str(idx)])) for row in rows for idx in range(6)),
        key=lambda t: t[2],
    )
    return best  # (seed, position_idx, error)


def lookup_error(error_csv, seed, position_idx):
    with open(error_csv) as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if row["seed"] == seed:
            return float(row[str(position_idx)])
    return None


def brighten_scene(model, factor):
    """Scale headlight + scene light intensities to brighten offscreen renders."""
    model.vis.headlight.ambient[:] = np.clip(model.vis.headlight.ambient * factor, 0.0, 1.0)
    model.vis.headlight.diffuse[:] = np.clip(model.vis.headlight.diffuse * factor, 0.0, 1.0)
    model.light_diffuse[:] = np.clip(model.light_diffuse * factor, 0.0, 1.0)
    model.light_ambient[:] = np.clip(model.light_ambient * factor + 0.05 * (factor - 1.0), 0.0, 1.0)


def brighten_image(img, gamma):
    """Gamma-correct (gamma < 1 brightens) an already-rendered uint8 RGB image."""
    if gamma == 1.0:
        return img
    normalized = img.astype(np.float32) / 255.0
    corrected = np.power(normalized, gamma)
    return np.clip(corrected * 255.0, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reward_dir", type=str, help="Experiment dir, e.g. runs/<project>/<exp_name>")
    parser.add_argument("--error-csv", type=str, default=None, help="final_error_table.csv (default: <reward_dir>/final_error_table.csv)")
    parser.add_argument("--seed", type=str, default=None, help="Seed subfolder to use (default: auto-pick the (seed, position) pair with smallest error)")
    parser.add_argument("--position", type=int, default=None, help="Preset position index 0-5 (required if --seed is given)")
    parser.add_argument("--num-captures", type=int, default=5)
    parser.add_argument("--camera", type=str, default="comparison")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--std-dev", type=float, default=0.4)
    parser.add_argument("--no-action-clipping", action="store_true")
    parser.add_argument("--out-dir", type=str, default="plots/paper_captures")
    parser.add_argument("--light-factor", type=float, default=1.0, help="Scale scene light intensities (>1 brightens)")
    parser.add_argument("--gamma", type=float, default=1.0, help="Gamma-correct rendered pixels, <1 brightens (e.g. 0.6)")
    args = parser.parse_args()

    reward_dir = Path(args.reward_dir)
    error_csv = Path(args.error_csv) if args.error_csv else reward_dir / "final_error_table.csv"
    if args.seed is not None:
        if args.position is None:
            sys.exit("--position is required when --seed is given")
        seed, position_idx = args.seed, args.position
        error = lookup_error(error_csv, seed, position_idx)
        print(f"Using seed={seed}, position={position_idx}, final_total_err={error}")
    else:
        seed, position_idx, error = find_best_seed_position(error_csv)
        print(f"Best (seed, position): seed={seed}, position={position_idx}, final_total_err={error:.4f}")

    model_zip = reward_dir / seed / "models" / "best.model.zip"
    env = PushT(render=False, random_TPos=False)
    if args.light_factor != 1.0:
        brighten_scene(env.mj_model, args.light_factor)
    params = load_policy_params(model_zip)

    policy = Policy(
        as_shape=env.single_action_space.shape,
        std_dev=args.std_dev,
        policy_observation_indices=jnp.arange(env.single_observation_space.shape[0]),
    )
    get_processed_action = get_processed_action_function(
        not args.no_action_clipping,
        jnp.array(env.single_action_space.low),
        jnp.array(env.single_action_space.high),
    )

    @jax.jit
    def step_fn(state):
        action_mean, _ = policy.apply(params, state.next_observation)
        action = get_processed_action(action_mean)
        return env.step(state, action)

    # single (unbatched) env at the winning preset position, batched to size 1
    # (mirrors eval_final_error.make_initial_batch, whose stacking also
    # promotes plain-python-float leaves in State.info to jnp arrays)
    single_state = make_initial_state_for_position(env, position_idx)
    state = jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves, axis=0), single_state)

    capture_steps = set(np.linspace(0, env.horizon - 1, args.num_captures, dtype=int).tolist())
    print(f"Capturing at steps: {sorted(capture_steps)}")

    renderer = mujoco.Renderer(env.mj_model, args.height, args.width)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for step_idx in range(env.horizon):
        if step_idx in capture_steps:
            data = mjx.get_data(env.mj_model, state.data)[0]
            renderer.update_scene(data, camera=args.camera)
            img = renderer.render()
            img = brighten_image(img, args.gamma)
            out_path = out_dir / f"seed{seed}_pos{position_idx}_step{step_idx:03d}.png"
            imageio.imwrite(out_path, img)
            saved.append(out_path)
            print(f"Saved {out_path}")
        state = step_fn(state)
    renderer.close()

    print(f"Done. {len(saved)} captures saved to {out_dir}/")


if __name__ == "__main__":
    main()
