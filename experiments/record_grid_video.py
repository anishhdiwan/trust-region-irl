"""Roll out several (seed, preset-position) policies and compose their
full-length rollouts into a single 1xN grid video, each panel captioned with
its seed and preset position.

Reuses the exact deterministic-rollout logic from eval_final_error.py /
capture_best_rollout.py (same policy restore, same env/step function, same
preset-position reset), but renders every step of the episode (rather than a
handful of evenly-spaced snapshots) with mujoco's offscreen Renderer, then
stitches the per-policy frame sequences side by side into one video.

Run with the same environment used for training (conda env "trirl"), from
the experiments/ directory:

    conda activate trirl
    cd experiments
    python record_grid_video.py runs/ICRA_TRIRL_10_2F/pusht_ppo_fb \
        --clips 1789307943:2 1789318417:3 1789328967:1
"""
import argparse
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
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trust_region_irl.environments.pusht_mjx.environment import PushT
from trust_region_irl.algorithms.trirl_ppo_fb.flax_full_jit.policy import Policy, get_processed_action_function
from eval_final_error import load_policy_params, make_initial_state_for_position
from capture_best_rollout import brighten_scene, brighten_image, lookup_error


def roll_out_frames(reward_dir, seed, position_idx, args, env):
    model_zip = reward_dir / seed / "models" / "best.model.zip"
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

    single_state = make_initial_state_for_position(env, position_idx)
    state = jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves, axis=0), single_state)

    def render(state):
        data = mjx.get_data(env.mj_model, state.data)[0]
        renderer.update_scene(data, camera=args.camera)
        return brighten_image(renderer.render(), args.gamma)

    renderer = mujoco.Renderer(env.mj_model, args.height, args.width)
    frames = []
    freeze_frame = None
    for step_idx in range(env.horizon):
        if freeze_frame is not None:
            frames.append(freeze_frame)
            continue
        frames.append(render(state))
        state = step_fn(state)
        if bool(np.asarray(state.terminated)[0]):
            print(f"  success at step {step_idx + 1}, freezing")
            freeze_frame = render(state)
    renderer.close()
    return frames


def caption_panel(frame, text, caption_height, font):
    h, w, _ = frame.shape
    panel = Image.new("RGB", (w, h + caption_height), color=(20, 20, 20))
    panel.paste(Image.fromarray(frame), (0, caption_height))
    draw = ImageDraw.Draw(panel)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((w - text_w) / 2, (caption_height - text_h) / 2 - bbox[1]), text, fill=(255, 255, 255), font=font)
    return np.array(panel)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reward_dir", type=str, help="Experiment dir, e.g. runs/<project>/<exp_name>")
    parser.add_argument("--clips", type=str, nargs="+", required=True,
                         help="One or more seed:position pairs, e.g. 1789307943:2 1789318417:3 1789328967:1")
    parser.add_argument("--error-csv", type=str, default=None, help="final_error_table.csv (default: <reward_dir>/final_error_table.csv)")
    parser.add_argument("--camera", type=str, default="comparison")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--std-dev", type=float, default=0.4)
    parser.add_argument("--no-action-clipping", action="store_true")
    parser.add_argument("--light-factor", type=float, default=1.0, help="Scale scene light intensities (>1 brightens)")
    parser.add_argument("--gamma", type=float, default=1.0, help="Gamma-correct rendered pixels, <1 brightens (e.g. 0.6)")
    parser.add_argument("--fps", type=int, default=20, help="20 Hz matches the env's control frequency (real-time playback)")
    parser.add_argument("--caption-height", type=int, default=44)
    parser.add_argument("--out", type=str, default="plots/policy_grid_comparison.mp4")
    args = parser.parse_args()

    reward_dir = Path(args.reward_dir)
    error_csv = Path(args.error_csv) if args.error_csv else reward_dir / "final_error_table.csv"
    clips = [tuple(c.split(":")) for c in args.clips]
    clips = [(seed, int(pos)) for seed, pos in clips]

    env = PushT(render=False, random_TPos=False)
    if args.light_factor != 1.0:
        brighten_scene(env.mj_model, args.light_factor)

    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)

    panel_sequences = []
    for seed, position_idx in clips:
        error = lookup_error(error_csv, seed, position_idx)
        label = f"seed {seed}  •  pos {position_idx}" + (f"  •  err {error:.2f}" if error is not None else "")
        print(f"Rolling out {label} ...")
        frames = roll_out_frames(reward_dir, seed, position_idx, args, env)
        captioned = [caption_panel(f, label, args.caption_height, font) for f in frames]
        panel_sequences.append(captioned)

    num_frames = min(len(seq) for seq in panel_sequences)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Compositing {len(clips)}x1 grid, {num_frames} frames, -> {out_path}")
    with imageio.get_writer(out_path, fps=args.fps, macro_block_size=None) as writer:
        for t in range(num_frames):
            grid_frame = np.hstack([seq[t] for seq in panel_sequences])
            writer.append_data(grid_frame)

    print(f"Saved grid video to {out_path}")


if __name__ == "__main__":
    main()
