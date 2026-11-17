"""Render videos of a Balance-FR3 policy (deterministic mean action): one episode per goal family plus all of them joined.

The episodes are the same fixed random set as eval_policies.py (same seed); per family the episode with the median
second-half goal distance is shown. The goal is the model's (transparent green) goal ball; the policy panel also draws
the episode's whole goal path on the plate (green line) and the ball's path so far (blue line). With --expert-raw, a
BaMPC demo of the same family (its own random episode) is shown on the right while the clip is no longer than the demos.

    MUJOCO_GL=egl python scripts/balance/render_policy.py CKPT OUT_DIR [--seconds 10] [--label TEXT]
        [--expert-raw balance_fr3_varied_300.npz] [--n-eval 100] [--dynamics ball_mass_scale=4 ...]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import mujoco
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

from trust_region_irl.environments.balance_fr3_mjx.environment import BalanceFR3
from trust_region_irl.algorithms.trirl_ppo_spr.flax_full_jit.policy import Policy
from trust_region_irl.policy_checkpoint import load_policy_params

parser = argparse.ArgumentParser()
parser.add_argument("ckpt")
parser.add_argument("out_dir")
parser.add_argument("--seconds", type=float, default=10.0, help="clip length; training episodes are 10 s")
parser.add_argument("--label", default="TRIRL policy (SPR reward features)")
parser.add_argument("--expert-raw", default="", help="raw BaMPC demos (collect_demos.py output) for a side-by-side expert panel")
parser.add_argument("--n-eval", type=int, default=100)
parser.add_argument("--impl", default="warp", help="MJX backend: jax | warp")
parser.add_argument("--seed", type=int, default=2026)
parser.add_argument("--dynamics", nargs="*", default=[], help="env dynamics variant, e.g. ball_mass_scale=4")
args = parser.parse_args()
OUT = Path(args.out_dir); OUT.mkdir(parents=True, exist_ok=True)
FAMS = ["still", "line", "ellipse", "lissajous", "waypoints"]
W, H_PX, CAM = 640, 480, "video"

env = BalanceFR3(render=False, nr_envs=args.n_eval, impl=args.impl, horizon=int(round(args.seconds * BalanceFR3.CONTROL_FREQ)),
                 **{k: float(v) for k, v in (x.split("=") for x in args.dynamics)})
T = env.horizon
policy = Policy((2,), 1.0, jnp.arange(env.single_observation_space.shape[0]))


@jax.jit
def rollout(params, keys):
    s0 = env.reset(keys, True)
    def step(s, _):
        s2 = env.step(s, policy.apply(params, s.next_observation)[0])
        return s2, dict(obs=s2.actual_next_observation, dist=s2.info["env_info/goal_dist"], lost=s2.info["env_info/ball_lost"], qpos=s2.data.qpos)
    _, rec = jax.lax.scan(step, s0, None, length=T)
    return s0.data.qpos, s0.next_observation, s0.info_episode_store["fam"], rec


qpos0, obs0, fam, rec = rollout(load_policy_params(args.ckpt), jax.random.split(jax.random.PRNGKey(args.seed), args.n_eval))
qpos0, obs0, fam = np.asarray(qpos0), np.asarray(obs0), np.asarray(fam)
rec = {k: np.asarray(v) for k, v in rec.items()}
# the last step's data is already reset (horizon end): use the last recorded obs's step T-1 qpos for the final frame
qpos = np.concatenate([qpos0[None], rec["qpos"][:-1]], 0)            # (T, N, nq): frame t = state at time t
obs = np.concatenate([obs0[None], rec["obs"][:-1]], 0)
dist = rec["dist"]

WITH_EXPERT = False
if args.expert_raw:
    ex = np.load(args.expert_raw)
    WITH_EXPERT = ex["qpos"].shape[1] >= T   # the expert panel only while the demos last
    ex_fam = ex["demo_meta"][:, 0].astype(int)
    ex_q, ex_goal, ex_goalq = ex["qpos"], ex["goal_pos"], ex["goal_quat"]
    ex_dist = np.linalg.norm(ex_q[..., 0:2] - ex_goal[..., 0:2], axis=-1)   # (demos, steps)

m = env.mj_model
goal_mocap = m.body_mocapid[m.body("goal").id]
renderer = mujoco.Renderer(m, H_PX, W)
scene_opt = mujoco.MjvOption(); scene_opt.sitegroup[:] = 0   # hide sites (the ball's 3 marker spheres; visual only)
data = mujoco.MjData(m)
try:
    font = ImageFont.truetype("DejaVuSans.ttf", 18)
except OSError:
    font = ImageFont.load_default()


def to_plate(o, idx):
    """Plate-local position of the world point o[..., idx:idx+3] (plate position 9:12, quaternion wxyz 12:16)."""
    p, q, x = o[..., 9:12], o[..., 12:16], o[..., idx:idx + 3]
    w, v = q[..., :1], -q[..., 1:4]
    d = x - p; t = 2 * np.cross(v, d)
    return d + w * t + np.cross(v, t)


LINE_Z = env.rest_z - 0.038   # just above the plate surface (ball centre height minus the 4 cm radius)


def add_path(scn, local_xy, rgba, width):
    """Draw a polyline given in plate-local xy, carried by the plate's current pose."""
    if len(local_xy) < 2:
        return
    P, R = data.xpos[env.plate_id], data.xmat[env.plate_id].reshape(3, 3)
    pts = P + np.c_[local_xy, np.full(len(local_xy), LINE_Z)] @ R.T
    for a, b in zip(pts[:-1], pts[1:]):
        if scn.ngeom >= scn.maxgeom or np.linalg.norm(b - a) < 1e-5:
            continue
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9), np.asarray(rgba, np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, a, b)
        scn.ngeom += 1


def frame(q, goal_pos, goal_quat, lines, paths=()):
    data.qpos[:] = q
    data.mocap_pos[goal_mocap] = goal_pos; data.mocap_quat[goal_mocap] = goal_quat
    mujoco.mj_forward(m, data)
    renderer.update_scene(data, camera=CAM, scene_option=scene_opt)
    for xy, rgba, width in paths:
        add_path(renderer.scene, xy, rgba, width)
    img = Image.fromarray(renderer.render())
    dr = ImageDraw.Draw(img)
    for k, s in enumerate(lines):
        dr.text((10, 8 + 22 * k), s, fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
    return np.asarray(img)


summary, all_frames = {}, []
for f, name in enumerate(FAMS):
    idx = np.where(fam == f)[0]
    if len(idx) == 0:
        continue
    e2 = dist[T // 2:, idx].mean(0)
    i = int(idx[np.argsort(e2)[len(e2) // 2]])                         # median episode of this family
    summary[name] = dict(policy_episode=i, policy_goal_dist_2nd_half_cm=float(100 * dist[T // 2:, i].mean()),
                         policy_goal_dist_after_2s_cm=float(100 * dist[2 * env.CONTROL_FREQ:, i].mean()),
                         policy_goal_dist_max_after_2s_cm=float(100 * dist[2 * env.CONTROL_FREQ:, i].max()),
                         family_median_after_2s_cm=float(100 * np.median(dist[2 * env.CONTROL_FREQ:, idx].mean(0))),
                         family_ball_lost=float(rec["lost"][:, idx].max(0).mean()))
    if WITH_EXPERT:
        ej = np.where(ex_fam == f)[0]; j = int(ej[np.argsort(ex_dist[ej, T // 2:T].mean(1))[len(ej) // 2]])
        summary[name].update(expert_demo=j, expert_goal_dist_2nd_half_cm=float(100 * ex_dist[j, T // 2:T].mean()))
    frames = []
    goal_xy, ball_xy = to_plate(obs[:, i], 19)[:, :2], to_plate(obs[:, i], 0)[:, :2]
    for t in range(T):
        a = frame(qpos[t, i], obs[t, i, 19:22], obs[t, i, 12:16],
                  [f"{args.label}  |  goal: {name}", f"t {t / env.CONTROL_FREQ:4.1f} s   ball-goal {100 * np.linalg.norm(obs[t, i, 0:2] - obs[t, i, 19:21]):5.1f} cm"],
                  paths=((goal_xy[::2], (0.1, 0.8, 0.1, 0.9), 0.003), (ball_xy[:t + 1:2], (0.2, 0.45, 0.95, 0.9), 0.002)))
        if WITH_EXPERT:
            b = frame(ex_q[j, t], ex_goal[j, t], ex_goalq[j, t],
                      [f"expert (BaMPC MPPI)  |  goal: {name}", f"t {t / env.CONTROL_FREQ:4.1f} s   ball-goal {100 * np.linalg.norm(ex_q[j, t, 0:2] - ex_goal[j, t, 0:2]):5.1f} cm"])
            a = np.concatenate([a, np.full((H_PX, 6, 3), 255, np.uint8), b], 1)
        frames.append(a)
    imageio.mimsave(OUT / f"balance_{name}.mp4", frames, fps=env.CONTROL_FREQ, quality=8, macro_block_size=1)
    all_frames += frames
    print(name, json.dumps(summary[name]), flush=True)
imageio.mimsave(OUT / "balance_all_families.mp4", all_frames, fps=env.CONTROL_FREQ, quality=8, macro_block_size=1)
json.dump(summary, open(OUT / "videos.json", "w"), indent=1)
print("wrote", OUT)
