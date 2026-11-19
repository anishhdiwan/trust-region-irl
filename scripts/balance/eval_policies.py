"""Evaluate Balance-FR3 policies (deterministic mean action) on fixed random episodes, next to a zero-action baseline and,
optionally, the expert demos.

Per checkpoint and goal family: ball-goal distance (whole episode and second half) and the fraction of episodes where
the ball is lost. Every checkpoint sees the same episodes (same seed). Writes to OUT_DIR: metrics.json, paths.png
(plate-local ball and goal paths of the first checkpoint, 4 episodes per family) and goal_dist.png (distance over time).

    python scripts/balance/eval_policies.py OUT_DIR CKPT [CKPT ...] [--expert trirl_dataset/rl_expert/balance_fr3_varied_300.npz]
        [--n-eval 100] [--impl warp] [--dynamics ball_mass_scale=4 tilt_gain_scale=0.5 ...]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from trust_region_irl.environments.balance_fr3_mjx.environment import BalanceFR3
from trust_region_irl.algorithms.trirl_ppo_spr.flax_full_jit.policy import Policy
from trust_region_irl.policy_checkpoint import load_policy_params

parser = argparse.ArgumentParser()
parser.add_argument("out_dir")
parser.add_argument("ckpts", nargs="+", help="model archives (TRIRL *.model.zip, PPO *.model)")
parser.add_argument("--expert", default="", help="TRIRL expert dataset (from make_expert_dataset.py) to report the expert's own distance")
parser.add_argument("--n-eval", type=int, default=100)
parser.add_argument("--impl", default="warp", help="MJX backend: jax | warp")
parser.add_argument("--seed", type=int, default=2026)
parser.add_argument("--dynamics", nargs="*", default=[], help="env dynamics variant, e.g. ball_mass_scale=4 tilt_gain_scale=0.5")
args = parser.parse_args()
OUT = Path(args.out_dir); OUT.mkdir(parents=True, exist_ok=True)
DYN = {k: float(v) for k, v in (x.split("=") for x in args.dynamics)}
FAMS = ["still", "line", "ellipse", "lissajous", "waypoints"]
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
C_GOAL, C_BALL = "#9a9994", "#2a78d6"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

env = BalanceFR3(render=False, nr_envs=args.n_eval, impl=args.impl, **DYN)
H = env.horizon
policy = Policy((2,), 1.0, jnp.arange(env.single_observation_space.shape[0]))


def rollout(act_fn):
    @jax.jit
    def run(params, keys):
        s0 = env.reset(keys, True)
        def step(s, _):
            s2 = env.step(s, act_fn(params, s.next_observation))
            return s2, dict(obs=s2.actual_next_observation, dist=s2.info["env_info/goal_dist"], lost=s2.info["env_info/ball_lost"])
        return s0.info_episode_store["fam"], jax.lax.scan(step, s0, None, length=H)[1]
    return run


def plate_local(obs, idx):
    """Plate-local xy of the world point obs[..., idx:idx+3] (plate position 9:12, quaternion wxyz 12:16)."""
    p, q, x = obs[..., 9:12], obs[..., 12:16], obs[..., idx:idx + 3]
    w, v = q[..., :1], -q[..., 1:4]   # inverse rotation
    d = x - p; t = 2 * np.cross(v, d)
    return (d + w * t + np.cross(v, t))[..., :2]


def summarise(dist, lost, fam):
    """dist, lost: (time, episodes); fam: (episodes,)."""
    def one(m):
        return dict(goal_dist_cm=float(100 * dist[:, m].mean()), goal_dist_2nd_half_cm=float(100 * dist[H // 2:, m].mean()),
                    ball_lost=float(lost[:, m].max(0).mean()), n=int(m.sum()))
    return {"all": one(np.ones_like(fam, bool)), **{name: one(fam == f) for f, name in enumerate(FAMS) if (fam == f).any()}}


def label(ckpt):
    p = Path(ckpt)
    return f"{p.parent.parent.name}/{p.name.split('.')[0]}"   # <run>/<best|latest>


keys = jax.random.split(jax.random.PRNGKey(args.seed), args.n_eval)
results, recs = {"dynamics": DYN}, {}
fam, rec = rollout(lambda p, o: jnp.zeros(o.shape[:-1] + (2,)))(None, keys)
fam = np.asarray(fam); recs["zero action"] = {k: np.asarray(v) for k, v in rec.items()}
run_policy = rollout(lambda p, o: policy.apply(p, o)[0])
for ckpt in args.ckpts:
    _, rec = run_policy(load_policy_params(ckpt), keys)
    recs[label(ckpt)] = {k: np.asarray(v) for k, v in rec.items()}
for name, rec in recs.items():
    results[name] = summarise(rec["dist"], rec["lost"], fam)
    print(name, json.dumps(results[name]["all"]), flush=True)

e_dist = e_fam = None
if args.expert:   # the expert's own distance: ball xy vs goal xy in the recorded states (no ball is lost in the demos)
    d = np.load(args.expert)
    T_e = int(d["episode_lengths"][0])
    ns = d["next_states"].reshape(-1, T_e, d["next_states"].shape[-1]).transpose(1, 0, 2)
    e_dist, e_fam = np.linalg.norm(ns[..., 0:2] - ns[..., 19:21], axis=-1), d["demo_meta"][:, 0].astype(int)
    e = dict(goal_dist_cm=float(100 * e_dist.mean()), goal_dist_2nd_half_cm=float(100 * e_dist[T_e // 2:].mean()), n=int(e_dist.shape[1]))
    results["expert"] = {"all": e, **{n: dict(goal_dist_cm=float(100 * e_dist[:, e_fam == f].mean()),
                                              goal_dist_2nd_half_cm=float(100 * e_dist[T_e // 2:, e_fam == f].mean())) for f, n in enumerate(FAMS)}}
    print("expert", json.dumps(e), flush=True)
json.dump(results, open(OUT / "metrics.json", "w"), indent=1)

# paths of the first checkpoint: one row per family, 4 episodes each
first = label(args.ckpts[0]); pr = recs[first]
fig, axes = plt.subplots(5, 4, figsize=(11, 13.5), facecolor=SURF)
for f in range(5):
    eps = np.where(fam == f)[0][:4]
    for j in range(4):
        ax = axes[f, j]; ax.set_facecolor(SURF)
        if j >= len(eps):
            ax.axis("off"); continue
        i = eps[j]
        ax.add_patch(plt.Rectangle((-19.5, -25), 39, 50, fill=False, lw=0.8, color=INK2))
        g, b = plate_local(pr["obs"][:, i], 19) * 100, plate_local(pr["obs"][:, i], 0) * 100
        ax.plot(g[:, 0], g[:, 1], color=C_GOAL, lw=2, ls="--"); ax.plot(b[:, 0], b[:, 1], color=C_BALL, lw=1.2)
        ax.plot(*b[0], "o", ms=5, color=C_BALL); ax.plot(*g[0], "o", ms=5, color=C_GOAL); ax.plot(*b[-1], "s", ms=5, color=C_BALL)
        ax.set_xlim(-21, 21); ax.set_ylim(-27, 27); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"mean {100 * pr['dist'][:, i].mean():.1f} cm, 2nd half {100 * pr['dist'][H // 2:, i].mean():.1f} cm", fontsize=8, color=INK2)
        if j == 0:
            ax.set_ylabel(FAMS[f], fontsize=11, color=INK)
fig.suptitle(f"Balance-FR3, {first}: ball vs goal on the plate (plate-local cm, plate 39 x 50 cm)", fontsize=11, color=INK, y=0.995)
fig.legend(handles=[plt.Line2D([], [], color=C_GOAL, lw=2, ls="--", label="goal path"), plt.Line2D([], [], color=C_BALL, lw=1.5, label="ball path"),
                    plt.Line2D([], [], color=INK2, marker="o", ls="", label="start"), plt.Line2D([], [], color=C_BALL, marker="s", ls="", label="end")],
           loc="upper center", bbox_to_anchor=(0.5, 0.975), ncol=4, frameon=False, fontsize=9)
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(OUT / "paths.png", dpi=120, facecolor=SURF); plt.close(fig)

# goal distance over time, per family
fig, axes = plt.subplots(1, 5, figsize=(18, 3.8), facecolor=SURF, sharey=True)
t = np.arange(1, H + 1) / env.CONTROL_FREQ
for f, ax in enumerate(axes):
    ax.set_facecolor(SURF)
    for k, (name, rec) in enumerate(recs.items()):
        c = INK2 if name == "zero action" else SERIES[(k - 1) % len(SERIES)]
        ax.plot(t, 100 * rec["dist"][:, fam == f].mean(1), color=c, lw=1.5, label=name)
    if e_dist is not None:
        te = np.arange(1, e_dist.shape[0] + 1) / env.CONTROL_FREQ
        ax.plot(te, 100 * e_dist[:, e_fam == f].mean(1), color=INK, lw=2, label="expert")
    ax.set_title(f"{FAMS[f]} (n={int((fam == f).sum())})", fontsize=10, color=INK); ax.set_xlabel("time (s)")
    ax.grid(alpha=0.25); ax.set_yscale("log")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axes[0].set_ylabel("ball-goal distance (cm, log)"); axes[-1].legend(frameon=False, fontsize=7)
fig.suptitle("Balance-FR3: mean ball-goal distance over the episode", fontsize=11, color=INK)
fig.tight_layout(); fig.savefig(OUT / "goal_dist.png", dpi=120, facecolor=SURF); plt.close(fig)
print("wrote", OUT)
