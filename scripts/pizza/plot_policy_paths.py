"""Roll out trained pizza_dr_mjx policies (deterministic mean action) on the same fixed random pizzas and plot the tool paths.

Every checkpoint is rolled out without the wander cut (full horizon), so you can see what a policy does after covering
the pizza; the step where a cut (off the pizza by more than --wander-margin for the env's hold time, while spreading)
would have ended the episode is marked. Paths are drawn in pizza-centred cm.

Writes to OUT_DIR: paths.png (8 pizzas per checkpoint), coverage.png (coverage over the episode), metrics.json and one
rollout_<label>.npz per checkpoint.

    python scripts/pizza/plot_policy_paths.py OUT_DIR CKPT [CKPT ...] [--n-eval 64] [--impl warp]
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

from trust_region_irl.environments.pizza_dr_mjx.environment import PizzaSpreadDR
from trust_region_irl.algorithms.trirl_ppo_spr.flax_full_jit.policy import Policy
from trust_region_irl.policy_checkpoint import load_policy_params

parser = argparse.ArgumentParser()
parser.add_argument("out_dir")
parser.add_argument("ckpts", nargs="+", help="model archives (e.g. runs/.../models/latest.model.zip)")
parser.add_argument("--n-eval", type=int, default=64, help="number of random pizzas")
parser.add_argument("--n-show", type=int, default=8, help="pizzas drawn per checkpoint")
parser.add_argument("--wander-margin", type=float, default=0.025, help="cut to mark, m beyond the rim")
parser.add_argument("--impl", default="warp", help="MJX backend: jax | warp")
parser.add_argument("--seed", type=int, default=12345)
args = parser.parse_args()
OUT = Path(args.out_dir); OUT.mkdir(parents=True, exist_ok=True)
N = args.n_eval
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
C_TRAVEL, C_SPREAD, C_AFTER, C_LIFT = "#9a9994", "#2a78d6", "#eb6834", "#e34948"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

env = PizzaSpreadDR(render=False, nr_envs=N, impl=args.impl, wander_margin=10.0)   # no cut during the rollout
H, HOLD = env.horizon, int(round(env.wander_hold_s * env.CONTROL_FREQ))
policy = Policy((2,), 1.0, jnp.arange(env.single_observation_space.shape[0]))


@jax.jit
def rollout(params, keys):
    s0 = env.reset(keys, True)
    def step(s, _):
        mean, _ = policy.apply(params, s.next_observation)
        s2 = env.step(s, mean)
        rec = dict(ee=s2.actual_next_observation[:, :3], landed=s2.info["env_info/touched_down"], lifted=s2.info["env_info/lifted"],
                   cov=jnp.where(s2.terminated | s2.truncated, s2.info["rollout/coverage"], s2.info["env_info/coverage"]))
        return s2, rec
    return s0.next_observation, jax.lax.scan(step, s0, None, length=H)[1]


def label(ckpt):
    p = Path(ckpt)
    return f"{p.parent.parent.name}/{p.name.split('.')[0]}"   # <run>/<best|latest>


keys = jax.random.split(jax.random.PRNGKey(args.seed), N)
results, runs = {}, []
for ckpt in args.ckpts:
    obs0, rec = rollout(load_policy_params(ckpt), keys)
    pizza = np.asarray(obs0)[:, 3:6]; rec = {k: np.asarray(v) for k, v in rec.items()}   # rec[k]: (H, N)
    r = np.linalg.norm(rec["ee"][:, :, :2] - pizza[None, :, :2], axis=-1); R = pizza[None, :, 2]
    landed, lifted = rec["landed"] > 0.5, rec["lifted"] > 0.5
    beyond = landed & (r - R > args.wander_margin)
    cut_step = np.full(N, -1)
    for i in range(N):   # first step with HOLD consecutive steps beyond the margin
        run_len = 0
        for t in range(H):
            run_len = run_len + 1 if beyond[t, i] else 0
            if run_len >= HOLD:
                cut_step[i] = t; break
    cov = rec["cov"]
    touch = np.where(landed.any(0), landed.argmax(0), -1)
    m = dict(coverage_final=float(cov[-1].mean()), success_final=float((cov[-1] >= env.success_threshold).mean()),
             touched_down=float((touch >= 0).mean()), lifted=float(lifted[-1].mean()), would_be_cut=float((cut_step >= 0).mean()),
             coverage_at_cut=float(np.mean(cov[cut_step[cut_step >= 0], np.where(cut_step >= 0)[0]])) if (cut_step >= 0).any() else None,
             off_pizza_frac_while_landed=float((landed & (r > R)).sum() / max(landed.sum(), 1)))
    name = label(ckpt); runs.append(name); results[name] = m
    print(name, json.dumps(m), flush=True)
    np.savez_compressed(OUT / f"rollout_{name.replace('/', '_')}.npz", pizza=pizza, cut_step=cut_step, **rec)
json.dump(results, open(OUT / "metrics.json", "w"), indent=1)

fig, axes = plt.subplots(len(runs), args.n_show, figsize=(2.3 * args.n_show, 2.5 * len(runs) + 0.6), facecolor=SURF, squeeze=False)
for i, name in enumerate(runs):
    d = np.load(OUT / f"rollout_{name.replace('/', '_')}.npz")
    for j in range(args.n_show):
        ax = axes[i, j]; ax.set_facecolor(SURF)
        cx, cy, R = d["pizza"][j]; p = (d["ee"][:, j, :2] - [cx, cy]) * 100; Rc = R * 100
        landed, lifted, cs = d["landed"][:, j] > 0.5, d["lifted"][:, j] > 0.5, d["cut_step"][j]
        after = np.arange(len(p)) > cs if cs >= 0 else np.zeros_like(landed)
        for rad, ls, c in ((Rc, "-", INK), (Rc - env.CRUST * 100, ":", INK2), (Rc + args.wander_margin * 100, "--", INK2)):
            ax.add_patch(plt.Circle((0, 0), rad, fill=False, ls=ls, lw=0.8, color=c))
        for mask, c in ((~landed & ~lifted, C_TRAVEL), (landed & ~after, C_SPREAD), (landed & after, C_AFTER), (lifted, C_LIFT)):
            ax.plot(np.where(mask, p[:, 0], np.nan), np.where(mask, p[:, 1], np.nan), color=c, lw=1.0)
        ax.plot(*p[0], "o", ms=4, color=C_TRAVEL); ax.plot(*p[-1], "s", ms=4, color=INK)
        if cs >= 0:
            ax.plot(*p[cs], "x", ms=7, mew=2, color=C_AFTER)
        L = Rc + 6; ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#d8d7d2")
        ax.set_title(f"R {Rc:.0f} cm  cov {d['cov'][-1, j]:.2f}" + (f" | cut@{d['cov'][cs, j]:.2f}" if cs >= 0 else ""), fontsize=7, color=INK2)
        if j == 0:
            ax.set_ylabel(name, fontsize=8, color=INK)
handles = [plt.Line2D([], [], color=c, lw=2, label=l) for c, l in ((C_TRAVEL, "travel (z +5 cm)"), (C_SPREAD, "spreading"),
           (C_AFTER, "after the cut would fire"), (C_LIFT, "lifted off"))] + \
          [plt.Line2D([], [], color=C_AFTER, marker="x", ls="", label="cut point"), plt.Line2D([], [], color=INK, marker="s", ls="", label="end"),
           plt.Line2D([], [], color=INK2, ls="--", label=f"rim + {args.wander_margin * 100:.1f} cm"), plt.Line2D([], [], color=INK2, ls=":", label="sauce edge (R - crust)")]
fig.suptitle("pizza_dr_mjx tool paths, deterministic policy, no cut during the rollout (pizza-centred cm)", fontsize=11, color=INK, y=0.995)
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.975), ncol=8, fontsize=8, frameon=False)
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(OUT / "paths.png", dpi=130, facecolor=SURF); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4), facecolor=SURF); ax.set_facecolor(SURF)
for k, name in enumerate(runs):
    d = np.load(OUT / f"rollout_{name.replace('/', '_')}.npz"); c = SERIES[k % len(SERIES)]
    ax.plot(d["cov"].mean(1), color=c, lw=2, label=name)
    ax.fill_between(np.arange(H), np.percentile(d["cov"], 25, 1), np.percentile(d["cov"], 75, 1), color=c, alpha=0.12, lw=0)
ax.axhline(env.success_threshold, color=INK2, ls="--", lw=0.8)
ax.set_title("sauce coverage over the episode (mean, interquartile band)", fontsize=10, color=INK)
ax.set_xlabel("step (20 Hz)"); ax.set_ylabel("coverage"); ax.legend(frameon=False, fontsize=7); ax.grid(alpha=0.25)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig(OUT / "coverage.png", dpi=130, facecolor=SURF); plt.close(fig)
print("wrote", OUT)
