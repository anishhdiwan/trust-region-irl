"""Where can the pizza be? Drive the tool home -> pizza centre -> clockwise spiral to the rim -> one rim circle at the
spreading height, for a grid of (centre x, centre y, radius), and record how well the IK copes. The reachable band in
pizza_dr_mjx (BAND_AT_RMIN, BAND_AT_RMAX) comes from this sweep.

A candidate is reachable when the path finishes with no singular damping (sigma_min > 0.01), no joint-velocity cap
engaged, joint-limit margin > 0.05 rad, height error < 5 mm, no contact, and tracking within 1 cm on the outer half.

    python scripts/pizza/workspace_sweep.py OUT_DIR      (env IMPL=jax|warp)
"""
import json, os, sys, time
import numpy as np
import jax, jax.numpy as jnp
from mujoco import mjx
from mujoco.mjx._src import support as mjx_support
from trust_region_irl.environments.pizza_dr_mjx.environment import PizzaSpreadDR

OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
IMPL = os.environ.get("IMPL", "jax")
CX = np.arange(0.30, 0.801, 0.05); CY = np.arange(-0.40, 0.401, 0.05); RS = np.array([0.10, 0.125, 0.15, 0.175, 0.20])
PITCH, SPEED, DS = 0.03, 0.25, 0.005   # spiral spacing (m, about two tool diameters), commanded speed (m/s), path resolution (m)
WIN, LOOK, T = 20, 4, 640
# pass criteria
MAX_TRACK, MAX_SAT_FRAC, SR_EPS, MIN_JOINT_MARGIN, MAX_Z_ERR = 0.01, 0.0, 1e-2, 0.05, 0.005   # tracking judged on the outer half of the pizza; any IK velocity-cap engagement fails

cands = np.array([(cx, cy, r) for r in RS for cx in CX for cy in CY], dtype=np.float32)
N = len(cands); print("candidates", N, flush=True)
env = PizzaSpreadDR(render=False, impl=IMPL, nr_envs=N)
data0 = mjx.forward(env.mjx_model, env.mjx_data.replace(qpos=env.initial_qpos, qvel=env.initial_qvel))   # arm at qhome, tool at its home pose
home = np.asarray(data0.xpos[env.ee_body_id], dtype=np.float32)

def path(cx, cy, r):
    """home -> centre (straight), clockwise Archimedean spiral to radius r, then one full rim circle; points every DS."""
    pts = [home[:2]]
    c = np.array([cx, cy]); d = np.linalg.norm(c - home[:2]); n = max(int(d / DS), 1)
    pts += list(home[:2] + (c - home[:2]) * (np.arange(1, n + 1) / n)[:, None])
    th, out = 0.0, []
    while True:   # r(th) = PITCH * th / (2 pi), clockwise
        rr = PITCH * th / (2 * np.pi)
        if rr >= r: break
        out.append(c + rr * np.array([np.cos(-th), np.sin(-th)]))
        th += DS / max(rr, DS)
    th0 = th
    for k in range(int(2 * np.pi * r / DS) + 1):
        a = th0 + k * DS / r
        out.append(c + r * np.array([np.cos(-a), np.sin(-a)]))
    return np.array(pts + out, dtype=np.float32)

paths = [path(*c) for c in cands]; L = max(len(p) for p in paths)
refs = np.stack([np.concatenate([p, np.repeat(p[-1:], L - len(p), 0)]) for p in paths]); last = np.array([len(p) - 1 for p in paths])
print("path points max", L, "| longest path (m)", round(L * DS, 2), flush=True)
refs, last = jnp.asarray(refs), jnp.asarray(last)
cent = jnp.asarray(cands[:, :2])
amp = SPEED / env.MAX_SPEED; dt = 1.0 / env.CONTROL_FREQ
lo, hi = np.asarray(env.mj_model.jnt_range[:7, 0]), np.asarray(env.mj_model.jnt_range[:7, 1])

@jax.vmap
def diag(d):
    J = jnp.concatenate([a.T for a in mjx_support.jac(env.mjx_model, d, d.xpos[env.ee_body_id], env.ee_body_id)], 0)[:, env.arm_dofadr]
    q = d.qpos[env.arm_qadr]
    return jnp.linalg.svd(J, compute_uv=False)[-1], jnp.minimum(q - lo, hi - q).min(), d.xpos[env.ee_body_id], jnp.min(jnp.where(jnp.arange(d.contact.dist.shape[0]) >= 0, d.contact.dist, 1.0))

def physics_step(data, a):
    """One control step: planar velocity a (in units of MAX_SPEED) through the env's IK, height held at Z_HOLD."""
    def sub(d, _):
        return mjx.step(env.mjx_model, d.replace(ctrl=env._differential_ik(d, a * env.MAX_SPEED, env.Z_HOLD))), None
    return jax.lax.scan(sub, data, None, length=env.nr_intermediate_steps)[0]

def step(carry, _):
    data, idx = carry
    _, _, p0, _ = diag(data)
    pos = p0[:, :2]
    win = jnp.minimum(idx[:, None] + jnp.arange(WIN)[None], last[:, None])
    cand = jnp.take_along_axis(refs, win[:, :, None], axis=1)
    dist = jnp.linalg.norm(cand - pos[:, None], axis=-1)
    idx = jnp.minimum(idx + jnp.argmin(dist, 1), last)
    tgt = jnp.take_along_axis(refs, jnp.minimum(idx + LOOK, last)[:, None, None], axis=1)[:, 0]
    d = tgt - pos; done = idx >= last
    a = (amp * ~done)[:, None] * d / (jnp.linalg.norm(d, axis=-1, keepdims=True) + 1e-9)
    dq = jax.vmap(lambda d, v: env._differential_ik(d, v, env.Z_HOLD))(data, a * env.MAX_SPEED)
    sat = jnp.max(jnp.abs(dq) / env.joint_vel_limits, axis=-1) > 0.999   # the IK scaled the joint velocities down (cap engaged)
    nd = jax.vmap(physics_step)(data, a)
    smin, jmarg, p1, cdist = diag(nd)
    moved = jnp.linalg.norm(p1[:, :2] - pos, axis=-1)
    ratio = sat.astype(jnp.float32)
    rad = jnp.linalg.norm(p1[:, :2] - cent, axis=-1)
    return (nd, idx), (jnp.min(dist, 1), ratio, smin, jmarg, jnp.abs(p1[:, 2] - env.Z_HOLD), cdist, done, idx, rad)

data = jax.vmap(lambda _: mjx.forward(env.mjx_model, env.mjx_data.replace(qpos=env.initial_qpos, qvel=env.initial_qvel)))(jnp.arange(N))   # batched like the env reset
t0 = time.time()
_, out = jax.jit(lambda d: jax.lax.scan(step, (d, jnp.zeros(N, jnp.int32)), None, length=T))(data)
track, ratio, smin, jmarg, zerr, cdist, done, idx, rad = [np.asarray(x) for x in out]
print("rollout + compile s", round(time.time() - t0, 1), flush=True)
travel_end = np.array([int(np.ceil(np.linalg.norm(c[:2] - home[:2]) / DS)) for c in cands])
on_pizza = idx > travel_end[None, :] + 2   # spiral and rim phase
outer = on_pizza & (rad > 0.5 * cands[None, :, 2])
m = lambda x, f, fill, msk=on_pizza: np.where(msk, x, fill).__getattribute__(f)(0)
res = dict(cands=cands, finished=done[-1], track_max=m(track, "max", 0, outer), sat_frac=np.where(on_pizza, ratio, 0).sum(0) / np.maximum(on_pizza.sum(0), 1), sigma_min=m(smin, "min", 1),
           joint_margin=m(jmarg, "min", 9), z_err_max=m(zerr, "max", 0), contact_min=m(cdist, "min", 1))
ok = (res["finished"] & (res["track_max"] < MAX_TRACK) & (res["sat_frac"] <= MAX_SAT_FRAC) & (res["sigma_min"] > SR_EPS)
      & (res["joint_margin"] > MIN_JOINT_MARGIN) & (res["z_err_max"] < MAX_Z_ERR) & (res["contact_min"] > -1e-4))
res["ok"] = ok
np.savez(f"{OUT}/sweep.npz", **res)
summary = {}
for r in RS:
    sel = np.isclose(cands[:, 2], r); c = cands[sel & ok]
    why = {k: int((~v[sel]).sum()) for k, v in dict(unfinished=res["finished"], track=res["track_max"] < MAX_TRACK, ik_cap=res["sat_frac"] <= MAX_SAT_FRAC,
            singular=res["sigma_min"] > SR_EPS, joint_limit=res["joint_margin"] > MIN_JOINT_MARGIN, height=res["z_err_max"] < MAX_Z_ERR, contact=res["contact_min"] > -1e-4).items()}
    summary[f"R={r:.3f}"] = dict(ok=int(len(c)), of=int(sel.sum()), x_range=[float(c[:, 0].min()), float(c[:, 0].max())] if len(c) else None,
                                 y_range=[float(c[:, 1].min()), float(c[:, 1].max())] if len(c) else None, fails=why)
    print(f"R={r:.3f}: {len(c)}/{sel.sum()} ok", summary[f"R={r:.3f}"], flush=True)
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1)
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    f, ax = plt.subplots(1, len(RS), figsize=(4 * len(RS), 4.2), sharey=True)
    for k, r in enumerate(RS):
        sel = np.isclose(cands[:, 2], r)
        ax[k].scatter(cands[sel & ~ok, 0], cands[sel & ~ok, 1], c="lightgray", s=25, label="fails")
        ax[k].scatter(cands[sel & ok, 0], cands[sel & ok, 1], c="tab:green", s=25, label="ok")
        ax[k].plot(home[0], home[1], "k+", ms=12, label="tool home")
        ax[k].set_title(f"R = {r*100:.1f} cm"); ax[k].set_xlabel("pizza centre x (m)"); ax[k].set_aspect("equal")
    ax[0].set_ylabel("pizza centre y (m)"); ax[0].legend(fontsize=7, loc="lower left")
    plt.tight_layout(); plt.savefig(f"{OUT}/feasible_region.png", dpi=110)
except Exception as e:
    print("plot failed", e)
print("done")
