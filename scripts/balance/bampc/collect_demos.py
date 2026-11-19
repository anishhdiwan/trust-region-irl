"""Varied-goal expert demos for Balance-FR3 from the BaMPC MPPI planner (task space, 20 Hz, 10 s).

Each demo draws a goal-path family and parameters (plate-local, within BaMPC's on-plate safe area):
  still      goal fixed at a random point
  line       back and forth along a random direction, amplitude 3-8 cm, period 12-25 s, random centre and phase
  ellipse    radii 3-8 cm each, period 12-25 s, clockwise or anticlockwise, random centre and phase
  lissajous  radii 3-6 cm, x:y frequency 1:2 or 2:1, base period 16-30 s
  waypoints  ramped: the goal glides at 2-4 cm/s to a random point, a new one once it arrived and the ball is within 3 cm
Moving paths are rejected if their peak speed exceeds 6 cm/s (the planner tracks well below that). Per step it records
the raw state, the plate-local goal position and velocity, the world goal pose, the mean 2-D action and its 5 sub-steps,
and the next state. Demos where the ball leaves the plate are dropped.

    python scripts/balance/bampc/collect_demos.py OUT.npz N_DEMOS SEED      (from a bampc-code checkout, its conda env)

Env vars: EPISODE_S (default 10) sets the demo length; PER_FAMILY=k keeps sampling until every family has k demos
(N_DEMOS is then ignored and becomes 5 * k); HOLD=1 holds one command per 50 ms step instead of updating it every
10 ms sub-step (what a 20 Hz policy can do; compare the two with compare_hold.py).
"""
import os, sys, time, math
import numpy as np
import mujoco
from bampc.config import planner as planner_profiles, reward as reward_profiles, scenarios as scen
from bampc.config.numerics import load as load_numerics
from bampc.planner import PlannerConfig, build_planner
from bampc.rollout import WarpRolloutEngine
from bampc.task.base import GoalDrift
from bampc.task.balance_fr3 import BalanceFr3
from experiments.common.dr.truth import WarpTruth
from experiments.common.dr.goal import GoalDriver, snapshot

OUT, N, SEED = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
PLAN_HZ, EPISODE_S, VMAX = 20, float(os.environ.get("EPISODE_S", 10.0)), 0.06
PER_FAMILY = int(os.environ.get("PER_FAMILY", 0))
HOLD = int(os.environ.get("HOLD", 0))   # 1: one command per 50 ms step, held over the sub-steps (like the TRIRL policy)
FAMILIES = ["still", "line", "ellipse", "lissajous", "waypoints"]
if PER_FAMILY:
    N = PER_FAMILY * len(FAMILIES)
bank = scen.load("balance_fr3")
cfg = PlannerConfig(algo="mppi", **planner_profiles.load("balance_fr3", sampling="task"))
task = BalanceFr3(sampling_space="task", shape=bank.shape, scale=bank.scale, goal_drift=GoalDrift(),
                  **reward_profiles.load("balance_fr3", sampling="task"), model_config=load_numerics("balance_fr3"),
                  ctrl_range_scale=cfg.ctrl_range_scale)
engine = WarpRolloutEngine(task, num_samples=cfg.num_samples, num_randomizations=1)
planner = build_planner(cfg, task, engine)
truth = WarpTruth(task, {})
m = task.mj_model; md = mujoco.MjData(m)
sub = int(round(1.0 / PLAN_HZ / task.dt)); T = int(round(EPISODE_S * PLAN_HZ))
SAFE = np.array([0.195 - 0.05, 0.25 - 0.05])          # plate half-extents minus BaMPC's 5 cm keep-out
xy_lo = np.min([s.start["block_xy"] for s in bank.scenarios], 0); xy_hi = np.max([s.start["block_xy"] for s in bank.scenarios], 0)
orig_goal_local = task._goal_local
rng = np.random.default_rng(SEED)


class DriftPath:
    """centre + r * sin(2 pi f t + phi) per plate axis; analytic velocity."""
    def __init__(self, centre, r, f, phi):
        self.c, self.r, self.f, self.phi = map(np.asarray, (centre, r, f, phi))
    def pos(self, t): return self.c + self.r * np.sin(2 * np.pi * self.f * t + self.phi)
    def vel(self, t): return self.r * 2 * np.pi * self.f * np.cos(2 * np.pi * self.f * t + self.phi)
    def peak_speed(self):
        ts = np.linspace(0, 2 / max(self.f.min(), 1e-6), 2000)
        return 0.0 if not np.any(self.r) else float(np.max(np.linalg.norm(np.stack([self.vel(t) for t in ts]), axis=1)))
    def __call__(self, t, mj_data): return np.array([*self.pos(t), task._rest_z_local])


class RampedWaypoints:
    """Goal glides at `speed` to a random on-plate point; a new point once it arrived and the ball is within `thr`."""
    def __init__(self, speed, thr, rng):
        self.speed, self.thr, self.rng = speed, thr, rng
        self.p = rng.uniform(-SAFE, SAFE); self.tgt = None; self.t = 0.0; self.v = np.zeros(2)
    def _new_target(self):
        for _ in range(20):
            c = self.rng.uniform(-SAFE, SAFE)
            if np.linalg.norm(c - self.p) >= 2 * self.thr: return c
        return c
    def pos(self, t): return self.p
    def vel(self, t): return self.v
    def __call__(self, t, mj_data):
        dt = max(t - self.t, 0.0); self.t = t
        ball = task._block_local_xyz(mj_data)[:2]
        arrived = self.tgt is not None and np.linalg.norm(self.p - self.tgt) < 1e-4
        if self.tgt is None or (arrived and np.linalg.norm(ball - self.p) < self.thr):
            self.tgt = self._new_target()
        d = self.tgt - self.p; n = np.linalg.norm(d)
        if dt > 0:
            step = min(n, self.speed * dt)
            self.v = d / n * self.speed if n > 1e-9 and step < n else np.zeros(2)
            self.p = self.p + (d / n * step if n > 1e-9 else 0.0)
        return np.array([self.p[0], self.p[1], task._rest_z_local])


def sample_path(counts=None):
    fam = FAMILIES[rng.integers(len(FAMILIES))]
    while PER_FAMILY and counts[FAMILIES.index(fam)] >= PER_FAMILY:   # family quota full: draw another
        fam = FAMILIES[rng.integers(len(FAMILIES))]
    if fam == "waypoints":
        sp = rng.uniform(0.02, 0.04)
        return fam, RampedWaypoints(sp, 0.03, np.random.default_rng(rng.integers(1 << 31))), dict(speed=sp)
    for _ in range(200):
        if fam == "still":
            path = DriftPath(rng.uniform(-SAFE + 0.02, SAFE - 0.02), [0, 0], [0, 0], [0, 0]); info = {}
        elif fam == "line":
            A, th, per = rng.uniform(0.03, 0.08), rng.uniform(0, np.pi), rng.uniform(12, 25)
            r = np.array([A * math.cos(th), A * math.sin(th)]); ph = rng.uniform(0, 2 * np.pi)
            path = DriftPath(rng.uniform(-(SAFE - np.abs(r)), SAFE - np.abs(r)), r, [1 / per] * 2, [ph, ph]); info = dict(amp=A, angle=th, period=per)
        elif fam == "ellipse":
            r = rng.uniform(0.03, 0.08, 2); per = rng.uniform(12, 25); d = rng.choice([-1, 1]); ph = rng.uniform(0, 2 * np.pi)
            path = DriftPath(rng.uniform(-(SAFE - r), SAFE - r), r, [1 / per] * 2, [ph, ph + d * np.pi / 2]); info = dict(rx=r[0], ry=r[1], period=per, dir=int(d))
        else:
            r = rng.uniform(0.03, 0.06, 2); per = rng.uniform(16, 30); f = np.array([1 / per, 2 / per]) if rng.random() < 0.5 else np.array([2 / per, 1 / per])
            path = DriftPath(rng.uniform(-(SAFE - r), SAFE - r), r, f, rng.uniform(0, 2 * np.pi, 2)); info = dict(rx=r[0], ry=r[1], period=per)
        if path.peak_speed() <= VMAX:
            return fam, path, info
    raise RuntimeError("could not sample a path under the speed cap")


keep = {k: [] for k in ("qpos", "qvel", "time", "goal_pos", "goal_quat", "goal_local", "goal_local_vel", "action", "action_sub", "next_qpos", "next_qvel",
                        "next_goal_local", "next_goal_local_vel")}
meta, fams, t0w, tried = [], [], time.time(), 0
while len(meta) < N:
    tried += 1
    fam, path, info = sample_path(np.bincount([int(mm[0]) for mm in meta], minlength=len(FAMILIES)))
    task.goal_xy, task.goal_drift, task.random_goal = (0.0, 0.0), GoalDrift(), None
    task._goal_local = path
    sc = scen.Scenario(tried, {"block_xy": [float(v) for v in rng.uniform(xy_lo, xy_hi)]}, 0.0)
    truth.reset(scen.pose(bank, task, sc))
    driver = GoalDriver(task, 0.0)
    planner.rng = np.random.default_rng([SEED, tried]); params = planner.init_params()
    ep = {k: [] for k in keep}
    for k in range(T):
        qpos, qvel, t = truth.state()
        params, _ = planner.optimize(snapshot(driver, qpos, qvel, t), params)
        goal = driver.mocap(t, qpos, qvel)
        gl, gv = np.array(path.pos(t), np.float64), np.array(path.vel(t), np.float64)
        us = []
        for _ in range(sub):
            _, _, tn = truth.state()
            u = np.asarray(planner.get_action(params, t if HOLD else tn), np.float32); us.append(u); truth.step(u)
        nq, nv, tn = truth.state()
        for key, v in (("qpos", qpos), ("qvel", qvel), ("time", t), ("goal_pos", np.asarray(goal[0]).reshape(-1, 3)[0]),
                       ("goal_quat", np.asarray(goal[1]).reshape(-1, 4)[0]), ("goal_local", gl), ("goal_local_vel", gv),
                       ("action", np.mean(us, 0)), ("action_sub", np.stack(us)), ("next_qpos", nq), ("next_qvel", nv)):
            ep[key].append(v)
    # next-step goal = the goal recorded at the following step (the last one from the path itself)
    for k in range(T):
        if k + 1 < T:
            ep["next_goal_local"].append(ep["goal_local"][k + 1]); ep["next_goal_local_vel"].append(ep["goal_local_vel"][k + 1])
        else:
            md.qpos[:] = nq; md.qvel[:] = nv; mujoco.mj_forward(m, md); path(tn, md)
            ep["next_goal_local"].append(np.array(path.pos(tn))); ep["next_goal_local_vel"].append(np.array(path.vel(tn)))
    md.qpos[:] = nq; md.qvel[:] = nv; mujoco.mj_forward(m, md)
    if not task.task_success(md):
        print(f"  dropped attempt {tried} ({fam}): ball left the plate", flush=True); continue
    for key in keep: keep[key].append(np.array(ep[key], dtype=np.float32))
    meta.append((FAMILIES.index(fam), sc.start["block_xy"][0], sc.start["block_xy"][1], *[float(info.get(k, np.nan)) for k in ("amp", "rx", "ry", "period", "speed")]))
    if len(meta) % 10 == 0:
        print(f"{len(meta)}/{N} demos | {tried} attempts | {time.time() - t0w:.0f}s", flush=True)
np.savez_compressed(OUT, **{k: np.stack(v) for k, v in keep.items()}, demo_meta=np.array(meta, np.float32),
                    demo_meta_cols=np.array(["family", "ball_x0", "ball_y0", "amp", "rx", "ry", "period", "glide_speed"]),
                    families=np.array(FAMILIES), plan_hz=PLAN_HZ, sim_dt=task.dt)
print(f"saved {OUT}: {len(meta)} demos x {T} steps (kept {len(meta)}/{tried}) | per family: {np.bincount(np.array(meta)[:, 0].astype(int), minlength=5).tolist()}")
