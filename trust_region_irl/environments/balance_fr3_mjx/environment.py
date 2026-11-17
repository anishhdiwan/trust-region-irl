"""Ball balancing with the FR3 (BaMPC's Balance-FR3), as an MJX env for TRIRL.

The FR3 holds a plate on its end effector and tilts it to roll a ball (radius 4 cm) to a goal that rides the plate. The
model is BaMPC's own (exported XML + solver options, identical dynamics to their MuJoCo 3.10 model, checked step for
step), and the action goes through an exact JAX port of their task-space IK (BalanceFr3.control_map_host).

Each episode draws a goal path on the plate, the same families and ranges as the BaMPC demos: still, line, ellipse,
Lissajous (centre + r sin(2 pi f t + phi), peak speed <= 6 cm/s) or ramped waypoints (the goal glides at 2-4 cm/s to a
random point, a new one once it arrived and the ball is within 3 cm).

Action: plate roll/pitch rate in [-1, 1]^2 (x MAX_TILT_VEL rad/s), held for the 5 sim sub-steps (10 ms) of each 50 ms step.
Observation (39): task part (25, world frame) = ball position, linear and angular velocity; plate position, quaternion and
angular velocity; goal position and velocity; then the arm joint positions (7) and velocities (7) for the policy and
critic only (the embodiment-free encoder uses encoder_input_indices = 0..24). Fixed horizon, no early termination (only
a NaN reset); ball loss and goal distance are metrics.
"""
import json
from functools import partial
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from mujoco.mjx._src import support as mjx_support

from trust_region_irl.environments.balance_fr3_mjx.state import State
from trust_region_irl.environments.balance_fr3_mjx.box_space import BoxSpace

DATA = Path(__file__).resolve().parent / "data"

try:   # Data fields MJX-warp shares across worlds (contact buffers, sizes); they must not be selected per world
    from mujoco.mjx.warp.types import DATA_NON_VMAP as _WARP_SHARED
except ImportError:
    _WARP_SHARED = set()


def _select_state(pred, a, b):
    """where(pred, a, b) over two States, keeping b's world-shared MJX-warp fields (identical to lax.cond for impl='jax')."""
    def sel(path, x, y):
        name = jax.tree_util.keystr(path).split(".")[-1]
        return y if name in _WARP_SHARED else jnp.where(pred, x, y)
    return jax.tree_util.tree_map_with_path(sel, a, b)


def _quat_mul(a, b):
    return jnp.array([a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
                      a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
                      a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
                      a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]])


def _sub_quat(qa, qb):
    """mju_subQuat: the 3-D rotation r with qb * quat(r) = qa (angle wrapped to [-pi, pi])."""
    d = _quat_mul(qb * jnp.array([1.0, -1.0, -1.0, -1.0]), qa)
    v = d[1:4]; s = jnp.linalg.norm(v)
    angle = 2.0 * jnp.arctan2(s, d[0])
    angle = jnp.where(angle > jnp.pi, angle - 2.0 * jnp.pi, angle)
    return jnp.where(s > 1e-12, v / jnp.maximum(s, 1e-12) * angle, jnp.zeros(3))


def _rotate(q, v):
    """Rotate v by unit quaternion q (w, x, y, z), as BaMPC's quat_rotate."""
    qv = q[1:4]; t = 2.0 * jnp.cross(qv, v)
    return v + q[0] * t + jnp.cross(qv, t)


class BalanceFR3:
    CONTROL_FREQ = 20
    OBS_CLIP = 10.0
    BALL_XY_BOX = ((-0.09, -0.092), (0.094, 0.099))   # plate-local ball start range, as BaMPC's scenario bank and the demos

    def __init__(self, render=False, horizon=200, impl="jax", nr_envs=64, contacts_per_env=64):
        self.horizon = horizon
        self.impl = impl
        # (the velocity tells which way the goal is heading, so two demos passing the same point in opposite directions are not confused)
        p = json.load(open(DATA / "balance_fr3_task_params.json"))
        m = mujoco.MjModel.from_xml_path(str(DATA / "balance_fr3_model.xml"))
        for k, v in json.load(open(DATA / "balance_fr3_opt.json")).items():   # solver options BaMPC writes after compiling
            try:
                setattr(m.opt, k, type(getattr(m.opt, k))(v) if np.ndim(v) == 0 else np.asarray(v))
            except Exception:
                pass
        inert = np.load(DATA / "balance_fr3_link0_inertial.npz")
        b0 = m.body("fr3_link0").id   # fixed base: XML recomputes its inertial, set BaMPC's (no effect on dynamics)
        for f in ("body_mass", "body_inertia", "body_ipos", "body_iquat"):
            getattr(m, f)[b0] = inert[f][b0]
        self.mj_model = m
        if impl == "warp":
            self.mjx_model = mjx.put_model(m, impl="warp")
            import inspect
            mk = inspect.signature(mjx.make_data).parameters
            kw = {"naconmax" if "naconmax" in mk else "nconmax": contacts_per_env * nr_envs, "njmax": m.njmax if m.njmax > 0 else 1200}
            if "naccdmax" in mk:   # convex-collision buffer: MJX defaults it to one world's size, which overflows with many envs
                kw["naccdmax"] = contacts_per_env * nr_envs
            self.mjx_data = mjx.make_data(m, impl="warp", **kw)
        else:
            self.mjx_model = mjx.put_model(m)
            self.mjx_data = mjx.make_data(self.mjx_model)
        self.nr_intermediate_steps = max(round((1.0 / self.CONTROL_FREQ) / m.opt.timestep), 1)

        self.ee_body_id = int(p["ee_body_id"]); self.plate_id = int(p["plate_body_id"]); self.ball_id = int(p["ball_body_id"])
        self.arm_dofadr = jnp.asarray(p["arm_dofadr"]); self.arm_qposadr = jnp.asarray(p["arm_qposadr"])
        self.k_pos, self.k_yaw, self.ik_damping = float(p["k_pos"]), float(p["k_yaw"]), float(p["ik_damping"])
        self.max_tilt_vel = float(p.get("max_tilt_vel", 1.0))
        self.ee_home_pos = jnp.asarray(p["ee_home_pos"], jnp.float32); self.ee_home_quat = jnp.asarray(p["ee_home_quat"], jnp.float32)
        self.rest_z = float(p["ball_local_z"]); self.lost_margin = float(p["lost_margin"])
        self.home_qpos = jnp.asarray(p["home_qpos"], jnp.float32)
        d0 = mujoco.MjData(m); d0.qpos[:] = p["home_qpos"]; mujoco.mj_forward(m, d0)
        self.plate_pos0 = jnp.asarray(d0.xpos[self.plate_id], jnp.float32); self.plate_R0 = jnp.asarray(d0.xmat[self.plate_id].reshape(3, 3), jnp.float32)
        self.ball_qadr = int(m.jnt_qposadr[0]); self.ball_dadr = int(m.jnt_dofadr[0])

        self.single_action_space = BoxSpace(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.single_observation_space = BoxSpace(low=-np.inf, high=np.inf, shape=(39,), dtype=np.float32)
        self.single_features_shape = BoxSpace(low=-np.inf, high=np.inf, shape=(39,), dtype=np.float32)
        self.encoder_input_indices = tuple(range(0, 25))   # embodiment-free: ball, plate, goal position and velocity; not the arm joints (25:39)
        self.viewer = None

    # ------------------------------------------------------------ control #
    def control_map(self, data, u):
        """BaMPC's task-space IK (BalanceFr3.control_map_host): 2-D plate tilt rate (rad/s) -> 7 joint velocities."""
        jacp, jacr = mjx_support.jac(self.mjx_model, data, data.xpos[self.ee_body_id], self.ee_body_id)
        J = jnp.concatenate([jacp.T, jacr.T], axis=0)[:, self.arm_dofadr]   # (6, 7)
        ee_pos, ee_quat = data.xpos[self.ee_body_id], data.xquat[self.ee_body_id]
        e_world = _rotate(ee_quat, _sub_quat(self.ee_home_quat, ee_quat))
        twist = jnp.concatenate([self.k_pos * (self.ee_home_pos - ee_pos), u, jnp.array([self.k_yaw * e_world[2]])])
        return J.T @ jnp.linalg.solve(J @ J.T + self.ik_damping * jnp.eye(6), twist)

    def physics_step(self, data, u_sub):
        """Apply one tilt command per sim sub-step (u_sub: (nsub, 2) in rad/s) through the IK."""
        def sub(d, u):
            return mjx.step(self.mjx_model, d.replace(ctrl=self.control_map(d, u))), None
        data, _ = jax.lax.scan(sub, data, u_sub)
        return data

    # ------------------------------------------------------------ goal paths #
    # Per-episode goal path on the plate, the same families and ranges as the BaMPC demos (scripts/balance/bampc/collect_demos.py):
    # 0 still, 1 line, 2 ellipse, 3 lissajous (all: centre + r * sin(2 pi f t + phi) per plate axis), 4 ramped waypoints.
    SAFE = (0.195 - 0.05, 0.25 - 0.05)   # plate half-extents minus BaMPC's 5 cm keep-out
    VMAX = 0.06                          # moving paths above this peak speed are re-drawn (the planner tracks well below it)
    WP_THR = 0.03                        # waypoints: a new point once the goal arrived and the ball is within this

    def _drift_candidates(self, key, fam, K=16):
        """K candidate (centre, r, f, phi) for drift family `fam`; returns the first under the speed cap."""
        S = jnp.array(self.SAFE)
        ks = jax.random.split(key, 8)
        U = lambda k, lo, hi, shape=(K,): jax.random.uniform(k, shape, minval=lo, maxval=hi)
        # line
        A, th, per_l, ph_l = U(ks[0], 0.03, 0.08), U(ks[1], 0.0, jnp.pi), U(ks[2], 12.0, 25.0), U(ks[3], 0.0, 2 * jnp.pi)
        r_line = jnp.stack([A * jnp.cos(th), A * jnp.sin(th)], -1)
        # ellipse
        r_ell = U(ks[4], 0.03, 0.08, (K, 2)); per_e = U(ks[2], 12.0, 25.0); dirn = jnp.where(U(ks[5], 0.0, 1.0) < 0.5, -1.0, 1.0)
        # lissajous
        r_lis = U(ks[6], 0.03, 0.06, (K, 2)); per_s = U(ks[2], 16.0, 30.0); swap = U(ks[7], 0.0, 1.0) < 0.5
        r = jnp.select([fam == 1, fam == 2, fam == 3], [r_line, r_ell, r_lis], jnp.zeros((K, 2)))
        per = jnp.select([fam == 1, fam == 2], [per_l, per_e], per_s)
        f1 = jnp.stack([1.0 / per, 1.0 / per], -1)
        f_lis = jnp.where(swap[:, None], jnp.stack([2.0 / per, 1.0 / per], -1), jnp.stack([1.0 / per, 2.0 / per], -1))
        f = jnp.where(fam == 3, f_lis, f1)
        ph2 = U(ks[3], 0.0, 2 * jnp.pi, (K, 2))
        phi = jnp.select([fam == 1, fam == 2], [jnp.stack([ph_l, ph_l], -1), jnp.stack([ph_l, ph_l + dirn * jnp.pi / 2], -1)], ph2)
        span = S - jnp.abs(r) - jnp.where(fam == 0, 0.02, 0.0)
        c = jax.random.uniform(ks[0], (K, 2), minval=-1.0, maxval=1.0) * span
        ts = jnp.linspace(0.0, 2.0 / jnp.min(f, axis=-1, keepdims=True), 64, axis=-1)   # (K, 64)
        v = r[:, None, :] * 2 * jnp.pi * f[:, None, :] * jnp.cos(2 * jnp.pi * f[:, None, :] * ts[..., None] + phi[:, None, :])
        ok = jnp.max(jnp.linalg.norm(v, axis=-1), axis=-1) <= self.VMAX
        i = jnp.argmax(ok)
        return jnp.concatenate([c[i], r[i], f[i], phi[i]])

    def _wp_target(self, key, pos, K=20):
        """A random on-plate waypoint at least 2 * WP_THR away from `pos` (first of K candidates that is)."""
        S = jnp.array(self.SAFE)
        cand = jax.random.uniform(key, (K, 2), minval=-S, maxval=S)
        ok = jnp.linalg.norm(cand - pos, axis=-1) >= 2 * self.WP_THR
        return cand[jnp.where(jnp.any(ok), jnp.argmax(ok), K - 1)]

    def _sample_goal(self, key):
        """Episode goal state: family, drift params (8), waypoint position/target/velocity/speed."""
        kf, kd, kp, kt, ks = jax.random.split(key, 5)
        fam = jax.random.randint(kf, (), 0, 5)
        path = self._drift_candidates(kd, fam)
        S = jnp.array(self.SAFE)
        wp_pos = jax.random.uniform(kp, (2,), minval=-S, maxval=S)
        return {"fam": fam, "path": path, "wp_pos": wp_pos, "wp_tgt": self._wp_target(kt, wp_pos), "wp_vel": jnp.zeros(2),
                "wp_speed": jax.random.uniform(ks, minval=0.02, maxval=0.04)}

    def goal_local(self, g, t):
        """Plate-local goal position and velocity (2-D each) at episode time t."""
        c, r, f, phi = g["path"][0:2], g["path"][2:4], g["path"][4:6], g["path"][6:8]
        arg = 2 * jnp.pi * f * t + phi
        dp, dv = c + r * jnp.sin(arg), r * 2 * jnp.pi * f * jnp.cos(arg)
        wp = g["fam"] == 4
        return jnp.where(wp, g["wp_pos"], dp), jnp.where(wp, g["wp_vel"], dv)

    def _advance_waypoint(self, g, ball_local_xy, key):
        """Ramped waypoint update for one control step, as RampedWaypoints in the demo collection."""
        p, tgt, sp = g["wp_pos"], g["wp_tgt"], g["wp_speed"]
        arrived = jnp.linalg.norm(p - tgt) < 1e-4
        tgt = jnp.where(arrived & (jnp.linalg.norm(ball_local_xy - p) < self.WP_THR), self._wp_target(key, p), tgt)
        d = tgt - p; n = jnp.linalg.norm(d); step = jnp.minimum(n, sp / self.CONTROL_FREQ)
        u = d / jnp.maximum(n, 1e-9)
        moving = n > 1e-9
        g = dict(g)
        g["wp_vel"] = jnp.where(moving & (step < n), u * sp, jnp.zeros(2))
        g["wp_pos"] = p + jnp.where(moving, u * step, jnp.zeros(2))
        g["wp_tgt"] = tgt
        return g

    # ------------------------------------------------------------ geometry #
    def _plate(self, data):
        return data.xpos[self.plate_id], data.xmat[self.plate_id].reshape(3, 3)

    def obs_from(self, data, gl, gv):
        """Observation from a state and the plate-local goal position/velocity (used by the env and the dataset builder)."""
        pp, R = self._plate(data)
        goal_w = pp + R @ jnp.array([gl[0], gl[1], self.rest_z])
        goal_v = R @ jnp.array([gv[0], gv[1], 0.0])
        ball_v = data.qvel[self.ball_dadr:self.ball_dadr + 6]   # free joint: linear (world), angular
        plate_w = data.cvel[self.plate_id][0:3]                 # angular velocity (rotational part of the com-based spatial velocity)
        o = jnp.concatenate([data.xpos[self.ball_id], ball_v, pp, data.xquat[self.plate_id], plate_w, goal_w, goal_v,
                             data.qpos[self.arm_qposadr], data.qvel[self.arm_dofadr]])
        return jnp.clip(jnp.nan_to_num(o, nan=0.0, posinf=self.OBS_CLIP, neginf=-self.OBS_CLIP), -self.OBS_CLIP, self.OBS_CLIP)

    def _metrics(self, data, gl):
        pp, R = self._plate(data)
        goal = pp + R @ jnp.array([gl[0], gl[1], self.rest_z])
        dist = jnp.linalg.norm(data.xpos[self.ball_id][:2] - goal[:2])
        local = R.T @ (data.xpos[self.ball_id] - pp)
        return dist, local[2] < self.rest_z - self.lost_margin, local[:2]

    # ------------------------------------------------------------ rollout #
    def _fresh_store(self, g):
        return {"episode_return": 0.0, "episode_length": 0, "lost": 0.0, "dist_sum": 0.0, "dist_sum_2nd": 0.0, **g}

    @partial(jax.vmap, in_axes=(None, 0, None))
    def reset(self, key, eval_mode):
        obs0 = jnp.zeros(self.single_observation_space.shape, dtype=jnp.float32)
        info = {k: 0.0 for k in ("rollout/episode_return", "rollout/goal_dist", "rollout/goal_dist_2nd_half", "rollout/ball_lost",
                                 "rollout/diverged", "env_info/goal_dist", "env_info/ball_lost", "env_info/diverged",
                                 "rollout/goal_family", "env_info/goal_speed")}
        info["rollout/episode_length"] = 0
        store = self._fresh_store(self._sample_goal(key))
        return self._reset(State(self.mjx_data, obs0, obs0, 0.0, False, False, info, store, key))

    @partial(jax.jit, static_argnums=(0,))
    def _reset(self, state):
        key, k_xy, k_g = jax.random.split(state.key, 3)
        lo, hi = jnp.array(self.BALL_XY_BOX[0]), jnp.array(self.BALL_XY_BOX[1])
        xy = jax.random.uniform(k_xy, (2,), minval=lo, maxval=hi)
        ball = self.plate_pos0 + self.plate_R0 @ jnp.array([xy[0], xy[1], self.rest_z])   # on the level plate at home, as BalanceFr3.set_initial_state
        qpos = self.home_qpos.at[self.ball_qadr:self.ball_qadr + 3].set(ball).at[self.ball_qadr + 3:self.ball_qadr + 7].set(jnp.array([1.0, 0, 0, 0]))
        data = self.mjx_data.replace(qpos=qpos, qvel=jnp.zeros(self.mj_model.nv), ctrl=jnp.zeros(self.mj_model.nu), time=jnp.asarray(0.0, self.mjx_data.time.dtype))
        data = mjx.forward(self.mjx_model, data)
        g = self._sample_goal(k_g)
        gl, gv = self.goal_local(g, 0.0)
        obs = self.obs_from(data, gl, gv)
        store = self._fresh_store(g)
        return state.replace(key=key, data=data, next_observation=obs, actual_next_observation=obs, reward=0.0, terminated=False,
                             truncated=False, info=dict(state.info), info_episode_store=store)

    @partial(jax.vmap, in_axes=(None, 0, 0))
    @partial(jax.jit, static_argnums=(0,))
    def step(self, state, action):
        return self._step(state, action)

    @partial(jax.jit, static_argnums=(0,))
    def _step(self, state, action):
        u = jnp.clip(action, -1.0, 1.0) * self.max_tilt_vel
        data = self.physics_step(state.data, jnp.tile(u[None], (self.nr_intermediate_steps, 1)))
        obs_data = mjx.kinematics(self.mjx_model, data)
        store = dict(state.info_episode_store)
        key, k_wp = jax.random.split(state.key)
        store["episode_length"] += 1
        t = store["episode_length"] / self.CONTROL_FREQ
        _, _, ball_local = self._metrics(obs_data, store["wp_pos"])
        adv = self._advance_waypoint(store, ball_local, k_wp)          # ramped waypoints move with the ball, as in the demos
        for k in ("wp_pos", "wp_tgt", "wp_vel"):
            store[k] = jnp.where(store["fam"] == 4, adv[k], store[k])
        gl, gv = self.goal_local(store, t)
        dist, lost, _ = self._metrics(obs_data, gl)
        store["lost"] = jnp.maximum(store["lost"], lost.astype(jnp.float32))
        store["dist_sum"] += dist
        store["dist_sum_2nd"] += jnp.where(store["episode_length"] > self.horizon // 2, dist, 0.0)
        reward = -dist   # logging only (env_reward_frac = 0 in IRL)
        diverged = jnp.any(jnp.isnan(data.qpos)) | jnp.any(jnp.isnan(data.qvel))
        at_horizon = store["episode_length"] >= self.horizon
        truncated = at_horizon | diverged
        done = truncated
        obs = self.obs_from(obs_data, gl, gv)
        store["episode_return"] += reward
        state.info.update({"env_info/goal_dist": dist, "env_info/ball_lost": lost.astype(jnp.float32), "env_info/diverged": diverged.astype(jnp.float32),
                           "env_info/goal_speed": jnp.linalg.norm(gv)})
        n = jnp.maximum(store["episode_length"], 1)
        for k, v in (("rollout/episode_return", store["episode_return"]), ("rollout/episode_length", store["episode_length"]),
                     ("rollout/goal_dist", store["dist_sum"] / n), ("rollout/goal_dist_2nd_half", store["dist_sum_2nd"] / jnp.maximum(n - self.horizon // 2, 1)),
                     ("rollout/ball_lost", store["lost"]), ("rollout/diverged", diverged.astype(jnp.float32)),
                     ("rollout/goal_family", store["fam"].astype(jnp.float32))):
            state.info[k] = jnp.where(done, v, state.info[k])
        state = state.replace(key=key, info_episode_store=store)

        def when_done(_):
            s = self._reset(state)
            return s.replace(actual_next_observation=obs, reward=reward, terminated=False, truncated=truncated)

        def when_not_done(_):
            return state.replace(data=data, next_observation=obs, actual_next_observation=obs, reward=reward, terminated=False, truncated=truncated)

        return _select_state(done, when_done(None), when_not_done(None))

    def feature_from_transition(self, observation, action, eps=1e-4):
        """Raw observation (no hand features), for the algorithms that ask for features."""
        return observation

    def render(self, state):
        return state

    def close(self):
        pass
