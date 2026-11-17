"""Pizza-sauce spreading with the FR3 and a pusher tool (MJX), with domain randomisation of the pizza and the tool start.

Each episode samples a pizza radius R in [R_MIN, R_MAX] and a centre in the part of the workspace where the whole spiral
reaches the rim without singularities (a band around the robot base, from scripts/pizza/workspace_sweep.py), and a random
tool start above the pizza at the travel height (from the precomputed data/start_table.npz). The policy only commands
the planar tool velocity (2-D action in [-1, 1] x MAX_SPEED, through a differential IK that holds height and orientation):
  1. travel at Z_TRAVEL until the tool is within TOUCH_R of the centre
  2. the env lowers the tool to the spreading height Z_HOLD, with the planar velocity held at zero while it descends
  3. spreading: coverage (tool-swept area, the same model the real robot computes from FK + the OptiTrack pizza centre)
     only counts after touchdown
  4. standing still on the pizza for STILL_STEPS steps lifts the tool back to the travel height (the natural end)

Observation (212, raw, no hand features): ee xyz (world), pizza centre xy, pizza radius, arm joint positions (7) and
velocities (7), swept-area grid (N_RAD x N_ANG polar cells anchored at the pizza centre, scaled to the sauce disc of
radius R - CRUST, 1 = swept). The policy and critic see all of it; encoder_input_indices (everything but the arm
joints) is the embodiment-free input of the learned reward (reward_type=byol-embodiment-free).

Coverage (the mean of the grid, a metric only) and the env reward (logging only, env_reward_frac=0 in IRL) never shape
the policy directly. The pizza board in the scene XML is visual only: the tool never touches it.
"""
from copy import deepcopy
from functools import partial
from pathlib import Path

import mujoco
import numpy as np
import jax
import jax.numpy as jnp
from mujoco import mjx
from mujoco.mjx._src import support as mjx_support

from trust_region_irl.environments.pizza_dr_mjx.state import State
from trust_region_irl.environments.pizza_dr_mjx.box_space import BoxSpace
from trust_region_irl.environments.pizza_dr_mjx.viewer import MujocoViewer

try:   # Data fields MJX-warp shares across worlds (contact buffers, sizes); they must not be selected per world
    from mujoco.mjx.warp.types import DATA_NON_VMAP as _WARP_SHARED
except ImportError:
    _WARP_SHARED = set()

DATA = Path(__file__).resolve().parent / "data"
START_TABLE = DATA / "start_table.npz"


def _select_state(pred, a, b):
    """where(pred, a, b) over two States, keeping b's world-shared MJX-warp fields (identical to lax.cond for impl='jax')."""
    def sel(path, x, y):
        name = jax.tree_util.keystr(path).split(".")[-1]
        return y if name in _WARP_SHARED else jnp.where(pred, x, y)
    return jax.tree_util.tree_map_with_path(sel, a, b)


ARM_JOINT_NAMES = [f"fr3_joint{i}" for i in range(1, 8)]

# measured on the rig (config/pizza_robot.yaml)
QHOME = np.array([-0.423000, 0.625100, 0.423600, -1.912700, -0.390000, 2.456700, -1.858600])
HOME_POS = np.array([0.6247, -0.0094, 0.0641])          # board centre x/y, 20 mm above it
GOAL_QUAT_EE = np.array([0.0, 0.7071068, 0.7071068, 0.0])   # tool-down, same as pushT


def _quat_normalize(q, eps=1e-8):
    return q / (jnp.linalg.norm(q) + eps)


def _quat_conj(q):
    return jnp.array([q[0], -q[1], -q[2], -q[3]])


def _quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return jnp.array([w1*w2 - x1*x2 - y1*y2 - z1*z2,
                      w1*x2 + x1*w2 + y1*z2 - z1*y2,
                      w1*y2 - x1*z2 + y1*w2 + z1*x2,
                      w1*z2 + x1*y2 - y1*x2 + z1*w2])


def _quat_to_rotvec(qe, eps=1e-8):
    qe = _quat_normalize(qe)
    angle = 2.0 * jnp.arccos(jnp.clip(qe[0], -1.0, 1.0))
    s = jnp.sqrt(jnp.maximum(1.0 - qe[0] * qe[0], eps))
    return qe[1:] * (angle / s)


def _quat_error_body(qd, q):
    qe = _quat_mul(_quat_normalize(qd), _quat_conj(_quat_normalize(q)))
    return _quat_to_rotvec(jnp.where(qe[0] < 0.0, -qe, qe))

class PizzaSpreadDR:
    CONTROL_FREQ = 20                   # Hz, the rig's rate
    MAX_SPEED = 0.354                   # m/s planar tool speed at |action| = 1 (the demonstrations run at up to ~0.3 m/s)
    KP_ORI = 0.5                        # IK null-space pull toward qhome
    KP_Z = 5.0                          # IK height-hold gain
    OBS_CLIP = 10.0
    TOOL_R = 0.015                      # tool sphere radius, m: the swept footprint
    Z_HOLD = HOME_POS[2]                # spreading height of the tool centre, m
    TRAVEL_DZ = 0.05                    # travel height above the spreading height, m
    WANDER_MARGIN = 0.025               # default off-pizza margin, m beyond the rim (after touchdown)
    WANDER_HOLD_S = 0.5                 # default time beyond that margin before the episode is cut, s
    N_RAD, N_ANG = 8, 24                # swept-area grid: 192 polar cells

    R_MIN, R_MAX = 0.10, 0.20           # pizza radius range, m (8" to 16")
    # reachable band of pizza centres (distance from the robot base, m) at R_MIN and R_MAX, from the workspace sweep,
    # shrunk by BAND_MARGIN on both sides; linear in R in between
    BAND_AT_RMIN, BAND_AT_RMAX, BAND_MARGIN = (0.42, 0.68), (0.51, 0.60), 0.02
    MAX_ANGLE = np.deg2rad(45.0)        # pizza centre bearing from the robot base, either side of straight ahead
    TOUCH_R = 0.01                      # start the descent when the tool is this close to the centre, m
    LANDED_DZ = 0.003                   # counts as touched down within this height of Z_HOLD, m
    CRUST = 0.015                       # sauce-free rim, m: coverage is measured over the sauce disc of radius R - CRUST
    STILL_ACT, STILL_STEPS = 0.05, 10   # lift-off: commanded planar action below STILL_ACT for STILL_STEPS steps (0.5 s) while spreading

    def __init__(self, render, horizon=800, success_threshold=0.90, end_on_success=False, wander_margin=None, wander_hold_s=None,
                 impl="jax", nr_envs=64, contacts_per_env=64, ls_iterations=None):
        self.horizon = horizon
        self.success_threshold = success_threshold
        self.end_on_success = end_on_success   # False: fixed horizon, coverage is only an evaluation metric (no termination bias)
        self.wander_margin = self.WANDER_MARGIN if wander_margin is None else float(wander_margin)
        self.wander_hold_s = self.WANDER_HOLD_S if wander_hold_s is None else float(wander_hold_s)
        self.Z_TRAVEL = self.Z_HOLD + self.TRAVEL_DZ

        self.mj_model = mujoco.MjModel.from_xml_path((DATA / "scene_mjx_pizza.xml").as_posix())
        self.mj_model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        if ls_iterations is not None:
            self.mj_model.opt.ls_iterations = ls_iterations
        self.mj_data = mujoco.MjData(self.mj_model)
        self.impl = impl   # MJX backend: "jax" or "warp" (MuJoCo 3.6, see requirements.txt)
        if impl == "warp":
            self.mjx_model = mjx.put_model(self.mj_model, impl="warp")
            o = getattr(self.mjx_model.opt, "_impl", None)
            if hasattr(o, "warn_overflow"):   # MuJoCo >= 4: silence solver/linesearch iteration-limit prints, keep buffer-overflow warnings
                self.mjx_model = self.mjx_model.replace(opt=self.mjx_model.opt.replace(_impl=o.replace(warn_overflow=o.warn_overflow & ~((1 << 9) | (1 << 10)))))
            # warp keeps ONE contact pool shared by all worlds: size it for every env, else contacts are silently dropped
            import inspect
            mk = inspect.signature(mjx.make_data).parameters
            kw = {"naconmax" if "naconmax" in mk else "nconmax": contacts_per_env * nr_envs,
                  "njmax": self.mj_model.njmax if self.mj_model.njmax > 0 else 1200}   # constraint rows per world
            if "naccdmax" in mk:   # convex-collision buffer: MJX defaults it to ONE world's size, which overflows with many envs
                kw["naccdmax"] = contacts_per_env * nr_envs
            self.mjx_data = mjx.make_data(self.mj_model, impl="warp", **kw)
            # warp kernels are not differentiable: the IK manipulability gradient uses a JAX-backend copy of the model
            self._ik_model = mjx.put_model(self.mj_model)
            self._ik_data = mjx.make_data(self._ik_model)
        else:
            self.mjx_model = mjx.put_model(self.mj_model)
            self.mjx_data = mjx.make_data(self.mjx_model)
        self.nr_intermediate_steps = max(round((1.0 / self.CONTROL_FREQ) / self.mj_model.opt.timestep), 1)

        self.ee_body_id = self.mj_model.body("ee_frame").id
        arm_ids = [self.mj_model.joint(n).id for n in ARM_JOINT_NAMES]
        self.arm_qadr = jnp.array(self.mj_model.jnt_qposadr[arm_ids])
        self.arm_dofadr = jnp.array(self.mj_model.jnt_dofadr[arm_ids])
        self.joint_vel_limits = jnp.array(self.mj_model.actuator_ctrlrange[:7, 1])
        self.qhome = jnp.array(QHOME, dtype=jnp.float32)
        self.GOAL_QUAT_EE = jnp.array(GOAL_QUAT_EE, dtype=jnp.float32)
        initial_qpos = np.array(self.mj_model.qpos0)
        initial_qpos[np.array(self.mj_model.jnt_qposadr[arm_ids])] = QHOME
        self.initial_qpos = jnp.array(initial_qpos, dtype=jnp.float32)
        self.initial_qvel = jnp.zeros(self.mj_model.nv, dtype=jnp.float32)

        self.base_xy = jnp.zeros(2, jnp.float32)   # robot base at the world origin (fr3 link0)
        ir = (np.arange(self.N_RAD) + 0.5) / self.N_RAD
        ia = (np.arange(self.N_ANG) + 0.5) / self.N_ANG * 2 * np.pi - np.pi
        self.unit_cells = jnp.asarray(np.stack([np.outer(ir, np.cos(ia)), np.outer(ir, np.sin(ia))], -1).reshape(-1, 2), jnp.float32)
        if START_TABLE.exists():   # arm poses with the tool at travel height over a grid of xy (scripts/pizza/build_start_table.py)
            t = np.load(START_TABLE)
            self.start_qpos, self.start_xy = jnp.asarray(t["qpos"], jnp.float32), jnp.asarray(t["xy"], jnp.float32)
        else:
            self.start_qpos, self.start_xy = None, None

        n_cells = self.N_RAD * self.N_ANG
        obs_dim = 6 + 14 + n_cells   # ee xyz, pizza cx/cy/R, arm joint pos (7) and vel (7), swept grid
        self.encoder_input_indices = tuple(range(0, 6)) + tuple(range(20, obs_dim))   # embodiment-free: no arm joints
        self.single_action_space = BoxSpace(low=-jnp.ones(2), high=jnp.ones(2), shape=(2,), dtype=jnp.float32)
        self.single_observation_space = BoxSpace(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.single_features_shape = BoxSpace(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.viewer = MujocoViewer(self.mj_model, 1 / self.CONTROL_FREQ) if render else None
        if self.viewer is not None:
            self.light_xdir = deepcopy(self.mj_data.light_xdir)
            self.light_xpos = deepcopy(self.mj_data.light_xpos)

    # ------------------------------------------------------------ sampling #
    def sample_pizza(self, key):
        """Radius, then a centre in the reachable band for that radius (polar around the robot base)."""
        k_r, k_d, k_a = jax.random.split(key, 3)
        R = jax.random.uniform(k_r, minval=self.R_MIN, maxval=self.R_MAX)
        f = (R - self.R_MIN) / (self.R_MAX - self.R_MIN)
        d_lo = self.BAND_AT_RMIN[0] + f * (self.BAND_AT_RMAX[0] - self.BAND_AT_RMIN[0]) + self.BAND_MARGIN
        d_hi = self.BAND_AT_RMIN[1] + f * (self.BAND_AT_RMAX[1] - self.BAND_AT_RMIN[1]) - self.BAND_MARGIN
        d = jax.random.uniform(k_d, minval=d_lo, maxval=d_hi)
        a = jax.random.uniform(k_a, minval=-self.MAX_ANGLE, maxval=self.MAX_ANGLE)
        return self.base_xy + d * jnp.array([jnp.cos(a), jnp.sin(a)]), R

    def sample_start(self, key, centre, R):
        """A random start-table pose with the tool above the pizza (uniform over the table points inside the disc)."""
        inside = jnp.linalg.norm(self.start_xy - centre, axis=-1) <= R
        score = jnp.where(inside, jax.random.uniform(key, (self.start_xy.shape[0],)), -1.0)
        return self.start_qpos[jnp.argmax(score)]

    # ------------------------------------------------------------ coverage #
    def _mark(self, visited, xy, centre, R):
        """Mark every polar cell (scaled by R) whose centre lies under the tool disc, plus the cell under the tool centre."""
        q = (xy - centre) / R
        r = jnp.linalg.norm(q)
        th = jnp.arctan2(q[1], q[0])
        ir = jnp.clip((r * self.N_RAD).astype(jnp.int32), 0, self.N_RAD - 1)
        ia = jnp.clip(((th + jnp.pi) / (2 * jnp.pi) * self.N_ANG).astype(jnp.int32), 0, self.N_ANG - 1)
        visited = jnp.where(r <= 1.0, visited.at[ir * self.N_ANG + ia].set(1.0), visited)
        d = jnp.linalg.norm(self.unit_cells * R - (xy - centre), axis=-1)
        return jnp.maximum(visited, (d <= self.TOOL_R).astype(jnp.float32))

    def _obs(self, data, store):
        p = data.xpos[self.ee_body_id]
        obs = jnp.concatenate([p, store["pizza"], data.qpos[self.arm_qadr], data.qvel[self.arm_dofadr], store["visited"]])
        return jnp.clip(jnp.nan_to_num(obs, nan=0.0, posinf=self.OBS_CLIP, neginf=-self.OBS_CLIP), -self.OBS_CLIP, self.OBS_CLIP)

    def _geom(self, data, store):
        xy = data.xpos[self.ee_body_id][:2]
        centre, R = store["pizza"][:2], store["pizza"][2]
        r = jnp.linalg.norm(xy - centre)
        return xy, centre, R, r, jnp.maximum(r - R, 0.0), jnp.mean(store["visited"])

    # ------------------------------------------------------------ rollout #
    @partial(jax.vmap, in_axes=(None, 0, None))
    def reset(self, key, eval_mode):
        assert self.start_qpos is not None, f"missing {START_TABLE}; run scripts/pizza/build_start_table.py"
        obs0 = jnp.zeros(self.single_observation_space.shape, dtype=jnp.float32)
        info = {k: 0.0 for k in ("rollout/episode_return", "rollout/is_success", "rollout/task_err", "rollout/coverage",
                                 "rollout/off_pizza_frac", "rollout/diverged", "rollout/wandered_off", "rollout/touchdown_step", "rollout/lift_step", "env_info/lifted",
                                 "rollout/success_clean", "rollout/clean_finish", "rollout/lifted", "rollout/coverage_at_wander", "rollout/episode_length_all",
                                 "env_info/coverage", "env_info/radius", "env_info/off_pizza", "env_info/is_success",
                                 "env_info/task_err", "env_info/diverged", "env_info/touched_down", "env_info/pizza_R")}
        info["rollout/episode_length"] = 0
        store = self._fresh_store(jnp.zeros(3), 0)
        return self._reset(State(self.mjx_data, obs0, obs0, 0.0, False, False, info, store, key))

    def _fresh_store(self, pizza, length):
        return {"episode_return": 0.0, "episode_length": length, "visited": jnp.zeros(self.N_RAD * self.N_ANG, jnp.float32),
                "pizza": jnp.asarray(pizza, jnp.float32), "phase": 0.0, "touchdown_step": 0.0, "off_steps": 0.0, "wander_steps": 0.0,
                "still_steps": 0.0, "lift_step": 0.0}

    @partial(jax.jit, static_argnums=(0,))
    def _reset(self, state):
        key, k_p, k_s = jax.random.split(state.key, 3)
        centre, R = self.sample_pizza(k_p)
        qpos = self.initial_qpos.at[self.arm_qadr].set(self.sample_start(k_s, centre, R))
        data = mjx.forward(self.mjx_model, self.mjx_data.replace(qpos=qpos, qvel=self.initial_qvel, ctrl=jnp.zeros(self.mjx_model.nu)))
        store = self._fresh_store(jnp.concatenate([centre, R[None]]), 0)
        info = dict(state.info)
        info.update({"env_info/coverage": 0.0, "env_info/is_success": 0.0, "env_info/diverged": 0.0, "env_info/touched_down": 0.0,
                     "env_info/pizza_R": R, "env_info/task_err": 1.0})
        obs = self._obs(data, store)
        return state.replace(key=key, data=data, next_observation=obs, actual_next_observation=obs, reward=0.0,
                             terminated=False, truncated=False, info=info, info_episode_store=store)

    @partial(jax.vmap, in_axes=(None, 0, 0))
    @partial(jax.jit, static_argnums=(0,))
    def step(self, state, action):
        return self._step(state, action)

    @partial(jax.jit, static_argnums=(0,))
    def _step(self, state, action):
        store = state.info_episode_store
        phase = store["phase"]   # 0 travel, 1 descending or spreading, 2 lifted off (finished)
        ee = state.data.xpos[self.ee_body_id]
        spreading_phase = (phase > 0.5) & (phase < 1.5)
        landed = spreading_phase & (ee[2] - self.Z_HOLD < self.LANDED_DZ)
        frozen = (spreading_phase & ~landed) | (phase > 1.5)   # xy held while descending and after lift-off
        vel_xy = jnp.clip(action, -1.0, 1.0) * self.MAX_SPEED * jnp.where(frozen, 0.0, 1.0)
        z_target = jnp.where(spreading_phase, self.Z_HOLD, self.Z_TRAVEL)

        def substep(data, _):
            dq = self._differential_ik(data, vel_xy, z_target)
            return mjx.step(self.mjx_model, data.replace(ctrl=dq)), None

        data, _ = jax.lax.scan(substep, state.data, xs=(), length=self.nr_intermediate_steps)
        obs_data = mjx.kinematics(self.mjx_model, data)

        store["episode_length"] += 1
        xy, centre, R, r, off_pizza, _ = self._geom(obs_data, store)
        start_descent = (phase < 0.5) & (r < self.TOUCH_R)
        store["touchdown_step"] = jnp.where(start_descent, store["episode_length"].astype(jnp.float32), store["touchdown_step"])
        store["phase"] = jnp.where(start_descent, 1.0, phase)
        landed = (store["phase"] > 0.5) & (store["phase"] < 1.5) & (obs_data.xpos[self.ee_body_id][2] - self.Z_HOLD < self.LANDED_DZ)
        store["visited"] = jnp.where(landed, self._mark(store["visited"], xy, centre, R - self.CRUST), store["visited"])
        # natural end: standing still on the pizza for STILL_STEPS steps lifts the tool off (scripted, like the touchdown)
        still = landed & (jnp.linalg.norm(jnp.clip(action, -1.0, 1.0)) < self.STILL_ACT)
        store["still_steps"] = jnp.where(still, store["still_steps"] + 1, 0.0)
        lift = landed & (store["still_steps"] >= self.STILL_STEPS)
        store["lift_step"] = jnp.where(lift, store["episode_length"].astype(jnp.float32), store["lift_step"])
        store["phase"] = jnp.where(lift, 2.0, store["phase"])
        coverage = jnp.mean(store["visited"])
        obs = self._obs(obs_data, store)

        is_success = (coverage >= self.success_threshold).astype(jnp.float32)
        reward = jnp.nan_to_num(10.0 * coverage - 5.0 * off_pizza * landed - 0.001 * jnp.sum(jnp.square(action)), nan=0.0, posinf=0.0, neginf=-10.0)
        terminated = (is_success > 0.5) & self.end_on_success
        diverged = jnp.any(jnp.isnan(data.qpos)) | jnp.any(jnp.isnan(data.qvel))
        beyond = landed & (off_pizza > self.wander_margin)   # wandering off only counts once spreading
        store["wander_steps"] = jnp.where(beyond, store["wander_steps"] + 1, 0.0)
        wandered_off = store["wander_steps"] >= self.wander_hold_s * self.CONTROL_FREQ
        store["off_steps"] += (landed & (off_pizza > 0.0)).astype(jnp.float32)
        at_horizon = store["episode_length"] >= self.horizon
        truncated = at_horizon | diverged | wandered_off
        done = terminated | truncated

        state.info.update({"env_info/coverage": coverage, "env_info/radius": r / R, "env_info/off_pizza": off_pizza,
                           "env_info/is_success": is_success, "env_info/task_err": 1.0 - coverage,
                           "env_info/diverged": diverged.astype(jnp.float32), "env_info/touched_down": landed.astype(jnp.float32),
                           "env_info/pizza_R": R, "env_info/lifted": (store["phase"] > 1.5).astype(jnp.float32)})
        store["episode_return"] += reward
        log_done = terminated | at_horizon
        for k, v in (("rollout/episode_return", store["episode_return"]), ("rollout/episode_length", store["episode_length"])):
            state.info[k] = jnp.where(log_done, v, state.info[k])
        for k, v in (("rollout/is_success", is_success), ("rollout/task_err", 1.0 - coverage),
                     ("rollout/coverage", coverage), ("rollout/diverged", diverged.astype(jnp.float32)),
                     ("rollout/wandered_off", wandered_off.astype(jnp.float32)), ("rollout/touchdown_step", store["touchdown_step"]), ("rollout/lift_step", store["lift_step"]),
                     ("rollout/off_pizza_frac", store["off_steps"] / jnp.maximum(store["episode_length"], 1)),
                     # honest finish metrics: success not ending by wandering off, the expert's ending (lifted with the coverage target), lengths of all episodes
                     ("rollout/success_clean", is_success * (1.0 - wandered_off.astype(jnp.float32))),
                     ("rollout/clean_finish", is_success * (store["phase"] > 1.5).astype(jnp.float32)),
                     ("rollout/lifted", (store["phase"] > 1.5).astype(jnp.float32)),
                     ("rollout/episode_length_all", store["episode_length"].astype(jnp.float32))):
            state.info[k] = jnp.where(done, v, state.info[k])
        state.info["rollout/coverage_at_wander"] = jnp.where(done & wandered_off, coverage, state.info["rollout/coverage_at_wander"])

        def when_done(_):
            s = self._reset(state)
            return s.replace(actual_next_observation=obs, reward=reward, terminated=terminated, truncated=truncated)

        def when_not_done(_):
            return state.replace(data=data, next_observation=obs, actual_next_observation=obs, reward=reward,
                                 terminated=terminated, truncated=truncated)

        return _select_state(done, when_done(None), when_not_done(None))

    # ------------------------------------------------------------ kinematics #
    def _differential_ik(self, data, vel_xy, z_target):
        """Planar tool velocity -> arm joint velocities, robust near singularities (the PushT IK).

        Damped least squares with damping that grows as the smallest singular value shrinks, a null-space pull to
        qhome plus a manipulability gradient that steers away from singular poses, and one common scale factor for
        the joint-velocity cap so the tool direction is kept. Height and orientation are held by servo terms."""
        SR_EPS = 1e-2  # singular-value threshold where damping kicks in
        SR_LAMBDA_MAX = 0.1  # max damping
        K_MANIP = 1.0  # null-space manipulability gain

        point = data.xpos[self.ee_body_id]
        jacp, jacr = mjx_support.jac(self.mjx_model, data, point, self.ee_body_id)
        J = jnp.concatenate([jacp.T, jacr.T], axis=0)[:, self.arm_dofadr]
        sigma_min = jnp.linalg.svd(J, compute_uv=False)[-1]
        lam2 = jnp.where(sigma_min < SR_EPS, (1.0 - (sigma_min / SR_EPS) ** 2) * SR_LAMBDA_MAX ** 2, 0.0)
        J_dls = J.T @ jnp.linalg.solve(J @ J.T + lam2 * jnp.eye(6), jnp.eye(6))  # (7, 6) damped least-squares inverse

        ee_pos = data.xpos[self.ee_body_id]
        goal_vec = _quat_error_body(self.GOAL_QUAT_EE, data.xquat[self.ee_body_id])
        twist = jnp.concatenate([vel_xy, jnp.array([self.KP_Z * (z_target - ee_pos[2])]), goal_vec])   # z_target: held tool height (travel or spreading)
        dq = J_dls @ twist

        qnow = data.qpos[self.arm_qadr]
        N = jnp.eye(J.shape[1]) - J_dls @ J
        dq = dq + N @ (self.KP_ORI * (self.qhome - qnow))

        # warp: differentiate through the JAX-backend copy (kinematics depends only on qpos, so the gradient is identical)
        m_ik, d_ik = (self._ik_model, self._ik_data.replace(qpos=data.qpos)) if self.impl == "warp" else (self.mjx_model, data)

        def _manip(q):  # Yoshikawa manipulability as a function of the arm joints, for autodiff
            d = d_ik.replace(qpos=d_ik.qpos.at[self.arm_qadr].set(q))
            d = mjx.kinematics(m_ik, d)
            d = mjx.com_pos(m_ik, d)
            jp, jr = mjx_support.jac(m_ik, d, d.xpos[self.ee_body_id], self.ee_body_id)
            Jq = jnp.concatenate([jp.T, jr.T], axis=0)[:, self.arm_dofadr]
            return jnp.sqrt(jnp.linalg.det(Jq @ Jq.T) + 1e-12)

        dq = dq + N @ (K_MANIP * jax.grad(_manip)(qnow))
        dq = jnp.nan_to_num(dq, nan=0.0, posinf=0.0, neginf=0.0)
        scale = jnp.minimum(1.0, jnp.min(self.joint_vel_limits / (jnp.abs(dq) + 1e-9)))   # one factor keeps the ee direction under the cap
        return dq * scale

    # ------------------------------------------------------------ misc #
    def feature_from_transition(self, observation, action, eps=1e-4):
        """Raw observation (no hand features); kept for the algorithms that ask for features."""
        return observation

    def render(self, state):
        data = mjx.get_data(self.mj_model, state.data)[0]
        data.light_xdir = self.light_xdir
        data.light_xpos = self.light_xpos
        self.viewer.render(data)
        return state

    def close(self):
        if self.viewer:
            self.viewer.close()
