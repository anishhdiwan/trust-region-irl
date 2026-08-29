"""PushT FR3 environment (MJX) for TRIRL.

Mirrors the expert-data generator (mtp-thesis/scripts/generate_pusht_expert.py)
so a learned policy's observations/actions match the expert distribution:

  * Fixed start: block at (0.6, -0.1, 90deg); fixed goal (model mocap body).
  * 2-D action in [-1, 1] -> *MAX_SPEED -> differential IK -> 7 joint-velocity
    ctrl. The IK pins the ee height (z) and orientation, so the pusher only
    moves in the horizontal plane (planar, fixed-height) -- matching the data.
  * 24-dim observation: block_pos_rel_goal(3) + block_quat_rel_goal(4)
    + ee_pos_rel_goal(3) + arm_qpos(7) + arm_qvel(7).

Control runs at 20 Hz; the model timestep is 0.025 s -> 2 sim substeps per
control step, and the IK is recomputed each substep (as in the generator).
"""
from copy import deepcopy
from pathlib import Path
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import mujoco.mjx._src.support as mjx_support
from scipy.spatial.transform import Rotation as R

from trust_region_irl.environments.pusht_mjx.state import State
from trust_region_irl.environments.pusht_mjx.box_space import BoxSpace
from trust_region_irl.environments.pusht_mjx.viewer import MujocoViewer


ARM_JOINT_NAMES = [f"fr3_joint{i}" for i in range(1, 8)]
QHOME = np.array([0.51199203, 0.1014329, -0.36340348, -2.9813132,
                  0.50339095, 3.06692214, -1.92271156])


# ----------------------------- quat helpers (port of hydrax.utils.utils) ---- #
def _quat_normalize(q, eps=1e-8):
    n = jnp.linalg.norm(q)
    return jnp.where(n < eps, jnp.array([1.0, 0.0, 0.0, 0.0]), q / jnp.maximum(n, eps))


def _quat_conj(q):
    w, x, y, z = q
    return jnp.array([w, -x, -y, -z])


def _quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return jnp.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _quat_to_rotvec(qe, eps=1e-8):
    w, x, y, z = qe
    w = jnp.clip(w, -1.0, 1.0)
    angle = 2.0 * jnp.arccos(w)
    s = jnp.sqrt(jnp.maximum(1.0 - w * w, 0.0))
    s_safe = jnp.where(s < eps, 1.0, s)  # avoid the untaken-branch x/0 NaN (autodiff-safe)
    axis = jnp.where(s < eps, jnp.array([1.0, 0.0, 0.0]), jnp.array([x, y, z]) / s_safe)
    return angle * axis


def _quat_error_body(qd, q):
    """Right-invariant error q_e = qd * q^{-1} as a rotation vector."""
    qd = _quat_normalize(qd)
    q = _quat_normalize(q)
    qe = _quat_mul(qd, _quat_conj(q))
    return _quat_to_rotvec(jnp.where(qe[0] < 0.0, -qe, qe))


class PushT:
    # fixed scene (examples/pusht_franka_free.py SEED 40 / the data generator)
    BLOCK_POS = (0.55, 0.0) # (0.6, -0.1)
    BLOCK_ANGLE = 0 # np.pi / 2
    GOAL_POS_EE = np.array([0.45, 0.1, 0.035])
    GOAL_QUAT_EE = jnp.array([0.0, 0.7071, 0.7071, 0.0])  # wxyz
    MAX_SPEED = 0.35
    CONTROL_FREQ = 20
    KP_ORI = 10.0
    Z_HOLD = 0.045  # ee height the IK servos toward
    # --- numerical-stability guards (prevent MJX divergence at large nr_envs) ---
    OBS_CLIP = 10.0       # clip observation magnitudes so a diverged env can't
                          # feed huge values into the discriminator/critic
                          # (expert obs absmax is 3.33, ~3x headroom)
    ERR_CLIP = 5.0        # cap pos/orient error (keeps reward finite/bounded)
    DIVERGE_BOUND = 1.0   # block farther than this (m) from goal -> end episode
                          # (~0.57 m table; 1.0 catches off-table blocks early)

    # Singularity Threshold to truncate if the eigenvalue is lesser
    SINGULARITY_THRESHOLD = 0.05

    def __init__(self, render, horizon=250, reward_style="dense",
                 success_threshold=0.05, feature_fn="base"):
        self.horizon = horizon
        self.reward_style = reward_style
        self.success_threshold = success_threshold
        self.feature_fn = feature_fn

        xml_path = (Path(__file__).resolve().parent / "data" / "scene_mjx_free.xml").as_posix()
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mjx_model = mjx.put_model(self.mj_model)
        self.mjx_data = mjx.make_data(self.mjx_model)

        # 20 Hz control; model timestep 0.025 s -> 2 sim steps per control step
        self.nr_intermediate_steps = max(
            round((1.0 / self.CONTROL_FREQ) / self.mj_model.opt.timestep), 1)

        # indices
        self.ee_body_id = self.mj_model.body("ee_frame").id
        self.block_body_id = self.mj_model.body("block").id
        self.goal_body_id = self.mj_model.body("goal").id
        arm_joint_ids = [self.mj_model.joint(n).id for n in ARM_JOINT_NAMES]
        self.arm_qadr = jnp.array(self.mj_model.jnt_qposadr[arm_joint_ids])   # qpos addr (7,)
        self.arm_dofadr = jnp.array(self.mj_model.jnt_dofadr[arm_joint_ids])  # dof addr (7,)
        self._arm_qadr_np = np.array(self.mj_model.jnt_qposadr[arm_joint_ids])
        self._arm_dofadr_np = np.array(self.mj_model.jnt_dofadr[arm_joint_ids])
        self.joint_limits = self.mj_model.jnt_range[arm_joint_ids]  # (7, 2)
        # Per-joint IK velocity cap = the velocity-actuator ctrlrange (J1-4 ±2.62,
        # J5/7 ±5.26, J6 ±4.18). Element-wise clamp before mjx (anti-flailing) that
        # actually binds, unlike a single scalar cap.
        self.joint_vel_limits = jnp.array(self.mj_model.actuator_ctrlrange[:, 1], dtype=jnp.float32)  # (7,)

        # Goal pose + mocap. The goal is a mocap body; mjx does NOT auto-fill
        # mocap_pos/quat and (in mjx 3.2.7) does NOT compute framepos/framequat
        # sensors with a reftype. So we set the goal mocap explicitly on reset and
        # compute block/ee pose relative to the goal from body kinematics (xpos/xquat).
        _gd = mujoco.MjData(self.mj_model)
        mujoco.mj_forward(self.mj_model, _gd)
        self.goal_pos = jnp.array(_gd.xpos[self.goal_body_id], dtype=jnp.float32)
        self.goal_quat = jnp.array(_gd.xquat[self.goal_body_id], dtype=jnp.float32)
        self.mocap_pos = jnp.array(_gd.mocap_pos, dtype=jnp.float32)
        self.mocap_quat = jnp.array(_gd.mocap_quat, dtype=jnp.float32)

        self.qhome = jnp.array(QHOME)

        # precompute the fixed initial qpos (block pose + IK'd arm) once on CPU
        self.initial_qpos = jnp.array(self._compute_initial_qpos(), dtype=jnp.float32)
        self.initial_qvel = jnp.zeros(self.mjx_model.nv, dtype=jnp.float32)

        # 2-D normalised action, 24-dim observation
        self.single_action_space = BoxSpace(
            low=jnp.array([-1.0, -1.0]), high=jnp.array([1.0, 1.0]),
            shape=(2,), dtype=jnp.float32)
        self.single_observation_space = BoxSpace(
            low=-jnp.inf, high=jnp.inf, shape=(24,), dtype=jnp.float32)
        if self.feature_fn == "state_action":
            feature_dim = self.single_observation_space.shape[0] + self.single_action_space.shape[0]
        elif self.feature_fn == "base":
            feature_dim = 4  # [-pos_err, -orient_err, -ee_block_dist, -ctrl]
        elif self.feature_fn == "base_rbf":
            feature_dim = 7  # [-pos_err, -orient_err, -ee_block, pos_bump_tight, pos_bump_wide, ori_bump, goal_bump]
        else:
            feature_dim = self.single_observation_space.shape[0]
        self.single_features_shape = BoxSpace(
            low=-jnp.inf, high=jnp.inf, shape=(feature_dim,), dtype=jnp.float32)

        self.viewer = None
        if render:
            dt = self.mj_model.opt.timestep * self.nr_intermediate_steps
            self.viewer = MujocoViewer(self.mj_model, dt)
            c_model = deepcopy(self.mj_model)
            c_data = mujoco.MjData(c_model)
            mujoco.mj_step(c_model, c_data, 1)
            self.light_xdir = c_data.light_xdir
            self.light_xpos = c_data.light_xpos
            del c_model, c_data

    # ----------------------------------------------------------------- setup #
    def _compute_initial_qpos(self):
        """Damped-least-squares IK on CPU to place the arm at the fixed start.

        Reproduces PushTFranka.reset() from the data generator so the env start
        state matches the expert dataset exactly.
        """
        m, d = self.mj_model, mujoco.MjData(self.mj_model)
        # fixed block pose (free joint qpos[0:7])
        d.qpos[0] = self.BLOCK_POS[0]
        d.qpos[1] = self.BLOCK_POS[1]
        d.qpos[2] = 0.045
        half = self.BLOCK_ANGLE * 0.5
        d.qpos[3:7] = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])  # wxyz, z-rot

        q = np.array([0.0, -np.pi / 4, 0.0, -9 * np.pi / 10, 0.0, 3 * np.pi / 4, np.pi / 4])
        goal_quat_ee = np.array(self.GOAL_QUAT_EE)
        damping, step_size = 0.1, 1.0
        for _ in range(1000):
            d.qpos[self._arm_qadr_np] = q
            mujoco.mj_forward(m, d)
            cur_pos = d.xpos[self.ee_body_id]
            cur_quat = d.xquat[self.ee_body_id]
            pos_err = self.GOAL_POS_EE - cur_pos
            r_cur = R.from_quat(np.array([cur_quat[1], cur_quat[2], cur_quat[3], cur_quat[0]]))
            r_des = R.from_quat(np.array([goal_quat_ee[1], goal_quat_ee[2], goal_quat_ee[3], goal_quat_ee[0]]))
            orn_err = (r_des * r_cur.inv()).as_rotvec()
            err = np.concatenate([pos_err, orn_err])
            if np.linalg.norm(err) < 1e-3:
                break
            J_pos = np.zeros((3, m.nv))
            J_rot = np.zeros((3, m.nv))
            mujoco.mj_jacBody(m, d, J_pos, J_rot, self.ee_body_id)
            J = np.vstack([J_pos[:, self._arm_dofadr_np], J_rot[:, self._arm_dofadr_np]])
            H = J.T @ J + damping * np.eye(len(self._arm_dofadr_np))
            dq = np.linalg.solve(H, J.T @ err)
            q = np.clip(q + step_size * dq, self.joint_limits[:, 0], self.joint_limits[:, 1])
        d.qpos[self._arm_qadr_np] = q
        return d.qpos.copy()

    # ------------------------------------------------------------- dynamics #
    def _differential_ik(self, data, vel_xy):
        """Map a planar ee velocity (m/s) to 7 arm joint velocities.

        Analytic body Jacobian + damped pinv, with a null-space pull toward
        qhome and servo terms holding ee height/orientation. Port of the
        generator's differential_IK (deterministic_headless.py).
        """
        point = data.xpos[self.ee_body_id]
        jacp, jacr = mjx_support.jac(self.mjx_model, data, point, self.ee_body_id)  # (nv,3) each
        J = jnp.concatenate([jacp.T, jacr.T], axis=0)        # (6, nv)
        J = J[:, self.arm_dofadr]                            # (6, 7)
        J_pinv = jnp.linalg.pinv(J)

        ee_pos = data.xpos[self.ee_body_id]      # world ee position (kinematics)
        ee_quat = data.xquat[self.ee_body_id]    # world ee orientation (kinematics)
        goal_vec = _quat_error_body(self.GOAL_QUAT_EE, ee_quat)            # (3,)

        temp = jnp.concatenate([vel_xy, jnp.array([self.Z_HOLD - ee_pos[2]])])
        twist_err = jnp.concatenate([temp, goal_vec])                     # (6,)
        dq = J_pinv @ twist_err

        qnow = data.qpos[self.arm_qadr]
        N = jnp.eye(J.shape[1]) - J_pinv @ J
        dq = dq + N @ (self.KP_ORI * (self.qhome - qnow))
        # clamp joint-velocity commands so a singular/extreme config can't drive
        # the arm (and free block) into MJX divergence (per-joint, at the actuator limits)
        dq = jnp.nan_to_num(dq, nan=0.0, posinf=0.0, neginf=0.0)
        return jnp.clip(dq, -self.joint_vel_limits, self.joint_vel_limits), J

    @partial(jax.vmap, in_axes=(None, 0, None))
    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key, eval_mode):
        data = self.mjx_data
        next_observation = jnp.zeros(self.single_observation_space.shape, dtype=jnp.float32)
        info = {
            "rollout/episode_return": 0.0,
            "rollout/episode_length": 0,
            "env_info/pos_err": 0.0,
            "env_info/orient_err": 0.0,
            "env_info/is_success": 0.0,
            "env_info/reward_dist": 0.0,
            "env_info/reward_ctrl": 0.0,
            "env_info/diverged": 0.0,
        }
        info_episode_store = {"episode_return": 0.0, "episode_length": 0}
        state = State(data, next_observation, next_observation, 0.0, False, False,
                      info, info_episode_store, key)
        return self._reset(state)

    @partial(jax.jit, static_argnums=(0,))
    def _reset(self, state):
        data = self.mjx_data
        data = data.replace(qpos=self.initial_qpos, qvel=self.initial_qvel,
                            ctrl=jnp.zeros(self.mjx_model.nu),
                            mocap_pos=self.mocap_pos, mocap_quat=self.mocap_quat)
        data = mjx.forward(self.mjx_model, data)

        next_observation = self.get_observation(data)
        info = dict(state.info)
        info["env_info/pos_err"] = 0.0
        info["env_info/orient_err"] = 0.0
        info["env_info/is_success"] = 0.0
        info["env_info/reward_dist"] = 0.0
        info["env_info/reward_ctrl"] = 0.0
        info["env_info/diverged"] = 0.0
        return state.replace(
            data=data,
            next_observation=next_observation,
            actual_next_observation=next_observation,
            reward=0.0, terminated=False, truncated=False,
            info=info,
            info_episode_store={"episode_return": 0.0, "episode_length": 0},
        )

    @partial(jax.vmap, in_axes=(None, 0, 0))
    @partial(jax.jit, static_argnums=(0,))
    def step(self, state, action):
        return self._step(state, action)

    @partial(jax.jit, static_argnums=(0,))
    def _step(self, state, action):
        vel_xy = jnp.clip(action, -1.0, 1.0) * self.MAX_SPEED

        def substep(data, _):
            dq, J = self._differential_ik(data, vel_xy)
            data = mjx.step(self.mjx_model, data.replace(ctrl=dq))
            return data, J

        data, J = jax.lax.scan(substep, state.data, xs=(), length=self.nr_intermediate_steps)

        state.info_episode_store["episode_length"] += 1
        next_observation = self.get_observation(data)
        reward, r_info = self.get_reward(data, action)
        terminated = r_info["env_info/is_success"] > 0.5
        # divergence guard: block left a sane region (huge-finite) or state went
        # NaN -> end the episode so it resets instead of accumulating garbage.
        # (NaN > bound is False, so explicit isnan checks are needed; qvel/xpos can
        #  go NaN a step before qpos, so cover qvel too.)
        block_pos_raw = data.xpos[self.block_body_id] - self.goal_pos
        diverged = (jnp.nan_to_num(jnp.linalg.norm(block_pos_raw), nan=jnp.inf) > self.DIVERGE_BOUND) \
            | jnp.any(jnp.isnan(data.qpos)) | jnp.any(jnp.isnan(data.qvel))
        at_horizon = state.info_episode_store["episode_length"] >= self.horizon

        singular_values = jnp.linalg.svd(J, compute_uv=False)
        near_singular = jnp.min(singular_values) < self.SINGULARITY_THRESHOLD

        truncated = at_horizon | diverged | near_singular
        done = terminated | truncated

        state.info.update(r_info)
        state.info["env_info/diverged"] = diverged.astype(jnp.float32)
        state.info_episode_store["episode_return"] += reward
        # Log episode aggregates ONLY for genuine completions (success or horizon),
        # NOT divergence-resets -- otherwise short blow-up returns pollute the means
        # (the -7e9-style artifact). Auto-reset below still uses the full `done`.
        log_done = terminated | at_horizon
        state.info["rollout/episode_return"] = jnp.where(log_done, state.info_episode_store["episode_return"], state.info["rollout/episode_return"])
        state.info["rollout/episode_length"] = jnp.where(log_done, state.info_episode_store["episode_length"], state.info["rollout/episode_length"])

        def when_done(_):
            start_state = self._reset(state)
            return start_state.replace(
                actual_next_observation=next_observation,
                reward=reward, terminated=terminated, truncated=truncated)

        def when_not_done(_):
            return state.replace(
                data=data,
                next_observation=next_observation,
                actual_next_observation=next_observation,
                reward=reward, terminated=terminated, truncated=truncated)

        return jax.lax.cond(done, when_done, when_not_done, None)

    # --------------------------------------------------------- observation #
    def _block_rel_goal(self, data):
        """Block pose relative to the goal, from body kinematics (version-robust).

        The goal frame is axis-aligned (identity quat), so the relative position
        is just the world difference; the relative quat is conj(goal)*block.
        """
        block_pos = data.xpos[self.block_body_id] - self.goal_pos
        block_quat = _quat_mul(_quat_conj(self.goal_quat), data.xquat[self.block_body_id])
        return block_pos, block_quat

    def get_observation(self, data):
        block_pos, block_quat = self._block_rel_goal(data)
        ee_rel = data.xpos[self.ee_body_id] - self.goal_pos
        arm_q = data.qpos[self.arm_qadr]
        arm_v = data.qvel[self.arm_dofadr]
        obs = jnp.concatenate([block_pos, block_quat, ee_rel, arm_q, arm_v])
        # nan_to_num + clip: a diverged env stays bounded so it can't poison the
        # discriminator/critic (expert obs are all < ~3, so clipping never alters them).
        obs = jnp.nan_to_num(obs, nan=0.0, posinf=self.OBS_CLIP, neginf=-self.OBS_CLIP)
        return jnp.clip(obs, -self.OBS_CLIP, self.OBS_CLIP)

    def _goal_errors(self, data):
        block_pos, block_quat = self._block_rel_goal(data)
        block_quat = jnp.where(block_quat[0] < 0.0, -block_quat, block_quat)
        pos_err = jnp.linalg.norm(block_pos)
        orn_err = jnp.linalg.norm(_quat_to_rotvec(block_quat))
        # NaN/huge -> max error (a diverged state is a failure, not a success), bounded
        pos_err = jnp.clip(jnp.nan_to_num(pos_err, nan=self.ERR_CLIP, posinf=self.ERR_CLIP), 0.0, self.ERR_CLIP)
        orn_err = jnp.clip(jnp.nan_to_num(orn_err, nan=jnp.pi, posinf=jnp.pi), 0.0, jnp.pi)
        return pos_err, orn_err

    def get_reward(self, data, action):
        pos_err, orn_err = self._goal_errors(data)
        is_success = ((pos_err + orn_err) < self.success_threshold).astype(jnp.float32)

        reward_dist = -pos_err
        reward_orient = -orn_err
        reward_ctrl = -jnp.sum(jnp.square(action))
        if self.reward_style == "sparse":
            reward = jnp.where(is_success > 0.5, 1.0, 0.0)
        else:
            reward = reward_dist + 0.5 * reward_orient + 0.001 * reward_ctrl
        # explicit bounded neginf (default is ~-3.4e38) so a future reward-sign
        # change can't reintroduce a huge-negative leak into the critic
        reward = jnp.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=-10.0)

        info = {
            "env_info/pos_err": pos_err,
            "env_info/orient_err": orn_err,
            "env_info/is_success": is_success,
            "env_info/reward_dist": reward_dist,
            "env_info/reward_ctrl": reward_ctrl,
        }
        return reward, info

    def render(self, state):
        env_id = 0
        data = mjx.get_data(self.mj_model, state.data)[env_id]
        data.light_xdir = self.light_xdir
        data.light_xpos = self.light_xpos
        self.viewer.render(data)
        return state

    def close(self):
        if self.viewer:
            self.viewer.close()

    def feature_from_transition(self, observation, action, eps=1e-4):
        observation = jnp.asarray(observation, dtype=jnp.float32)
        action = jnp.asarray(action, dtype=jnp.float32)
        squeeze_output = observation.ndim == 1
        if squeeze_output:
            observation = observation[None, :]
            action = action[None, :]

        if self.feature_fn == "base_rbf":
            # Engineered features that make matching the expert REQUIRE reaching the
            # goal (not just matching its average distance). The broad "-err" terms
            # pull toward the goal; the goal-centered Gaussian "bumps" are peaked at
            # the goal -> high for the expert (which arrives) and ~0 for a policy
            # parked partway, AND steep near the goal (a real finishing gradient).
            block_pos = observation[:, 0:3]
            w = observation[:, 3]
            ee_rel = observation[:, 7:10]
            pos_err = jnp.linalg.norm(block_pos, axis=-1)
            orient_err = 1.0 - jnp.clip(w * w, 0.0, 1.0)          # 0 aligned .. 1 at 90deg
            ee_block = jnp.linalg.norm(ee_rel - block_pos, axis=-1)
            pos_bump_tight = jnp.exp(-(pos_err / 0.04) ** 2)      # ~1 only when essentially placed
            pos_bump_wide = jnp.exp(-(pos_err / 0.12) ** 2)
            ori_bump = jnp.exp(-(orient_err / 0.15) ** 2)
            goal_bump = jnp.exp(-(((pos_err + orient_err)) / 0.10) ** 2)  # smooth success proxy
            features = jnp.stack([-pos_err, -orient_err, -ee_block,
                                  pos_bump_tight, pos_bump_wide, ori_bump, goal_bump], axis=-1)
        elif self.feature_fn == "state_action":
            features = jnp.concatenate([observation, action], axis=-1)

        elif self.feature_fn == "base_without_ctrl":
            block_pos = observation[:, 0:3]            # block pos rel goal
            w = observation[:, 3]                      # block quat w (rel goal)
            ee_rel = observation[:, 7:10]              # ee pos rel goal
            pos_err = jnp.linalg.norm(block_pos, axis=-1)
            orient_err = 1.0 - jnp.clip(w * w, 0.0, 1.0)        # sin^2(theta/2); 0 = aligned
            ee_block = jnp.linalg.norm(ee_rel - block_pos, axis=-1)  # ee-to-block dist (contact)
            features = jnp.stack([-pos_err, -orient_err, -ee_block], axis=-1)

        elif self.feature_fn == "base":
            # Engineered, goal-relevant features so the LINEAR reward r=theta^T phi
            # can express the task (a linear reward over raw block coords cannot).
            # All are negative "rewards" (higher = better), mirroring point_maze.
            block_pos = observation[:, 0:3]            # block pos rel goal
            w = observation[:, 3]                      # block quat w (rel goal)
            ee_rel = observation[:, 7:10]              # ee pos rel goal
            pos_err = jnp.linalg.norm(block_pos, axis=-1)
            orient_err = 1.0 - jnp.clip(w * w, 0.0, 1.0)        # sin^2(theta/2); 0 = aligned
            ee_block = jnp.linalg.norm(ee_rel - block_pos, axis=-1)  # ee-to-block dist (contact)
            ctrl = jnp.sum(jnp.square(action), axis=-1)
            features = jnp.stack([-pos_err, -orient_err, -ee_block, -ctrl], axis=-1)
        else:
            features = observation

        if squeeze_output:
            return features[0]
        return features
