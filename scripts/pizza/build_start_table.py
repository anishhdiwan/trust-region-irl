"""Arm poses with the tool at the pizza_dr travel height over a polar grid of xy around the robot base.

pizza_dr_mjx starts each episode from one of these poses (picked inside the sampled pizza disc). The env's own robust IK
drives the tool from home to every grid point; points the tool does not reach accurately and comfortably are dropped.

    python scripts/pizza/build_start_table.py
"""
import numpy as np
import jax, jax.numpy as jnp
from mujoco import mjx
from mujoco.mjx._src import support as mjx_support
from trust_region_irl.environments.pizza_dr_mjx.environment import PizzaSpreadDR, START_TABLE

env = PizzaSpreadDR(render=False)
D, A, STEP = np.arange(0.30, 0.801, 0.01), np.deg2rad(65.0), 0.01   # distance from the base, max bearing, grid spacing (m)
pts = np.array([(d * np.cos(a), d * np.sin(a)) for d in D for a in np.arange(-A, A + 1e-9, STEP / d)], np.float32)
print("grid points", len(pts), flush=True)
T, GAIN = 240, 4.0

def drive(target):
    data = mjx.forward(env.mjx_model, env.mjx_data.replace(qpos=env.initial_qpos, qvel=env.initial_qvel))
    def body(data, _):
        xy = data.xpos[env.ee_body_id][:2]
        v = GAIN * (target - xy); v = v * jnp.minimum(1.0, env.MAX_SPEED / (jnp.linalg.norm(v) + 1e-9))
        def sub(d, _):
            return mjx.step(env.mjx_model, d.replace(ctrl=env._differential_ik(d, v, env.Z_TRAVEL))), None
        data, _ = jax.lax.scan(sub, data, None, length=env.nr_intermediate_steps)
        return mjx.kinematics(env.mjx_model, data), None
    data, _ = jax.lax.scan(body, data, None, length=T)
    J = jnp.concatenate([a.T for a in mjx_support.jac(env.mjx_model, data, data.xpos[env.ee_body_id], env.ee_body_id)], 0)[:, env.arm_dofadr]
    return data.qpos[env.arm_qadr], data.xpos[env.ee_body_id], jnp.linalg.svd(J, compute_uv=False)[-1], jnp.max(jnp.abs(data.qvel[env.arm_dofadr]))

q, p, smin, vmax = [np.asarray(x) for x in jax.jit(jax.vmap(drive))(jnp.asarray(pts))]
lo, hi = np.asarray(env.mj_model.jnt_range[:7, 0]), np.asarray(env.mj_model.jnt_range[:7, 1])
margin = np.minimum(q - lo, hi - q).min(1)
err_xy = np.linalg.norm(p[:, :2] - pts, axis=1); err_z = np.abs(p[:, 2] - float(env.Z_TRAVEL))
ok = (err_xy < 0.003) & (err_z < 0.003) & (smin > 0.02) & (margin > 0.05) & (vmax < 0.05) & np.isfinite(q).all(1)
print(f"kept {ok.sum()}/{len(pts)} | rejected: xy {(err_xy >= 0.003).sum()} z {(err_z >= 0.003).sum()} singular {(smin <= 0.02).sum()} joint-limit {(margin <= 0.05).sum()} still-moving {(vmax >= 0.05).sum()}", flush=True)
START_TABLE.parent.mkdir(parents=True, exist_ok=True)
np.savez(START_TABLE, qpos=q[ok].astype(np.float32), xy=p[ok, :2].astype(np.float32), z_travel=float(env.Z_TRAVEL))
print("saved", START_TABLE)
