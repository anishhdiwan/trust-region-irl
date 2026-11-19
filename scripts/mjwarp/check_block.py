"""Is the T's motion on Warp right? Arm held still (zero action); T at each preset; JAX, Warp and MuJoCo C side by side."""
import numpy as np, jax, jax.numpy as jnp, mujoco
from mujoco import mjx
from trust_region_irl.environments.pusht_mjx.environment import PushT

N = 6
env = {k: PushT(render=False, feature_fn="base_without_ctrl", random_TPos=False, impl=k, nr_envs=N) for k in ("jax", "warp")}
m = env["jax"].mj_model; T = 20 * env["jax"].nr_intermediate_steps
# simpler: reset each env to preset i via reset_to_preset on a batch of 1
q0 = []
e0 = env["jax"]
for i in range(N):   # same construction as PushT._reset for preset i
    bx, by, ba = [float(v) for v in np.asarray(e0.preset_block_poses[i])[:3]]
    q = np.asarray(e0.initial_qpos).copy(); q[0:7] = [bx, by, 0.045, np.cos(ba / 2), 0, 0, np.sin(ba / 2)]
    q0.append(q)
q0 = np.stack(q0)
print("block start z per preset:", q0[:, 2].round(4).tolist(), flush=True)
res = {}
for k, e in env.items():
    s = e.reset(jax.random.split(jax.random.PRNGKey(0), N), False)
    d = jax.vmap(lambda dd, q: mjx.forward(e.mjx_model, dd.replace(qpos=q, qvel=jnp.zeros_like(dd.qvel), ctrl=jnp.zeros_like(dd.ctrl))))(s.data, jnp.asarray(q0))
    run = jax.jit(jax.vmap(lambda x: jax.lax.scan(lambda y, _: (mjx.step(e.mjx_model, y), y.qpos[0:3]), x, None, length=T)[1]))
    res[k] = np.asarray(run(d))   # (N, T, 3) block xyz
c = []
for i in range(N):
    cd = mujoco.MjData(m); cd.qpos[:] = q0[i]; cd.mocap_pos[:] = np.asarray(env["jax"].mocap_pos); cd.mocap_quat[:] = np.asarray(env["jax"].mocap_quat)
    mujoco.mj_forward(m, cd); tr = []
    for _ in range(T): tr.append(cd.qpos[0:3].copy()); mujoco.mj_step(m, cd)
    c.append(np.stack(tr))
res["C"] = np.stack(c)
for k in ("C", "jax", "warp"):
    r = res[k]
    print(f"{k:4s} | block xy drift over {T} substeps (cm): {(100*np.linalg.norm(r[:, -1, :2] - r[:, 0, :2], axis=1)).round(3).tolist()} | final z: {r[:, -1, 2].round(4).tolist()}", flush=True)
print("warp vs C final xy diff (cm):", (100*np.linalg.norm(res['warp'][:, -1, :2] - res['C'][:, -1, :2], axis=1)).round(3).tolist())
print("jax  vs C final xy diff (cm):", (100*np.linalg.norm(res['jax'][:, -1, :2] - res['C'][:, -1, :2], axis=1)).round(3).tolist())
d = mujoco.MjData(m); d.qpos[:] = q0[0]; mujoco.mj_forward(m, d); print("MuJoCo C contacts at preset 0 start:", d.ncon, "| T geoms:", [m.geom(g).name or g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body('block').id][:8])
