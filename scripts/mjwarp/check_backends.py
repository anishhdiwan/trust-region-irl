"""Warp vs JAX backend check for this repo's PushT env (MuJoCo >= 3.6, conda env trirl_warp36).

1. same state -> kinematics and full IK output (incl. the manipulability gradient) on both backends
2. same state + fixed ctrl, K physics steps -> both backends vs MuJoCo C
3. same actions, closed-loop env.step rollout (random_TPos presets) on both backends
"""
import numpy as np, jax, jax.numpy as jnp, mujoco
from mujoco import mjx
from trust_region_irl.environments.pusht_mjx.environment import PushT

N = 8
envs = {impl: PushT(render=False, feature_fn="base_without_ctrl", random_TPos=True, impl=impl, nr_envs=N) for impl in ("jax", "warp")}
keys = jax.random.split(jax.random.PRNGKey(3), N)
st = {k: e.reset(keys, False) for k, e in envs.items()}
cmp = lambda a, b: float(jnp.max(jnp.abs(jnp.asarray(a) - jnp.asarray(b))))
print("mujoco", mujoco.__version__, "| reset obs diff", cmp(st["jax"].next_observation, st["warp"].next_observation), flush=True)

# 1. perturbed arm, same state on both backends
q = st["jax"].data.qpos + 0.05 * jax.random.normal(jax.random.PRNGKey(4), st["jax"].data.qpos.shape) * (jnp.arange(st["jax"].data.qpos.shape[-1]) >= 7)
d = {k: jax.vmap(lambda dd, qq, e=e: mjx.forward(e.mjx_model, dd.replace(qpos=qq)))(st[k].data, q) for k, e in envs.items()}
v = jax.random.uniform(jax.random.PRNGKey(5), (N, 2), minval=-1, maxval=1) * PushT.MAX_SPEED
ik = {k: jax.jit(jax.vmap(e._differential_ik))(d[k], v) for k, e in envs.items()}
print("same state | xpos diff", cmp(d["jax"].xpos, d["warp"].xpos), "| IK dq diff", cmp(ik["jax"], ik["warp"]), "| |dq| max", float(jnp.abs(ik["jax"]).max()), flush=True)

# 2. fixed ctrl physics vs MuJoCo C (env 0)
e = envs["jax"]; m = e.mj_model; K = e.nr_intermediate_steps * 4
q0, v0, c0 = np.asarray(d["jax"].qpos[0]), np.asarray(d["jax"].qvel[0]), np.asarray(ik["jax"][0])
cd = mujoco.MjData(m); cd.qpos[:], cd.qvel[:], cd.ctrl[:] = q0, v0, c0
cd.mocap_pos[:], cd.mocap_quat[:] = np.asarray(d["jax"].mocap_pos[0]), np.asarray(d["jax"].mocap_quat[0])
mujoco.mj_forward(m, cd)
for _ in range(K): mujoco.mj_step(m, cd)
for k, en in envs.items():
    dd = jax.tree_util.tree_map(lambda x: x, d[k])
    run = jax.jit(jax.vmap(lambda x, c, en=en: jax.lax.fori_loop(0, K, lambda i, y: mjx.step(en.mjx_model, y.replace(ctrl=c)), x)))
    out = run(dd, ik["jax"])
    print(f"{k} vs MuJoCo C after {K} steps | qpos max diff {cmp(out.qpos[0], cd.qpos):.2e}", flush=True)

# 3. closed-loop rollout with the env's own step (IK + physics + auto-reset), same actions
acts = jax.random.uniform(jax.random.PRNGKey(6), (60, N, 2), minval=-1, maxval=1)
obs = {}
for k, en in envs.items():
    s = en.reset(keys, False); o = []
    for t in range(60):
        s = en.step(s, acts[t]); o.append(s.next_observation)
    obs[k] = jnp.stack(o)
    print(f"{k} rollout | finite {bool(jnp.isfinite(obs[k]).all())} | diverged any {bool(jnp.any(s.info['env_info/diverged']))}", flush=True)
for t in (0, 9, 29, 59):
    print(f"rollout step {t+1}: block pos diff (cm) max {100*cmp(obs['jax'][t,:,0:3], obs['warp'][t,:,0:3]):.4f} | ee_rel diff (cm) max {100*cmp(obs['jax'][t,:,7:10], obs['warp'][t,:,7:10]):.4f}", flush=True)
print("done")

# 4. new domain-randomised spawn: per-env T poses after reset, and distance to the end effector
for k, en in envs.items():
    s = en.reset(jax.random.split(jax.random.PRNGKey(11), 256), False)
    xy = np.asarray(s.data.qpos[:, 0:2]); g = np.asarray(en.goal_pos[0:2])
    r = np.linalg.norm(xy - g, axis=1)
    print(f"{k} spawn | unique poses {len(np.unique(xy.round(5), axis=0))}/256 | dist to goal max {r.max():.3f} m | finite {np.isfinite(xy).all()}", flush=True)
