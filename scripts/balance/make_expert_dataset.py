"""BaMPC demos -> TRIRL expert dataset for balance_fr3_mjx: observations from the env's own obs_from on the recorded states
and the recorded plate-local goal position and velocity,
action = the step's mean tilt command / MAX_TILT_VEL, absorbing = 1 on each demo's last step (episode boundary marker).

    python scripts/balance/make_expert_dataset.py RAW_DEMOS.npz trirl_dataset/rl_expert/balance_fr3_varied_300.npz
"""
import sys, numpy as np, jax, jax.numpy as jnp
from mujoco import mjx
from trust_region_irl.environments.balance_fr3_mjx.environment import BalanceFR3
SRC, OUT = sys.argv[1], sys.argv[2]
r = np.load(SRC); E, T = r["qpos"].shape[:2]
env = BalanceFR3(impl="jax", nr_envs=1); base = env.mjx_data
@jax.jit
@jax.vmap
def obs_of(q, v, t, gl, gv):
    d = mjx.forward(env.mjx_model, base.replace(qpos=q, qvel=v, time=jnp.asarray(t, base.time.dtype)))
    return env.obs_from(d, gl, gv)
dt = 1.0 / env.CONTROL_FREQ
flat = lambda k: jnp.asarray(r[k].reshape(E * T, *r[k].shape[2:]))
S = np.asarray(obs_of(flat("qpos"), flat("qvel"), jnp.asarray(r["time"].reshape(-1)), flat("goal_local"), flat("goal_local_vel")))
NS = np.asarray(obs_of(flat("next_qpos"), flat("next_qvel"), jnp.asarray(r["time"].reshape(-1) + dt), flat("next_goal_local"), flat("next_goal_local_vel")))
A = (r["action"].reshape(E * T, 2) / env.max_tilt_vel).astype(np.float32)
ab = np.zeros((E, T), np.float32); ab[:, -1] = 1.0
ab = ab.reshape(-1)
cont = np.abs(S.reshape(E, T, -1)[:, 1:] - NS.reshape(E, T, -1)[:, :-1]).max()   # next_state(t) must equal state(t+1)
np.savez_compressed(OUT, states=S.astype(np.float32), actions=A, next_states=NS.astype(np.float32), rewards=np.zeros(E * T, np.float32), absorbing=ab,
                    episode_lengths=np.full(E, T, np.int32), demo_meta=r["demo_meta"], demo_meta_cols=r["demo_meta_cols"])
print(f"saved {OUT}: {E} demos x {T} = {E*T} transitions | obs {S.shape[1]} | action range {A.min():.3f}..{A.max():.3f} | "
      f"next_state(t) vs state(t+1) max diff {cont:.2e} | finite {np.isfinite(S).all() and np.isfinite(NS).all()}")
