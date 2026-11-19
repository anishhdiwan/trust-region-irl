"""Check balance_fr3_mjx against BaMPC demos: one-step replay of recorded commands through the ported IK vs the recorded
next state (BaMPC: MuJoCo 3.10 Warp), with the recorded 10 ms sub-step commands and with their 50 ms mean held. Then a
short env smoke run.

    python scripts/balance/check_replay.py RAW_DEMOS.npz [jax] [warp]      (RAW_DEMOS: bampc/collect_demos.py output)
"""
import sys, numpy as np, jax, jax.numpy as jnp
from trust_region_irl.environments.balance_fr3_mjx.environment import BalanceFR3
D = sys.argv[1]
r = np.load(D); rng = np.random.default_rng(1); N = 300
E, T = r["qpos"].shape[:2]
ei, ti = rng.integers(0, E, N), rng.integers(0, T, N)
q0, v0, t0 = r["qpos"][ei, ti], r["qvel"][ei, ti], r["time"][ei, ti]
for impl in sys.argv[2:] or ["jax", "warp"]:
    env = BalanceFR3(impl=impl, nr_envs=N)
    base = env.mjx_data
    def mk(q, v, t):
        from mujoco import mjx
        d = base.replace(qpos=q, qvel=v, time=jnp.asarray(t, base.time.dtype))
        return mjx.forward(env.mjx_model, d)
    data = jax.jit(jax.vmap(mk))(jnp.asarray(q0), jnp.asarray(v0), jnp.asarray(t0))
    # 1. one-step replay: recorded sub-step commands, and the held mean command
    step = jax.jit(jax.vmap(env.physics_step))
    for name, us in (("recorded 5 sub-steps", r["action_sub"][ei, ti]), ("held mean action", np.repeat(r["action"][ei, ti][:, None], env.nr_intermediate_steps, 1))):
        out = step(data, jnp.asarray(us))
        bq = np.linalg.norm(np.asarray(out.qpos)[:, 0:3] - r["next_qpos"][ei, ti, 0:3], axis=1) * 1000
        bv = np.linalg.norm(np.asarray(out.qvel)[:, 0:3] - r["next_qvel"][ei, ti, 0:3], axis=1) * 1000
        aq = np.abs(np.asarray(out.qpos)[:, 7:] - r["next_qpos"][ei, ti, 7:]).max(1)
        print(f"[{impl}] {name:22s}: ball pos diff mm median {np.median(bq):.4f} 99pct {np.percentile(bq,99):.4f} max {bq.max():.4f} | "
              f"ball vel diff mm/s median {np.median(bv):.3f} max {bv.max():.3f} | arm joint diff rad max {aq.max():.2e}", flush=True)
    # 2. env smoke: reset + 50 random steps
    M = 64; e2 = BalanceFR3(impl=impl, nr_envs=M)
    s = e2.reset(jax.random.split(jax.random.PRNGKey(0), M), False)
    acts = jax.random.uniform(jax.random.PRNGKey(1), (50, M, 2), minval=-0.3, maxval=0.3)
    for a in acts: s = e2.step(s, a)
    o = np.asarray(s.next_observation)
    print(f"[{impl}] smoke: obs {o.shape} finite {np.isfinite(o).all()} | goal dist mean {float(np.mean(s.info['env_info/goal_dist']))*100:.1f} cm | ball lost frac {float(np.mean(s.info['env_info/ball_lost'])):.2f}", flush=True)
