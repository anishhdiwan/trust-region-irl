"""Zero-shot check of trained Balance-FR3 policies under changed dynamics (ball mass, IK gains): which variants make the
nominal policy worse, i.e. where a reward-transfer test is informative. Deterministic policy, fixed random 10 s
episodes, plus the zero-action baseline per variant (how fast the ball falls off). Writes one JSON row per variant.

    python scripts/balance/dynamics_sweep.py OUT.json CKPT [CKPT ...] [--variants '[{"ball_mass_scale": 4.0}, ...]']
"""
import argparse
import json

import numpy as np
import jax
import jax.numpy as jnp

from trust_region_irl.environments.balance_fr3_mjx.environment import BalanceFR3
from trust_region_irl.algorithms.trirl_ppo_spr.flax_full_jit.policy import Policy
from trust_region_irl.policy_checkpoint import load_policy_params

DEFAULT_VARIANTS = [{}] + [dict(ball_mass_scale=s) for s in (0.25, 0.5, 2.0, 4.0)] + [dict(tilt_gain_scale=s) for s in (0.5, 0.75, 1.5, 2.0)] + \
                   [dict(ik_k_pos_scale=s) for s in (0.25, 4.0)] + [dict(ik_damping_scale=s) for s in (100.0, 1000.0)]
parser = argparse.ArgumentParser()
parser.add_argument("out")
parser.add_argument("ckpts", nargs="+")
parser.add_argument("--variants", default="", help="JSON list of env keyword dicts; default: the sweep above ({} = nominal)")
parser.add_argument("--n-eval", type=int, default=100)
parser.add_argument("--seed", type=int, default=2026)
args = parser.parse_args()
OUT, CKPTS, N, H = args.out, args.ckpts, args.n_eval, 200
VARIANTS = json.loads(args.variants) if args.variants else DEFAULT_VARIANTS


params = [load_policy_params(c) for c in CKPTS]
keys = jax.random.split(jax.random.PRNGKey(args.seed), N)
results = []
for v in VARIANTS:
    env = BalanceFR3(render=False, nr_envs=N, impl="warp", horizon=H, **v)
    policy = Policy((2,), 1.0, jnp.arange(env.single_observation_space.shape[0]))

    def make(act):
        @jax.jit
        def run(p):
            s0 = env.reset(keys, True)
            def step(s, _):
                s2 = env.step(s, act(p, s.next_observation))
                return s2, (s2.info["env_info/goal_dist"], s2.info["env_info/ball_lost"])
            _, (d, l) = jax.lax.scan(step, s0, None, length=H)
            return d, l
        return run

    row = dict(variant=v or "nominal")
    d, l = make(lambda p, o: jnp.zeros(o.shape[:-1] + (2,)))(None)
    row["zero_action"] = dict(goal_dist_2nd_half_cm=float(100 * np.asarray(d)[H // 2:].mean()), ball_lost=float(np.asarray(l).max(0).mean()))
    run = make(lambda p, o: policy.apply(p, o)[0])
    for c, p in zip(CKPTS, params):
        d, l = run(p); d, l = np.asarray(d), np.asarray(l)
        lost = l.max(0) > 0.5
        row[c.split("/")[-3][-5:]] = dict(goal_dist_2nd_half_cm=float(100 * d[H // 2:].mean()),
                                         goal_dist_2nd_half_kept_cm=float(100 * d[H // 2:, ~lost].mean()) if (~lost).any() else None,
                                         ball_lost=float(lost.mean()))
    results.append(row)
    print(json.dumps(row), flush=True)
    del env
json.dump(results, open(OUT, "w"), indent=1)
