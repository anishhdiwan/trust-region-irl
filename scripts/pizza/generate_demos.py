"""Synthetic expert demos for pizza_dr_mjx: travel to the centre, (env) touchdown, clockwise centre-out spiral.

Natural ending: the spiral reaches the sauce edge (R - crust), closes one full loop there, slows to a stop, the env
lifts the tool off after 0.5 s standstill, and the tool hovers until the fixed horizon (no early termination).
Every demo is played through the env itself (random pizza position/size and start pose from its reset), so states are
exactly what the agent sees. Spiral pitch, start angle and speed vary per demo so the discriminator cannot memorise
one pattern; the direction is always clockwise. Only demos that run the full horizon, lift off and cover at least the
env's success threshold of the sauce area are kept (no wander-off, no divergence).

Output keys: states, actions, next_states, rewards, absorbing (1 on each demo's last step), episode_lengths, demo_meta
(per demo: cx cy R pitch speed length touchdown_step coverage off_pizza_frac lift_step).

    python scripts/pizza/generate_demos.py OUT.npz N_ENVS ROUNDS      (e.g. 512 envs x 4 rounds -> about 2,000 demos)
"""
import sys, time
import numpy as np
import jax, jax.numpy as jnp
from trust_region_irl.environments.pizza_dr_mjx.environment import PizzaSpreadDR

OUT, N, ROUNDS = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
PITCH = (0.022, 0.030)      # m between spiral turns; <= tool diameter (30 mm) so no gaps
SPEED = (0.18, 0.30)        # m/s along the spiral
TRAVEL_SPEED = 0.25         # m/s to the centre
DS, WIN, LOOK, K_STOP, ARRIVE = 0.004, 24, 4, 2.0, 0.004
env = PizzaSpreadDR(render=False, nr_envs=N)
T = env.horizon

def spiral(c, R, pitch, th0):
    """Clockwise Archimedean spiral from the centre to the sauce edge, then one full loop there, points every DS (world)."""
    r_out = (R - env.CRUST) - 0.5 * env.TOOL_R
    out, th = [], 0.0
    while True:
        r = pitch * th / (2 * np.pi)
        if r >= r_out: break
        out.append(c + r * np.array([np.cos(th0 - th), np.sin(th0 - th)])); th += DS / max(r, DS)
    r = r_out
    for k in range(int(2 * np.pi * r / DS) + 1):
        a = th0 - th - k * DS / r; out.append(c + r * np.array([np.cos(a), np.sin(a)]))
    return np.array(out, np.float32)

def run_round(seed):
    rng = np.random.default_rng(seed)
    state = env.reset(jax.random.split(jax.random.PRNGKey(seed), N), False)
    piz = np.asarray(state.info_episode_store["pizza"])
    pitch, th0, speed = rng.uniform(*PITCH, N), rng.uniform(0, 2 * np.pi, N), rng.uniform(*SPEED, N)
    paths = [spiral(piz[i, :2], piz[i, 2], pitch[i], th0[i]) for i in range(N)]
    L = max(len(p) for p in paths)
    refs = jnp.asarray(np.stack([np.concatenate([p, np.repeat(p[-1:], L - len(p), 0)]) for p in paths]))
    last = jnp.asarray([len(p) - 1 for p in paths]); centre = jnp.asarray(piz[:, :2])
    amp, amp_t = jnp.asarray(speed / env.MAX_SPEED, jnp.float32), TRAVEL_SPEED / env.MAX_SPEED

    def step(carry, _):
        s, idx, arrived = carry
        obs = s.next_observation; xy = obs[:, 0:2]
        spreading = s.info_episode_store["phase"] > 0.5
        # travel: straight to the centre, slowing over the last 3 cm
        d_c = centre - xy; n_c = jnp.linalg.norm(d_c, axis=-1, keepdims=True)
        a_travel = amp_t * jnp.minimum(1.0, n_c / 0.03) * d_c / (n_c + 1e-9)
        # spreading: follow the spiral at constant speed (env freezes xy during the descent)
        win = jnp.minimum(idx[:, None] + jnp.arange(WIN)[None], last[:, None])
        cand = jnp.take_along_axis(refs, win[:, :, None], axis=1)
        idx = jnp.where(spreading, jnp.minimum(idx + jnp.argmin(jnp.linalg.norm(cand - xy[:, None], axis=-1), 1), last), idx)
        tgt = jnp.take_along_axis(refs, jnp.minimum(idx + LOOK, last)[:, None, None], axis=1)[:, 0]
        d = tgt - xy
        end = jnp.take_along_axis(refs, last[:, None, None], axis=1)[:, 0]; d_end = jnp.linalg.norm(end - xy, axis=-1)
        final = idx + LOOK >= last
        arrived = arrived | (spreading & final & (d_end < ARRIVE))
        mag = jnp.where(final, jnp.minimum(amp, K_STOP * d_end / env.MAX_SPEED), amp) * ~arrived   # slow to a stop at the end, then stand still (env lifts off)
        a_spiral = mag[:, None] * d / (jnp.linalg.norm(d, axis=-1, keepdims=True) + 1e-9)
        a = jnp.where(spreading[:, None], a_spiral, a_travel)
        ns = env.step(s, a)
        done = ns.terminated | ns.truncated
        return (ns, idx, arrived), (obs, a, ns.actual_next_observation, ns.reward, ns.terminated, done, ns.info["rollout/wandered_off"],
                           ns.info["rollout/diverged"], ns.info["rollout/coverage"], ns.info["rollout/touchdown_step"], ns.info["rollout/off_pizza_frac"],
                           ns.info["rollout/lift_step"])

    _, out = jax.jit(lambda s: jax.lax.scan(step, (s, jnp.zeros(N, jnp.int32), jnp.zeros(N, bool)), None, length=T))(state)
    return [np.asarray(x) for x in out], piz, pitch, speed

S, A, NS, Rw, AB, meta = [], [], [], [], [], []
t0 = time.time()
for rd in range(ROUNDS):
    (obs, act, nobs, rew, term, done, wand, div, cov, tds, offf, lifts), piz, pitch, speed = run_round(rd)
    for i in range(N):
        ends = np.where(done[:, i])[0]
        if len(ends) == 0: continue
        e = ends[0]   # first episode of this env only (the auto-reset starts a different pizza)
        if e != T - 1 or wand[e, i] or div[e, i] or lifts[e, i] <= 0 or cov[e, i] < env.success_threshold: continue   # full horizon, lifted off, sauce area covered
        S.append(obs[:e + 1, i]); A.append(act[:e + 1, i]); NS.append(nobs[:e + 1, i]); Rw.append(rew[:e + 1, i])
        ab = np.zeros(e + 1, np.float32); ab[-1] = 1.0; AB.append(ab)
        meta.append((piz[i, 0], piz[i, 1], piz[i, 2], pitch[i], speed[i], e + 1, tds[e, i], cov[e, i], offf[e, i], lifts[e, i]))
    print(f"round {rd}: kept {len(meta)} demos so far | {time.time() - t0:.0f}s", flush=True)
meta = np.array(meta, np.float32)
np.savez_compressed(OUT, states=np.concatenate(S), actions=np.concatenate(A), next_states=np.concatenate(NS), rewards=np.concatenate(Rw),
                    absorbing=np.concatenate(AB), episode_lengths=meta[:, 5].astype(np.int32),
                    demo_meta=meta, demo_meta_cols=np.array("cx cy R pitch speed length touchdown_step coverage off_pizza_frac lift_step".split()))
L = meta[:, 5]
print(f"saved {OUT}: {len(meta)} demos, {int(L.sum())} transitions | length {L.min():.0f}-{L.max():.0f} (mean {L.mean():.0f}) | touchdown at step {meta[:,6].mean():.0f} | "
      f"lift-off at step {meta[:,9].min():.0f}-{meta[:,9].max():.0f} | R {meta[:,2].min():.3f}-{meta[:,2].max():.3f} | coverage {meta[:,7].min():.3f}-{meta[:,7].max():.3f} | off-pizza frac max {meta[:,8].max():.3f}")
