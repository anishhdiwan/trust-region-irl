import numpy as np
import jax
import jax.numpy as jnp


def prepare_expert_data(data_path, cutoff=1):
    dataset = dict()
    expert_files = np.load(data_path)

    def _flatten_feature_array(x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim <= 2:
            return x
        return x.reshape(-1, x.shape[-1])

    def _flatten_scalar_array(x):
        return np.asarray(x, dtype=np.float32).reshape(-1)

    states = _flatten_feature_array(expert_files["states"])
    actions = _flatten_feature_array(expert_files["actions"])

    cutoff = int(states.shape[0] / cutoff)

    dataset["states"] = states[:cutoff]
    dataset["actions"] = actions[:cutoff]
    dataset["next_states"] = _flatten_feature_array(expert_files["next_states"])[:cutoff]
    dataset["absorbing"] = _flatten_scalar_array(expert_files["absorbing"])[:cutoff]

    return dataset


def expert_data_spec(num_samples, state_dim, action_dim):
    """
    Dummy spec for use with full jittting
    """
    return {
        "states": jax.ShapeDtypeStruct(
            shape=(num_samples, state_dim),
            dtype=jnp.float32,
        ),
        "actions": jax.ShapeDtypeStruct(
            shape=(num_samples, action_dim),
            dtype=jnp.float32,
        ),
        "next_states": jax.ShapeDtypeStruct(
            shape=(num_samples, state_dim),
            dtype=jnp.float32,
        ),
        "absorbing": jax.ShapeDtypeStruct(
            shape=(num_samples,),
            dtype=jnp.float32,
        ),
    }

def agent_windows(actions, next_states, dones, k):
    T, N = dones.shape
    idx = jnp.arange(T)[:, None] + jnp.arange(k)[None, :]
    clamped = jnp.minimum(idx, T-1)

    action_windows = jnp.swapaxes(actions[clamped], 1, 2).reshape((T * N, k) + actions.shape[2:])
    next_states_windows = jnp.swapaxes(next_states[clamped], 1, 2).reshape((T * N, k) + next_states.shape[2:])

    in_range = (jnp.arange(T) + k - 1) <= (T - 1)
    crosses_episode_end = jnp.any(dones[clamped[:, :k - 1]], axis=1)

    valid = (in_range[:, None] & ~crosses_episode_end).reshape(-1)

    weight = valid / jnp.maximum(jnp.mean(valid), 1e-8)

    return action_windows, next_states_windows, valid, weight

def expert_windows(expert_actions, expert_next_states, expert_absorbing, expert_idx, k):
    M = expert_absorbing.shape[0]

    idx = expert_idx[:, None] + jnp.arange(k)[None, :]
    clamped = jnp.minimum(idx, M-1)

    action_windows = expert_actions[clamped]
    next_states_windows = expert_next_states[clamped]

    in_range = (expert_idx + k - 1) <= (M - 1)
    crosses_episode_end = jnp.any(expert_absorbing[clamped[:, :k - 1]] > 0, axis=1)

    valid = in_range & ~crosses_episode_end

    weight = valid / jnp.maximum(jnp.mean(valid), 1e-8)

    return action_windows, next_states_windows, valid, weight