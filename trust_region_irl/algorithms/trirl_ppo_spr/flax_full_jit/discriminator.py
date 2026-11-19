import numpy as np
from flax.linen.initializers import constant, orthogonal
from rl_x.environments.action_space_type import ActionSpaceType
from rl_x.environments.observation_space_type import ObservationSpaceType
from collections import deque
from trust_region_irl.algorithms.trirl_ppo_spr.flax_full_jit.byol import BYOL

def get_discriminator(config, env, reward_type='feature-based'):
    action_space_type = env.general_properties.action_space_type
    observation_space_type = env.general_properties.observation_space_type

    if action_space_type == ActionSpaceType.CONTINUOUS and observation_space_type == ObservationSpaceType.FLAT_VALUES:
        if reward_type == 'byol-observation-based':
            return BYOL()
        elif reward_type == 'byol-task-based-observation':
            # T position [0, 1, 2], T quaternion [3, 4, 5, 6]
            return BYOL(input_indices=(0, 1, 2, 3, 4, 5, 6))
        elif reward_type == 'byol-embodiment-free':
            # the encoder (and so the reward) sees only the task part of the observation: the env lists its non-robot entries
            # (no arm joint positions/velocities), while the policy and critic keep the full observation
            return BYOL(input_indices=tuple(int(i) for i in env.encoder_input_indices))
        else:
            raise NotImplementedError

    else:
        raise NotImplementedError
