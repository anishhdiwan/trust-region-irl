import flax.linen as nn
import jax.numpy as jnp
from flax.linen.initializers import constant
from rl_x.environments.action_space_type import ActionSpaceType
from rl_x.environments.observation_space_type import ObservationSpaceType


######################################################
######################################################
"""
All reward functions are linear in latent features. r = theta^T z where z = psi_BYOL(.) returns latent features.

Discriminator:
- D(z) = theta^T z : feature-based (linear in BYOL latent features z)
"""
######################################################
######################################################

def get_discriminator(config, env, reward_type='feature-based'):
    action_space_type = env.general_properties.action_space_type
    observation_space_type = env.general_properties.observation_space_type

    if action_space_type == ActionSpaceType.CONTINUOUS and observation_space_type == ObservationSpaceType.FLAT_VALUES:
        if reward_type == 'feature-based':
            return DiscriminatorFeatureBased()
        else:
            raise NotImplementedError


class DiscriminatorFeatureBased(nn.Module):
    @nn.compact
    def __call__(self, f, x, a, x_n, absorbing, shaping=None):
        """
        D(s,a) = theta^T z

        Args:
            f : BYOL latent features z of shape (..., byol_latent_dim)
        """
        theta = self.param("theta", constant(0.001), (f.shape[-1],))
        return jnp.dot(f, theta)
