import numpy as np
import flax.linen as nn
import jax.numpy as jnp
from flax.linen.initializers import constant, orthogonal
from rl_x.environments.action_space_type import ActionSpaceType
from rl_x.environments.observation_space_type import ObservationSpaceType
from collections import deque


######################################################
######################################################
"""
All reward functions are linear in latent features. r = theta^T phi(.) where phi(.) returns latent features from either state-action or base features

Discriminators. Available types:
- D(phi(.)) : feature-based
- D(psi(phi(.))) : boltzmann-feature-based
- D(psi(phi(x))) + gamma h(psi(phi(x'))) = h(psi(phi(x))) : shapedboltzmann-feature-based

All accept input as f, s, a, s', absorbing: bool for implementation convenience. All handle absorbing states to avoid termination/survival bias (https://arxiv.org/pdf/1809.02925)
"""
######################################################
######################################################

def get_discriminator(config, env, reward_type='feature-based'):
    action_space_type = env.general_properties.action_space_type
    observation_space_type = env.general_properties.observation_space_type

    if action_space_type == ActionSpaceType.CONTINUOUS and observation_space_type == ObservationSpaceType.FLAT_VALUES:       
        if reward_type == 'feature-based':
            return DiscriminatorFeatureBased()
        elif reward_type == 'boltzmann-feature-based':
            # return BoltzmannDiscriminatorFeatureBased()
            return BoltzmannDiscriminatorFeatureBased(
                hidden_dims=config.algorithm.boltzmann_hidden_dims,
                latent_dim=config.algorithm.boltzmann_latent_dim,
                energy_hidden_dim=config.algorithm.boltzmann_energy_hidden_dim,
            )
        elif reward_type == 'shapedboltzmann-feature-based':
            raise NotImplementedError
        else:
            raise NotImplementedError

class DiscriminatorFeatureBased(nn.Module):
    @nn.compact
    def __call__(self, f, x, a, x_n, absorbing, shaping=None):
        """
        D(s,a) = theta^T phi(.)

        Args:
            f : features
        """
        theta = self.param("theta", constant(0.001), (f.shape[-1],))
        return jnp.dot(f, theta)


# Leave this commented code because I need it for old policies learned on hard-coded boltzmann network
# class BoltzmannDiscriminatorFeatureBased(nn.Module):
#     def setup(self):
#         hidden1 = 16
#         hidden2 = 32
#         hidden3 = 64
#         hidden4 = 128
#         latent_dim = 16
#
#         self.feat_dense1 = nn.Dense(hidden1, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))
#         self.feat_dense2 = nn.Dense(hidden2, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))
#         self.feat_dense3 = nn.Dense(hidden3, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))
#         self.feat_dense4 = nn.Dense(latent_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0))
#
#         self.theta = self.param("theta", constant(0.0), (latent_dim,))
#
#         self.energy_dense1 = nn.Dense(32, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))
#         self.energy_dense2 = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))
#
#     def encode_feature(self, f):
#         z = self.feat_dense1(f)
#         z = nn.relu(z)
#         z = self.feat_dense2(z)
#         z = nn.relu(z)
#         z = self.feat_dense3(z)
#         z = nn.relu(z)
#         z = self.feat_dense4(z)
#         return z
#
#     def energy_from_z(self, z):
#         e = self.energy_dense1(z)
#         e = nn.tanh(e)
#         e = self.energy_dense2(e)
#         return jnp.squeeze(e, axis=-1)
#
#     def energy_only(self, f):
#         z = self.encode_feature(f)
#         return self.energy_from_z(z)
#
#     def reward_only(self, f):
#         z = self.encode_feature(f)
#         r = jnp.dot(z, self.theta)
#         return r
#
#     def __call__(self, f, x, a, x_n, absorbing, shaping: float = 1.0):
#         """
#         D(s,a) = theta^T psi(phi(.)) where psi is a feature encoder trained to return Boltzmann features (using denoising score matching)
#
#         Args:
#             f : features
#         """
#         zf = self.encode_feature(f)
#         _ = self.energy_from_z(zf)
#         r = jnp.dot(zf, self.theta)
#         return r


class BoltzmannDiscriminatorFeatureBased(nn.Module):
    hidden_dims: str = "16,32,64"
    latent_dim: int = 16
    energy_hidden_dim: int = 32

    def setup(self):
        hidden_dims = [int(d) for d in self.hidden_dims.split(",")]

        self.feat_denses = [
            nn.Dense(dim, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))
            for dim in hidden_dims
        ]
        self.feat_densefinal = nn.Dense(self.latent_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0))

        self.theta = self.param("theta", constant(0.0), (self.latent_dim,))

        self.energy_dense1 = nn.Dense(self.energy_hidden_dim, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))
        self.energy_dense2 = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))

    def encode_feature(self, f):
        z = f
        for feat_dense in self.feat_denses:
            z = feat_dense(z)
            z = nn.relu(z)
        z = self.feat_densefinal(z)
        return z

    def energy_from_z(self, z):
        e = self.energy_dense1(z)
        e = nn.tanh(e)
        e = self.energy_dense2(e)
        return jnp.squeeze(e, axis=-1)

    def energy_only(self, f):
        z = self.encode_feature(f)
        return self.energy_from_z(z)

    def reward_only(self, f):
        z = self.encode_feature(f)
        r = jnp.dot(z, self.theta)
        return r

    def __call__(self, f, x, a, x_n, absorbing, shaping: float = 1.0):
        """
        D(s,a) = theta^T psi(phi(.)) where psi is a feature encoder trained to return Boltzmann features (using denoising score matching)

        Args:
            f : features
        """
        zf = self.encode_feature(f)
        _ = self.energy_from_z(zf)
        r = jnp.dot(zf, self.theta)
        return r