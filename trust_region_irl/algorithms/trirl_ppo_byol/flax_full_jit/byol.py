import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence

class MLP(nn.Module):
    """Encoder for state vector"""
    hidden_dims:Sequence[int]
    latent_dim:int

    @nn.compact
    def __call__(self, x):
        for dim in self.hidden_dims:
            x = nn.Dense(dim, use_bias=False)(x)
            x = nn.relu(x)

        x = nn.Dense(self.latent_dim)(x)
        return x

class CNN(nn.Module):
    latent_dim:int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4))(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1))(x)
        x = nn.relu(x)

        # Flatten and project to latent space
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(512)(x)
        x = nn.relu(x)
        x = nn.Dense(self.latent_dim)(x)
        return x

class BYOL(nn.Module):
    input_type: str = "state"
    state_hidden_dims: Sequence[int] = (64, 64)
    latent_dim: int = 128
    proj_hidden_dim: int = 256
    proj_dim: int = 128

    def setup(self):
        if self.input_type == "state":
            self.encoder = MLP(hidden_dims=self.state_hidden_dims, latent_dim=self.latent_dim)
        elif self.input_type == "image":
            self.encoder = CNN(latent_dim=self.latent_dim)
        else:
            raise NotImplementedError

        self.proj_dense1 = nn.Dense(self.proj_hidden_dim)
        self.proj_dense2 = nn.Dense(self.proj_dim)

        self.pred_dense1 = nn.Dense(self.proj_hidden_dim)
        self.pred_dense2 = nn.Dense(self.proj_dim)

    def projector(self, z):
        p = self.proj_dense1(z)
        p = nn.relu(p)
        p = self.proj_dense2(p)

        return p

    def predictor(self, z):
        p = self.pred_dense1(z)
        p = nn.relu(p)
        p = self.pred_dense2(p)
        return p

    def online(self, x):
        z = self.encoder(x)
        p = self.projector(z)
        q = self.predictor(p)

        return q

    def target(self, x):
        z = self.encoder(x)
        p = self.projector(z)
        return p

    def encode(self, x):
        return self.encoder(x)

    def __call__(self, x):
        return self.online(x)
