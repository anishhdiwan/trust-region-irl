import flax.linen as nn
import jax.numpy as jnp
from flax.linen.initializers import constant, orthogonal

class BYOL(nn.Module):
    # Input indices to select what observations to feed into the networks
    input_indices: tuple = ()

    def setup(self):
        # Online Encoder Layers
        self.online_enc1 = nn.Dense(256, kernel_init=orthogonal(1.0), bias_init=constant(0.0))
        self.online_enc_norm = nn.LayerNorm()
        self.online_enc2 = nn.Dense(256, kernel_init=orthogonal(1.0), bias_init=constant(0.0))
        self.online_enc3 = nn.Dense(64, kernel_init=orthogonal(1.0), bias_init=constant(0.0))

        # Online Projector Layers
        self.online_proj1 = nn.Dense(256, kernel_init=orthogonal(1.0), bias_init=constant(0.0))
        self.online_proj_norm = nn.LayerNorm()
        self.online_proj2 = nn.Dense(128, kernel_init=orthogonal(1.0), bias_init=constant(0.0))

        # Online Predictor Layers
        self.online_pred1 = nn.Dense(256, kernel_init=orthogonal(1.0), bias_init=constant(0.0))
        self.online_pred_norm = nn.LayerNorm()
        self.online_pred2 = nn.Dense(128, kernel_init=orthogonal(1.0), bias_init=constant(0.0))

        # Target Encoder Layers
        self.target_enc1 = nn.Dense(256, kernel_init=orthogonal(1.0), bias_init=constant(0.0))
        self.target_enc_norm = nn.LayerNorm()
        self.target_enc2 = nn.Dense(256, kernel_init=orthogonal(1.0), bias_init=constant(0.0))
        self.target_enc3 = nn.Dense(64, kernel_init=orthogonal(1.0), bias_init=constant(0.0))

        # Target Projector Layers
        self.target_proj1 = nn.Dense(256, kernel_init=orthogonal(1.0), bias_init=constant(0.0))
        self.target_proj_norm = nn.LayerNorm()
        self.target_proj2 = nn.Dense(128, kernel_init=orthogonal(1.0), bias_init=constant(0.0))

        # Reward Parameters
        self.theta = self.param("theta", constant(0.001), (64,))

    def input(self, x):
        """
        Chooses the needed observations from the environment
        If empty, the 24D observation will be fed to the networks
        If not empty, the chosen indices will be fed to the networks
        """
        if len(self.input_indices) == 0:
            return x
        else:
            return x[..., jnp.array(self.input_indices)]

    def _online_encoder(self, x):
        z = self.online_enc1(x)
        z = self.online_enc_norm(z)
        z = nn.relu(z)
        z = self.online_enc2(z)
        z = nn.relu(z)
        z = self.online_enc3(z)

        return z

    def _online_projector(self, z):
        y = self.online_proj1(z)
        y = self.online_proj_norm(y)
        y = nn.relu(y)
        y = self.online_proj2(y)

        return y

    def _online_predictor(self,y ,a):
        p = jnp.concatenate((y,a),axis=-1)
        p = self.online_pred1(p)
        p = self.online_pred_norm(p)
        p = nn.relu(p)
        p = self.online_pred2(p)

        return p

    def _multi_step_prediction(self, y, actions, next_states):
        p = self._online_predictor(y, actions[0])
        for a in actions[1:]:
            p = jnp.concatenate((p, self._online_predictor(p, a)), axis=-1)

        return p


    def _target_encoder(self, x):
        z = self.target_enc1(x)
        z = self.target_enc_norm(z)
        z = nn.relu(z)
        z = self.target_enc2(z)
        z = nn.relu(z)
        z = self.target_enc3(z)

        return z

    def _target_projector(self, z):
        y = self.target_proj1(z)
        y = self.target_proj_norm(y)
        y = nn.relu(y)
        y = self.target_proj2(y)

        return y

    def _online_network(self, x, a):
        x = self.input(x)
        z = self._online_encoder(x)
        y = self._online_projector(z)
        p = self._online_predictor(y,a)

        return z, p

    def _target_network(self, x):
        x = self.input(x)
        z = self._target_encoder(x)
        y = self._target_projector(z)

        return y

    def _reward(self, x,a):
        z, _ = self._online_network(x, a)

        return z @ self.theta

    def __call__(self, x, a, x_next):
        # Predict the embedding of the next state give state and action
        z, p_online = self._online_network(x, a)
        y_target = self._target_network(x_next)

        return z, p_online, y_target

    def multi_step_byol(self, x, actions, next_states):
        x = self.input(x)

        # Online: encode s_t once, then chain the predictor through the k actions
        z = self._online_encoder(x)
        y = self._online_projector(z)

        p = self._online_predictor(y, actions[0])
        ps = [p]
        for a in actions[1:]:
            p = self._online_predictor(p, a)
            ps.append(p)
        ps = jnp.stack(ps)  # (k, 128)

        # Target: embed the k real future states in one call
        y_targets = self._target_network(next_states)  # (k, 128)

        return z, ps, y_targets


