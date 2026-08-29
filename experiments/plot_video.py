import os
import shutil
import tempfile
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import orbax.checkpoint as ocp
import matplotlib.pyplot as plt
import imageio.v2 as imageio


def extract_features(observation):
    block_pos = observation[:, 0:3]
    w = observation[:, 3]
    ee_rel = observation[:, 7:10]
    pos_err = jnp.linalg.norm(block_pos, axis=-1)
    orient_err = 1.0 - jnp.clip(w * w, 0.0, 1.0)
    ee_block = jnp.linalg.norm(ee_rel - block_pos, axis=-1)
    features = jnp.stack([-pos_err, -orient_err, -ee_block], axis=-1)
    return features


class BoltzmannDiscriminatorFeatureBased(nn.Module):
    def setup(self):
        hidden1 = 16
        hidden2 = 32
        hidden3 = 64
        latent_dim = 16
        energy1 = 32

        self.feat_dense1 = nn.Dense(hidden1, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))
        self.feat_dense2 = nn.Dense(hidden2, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))
        self.feat_dense3 = nn.Dense(hidden3, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))
        self.feat_dense4 = nn.Dense(latent_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0))

        self.theta = self.param("theta", constant(0.0), (latent_dim,))

        self.energy_dense1 = nn.Dense(energy1, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))
        self.energy_dense2 = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))

    def encode_feature(self, f):
        z = self.feat_dense1(f)
        z = nn.relu(z)
        z = self.feat_dense2(z)
        z = nn.relu(z)
        z = self.feat_dense3(z)
        z = nn.relu(z)
        z = self.feat_dense4(z)
        return z

    def energy_from_z(self, z):
        e = self.energy_dense1(z)
        e = nn.tanh(e)
        e = self.energy_dense2(e)
        return jnp.squeeze(e, axis=-1)

    def energy_only(self, f):
        z = self.encode_feature(f)
        return self.energy_from_z(z)

    def __call__(self, f, x, a, x_n, absorbing, shaping: float = 1.0):
        zf = self.encode_feature(f)
        _ = self.energy_from_z(zf)
        r = jnp.dot(zf, self.theta)
        return r


def load_checkpoint(model_zip):
    tmp = tempfile.mkdtemp()
    try:
        shutil.unpack_archive(model_zip, tmp, "zip")
        ckpt = ocp.PyTreeCheckpointer().restore(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    disc_params = ckpt["discriminator"]["params"]
    theta_best = np.asarray(disc_params["params"]["theta"], dtype=np.float32)

    return theta_best, disc_params


def get_best_model_limits(theta_best, disc_params, boltzmann_encoder):
    """
    Calculates the exact min and max reward values ONLY for the Position heatmap
    using the best.model.zip's theta to keep the color scale static.
    """
    print("Calculating fixed scale limits for Position Reward based on best.model.zip theta...")

    # Position features
    xs_grid = np.linspace(-0.35, 0.35, 100)
    ys_grid = np.linspace(-0.35, 0.35, 100)
    XX, YY = np.meshgrid(xs_grid, ys_grid)
    base = np.zeros(24, dtype=np.float32)
    G = np.tile(base, (100 * 100, 1))
    G[:, 0] = XX.ravel()
    G[:, 1] = YY.ravel()
    features = extract_features(G)
    features = features.at[:, 1].set(0.0)
    features = features.at[:, 2].set(0.0)
    f_pos = jnp.stack([-features[:, 0], -features[:, 1], -features[:, 2]], axis=-1)
    z_pos = boltzmann_encoder.apply(disc_params, f_pos, method=BoltzmannDiscriminatorFeatureBased.encode_feature)
    r_pos = np.array(z_pos @ theta_best)
    pos_lims = (r_pos.min(), r_pos.max())

    # Add 5% padding to the limits
    span = pos_lims[1] - pos_lims[0]
    if span == 0: span = 1.0
    padded_pos_lims = (pos_lims[0] - 0.05 * span, pos_lims[1] + 0.05 * span)

    return padded_pos_lims


def plot_rewards_frame(theta, disc_params, boltzmann_encoder, frame_idx, output_dir, pos_lims):
    """
    Renders a single frame. Line charts auto-scale, heatmap uses fixed pos_lims.
    """
    qs = np.linspace(-np.pi, np.pi, 100).reshape(-1, 1)
    w = np.cos(qs / 2.0)
    orient_feat = -(1.0 - np.clip(w * w, 0.0, 1.0))

    features = jnp.hstack([np.zeros((100, 1)), orient_feat, np.zeros((100, 1))])
    encoded_features = boltzmann_encoder.apply(disc_params, features,
                                               method=BoltzmannDiscriminatorFeatureBased.encode_feature)
    ort_reward = jnp.array(encoded_features @ theta)

    xs = np.linspace(0.0, 0.35, 100).reshape(-1, 1)
    features = jnp.hstack([jnp.zeros((100, 1)), xs, np.zeros((100, 1))])

    pos_err = features[:, 0]
    orient_err = features[:, 1]
    ee_block = features[:, 2]

    features = jnp.stack([-pos_err, -orient_err, -ee_block], axis=-1)
    encoded_features = boltzmann_encoder.apply(disc_params, features,
                                               method=BoltzmannDiscriminatorFeatureBased.encode_feature)
    ee_reward = jnp.array(encoded_features @ theta)

    xs = np.linspace(-0.35, 0.35, 100)
    ys = np.linspace(-0.35, 0.35, 100)
    XX, YY = np.meshgrid(xs, ys)

    base = np.zeros(24, dtype=np.float32)
    G = np.tile(base, (100 * 100, 1))
    G[:, 0] = XX.ravel()
    G[:, 1] = YY.ravel()

    features = extract_features(G)
    features = features.at[:, 1].set(0.0)
    features = features.at[:, 2].set(0.0)

    pos_err = features[:, 0]
    orient_err = features[:, 1]
    ee_block = features[:, 2]

    features = jnp.stack([-pos_err, -orient_err, -ee_block], axis=-1)
    encoded_features = boltzmann_encoder.apply(disc_params, features,
                                               method=BoltzmannDiscriminatorFeatureBased.encode_feature)

    pos_reward = jnp.array(encoded_features @ theta).reshape(100, 100)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Orientation Reward
    axes[0].set_title(f"Orientation Reward (Step {frame_idx})")
    axes[0].plot(qs, ort_reward, color="#e11d48", label="reward")
    axes[0].set_xlabel("orientation error")
    axes[0].set_ylabel("reward")
    # No ylim set here -> let Matplotlib auto-scale to the current theta
    axes[0].legend()

    # 2. End-Effector Reward
    xs_ee = np.linspace(0.0, 0.35, 100)
    axes[1].set_title("End-Effector Distance")
    axes[1].plot(xs_ee, ee_reward, color="#e11d48", label="reward")
    axes[1].set_xlabel("distance between ee and block")
    axes[1].set_ylabel("reward")
    # No ylim set here -> let Matplotlib auto-scale to the current theta
    axes[1].legend()

    # 3. Position Reward
    axes[2].set_title("Position Reward")
    im = axes[2].imshow(pos_reward, origin="lower", extent=[xs[0], xs[-1], ys[0], ys[-1]],
                        cmap="viridis", vmin=pos_lims[0], vmax=pos_lims[1])  # <-- Fixed limits applied here
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    axes[2].plot(0, 0, marker="*", color="red", markersize=16, zorder=5, label="goal")
    axes[2].legend()
    fig.colorbar(im, ax=axes[2])

    plt.tight_layout()

    frame_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.png")
    plt.savefig(frame_path, dpi=125)
    plt.close(fig)  # Critical to prevent memory leaks

    return frame_path


if __name__ == "__main__":
    boltzmann_encoder = BoltzmannDiscriminatorFeatureBased()

    experiment_id = "1785933526"
    target_exp_dir = f'runs/curriculum_rl_eval_based/pusht_ppo_fb/{experiment_id}'

    checkpoint_path = os.path.join(target_exp_dir, "models/best.model.zip")
    theta_matrix_path = os.path.join(target_exp_dir, "models/theta_matrix.npy")
    frames_dir = "temp_video_frames"
    os.makedirs(frames_dir, exist_ok=True)

    print("Loading networks and thetas...")
    theta_best, disc_params = load_checkpoint(checkpoint_path)
    thetas = np.load(theta_matrix_path)

    # Calculate limits for Position only
    pos_lims = get_best_model_limits(theta_best, disc_params, boltzmann_encoder)
    print(f"Calculated Best Limits -> Pos: {pos_lims}")

    num_frames = thetas.shape[0]
    frame_files = []

    print(f"Generating {num_frames} frames...")
    for i in range(num_frames):
        theta = thetas[i]
        # Passed only pos_lims to the plotting function
        path = plot_rewards_frame(theta, disc_params, boltzmann_encoder, i, frames_dir, pos_lims)
        frame_files.append(path)
        if i % 10 == 0:
            print(f"Processed {i}/{num_frames} frames")

    print("Compiling video...")
    output_video_path = f"plots/reward_evolution_{experiment_id}.mp4"

    with imageio.get_writer(output_video_path, fps=15) as writer:
        for filename in frame_files:
            image = imageio.imread(filename)
            writer.append_data(image)

    print(f"Video saved successfully to {output_video_path}!")
    shutil.rmtree(frames_dir)