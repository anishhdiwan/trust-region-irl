import os
import sys
import json
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import shutil
import tempfile
import numpy as np
import jax.numpy as jnp
import orbax.checkpoint as ocp
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trust_region_irl.algorithms.trirl_ppo_fb.flax_full_jit.discriminator import BoltzmannDiscriminatorFeatureBased

def extract_features(observation):
    """
    Extract features from a 24D observation.
    :param observation: 24D Observation
    :return: 3D features
    """
    block_pos = observation[:, 0:3]
    w = observation[:, 3]
    ee_rel = observation[:, 7:10]
    pos_err = jnp.linalg.norm(block_pos, axis=-1)
    orient_err = 1.0 - jnp.clip(w * w, 0.0, 1.0)
    ee_block = jnp.linalg.norm(ee_rel - block_pos, axis=-1)
    features = jnp.stack([-pos_err, -orient_err, -ee_block], axis=-1)

    return features

def load_checkpoint(model_zip):
    tmp = tempfile.mkdtemp()
    try:
        shutil.unpack_archive(model_zip, tmp, "zip")
        ckpt = ocp.PyTreeCheckpointer().restore(tmp)
        with open(f"{tmp}/config_algorithm.json", "r") as f:
            config_algorithm = json.load(f)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    params = ckpt
    theta = np.asarray(ckpt["discriminator"]["params"]["params"]["theta"], dtype=np.float32)
    disc_params = ckpt["discriminator"]["params"]
    encoder = BoltzmannDiscriminatorFeatureBased(
        hidden_dims=config_algorithm["boltzmann_hidden_dims"],
        latent_dim=config_algorithm["boltzmann_latent_dim"],
        energy_hidden_dim=config_algorithm["boltzmann_energy_hidden_dim"],
    )
    return theta, disc_params, params, encoder

def plot_16features(encoded_features):
    fig, axes = plt.subplots(4, 4, figsize=(13, 10))
    for k, ax in enumerate(axes.ravel()):
        ax.hist(encoded_features[:, k], bins=30, color="#334155", edgecolor="white", linewidth=0.3)
        ax.set_title(f"z[{k}]  θ={theta[k]:.0f}", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("16 latent energy over 512 expert transitions")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig("energy16_histograms.png", dpi=125)

def plot_rewards(exp_id, boltzmann_encoder, disc_params, theta):
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

    # base = np.median(S, axis=0).astype(np.float32)
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
    axes[0].set_title("Reward with fixed pos and ee")
    axes[0].plot(qs, ort_reward, color="#e11d48", label="reward")
    axes[0].set_xlabel("orientation error")
    axes[0].set_ylabel("reward")
    axes[0].legend()

    # 2. End-Effector Reward
    # Recreating the xs range for EE since the variable was overwritten by the position grid
    xs_ee = np.linspace(0.0, 0.35, 100)
    axes[1].set_title("Reward with fixed position and orientation")
    axes[1].plot(xs_ee, ee_reward, color="#e11d48", label="reward")
    axes[1].set_xlabel("distance between ee and block")
    axes[1].set_ylabel("reward")
    axes[1].legend()

    # 3. Position Reward
    # xs and ys here refer to the last assignment (-0.35 to 0.35)
    axes[2].set_title("Reward with fixed orientation and ee")
    im = axes[2].imshow(pos_reward, origin="lower", extent=[xs[0], xs[-1], ys[0], ys[-1]], cmap="viridis")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    axes[2].plot(0, 0, marker="*", color="red", markersize=16, zorder=5, label="goal")
    axes[2].legend()
    fig.colorbar(im, ax=axes[2])

    plt.tight_layout()
    plt.savefig(f"plots/reward_{exp_id}.png", dpi=300)

# target_directory = 'runs/<PROJECT_NAME>/<EXPERIMENT_NAME>'
target_directory = 'runs/data_scaling_effect/pusht_ppo_fb'

exp_ids = [f.name for f in Path(target_directory).iterdir() if f.is_dir()]

for id in exp_ids:
    try:
        theta, disc_params, params, boltzmann_encoder = load_checkpoint(f"{target_directory}/{id}/models/best.model.zip")
        plot_rewards(id, boltzmann_encoder, disc_params, theta)
    except Exception as e:
        print(f"Skipping {id}, error: {e}")
        continue




