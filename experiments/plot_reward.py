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
    # encoder = BoltzmannDiscriminatorFeatureBased(
    #     hidden_dims=config_algorithm["boltzmann_hidden_dims"],
    #     latent_dim=config_algorithm["boltzmann_latent_dim"],
    #     energy_hidden_dim=config_algorithm["boltzmann_energy_hidden_dim"],
    # )

    encoder = BoltzmannDiscriminatorFeatureBased()
    return theta, disc_params, params, encoder

def normalize_01(x):
    x_min = jnp.min(x)
    x_max = jnp.max(x)
    return (x - x_min) / (x_max - x_min + 1e-8)


def global_normalize(*rewards):
    """
    Takes any number of reward arrays, finds the global min and max across all of them,
    and normalizes them all to a 0-1 scale based on those global bounds.
    """
    # Find global min and max across all provided reward arrays
    global_min = float(min([jnp.min(r) for r in rewards]))
    global_max = float(max([jnp.max(r) for r in rewards]))

    # Normalize and return them in the same order
    normalized = tuple((r - global_min) / (global_max - global_min + 1e-8) for r in rewards)

    return normalized

def plot_16features(encoded_features):
    fig, axes = plt.subplots(4, 4, figsize=(13, 10))
    for k, ax in enumerate(axes.ravel()):
        ax.hist(encoded_features[:, k], bins=30, color="#334155", edgecolor="white", linewidth=0.3)
        ax.set_title(f"z[{k}]  θ={theta[k]:.0f}", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("16 latent energy over 512 expert transitions")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig("energy16_histograms.png", dpi=125)

def plot_rewards(exp_id, boltzmann_encoder, disc_params, theta, features_dim=3):
    qs = np.linspace(-np.pi, np.pi, 100).reshape(-1, 1)
    w = np.cos(qs / 2.0)
    orient_feat = -(1.0 - np.clip(w * w, 0.0, 1.0))

    features = jnp.hstack([np.zeros((100, 1)), orient_feat, np.zeros((100, 1))]) if features_dim == 3 else jnp.hstack([np.zeros((100, 1)), orient_feat])

    encoded_features = boltzmann_encoder.apply(disc_params, features,
                                               method=BoltzmannDiscriminatorFeatureBased.encode_feature)
    ort_reward = jnp.array(encoded_features @ theta)
    ort_reward = normalize_01(ort_reward)

    xs = np.linspace(0.0, 0.4, 100).reshape(-1, 1)

    features = jnp.hstack([jnp.zeros((100, 1)), xs, np.zeros((100, 1))])

    pos_err = features[:, 0]
    orient_err = features[:, 1]
    ee_block = features[:, 2]

    features = jnp.stack([-pos_err, -orient_err, -ee_block], axis=-1) if features_dim == 3 else jnp.stack([pos_err, orient_err], axis=-1)

    encoded_features = boltzmann_encoder.apply(disc_params, features,
                                               method=BoltzmannDiscriminatorFeatureBased.encode_feature)

    ee_reward = jnp.array(encoded_features @ theta)
    ee_reward = normalize_01(ee_reward)

    xs = np.linspace(-0.4, 0.4, 100)
    ys = np.linspace(-0.4, 0.4, 100)

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

    features = jnp.stack([-pos_err, -orient_err, -ee_block], axis=-1) if features_dim == 3 else jnp.stack([pos_err, orient_err], axis=-1)

    encoded_features = boltzmann_encoder.apply(disc_params, features,
                                               method=BoltzmannDiscriminatorFeatureBased.encode_feature)

    pos_reward = jnp.array(encoded_features @ theta).reshape(100, 100)
    pos_reward = normalize_01(pos_reward)
    # ---------------------------------------------
    # CREATE THE LAYOUT
    # ---------------------------------------------
    if features_dim == 3:
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2)  # 2 rows, 2 columns

        ax_ort = fig.add_subplot(gs[0, 0])  # Row 0, Col 0 (Top Left)
        ax_ee = fig.add_subplot(gs[1, 0])  # Row 1, Col 0 (Bottom Left)
        ax_pos = fig.add_subplot(gs[:, 1])  # Both Rows, Col 1 (Entire Right Side)
    else:
        fig = plt.figure(figsize=(14, 6))
        gs = fig.add_gridspec(1, 2)  # 1 row, 2 columns

        ax_ort = fig.add_subplot(gs[0, 0])  # Row 0, Col 0 (Left)
        ax_pos = fig.add_subplot(gs[0, 1])  # Row 0, Col 1 (Right)

    # ---------------------------------------------
    # 1. Orientation Reward (Top Left)
    # ---------------------------------------------
    ax_ort.set_title("Orientation Reward")
    ax_ort.plot(qs, ort_reward, color="#e11d48")
    ax_ort.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax_ort.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])

    # ---------------------------------------------
    # 2. End-Effector Reward (Bottom Left)
    # ---------------------------------------------
    if features_dim == 3:
        xs_ee = np.linspace(0.0, 0.4, 100)
        ax_ee.set_title("End-Effector Reward")
        ax_ee.plot(xs_ee, ee_reward, color="#e11d48")
        ax_ee.set_xticks(np.linspace(0.0, 0.4, 5))

    # ---------------------------------------------
    # 3. Position Reward (Right Side)
    # ---------------------------------------------
    ax_pos.set_title("Position Reward")
    im = ax_pos.imshow(pos_reward, origin="lower", extent=[xs[0], xs[-1], ys[0], ys[-1]], cmap="Blues")
    ax_pos.plot(0, 0, marker="*", color="red", markersize=10, linestyle="None", zorder=5, label="goal")
    ax_pos.legend()
    fig.colorbar(im, ax=ax_pos, ticks=[0, 1], fraction=0.046, pad=0.04)
    ax_pos.set_xticks(np.linspace(-0.4, 0.4, 5))
    ax_pos.set_yticks(np.linspace(-0.4, 0.4, 5))

    plt.rcParams.update({'font.size': 18})
    plt.tight_layout()
    plt.savefig(f"plots/reward_{exp_id}.png", dpi=300)
    plt.close(fig)  # Good practice to close the figure after saving to free memory!


# target_directory = 'runs/<PROJECT_NAME>/<EXPERIMENT_NAME>'
target_directory = 'runs/without_annealing_disc/pusht_ppo_fb'
features_dim = 2

exp_ids = [f.name for f in Path(target_directory).iterdir() if f.is_dir()]

for id in exp_ids:
    try:
        theta, disc_params, params, boltzmann_encoder = load_checkpoint(f"{target_directory}/{id}/models/latest.model.zip")
        plot_rewards(id, boltzmann_encoder, disc_params, theta, features_dim)
    except Exception as e:
        print(f"Skipping {id}, error: {e}")
        continue




