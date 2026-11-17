
> #### `install.sh` executes the following commands automatically! Please try it first: `./install.sh [ENV_NAME]`

## Manual Installation Instructions

The environments run on MuJoCo / MJX 3.6, either on its JAX backend (`--environment.impl=jax`) or on the much faster
Warp backend (`--environment.impl=warp`). The pinned versions in `requirements.txt` use the CUDA 12 build of JAX
0.7.1, which runs on V100 and newer GPUs.

Create the environment and install the pinned stack
```bash
conda create -n trirl_warp36 python=3.11.4
conda activate trirl_warp36
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu   # TensorBoard logging only
```

Install RL-X next to this repository, without its dependency list (it would pull other MuJoCo and JAX versions)
```bash
git clone https://github.com/nico-bohlinger/RL-X.git && cd RL-X && git checkout 14379ff # clone a specific commit to ensure reproducibility
pip install -e . --no-deps --config-settings editable_mode=compat
```

Install this package
```bash
git clone https://github.com/anishhdiwan/trust-region-irl
cd trust-region-irl
pip install -e . --no-deps
```

Check that MuJoCo, JAX and Warp work and JAX sees a GPU
```bash
python -c "import mujoco, jax, warp; print(mujoco.__version__, jax.devices(), warp.config.version)"
```

The paper's original environment (MuJoCo 3.2 with LocoMujoco for the G1 and Go2 tasks) is described in this file at
the `icml26` tag.

#### Dataset

We host the dataset on **HuggingFace [https://huggingface.co/datasets/anishdiwan/trirl_dataset](https://huggingface.co/datasets/anishdiwan/trirl_dataset)**.
```bash
# Make sure git-xet is installed (https://hf.co/docs/hub/git-xet)
curl -sSfL https://hf.co/git-xet/install.sh | sh
git clone https://huggingface.co/datasets/anishdiwan/trirl_dataset
```

The demonstrations for the pizza and ball-balancing tasks are generated with the scripts in `scripts/pizza` and
`scripts/balance` and saved into `trirl_dataset/rl_expert`; see [usage.md](usage.md).
