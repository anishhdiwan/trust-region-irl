<img align="right" src="figs/ICML-logo.svg" height="45" alt="ICML 2026">
<span>
  <img src="figs/ias_logo_dark.svg" height="32" alt="Intelligent Autonomous Systems Group">&nbsp;&nbsp;
  <img src="figs/tuda_logo_dark.svg" height="32" alt="Technical University of Darmstadt">&nbsp;&nbsp;
  <img src="figs/lund_logo_dark.svg" height="42" alt="Lund University">&nbsp;&nbsp;
  <img src="figs/huawei_logo_dark.svg" height="40" alt="Huawei Noah's Ark Lab">&nbsp;&nbsp;
  <img src="figs/hessian_AI_logo_dark.svg" height="24" alt="hessian.AI">&nbsp;&nbsp;
  <img src="figs/dfki_logo_dark.svg" height="28" alt="DFKI">&nbsp;&nbsp;
  <img src="figs/rig_logo_dark.svg" height="36" alt="Robotics Institute Germany">
</span>
<br clear="all">

[[`Webpage`](https://anishhdiwan.github.io/trust-region-irl/)] [[`Paper`](https://arxiv.org/pdf/2605.11020)] <br>
Official code release for our **ICML 2026** paper :page_facing_up:<br>
<big><strong>Trust Region Inverse Reinforcement Learning: Explicit Dual Ascent using Local Policy Updates</strong></big><br>
<em><a href="https://anishhdiwan.github.io/">Anish Diwan</a>, <a href="https://www.ias.informatik.tu-darmstadt.de/Team/DavideTateo">Davide Tateo</a>, <a href="https://cmower.github.io/">Christopher E. Mower</a>, <a href="http://bouammar.com/blog/">Haitham Bou-Ammar</a>, <a href="https://www.ias.tu-darmstadt.de/Team/JanPeters">Jan Peters</a>, <a href="https://www.ias.informatik.tu-darmstadt.de/Team/OlegArenz">Oleg Arenz</a></em>

<strong>TL;DR</strong>&ensp;We present an inverse RL method that explicitly optimizes the IRL Lagrangian and it's dual using local trust-region policy updates and a reward correction step. Our key theoretical insight is that a trust-region-optimal policy for a reward update can be globally optimal for a smaller update in the same direction. Our method yields monotonic performance improvement and can learn global reward functions.

<p align="center">
  <img src="figs/correction_illustration.gif" alt="An illustration of TRIRL." width="500">
  <br>
  <em>TRIRL uses cheap, trust-region policy updates and corrects the reward to account for this local policy optimization.</em>
</p>

> **Note:** This official codebase contains the main method, baselines, and all options used in the paper. This version is intended for reproducibility. A bare-bones version of the algorithm that will be maintained in the long run can be found at [https://github.com/nico-bohlinger/RL-X](https://github.com/nico-bohlinger/RL-X). 

### Install
The following command creates a conda environment and installs this package, the dataset, and its dependencies. For manual installation instructions, please follow [docs/install.md](docs/install.md). We host the dataset on **HuggingFace @ [https://huggingface.co/datasets/anishdiwan/trirl_dataset](https://huggingface.co/datasets/anishdiwan/trirl_dataset)**. 
```bash
./install.sh
```

### Training
<p align="center">
  <img src="figs/g1_walk.gif" width="12%">&emsp;<img src="figs/g1_run.gif" width="12%">&emsp;<img src="figs/go2.gif" width="12%">
</p>

```bash
cd trust-region-irl/experiments
./run_experiment.sh
```

**Generic Command for Any Algo/Env**
```bash
python experiment.py \
  --algorithm.name="<algorithm>.<implementation>" \
  --algorithm.total_timesteps=xe6 \
  --environment.name="<environment>" \
  --environment.nr_envs=4096 \
  --environment.seed=0 \
  --runner.mode="train" \

  # optional
  --runner.wandb_entity="<wandb>" \
  --runner.project_name="<project>" \
  --runner.exp_name="<exp>" \
```

- **Supported algorithms:** `[trirl_ppo, trirl_trpl, trirl_trpl_fb, gail_ppo, airl_ppo, amp_ppo, near_ppo, lsiq_sac]`. By default pass `flax_full_jit` as the implementation.
  - *Locomujoco implementation*: for robotics, we use the Locomujoco library. Pass `flax_loco_mjx` as the implementation. 
- **Supported environments:** `[half_cheetah_mjx, ant_mjx, walker_mjx, hopper_mjx, humanoid_mjx, loco_mjx, pushT_franka]`
  - *Robotics environments:* robotics envs can be accessed by passing `loco_mjx` with the following `[--environment.agent="MjxUnitreeG1" --environment.task="run/walk" , --environment.agent="MjxUnitreeGo2" --environment.task="rl"]`
- The full list of config options of each algorithm is stored in `trust-region-irl/algorithms/<alg_name>/flax_full_jit/default_config.py`



### Viewing Results

**View a model learned by IRL** (e.g. TRIRL-PPO-FB) by rendering it in the Push-T viewer. Point `--runner.load_model` at the checkpoint you want to inspect:
```bash
cd trust-region-irl/experiments
python experiment.py \
  --algorithm.name="trirl_ppo_fb.flax_full_jit" \
  --algorithm.data_path="../trirl_dataset/rl_expert/expert_dataset_pusht_mtp_clean_93_episodes_trirl_f32abs.npz" \
  --algorithm.boltzmann_hidden_dims="16,32,64" \
  --environment.name="pusht_mjx" \
  --environment.nr_envs=1 \
  --environment.seed=0 \
  --environment.feature_fn="base_without_ctrl" \
  --environment.render=True \
  --runner.mode="test" \
  --runner.load_model="<path/to/best.model.zip>" \
  --runner.track_tb=False \
  --runner.track_wandb=False \
  --runner.save_model=False \
  --runner.track_console=True
```
or just run `./view_model.sh` which already points at a saved checkpoint.

**View a model learned by PPO** (retraining on the recovered/frozen reward) the same way, but with `--algorithm.name="ppo_retraining"`:
```bash
python experiment.py \
  --algorithm.name="ppo_retraining" \
  --algorithm.data_path="../trirl_dataset/rl_expert/expert_dataset_pusht_mtp_clean_93_episodes_trirl_f32abs.npz" \
  --environment.name="pusht_mjx" \
  --environment.nr_envs=1 \
  --environment.seed=0 \
  --environment.feature_fn="base_without_ctrl" \
  --environment.render=True \
  --runner.mode="test" \
  --runner.load_model="<path/to/best.model.zip>" \
  --runner.track_tb=False \
  --runner.track_wandb=False \
  --runner.save_model=False \
  --runner.track_console=True
```
or `./vis_ppo.sh`.

**View expert data** (real-robot Push-T episodes) in the MuJoCo viewer:
```bash
python vis_expert_data.py --data-dir="../trirl_dataset/rl_expert/expert_data" --episodes 11
```
Drop `--episodes` to play every episode in the folder, in order, or `./vis_expert_data.sh` for the same thing.

**Plot the learned reward** (feature-based Boltzmann discriminator) for every checkpoint in a run directory. Edit `target_directory` at the bottom of `plot_reward.py` to point at your run folder, then:
```bash
python plot_reward.py
```
This loops over every experiment subfolder, loads `models/best.model.zip`, and saves a 3-panel reward plot (orientation, end-effector distance, position heatmap) to `plots/reward_<exp_id>.png`.

**Plot how the reward evolves over training** as a video. Edit `experiment_id` / `target_exp_dir` at the bottom of `plot_video.py` to point at a run with a saved `models/theta_matrix.npy` (the theta history), then:
```bash
python plot_video.py
```
This renders one reward-plot frame per saved theta and stitches them into `plots/reward_evolution_<experiment_id>.mp4`.

### Discrete Setting
`trust_region_irl_discrete` contains a numpy-only gridworld implementation of TRIRL. It can be used to reproduce Figure 2 in the paper.

```bash
cd trust_region_irl_discrete
python experiment.py # generates .pdf plots
```

<p align="center">
  <img src="figs/trirl_discrete.svg" width="20%">
</p>

### Cite

```bibtex
@inproceedings{diwan2026trirl,
  title     = {Trust Region Inverse Reinforcement Learning: Explicit Dual Ascent using Local Policy Updates},
  author    = {Diwan, Anish and Tateo, Davide and Mower, Christopher E. and Bou-Ammar, Haitham and Peters, Jan and Arenz, Oleg},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026},
  url={https://openreview.net/pdf?id=XSYX75R6RC}
}
```