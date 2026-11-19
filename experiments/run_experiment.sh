#!/bin/sh

# TRIRL PPO-FB to solve the Push-T task
# TRIRL PPO-BYOL to solve the Push-T task
python experiment.py \
   --algorithm.name="trirl_ppo_byol_s.flax_full_jit" \
   --algorithm.data_path="../trirl_dataset/rl_expert/trirl_expert_data_100_episodes.npz" \
   --algorithm.total_timesteps=150e6 \
   --algorithm.entropy_coef=0.001 \
   --algorithm.clip_range=0.2 \
   --algorithm.env_reward_frac=0.0 \
   --algorithm.nr_steps=128 \
   --algorithm.nr_epochs=10 \
   --algorithm.nr_epochs_disc=10 \
   --algorithm.minibatch_size=512 \
   --algorithm.learning_rate_disc=1e-04 \
   --algorithm.learning_rate=4e-04 \
   --algorithm.reward_type="feature-based" \
   --algorithm.byol_input_type="state" \
   --algorithm.byol_latent_dim=128 \
   --algorithm.byol_proj_hidden_dim=256 \
   --algorithm.byol_proj_dim=128 \
   --algorithm.learning_rate_byol=1e-03 \
   --algorithm.byol_tau=0.99 \
   --algorithm.byol_noise_scale=0.05 \
   --algorithm.nr_epochs_byol=1 \
   --algorithm.epsilon=0.2 \
   --algorithm.init_eta=30.0 \
   --algorithm.gp_lambda=0.5 \
   --algorithm.gae_lambda=0.95 \
   --algorithm.std_dev=0.4 \
   --algorithm.anneal_learning_rate=True \
   --algorithm.evaluation_and_save_frequency=2097152 \
   --environment.name="pusht_mjx" \
   --environment.random_TPos=True \
   --environment.nr_envs=4096 \
   --environment.seed=0 \
   --environment.feature_fn="base_without_ctrl" \
   --runner.mode="train" \
   --runner.track_tb=True \
   --runner.track_console=True \
   --runner.track_wandb=False \
   --runner.save_model=False \
   --runner.wandb_entity="s-mohddheia-tu-darmstadt" \
   --runner.project_name="garbage_data" \
   --runner.exp_name="pusht_ppo_byol_seed0"