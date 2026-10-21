#!/bin/sh

# TRIRL PPO-FB to solve the Push-T task
python experiment.py \
   --algorithm.name="trirl_ppo_fb.flax_full_jit" \
   --algorithm.data_path="../trirl_dataset/rl_expert/trirl_expert_data_100_episodes.npz" \
   --algorithm.total_timesteps=3e6 \
   --algorithm.entropy_coef=0.001 \
   --algorithm.clip_range=0.2 \
   --algorithm.env_reward_frac=0.0 \
   --algorithm.nr_steps=128 \
   --algorithm.nr_epochs=10 \
   --algorithm.nr_epochs_disc=10 \
   --algorithm.minibatch_size=512 \
   --algorithm.learning_rate_disc=1e-04 \
   --algorithm.learning_rate=4e-04 \
   --algorithm.reward_type=boltzmann-feature-based \
   --algorithm.dsm_alpha=0.001 \
   --algorithm.dsm_sigma=0.5 \
   --algorithm.epsilon=0.2 \
   --algorithm.init_eta=30.0 \
   --algorithm.gp_lambda=0.5 \
   --algorithm.gae_lambda=0.95 \
   --algorithm.std_dev=0.4 \
   --algorithm.anneal_learning_rate=False \
   --algorithm.evaluation_and_save_frequency=2097152 \
   --algorithm.boltzmann_hidden_dims="16,32,64" \
   --environment.name="pusht_mjx" \
   --environment.nr_envs=4096 \
   --environment.seed=0 \
   --environment.feature_fn="pos_ort" \
   --runner.mode="train" \
   --runner.track_tb=True \
   --runner.track_console=True \
   --runner.track_wandb=True \
   --runner.save_model=True \
   --runner.wandb_entity="s-mohddheia-tu-darmstadt" \
   --runner.project_name="without_annealing_disc" \
   --runner.exp_name="pusht_ppo_fb"
