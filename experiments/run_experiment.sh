#!/bin/sh

# TRIRL max eta on Ant
#python experiment.py \
#    --algorithm.name="trirl_trpl.flax_full_jit" \
#    --algorithm.data_path="../trirl_dataset/rl_expert/Ant-v5_30_PPO.npz" \
#    --algorithm.total_timesteps=30e6 \
#    --environment.name="ant_mjx" \
#    --environment.nr_envs=4096 \
#    --environment.seed=0 \
#    --runner.mode="train" \
#    --runner.track_tb=True \
#    --runner.track_console=True \
#    --runner.track_wandb=False \
#    --runner.save_model=False \
#    --runner.wandb_entity="your-wandb-entity" \
#    --runner.project_name="trust_region_irl" \
#    --runner.exp_name="ant_trirl" \


# TRIRL PPO-FB to solve the Push-T task
python experiment.py \
   --algorithm.name="trirl_ppo_fb.flax_full_jit" \
   --algorithm.data_path="../trirl_dataset/rl_expert/expert_dataset_pusht_mtp_clean_93_episodes_trirl_f32abs_60_episodes.npz" \
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
   --algorithm.reward_type=boltzmann-feature-based \
   --algorithm.dsm_alpha=0.001 \
   --algorithm.dsm_sigma=0.5 \
   --algorithm.epsilon=0.2 \
   --algorithm.init_eta=30.0 \
   --algorithm.gp_lambda=0.5 \
   --algorithm.gae_lambda=0.95 \
   --algorithm.std_dev=0.4 \
   --algorithm.anneal_learning_rate=True \
   --algorithm.evaluation_and_save_frequency=2097152 \
   --algorithm.boltzmann_hidden_dims="16,32,64" \
   --environment.name="pusht_mjx" \
   --environment.nr_envs=4096 \
   --environment.seed=0 \
   --environment.feature_fn="base_without_ctrl" \
   --runner.mode="train" \
   --runner.track_tb=True \
   --runner.track_console=True \
   --runner.track_wandb=True \
   --runner.save_model=True \
   --runner.wandb_entity="s-mohddheia-tu-darmstadt" \
   --runner.project_name="data_scaling_effect" \
   --runner.exp_name="pusht_ppo_fb"
