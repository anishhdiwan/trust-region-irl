#!/bin/bash
#SBATCH --job-name=trirl_ppo
#SBATCH -t 08:00:00
#SBATCH --array=0-9
#SBATCH -c 1
#SBATCH --mem-per-cpu=6G
#SBATCH -p stud
#SBATCH --gres=gpu:1
#SBATCH -C 'rtx2080|rtx3080|v100|rtx3090|a5000'
#SBATCH --output=logs/output_%A_%a.txt
#SBATCH --error=logs/error_%A_%a.txt

mkdir -p logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate trirl

# TRIRL PPO-FB to solve the Push-T task
python experiment.py \
   --algorithm.name="trirl_ppo_fb.flax_full_jit" \
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
   --environment.seed=$SLURM_ARRAY_TASK_ID \
   --environment.feature_fn="pos_ort" \
   --environment.random_TPos=True \
   --runner.mode="train" \
   --runner.track_tb=True \
   --runner.track_console=True \
   --runner.track_wandb=True \
   --runner.save_model=True \
   --runner.wandb_entity="s-mohddheia-tu-darmstadt" \
   --runner.project_name="ICRA_TRIRL_6_2F" \
   --runner.exp_name="pusht_ppo_fb"
