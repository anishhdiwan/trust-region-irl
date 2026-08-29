python experiment.py \
  --algorithm.name="ppo_retraining" \
  --algorithm.data_path="../trirl_dataset/rl_expert/expert_dataset_pusht_mtp_clean_93_episodes_trirl_f32abs.npz" \
  --environment.name="pusht_mjx" \
  --environment.nr_envs=1 \
  --environment.seed=0 \
  --environment.feature_fn="base_without_ctrl" \
  --environment.render=True \
  --runner.mode="test" \
  --runner.load_model="runs/PPO_RETRAINING_EXP/pusht_ppo_retraining/1787980369_ppo_retraining/models/latest.model.zip" \
  --runner.track_tb=False \
  --runner.track_wandb=False \
  --runner.save_model=False \
  --runner.track_console=True

#