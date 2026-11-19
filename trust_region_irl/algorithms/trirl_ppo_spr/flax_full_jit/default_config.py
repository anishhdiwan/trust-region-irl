from ml_collections import config_dict


def get_config(algorithm_name):
    config = config_dict.ConfigDict()

    config.name = algorithm_name

    config.device = "gpu"  # cpu, gpu
    config.nr_parallel_seeds = 1
    config.total_timesteps = 2e9
    config.learning_rate = 4e-4
    config.anneal_learning_rate = True
    config.nr_steps = 128
    config.nr_epochs = 10
    config.minibatch_size = 32768
    config.gamma = 0.99
    config.gae_lambda = 0.9
    config.clip_range = 0.1
    config.entropy_coef = 0.001
    config.critic_coef = 1.0
    config.max_grad_norm = 5.0
    config.std_dev = 1.0
    config.action_clipping_and_rescaling = False
    config.evaluation_and_save_frequency = -1  # -1 to disable
    config.evaluation_active = True

    # TRIRL Params
    config.nr_hidden_units_disc = 256
    config.learning_rate_disc = 0.0002015531860350999
    config.nr_epochs_disc = 30
    config.env_reward_frac = 0.0
    config.data_path = "../trirl_dataset/rl_expert/Ant-v5_30_PPO.npz"

    config.epsilon = 0.693139308915975
    config.init_eta = 100.0
    config.const_eta = False
    config.beta = float(1/config.entropy_coef)
    config.gp_lambda = 0.03267622091691947
    config.gp_alpha = 0.5

    # BYOL Params
    config.target_momentum = 0.99 # For EMA
    config.pretrain_env_steps = 50e6 # Environment steps of encoder-only pre-training before TRIRL (not counted in total_timesteps)
    config.warmup_env_steps = 0 # TRIRL steps during which the discriminator loss does not train the encoder
    config.w_disc_loss = 1.0
    config.prediction_steps = 1

    config.handle_absorbing_states = True
    config.reward_type = 'byol-observation-based'
    config.retraining = False
    config.curriculum_learning = False

    return config
