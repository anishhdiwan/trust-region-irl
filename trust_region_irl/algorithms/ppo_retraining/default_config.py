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

    config.dsm_alpha = 1.0
    config.dsm_sigma = 0.10
    config.feature_var_target = 0.25
    config.feature_var_weight = 0.001

    # Boltzmann feature-encoder architecture (used when reward_type='boltzmann-feature-based').
    # hidden_dims: comma-separated encoder layer widths; latent_dim: encoder output (= reward theta dim).
    config.boltzmann_hidden_dims = "4,8"
    config.boltzmann_latent_dim = 16
    config.boltzmann_energy_hidden_dim = 32
    # point maze
    # dsm_alpha = 0.5
    # dsm_sigma = 0.25
    # feature_var_target = 1.0
    # feature_var_weight = 0.5

    config.handle_absorbing_states = True
    config.reward_type = 'feature-based' # options: feature-based, boltzmann-feature-based
    config.retraining = False

    return config
