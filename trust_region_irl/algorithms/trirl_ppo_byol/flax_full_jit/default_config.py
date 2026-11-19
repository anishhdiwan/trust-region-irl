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

    # BYOL Configuration
    config.byol_input_type = "state"  # "state" | "image"
    config.byol_input = "obs"  # "obs": encoder sees the raw observation | "features": encoder sees env.feature_from_transition(s, a) (set environment.feature_fn)
    config.byol_hidden_dims = (64, 64)
    config.byol_latent_dim = 128
    config.byol_proj_hidden_dim = 256
    config.byol_proj_dim = 128
    config.learning_rate_byol = 1e-3
    config.byol_tau = 0.99
    config.byol_noise_scale = 0.05
    config.augmentation = True  # False: no augmentation, both BYOL views are the unchanged input (byol_noise_scale is ignored)
    config.nr_epochs_byol = 1  # number of full-rollout-batch BYOL steps per TRIRL iteration (online step + EMA each)
    config.byol_pretrain = False  # True: pretrain BYOL on expert + random-policy states before TRIRL, then keep training online
    config.byol_pretrain_steps = 1000  # number of BYOL gradient steps in the pretraining phase
    config.byol_pretrain_log_frequency = 100  # log pretraining metrics every N steps (every N rollout rounds if byol_pretrain_env_steps > 0)
    config.byol_pretrain_env_steps = 0  # 0: pretrain on ONE rollout for byol_pretrain_steps steps. >0: stream ceil(env_steps / (nr_envs * nr_steps)) fresh rollouts
    config.byol_pretrain_steps_per_rollout = 1024  # BYOL steps per streamed rollout (only used if byol_pretrain_env_steps > 0)

    config.handle_absorbing_states = True
    config.reward_type = 'feature-based'
    config.retraining = False
    config.curriculum_learning = False

    return config
