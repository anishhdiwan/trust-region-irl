from ml_collections import config_dict


def get_config(environment_name):
    config = config_dict.ConfigDict()

    config.name = environment_name

    config.seed = 1
    config.nr_envs = 64
    config.render = False
    config.device = "gpu"
    config.copy_train_env_for_eval = True
    config.feature_fn = "base"  # IRL feature basis: base | base_rbf | base_without_ctrl | pos_ort | state_action
    config.random_TPos = False
    config.impl = "jax"              # MJX backend: jax | warp (warp needs the trirl_warp36 conda env: MuJoCo 3.6, CUDA 12)
    config.contacts_per_env = 64     # warp only: contact slots per env in the shared contact pools (raise for contact-heavy tasks)
    config.ls_iterations = 0         # >0: override the solver linesearch iterations (0 = keep the XML value)
    config.success_threshold = 0.05  # Success: position error AND orientation error below this (used when the curriculum is off)
    config.curriculum_success = False  # The success threshold changes over the course of training (trirl_ppo_spr)
    config.success_threshold_values = "0.5,0.25,0.1,0.05" # Success thresholds over the course of training
    config.success_threshold_fractions = "0.2,0.2,0.2,0.4" # How long should each threshold from the training steps
    """
    Example usage for success threshold values and fractions:
    In the default values, 0.5 will be the success threshold for the first 20% of training steps and 0.05 for the last 40% of training steps.
    """

    return config
