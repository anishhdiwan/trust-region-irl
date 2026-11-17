from ml_collections import config_dict


def get_config(environment_name):
    config = config_dict.ConfigDict()
    config.name = environment_name
    config.seed = 1
    config.nr_envs = 64
    config.render = False
    config.device = "gpu"
    config.copy_train_env_for_eval = True
    config.horizon = 200               # steps at 20 Hz (10 s), as the BaMPC demos; fixed, no early termination
    config.impl = "jax"                # MJX backend: jax | warp (MuJoCo 3.6, see requirements_warp.txt)
    config.contacts_per_env = 64       # warp only: contact slots per env in the shared contact pools
    return config
