from ml_collections import config_dict


def get_config(environment_name):
    config = config_dict.ConfigDict()
    config.name = environment_name
    config.seed = 1
    config.nr_envs = 64
    config.render = False
    config.device = "gpu"
    config.copy_train_env_for_eval = True
    config.horizon = 800               # steps at 20 Hz (40 s): travel, touchdown, spiral of the largest pizza, lift-off
    config.success_threshold = 0.90    # sauce-area coverage counted as success (evaluation metric)
    config.end_on_success = False      # True: end the episode at success (False: fixed horizon, no termination bias)
    config.wander_margin = 0.0         # 0 = env default (m beyond the rim, after touchdown)
    config.wander_hold_s = 0.0         # 0 = env default
    config.impl = "jax"                # MJX backend: jax | warp (MuJoCo 3.6, see requirements.txt)
    config.contacts_per_env = 64       # warp only: contact slots per env in the shared contact pools
    config.ls_iterations = 0           # >0: override the solver linesearch iterations
    return config
