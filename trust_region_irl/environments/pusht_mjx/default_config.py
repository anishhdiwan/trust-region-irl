from ml_collections import config_dict


def get_config(environment_name):
    config = config_dict.ConfigDict()

    config.name = environment_name

    config.seed = 1
    config.nr_envs = 64
    config.render = False
    config.device = "gpu"
    config.copy_train_env_for_eval = True
    config.feature_fn = "base"  # IRL feature basis: base | base_rbf | rbf | state_action
    config.block_type = "free"  # block physics: free (6-DOF) | 3dof (planar slide/slide/hinge)

    return config
