from trust_region_irl.environments.pusht_mjx.environment import PushT
from trust_region_irl.environments.pusht_mjx.general_properties import GeneralProperties


def initial_success_threshold(env_config):
    """The threshold envs start with: the first curriculum value if the curriculum is on, else success_threshold."""
    if env_config.curriculum_success:
        return float(env_config.success_threshold_values.split(",")[0])
    return env_config.success_threshold


def create_train_and_eval_env(config):
    success_threshold = initial_success_threshold(config.environment)

    train_env = PushT(render=config.environment.render, feature_fn=config.environment.feature_fn, random_TPos=config.environment.random_TPos,
                      success_threshold=success_threshold,
                      impl=getattr(config.environment, "impl", "jax"), nr_envs=config.environment.nr_envs,
                      contacts_per_env=getattr(config.environment, "contacts_per_env", 64),
                      ls_iterations=getattr(config.environment, "ls_iterations", 0) or None)
    train_env.general_properties = GeneralProperties

    if config.environment.copy_train_env_for_eval:
        return train_env, train_env

    eval_env = PushT(render=config.environment.render, feature_fn=config.environment.feature_fn, random_TPos=config.environment.random_TPos,
                     success_threshold=success_threshold,
                     impl=getattr(config.environment, "impl", "jax"), nr_envs=config.environment.nr_envs,
                     contacts_per_env=getattr(config.environment, "contacts_per_env", 64),
                     ls_iterations=getattr(config.environment, "ls_iterations", 0) or None)
    eval_env.general_properties = GeneralProperties

    return train_env, eval_env
