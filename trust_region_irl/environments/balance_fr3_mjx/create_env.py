from trust_region_irl.environments.balance_fr3_mjx.environment import BalanceFR3
from trust_region_irl.environments.balance_fr3_mjx.general_properties import GeneralProperties


def _make(config):
    e = config.environment
    env = BalanceFR3(render=e.render, horizon=e.horizon, impl=e.impl, nr_envs=e.nr_envs, contacts_per_env=e.contacts_per_env,
                     ball_mass_scale=e.ball_mass_scale, tilt_gain_scale=e.tilt_gain_scale, ik_k_pos_scale=e.ik_k_pos_scale,
                     ik_k_yaw_scale=e.ik_k_yaw_scale, ik_damping_scale=e.ik_damping_scale)
    env.general_properties = GeneralProperties
    return env


def create_train_and_eval_env(config):
    train_env = _make(config)
    if config.environment.copy_train_env_for_eval:
        return train_env, train_env
    return train_env, _make(config)
