from trust_region_irl.environments.pizza_dr_mjx.environment import PizzaSpreadDR
from trust_region_irl.environments.pizza_dr_mjx.general_properties import GeneralProperties


def _make(config):
    e = config.environment
    env = PizzaSpreadDR(render=e.render, horizon=e.horizon, success_threshold=e.success_threshold, end_on_success=e.end_on_success, wander_margin=e.wander_margin or None,
                        wander_hold_s=e.wander_hold_s or None, impl=e.impl, nr_envs=e.nr_envs, contacts_per_env=e.contacts_per_env,
                        ls_iterations=e.ls_iterations or None)
    env.general_properties = GeneralProperties
    return env


def create_train_and_eval_env(config):
    train_env = _make(config)
    if config.environment.copy_train_env_for_eval:
        return train_env, train_env
    return train_env, _make(config)
