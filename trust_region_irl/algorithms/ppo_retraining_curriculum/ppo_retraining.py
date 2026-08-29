import os
import shutil
import json
from copy import deepcopy
import logging
import time
import tree
import numpy as np
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from flax.training import orbax_utils
import orbax.checkpoint
import optax
import wandb

from rl_x.algorithms.ppo.flax_full_jit.general_properties import GeneralProperties
from rl_x.algorithms.ppo.flax_full_jit.policy import get_policy
from rl_x.algorithms.ppo.flax_full_jit.critic import get_critic

from trust_region_irl.algorithms.ppo_retraining import PPO_RETRAINING

rlx_logger = logging.getLogger("rl_x")

class Batch:
    def __init__(self, states, next_states, actions, rewards, values, terminations, log_probs, advantages, returns):
        self.states = states
        self.next_states = next_states
        self.actions = actions
        self.rewards = rewards
        self.values = values
        self.terminations = terminations
        self.log_probs = log_probs
        self.advantages = advantages
        self.returns = returns

class PPO_RETRAINING_CURRICULUM:
    def __init__(self, config, train_env, eval_env, run_path, writer, reward_function=None):
        self.config = config
        self.train_env = train_env
        self.eval_env = eval_env
        self.writer = writer

        self.save_model = config.runner.save_model
        self.save_path = os.path.join(run_path, "models")
        self.track_console = config.runner.track_console
        self.track_tb = config.runner.track_tb
        self.track_wandb = config.runner.track_wandb
        self.seed = config.environment.seed
        self.nr_parallel_seeds = config.algorithm.nr_parallel_seeds
        self.total_timesteps = config.algorithm.total_timesteps
        self.nr_envs = config.environment.nr_envs
        self.render = config.environment.render
        self.learning_rate = config.algorithm.learning_rate
        self.anneal_learning_rate = config.algorithm.anneal_learning_rate
        self.nr_steps = config.algorithm.nr_steps
        self.nr_epochs = config.algorithm.nr_epochs
        self.minibatch_size = config.algorithm.minibatch_size
        self.gamma = config.algorithm.gamma
        self.gae_lambda = config.algorithm.gae_lambda
        self.clip_range = config.algorithm.clip_range
        self.entropy_coef = config.algorithm.entropy_coef
        self.critic_coef = config.algorithm.critic_coef
        self.max_grad_norm = config.algorithm.max_grad_norm
        self.std_dev = config.algorithm.std_dev
        self.evaluation_and_save_frequency = config.algorithm.evaluation_and_save_frequency
        self.evaluation_active = config.algorithm.evaluation_active
        self.batch_size = config.environment.nr_envs * config.algorithm.nr_steps
        self.nr_updates = config.algorithm.total_timesteps // self.batch_size
        self.nr_minibatches = self.batch_size // self.minibatch_size
        if config.algorithm.evaluation_and_save_frequency == -1:
            self.evaluation_and_save_frequency = self.batch_size * (self.total_timesteps // self.batch_size)
        self.nr_multi_learning_and_eval_save_iterations = self.total_timesteps // self.evaluation_and_save_frequency
        self.nr_updates_per_multi_learning_iteration = self.evaluation_and_save_frequency // self.batch_size
        self.os_shape = self.train_env.single_observation_space.shape
        self.as_shape = self.train_env.single_action_space.shape
        self.horizon = self.train_env.horizon

        # Use the saved reward function
        self.reward_function = reward_function

        # Determine the correct directory to look for theta_matrix.npy
        if hasattr(config.runner, "load_model") and config.runner.load_model:
            # Look in the folder of the model you are loading
            model_folder = os.path.dirname(config.runner.load_model)
        else:
            # Fallback to the current run's save folder (e.g., run_path/models)
            model_folder = self.save_path

        theta_path = os.path.join(model_folder, "theta_matrix.npy")

        # Upload the thetas
        try:
            # Add jnp.array() around the loaded numpy array
            self.thetas = jnp.array(np.load(theta_path))
            self.nr_thetas = self.thetas.shape[0]
            rlx_logger.info(f"Loaded {self.nr_thetas} thetas from {theta_path} for Curriculum Learning.")
        except Exception as e:
            rlx_logger.error(f"Runtime Error: the thetas weren't found. Exception: {e}")
            raise FileNotFoundError(f"Missing theta matrix")

        self.nr_thetas = self.thetas.shape[0]
        rlx_logger.info(f"Loaded {self.nr_thetas} thetas from {theta_path} for Curriculum Learning.")

        if self.evaluation_and_save_frequency % self.batch_size != 0:
            raise ValueError("Evaluation and save frequency must be a multiple of batch size")

        if self.nr_parallel_seeds > 1:
            raise ValueError(
                "Parallel seeds are not supported yet. This is mainly limited by not being able to log mutliple wandb runs at the same time.")

        rlx_logger.info(f"Using device: {jax.default_backend()}")

        self.key = jax.random.PRNGKey(self.seed)
        self.key, policy_key, critic_key, reset_key = jax.random.split(self.key, 4)
        reset_key = jax.random.split(reset_key, 1)

        self.policy, self.get_processed_action = get_policy(self.config, self.train_env)
        self.critic = get_critic(self.config, self.train_env)

        def linear_schedule(count):
            fraction = 1.0 - (count // (self.nr_minibatches * self.nr_epochs)) / self.nr_updates
            return self.learning_rate * fraction

        learning_rate = linear_schedule if self.anneal_learning_rate else self.learning_rate

        env_state = self.train_env.reset(reset_key, False)

        self.policy_state = TrainState.create(
            apply_fn=self.policy.apply,
            params=self.policy.init(policy_key, env_state.next_observation),
            tx=optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.inject_hyperparams(optax.adam)(learning_rate=learning_rate),
            )
        )

        self.critic_state = TrainState.create(
            apply_fn=self.critic.apply,
            params=self.critic.init(critic_key, env_state.next_observation),
            tx=optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.inject_hyperparams(optax.adam)(learning_rate=learning_rate),
            )
        )

        if self.save_model:
            os.makedirs(self.save_path, exist_ok=True)
            self.latest_model_file_name = "latest.model"
            self.best_model_file_name = "best.model"
            self.best_eval_return = -np.inf
            self.latest_model_checkpointer = orbax.checkpoint.PyTreeCheckpointer()

    def train(self):
        def jitable_train_function(key, parallel_seed_id):
            key, reset_key = jax.random.split(key, 2)
            reset_keys = jax.random.split(reset_key, self.nr_envs)
            env_state = self.train_env.reset(reset_keys, False)

            policy_state = self.policy_state
            critic_state = self.critic_state

            def multi_learning_and_eval_save_iteration(multi_learning_and_eval_save_iteration_carry,
                                                       multi_learning_iteration_step):
                policy_state, critic_state, env_state, key = multi_learning_and_eval_save_iteration_carry

                def learning_iteration(learning_iteration_carry, learning_iteration_step):
                    policy_state, critic_state, env_state, key = learning_iteration_carry

                    current_global_iteration = (multi_learning_iteration_step * self.nr_updates_per_multi_learning_iteration) + learning_iteration_step
                    total_iterations = self.nr_multi_learning_and_eval_save_iterations * self.nr_updates_per_multi_learning_iteration

                    # 0.0 at the start of training, 1.0 at the end
                    progress_fraction = current_global_iteration / jnp.maximum(1, total_iterations - 1)

                    # Map fraction to an index in the theta matrix (e.g. 0 to N-1)
                    continuous_idx = progress_fraction * (self.nr_thetas - 1)

                    idx_low = jnp.floor(continuous_idx).astype(jnp.int32)
                    idx_high = jnp.ceil(continuous_idx).astype(jnp.int32)
                    alpha = continuous_idx - idx_low

                    # Linearly interpolate between the two nearest thetas
                    theta_low = self.thetas[idx_low]
                    theta_high = self.thetas[idx_high]
                    current_theta = (1.0 - alpha) * theta_low + alpha * theta_high

                    # Acting
                    def single_rollout(single_rollout_carry, _):
                        policy_state, critic_state, env_state, key = single_rollout_carry

                        key, subkey = jax.random.split(key)
                        observation = env_state.next_observation
                        action_mean, action_logstd = self.policy.apply(policy_state.params, observation)
                        action_std = jnp.exp(action_logstd)
                        action = action_mean + action_std * jax.random.normal(subkey, shape=action_mean.shape)
                        log_prob = (-0.5 * ((action - action_mean) / action_std) ** 2 - 0.5 * jnp.log(
                            2.0 * jnp.pi) - action_logstd).sum(1)
                        processed_action = self.get_processed_action(action)
                        value = self.critic.apply(critic_state.params, observation).squeeze(-1)
                        env_state = self.train_env.step(env_state, processed_action)
                        transition = (observation, env_state.actual_next_observation, action, env_state.reward, value,
                                      env_state.terminated, log_prob, env_state.info)

                        if self.render:
                            def render(env_state):
                                return self.train_env.render(env_state)

                            env_state = jax.experimental.io_callback(render, env_state, env_state)

                        return (policy_state, critic_state, env_state, key), transition

                    single_rollout_carry, batch = jax.lax.scan(single_rollout, learning_iteration_carry, None,
                                                               self.nr_steps)
                    policy_state, critic_state, env_state, key = single_rollout_carry
                    states, next_states, actions, rewards, values, terminations, log_probs, infos = batch

                    # Use learnt reward
                    rewards = self.reward_function(current_theta, states, actions, next_states, terminations, log_probs)

                    # Calculating advantages and returns
                    def calculate_gae_advantages(critic_state, next_states, rewards, values, terminations):
                        def compute_advantages(carry, t):
                            prev_advantage = carry[0]
                            advantage = delta[t] + self.gamma * self.gae_lambda * (1 - terminations[t]) * prev_advantage
                            return (advantage,), advantage

                        next_values = self.critic.apply(critic_state.params, next_states).squeeze(-1)
                        delta = rewards + self.gamma * next_values * (1.0 - terminations) - values
                        init_advantages = delta[-1]
                        _, advantages = jax.lax.scan(compute_advantages, (init_advantages,),
                                                     jnp.arange(self.nr_steps - 2, -1, -1))
                        advantages = jnp.concatenate([advantages[::-1], jnp.array([init_advantages])])
                        returns = advantages + values
                        return advantages, returns

                    advantages, returns = calculate_gae_advantages(critic_state, next_states, rewards, values,
                                                                   terminations)

                    # Optimizing
                    def loss_fn(policy_params, critic_params, state_b, action_b, log_prob_b, return_b, advantage_b):
                        # Policy loss
                        action_mean, action_logstd = self.policy.apply(policy_params, state_b)
                        action_std = jnp.exp(action_logstd)
                        new_log_prob = -0.5 * ((action_b - action_mean) / action_std) ** 2 - 0.5 * jnp.log(
                            2.0 * jnp.pi) - action_logstd
                        new_log_prob = new_log_prob.sum(1)
                        entropy = action_logstd + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e)
                        entropy = self.entropy_coef * entropy  # scaling to reduce critic loss (has no effect on solution since the same is also applied to the reward function)

                        logratio = new_log_prob - log_prob_b
                        ratio = jnp.exp(logratio)
                        approx_kl_div = (ratio - 1) - logratio
                        clip_fraction = jnp.float32((jnp.abs(ratio - 1) > self.clip_range))

                        pg_loss1 = -advantage_b * ratio
                        pg_loss2 = -advantage_b * jnp.clip(ratio, 1 - self.clip_range, 1 + self.clip_range)
                        pg_loss = jnp.maximum(pg_loss1, pg_loss2)

                        entropy_loss = entropy.sum(1)

                        # Critic loss
                        new_value = self.critic.apply(critic_params, state_b)
                        critic_loss = 0.5 * (new_value - return_b) ** 2

                        # Combine losses
                        loss = pg_loss - self.entropy_coef * entropy_loss + self.critic_coef * critic_loss

                        # Create metrics
                        metrics = {
                            "loss/policy_gradient_loss": pg_loss,
                            "loss/critic_loss": critic_loss,
                            "loss/entropy_loss": entropy_loss,
                            "policy_ratio/approx_kl": approx_kl_div,
                            "policy_ratio/clip_fraction": clip_fraction,
                        }

                        return loss, (metrics)

                    batch_states = states.reshape((-1,) + self.os_shape)
                    batch_actions = actions.reshape((-1,) + self.as_shape)
                    batch_advantages = advantages.reshape(-1)
                    batch_returns = returns.reshape(-1)
                    batch_log_probs = log_probs.reshape(-1)

                    vmap_loss_fn = jax.vmap(loss_fn, in_axes=(None, None, 0, 0, 0, 0, 0), out_axes=0)
                    safe_mean = lambda x: jnp.mean(x) if x is not None else x
                    mean_vmapped_loss_fn = lambda *a, **k: tree.map_structure(safe_mean, vmap_loss_fn(*a, **k))
                    grad_loss_fn = jax.value_and_grad(mean_vmapped_loss_fn, argnums=(0, 1), has_aux=True)

                    key, subkey = jax.random.split(key)
                    batch_indices = jnp.tile(jnp.arange(self.batch_size), (self.nr_epochs, 1))
                    batch_indices = jax.random.permutation(subkey, batch_indices, axis=1, independent=True)
                    batch_indices = batch_indices.reshape((self.nr_epochs * self.nr_minibatches, self.minibatch_size))

                    def minibatch_update(carry, minibatch_indices):
                        policy_state, critic_state = carry

                        minibatch_advantages = batch_advantages[minibatch_indices]
                        minibatch_advantages = (minibatch_advantages - jnp.mean(minibatch_advantages)) / (
                                    jnp.std(minibatch_advantages) + 1e-8)

                        (loss, (metrics)), (policy_gradients, critic_gradients) = grad_loss_fn(
                            policy_state.params,
                            critic_state.params,
                            batch_states[minibatch_indices],
                            batch_actions[minibatch_indices],
                            batch_log_probs[minibatch_indices],
                            batch_returns[minibatch_indices],
                            minibatch_advantages
                        )

                        policy_state = policy_state.apply_gradients(grads=policy_gradients)
                        critic_state = critic_state.apply_gradients(grads=critic_gradients)

                        metrics["gradients/policy_grad_norm"] = optax.global_norm(policy_gradients)
                        metrics["gradients/critic_grad_norm"] = optax.global_norm(critic_gradients)

                        carry = (policy_state, critic_state)

                        return carry, (metrics)

                    init_carry = (policy_state, critic_state)
                    carry, (optimization_metrics) = jax.lax.scan(minibatch_update, init_carry, batch_indices)
                    policy_state, critic_state = carry

                    optimization_metrics["lr/learning_rate"] = policy_state.opt_state[1].hyperparams["learning_rate"]
                    optimization_metrics["v_value/explained_variance"] = 1 - jnp.var(returns - values) / (
                                jnp.var(returns) + 1e-8)
                    optimization_metrics["policy/std_dev"] = jnp.mean(
                        jnp.exp(policy_state.params["params"]["policy_logstd"]))

                    optimization_metrics["curriculum/progress_fraction"] = progress_fraction
                    optimization_metrics["curriculum/theta_mean"] = jnp.mean(current_theta)

                    # Logging
                    combined_metrics = {**infos, **optimization_metrics}
                    combined_metrics = tree.map_structure(lambda x: jnp.mean(x), combined_metrics)

                    def callback(carry):
                        metrics, learning_iteration_step, combined_learning_iteration_step, parallel_seed_id = carry
                        current_time = time.time()
                        metrics["time/sps"] = int(
                            (self.nr_steps * self.nr_envs) / (current_time - self.last_time[parallel_seed_id]))
                        self.last_time[parallel_seed_id] = current_time
                        global_step = int(combined_learning_iteration_step.item() * self.nr_steps * self.nr_envs)
                        metrics["steps/nr_env_steps"] = global_step
                        metrics[
                            "steps/nr_updates"] = combined_learning_iteration_step.item() * self.nr_epochs * self.nr_minibatches
                        is_last_train_update_before_eval = self.evaluation_active and (
                                    learning_iteration_step + 1 == self.nr_updates_per_multi_learning_iteration)
                        self.start_logging(global_step)
                        for key, value in metrics.items():
                            self.log(f"{key}", np.asarray(value), global_step)
                        self.end_logging(wandb_commit=not is_last_train_update_before_eval)

                    combined_learning_iteration_step = (
                                                                   multi_learning_iteration_step * self.nr_updates_per_multi_learning_iteration) + learning_iteration_step + 1
                    jax.debug.callback(callback,
                                       (combined_metrics, learning_iteration_step, combined_learning_iteration_step,
                                        parallel_seed_id))

                    return (policy_state, critic_state, env_state, key), None

                key, subkey = jax.random.split(key)
                learning_iteration_carry, _ = jax.lax.scan(learning_iteration,
                                                           (policy_state, critic_state, env_state, subkey),
                                                           jnp.arange(self.nr_updates_per_multi_learning_iteration))
                policy_state, critic_state, env_state, key = learning_iteration_carry

                # Evaluating
                if self.evaluation_active:
                    def single_eval_rollout(single_eval_rollout_carry, _):
                        policy_state, eval_env_state = single_eval_rollout_carry

                        eval_action_mean, _ = self.policy.apply(policy_state.params, eval_env_state.next_observation)
                        eval_action = eval_action_mean
                        eval_processed_action = self.get_processed_action(eval_action)
                        eval_env_state = self.eval_env.step(eval_env_state, eval_processed_action)

                        return (policy_state, eval_env_state), None

                    key, reset_key = jax.random.split(key)
                    reset_keys = jax.random.split(reset_key, self.nr_envs)
                    eval_env_state = self.eval_env.reset(reset_keys, True)
                    single_eval_rollout_carry, _ = jax.lax.scan(single_eval_rollout, (policy_state, eval_env_state),
                                                                jnp.arange(self.horizon))
                    _, eval_env_state = single_eval_rollout_carry

                    eval_metrics = {
                        "eval/episode_return": jnp.mean(eval_env_state.info["rollout/episode_return"]),
                        "eval/episode_length": jnp.mean(eval_env_state.info["rollout/episode_length"]),
                    }

                    if "rollout/task_err" in eval_env_state.info:
                        eval_metrics["eval/task_err"] = jnp.mean(eval_env_state.info["rollout/task_err"])

                    def callback(metrics_and_global_step):
                        metrics, combined_learning_iteration_step = metrics_and_global_step
                        global_step = int(combined_learning_iteration_step.item() * self.nr_steps * self.nr_envs)
                        self.start_logging(global_step)
                        for key, value in metrics.items():
                            self.log(f"{key}", np.asarray(value), global_step)
                        self.end_logging()

                    combined_learning_iteration_step = (multi_learning_iteration_step + 1) * self.nr_updates_per_multi_learning_iteration
                    jax.debug.callback(callback, (eval_metrics, combined_learning_iteration_step))

                    # Saving
                    if self.save_model:
                        # Fetch episode_return from eval_metrics, fallback to -inf if not evaluating
                        eval_return_for_save = (eval_metrics["eval/episode_return"]
                                                if self.evaluation_active else jnp.asarray(-jnp.inf))

                        def save_with_check(policy_state, critic_state, eval_return):
                            self.save(policy_state, critic_state)  # always save latest.model

                            if self.evaluation_active:
                                current_return = float(np.asarray(eval_return).reshape(-1)[0])

                                # Check if the current return is GREATER than the best return
                                if current_return > self.best_eval_return:
                                    self.best_eval_return = current_return
                                    self.save(policy_state, critic_state, file_name=self.best_model_file_name)
                                    rlx_logger.info(
                                        f"[save-best] new best eval/episode_return={current_return:.4f} -> {self.best_model_file_name}")

                        jax.debug.callback(save_with_check, policy_state, critic_state, eval_return_for_save)

                return (policy_state, critic_state, env_state, key), None

            jax.lax.scan(multi_learning_and_eval_save_iteration, (policy_state, critic_state, env_state, key),
                         jnp.arange(self.nr_multi_learning_and_eval_save_iterations))

        self.key, subkey = jax.random.split(self.key)
        seed_keys = jax.random.split(subkey, self.nr_parallel_seeds)
        train_function = jax.jit(jax.vmap(jitable_train_function))
        self.last_time = [time.time() for _ in range(self.nr_parallel_seeds)]
        self.start_time = deepcopy(self.last_time)
        jax.block_until_ready(train_function(seed_keys, jnp.arange(self.nr_parallel_seeds)))
        rlx_logger.info(f"Average time: {max([time.time() - t for t in self.start_time]):.2f} s")

    def log(self, name, value, step):
        if self.track_tb:
            self.writer.add_scalar(name, value, step)
        if self.track_wandb:
            self.wandb_log_cache[name] = value
        if self.track_console:
            self.log_console(name, value)

    def log_console(self, name, value):
        value = np.format_float_positional(value, trim="-")
        rlx_logger.info(f"│ {name.ljust(30)}│ {str(value).ljust(14)[:14]} │", flush=False)

    def start_logging(self, step):
        if self.track_wandb:
            self.wandb_log_cache = {"global_step": int(step)}
        if self.track_console:
            rlx_logger.info("┌" + "─" * 31 + "┬" + "─" * 16 + "┐", flush=False)
        else:
            rlx_logger.info(f"Step: {step}")

    def end_logging(self, wandb_commit=True):
        if self.track_wandb:
            wandb.log(self.wandb_log_cache, commit=wandb_commit)
        if self.track_console:
            rlx_logger.info("└" + "─" * 31 + "┴" + "─" * 16 + "┘")

    def save(self, policy_state, critic_state, file_name=None):
        file_name = file_name or self.latest_model_file_name
        checkpoint = {
            "policy": policy_state,
            "critic": critic_state
        }
        save_args = orbax_utils.save_args_from_target(checkpoint)
        self.latest_model_checkpointer.save(f"{self.save_path}/tmp", checkpoint, save_args=save_args)

        with open(f"{self.save_path}/tmp/config_algorithm.json", "w") as f:
            json.dump(self.config.algorithm.to_dict(), f)

        # shutil.make_archive automatically appends '.zip' to the target file
        shutil.make_archive(f"{self.save_path}/{file_name}", "zip", f"{self.save_path}/tmp")

        # We REMOVED the os.rename line that was deleting the .zip extension
        shutil.rmtree(f"{self.save_path}/tmp")

        if self.track_wandb:
            # Note: Added .zip here so wandb uploads the correctly named archive
            wandb.save(f"{self.save_path}/{file_name}.zip", base_path=self.save_path)

    def load(config, train_env, eval_env, run_path, writer, explicitly_set_algorithm_params):
        splitted_path = config.runner.load_model.split("/")
        checkpoint_dir = os.path.abspath("/".join(splitted_path[:-1]))
        checkpoint_file_name = splitted_path[-1]
        shutil.unpack_archive(f"{checkpoint_dir}/{checkpoint_file_name}", f"{checkpoint_dir}/tmp", "zip")
        checkpoint_dir = f"{checkpoint_dir}/tmp"

        loaded_algorithm_config = json.load(open(f"{checkpoint_dir}/config_algorithm.json", "r"))
        for key, value in loaded_algorithm_config.items():
            if f"algorithm.{key}" not in explicitly_set_algorithm_params and key in config.algorithm:
                config.algorithm[key] = value
        model = PPO_RETRAINING(config, train_env, eval_env, run_path, writer)

        target = {
            "policy": model.policy_state,
            "critic": model.critic_state
        }
        restore_args = orbax_utils.restore_args_from_target(target)
        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        checkpoint = checkpointer.restore(checkpoint_dir, item=target, restore_args=restore_args)

        model.policy_state = checkpoint["policy"]
        model.critic_state = checkpoint["critic"]

        shutil.rmtree(checkpoint_dir)

        return model

    def test(self, episodes):
        self.test_ppo(episodes)


    def test_ppo(self, episodes):
        def visualize():
            rlx_logger.info("Testing runs infinitely. The episodes parameter is ignored.")

            @jax.jit
            def rollout(env_state, key):
                # Use deterministic action mean for evaluation
                action_mean, _ = self.policy.apply(self.policy_state.params, env_state.next_observation)
                action = action_mean
                processed_action = self.get_processed_action(action)
                env_state = self.eval_env.step(env_state, processed_action)
                return env_state, key

            self.key, subkey = jax.random.split(self.key)
            reset_keys = jax.random.split(subkey, self.nr_envs)
            env_state = self.eval_env.reset(reset_keys, True)

            while True:
                env_state, self.key = rollout(env_state, self.key)
                if self.render:
                    env_state = self.eval_env.render(env_state)

        def evaluate():
            rlx_logger.info("Running and saving episodes.")
            nr_steps = 1000
            batch = Batch(
                states=np.zeros((1, nr_steps, self.nr_envs) + self.os_shape),
                next_states=np.zeros((1, nr_steps, self.nr_envs) + self.os_shape),
                actions=np.zeros((1, nr_steps, self.nr_envs) + self.as_shape),
                rewards=np.zeros((1, nr_steps, self.nr_envs)),
                terminations=np.zeros((1, nr_steps, self.nr_envs)),

                values=np.zeros((1, nr_steps, self.nr_envs)),
                log_probs=np.zeros((1, nr_steps, self.nr_envs)),
                advantages=np.zeros((1, nr_steps, self.nr_envs)),
                returns=np.zeros((1, nr_steps, self.nr_envs)),
            )

            episode_return = jnp.zeros((self.nr_envs))
            self.key, subkey = jax.random.split(self.key)
            reset_keys = jax.random.split(subkey, self.nr_envs)
            env_state = self.eval_env.reset(reset_keys, True)

            for step in range(nr_steps):
                batch.states[0, step] = env_state.next_observation

                # step
                action_mean, _ = self.policy.apply(self.policy_state.params, env_state.next_observation)
                action = action_mean
                processed_action = self.get_processed_action(action)
                env_state = self.eval_env.step(env_state, processed_action)

                batch.next_states[0, step] = env_state.actual_next_observation
                batch.actions[0, step] = processed_action
                batch.rewards[0, step] = env_state.reward
                batch.terminations[0, step] = env_state.terminated
                episode_return += env_state.reward

                if self.render:
                    env_state = self.eval_env.render(env_state)

            mean_episode_return = episode_return.mean()
            rlx_logger.info(f"Mean Episode Return: {mean_episode_return}")

            def flatten_and_prune(arr):
                flat = arr.reshape(-1, arr.shape[-1])
                nan_mask = ~np.isnan(flat).any(axis=1)
                return flat[nan_mask]

            exp_states = flatten_and_prune(batch.states)
            exp_actions = flatten_and_prune(batch.actions)
            exp_next_states = flatten_and_prune(batch.next_states)
            exp_absorbing = flatten_and_prune(batch.terminations).flatten()
            exp_rewards = flatten_and_prune(batch.rewards).flatten()

            print(f"save path: {self.save_path}")
            print(f"states shape: {exp_states.shape}")
            print(f"actions shape: {exp_actions.shape}")
            print(f"rewards shape: {exp_rewards.shape}")
            print(f"absorbing shape: {exp_absorbing.shape}")

            os.makedirs(self.save_path, exist_ok=True)
            np.savez(f"{self.save_path}/expert_dataset_PPO_retrained_{self.nr_envs}", states=exp_states,
                     actions=exp_actions,
                     next_states=exp_next_states, absorbing=exp_absorbing, rewards=exp_rewards)

        visualize()
        # evaluate() # Uncomment this line and comment visualize() if you want to generate the dataset instead

    def general_properties(self):
        return GeneralProperties