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
from flax import linen as nn
import orbax.checkpoint
import optax
import wandb
import re
from flax.core import unfreeze

from trust_region_irl.algorithms.ppo_retraining_curriculum import PPO_RETRAINING_CURRICULUM
from trust_region_irl.algorithms.trirl_ppo_spr.flax_full_jit.general_properties import GeneralProperties
from trust_region_irl.algorithms.trirl_ppo_spr.flax_full_jit.policy import get_policy
from trust_region_irl.algorithms.trirl_ppo_spr.flax_full_jit.critic import get_critic
from trust_region_irl.algorithms.trirl_ppo_spr.flax_full_jit.discriminator import get_discriminator
from trust_region_irl.algorithms.trirl_ppo_spr.flax_full_jit.data_utils import prepare_expert_data, expert_data_spec, agent_windows, expert_windows

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

class TRIRL_PPO:
    def __init__(self, config, train_env, eval_env, run_path, writer):
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
        self.retraining = config.algorithm.retraining
        self.init_eta = config.algorithm.init_eta
        self.const_eta = config.algorithm.const_eta

        # TRIRL Specific
        self.data_path = config.algorithm.data_path
        self.nr_epochs_disc = config.algorithm.nr_epochs_disc
        self.learning_rate_disc = config.algorithm.learning_rate_disc
        self.env_reward_frac = config.algorithm.env_reward_frac
        self.handle_absorbing_states = config.algorithm.handle_absorbing_states
        self.epsilon = config.algorithm.epsilon
        self.gp_lambda = config.algorithm.gp_lambda
        self.gp_alpha = config.algorithm.gp_alpha
        self.beta = 1/config.algorithm.entropy_coef
        self.reward_type = config.algorithm.reward_type
        self.num_data_samples = prepare_expert_data(self.data_path)["states"].shape[0]

        # BYOL Specific
        self.pretrain_env_steps = config.algorithm.pretrain_env_steps
        self.nr_pretrain_iterations = int(self.pretrain_env_steps // self.batch_size)  # one iteration = one rollout of nr_envs * nr_steps
        self.warmup_env_steps = config.algorithm.warmup_env_steps
        self.w_disc_loss = config.algorithm.w_disc_loss
        self.target_momentum = config.algorithm.target_momentum
        self.prediction_steps = config.algorithm.prediction_steps

        self.curriculum_success = config.environment.get("curriculum_success", False)
        if self.curriculum_success:
            # load thresholds from config
            threshold_values = [float(v) for v in config.environment.success_threshold_values.split(",")]
            threshold_fractions = [float(f) for f in config.environment.success_threshold_fractions.split(",")]

            # Check if number of thresholds is the same as number of fractions
            if len(threshold_values) != len(threshold_fractions):
                raise ValueError("success_threshold_values and success_threshold_fractions must have the same number of entries")
            if abs(sum(threshold_fractions) - 1.0) > 1e-6:
                raise ValueError(f"success_threshold_fractions must sum to 1, got {sum(threshold_fractions)}")

            self.success_threshold_values = jnp.array(threshold_values, dtype=jnp.float32)
            # Env step at which each threshold after the first one starts
            self.success_threshold_boundaries = jnp.array(np.cumsum(threshold_fractions)[:-1] * self.total_timesteps, dtype=jnp.float32)
            self.final_success_threshold = threshold_values[-1]  # evaluation always uses the final threshold

        # Curriculum Thetas: corrected-reward theta of every TRIRL iteration, keyed by outer (eval/save) iteration.
        # Written to models/theta_matrix.npy at every save (save_thetas) and once more at the end of training.
        self.thetas_by_save_iteration = {}
        self.curriculum_learning_mode = config.algorithm.curriculum_learning

        if self.minibatch_size > self.batch_size:
            raise ValueError("Minibatch size must not be larger than batch size")

        if self.evaluation_and_save_frequency % self.batch_size != 0:
            raise ValueError("Evaluation and save frequency must be a multiple of batch size")

        if self.nr_parallel_seeds > 1:
            raise ValueError("Parallel seeds are not supported yet. This is mainly limited by not being able to log mutliple wandb runs at the same time.")

        rlx_logger.info(f"Using device: {jax.default_backend()}")

        self.key = jax.random.PRNGKey(self.seed)
        self.key, policy_key, critic_key, discriminator_key, reset_key = jax.random.split(self.key, 5)
        reset_key = jax.random.split(reset_key, 1)

        self.policy, self.get_processed_action = get_policy(self.config, self.train_env)
        self.critic = get_critic(self.config, self.train_env)
        self.discriminator = get_discriminator(config, self.train_env, reward_type=self.reward_type)

        def linear_schedule(count):
            fraction = 1.0 - (count // (self.nr_minibatches * self.nr_epochs)) / self.nr_updates
            return self.learning_rate * fraction

        def linear_schedule_disc(count):
            fraction = 1.0 - (count // (self.nr_minibatches * self.nr_epochs_disc)) / self.nr_updates
            return self.learning_rate_disc * fraction

        learning_rate = linear_schedule if self.anneal_learning_rate else self.learning_rate
        learning_rate_disc = linear_schedule_disc if self.anneal_learning_rate else self.learning_rate_disc

        self.key, sampling_key = jax.random.split(self.key)
        env_state = self.train_env.reset(reset_key, False)
        action = jnp.array([self.train_env.single_action_space.sample(sampling_key)])
        self.H_terminal = jnp.sum(jnp.log(self.train_env.single_action_space.high - self.train_env.single_action_space.low))

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

        # Create online network params and copy them to target network
        discriminator_params = self.discriminator.init(discriminator_key, env_state.next_observation, action, env_state.actual_next_observation)
        discriminator_params = unfreeze(discriminator_params)
        for layer in ["enc1", "enc_norm", "enc2", "enc3", "proj1", "proj_norm", "proj2"]:
            discriminator_params["params"][f"target_{layer}"] = discriminator_params["params"][f"online_{layer}"]

        self.discriminator_state = TrainState.create(
            apply_fn=self.discriminator.apply,
            params=discriminator_params,
            tx=optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.inject_hyperparams(optax.adam)(learning_rate=learning_rate_disc),
            )
        )
        self.corrected_reward_state = self.discriminator_state

        # Encoder-only pre-training uses its own optimiser state and a constant learning rate,
        # so it does not advance the discriminator's learning-rate schedule
        self.pretrain_state = TrainState.create(
            apply_fn=self.discriminator.apply,
            params=discriminator_params,
            tx=optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.inject_hyperparams(optax.adam)(learning_rate=self.learning_rate_disc),
            )
        )

        if self.save_model:
            os.makedirs(self.save_path, exist_ok=True)
            self.latest_model_file_name = "latest.model"
            self.best_model_file_name = "best.model"      # lowest eval/task_err so far
            self.best_eval_return = -np.inf
            #self.best_eval_task_err = np.inf    # host-side tracker, updated in save callback
            self.latest_model_checkpointer = orbax.checkpoint.PyTreeCheckpointer()

    def train(self):
        def jitable_train_function(key, parallel_seed_id):
            key, reset_key = jax.random.split(key, 2)
            reset_keys = jax.random.split(reset_key, self.nr_envs)
            env_state = self.train_env.reset(reset_keys, False)

            # Expert demonstrations
            def _prepare_expert_data():
                return prepare_expert_data(self.data_path)

            demonstrations = jax.experimental.io_callback(_prepare_expert_data, expert_data_spec(num_samples=self.num_data_samples, state_dim=self.train_env.single_observation_space.shape[0], action_dim=self.train_env.single_action_space.shape[0]))
            expert_states = demonstrations["states"]
            expert_next_states = demonstrations["next_states"]
            expert_actions = demonstrations["actions"]
            expert_absorbing = demonstrations["absorbing"].flatten()
            expert_features = self.train_env.feature_from_transition(expert_states, expert_actions)

            # Set up carry objects
            policy_state = self.policy_state
            critic_state = self.critic_state
            discriminator_state = self.discriminator_state
            corrected_reward_state = self.corrected_reward_state

            """ BYOL Pre-training """
            def pretrain_loss_fn(discriminator_params, state, action_window, next_state_window, weight,
                                 expert_state, expert_action_window, expert_next_state_window, expert_weight):
                # k-step chain: p, y have shape (k, 128), one row per predicted future state
                _, p, y = self.discriminator.apply(discriminator_params, state, action_window, next_state_window, method=self.discriminator.multi_step_byol)
                _, expert_p, expert_y = self.discriminator.apply(discriminator_params, expert_state, expert_action_window, expert_next_state_window, method=self.discriminator.multi_step_byol)

                y = jax.lax.stop_gradient(y)
                expert_y = jax.lax.stop_gradient(expert_y)

                # normalise every step's vector on its own (axis=-1), not the whole (k, 128) array
                p = p / (jnp.linalg.norm(p, axis=-1, keepdims=True) + 1e-12)
                expert_p = expert_p / (jnp.linalg.norm(expert_p, axis=-1, keepdims=True) + 1e-12)
                y = y / (jnp.linalg.norm(y, axis=-1, keepdims=True) + 1e-12)
                expert_y = expert_y / (jnp.linalg.norm(expert_y, axis=-1, keepdims=True) + 1e-12)

                # cosine loss per step (k,); weight = 0 for windows that leave the rollout or cross an episode end
                step_loss = weight * (2 - 2 * jnp.sum(p * y, axis=-1))
                expert_step_loss = expert_weight * (2 - 2 * jnp.sum(expert_p * expert_y, axis=-1))

                byol_loss = 0.5 * (jnp.sum(step_loss) + jnp.sum(expert_step_loss))
                metrics = {
                    "pretrain/byol_loss": byol_loss,
                }
                for i in range(self.prediction_steps):
                    metrics[f"pretrain/byol_step_{i + 1}"] = 0.5 * (step_loss[i] + expert_step_loss[i])
                return byol_loss, (metrics)

            vmap_pretrain_loss_fn = jax.vmap(pretrain_loss_fn, in_axes=(None, 0, 0, 0, 0, 0, 0, 0, 0), out_axes=0)
            mean_vmapped_pretrain_loss_fn = lambda *a, **k: tree.map_structure(jnp.mean, vmap_pretrain_loss_fn(*a, **k))
            grad_pretrain_loss_fn = jax.value_and_grad(mean_vmapped_pretrain_loss_fn, argnums=(0), has_aux=True)

            def update_target(state):
                def ema(target_param, online_param):
                    return self.target_momentum * target_param + (1 - self.target_momentum) * online_param

                params = dict(state.params)
                params["params"] = dict(state.params["params"])
                for layer in ["enc1", "enc_norm", "enc2", "enc3", "proj1", "proj_norm", "proj2"]:
                    params["params"][f"target_{layer}"] = jax.tree_util.tree_map(ema, params["params"][f"target_{layer}"], params["params"][f"online_{layer}"])
                return state.replace(params=params)

            def pretrain_rollout(pretrain_rollout_carry, _):
                env_state, key = pretrain_rollout_carry

                key, subkey = jax.random.split(key)
                observation = env_state.next_observation
                action_mean, action_logstd = self.policy.apply(policy_state.params, observation)
                action = action_mean + jnp.exp(action_logstd) * jax.random.normal(subkey, shape=action_mean.shape)
                env_state = self.train_env.step(env_state, self.get_processed_action(action))

                return (env_state, key), (observation, env_state.actual_next_observation, action, env_state.terminated | env_state.truncated)

            def pretrain_iteration(pretrain_iteration_carry, pretrain_iteration_step):
                pretrain_state, env_state, key = pretrain_iteration_carry

                (env_state, key), (states, next_states, actions, dones) = jax.lax.scan(pretrain_rollout, (env_state, key), None, self.nr_steps)
                batch_states = states.reshape((-1,) + self.os_shape)
                batch_next_states = next_states.reshape((-1,) + self.os_shape)
                # batch_actions = actions.reshape((-1,) + self.as_shape)
                batch_action_windows, batch_next_states_windows, batch_valid, batch_weights = agent_windows(actions, next_states, dones, self.prediction_steps)

                # Expert batch sampled with replacement, as in the discriminator update
                key, expert_key, permutation_key = jax.random.split(key, 3)
                expert_idx = jax.random.randint(expert_key, (self.batch_size,), 0, expert_states.shape[0])
                batch_expert_states = expert_states[expert_idx]
                batch_expert_actions_windows, batch_expert_next_states_windows, batch_expert_valid, batch_expert_weights = expert_windows(expert_actions, expert_next_states, expert_absorbing, expert_idx, self.prediction_steps)
                # batch_expert_actions = expert_actions[expert_idx]
                # batch_expert_next_states = expert_next_states[expert_idx]

                batch_indices = jnp.tile(jnp.arange(self.batch_size), (self.nr_epochs_disc, 1))
                batch_indices = jax.random.permutation(permutation_key, batch_indices, axis=1, independent=True)
                batch_indices = batch_indices[:, :self.nr_minibatches * self.minibatch_size]
                batch_indices = batch_indices.reshape((self.nr_epochs_disc * self.nr_minibatches, self.minibatch_size))

                def pretrain_minibatch_update(pretrain_state, minibatch_indices):
                    (pretrain_loss, (metrics)), (pretrain_gradients) = grad_pretrain_loss_fn(
                        pretrain_state.params,
                        batch_states[minibatch_indices],
                        batch_action_windows[minibatch_indices],
                        batch_next_states_windows[minibatch_indices],
                        batch_weights[minibatch_indices],
                        batch_expert_states[minibatch_indices],
                        batch_expert_actions_windows[minibatch_indices],
                        batch_expert_next_states_windows[minibatch_indices],
                        batch_expert_weights[minibatch_indices],
                    )
                    pretrain_state = pretrain_state.apply_gradients(grads=pretrain_gradients)
                    pretrain_state = update_target(pretrain_state)
                    return pretrain_state, (metrics)

                pretrain_state, (pretrain_metrics) = jax.lax.scan(pretrain_minibatch_update, pretrain_state, batch_indices)
                pretrain_metrics = tree.map_structure(lambda x: jnp.mean(x), pretrain_metrics)

                def callback(carry):
                    metrics, pretrain_iteration_step, parallel_seed_id = carry
                    current_time = time.time()
                    metrics["time/sps"] = int((self.nr_steps * self.nr_envs) / (current_time - self.last_time[parallel_seed_id]))
                    self.last_time[parallel_seed_id] = current_time
                    pretrain_step = int((pretrain_iteration_step.item() + 1) * self.nr_steps * self.nr_envs)
                    metrics["steps/nr_pretrain_env_steps"] = pretrain_step
                    self.start_logging(pretrain_step)
                    for key, value in metrics.items():
                        self.log(f"{key}", np.asarray(value), pretrain_step)
                    self.end_logging()

                jax.debug.callback(callback, (pretrain_metrics, pretrain_iteration_step, parallel_seed_id))

                return (pretrain_state, env_state, key), None

            if self.nr_pretrain_iterations > 0:
                key, pretrain_key = jax.random.split(key)
                (pretrain_state, env_state, _), _ = jax.lax.scan(pretrain_iteration, (self.pretrain_state, env_state, pretrain_key), jnp.arange(self.nr_pretrain_iterations))

                # TRIRL starts from the pre-trained encoder, projector, predictor and target; its optimiser state is still fresh
                discriminator_state = discriminator_state.replace(params=pretrain_state.params)
                corrected_reward_state = corrected_reward_state.replace(params=pretrain_state.params)

            def multi_learning_and_eval_save_iteration(multi_learning_and_eval_save_iteration_carry, multi_learning_iteration_step):
                policy_state, critic_state, discriminator_state, corrected_reward_state, \
                (expert_states, expert_actions, expert_next_states, expert_absorbing, expert_features), \
                env_state, key = multi_learning_and_eval_save_iteration_carry

                def learning_iteration(learning_iteration_carry, learning_iteration_step):
                    policy_state, critic_state, discriminator_state, corrected_reward_state, \
                    (expert_states, expert_actions, expert_next_states, expert_absorbing, expert_features), \
                    env_state, key = learning_iteration_carry

                    # Success-threshold curriculum: pick the threshold from the env steps done before this rollout
                    if self.curriculum_success:
                        steps_done = ((multi_learning_iteration_step * self.nr_updates_per_multi_learning_iteration) + learning_iteration_step) * self.batch_size
                        stage = jnp.searchsorted(self.success_threshold_boundaries, steps_done, side="right")
                        env_state = env_state.replace(success_threshold=jnp.full_like(env_state.success_threshold, self.success_threshold_values[stage]))

                        # Acting
                    def single_rollout(single_rollout_carry, _):
                        policy_state, critic_state, env_state, key = single_rollout_carry

                        key, subkey = jax.random.split(key)
                        observation = env_state.next_observation
                        action_mean, action_logstd = self.policy.apply(policy_state.params, observation)
                        action_std = jnp.exp(action_logstd)
                        action = action_mean + action_std * jax.random.normal(subkey, shape=action_mean.shape)
                        log_prob = (-0.5 * ((action - action_mean) / action_std) ** 2 - 0.5 * jnp.log(2.0 * jnp.pi) - action_logstd).sum(1)
                        processed_action = self.get_processed_action(action)
                        value = self.critic.apply(critic_state.params, observation).squeeze(-1)
                        action_logstd = jnp.repeat(action_logstd[None, :], action_mean.shape[0], axis=0)

                        env_state = self.train_env.step(env_state, processed_action)
                        transition = (observation, env_state.actual_next_observation, action, env_state.reward, value, env_state.terminated, log_prob, action_mean, action_logstd, env_state.info, env_state.terminated | env_state.truncated)
                        if self.render:
                            def render(env_state):
                                return self.train_env.render(env_state)

                            env_state = jax.experimental.io_callback(render, env_state, env_state)

                        return (policy_state, critic_state, env_state, key), transition

                    single_rollout_carry, batch = jax.lax.scan(single_rollout, (policy_state, critic_state, env_state, key), None, self.nr_steps)
                    policy_state, critic_state, env_state, key = single_rollout_carry
                    states, next_states, actions, rewards, values, terminations, log_probs, old_action_means, old_action_logstd, infos, dones = batch

                    """ Discriminator Update """

                    # encoder_switch: 0.0 during warm-up (discriminator loss does not train the encoder), 1.0 afterwards
                    def trirl_loss_fn(discriminator_params, state, action, expert_state, expert_action, label, expert_label,
                                        action_window, next_state_window, weight, absorbing,
                                        expert_action_window, expert_next_state_window, expert_weight, expert_absorbing, encoder_switch):
                        def update_param(w):
                            return encoder_switch * w + (1.0 - encoder_switch) * jax.lax.stop_gradient(w)

                        disc_params = dict(discriminator_params)
                        disc_params["params"] = dict(discriminator_params["params"])

                        disc_params["params"]["online_enc1"] = jax.tree_util.tree_map(update_param, disc_params["params"]["online_enc1"])
                        disc_params["params"]["online_enc_norm"] = jax.tree_util.tree_map(update_param, disc_params["params"]["online_enc_norm"])
                        disc_params["params"]["online_enc2"] = jax.tree_util.tree_map(update_param, disc_params["params"]["online_enc2"])
                        disc_params["params"]["online_enc3"] = jax.tree_util.tree_map(update_param, disc_params["params"]["online_enc3"])


                        logits = self.discriminator.apply(disc_params, state, action, method=self.discriminator._reward)
                        expert_logits = self.discriminator.apply(disc_params, expert_state, expert_action, method=self.discriminator._reward)

                        bce_agent = optax.sigmoid_binary_cross_entropy(logits, label).mean()
                        bce_expert = optax.sigmoid_binary_cross_entropy(expert_logits, expert_label).mean()

                        # Gradient penalty on an interpolated sample between the expert and agent
                        alpha = self.gp_alpha
                        gp_lambda = self.gp_lambda
                        interpolated_state = alpha * expert_state + (1 - alpha) * state

                        grad_state = jax.grad(lambda s: jnp.sum(self.discriminator.apply(disc_params, s, action, method=self.discriminator._reward)))(interpolated_state)
                        grad_norm = jnp.sqrt(jnp.sum(jnp.square(grad_state)) + 1e-12)
                        gp = (grad_norm - 1.0) ** 2

                        bce_loss = bce_agent + bce_expert + gp_lambda * gp

                        # BYOL Loss: k-step chain, p and y have shape (k, 128); real params so the encoder trains during warm-up
                        z, p, y = self.discriminator.apply(discriminator_params, state, action_window, next_state_window, method=self.discriminator.multi_step_byol)
                        expert_z, expert_p, expert_y = self.discriminator.apply(discriminator_params, expert_state, expert_action_window, expert_next_state_window, method=self.discriminator.multi_step_byol)

                        y = jax.lax.stop_gradient(y)
                        expert_y = jax.lax.stop_gradient(expert_y)

                        # normalise every step's vector on its own (axis=-1), not the whole (k, 128) array
                        p = p / (jnp.linalg.norm(p, axis=-1, keepdims=True) + 1e-12)
                        expert_p = expert_p / (jnp.linalg.norm(expert_p, axis=-1, keepdims=True) + 1e-12)
                        y = y / (jnp.linalg.norm(y, axis=-1, keepdims=True) + 1e-12)
                        expert_y = expert_y / (jnp.linalg.norm(expert_y, axis=-1, keepdims=True) + 1e-12)

                        # cosine loss per step (k,); weight = 0 for windows that leave the rollout or cross an episode end
                        step_loss = weight * (2 - 2 * jnp.sum(p * y, axis=-1))
                        expert_step_loss = expert_weight * (2 - 2 * jnp.sum(expert_p * expert_y, axis=-1))

                        BYOL_loss = 0.5 * (jnp.sum(step_loss) + jnp.sum(expert_step_loss))

                        total_loss = BYOL_loss + self.w_disc_loss * bce_loss

                        metrics = {
                            "loss/byol_loss": BYOL_loss,
                            "loss/total_loss": total_loss,
                            "loss/encoder_switch":encoder_switch,
                            "loss/discriminator_loss": bce_loss,
                            "loss/discriminator_agent_loss": bce_agent,
                            "loss/discriminator_expert_loss": bce_expert,
                            "loss/discriminator_gp": gp,
                        }
                        for i in range(self.prediction_steps):
                            metrics[f"loss/byol_step_{i + 1}"] = 0.5 * (step_loss[i] + expert_step_loss[i])
                        return total_loss, (metrics)

                    batch_states = states.reshape((-1,) + self.os_shape)
                    batch_actions = actions.reshape((-1,) + self.as_shape)

                    # Expert batch: sample batch_size transitions WITH REPLACEMENT so the
                    # expert minibatch is correct even when batch_size > number of expert
                    # transitions. (The original perm[:batch_size] returned only len(expert)
                    # rows; the downstream arange(batch_size) indexing then clamped the
                    # out-of-range indices to the last row, collapsing the expert batch to
                    # ~one repeated sample whenever nr_envs*nr_steps > len(expert_data).)
                    key, shuffle_key = jax.random.split(key)
                    expert_idx = jax.random.randint(shuffle_key, (self.batch_size,), 0, expert_states.shape[0])
                    batch_expert_states = expert_states[expert_idx]
                    batch_expert_actions = expert_actions[expert_idx]
                    expert_labels = jnp.ones((self.batch_size, 1), dtype=jnp.float32)
                    rollout_labels = jnp.zeros((self.batch_size, 1), dtype=jnp.float32)

                    batch_absorbing = terminations.reshape(-1)
                    batch_expert_absorbing = expert_absorbing[expert_idx]

                    # k-step windows for the BYOL loss: actions a_t..a_{t+k-1}, next states s_{t+1}..s_{t+k} (same order as batch_states)
                    batch_action_windows, batch_next_states_windows, batch_valid, batch_weights = agent_windows(actions, next_states, dones, self.prediction_steps)
                    batch_expert_actions_windows, batch_expert_next_states_windows, batch_expert_valid, batch_expert_weights = expert_windows(expert_actions, expert_next_states, expert_absorbing, expert_idx, self.prediction_steps)

                    combined_learning_iteration_step = (multi_learning_iteration_step * self.nr_updates_per_multi_learning_iteration) + learning_iteration_step + 1
                    global_step = combined_learning_iteration_step * self.nr_steps * self.nr_envs
                    encoder_switch = jnp.where(global_step < self.warmup_env_steps, 0.0, 1.0)

                    vmap_trirl_loss_fn = jax.vmap(trirl_loss_fn, in_axes=(None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None), out_axes=0)
                    safe_mean = lambda x: jnp.mean(x) if x is not None else x
                    mean_vmapped_trirl_loss_fn = lambda *a, **k: tree.map_structure(safe_mean, vmap_trirl_loss_fn(*a, **k))
                    grad_trirl_loss_fn = jax.value_and_grad(mean_vmapped_trirl_loss_fn, argnums=(0), has_aux=True)

                    key, subkey = jax.random.split(key)
                    batch_indices_disc = jnp.tile(jnp.arange(self.batch_size), (self.nr_epochs_disc, 1))
                    batch_indices_disc = jax.random.permutation(subkey, batch_indices_disc, axis=1, independent=True)
                    batch_indices_disc = batch_indices_disc[:, :self.nr_minibatches * self.minibatch_size]
                    batch_indices_disc = batch_indices_disc.reshape((self.nr_epochs_disc * self.nr_minibatches, self.minibatch_size))

                    def trirl_minibatch_update(carry, minibatch_indices_disc):
                        def ema(target_param, online_param):
                            return self.target_momentum * target_param + (1 - self.target_momentum) * online_param

                        discriminator_state, key = carry

                        # TRIRL UPDATE
                        (trirl_loss, (metrics)), (discriminator_gradients) = grad_trirl_loss_fn(
                            discriminator_state.params,
                            batch_states[minibatch_indices_disc],
                            batch_actions[minibatch_indices_disc],
                            batch_expert_states[minibatch_indices_disc],
                            batch_expert_actions[minibatch_indices_disc],
                            rollout_labels[minibatch_indices_disc],
                            expert_labels[minibatch_indices_disc],
                            batch_action_windows[minibatch_indices_disc],
                            batch_next_states_windows[minibatch_indices_disc],
                            batch_weights[minibatch_indices_disc],
                            batch_absorbing[minibatch_indices_disc],
                            batch_expert_actions_windows[minibatch_indices_disc],
                            batch_expert_next_states_windows[minibatch_indices_disc],
                            batch_expert_weights[minibatch_indices_disc],
                            batch_expert_absorbing[minibatch_indices_disc],
                            encoder_switch,
                        )

                        discriminator_state = discriminator_state.apply_gradients(grads=discriminator_gradients)

                        disc_params = dict(discriminator_state.params)
                        disc_params["params"] = dict(discriminator_state.params["params"])

                        disc_params["params"]["target_enc1"] = jax.tree_util.tree_map(ema, disc_params["params"]["target_enc1"], disc_params["params"]["online_enc1"])
                        disc_params["params"]["target_enc_norm"] = jax.tree_util.tree_map(ema, disc_params["params"]["target_enc_norm"], disc_params["params"]["online_enc_norm"])
                        disc_params["params"]["target_enc2"] = jax.tree_util.tree_map(ema, disc_params["params"]["target_enc2"], disc_params["params"]["online_enc2"])
                        disc_params["params"]["target_enc3"] = jax.tree_util.tree_map(ema, disc_params["params"]["target_enc3"], disc_params["params"]["online_enc3"])
                        disc_params["params"]["target_proj1"] = jax.tree_util.tree_map(ema, disc_params["params"][ "target_proj1"], disc_params["params"]["online_proj1"])
                        disc_params["params"]["target_proj_norm"] = jax.tree_util.tree_map(ema, disc_params["params"][ "target_proj_norm"], disc_params["params"]["online_proj_norm"])
                        disc_params["params"]["target_proj2"] = jax.tree_util.tree_map(ema, disc_params["params"]["target_proj2"], disc_params["params"]["online_proj2"])

                        # Copy the updates after ema to discriminator state
                        discriminator_state = discriminator_state.replace(params=disc_params)

                        carry = (discriminator_state, key)

                        return carry, (metrics)

                    init_carry = (discriminator_state, key)
                    carry, (disc_optimization_metrics) = jax.lax.scan(trirl_minibatch_update, init_carry, batch_indices_disc)
                    discriminator_state, key = carry

                    """ Reward Update """

                    # Define Apply Functions
                    def _get_log_density_ratio(inputs, discriminator_state):
                        state, action = inputs
                        logits = self.discriminator.apply(discriminator_state.params, state, action, method=self.discriminator._reward)
                        return logits

                    _get_log_density_ratio = jax.vmap(_get_log_density_ratio, in_axes=(0, None), out_axes=0)


                    _inputs = (states.reshape((-1,) + self.os_shape), actions.reshape((-1,) + self.as_shape))
                    corr_reward = self.entropy_coef * ((1 - self.epsilon) * _get_log_density_ratio(_inputs, corrected_reward_state).reshape(rewards.shape) \
                        + self.epsilon * self.beta * _get_log_density_ratio(_inputs, discriminator_state).reshape(rewards.shape))

                    if self.handle_absorbing_states:
                        _inputs_absorbing = (next_states.reshape((-1,) + self.os_shape), actions.reshape((-1,) + self.as_shape))
                        corr_reward_absorbing_state = self.entropy_coef * ((1 - self.epsilon) * _get_log_density_ratio(_inputs_absorbing, corrected_reward_state).reshape(rewards.shape) \
                            + self.epsilon * self.beta * _get_log_density_ratio(_inputs_absorbing, discriminator_state).reshape(rewards.shape))
                    else:
                        corr_reward_absorbing_state = jnp.asarray(0.0)

                    corr_reward = self.env_reward_frac * rewards + (1 - self.env_reward_frac) * corr_reward
                    corr_reward_absorbing_state = (1 - self.env_reward_frac) * corr_reward_absorbing_state # the environment pays no reward in the absorbing state


                    """ PPO + KL Penalty """

                    # Calculating advantages and returns
                    def calculate_gae_advantages(critic_state, next_states, rewards, values, terminations):
                        def compute_advantages(carry, t):
                            prev_advantage = carry[0]
                            advantage = delta[t] + self.gamma * self.gae_lambda * (1 - terminations[t]) * prev_advantage
                            return (advantage,), advantage

                        next_values = self.critic.apply(critic_state.params, next_states).squeeze(-1)
                        delta = rewards + self.gamma * next_values * (1.0 - terminations) - values
                        init_advantages = delta[-1]
                        _, advantages = jax.lax.scan(compute_advantages, (init_advantages,), jnp.arange(self.nr_steps - 2, -1, -1))
                        advantages = jnp.concatenate([advantages[::-1], jnp.array([init_advantages])])
                        returns = advantages + values
                        return advantages, returns

                    def calculate_gae_advantages_absorbing(critic_state, next_states, rewards, rewards_next_state, values, terminations):
                        """
                        Correctly handle absorbing state value and entropy (instead of setting to 0.0)
                        """
                        def compute_advantages(carry, t):
                            prev_advantage = carry[0]
                            advantage = delta[t] + self.gamma * self.gae_lambda * (1 - terminations[t]) * prev_advantage
                            return (advantage,), advantage

                        next_values = self.critic.apply(critic_state.params, next_states).squeeze(-1)
                        terminal_tail = (self.gamma / (1.0 - self.gamma)) * (rewards_next_state + self.entropy_coef * self.H_terminal)
                        delta = rewards + self.gamma * next_values * (1.0 - terminations) + (terminations * terminal_tail) - values
                        init_advantages = delta[-1]
                        _, advantages = jax.lax.scan(compute_advantages, (init_advantages,), jnp.arange(self.nr_steps - 2, -1, -1))
                        advantages = jnp.concatenate([advantages[::-1], jnp.array([init_advantages])])
                        returns = advantages + values
                        return advantages, returns


                    if self.handle_absorbing_states:
                        advantages, returns = calculate_gae_advantages_absorbing(critic_state, next_states, corr_reward, corr_reward_absorbing_state, values, terminations)
                    else:
                        advantages, returns = calculate_gae_advantages(critic_state, next_states, corr_reward, values, terminations)


                    # Compute eta (linear schedule)
                    def linear_schedule_eta(global_step):
                        if self.const_eta:
                            return self.init_eta
                        else:
                            fraction = 1.0 - jnp.clip(global_step, min=None, max=self.total_timesteps) / self.total_timesteps
                            return self.init_eta * fraction

                    eta = linear_schedule_eta(global_step)


                    # Optimizing
                    def loss_fn(policy_params, critic_params, state_b, action_b, log_prob_b, return_b, advantage_b, old_mean_b, old_logstd_b, eta):
                        # Policy loss
                        action_mean, action_logstd = self.policy.apply(policy_params, state_b)
                        action_std = jnp.exp(action_logstd)
                        new_log_prob = -0.5 * ((action_b - action_mean) / action_std) ** 2 - 0.5 * jnp.log(2.0 * jnp.pi) - action_logstd
                        new_log_prob = new_log_prob.sum(1)
                        entropy = action_logstd + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e)
                        entropy = self.entropy_coef * entropy # scaling to reduce critic loss

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

                        # Trust Region Loss
                        # Analytical KL (across all samples)
                        old_logstd_b = jnp.expand_dims(old_logstd_b, 0) # shape (1, as_shape)
                        old_action_std = jnp.exp(old_logstd_b)
                        tr_loss_maha = 0.5 * jnp.sum(((old_mean_b - action_mean)/old_action_std) ** 2, axis=1)
                        tr_loss_cov_part = 0.5 * jnp.sum(2.0 * (jnp.log(old_action_std) - jnp.log(action_std)) + (action_std/old_action_std)**2 - 1.0, axis=1)
                        trust_region_loss = tr_loss_maha + tr_loss_cov_part

                        # Combine losses
                        loss = pg_loss - self.entropy_coef * entropy_loss + self.critic_coef * critic_loss + eta * trust_region_loss

                        # Create metrics
                        metrics = {
                            "loss/policy_gradient_loss": pg_loss,
                            "loss/critic_loss": critic_loss,
                            "loss/entropy_loss": entropy_loss,
                            "loss/trust_region_loss": trust_region_loss,
                            "policy_ratio/approx_kl": approx_kl_div,
                            "policy_ratio/clip_fraction": clip_fraction,
                        }

                        return loss, (metrics)

                    # Gradients must only flow through the predicted ones
                    old_action_means = jax.lax.stop_gradient(old_action_means)
                    old_action_logstd = jax.lax.stop_gradient(old_action_logstd)
                    old_action_logstd = old_action_logstd.squeeze()
                    old_action_logstd = old_action_logstd[0,0,:]

                    batch_states = states.reshape((-1,) + self.os_shape)
                    batch_actions = actions.reshape((-1,) + self.as_shape)
                    batch_advantages = advantages.reshape(-1)
                    batch_returns = returns.reshape(-1)
                    batch_log_probs = log_probs.reshape(-1)
                    batch_action_means = old_action_means.reshape((-1,) + self.as_shape)

                    vmap_loss_fn = jax.vmap(loss_fn, in_axes=(None, None, 0, 0, 0, 0, 0, 0, None, None), out_axes=0)
                    safe_mean = lambda x: jnp.mean(x) if x is not None else x
                    mean_vmapped_loss_fn = lambda *a, **k: tree.map_structure(safe_mean, vmap_loss_fn(*a, **k))
                    grad_loss_fn = jax.value_and_grad(mean_vmapped_loss_fn, argnums=(0, 1), has_aux=True)

                    key, subkey = jax.random.split(key)
                    batch_indices = jnp.tile(jnp.arange(self.batch_size), (self.nr_epochs, 1))
                    batch_indices = jax.random.permutation(subkey, batch_indices, axis=1, independent=True)
                    batch_indices = batch_indices[:, :self.nr_minibatches * self.minibatch_size]
                    batch_indices = batch_indices.reshape((self.nr_epochs * self.nr_minibatches, self.minibatch_size))

                    def ppo_minibatch_update(carry, minibatch_indices):
                        policy_state, critic_state = carry

                        minibatch_advantages = batch_advantages[minibatch_indices]
                        minibatch_advantages = (minibatch_advantages - jnp.mean(minibatch_advantages)) / (jnp.std(minibatch_advantages) + 1e-8)

                        (loss, (metrics)), (policy_gradients, critic_gradients) = grad_loss_fn(
                            policy_state.params,
                            critic_state.params,
                            batch_states[minibatch_indices],
                            batch_actions[minibatch_indices],
                            batch_log_probs[minibatch_indices],
                            batch_returns[minibatch_indices],
                            minibatch_advantages,
                            batch_action_means[minibatch_indices],
                            old_action_logstd,
                            eta
                        )

                        policy_state = policy_state.apply_gradients(grads=policy_gradients)
                        critic_state = critic_state.apply_gradients(grads=critic_gradients)

                        metrics["gradients/policy_grad_norm"] = optax.global_norm(policy_gradients)
                        metrics["gradients/critic_grad_norm"] = optax.global_norm(critic_gradients)

                        carry = (policy_state, critic_state)
                        return carry, (metrics)

                    init_carry = (policy_state, critic_state)
                    carry, (ppo_optimization_metrics) = jax.lax.scan(ppo_minibatch_update, init_carry, batch_indices)
                    policy_state, critic_state = carry

                    ppo_optimization_metrics["lr/learning_rate"] = policy_state.opt_state[1].hyperparams["learning_rate"]
                    ppo_optimization_metrics["v_value/explained_variance"] = 1 - jnp.var(returns - values) / (jnp.var(returns) + 1e-8)
                    ppo_optimization_metrics["policy/std_dev"] = jnp.mean(jnp.exp(policy_state.params["params"]["policy_logstd"]))
                    ppo_optimization_metrics["correction/eta"] = eta


                    """ Reward Correction """
                    step = self.epsilon / (1.0 + eta)
                    old_params = corrected_reward_state.params
                    updated_params = discriminator_state.params
                    new_inner_params = {
                        **updated_params["params"],
                        "theta": (1.0 - step) * old_params["params"]["theta"] + step * self.beta * updated_params["params"]["theta"],
                    }
                    if self.reward_type == "shapedboltzmann-feature-based":
                        new_inner_params["theta_h"] = (1.0 - step) * old_params["params"]["theta_h"] + step * self.beta * updated_params["params"]["theta_h"]
                    new_params = {
                        **updated_params,
                        "params": new_inner_params,
                    }
                    corrected_reward_state = discriminator_state.replace(params=new_params)

                    # Logging
                    combined_metrics = {**infos, **disc_optimization_metrics, **ppo_optimization_metrics}
                    if self.curriculum_success:
                        combined_metrics["env/success_threshold"] = env_state.success_threshold
                    combined_metrics = tree.map_structure(lambda x: jnp.mean(x), combined_metrics)

                    def callback(carry):
                        metrics, learning_iteration_step, combined_learning_iteration_step, parallel_seed_id = carry
                        current_time = time.time()
                        metrics["time/sps"] = int((self.nr_steps * self.nr_envs) / (current_time - self.last_time[parallel_seed_id]))
                        self.last_time[parallel_seed_id] = current_time
                        global_step = int(combined_learning_iteration_step.item() * self.nr_steps * self.nr_envs)
                        metrics["steps/nr_env_steps"] = global_step
                        metrics["steps/nr_updates"] = combined_learning_iteration_step.item() * self.nr_epochs * self.nr_minibatches
                        is_last_train_update_before_eval = self.evaluation_active and (learning_iteration_step + 1 == self.nr_updates_per_multi_learning_iteration)
                        self.start_logging(global_step)
                        for key, value in metrics.items():
                            self.log(f"{key}", np.asarray(value), global_step)
                        self.end_logging(wandb_commit=not is_last_train_update_before_eval)

                    combined_learning_iteration_step = (multi_learning_iteration_step * self.nr_updates_per_multi_learning_iteration) + learning_iteration_step + 1
                    jax.debug.callback(callback, (combined_metrics, learning_iteration_step, combined_learning_iteration_step, parallel_seed_id))

                    return (policy_state, critic_state, discriminator_state, corrected_reward_state, (expert_states, expert_actions, expert_next_states, expert_absorbing, expert_features), env_state, key), corrected_reward_state.params['params']['theta']  # the reward theta the policy uses (retraining / curriculum)

                key, subkey = jax.random.split(key)
                learning_iteration_carry, inner_thetas = jax.lax.scan(learning_iteration, (policy_state, critic_state, discriminator_state, corrected_reward_state,
                                                                                (expert_states, expert_actions, expert_next_states, expert_absorbing, expert_features),
                                                                                env_state, subkey), jnp.arange(self.nr_updates_per_multi_learning_iteration))

                policy_state, critic_state, discriminator_state, corrected_reward_state, \
                (expert_states, expert_actions, expert_next_states, expert_absorbing, expert_features), \
                env_state, key = learning_iteration_carry


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
                    if self.curriculum_success:
                        # Fixed final threshold, so eval/episode_return and the best-model choice are comparable over training
                        eval_env_state = eval_env_state.replace(success_threshold=jnp.full_like(eval_env_state.success_threshold, self.final_success_threshold))
                    single_eval_rollout_carry, _ = jax.lax.scan(single_eval_rollout, (policy_state, eval_env_state), jnp.arange(self.horizon))
                    _, eval_env_state = single_eval_rollout_carry

                    eval_metrics = {
                        "eval/episode_return": jnp.mean(eval_env_state.info["rollout/episode_return"]),
                        "eval/episode_length": jnp.mean(eval_env_state.info["rollout/episode_length"]),
                    }

                    if "rollout/is_success" in eval_env_state.info:
                        eval_metrics["eval/is_success"] = jnp.mean(eval_env_state.info["rollout/is_success"])

                    # if "rollout/task_err" in eval_env_state.info:
                    #     eval_metrics["eval/task_err"] = jnp.mean(eval_env_state.info["rollout/task_err"])

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
                    eval_return_for_save = (eval_metrics["eval/episode_return"]
                                            if self.evaluation_active else jnp.asarray(-jnp.inf))

                    def save_with_check(policy_state, critic_state, discriminator_state, corrected_reward_state,
                                        eval_return):
                        self.save(policy_state, critic_state, discriminator_state,
                                  corrected_reward_state)  # latest.model

                        if self.evaluation_active:
                            current_return = float(np.asarray(eval_return).reshape(-1)[0])

                            # Check if the current return is GREATER than the best return
                            if current_return > self.best_eval_return:
                                self.best_eval_return = current_return
                                self.save(policy_state, critic_state, discriminator_state, corrected_reward_state,
                                          file_name=self.best_model_file_name)
                                rlx_logger.info(
                                    f"[save-best] new best eval/episode_return={current_return:.4f} -> {self.best_model_file_name}")

                    jax.debug.callback(save_with_check, policy_state, critic_state, discriminator_state,
                                       corrected_reward_state, eval_return_for_save)
                    jax.debug.callback(self.save_thetas, inner_thetas, multi_learning_iteration_step)

                return (policy_state, critic_state, discriminator_state, corrected_reward_state, (expert_states, expert_actions, expert_next_states, expert_absorbing, expert_features), env_state, key), inner_thetas


            _, all_thetas =  jax.lax.scan(multi_learning_and_eval_save_iteration, (policy_state, critic_state, discriminator_state, corrected_reward_state,
                                                                  (expert_states, expert_actions, expert_next_states, expert_absorbing, expert_features),
                                                                  env_state, key), jnp.arange(self.nr_multi_learning_and_eval_save_iterations))

            # Make the jitted function return the data
            return all_thetas

        self.key, subkey = jax.random.split(self.key)
        seed_keys = jax.random.split(subkey, self.nr_parallel_seeds)
        train_function = jax.jit(jax.vmap(jitable_train_function))
        self.last_time = [time.time() for _ in range(self.nr_parallel_seeds)]
        self.start_time = deepcopy(self.last_time)
        all_thetas = jax.block_until_ready(train_function(seed_keys, jnp.arange(self.nr_parallel_seeds)))
        rlx_logger.info(f"Average time: {max([time.time() - t for t in self.start_time]):.2f} s")

        all_thetas = np.asarray(all_thetas)
        thetas_seed0 = all_thetas[0]
        thetas_flat = thetas_seed0.reshape(-1, *thetas_seed0.shape[2:])
        thetas_to_save = thetas_flat
        os.makedirs(self.save_path, exist_ok=True)
        file_path = os.path.join(self.save_path, "theta_matrix.npy")
        np.save(file_path, thetas_to_save)

        rlx_logger.info(f"Saved theta matrix of shape {thetas_to_save.shape} to {file_path}")


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


    def save_thetas(self, inner_thetas, save_iteration):
        """Write models/theta_matrix.npy with the corrected-reward thetas recorded so far (one row per TRIRL iteration),
        so the file exists even if the run is killed before the end. Keyed by save iteration, so rows stay in order."""
        self.thetas_by_save_iteration[int(np.asarray(save_iteration))] = np.asarray(inner_thetas)
        thetas = np.concatenate([self.thetas_by_save_iteration[k] for k in sorted(self.thetas_by_save_iteration)], axis=0)
        os.makedirs(self.save_path, exist_ok=True)
        np.save(os.path.join(self.save_path, "theta_matrix.npy"), thetas)

    def save(self, policy_state, critic_state, discriminator_state, corrected_reward_state, file_name=None):
        file_name = file_name or self.latest_model_file_name
        checkpoint = {
            "policy": policy_state,
            "critic": critic_state,
            "discriminator": discriminator_state,
            "corrected_reward": corrected_reward_state,
        }
        save_args = orbax_utils.save_args_from_target(checkpoint)
        self.latest_model_checkpointer.save(f"{self.save_path}/tmp", checkpoint, save_args=save_args)
        with open(f"{self.save_path}/tmp/config_algorithm.json", "w") as f:
            json.dump(self.config.algorithm.to_dict(), f)
        shutil.make_archive(f"{self.save_path}/{file_name}", "zip", f"{self.save_path}/tmp")
        shutil.rmtree(f"{self.save_path}/tmp")

        if self.track_wandb:
            wandb.save(f"{self.save_path}/{file_name}.zip", base_path=self.save_path)

    def load(config, train_env, eval_env, run_path, writer, explicitly_set_algorithm_params):
        splitted_path = config.runner.load_model.split("/")
        checkpoint_dir = os.path.abspath("/".join(splitted_path[:-1]))
        checkpoint_file_name = splitted_path[-1]
        if os.path.exists(f"{checkpoint_dir}/tmp"):
            shutil.rmtree(f"{checkpoint_dir}/tmp")
        shutil.unpack_archive(f"{checkpoint_dir}/{checkpoint_file_name}", f"{checkpoint_dir}/tmp", "zip")
        checkpoint_dir = f"{checkpoint_dir}/tmp"

        loaded_algorithm_config = json.load(open(f"{checkpoint_dir}/config_algorithm.json", "r"))
        for key, value in loaded_algorithm_config.items():
            if f"algorithm.{key}" not in explicitly_set_algorithm_params and key in config.algorithm:
                config.algorithm[key] = value
        model = TRIRL_PPO(config, train_env, eval_env, run_path, writer)

        target = {
            "policy": model.policy_state,
            "critic": model.critic_state,
            "discriminator": model.discriminator_state,
            "corrected_reward": model.corrected_reward_state,
        }
        restore_args = orbax_utils.restore_args_from_target(target)
        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        checkpoint = checkpointer.restore(checkpoint_dir, item=target, restore_args=restore_args)

        model.policy_state = checkpoint["policy"]
        model.critic_state = checkpoint["critic"]
        model.discriminator_state = checkpoint["discriminator"]
        model.corrected_reward_state = checkpoint["corrected_reward"]

        shutil.rmtree(checkpoint_dir)

        return model

    def make_reward_function(self):
        discriminator = self.discriminator
        corrected_reward_params = self.corrected_reward_state.params
        entropy_coef = self.entropy_coef

        def reward_function(state, action, next_state, absorbing, log_prob=None):
            params = corrected_reward_params

            reward = discriminator.apply(params, state, action, method=discriminator._reward)
            return entropy_coef * reward

        reward_function = jax.vmap(reward_function, in_axes=(0, 0, 0, 0, None), out_axes=0)
        return reward_function

    def make_reward_function_fc(self):
        discriminator = self.discriminator
        entropy_coef = self.entropy_coef
        base_params = self.corrected_reward_state.params

        def reward_function(theta, state, action, next_state, absorbing, log_prob=None):
            params = {
                **base_params,
                "params": {
                    **base_params["params"],
                    "theta": theta  # Override base theta with the curriculum theta
                }
            }

            reward = discriminator.apply(params, state, action, method=discriminator._reward)
            return entropy_coef * reward

        reward_function = jax.vmap(reward_function, in_axes=(None, 0, 0, 0, 0, None), out_axes=0)
        return reward_function


    def test(self, episodes):
        if self.retraining:
            self.retrain_ppo()
        else:
            self.test_irl(episodes)


    def retrain_ppo(self):
        from trust_region_irl.algorithms.ppo_retraining.ppo_retraining import PPO_RETRAINING
        self.config.runner.track_tb = True
        self.config.runner.track_wandb = True
        self.config.runner.track_console = True
        self.config.runner.save_model = True   # RL-X test mode forbids saving; the retrained policy is the result, so keep it
        run_name = f"{self.config.runner.run_name}_ppo_retraining"
        run_path = f"runs/{self.config.runner.project_name}/{self.config.runner.exp_name}/{run_name}"
        run_path = os.path.abspath(run_path)
        if self.config.runner.save_model or self.config.runner.track_tb or self.config.runner.track_wandb:
            os.makedirs(run_path, exist_ok=True)

        if self.config.runner.track_wandb:
            import wandb
            wandb.init(
                entity=self.config.runner.wandb_entity,
                project=self.config.runner.project_name,
                group=self.config.runner.exp_name,
                name=run_name,
                notes=self.config.runner.notes,
                sync_tensorboard=True,
                config=self.config.to_dict(),
                monitor_gym=True,
                save_code=True,
            )

        writer = None
        if self.config.runner.track_tb:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(run_path)
            all_config_items = self.config.runner.items() + self.config.algorithm.items() + self.config.environment.items()
            writer.add_text(
                "hyperparameters",
                "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in all_config_items])),
            )

        # Switch to use dynamic thetas in case of curriculum learning
        if self.curriculum_learning_mode:
            print("TRAINING WITH CURRICULUM MODE")
            reward_function = self.make_reward_function_fc()
            model = PPO_RETRAINING_CURRICULUM(self.config, self.train_env, self.eval_env, run_path, writer, reward_function)
        else:
            print("TRAINING WITHOUT CURRICULUM MODE")
            reward_function = self.make_reward_function()
            model = PPO_RETRAINING(self.config, self.train_env, self.eval_env, run_path, writer, reward_function)

        try:
            model.train()
        finally:
            self.train_env.close()
            self.eval_env.close()
            if self.config.runner.track_tb:
                writer.close()
            if self.config.runner.track_wandb:
                wandb.finish()


    def test_irl(self, episodes):
        def visualize():
            rlx_logger.info("Testing runs infinitely. The episodes parameter is ignored.")

            @jax.jit
            def rollout(env_state, key):
                action_mean, action_logstd = self.policy.apply(self.policy_state.params, env_state.next_observation)
                action = action_mean # + action_std * jax.random.normal(subkey, shape=action_mean.shape)
                processed_action = self.get_processed_action(action)
                env_state = self.train_env.step(env_state, processed_action)
                return env_state, key

            self.key, subkey = jax.random.split(self.key)
            reset_keys = jax.random.split(subkey, self.nr_envs)
            env_state = self.train_env.reset(reset_keys, True)
            while True:
                env_state, self.key = rollout(env_state, self.key)
                if self.render:
                    env_state = self.train_env.render(env_state)

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
            ).shape

            episode_return = jnp.zeros((self.nr_envs))
            self.key, subkey = jax.random.split(self.key)
            reset_keys = jax.random.split(subkey, self.nr_envs)
            env_state = self.train_env.reset(reset_keys, True)

            for step in range(nr_steps):
                batch.states[0, step] = env_state.next_observation

                # step
                action_mean, action_logstd = self.policy.apply(self.policy_state.params, env_state.next_observation)
                action = action_mean
                processed_action = self.get_processed_action(action)
                env_state = self.train_env.step(env_state, processed_action)

                batch.next_states[0, step] = env_state.actual_next_observation
                batch.actions[0, step] = processed_action
                batch.rewards[0, step] = env_state.reward
                batch.terminations[0, step] = env_state.terminated
                episode_return += env_state.reward

                if self.render:
                    env_state = self.train_env.render(env_state)

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
            np.savez(f"{self.save_path}/expert_dataset_Humanoid-v5_{self.nr_envs}_PPONew", states=exp_states, actions=exp_actions,
                next_states=exp_next_states, absorbing=exp_absorbing, rewards=exp_rewards)


        visualize()

    def general_properties(self):
        return GeneralProperties