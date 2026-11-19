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

from trust_region_irl.algorithms.trirl_ppo_byol.flax_full_jit.byol import BYOL
from trust_region_irl.algorithms.ppo_retraining_curriculum import PPO_RETRAINING_CURRICULUM
from trust_region_irl.algorithms.trirl_ppo_byol.flax_full_jit.general_properties import GeneralProperties
from trust_region_irl.algorithms.trirl_ppo_byol.flax_full_jit.policy import get_policy
from trust_region_irl.algorithms.trirl_ppo_byol.flax_full_jit.critic import get_critic
from trust_region_irl.algorithms.trirl_ppo_byol.flax_full_jit.discriminator import get_discriminator
from trust_region_irl.algorithms.trirl_ppo_byol.flax_full_jit.data_utils import prepare_expert_data, expert_data_spec

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
        self.byol_input_type = config.algorithm.byol_input_type
        self.byol_input = config.algorithm.byol_input
        if self.byol_input not in ("obs", "features"):
            raise ValueError(f"byol_input must be 'obs' or 'features', got '{self.byol_input}'")
        if self.byol_input == "features" and not hasattr(self.train_env, "feature_from_transition"):
            raise ValueError("byol_input='features' needs an env with feature_from_transition")
        self.byol_hidden_dims = config.algorithm.byol_hidden_dims
        self.byol_latent_dim = config.algorithm.byol_latent_dim
        self.byol_proj_hidden_dim = config.algorithm.byol_proj_hidden_dim
        self.byol_proj_dim = config.algorithm.byol_proj_dim
        self.learning_rate_byol = config.algorithm.learning_rate_byol
        self.byol_tau = config.algorithm.byol_tau
        self.byol_noise_scale = config.algorithm.byol_noise_scale
        self.augmentation = config.algorithm.augmentation
        self.nr_epochs_byol = config.algorithm.nr_epochs_byol
        self.byol_pretrain = config.algorithm.byol_pretrain
        self.byol_pretrain_steps = int(config.algorithm.byol_pretrain_steps)
        self.byol_pretrain_log_frequency = int(config.algorithm.byol_pretrain_log_frequency)
        self.byol_pretrain_env_steps = int(float(config.algorithm.byol_pretrain_env_steps))
        self.byol_pretrain_steps_per_rollout = int(config.algorithm.byol_pretrain_steps_per_rollout)
        if self.byol_pretrain and self.byol_pretrain_steps < 1:
            raise ValueError("byol_pretrain_steps must be >= 1 when byol_pretrain is True")
        if self.byol_pretrain and self.byol_pretrain_env_steps > 0 and self.byol_pretrain_steps_per_rollout < 1:
            raise ValueError("byol_pretrain_steps_per_rollout must be >= 1 when byol_pretrain_env_steps > 0")

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
        self.key, policy_key, critic_key, discriminator_key, byol_key, target_byol_key, reset_key = jax.random.split(self.key, 7)
        reset_key = jax.random.split(reset_key, 1)

        self.policy, self.get_processed_action = get_policy(self.config, self.train_env)
        self.critic = get_critic(self.config, self.train_env)
        self.discriminator = get_discriminator(config, self.train_env, reward_type=self.reward_type)
        self.byol = BYOL(
            input_type=self.byol_input_type,
            state_hidden_dims=self.byol_hidden_dims,
            latent_dim=self.byol_latent_dim,
            proj_hidden_dim=self.byol_proj_hidden_dim,
            proj_dim=self.byol_proj_dim,
        )

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

        byol_params = self.byol.init(byol_key, self.byol_inputs(env_state.next_observation, action))
        self.byol_state = TrainState.create(
            apply_fn=self.byol.apply,
            params=byol_params,
            tx=optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.inject_hyperparams(optax.adam)(learning_rate=self.learning_rate_byol),
            )
        )
        # BYOL: the target network starts as an exact copy of the online network and then follows it by EMA
        self.byol_target_params = jax.tree_util.tree_map(jnp.copy, byol_params)
        feature = self.byol.apply(self.byol_state.params, self.byol_inputs(env_state.next_observation, action), method=self.byol.encode)

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

        self.discriminator_state = TrainState.create(
            apply_fn=self.discriminator.apply,
            params=self.discriminator.init(discriminator_key, feature, env_state.next_observation, action, env_state.actual_next_observation, env_state.terminated),
            tx=optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.inject_hyperparams(optax.adam)(learning_rate=learning_rate_disc),
            )
        )
        self.corrected_reward_state = self.discriminator_state

        if self.save_model:
            os.makedirs(self.save_path, exist_ok=True)
            self.latest_model_file_name = "latest.model"
            self.best_model_file_name = "best.model"      # lowest eval/task_err so far
            self.best_eval_return = -np.inf
            #self.best_eval_task_err = np.inf    # host-side tracker, updated in save callback
            self.latest_model_checkpointer = orbax.checkpoint.PyTreeCheckpointer()

    def byol_inputs(self, observation, action):
        """Input of the BYOL encoder: the raw observation, or the env's hand-crafted features if byol_input='features'."""
        if self.byol_input == "features":
            return self.train_env.feature_from_transition(observation, action)
        return observation

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
            expert_byol_inputs = self.byol_inputs(expert_states, expert_actions)  # BYOL pretraining data (expert half)

            # Set up carry objects
            policy_state = self.policy_state
            critic_state = self.critic_state
            discriminator_state = self.discriminator_state
            corrected_reward_state = self.corrected_reward_state
            byol_state = self.byol_state
            byol_target_params = self.byol_target_params

            """ BYOL loss (shared by the pretraining phase and the online update inside TRIRL) """

            def augment_state(x, rng_key):
                if not self.augmentation:
                    return x
                return x + self.byol_noise_scale * jax.random.normal(rng_key, shape=x.shape)

            def byol_loss_fn(online_params, target_params, x_batch, rng_key):
                rng1, rng2 = jax.random.split(rng_key)
                view_1 = augment_state(x_batch, rng1)
                view_2 = augment_state(x_batch, rng2)

                pred_1 = self.byol.apply(online_params, view_1, method=self.byol.online)
                pred_2 = self.byol.apply(online_params, view_2, method=self.byol.online)

                proj_1 = jax.lax.stop_gradient(self.byol.apply(target_params, view_1, method=self.byol.target))
                proj_2 = jax.lax.stop_gradient(self.byol.apply(target_params, view_2, method=self.byol.target))

                def cosine_loss(p, z):
                    p = p / (jnp.linalg.norm(p, axis=-1, keepdims=True) + 1e-8)
                    z = z / (jnp.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)
                    return 2 - 2 * jnp.sum(p * z, axis=-1)

                return jnp.mean(cosine_loss(pred_1, proj_2) + cosine_loss(pred_2, proj_1))

            grad_byol_loss_fn = jax.value_and_grad(byol_loss_fn)

            """ Optional BYOL pretraining (byol_pretrain=True) """
            # Runs once, before TRIRL. Data = expert states + states from ONE rollout (nr_envs x nr_steps)
            # of the randomly initialised policy, collected on a separate env instance so the TRIRL env
            # state and its episode statistics are untouched. Each pretraining step samples minibatch_size
            # rollout states + minibatch_size expert states (with replacement), computes L_BYOL, takes one
            # online gradient step and EMA-updates the target. Afterwards the encoder keeps training
            # online inside TRIRL exactly as without pretraining (the Adam state is carried over).
            # With byol_pretrain_env_steps > 0 the data is streamed instead: ceil(env_steps / (nr_envs x nr_steps))
            # rounds, each collecting a FRESH rollout (continuing the same separate env, random policy) followed by
            # byol_pretrain_steps_per_rollout steps on it. Only one rollout is held in memory; metrics are averaged per round.
            if self.byol_pretrain:
                key, pre_reset_key, pre_rollout_key, pre_loop_key = jax.random.split(key, 4)
                pre_env_state = self.train_env.reset(jax.random.split(pre_reset_key, self.nr_envs), False)

                def pretrain_rollout(carry, _):
                    pre_env_state, rollout_key = carry
                    rollout_key, subkey = jax.random.split(rollout_key)
                    observation = pre_env_state.next_observation
                    action_mean, action_logstd = self.policy.apply(policy_state.params, observation)
                    action = action_mean + jnp.exp(action_logstd) * jax.random.normal(subkey, shape=action_mean.shape)
                    pre_env_state = self.train_env.step(pre_env_state, self.get_processed_action(action))
                    return (pre_env_state, rollout_key), self.byol_inputs(observation, action)

                # Live logging: one host callback per pretraining step (single-rollout mode) or per round (streamed mode).
                # log_byol_pretraining decides on the host which entries to log (every byol_pretrain_log_frequency + the last).
                def log_pretrain_entry(metrics, entry_idx, nr_entries):
                    jax.debug.callback(lambda m, i, p: self.log_byol_pretraining(m, i, nr_entries, p), metrics, entry_idx, parallel_seed_id)

                def run_byol_pretrain_steps(byol_state, byol_target_params, loop_key, pretrain_rollout_states, nr_pretrain_steps, log_each_step):
                    def byol_pretrain_step(carry, step_idx):
                        byol_state, byol_target_params, loop_key = carry
                        loop_key, rollout_idx_key, expert_idx_key, loss_key = jax.random.split(loop_key, 4)
                        rollout_idx = jax.random.randint(rollout_idx_key, (self.minibatch_size,), 0, pretrain_rollout_states.shape[0])
                        expert_idx = jax.random.randint(expert_idx_key, (self.minibatch_size,), 0, expert_byol_inputs.shape[0])
                        x_batch = jnp.concatenate([pretrain_rollout_states[rollout_idx], expert_byol_inputs[expert_idx]], axis=0)

                        loss, grads = grad_byol_loss_fn(byol_state.params, byol_target_params, x_batch, loss_key)
                        byol_state = byol_state.apply_gradients(grads=grads)
                        byol_target_params = optax.incremental_update(byol_state.params, byol_target_params, 1.0 - self.byol_tau)

                        latent = self.byol.apply(byol_state.params, x_batch, method=self.byol.encode)
                        metrics = {
                            "pretrain/byol_loss": loss,
                            "pretrain/byol_grad_norm": optax.global_norm(grads),
                            "pretrain/latent_std": jnp.mean(jnp.std(latent, axis=0)),
                            "pretrain/latent_norm": jnp.mean(jnp.linalg.norm(latent, axis=-1)),
                        }
                        if log_each_step:
                            log_pretrain_entry(metrics, step_idx, nr_pretrain_steps)
                        return (byol_state, byol_target_params, loop_key), metrics

                    return jax.lax.scan(byol_pretrain_step, (byol_state, byol_target_params, loop_key), jnp.arange(nr_pretrain_steps))

                if self.byol_pretrain_env_steps > 0:
                    nr_pretrain_rounds = -(-self.byol_pretrain_env_steps // (self.nr_envs * self.nr_steps))  # ceil

                    def byol_pretrain_round(carry, round_idx):
                        pre_env_state, rollout_key, byol_state, byol_target_params, loop_key = carry
                        (pre_env_state, rollout_key), pretrain_rollout_states = jax.lax.scan(pretrain_rollout, (pre_env_state, rollout_key), None, self.nr_steps)
                        pretrain_rollout_states = pretrain_rollout_states.reshape((-1, pretrain_rollout_states.shape[-1]))
                        (byol_state, byol_target_params, loop_key), round_metrics = run_byol_pretrain_steps(
                            byol_state, byol_target_params, loop_key, pretrain_rollout_states, self.byol_pretrain_steps_per_rollout, log_each_step=False
                        )
                        round_metrics = jax.tree_util.tree_map(jnp.mean, round_metrics)
                        log_pretrain_entry(round_metrics, round_idx, nr_pretrain_rounds)
                        return (pre_env_state, rollout_key, byol_state, byol_target_params, loop_key), None

                    (_, _, byol_state, byol_target_params, _), _ = jax.lax.scan(
                        byol_pretrain_round, (pre_env_state, pre_rollout_key, byol_state, byol_target_params, pre_loop_key), jnp.arange(nr_pretrain_rounds)
                    )
                else:
                    _, pretrain_rollout_states = jax.lax.scan(pretrain_rollout, (pre_env_state, pre_rollout_key), None, self.nr_steps)
                    pretrain_rollout_states = pretrain_rollout_states.reshape((-1, pretrain_rollout_states.shape[-1]))
                    (byol_state, byol_target_params, _), _ = run_byol_pretrain_steps(
                        byol_state, byol_target_params, pre_loop_key, pretrain_rollout_states, self.byol_pretrain_steps, log_each_step=True
                    )

            def multi_learning_and_eval_save_iteration(multi_learning_and_eval_save_iteration_carry, multi_learning_iteration_step):
                policy_state, critic_state, discriminator_state, corrected_reward_state, \
                (expert_states, expert_actions, expert_next_states, expert_absorbing), \
                env_state, byol_state, byol_target_params, key = multi_learning_and_eval_save_iteration_carry

                def learning_iteration(learning_iteration_carry, learning_iteration_step):
                    policy_state, critic_state, discriminator_state, corrected_reward_state, \
                    (expert_states, expert_actions, expert_next_states, expert_absorbing), \
                    env_state, byol_state, byol_target_params, key = learning_iteration_carry

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
                        transition = (observation, env_state.actual_next_observation, action, env_state.reward, value, env_state.terminated, log_prob, action_mean, action_logstd, env_state.info)
                        if self.render:
                            def render(env_state):
                                return self.train_env.render(env_state)

                            env_state = jax.experimental.io_callback(render, env_state, env_state)

                        return (policy_state, critic_state, env_state, key), transition

                    single_rollout_carry, batch = jax.lax.scan(single_rollout, (policy_state, critic_state, env_state, key), None, self.nr_steps)
                    policy_state, critic_state, env_state, key = single_rollout_carry
                    states, next_states, actions, rewards, values, terminations, log_probs, old_action_means, old_action_logstd, infos = batch

                    """ BYOL UPDATE """
                    # (augment_state / byol_loss_fn / grad_byol_loss_fn are defined once in jitable_train_function
                    #  and shared with the optional BYOL pretraining phase)

                    # BYOL data: the FULL rollout batch (agent states only, no expert states, no minibatching).
                    # Per step: both views of all rollout states go through the online and target networks,
                    # L_BYOL is computed, the online network takes one gradient step, then the target is EMA-updated.
                    # nr_epochs_byol = number of such full-batch steps per TRIRL iteration (default 1).
                    batch_states = states.reshape((-1,) + self.os_shape)
                    batch_byol_inputs = self.byol_inputs(batch_states, actions.reshape((-1,) + self.as_shape))

                    def byol_full_batch_update(carry, _):
                        byol_state, byol_target_params, loop_key = carry
                        loop_key, rng = jax.random.split(loop_key)

                        # 1) L_BYOL on all rollout states (online: enc->proj->pred, target: enc->proj with stop-grad)
                        loss, grads = grad_byol_loss_fn(byol_state.params, byol_target_params, batch_byol_inputs, rng)

                        # 2) update the online network
                        byol_state = byol_state.apply_gradients(grads=grads)

                        # 3) EMA update of the target network: xi <- tau * xi + (1 - tau) * theta
                        byol_target_params = optax.incremental_update(
                            byol_state.params, byol_target_params, 1.0 - self.byol_tau
                        )

                        metrics = {
                            "loss/byol_loss": loss,
                            "gradients/byol_grad_norm": optax.global_norm(grads),
                        }
                        return (byol_state, byol_target_params, loop_key), metrics

                    key, byol_loop_key = jax.random.split(key)
                    carry, byol_optimization_metrics = jax.lax.scan(
                        byol_full_batch_update, (byol_state, byol_target_params, byol_loop_key), None, self.nr_epochs_byol
                    )
                    byol_state, byol_target_params, _ = carry

                    # 4) continue with TRIRL: features below come from the updated ONLINE encoder
                    byol_optimization_metrics["lr/learning_rate_byol"] = byol_state.opt_state[1].hyperparams["learning_rate"]

                    """ Discriminator Update """

                    def trirl_loss_fn(discriminator_params, state, action, expert_state, expert_action, label, expert_label,
                                        next_state=None, absorbing=None, expert_next_state=None, expert_absorbing=None, feature=None, expert_feature=None):
                        logits = self.discriminator.apply(discriminator_params, feature, state, action, next_state, absorbing)
                        expert_logits = self.discriminator.apply(discriminator_params, expert_feature, expert_state, expert_action, expert_next_state, expert_absorbing)
                        bce_agent = optax.sigmoid_binary_cross_entropy(logits, label).mean()
                        bce_expert = optax.sigmoid_binary_cross_entropy(expert_logits, expert_label).mean()

                        # Gradient penalty on an interpolated sample between the expert and agent
                        alpha = self.gp_alpha
                        gp_lambda = self.gp_lambda
                        interpolated_state = alpha * expert_state + (1 - alpha) * state
                        interpolated_action = alpha * expert_action + (1 - alpha) * action
                        interpolated_next_state = alpha * expert_next_state + (1 - alpha) * next_state
                        interpolated_abs = 0.0 * expert_absorbing # assume interpolated state to be non-absorbing
                        interpolated_feature = alpha * expert_feature + (1 - alpha) * feature
                        grad_feature, grad_state, grad_action, grad_next_state = jax.grad(lambda f, s, a, sn, ab: jnp.sum(self.discriminator.apply(discriminator_params, f, s, a, sn, ab)), argnums=(0, 1, 2, 3))(interpolated_feature, interpolated_state, interpolated_action, interpolated_next_state, interpolated_abs)
                        grad_norm = jnp.sqrt(jnp.sum(jnp.square(grad_feature)) + 1e-12)
                        gp = (grad_norm - 1.0) ** 2

                        bce_loss = bce_agent + bce_expert + gp_lambda * gp
                        metrics = {
                            "loss/discriminator_loss": bce_loss,
                            "loss/discriminator_agent_loss": bce_agent,
                            "loss/discriminator_expert_loss": bce_expert,
                            "loss/discriminator_gp": gp,
                        }
                        return bce_loss, (metrics)

                    batch_states = states.reshape((-1,) + self.os_shape)
                    batch_actions = actions.reshape((-1,) + self.as_shape)
                    batch_features = jax.lax.stop_gradient(self.byol.apply(byol_state.params, batch_byol_inputs, method=self.byol.encode)) # get features
                    byol_optimization_metrics["byol/latent_std"] = jnp.mean(jnp.std(batch_features, axis=0))
                    byol_optimization_metrics["byol/latent_norm"] = jnp.mean(jnp.linalg.norm(batch_features, axis=-1))

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
                    batch_expert_features = jax.lax.stop_gradient(self.byol.apply(byol_state.params, self.byol_inputs(batch_expert_states, batch_expert_actions), method=self.byol.encode)) # get features
                    expert_labels = jnp.ones((self.batch_size, 1), dtype=jnp.float32)
                    rollout_labels = jnp.zeros((self.batch_size, 1), dtype=jnp.float32)

                    batch_next_states = next_states.reshape((-1,) + self.os_shape)
                    batch_absorbing = terminations.reshape(-1)
                    batch_expert_next_states = expert_next_states[expert_idx]
                    batch_expert_absorbing = expert_absorbing[expert_idx]

                    vmap_trirl_loss_fn = jax.vmap(trirl_loss_fn, in_axes=(None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), out_axes=0)
                    safe_mean = lambda x: jnp.mean(x) if x is not None else x
                    mean_vmapped_trirl_loss_fn = lambda *a, **k: tree.map_structure(safe_mean, vmap_trirl_loss_fn(*a, **k))
                    grad_trirl_loss_fn = jax.value_and_grad(mean_vmapped_trirl_loss_fn, argnums=(0), has_aux=True)

                    key, subkey = jax.random.split(key)
                    batch_indices_disc = jnp.tile(jnp.arange(self.batch_size), (self.nr_epochs_disc, 1))
                    batch_indices_disc = jax.random.permutation(subkey, batch_indices_disc, axis=1, independent=True)
                    batch_indices_disc = batch_indices_disc[:, :self.nr_minibatches * self.minibatch_size]
                    batch_indices_disc = batch_indices_disc.reshape((self.nr_epochs_disc * self.nr_minibatches, self.minibatch_size))

                    def trirl_minibatch_update(carry, minibatch_indices_disc):
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
                            batch_next_states[minibatch_indices_disc],
                            batch_absorbing[minibatch_indices_disc],
                            batch_expert_next_states[minibatch_indices_disc],
                            batch_expert_absorbing[minibatch_indices_disc],
                            batch_features[minibatch_indices_disc],
                            batch_expert_features[minibatch_indices_disc],
                        )

                        discriminator_state = discriminator_state.apply_gradients(grads=discriminator_gradients)
                        carry = (discriminator_state, key)

                        return carry, (metrics)

                    init_carry = (discriminator_state, key)
                    carry, (disc_optimization_metrics) = jax.lax.scan(trirl_minibatch_update, init_carry, batch_indices_disc)
                    discriminator_state, key = carry

                    """ Reward Update """

                    # Define Apply Functions
                    def _get_log_density_ratio(inputs, discriminator_state):
                        feature, state, action, next_state, absorbing = inputs
                        logits = self.discriminator.apply(discriminator_state.params, feature, state, action, next_state, absorbing)
                        return logits

                    _get_log_density_ratio = jax.vmap(_get_log_density_ratio, in_axes=(0, None), out_axes=0)


                    _inputs = (batch_features, states.reshape((-1,) + self.os_shape), actions.reshape((-1,) + self.as_shape), next_states.reshape((-1,) + self.os_shape), terminations.flatten())
                    corr_reward = self.entropy_coef * ((1 - self.epsilon) * _get_log_density_ratio(_inputs, corrected_reward_state).reshape(rewards.shape) \
                        + self.epsilon * self.beta * _get_log_density_ratio(_inputs, discriminator_state).reshape(rewards.shape))

                    if self.handle_absorbing_states:
                        features_absorbing = jax.lax.stop_gradient(self.byol.apply(byol_state.params, self.byol_inputs(next_states.reshape((-1,) + self.os_shape), actions.reshape((-1,) + self.as_shape)), method=self.byol.encode))
                        _inputs_absorbing = (features_absorbing, next_states.reshape((-1,) + self.os_shape), actions.reshape((-1,) + self.as_shape), next_states.reshape((-1,) + self.os_shape), jnp.ones_like(terminations.flatten()))
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

                    combined_learning_iteration_step = (multi_learning_iteration_step * self.nr_updates_per_multi_learning_iteration) + learning_iteration_step + 1
                    global_step = combined_learning_iteration_step * self.nr_steps * self.nr_envs
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
                    new_params = {
                        **updated_params,
                        "params": new_inner_params,
                    }
                    corrected_reward_state = discriminator_state.replace(params=new_params)

                    # Logging
                    combined_metrics = {**infos, **byol_optimization_metrics, **disc_optimization_metrics, **ppo_optimization_metrics}
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

                    return (policy_state, critic_state, discriminator_state, corrected_reward_state, (expert_states, expert_actions, expert_next_states, expert_absorbing), env_state, byol_state, byol_target_params, key), corrected_reward_state.params['params']['theta']  # the reward theta the policy uses (retraining / curriculum)

                key, subkey = jax.random.split(key)
                learning_iteration_carry, inner_thetas = jax.lax.scan(learning_iteration, (policy_state, critic_state, discriminator_state, corrected_reward_state,
                                                                                (expert_states, expert_actions, expert_next_states, expert_absorbing),
                                                                                env_state, byol_state, byol_target_params, subkey), jnp.arange(self.nr_updates_per_multi_learning_iteration))

                policy_state, critic_state, discriminator_state, corrected_reward_state, \
                (expert_states, expert_actions, expert_next_states, expert_absorbing), \
                env_state, byol_state, byol_target_params, key = learning_iteration_carry


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
                    # best = lowest deterministic eval task error (pos+orient)
                    # eval_task_err_for_save = (eval_metrics["eval/task_err"]
                    #                           if self.evaluation_active else jnp.asarray(jnp.inf))
                    eval_return_for_save = (eval_metrics["eval/episode_return"]
                                            if self.evaluation_active else jnp.asarray(-jnp.inf))

                    #eval_task_err_for_save = jnp.zeros(self.nr_envs)
                    # def save_with_check(policy_state, critic_state, discriminator_state, corrected_reward_state, eval_task_err):
                    #     self.save(policy_state, critic_state, discriminator_state, corrected_reward_state)  # latest.model
                    #     if self.evaluation_active:
                    #         te = float(np.asarray(eval_task_err).reshape(-1)[0])
                    #         if te < self.best_eval_task_err:
                    #             self.best_eval_task_err = te
                    #             self.save(policy_state, critic_state, discriminator_state, corrected_reward_state,
                    #                       file_name=self.best_model_file_name)
                    #             rlx_logger.info(f"[save-best] new best eval/task_err={te:.4f} -> {self.best_model_file_name}")
                    # jax.debug.callback(save_with_check, policy_state, critic_state, discriminator_state, corrected_reward_state, eval_task_err_for_save)

                    def save_with_check(policy_state, critic_state, discriminator_state, corrected_reward_state,
                                        byol_state, byol_target_params, eval_return):
                        self.save(policy_state, critic_state, discriminator_state,
                                  corrected_reward_state, byol_state, byol_target_params)  # latest.model

                        if self.evaluation_active:
                            current_return = float(np.asarray(eval_return).reshape(-1)[0])

                            # Check if the current return is GREATER than the best return
                            if current_return > self.best_eval_return:
                                self.best_eval_return = current_return
                                self.save(policy_state, critic_state, discriminator_state, corrected_reward_state,
                                          byol_state, byol_target_params, file_name=self.best_model_file_name)
                                rlx_logger.info(
                                    f"[save-best] new best eval/episode_return={current_return:.4f} -> {self.best_model_file_name}")

                    jax.debug.callback(save_with_check, policy_state, critic_state, discriminator_state,
                                       corrected_reward_state, byol_state, byol_target_params, eval_return_for_save)
                    jax.debug.callback(self.save_thetas, inner_thetas, multi_learning_iteration_step)

                return (policy_state, critic_state, discriminator_state, corrected_reward_state, (expert_states, expert_actions, expert_next_states, expert_absorbing), env_state, byol_state, byol_target_params, key), inner_thetas


            _, all_thetas =  jax.lax.scan(multi_learning_and_eval_save_iteration, (policy_state, critic_state, discriminator_state, corrected_reward_state,
                                                                  (expert_states, expert_actions, expert_next_states, expert_absorbing),
                                                                  env_state, byol_state, byol_target_params, key), jnp.arange(self.nr_multi_learning_and_eval_save_iterations))

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


    def log_byol_pretraining(self, metrics, entry_idx, nr_entries, parallel_seed_id):
        """Host-side logging of one BYOL pretraining entry, called live from inside the pretraining scan.
        An entry is one BYOL step, or one rollout round (metrics averaged over its steps) if byol_pretrain_env_steps > 0.
        Only every byol_pretrain_log_frequency-th entry and the last one are logged (TensorBoard, W&B, console)."""
        step = int(np.asarray(entry_idx))
        is_last = step == nr_entries - 1
        if step % max(1, self.byol_pretrain_log_frequency) != 0 and not is_last:
            return
        values = {k: float(np.asarray(v)) for k, v in metrics.items()}
        unit = "round" if self.byol_pretrain_env_steps > 0 else "step"
        if self.track_tb:
            for name, value in values.items():
                self.writer.add_scalar(name, value, step + 1)
        if self.track_wandb:
            # rl_x plots every metric against global_step (define_metric("*", step_metric="global_step")), so entries
            # without it get no chart. During pretraining global_step is the pretraining step/round; pretrain/* only exist here.
            wandb.log({"global_step": step + 1, "pretrain/step": step + 1, **values})
        if self.track_console:
            rlx_logger.info(f"[byol-pretrain] {unit} {step + 1}/{nr_entries} | " + " | ".join(f"{k.split('/')[-1]}={v:.4f}" for k, v in values.items()))
        if is_last:
            # Reset the SPS timer so the first TRIRL update does not count the pretraining time
            self.last_time[int(np.asarray(parallel_seed_id))] = time.time()

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

    def save(self, policy_state, critic_state, discriminator_state, corrected_reward_state, byol_state, byol_target_params, file_name=None):
        file_name = file_name or self.latest_model_file_name
        checkpoint = {
            "policy": policy_state,
            "critic": critic_state,
            "discriminator": discriminator_state,
            "corrected_reward": corrected_reward_state,
            "byol": byol_state,
            "byol_target": byol_target_params,
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
            "byol": model.byol_state,
            "byol_target": model.byol_target_params,
        }
        restore_args = orbax_utils.restore_args_from_target(target)
        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        checkpoint = checkpointer.restore(checkpoint_dir, item=target, restore_args=restore_args)

        model.policy_state = checkpoint["policy"]
        model.critic_state = checkpoint["critic"]
        model.discriminator_state = checkpoint["discriminator"]
        model.corrected_reward_state = checkpoint["corrected_reward"]
        model.byol_state = checkpoint["byol"]
        model.byol_target_params = checkpoint["byol_target"]

        shutil.rmtree(checkpoint_dir)

        return model

    def make_reward_function(self):
        discriminator = self.discriminator
        corrected_reward_params = self.corrected_reward_state.params
        byol = self.byol
        byol_params = self.byol_state.params
        entropy_coef = self.entropy_coef
        byol_inputs = self.byol_inputs

        def reward_function(state, action, next_state, absorbing, log_prob=None):
            feature = byol.apply(byol_params, byol_inputs(state, action), method=byol.encode)
            params = corrected_reward_params

            reward = discriminator.apply(params, feature, state, action, next_state, absorbing)
            return entropy_coef * reward

        reward_function = jax.vmap(reward_function, in_axes=(0, 0, 0, 0, None), out_axes=0)
        return reward_function

    def make_reward_function_fc(self):
        discriminator = self.discriminator
        byol = self.byol
        byol_params = self.byol_state.params
        entropy_coef = self.entropy_coef
        base_params = self.corrected_reward_state.params
        byol_inputs = self.byol_inputs

        def reward_function(theta, state, action, next_state, absorbing, log_prob=None):
            feature = byol.apply(byol_params, byol_inputs(state, action), method=byol.encode)
            params = {
                **base_params,
                "params": {
                    **base_params["params"],
                    "theta": theta  # Override base theta with the curriculum theta
                }
            }

            reward = discriminator.apply(params, feature, state, action, next_state, absorbing)
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
