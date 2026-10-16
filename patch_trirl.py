file_path = "/home/kukoboshi/Artificial Intelligence/IP1/trust-region-irl/trust_region_irl/algorithms/trirl_ppo_fb/flax_full_jit/trirl_ppo.py"
with open(file_path, "r") as f:
    content = f.read()

target = """                if self.render:
                    env_state = self.eval_env.render(env_state)"""

replacement = """                if self.render:
                    env_state = self.eval_env.render(env_state)
                    if hasattr(self.eval_env, "viewer") and self.eval_env.viewer is not None:
                        if getattr(self.eval_env.viewer, "reset_requested", False):
                            self.eval_env.viewer.reset_requested = False
                            self.key, subkey = jax.random.split(self.key)
                            reset_keys = jax.random.split(subkey, self.nr_envs)
                            env_state = self.eval_env.reset(reset_keys, True)"""

content = content.replace(target, replacement, 1) # Only replace the first occurrence (which is inside visualize())

with open(file_path, "w") as f:
    f.write(content)
print("Updated trirl_ppo.")

