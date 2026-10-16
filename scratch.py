import re

file_path = "/home/kukoboshi/Artificial Intelligence/IP1/trust-region-irl/trust_region_irl/algorithms/trirl_ppo_fb/flax_full_jit/trirl_ppo.py"
with open(file_path, "r") as f:
    content = f.read()

# Split into part before test_irl and test_irl itself
parts = content.split("def test_irl(self, episodes):")
if len(parts) == 2:
    pre = parts[0]
    post = "def test_irl(self, episodes):" + parts[1]
    
    # In test_irl, replace self.train_env with self.eval_env
    post_mod = post.replace("self.train_env", "self.eval_env")
    
    with open(file_path, "w") as f:
        f.write(pre + post_mod)
    print("Modified successfully.")
else:
    print("Could not find test_irl.")

