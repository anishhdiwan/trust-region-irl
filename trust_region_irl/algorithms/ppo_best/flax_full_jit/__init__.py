# RL-X's PPO (flax_full_jit), copied unchanged except that it also keeps best.model (highest eval/episode_return)
from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from trust_region_irl.algorithms.ppo_best.flax_full_jit.ppo import PPO
from trust_region_irl.algorithms.ppo_best.flax_full_jit.default_config import get_config
from rl_x.algorithms.ppo.flax_full_jit.general_properties import GeneralProperties


PPO_BEST_FLAX_FULL_JIT = extract_algorithm_name_from_file(__file__)
register_algorithm(PPO_BEST_FLAX_FULL_JIT, get_config, PPO, GeneralProperties)
