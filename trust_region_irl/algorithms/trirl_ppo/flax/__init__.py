from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from trust_region_irl.algorithms.trirl_ppo.flax.trirl_ppo import TRIRL_PPO
from trust_region_irl.algorithms.trirl_ppo.flax.default_config import get_config
from trust_region_irl.algorithms.trirl_ppo.flax.general_properties import GeneralProperties


TRIRL_PPO_FLAX = extract_algorithm_name_from_file(__file__)
register_algorithm(TRIRL_PPO_FLAX, get_config, TRIRL_PPO, GeneralProperties)
