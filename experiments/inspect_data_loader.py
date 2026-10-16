"""Print what the expert data loader actually hands training.

Calls the same `prepare_expert_data` used inside trirl_ppo.py's training
loop, then reproduces its per-iteration expert minibatch sampling
(jax.random.randint with replacement, trirl_ppo.py lines ~328-345) so you
can see exactly what one training minibatch looks like.
"""

import os
import sys
import argparse
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import jax

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trust_region_irl.algorithms.data_utils import prepare_expert_data


def print_array(name, arr, num_rows):
    arr = np.asarray(arr)
    print(f"  {name}: shape={arr.shape} dtype={arr.dtype}")
    preview = arr[:num_rows]
    print(f"    first {min(num_rows, arr.shape[0])} rows:\n{preview}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        default="../trirl_dataset/rl_expert/expert_dataset_pusht_mtp_clean_93_episodes_trirl_f32abs.npz",
    )
    parser.add_argument("--batch-size", type=int, default=512, help="Mirrors algorithm.minibatch_size")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-rows", type=int, default=3, help="Rows to preview per array")
    args = parser.parse_args()

    dataset = prepare_expert_data(args.data_path)

    print(f"\nLoaded expert data from: {args.data_path}")
    print(f"Keys present: {sorted(dataset.keys())}\n")
    for key, arr in dataset.items():
        print_array(key, arr, args.num_rows)

    num_transitions = dataset["states"].shape[0]
    print(f"\nTotal expert transitions available: {num_transitions}")

    key = jax.random.PRNGKey(args.seed)
    _, shuffle_key = jax.random.split(key)
    expert_idx = jax.random.randint(shuffle_key, (args.batch_size,), 0, num_transitions)

    print(f"\nSampled training minibatch (batch_size={args.batch_size}, seed={args.seed}):")
    print(f"  sampled indices (first {args.num_rows}): {np.asarray(expert_idx)[:args.num_rows]}")
    for key_name in ("states", "actions"):
        if key_name in dataset:
            batch = np.asarray(dataset[key_name])[np.asarray(expert_idx)]
            print_array(f"batch_expert_{key_name}", batch, args.num_rows)


if __name__ == "__main__":
    main()
