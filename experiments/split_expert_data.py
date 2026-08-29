import os
import shutil
import numpy as np
import jax
import orbax.checkpoint
from flax.training import orbax_utils
from flax.traverse_util import flatten_dict


def convert_model_to_npz(model_path, output_path="best_model.npz"):
    """
    Takes a zipped Orbax checkpoint, extracts it, builds a dummy structure
    to bypass hardware/sharding mismatches, and saves it to an .npz file.
    """
    model_path = os.path.abspath(model_path)
    base_dir = os.path.dirname(model_path)
    tmp_dir = os.path.join(base_dir, "tmp_extraction")

    # Unpack the zip archive into a temporary folder
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    print(f"Unpacking {model_path}...")
    shutil.unpack_archive(model_path, tmp_dir, "zip")

    print("Reading Orbax checkpoint structure...")
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()

    # Get the structure (shapes and dtypes) of the checkpoint without loading the actual data
    structure = checkpointer.structure(tmp_dir)

    # Create a "dummy target" of pure NumPy zeros matching the checkpoint's shape.
    # This tricks Orbax into loading the weights as plain CPU arrays, ignoring saved GPU sharding.
    def create_dummy_array(leaf):
        if hasattr(leaf, 'shape') and hasattr(leaf, 'dtype'):
            return np.zeros(leaf.shape, dtype=leaf.dtype)
        return leaf

    dummy_target = jax.tree_util.tree_map(create_dummy_array, structure)

    # Generate restore_args from our un-sharded dummy target
    restore_args = orbax_utils.restore_args_from_target(dummy_target)

    print("Restoring checkpoint weights to CPU...")
    # Restore the actual checkpoint data safely using the dummy blueprint
    raw_checkpoint = checkpointer.restore(tmp_dir, item=dummy_target, restore_args=restore_args)

    print("Converting to flat NumPy arrays...")
    numpy_weights = {}

    # Flatten the nested dictionaries and ensure they are standard NumPy arrays
    flat_params = flatten_dict(raw_checkpoint, sep='_')

    for key, value in flat_params.items():
        numpy_weights[key] = np.array(value)

    # Save to .npz
    np.savez_compressed(output_path, **numpy_weights)
    print(f"Success! Model weights saved to '{output_path}'.")

    # Clean up the temporary extracted folder
    shutil.rmtree(tmp_dir)

    return output_path


# Run the function
if __name__ == "__main__":
    convert_model_to_npz(
        "runs/different-boltzmann-design-base-features-ppo-fb/pusht_ppo_fb/1784559157/models/best.model.zip",
        "best_model.npz"
    )