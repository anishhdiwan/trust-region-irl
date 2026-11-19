"""Load the policy parameters of a saved model (TRIRL best/latest.model.zip, PPO retraining or PPO best/latest.model).

Checkpoints are orbax PyTrees zipped together with config_algorithm.json; they are saved with CPU sharding, so the
arrays are restored as numpy arrays. The policy network is the same in all of them (trirl_ppo_spr/flax_full_jit/policy.py).
"""
import shutil
import tempfile

import jax
import numpy as np
import orbax.checkpoint


def load_policy_params(path):
    """Policy params of the checkpoint archive at `path`, ready for Policy.apply."""
    tmp = tempfile.mkdtemp()
    try:
        shutil.unpack_archive(path, tmp, "zip")
        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        meta = checkpointer.metadata(tmp)
        meta = getattr(meta, "item_metadata", meta)
        meta = getattr(meta, "tree", meta)
        restore_args = jax.tree.map(lambda _: orbax.checkpoint.RestoreArgs(restore_type=np.ndarray), meta)
        checkpoint = checkpointer.restore(tmp, restore_args=restore_args)
    finally:
        shutil.rmtree(tmp)
    params = checkpoint["policy"]["params"]
    return params if "params" in params else {"params": params}
