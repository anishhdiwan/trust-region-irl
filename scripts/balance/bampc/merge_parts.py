"""Concatenate demo-collection parts along the demo axis.

    python scripts/balance/bampc/merge_parts.py OUT.npz PART1.npz PART2.npz ...
"""
import sys
import numpy as np
out, parts = sys.argv[1], [np.load(p) for p in sys.argv[2:]]
n = [len(p["demo_meta"]) for p in parts]
merged = {}
for k in parts[0].files:
    a = parts[0][k]
    merged[k] = np.concatenate([p[k] for p in parts], 0) if a.ndim > 0 and a.shape[0] == n[0] and k != "demo_meta_cols" and k != "families" else a
merged["source_part"] = np.concatenate([np.full(m, i, np.int32) for i, m in enumerate(n)])
np.savez_compressed(out, **merged)
print(f"saved {out}: {sum(n)} demos from {len(parts)} parts | per family {np.bincount(merged['demo_meta'][:, 0].astype(int), minlength=5).tolist()} | steps {merged['qpos'].shape[1]}")
