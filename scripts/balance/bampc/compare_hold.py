"""Compare BaMPC per-substep control (100 Hz) with one command held per 50 ms step (20 Hz), ball-goal error after 2 s.

    python scripts/balance/bampc/compare_hold.py HOLD0.npz HOLD1.npz      (collect_demos.py with HOLD=0 / HOLD=1, same seed)
"""
import sys, numpy as np
FAMS = ["still", "line", "ellipse", "lissajous", "waypoints"]
for lab, f in (("100 Hz (per sub-step, as the demos)", sys.argv[1]), ("20 Hz held (like the policy)", sys.argv[2])):
    r = np.load(f); d = 100 * np.linalg.norm(r["qpos"][..., 0:2] - r["goal_pos"][..., 0:2], axis=-1)[:, 40:]; fam = r["demo_meta"][:, 0].astype(int)
    print(f"{lab:38s} all {d.mean():.2f} cm | 2nd half {d[:, 60:].mean():.2f} | worst-moment median {np.median(d.max(1)):.2f} | " +
          " | ".join(f"{n} {d[fam == i].mean():.2f}" for i, n in enumerate(FAMS)))
