"""Pick K diverse demos from a full pizza demo set (k-means over the randomisation: centre distance and bearing, radius,
spiral pitch, speed and tool start; the demo nearest each centroid) and plot the selection.

    python scripts/pizza/select_demos.py FULL.npz OUT.npz PLOT_DIR [K]      (K = 100 by default)
"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

FULL, OUT, PLOTS = sys.argv[1:4]; K = int(sys.argv[4]) if len(sys.argv) > 4 else 100
d = dict(np.load(FULL)); S, ab, m = d["states"], d["absorbing"], d["demo_meta"]
ends = np.where(ab > 0)[0]; starts = np.r_[0, ends[:-1] + 1]
cx, cy, R, pitch, speed = m[:, 0], m[:, 1], m[:, 2], m[:, 3], m[:, 4]
s0 = S[starts, :2] - m[:, :2]; ang0 = np.arctan2(s0[:, 1], s0[:, 0]); r0 = np.linalg.norm(s0, axis=1) / R
X = np.stack([np.hypot(cx, cy), np.arctan2(cy, cx), R, pitch, speed, np.cos(ang0) * r0, np.sin(ang0) * r0], 1)
X = (X - X.mean(0)) / X.std(0)
best = None
for seed in range(8):   # k-means++ init, Lloyd iterations, keep the lowest inertia
    rng = np.random.default_rng(seed); C = [X[rng.integers(len(X))]]
    for _ in range(K - 1):
        d2 = np.min(((X[:, None] - np.array(C)[None]) ** 2).sum(-1), 1); C.append(X[rng.choice(len(X), p=d2 / d2.sum())])
    C = np.array(C)
    for _ in range(100):
        lab = np.argmin(((X[:, None] - C[None]) ** 2).sum(-1), 1)
        C2 = np.array([X[lab == k].mean(0) if (lab == k).any() else C[k] for k in range(K)])
        if np.allclose(C2, C): break
        C = C2
    inertia = ((X - C[lab]) ** 2).sum()
    if best is None or inertia < best[0]: best = (inertia, C)
sel = []
for k in range(K):
    sel.append(next(int(i) for i in np.argsort(((X - best[1][k]) ** 2).sum(-1)) if int(i) not in sel))
sel = np.array(sorted(sel))
idx = np.concatenate([np.arange(starts[i], ends[i] + 1) for i in sel])
out = {k: d[k][idx] for k in ("states", "actions", "next_states", "rewards", "absorbing")}
out.update(episode_lengths=d["episode_lengths"][sel], demo_meta=m[sel], demo_meta_cols=d["demo_meta_cols"], source_demo_index=sel)
np.savez_compressed(OUT, **out)
mm = m[sel]
lines = [f"{K} demos, {len(idx)} transitions (selected from {len(m)} by k-means over the randomisation)"]
for name, a, b in (("pizza radius (cm)", R * 100, mm[:, 2] * 100), ("centre distance from robot base (m)", np.hypot(cx, cy), np.hypot(mm[:, 0], mm[:, 1])),
                   ("centre bearing (deg)", np.degrees(np.arctan2(cy, cx)), np.degrees(np.arctan2(mm[:, 1], mm[:, 0]))),
                   ("spiral pitch (cm)", pitch * 100, mm[:, 3] * 100), ("speed (m/s)", speed, mm[:, 4]), ("touchdown step", m[:, 6], mm[:, 6]),
                   ("lift-off step", m[:, 9], mm[:, 9]), ("final sauce coverage", m[:, 7], mm[:, 7])):
    lines.append(f"| {name} | {a.min():.2f}-{a.max():.2f} (mean {a.mean():.2f}) | {b.min():.2f}-{b.max():.2f} (mean {b.mean():.2f}) |")
print("\n".join(lines), flush=True)

os.makedirs(PLOTS, exist_ok=True)
f, ax = plt.subplots(1, 3, figsize=(15, 4.6))
ax[0].scatter(cx, cy, s=4, c="lightgray", label=f"all {len(m)}"); sc = ax[0].scatter(mm[:, 0], mm[:, 1], s=18, c=mm[:, 2] * 100, cmap="viridis", label=f"selected {K}")
ax[0].plot(0, 0, "k^", label="robot base"); ax[0].set_aspect("equal"); ax[0].set_title("pizza centres (colour = radius cm)"); ax[0].legend(fontsize=7); plt.colorbar(sc, ax=ax[0], fraction=0.04)
ax[1].scatter(R * 100, pitch * 100, s=4, c="lightgray"); ax[1].scatter(mm[:, 2] * 100, mm[:, 3] * 100, s=18, c="tab:red"); ax[1].set_xlabel("radius (cm)"); ax[1].set_ylabel("spiral pitch (cm)")
ax[2].scatter(np.cos(ang0) * r0, np.sin(ang0) * r0, s=4, c="lightgray"); ax[2].scatter((np.cos(ang0) * r0)[sel], (np.sin(ang0) * r0)[sel], s=18, c="tab:red")
ax[2].add_patch(plt.Circle((0, 0), 1, fill=False)); ax[2].set_aspect("equal"); ax[2].set_title("tool start relative to pizza (units of R)")
plt.tight_layout(); plt.savefig(f"{PLOTS}/selection.png", dpi=95); plt.close(f)

zh = 0.0641; crust = 0.015
ends_s = np.where(out["absorbing"] > 0)[0]; starts_s = np.r_[0, ends_s[:-1] + 1]
pick = np.argsort(mm[:, 2])[np.linspace(0, K - 1, 8).astype(int)]
f, ax = plt.subplots(2, 4, figsize=(16, 8.4))
for k, i in enumerate(pick):
    a = ax.flat[k]; s = out["states"][starts_s[i]:ends_s[i] + 1]; c0, c1, r = s[0, 3:6]; td, lift = int(mm[i, 6]), int(mm[i, 9])
    a.add_patch(plt.Circle((c0, c1), r, color="#e8c08a", alpha=0.5)); a.add_patch(plt.Circle((c0, c1), r - crust, fill=False, ls=":", color="#a0522d"))
    landed = (np.arange(len(s)) >= td) & (np.arange(len(s)) <= lift) & (s[:, 2] - zh < 0.003)
    a.plot(s[:td + 1, 0], s[:td + 1, 1], "--", lw=1.2, color="tab:blue", label="travel (5 cm up)")
    a.plot(s[landed, 0], s[landed, 1], "-", lw=1, color="tab:red", label="spreading")
    a.plot(s[lift, 0], s[lift, 1], "ks", ms=5, label="lift-off"); a.plot(s[0, 0], s[0, 1], "go", ms=6, label="start"); a.set_aspect("equal")
    a.set_title(f"R={r*100:.1f} cm, pitch {mm[i,3]*100:.1f} cm, {mm[i,4]:.2f} m/s\ntouchdown {td}, lift-off {lift}, coverage {mm[i,7]:.2f}", fontsize=8)
ax.flat[0].legend(fontsize=7, loc="lower right")
plt.suptitle("pizza_dr expert demos: travel -> touchdown -> clockwise spiral to the sauce edge (dotted) -> closing loop -> stop -> lift-off -> hover", fontsize=10)
plt.tight_layout(); plt.savefig(f"{PLOTS}/demo_paths.png", dpi=95); plt.close(f)

print("saved", OUT, "| plots in", PLOTS)
