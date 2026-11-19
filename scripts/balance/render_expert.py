"""Render BaMPC expert demos (any length) for Balance-FR3: per goal family the demo with the median error after 2 s,
plus one combined video. Sites (the ball's marker spheres) are hidden. Writes videos and videos.json to OUT_DIR.

    MUJOCO_GL=egl python scripts/balance/render_expert.py RAW_DEMOS.npz OUT_DIR      (RAW_DEMOS: bampc/collect_demos.py output)
"""
import sys, json
from pathlib import Path
import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
from trust_region_irl.environments.balance_fr3_mjx.environment import BalanceFR3

SRC, OUT = sys.argv[1], Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
FAMS = ["still", "line", "ellipse", "lissajous", "waypoints"]
W, H_PX, CAM, HZ = 640, 480, "video", BalanceFR3.CONTROL_FREQ

r = np.load(SRC)
q, gp, gq, fam = r["qpos"], r["goal_pos"], r["goal_quat"], r["demo_meta"][:, 0].astype(int)
T = q.shape[1]
dist = np.linalg.norm(q[..., 0:2] - gp[..., 0:2], axis=-1)   # (E, T) ball vs goal, world xy

m = BalanceFR3(render=False, nr_envs=1, impl="jax").mj_model
goal_mocap = m.body_mocapid[m.body("goal").id]
renderer = mujoco.Renderer(m, H_PX, W)
scene_opt = mujoco.MjvOption(); scene_opt.sitegroup[:] = 0
data = mujoco.MjData(m)
try:
    font = ImageFont.truetype("DejaVuSans.ttf", 18)
except OSError:
    font = ImageFont.load_default()

summary, all_frames = {}, []
for f, name in enumerate(FAMS):
    idx = np.where(fam == f)[0]
    e = dist[idx, 2 * HZ:].mean(1)
    j = int(idx[np.argsort(e)[len(e) // 2]])
    summary[name] = dict(demo=j, goal_dist_after_2s_cm=float(100 * dist[j, 2 * HZ:].mean()), goal_dist_max_after_2s_cm=float(100 * dist[j, 2 * HZ:].max()),
                         family_median_after_2s_cm=float(100 * np.median(e)), family_mean_last_10s_cm=float(100 * dist[idx, -10 * HZ:].mean()))
    frames = []
    for t in range(T):
        data.qpos[:] = q[j, t]; data.mocap_pos[goal_mocap] = gp[j, t]; data.mocap_quat[goal_mocap] = gq[j, t]
        mujoco.mj_forward(m, data)
        renderer.update_scene(data, camera=CAM, scene_option=scene_opt)
        img = Image.fromarray(renderer.render()); dr = ImageDraw.Draw(img)
        for k, s in enumerate([f"expert (BaMPC MPPI)  |  goal: {name}", f"t {t / HZ:4.1f} s   ball-goal {100 * dist[j, t]:5.1f} cm"]):
            dr.text((10, 8 + 22 * k), s, fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
        frames.append(np.asarray(img))
    imageio.mimsave(OUT / f"expert_{name}.mp4", frames, fps=HZ, quality=8, macro_block_size=1)
    all_frames += frames
    print(name, json.dumps(summary[name]), flush=True)
imageio.mimsave(OUT / "expert_all_families.mp4", all_frames, fps=HZ, quality=8, macro_block_size=1)
json.dump(summary, open(OUT / "videos.json", "w"), indent=1)
print("wrote", OUT)
