"""Replay collected real-robot expert episodes in the real_pusht_mjx MuJoCo viewer.

Each episode npz stores `states` (N, 24) with the same layout the PushT env
observes: block_pos_rel_goal(3) + block_quat_rel_goal(4) + ee_pos_rel_goal(3)
+ arm_qpos(7) + arm_qvel(7), all expressed relative to the real setup's goal
frame (see franka_pusht/observation_node.py). We reconstruct world qpos (block
free joint + 7 arm joints) from that and just render each frame -- no
dynamics/IK needed. Uses the real_pusht_mjx environment's own model/viewer.

Recorded goal-relative values are added directly to the sim's own goal
placement (env.goal_pos / env.goal_quat).
"""

import argparse
import time
from pathlib import Path

import numpy as np
import mujoco

"""
Add a transformation on the T block and goal block to place them correctly to the robot.
"""
from trust_region_irl.environments.pusht_mjx.environment import PushT


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="../trirl_dataset/rl_expert/expert_data",
                         help="Directory with session_<timestamp>_episode_XXXX.npz files")
    parser.add_argument("--episodes", nargs="*", default=None,
                         help="Specific episode indices to play, e.g. 0 3 7. Default: all, in order.")
    parser.add_argument("--pause-between-episodes", type=float, default=1.0)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    episode_files = sorted(data_dir.glob("session_*_episode_*.npz"))
    if args.episodes is not None:
        wanted = {int(e) for e in args.episodes}
        episode_files = [f for f in episode_files if int(f.stem.split("_")[-1]) in wanted]
    if not episode_files:
        raise SystemExit(f"No session_*_episode_*.npz files found in {data_dir}")

    env = PushT(render=True)
    model = env.mj_model
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    for ep_file in episode_files:
        print(f"Playing {ep_file.name}")
        ep = np.load(ep_file)
        states = ep["states"]
        for state in states:
            block_pos_rel, block_quat_rel = state[0:3], state[3:7]
            arm_q = state[10:17]

            # map recorded goal-relative pose into the sim's world frame
            block_pos_world = np.asarray(env.goal_pos) + block_pos_rel
            block_quat_world = quat_mul(np.asarray(env.goal_quat), block_quat_rel)

            data.qpos[0:3] = block_pos_world + np.array([0.0, 0.0, 0.0])
            data.qpos[3:7] = block_quat_world
            data.qpos[7:14] = arm_q
            mujoco.mj_forward(model, data)

            data.light_xdir = env.light_xdir
            data.light_xpos = env.light_xpos
            env.viewer.render(data)

        time.sleep(args.pause_between_episodes)

    env.close()


if __name__ == "__main__":
    main()
