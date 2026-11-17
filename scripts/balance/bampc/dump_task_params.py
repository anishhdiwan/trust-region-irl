"""Dump the BalanceFr3 constants the JAX env needs (IK gains, EE home, ball placement, goal), plus a placement check.

Run from a bampc-code checkout in its conda env:  python scripts/balance/bampc/dump_task_params.py OUT_DIR
"""
import os, sys, json, numpy as np, mujoco
from bampc.config import planner as planner_profiles, reward as reward_profiles, scenarios as scen
from bampc.config.numerics import load as load_numerics
from bampc.planner import PlannerConfig
from bampc.task.balance_fr3 import BalanceFr3
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
bank = scen.load("balance_fr3"); cfg = PlannerConfig(algo="mppi", **planner_profiles.load("balance_fr3", sampling="task"))
rk = reward_profiles.load("balance_fr3", sampling="task")
task = BalanceFr3(sampling_space="task", shape=bank.shape, scale=bank.scale, goal_drift=bank.goal_drift, **rk, model_config=load_numerics("balance_fr3"), ctrl_range_scale=cfg.ctrl_range_scale)
m = task.mj_model
p = {k: getattr(task, k) for k in ("k_pos", "k_yaw", "ik_damping", "ee_body_id", "max_tilt_vel", "lost_margin") if hasattr(task, k)}
p.update(ee_home_pos=task.ee_home_pos.tolist(), ee_home_quat=task.ee_home_quat.tolist(), arm_dofadr=np.asarray(task.arm_dofadr).tolist(),
         arm_qposadr=np.asarray(task.arm_qposadr).tolist(), goal_xy=list(rk.get("goal_xy", (0.0, 0.0))),
         drift=dict(radius_xy=list(bank.goal_drift.radius_xy), freq_xy=list(bank.goal_drift.freq_xy), phase=list(bank.goal_drift.phase)),
         home_qpos=m.key("home").qpos.tolist(), plate_body_id=int(m.body("plate_body").id), ball_body_id=int(m.jnt_bodyid[0]))
p = {k: (float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer,)) else v) for k, v in p.items()}
# ball placement: pose a scenario and express the ball in the plate frame at home
sc = bank.scenarios[0]; md = scen.pose(bank, task, sc); mujoco.mj_forward(m, md)
R = md.xmat[p["plate_body_id"]].reshape(3, 3); loc = R.T @ (md.xpos[p["ball_body_id"]] - md.xpos[p["plate_body_id"]])
p["ball_local_z"] = float(loc[2]); p["placement_check_xy"] = [float(loc[0]), float(loc[1]), *sc.start["block_xy"]]
# goal: the base goal pose the drift is applied to (plate-local), from the goal mocap at the scenario's start
print({k: p[k] for k in ("k_pos", "k_yaw", "ik_damping", "max_tilt_vel", "lost_margin", "goal_xy", "ball_local_z", "placement_check_xy") if k in p})
json.dump(p, open(f"{OUT}/balance_fr3_task_params.json", "w"), indent=1)
