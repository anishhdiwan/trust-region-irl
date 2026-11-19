"""Export the Balance-FR3 model used for the demos as portable XML + solver options, and verify the export.

Run from a bampc-code checkout in its conda env (MuJoCo 3.10): builds BalanceFr3 exactly as collect_demos.py, writes
  balance_fr3_model.xml       (MjSpec.to_xml; meshdir is set to "assets" at the end, the FR3 meshes shared with pusht_mjx)
  balance_fr3_opt.json        (mj_model.opt fields BaMPC writes after compiling)
  balance_fr3_ref_310.npz     (every numeric MjModel array of the compiled reference model)
then reloads the XML + options in 3.10 and reports every model field that differs from the reference.

    python scripts/balance/bampc/export_model.py OUT_DIR

The files in trust_region_irl/environments/balance_fr3_mjx/data were made with this and dump_task_params.py.
"""
import sys, os, json
import numpy as np
import mujoco
from bampc.config import planner as planner_profiles, reward as reward_profiles, scenarios as scen
from bampc.config.numerics import load as load_numerics
from bampc.planner import PlannerConfig
from bampc.task.balance_fr3 import BalanceFr3, _build_model

OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
bank = scen.load("balance_fr3")
cfg = PlannerConfig(algo="mppi", **planner_profiles.load("balance_fr3", sampling="task"))
task = BalanceFr3(sampling_space="task", shape=bank.shape, scale=bank.scale, goal_drift=bank.goal_drift,
                  **reward_profiles.load("balance_fr3", sampling="task"), model_config=load_numerics("balance_fr3"),
                  ctrl_range_scale=cfg.ctrl_range_scale)
ref = task.mj_model

_, spec, _ = _build_model(bank.shape, bank.scale)
spec.compiler.meshdir = os.path.abspath(os.path.join(os.path.dirname(spec.modelfiledir or "."), spec.compiler.meshdir or "")) if not os.path.isabs(spec.compiler.meshdir or "") else spec.compiler.meshdir
xml = spec.to_xml()
import re
xml = re.sub(r"\n\s*<default/>", "", xml)   # an empty unnamed default left by the attach/graft steps; carries no settings, and XML loading rejects it
open(f"{OUT}/balance_fr3_model.xml", "w").write(xml)
OPT = ["timestep", "impratio", "iterations", "ls_iterations", "tolerance", "ls_tolerance", "noslip_iterations", "noslip_tolerance",
       "ccd_iterations", "ccd_tolerance", "solver", "integrator", "cone", "jacobian", "disableflags", "enableflags", "o_margin"]
opt = {k: (np.asarray(getattr(ref.opt, k)).tolist()) for k in OPT if hasattr(ref.opt, k)}
json.dump(opt, open(f"{OUT}/balance_fr3_opt.json", "w"), indent=1)

def fields(m):
    out = {}
    for name in dir(m):
        if name.startswith("_"): continue
        try: v = getattr(m, name)
        except Exception: continue
        if isinstance(v, np.ndarray) and v.dtype.kind in "fiub" and v.size: out[name] = v.copy()
    return out
ref_f = fields(ref); np.savez(f"{OUT}/balance_fr3_ref_310.npz", **ref_f)

def load_xml(path):
    m = mujoco.MjModel.from_xml_path(path)
    for k, v in json.load(open(f"{OUT}/balance_fr3_opt.json")).items():
        try: setattr(m.opt, k, type(getattr(m.opt, k))(v) if np.ndim(v) == 0 else np.asarray(v))
        except Exception as e: print("opt", k, e)
    return m
m2 = load_xml(f"{OUT}/balance_fr3_model.xml")
print("reference: nq", ref.nq, "nv", ref.nv, "nu", ref.nu, "ngeom", ref.ngeom, "nbody", ref.nbody, "| xml: nq", m2.nq, "nv", m2.nv, "nu", m2.nu, "ngeom", m2.ngeom, "nbody", m2.nbody)
diff = []
for k, v in fields(m2).items():
    if k not in ref_f: continue
    r = ref_f[k]
    if r.shape != v.shape: diff.append((k, "shape", r.shape, v.shape)); continue
    d = np.max(np.abs(r.astype(float) - v.astype(float)))
    if d > 1e-6: diff.append((k, "max abs diff", float(d)))
print(f"fields compared {len(ref_f)} | differing {len(diff)}")
for x in diff[:40]: print("  ", x)
# portable copy: meshes from the env's data/assets (identical to BaMPC's FR3 meshes)
xml = re.sub(r'meshdir="[^"]*"', 'meshdir="assets"', open(f"{OUT}/balance_fr3_model.xml").read())
open(f"{OUT}/balance_fr3_model.xml", "w").write(xml)
