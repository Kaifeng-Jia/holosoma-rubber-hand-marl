#!/usr/bin/env python3
"""T13 round 2, read-only diagnosis: where the MuJoCo closed loop leaves the Isaac one on the nominal test state of a finished
run (same state, same actor ONNX / SONIC, Isaac PD in MuJoCo, observation noise off). Compares the two nominal trajectories
(logs_system1/t13v4/finals/<run>_pm<level>/nominal.trajectory.npz and logs_system1/t13v4/mj_finals/<run>_pm<level>/...) frame by
frame over the frames both have: the object position error to the reference in each simulator, the object distance between the
simulators, the largest robot-root distance between the simulators, and the object height minus the reference's; plus the first
frame at which the object / root distances exceed 1, 5 and 10 cm. No simulation is run.

    sim2sim_s1/bin/python -B system1/t13v4/nominal_diverge.py --run c_out_table --level 2 --out results_system1/T13_runs/c_out_table/nominal_diverge.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--every", type=int, default=20, help="table row spacing (control steps, 50 Hz)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    pi = ROOT / f"logs_system1/t13v4/finals/{a.run}_pm{a.level}/nominal.trajectory.npz"
    pm = ROOT / f"logs_system1/t13v4/mj_finals/{a.run}_pm{a.level}/nominal.trajectory.npz"
    i, m = np.load(pi), np.load(pm)
    n = min(len(i["object_pos_w"]), len(m["object_pos_w"]))
    assert np.array_equal(i["reference_frame"][:n], m["reference_frame"][:n])
    err_i = np.linalg.norm(i["object_pos_w"][:n] - i["reference_object_pos_w"][:n], axis=1)
    err_m = np.linalg.norm(m["object_pos_w"][:n] - m["reference_object_pos_w"][:n], axis=1)
    obj = np.linalg.norm(m["object_pos_w"][:n] - i["object_pos_w"][:n], axis=1)
    root = np.linalg.norm(m["root_pos"][:n] - i["root_pos"][:n], axis=2).max(axis=1)
    dz_i = i["object_pos_w"][:n, 2] - i["reference_object_pos_w"][:n, 2]
    dz_m = m["object_pos_w"][:n, 2] - m["reference_object_pos_w"][:n, 2]

    def first(x, th):
        k = np.flatnonzero(x > th)
        return int(k[0]) if len(k) else None
    rows = [dict(step=int(t), time_s=round(t / 50, 2), object_err_isaac_m=round(float(err_i[t]), 4), object_err_mujoco_m=round(float(err_m[t]), 4),
                 object_mujoco_vs_isaac_m=round(float(obj[t]), 4), root_mujoco_vs_isaac_max_m=round(float(root[t]), 4),
                 object_dz_isaac_m=round(float(dz_i[t]), 4), object_dz_mujoco_m=round(float(dz_m[t]), 4))
            for t in sorted(set(range(0, n, a.every)) | {n - 1})]
    out = dict(rules=__doc__, run=a.run, level=a.level, frames_isaac=len(i["object_pos_w"]), frames_mujoco=len(m["object_pos_w"]), frames_compared=n,
               first_step_object_mujoco_vs_isaac_over_m={str(th): first(obj, th) for th in (0.01, 0.05, 0.1)},
               first_step_root_mujoco_vs_isaac_over_m={str(th): first(root, th) for th in (0.01, 0.05, 0.1)},
               inputs={str(p.relative_to(ROOT)): __import__("hashlib").sha256(p.read_bytes()).hexdigest() for p in (pi, pm)}, rows=rows)
    a.out.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: out[k] for k in ("frames_compared", "first_step_object_mujoco_vs_isaac_over_m", "first_step_root_mujoco_vs_isaac_over_m")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
