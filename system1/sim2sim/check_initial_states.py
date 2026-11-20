#!/usr/bin/env python3
"""T10 step 1 check: every frozen initial state (Table, Bin; +/-2 and +/-8; nominal + 50) loaded into MuJoCo (reference frame 0,
then the evaluator's yaw_transform of each robot root) versus the state Isaac actually had after its reset (the
<state>.initial.json 'actual_initial_state' of the laptop-5070 C-in seed-721 final evaluations, same state files, sha256 checked).
Thresholds (task book): joints 1e-4 rad, positions 1e-4 m. Also reported: root / object orientation angle, velocities.

    sim2sim_s1/bin/python -B system1/sim2sim/check_initial_states.py --out results_system1/T10_sim2sim/consistency
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from system1.sim2sim.mj_env import PairEnv  # noqa: E402

ISAAC = {("table", 2): "cin_table_m5kg_seed721_fresh12000_pm2", ("table", 8): "cin_table_m5kg_seed721_fresh12000_pm8",
         ("bucket", 2): "cin_bucket_m1kg_seed721_fresh12000_pm2", ("bucket", 8): "cin_bucket_m1kg_seed721_fresh12000_pm8"}
DUMP = dict(table="logs_system1/t10/isaac/d_table/dump", bucket="logs_system1/t10/isaac/d_bucket/dump")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def qangle(a, b):
    d = abs(float(np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b))))
    return 2 * np.arccos(min(1.0, d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rows, files = [], []
    for (task, lvl), name in ISAAC.items():
        idir = ROOT / "logs_system1/post_train/final" / name
        launch = json.loads((idir.parent / (name + ".launch.json")).read_text())
        argv = launch["argv"]
        states_path = Path(argv[argv.index("--states") + 1])
        summ = json.loads((idir / "summary.json").read_text())
        assert sha(states_path) == summ["states_sha256"]
        files.append(dict(task=task, level=lvl, states=str(states_path), states_sha256=summ["states_sha256"], isaac_eval_dir=str(idir.relative_to(ROOT))))
        env = PairEnv(ROOT / DUMP[task], task)
        for init in json.loads(states_path.read_text())["states"]:
            env.reset(init)
            st = env.state()
            isa = json.loads((idir / f"{init['state_id']}.initial.json").read_text())["actual_initial_state"]
            ir = np.array(isa["root_pos"])
            iq = np.array(isa["root_quat_xyzw"])
            iv = np.array(isa["root_velocity"])
            r = dict(task=task, level=lvl, state_id=init["state_id"],
                     joint_pos_max_abs_rad=float(np.abs(st["dof_pos"] - np.array(isa["dof_pos"])).max()),
                     joint_vel_max_abs=float(np.abs(st["dof_vel"] - np.array(isa["dof_vel"])).max()),
                     root_pos_max_abs_m=float(np.abs(st["root"][:, :3] - ir).max()),
                     root_ori_max_angle_rad=max(qangle(st["root"][i, 3:7], iq[i]) for i in range(2)),
                     root_lin_vel_max_abs=float(np.abs(st["root"][:, 7:10] - iv[:, :3]).max()),
                     root_ang_vel_max_abs=float(np.abs(st["root"][:, 10:13] - iv[:, 3:6]).max()),
                     object_pos_max_abs_m=float(np.abs(st["obj_pos"] - np.array(isa["object_pos_w"])).max()),
                     object_ori_angle_rad=qangle(st["obj_quat_xyzw"], np.array(isa["object_quat_xyzw"])))
            r["passed"] = (r["joint_pos_max_abs_rad"] < 1e-4 and r["root_pos_max_abs_m"] < 1e-4 and r["object_pos_max_abs_m"] < 1e-4)
            rows.append(r)
        print(task, lvl, "max joint", max(x["joint_pos_max_abs_rad"] for x in rows if x["task"] == task and x["level"] == lvl),
              "max root pos", max(x["root_pos_max_abs_m"] for x in rows if x["task"] == task and x["level"] == lvl),
              "max obj pos", max(x["object_pos_max_abs_m"] for x in rows if x["task"] == task and x["level"] == lvl), flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / "initial_states.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = {}
    for k in rows[0]:
        if k.endswith(("_rad", "_m", "_abs")):
            summary[k] = max(x[k] for x in rows)
    (a.out / "initial_states_summary.json").write_text(json.dumps(dict(n_states=len(rows), all_passed=all(x["passed"] for x in rows), max=summary,
                                                                        thresholds=dict(joint_rad=1e-4, position_m=1e-4), state_files=files), indent=1))
    print("all passed" if all(x["passed"] for x in rows) else "FAILED", len(rows), summary)


if __name__ == "__main__":
    main()
