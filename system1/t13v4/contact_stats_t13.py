#!/usr/bin/env python3
"""T13 v4 section 5 item 5 (internal diagnostic, not for the paper): which robot parts touch the object, per robot, in the saved
per-step states of one run's final evaluations (Isaac noise off / on, MuJoCo; +/-2, +/-8). Read only, CPU, no simulation.

Exactly the rules of T12-v2 part 1 (system1/t12v2/contact_stats.py, imported, unchanged: URDF collision geometry, a link is in
contact when its signed distance to the object is >= -1.5 cm, executed rows, window from the reference's first hand contact to the
last executed row, the same per-episode fractions: per robot hand / non-hand / legs / torso / arms, both hands at once). Only the
inputs differ: the evaluation directories of this run, and the reference = the run's runtime reference as Isaac's command
sampled it (logs_system1/t13v4/isaac_dump/<run>/dump/runtime_reference.npz, T14 round 6 C).
Summary over the 50 perturbed states of each group (completed / failed / all); nominal episodes in the per-episode table only.

    OMP_NUM_THREADS=1 hssim_s1/bin/python -B system1/t13v4/contact_stats_t13.py --run d_push --workers 8
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import system1.t12v2.contact_stats as cs  # noqa: E402

CASE = {"d_push": "push", "d_table": "table", "d_table_pa": "table", "c_out_table": "table", "c_out_push": "push"}  # round 2: c_out_*


def jobs(run):
    out = []
    for name, sim, noise, lv in [(f"{run}_pm{lv}", "isaac", "off", lv) for lv in (2, 8)] + [(f"{run}_pm{lv}_noise_on", "isaac", "on", lv) for lv in (2, 8)]:
        out.append(dict(group=f"isaac_noise_{noise}_pm{lv}", simulator=sim, noise=noise, level=lv, eval_dir=ROOT / "logs_system1/t13v4/finals" / name))
    for lv in (2, 8):
        out.append(dict(group=f"mujoco_noise_off_pm{lv}", simulator="mujoco", noise="off", level=lv, eval_dir=ROOT / "logs_system1/t13v4/mj_finals" / f"{run}_pm{lv}"))
    return [j for j in out if (j["eval_dir"] / "summary.json").exists()]


def reference(run):
    z = np.load(ROOT / "logs_system1/t13v4/isaac_dump" / run / "dump/runtime_reference.npz", allow_pickle=False)
    bodies = [str(x) for x in z["body_names"]]
    pel = bodies.index("pelvis")
    return dict(N=int(z["agent_joint_pos"].shape[0]), root_pos=z["agent_body_pos_w"][:, :, pel].astype(np.float64),
                root_quat=z["agent_body_quat_w"][:, :, pel].astype(np.float64), dof=z["agent_joint_pos"].astype(np.float64),
                obj_pos=z["object_pos_w"].astype(np.float64), obj_quat=z["object_quat_w"].astype(np.float64))


G = {}


def run_job(job):
    robot, obj, ref = G["robot"], G["obj"], G["ref"]
    s = json.loads((job["eval_dir"] / "summary.json").read_text())
    rows = []
    for e in s["episodes"]:
        tr = Path(e["trajectory"])
        with np.load(tr, allow_pickle=False) as z:
            a = {k: z[k] for k in ("root_pos", "root_quat_xyzw", "dof_pos", "object_pos_w", "object_quat_xyzw", "reference_frame")}
        n = cs.executed_rows(e, len(a["dof_pos"]))
        if not np.array_equal(a["reference_frame"][:n], np.minimum(np.arange(n), ref["N"] - 1)):
            raise ValueError(f"{tr}: saved reference frame is not min(row, N-1)")
        D = cs.link_distances(robot, obj, a["root_pos"][:n], a["root_quat_xyzw"][:n], a["dof_pos"][:n], a["object_pos_w"][:n], a["object_quat_xyzw"][:n])
        rows.append(dict(group=job["group"], simulator=job["simulator"], noise=job["noise"], level_cm=job["level"], state_id=e["state_id"], kind=e["kind"],
                         completed=bool(e["completed_reference"]), executed_steps=e["steps"], executed_rows=n, window_start_row=ref["k0"],
                         **cs.window_stats(D, robot, ref["k0"], n), trajectory=str(tr), trajectory_sha256=cs.sha(tr)))
    return job, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, choices=sorted(CASE))
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    out = ROOT / "results_system1/T13_runs" / a.run
    out.mkdir(parents=True, exist_ok=True)
    js = jobs(a.run)
    # object URDF of the run's Isaac final (the MuJoCo summaries do not carry it; the MuJoCo scene is the export of that Isaac scene)
    object_urdf = json.loads((ROOT / "logs_system1/t13v4/finals" / f"{a.run}_pm2" / "run_manifest.json").read_text())["object_urdf"]
    robot = cs.Robot("urdf")
    obj = cs.Object(cs.DATA / object_urdf, workers=a.workers, geometry="urdf", task=CASE[a.run])
    ref = reference(a.run)
    D = cs.link_distances(robot, obj, ref["root_pos"], ref["root_quat"], ref["dof"], ref["obj_pos"], ref["obj_quat"])
    hand_any = (D[:, :, :2] >= cs.CONTACT_M).any(axis=(1, 2))
    assert hand_any.any(), "the reference hands never touch the object"
    ref["k0"] = int(np.flatnonzero(hand_any)[0])
    ref_row = dict(group="reference", frames=ref["N"], first_contact_frame=ref["k0"], **cs.window_stats(D, robot, ref["k0"], ref["N"]))
    G.update(robot=robot, obj=obj, ref=ref)
    eps = []
    with Pool(min(a.workers, len(js))) as pool:
        for job, rows in pool.imap_unordered(run_job, js):
            eps += rows
            print("done", job["group"], len(rows), flush=True)
    eps.sort(key=lambda r: (r["group"], r["state_id"]))
    keys = list(dict.fromkeys(k for r in eps for k in r))
    with (out / "contact_episodes.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(eps)
    metrics = [k for k in keys if k.startswith("frac_")]
    summ = [dict(ref_row, episodes="reference", n=1)]
    for g in sorted({r["group"] for r in eps}):
        sel = [r for r in eps if r["group"] == g and r["kind"] == "perturbed"]
        for grp, rs in (("completed", [r for r in sel if r["completed"]]), ("failed", [r for r in sel if not r["completed"]]), ("all", sel)):
            row = dict(group=g, episodes=grp, n=len(rs), n_empty_window=sum(1 for r in rs if r["window_rows"] == 0))
            for m in metrics:
                v = np.array([r[m] for r in rs if r.get(m) is not None], dtype=float)
                row[f"{m}_mean"] = round(float(v.mean()), 4) if len(v) else None
            summ.append(row)
    keys = list(dict.fromkeys(k for r in summ for k in r))
    with (out / "contact_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(summ)
    print(json.dumps(ref_row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
