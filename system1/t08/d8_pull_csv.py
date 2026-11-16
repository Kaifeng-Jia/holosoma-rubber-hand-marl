#!/usr/bin/env python3
"""T08 D8 CSV (docs/system1/T08_slides_assets_zh.md): per-step pelvis vs reference for scheme A and D (seed 721) on Pull +/-2
perturbed_000. Alignment and d are those of results_system1/T01b/preregistration_drift_in_task.md, computed by its code
(system1/sonic/drift_in_task.episode_metrics, series=True): reference pelvis = the paper-environment reference at
reference_frame[j] plus the environment-origin offset reference_object_pos_w[j] - reference object[reference_frame[j]]
(checked constant < 3e-6 m); d(j) = |pelvis_xy(j) - ref_pelvis_xy(j)|.

  * A: logs_system1/T01b/schemeA_main/pull_pm2/perturbed_000.trajectory.npz (scheme A is deterministic, no seed);
  * D: the laptop-5070 final evaluation of seed 721 (results_system1/T03/c2_main/per_seed_summary.csv, Pull, +/-2, D, 721:
    logs_system1/post_train/d_curve/D_pull_paper_seed721/it08050_pm2/perturbed_000.trajectory.npz).
Each scheme keeps its own executed steps (no alignment, no truncation). One row per saved row j and robot; step = j (row 0 =
initial state, row j = state after j policy steps), time_s = j / 50.

Check (task book): A's final and maximum d must equal the A Pull perturbed_000 rows of results_system1/T01b/drift_in_task/
episodes.csv; otherwise exit non-zero (stop and report).

    hssim_s1/bin/python -B system1/t08/d8_pull_csv.py --out results_system1/T08_slides
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
from system1.sonic.drift_in_task import Refs, episode_metrics  # noqa: E402

A_SUM = ROOT / "logs_system1/T01b/schemeA_main/pull_pm2/summary.json"
D_SUM = ROOT / "logs_system1/post_train/d_curve/D_pull_paper_seed721/it08050_pm2/summary.json"
DRIFT = ROOT / "results_system1/T01b/drift_in_task/episodes.csv"
COLS = ["scheme", "task", "state_id", "step", "time_s", "reference_frame", "robot", "ref_pelvis_x", "ref_pelvis_y", "pelvis_x", "pelvis_y",
        "planar_error_m", "ref_object_x", "ref_object_y", "object_x", "object_y"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    refs = Refs()
    rows, info = [], []
    for scheme, sp in (("A", A_SUM), ("D", D_SUM)):
        s = json.loads(sp.read_text())
        ep = next(e for e in s["episodes"] if e["state_id"] == "perturbed_000")
        traj = Path(ep["trajectory"])
        ms = episode_metrics(refs, "pull", traj, series=True)
        with np.load(traj, allow_pickle=False) as z:
            ro, oo = z["reference_object_pos_w"].astype(np.float64), z["object_pos_w"].astype(np.float64)
        for j in range(len(ro)):
            for m in ms:
                rows.append(dict(scheme=scheme, task="pull", state_id="perturbed_000", step=j, time_s=j / 50, reference_frame=int(m["reference_frame"][j]),
                                 robot=m["robot"], ref_pelvis_x=m["ref_pelvis_xy"][j, 0], ref_pelvis_y=m["ref_pelvis_xy"][j, 1],
                                 pelvis_x=m["pelvis_xy"][j, 0], pelvis_y=m["pelvis_xy"][j, 1], planar_error_m=float(m["d_series"][j]),
                                 ref_object_x=ro[j, 0], ref_object_y=ro[j, 1], object_x=oo[j, 0], object_y=oo[j, 1]))
        info.append(dict(scheme=scheme, summary=str(sp), summary_sha256=sha(sp), trajectory=str(traj), trajectory_sha256=sha(traj),
                         checkpoint_sha256=s["checkpoint_sha256"], states_sha256=s["states_sha256"], gpu_name=s.get("gpu_name"),
                         completed=ep["completed_reference"], executed_steps=ep["steps"], terminal_terms=ep["terminal_terms"], saved_rows=len(ro),
                         per_robot=[dict(robot=m["robot"], d_final_m=float(m["d_series"][-1]), d_max_m=float(m["d_series"].max()),
                                         d_final_cm_episodes_csv_definition=m["d_final_cm"], d_max_cm=m["d_max_cm"]) for m in ms]))
    # check against the preregistered drift table (A)
    with DRIFT.open() as f:
        dr = [r for r in csv.DictReader(f) if r["scheme"] == "A" and r["task"] == "pull" and r["state_id"] == "perturbed_000"]
    assert len(dr) == 2, dr
    check = []
    a_info = info[0]
    for r in dr:
        m = next(x for x in a_info["per_robot"] if x["robot"] == int(r["robot"]))
        ok = abs(100 * m["d_final_m"] - float(r["d_final_cm"])) < 1e-9 and abs(100 * m["d_max_m"] - float(r["d_max_cm"])) < 1e-9 \
            and Path(r["trajectory"]).resolve() == Path(a_info["trajectory"]).resolve() and int(r["executed_steps"]) == a_info["executed_steps"]
        check.append(dict(robot=int(r["robot"]), csv_d_final_cm=100 * m["d_final_m"], drift_table_d_final_cm=float(r["d_final_cm"]),
                          csv_d_max_cm=100 * m["d_max_m"], drift_table_d_max_cm=float(r["d_max_cm"]), match=ok))
    a.out.mkdir(parents=True, exist_ok=True)
    dest = a.out / "d8_pull_A_pelvis_vs_reference.csv"
    with dest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    rec = dict(csv=dest.name, csv_sha256=sha(dest), rows=len(rows), inputs=info, drift_table=str(DRIFT), drift_table_sha256=sha(DRIFT), check=check,
               definition="results_system1/T01b/preregistration_drift_in_task.md (alignment, d); code system1/sonic/drift_in_task.episode_metrics(series=True)",
               reference_loader_meta=refs.meta.get("pull"), code_sha256=sha(__file__), drift_code_sha256=sha(ROOT / "system1/sonic/drift_in_task.py"),
               command=[sys.executable, *sys.argv])
    (a.out / "d8_pull_A_pelvis_vs_reference.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
    for c in check:
        print(c)
    if not all(c["match"] for c in check):
        print("CHECK FAILED: A final / max d differ from drift_in_task/episodes.csv - stop and report", flush=True)
        return 1
    print(dest, len(rows), "rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
