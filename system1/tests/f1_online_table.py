#!/usr/bin/env python3
"""F1 online record (no threshold): per robot, the dual-robot run (object moved away) next to the
single-robot run of the same role reference. Writes CSV + JSON."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

L = Path(sys.argv[1] if len(sys.argv) > 1 else "logs_system1/T01b")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "results_system1/T01b")
rows = []
for c in ("push", "pull", "kick", "chair", "bucket", "table"):
    f = L / "f1_online" / c / "result.json"
    d = json.loads(f.read_text()) if f.exists() else None
    for k in (0, 1):
        s = json.loads((L / "drift" / f"single_{c}_agent{k}" / "result.json").read_text())
        st = np.load(L / "drift" / f"single_{c}_agent{k}" / "trajectory.npz")
        ref = np.load(L / "refs" / f"{c}_agent{k}.sonic.npz")
        K = int(s["steps_executed"])
        rob, rf = st["body_pos14"][:K], ref["body_pos_w"][1 : K + 1]
        rel = float(np.linalg.norm((rob - rob[:, :1]) - (rf - rf[:, :1]), axis=-1).mean() * 1000)
        dxy = np.linalg.norm(rob[:, 0, :2] - rf[:, 0, :2], axis=-1)
        r = d["per_robot"][k] if d else None
        rows.append(dict(task=c, robot=k, dual_done=d is not None, dual_steps=d["steps"] if d else None,
                         dual_termination_kept=d["termination_kept"] if d else None,
                         dual_fell=r["fell"] if r else None, dual_fall_step=r["fall_step"] if r else None,
                         dual_mpjpe_g_mm=r["mpjpe_g_mm"] if r else None, dual_mpjpe_pelvis_rel_mm=r["mpjpe_pelvis_rel_mm"] if r else None,
                         dual_drift_final_mm=r["pelvis_xy_drift_final_mm"] if r else None, dual_drift_max_mm=r["pelvis_xy_drift_max_mm"] if r else None,
                         dual_pd_target_minus_qstar_max_abs=d["pd_target_minus_qstar_max_abs"] if d else None,
                         single_steps=K, single_fell=s["fell"], single_fall_step=s["fall_step"], single_mpjpe_g_mm=s["mpjpe_g_mm"],
                         single_mpjpe_pelvis_rel_mm=rel, single_drift_final_mm=float(dxy[-1] * 1000), single_drift_max_mm=float(dxy.max() * 1000)))
with open(OUT / "F1_online_dual_vs_single.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
(OUT / "F1_online_dual_vs_single.json").write_text(json.dumps(dict(
    note="no threshold (record only). Dual: SonicDualDriver in the paired env, object moved to (50, 50) m; Plan5 tasks keep the "
         "timeout term only, CORE4D tasks keep reference_horizon plus joint_bad_tracking with its thresholds set to 1e9. Single: "
         "validate_single.py, same role reference, frame 0, no object. Both GPU PhysX, TF32 off.",
    rows=rows), indent=1))
for r in rows:
    print(r["task"], r["robot"], r["dual_mpjpe_g_mm"] and round(r["dual_mpjpe_g_mm"], 1), round(r["single_mpjpe_g_mm"], 1),
          r["dual_drift_final_mm"] and round(r["dual_drift_final_mm"]), round(r["single_drift_final_mm"]), r["dual_fell"], r["single_fell"])
