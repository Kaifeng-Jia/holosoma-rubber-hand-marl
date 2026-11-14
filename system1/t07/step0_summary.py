#!/usr/bin/env python3
"""T07 step 0: SONIC with Sharpa hands vs the rubber hand, scheme A Bin nominal - one row per run from the evaluator's
own outputs (summary.json, <state>.torques.npz when --record-torques was on). CPU only, read-only.

    hssim_s1/bin/python -B system1/t07/step0_summary.py --step0 exports_system1/cloud_sync/logs/t07_step0 --out results_system1/T07/step0

Torque saturation: |tau| >= 0.95 x the joint's torque limit (the evaluator's rule), over every physics substep of the
episode; reported for the 29 body joints (max over joints, the joints with the largest fraction, and the fraction of
substeps in which any body joint is saturated) and, for Sharpa, the 44 finger joints separately.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

RUNS = [  # output dir, robot, finger mode, note
    ("A_bucket_rubber", "rubber", "-", "first pair (no torque recording)"),
    ("A_bucket_sharpa_track", "sharpa", "track", "first pair (no torque recording)"),
    ("t07s0b_A_bucket_rubber_tq", "rubber", "-", "with torque recording"),
    ("t07s0b_A_bucket_sharpa_locked_tq", "sharpa", "locked", "with torque recording (task book: fingers locked open)"),
    ("t07s0b_A_bucket_sharpa_track_tq", "sharpa", "track", "with torque recording"),
]


def sha(p: Path):
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step0", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rows = []
    for name, robot, fm, note in RUNS:
        d = a.step0 / name
        if not (d / "summary.json").exists():
            rows.append(dict(run=name, robot=robot, finger_mode=fm, note=note, status="no summary.json"))
            continue
        s = json.loads((d / "summary.json").read_text())
        e = s["episodes"][0]
        r = dict(run=name, robot=robot, finger_mode=fm, note=note, status="complete" if s.get("complete", True) else "incomplete",
                 state_id=e["state_id"], steps=e["steps"], horizon_steps=e["horizon_steps"], completed_reference=e["completed_reference"],
                 progress=round(e["progress"], 4), terminal_terms=";".join(k for k, v in e["terminal_terms"].items() if v),
                 sonic_fell_any=e.get("sonic_fell_any"), sonic_fall_step_r0=e["sonic_fall_step_per_robot"][0],
                 sonic_fall_step_r1=e["sonic_fall_step_per_robot"][1],
                 sonic_mpjpe_mm_r0=round(e["sonic_mpjpe_mm_per_robot"][0], 1) if e["sonic_mpjpe_mm_per_robot"][0] is not None else None,
                 sonic_mpjpe_mm_r1=round(e["sonic_mpjpe_mm_per_robot"][1], 1) if e["sonic_mpjpe_mm_per_robot"][1] is not None else None,
                 pelvis_dxy_final_m_r0=round(e["sonic_pelvis_dxy_final_m_per_robot"][0], 4),
                 pelvis_dxy_final_m_r1=round(e["sonic_pelvis_dxy_final_m_per_robot"][1], 4),
                 object_rmse_3d_cm=round(e["object_rmse_3d_m"] * 100, 2), robot_urdf=s["robot_urdf"],
                 hand_reference_sha256=(s.get("system1_hand") or {}).get("reference_sha256"), states_sha256=s["states_sha256"],
                 sonic_checkpoint_sha256=s.get("sonic_checkpoint_sha256"), gpu=s.get("gpu_name"), summary_sha256=sha(d / "summary.json"))
        tq = d / f"{e['state_id']}.torques.npz"
        if tq.exists():
            z = np.load(tq, allow_pickle=True)
            T = z["torques"]  # [control step, substep, agent, dof]
            lim = z["torque_limits"]
            names = [str(x) for x in z["dof_names"]]
            sat = np.abs(T) >= 0.95 * lim[None, None, None, :]
            body = list(range(29))
            fingers = list(range(29, len(names)))
            for ag in range(T.shape[2]):
                fb = sat[:, :, ag][..., body].reshape(-1, len(body))
                frac = fb.mean(axis=0)
                top = np.argsort(-frac)[:3]
                r[f"body_sat_max_frac_r{ag}"] = round(float(frac.max()), 4)
                r[f"body_sat_top_r{ag}"] = ";".join(f"{names[body[i]]}:{frac[i]:.3f}" for i in top if frac[i] > 0)
                r[f"body_sat_any_joint_frac_r{ag}"] = round(float(fb.any(axis=1).mean()), 4)
                r[f"body_tau_over_limit_max_r{ag}"] = round(float((np.abs(T[:, :, ag][..., body]) / lim[body]).max()), 3)
                if fingers:
                    ff = sat[:, :, ag][..., fingers].reshape(-1, len(fingers))
                    r[f"finger_sat_max_frac_r{ag}"] = round(float(ff.mean(axis=0).max()), 4)
            r["torque_substeps"] = int(T.shape[0] * T.shape[1])
            r["torques_sha256"] = sha(tq)
        rows.append(r)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / "step0.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    # SONIC sample walk A001, 10 s (system1/sonic/validate_single.py --max-steps 500 --record-torques [--sharpa])
    walk = []
    for name, robot in (("walk_rubber", "rubber"), ("walk_sharpa_locked", "sharpa (fingers locked open)")):
        f = a.step0 / name / "result.json"
        if not f.exists():
            walk.append(dict(run=name, robot=robot, status="no result.json"))
            continue
        r = json.loads(f.read_text())
        fr = r["torque_saturation_fraction_per_joint"]
        top = sorted(fr.items(), key=lambda kv: -kv[1])[:3]
        walk.append(dict(run=name, robot=robot, status="complete", steps_executed=r["steps_executed"], horizon_steps=r["horizon_steps"],
                         fell=r["fell"], fall_step=r["fall_step"], mpjpe_g_mm=round(r["mpjpe_g_mm"], 2),
                         max_pelvis_dz_m=round(r["max_pelvis_dz_m"], 4), max_ee_dz_m=round(r["max_ee_dz_m"], 4),
                         pelvis_xy_err_final_m=round(r["pelvis_xy_err_final_m"], 4),
                         body_torque_saturation_max_fraction=r["body_torque_saturation_max_fraction"],
                         body_torque_saturation_any_joint_fraction=r["body_torque_saturation_any_joint_fraction"],
                         body_tau_over_limit_max=round(r["body_tau_over_limit_max"], 3),
                         top_saturated_joints=";".join(f"{k}:{v:.4f}" for k, v in top if v > 0),
                         torque_substeps=r["torque_substeps"], device=r["device"], sonic_pd=r["sonic_pd"],
                         result_sha256=sha(f), torques_sha256=sha(a.step0 / name / "torques.npz")))
    wk = []
    for r in walk:
        wk += [k for k in r if k not in wk]
    with (a.out / "walk.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=wk)
        w.writeheader()
        w.writerows(walk)
    for r in walk:
        print(r)
    for r in rows:
        print({k: r.get(k) for k in ("run", "steps", "progress", "terminal_terms", "sonic_fell_any", "sonic_mpjpe_mm_r0", "sonic_mpjpe_mm_r1",
                                     "pelvis_dxy_final_m_r0", "pelvis_dxy_final_m_r1", "body_sat_max_frac_r0", "body_sat_max_frac_r1",
                                     "body_sat_top_r0", "body_sat_top_r1", "body_tau_over_limit_max_r0", "finger_sat_max_frac_r0")})


if __name__ == "__main__":
    raise SystemExit(main())
