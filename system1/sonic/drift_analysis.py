#!/usr/bin/env python3
"""Drift analysis (T01b section 2) following results_system1/T01b/preregistration_drift.md.

Inputs: single-robot runs (validate_single.py outputs) and their references
(<ref>.holosoma.npz for body/foot poses, <ref>.sonic.npz for pelvis/14-body reference).
Outputs (--out dir): drift_segments.csv, drift_per_second.csv, foot_slip.csv,
correlations.json, scatter_*.png, checks_v.json. Control-group rows (iii) are added from
--control dirs when given.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.spatial.transform import Rotation

FPS = 50
H_CONTACT = 0.03  # preregistered: contact if foot z <= clip minimum + 0.03 m
MIN_PATH = 0.10   # preregistered: below this reference path length the per-metre rates are not reported
MIN_SEC_DISP = 0.01


def yaw_wxyz(q):
    return Rotation.from_quat(np.asarray(q)[..., [1, 2, 3, 0]]).as_euler("ZYX")[..., 0]


def path_length(xy):
    return float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())


def segment_metrics(run_dir: Path, ref_prefix: Path, name: str):
    res = json.loads((run_dir / "result.json").read_text())
    t = np.load(run_dir / "trajectory.npz")
    sref = np.load(f"{ref_prefix}.sonic.npz")
    K = int(res["steps_executed"])
    rob = t["body_pos14"][:K]                     # post-step k=1..K
    ref = sref["body_pos_w"][1 : K + 1]           # reference frames 1..K
    glob = np.linalg.norm(rob - ref, axis=-1).mean() * 1000
    rel = np.linalg.norm((rob - rob[:, :1]) - (ref - ref[:, :1]), axis=-1).mean() * 1000
    ref_xy = sref["body_pos_w"][: K + 1, 0, :2]   # frames 0..K
    rob_xy = np.concatenate([ref_xy[:1], rob[:, 0, :2]])  # robot frame 0 = reference frame 0 (initial alignment check in result)
    d = np.linalg.norm(rob_xy - ref_xy, axis=1)
    L = path_length(ref_xy)
    dur = K / FPS
    rows_sec = []
    yaw = yaw_wxyz(sref["root_quat_w"][: K + 1])
    for s0 in range(0, K, FPS):
        s1 = min(s0 + FPS, K)
        dq, dr = ref_xy[s1] - ref_xy[s0], rob_xy[s1] - rob_xy[s0]
        if np.linalg.norm(dq) >= MIN_SEC_DISP:
            u = dq / np.linalg.norm(dq)
        else:
            u = np.array([np.cos(yaw[s0]), np.sin(yaw[s0])])
        n = np.array([-u[1], u[0]])
        e = dr - dq
        cos = float(dr @ dq / (np.linalg.norm(dr) * np.linalg.norm(dq))) if np.linalg.norm(dq) >= 0.05 and np.linalg.norm(dr) > 0 else None
        rows_sec.append(dict(segment=name, second=s0 // FPS, duration_s=(s1 - s0) / FPS, along_m=float(e @ u), lateral_m=float(e @ n),
                             ref_disp_m=float(np.linalg.norm(dq)), robot_disp_m=float(np.linalg.norm(dr)), cos_robot_ref=cos))
    seg = dict(segment=name, steps=K, horizon=res["horizon_steps"], fell=res["fell"], fall_step=res["fall_step"],
               mpjpe_g_mm=float(glob), mpjpe_pelvis_rel_mm=float(rel), drift_final_cm=float(d[-1] * 100), drift_max_cm=float(d.max() * 100),
               drift_rate_cm_per_s=float(d[-1] * 100 / dur), ref_path_m=L,
               drift_per_m_ref_cm=float(d[-1] * 100 / L) if L >= MIN_PATH else None,
               initial_pelvis_xy_err_m=res.get("initial_pelvis_xy_err_m"), initial_yaw_err_deg=res.get("initial_yaw_err_deg"),
               pd_target_minus_qstar_max_abs=res.get("pd_target_minus_qstar_max_abs"))
    return seg, rows_sec


def foot_slip(ref_prefix: Path, name: str):
    h = np.load(f"{ref_prefix}.holosoma.npz")
    names = [str(x) for x in h["body_names"]]
    bp = h["body_pos_w"]
    total, frac = 0.0, {}
    for foot in ("left_foot_contact_point", "right_foot_contact_point"):
        p = bp[:, names.index(foot)]
        c = p[:, 2] <= p[:, 2].min() + H_CONTACT
        both = c[1:] & c[:-1]
        total += float(np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1)[both].sum())
        frac[foot] = float(c.mean())
    L = path_length(bp[:, names.index("pelvis"), :2])
    return dict(segment=name, frames=len(bp), slip_total_m=total, ref_path_m=L,
                slip_per_m_ref_cm=total * 100 / L if L >= MIN_PATH else None,
                contact_frac_left=frac["left_foot_contact_point"], contact_frac_right=frac["right_foot_contact_point"])


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return dict(n=len(x), pearson_r=None, pearson_p=None, spearman_rho=None, spearman_p=None)
    pr, pp = stats.pearsonr(x, y)
    sr, sp = stats.spearmanr(x, y)
    return dict(n=len(x), pearson_r=float(pr), pearson_p=float(pp), spearman_rho=float(sr), spearman_p=float(sp))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, required=True, help="dir with single_<name>/ outputs")
    p.add_argument("--segments", nargs="+", required=True, help="name=ref_prefix")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--control", nargs="*", default=[], help="name=run_dir=ref_prefix for (iii)")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    segs, secs, slips = [], [], []
    for item in a.segments:
        name, ref = item.split("=")
        s, rs = segment_metrics(a.runs / f"single_{name}", Path(ref), name)
        segs.append(dict(group="sonic", **s))
        secs += rs
        slips.append(foot_slip(Path(ref), name))
    for item in a.control:
        name, run, ref = item.split("=")
        s, rs = segment_metrics(Path(run), Path(ref), name)
        segs.append(dict(group="prior_control", **s))
        secs += [dict(r, segment=f"control:{name}") for r in rs]
    write_csv(a.out / "drift_segments.csv", segs)
    write_csv(a.out / "drift_per_second.csv", secs)
    write_csv(a.out / "foot_slip.csv", slips)
    sl = {r["segment"]: r for r in slips}
    son = [s for s in segs if s["group"] == "sonic"]
    pts_m = [(sl[s["segment"]]["slip_per_m_ref_cm"], s["drift_per_m_ref_cm"], s["segment"]) for s in son
             if sl[s["segment"]]["slip_per_m_ref_cm"] is not None and s["drift_per_m_ref_cm"] is not None]
    pts_s = [(sl[s["segment"]]["slip_total_m"], s["drift_rate_cm_per_s"], s["segment"]) for s in son]
    used = {q[2] for q in pts_m}
    excluded = [s["segment"] for s in son if s["segment"] not in used]
    out = dict(primary_cm_per_m=corr([q[0] for q in pts_m], [q[1] for q in pts_m]),
               primary_excluded_short_path=excluded,
               supplementary_cm_per_s_vs_slip_m=corr([q[0] for q in pts_s], [q[1] for q in pts_s]))
    (a.out / "correlations.json").write_text(json.dumps(out, indent=1))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for fname, pts, xl, yl in (("scatter_drift_vs_slip_per_m.png", pts_m, "reference foot slip per m pelvis path (cm/m)", "SONIC drift per m reference path (cm/m)"),
                               ("scatter_drift_rate_vs_slip_total.png", pts_s, "reference foot slip total (m)", "SONIC drift rate (cm/s)")):
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for x, y, nm in pts:
            ax.scatter(x, y, color="tab:orange" if nm.startswith("sonic_sample") else "tab:blue")
            ax.annotate(nm.replace("_agent", "/").replace("sonic_sample_", "SONIC "), (x, y), fontsize=7)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(f"n={len(pts)}")
        fig.tight_layout()
        fig.savefig(a.out / fname, dpi=130)
        plt.close(fig)
    # (v) checks
    v = dict(direction=[], frame_rate=dict(reference_fps=FPS, control_dt_s=0.02, consistent=True), coordinates=[])
    for s in son:
        rs = [r for r in secs if r["segment"] == s["segment"] and r["cos_robot_ref"] is not None]
        neg = sum(r["cos_robot_ref"] < 0 for r in rs)
        v["direction"].append(dict(segment=s["segment"], seconds_with_ref_disp_ge_5cm=len(rs), seconds_cos_negative=neg,
                                   flagged=len(rs) > 0 and neg > len(rs) / 2))
        v["coordinates"].append(dict(segment=s["segment"], initial_pelvis_xy_err_m=s["initial_pelvis_xy_err_m"],
                                     initial_yaw_err_deg=s["initial_yaw_err_deg"],
                                     flagged=(s["initial_pelvis_xy_err_m"] or 0) > 0.01 or (s["initial_yaw_err_deg"] or 0) > 2.0))
    v["integration_problem_found"] = any(x["flagged"] for x in v["direction"]) or any(x["flagged"] for x in v["coordinates"])
    (a.out / "checks_v.json").write_text(json.dumps(v, indent=1))
    print(json.dumps(dict(correlations=out, integration_problem_found=v["integration_problem_found"]), indent=1))


if __name__ == "__main__":
    main()
