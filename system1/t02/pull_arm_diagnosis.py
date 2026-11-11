#!/usr/bin/env python3
"""A Pull arm diagnosis (Kevin 2026-11-11); definitions in results_system1/T02/pull_arm_diagnosis/DEFINITIONS.md
(commit f6f6338, written before computing A and D). Existing trajectories only; CPU.

    hssim_s1/bin/python -B system1/t02/pull_arm_diagnosis.py --out results_system1/T02/pull_arm_diagnosis
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
from system1.t02.object_analysis import A_CSV, V3, qmat  # noqa: E402

LEG_HALF = 0.025
LEG_Y = (-0.17835 - 0.3793 / 2, -0.17835 + 0.3793 / 2)          # leg axis, table-local y
LEG_XZ = {0: np.array([-0.23525, -0.6742736]), 1: np.array([-0.23525, 0.6742736])}  # robot -> its near-side leg (x, z)
OUTER = {0: "left", 1: "right"}                                    # outer hand per robot (reference frame 120)
T_F = 100
BEHIND_RADIUS = 0.12
BEHIND_YMAX = LEG_Y[1] + 0.05
CONTACT_GAP = 0.05
SEP_HOLD = 10
ARM_JOINTS = [f"{s}_{j}_joint" for s in ("left", "right") for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow")]
LINKS = ["left_elbow_link", "left_rubber_hand_link", "right_elbow_link", "right_rubber_hand_link"]
NSAMP = 41


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()


class Ctx:
    def __init__(self):
        import torch
        from prepare_experiments_r2_cpu import JOINTS, BatchFK
        from system1.sonic.task_refs import load_task_reference
        self.fk = BatchFK()
        self.jidx = [list(JOINTS).index(j) for j in ARM_JOINTS]
        ref, bodies, _, meta = load_task_reference("pull")
        a = {k: v.numpy().astype(np.float64) for k, v in ref.sample(torch.arange(ref.num_frames)).items()}
        pel = bodies.index("pelvis")
        self.ref = dict(root_pos=a["agent_body_pos_w"][:, :, pel], root_quat_xyzw=a["agent_body_quat_w"][:, :, pel], dof_pos=a["agent_joint_pos"],
                        object_pos_w=a["object_pos_w"], object_quat_xyzw=a["object_quat_w"], reference_frame=np.arange(ref.num_frames))
        self.ref_q = a["agent_joint_pos"]
        self.meta = meta
        T = self.fk.poses(self.ref["root_pos"], self.ref["root_quat_xyzw"], self.ref["dof_pos"], ["left_elbow_link", "left_wrist_yaw_link"])
        self.fk_check_m = float(np.abs(T[..., :3, 3] - a["agent_body_pos_w"][:, :, [bodies.index("left_elbow_link"), bodies.index("left_wrist_yaw_link")]]).max())
        d = a["object_pos_w"][-1] - a["object_pos_w"][0]
        dl = qmat(a["object_quat_w"][0]).T @ d
        self.dhat = np.array([dl[0], dl[2]]) / np.hypot(dl[0], dl[2])     # table-local horizontal (x, z)


def geometry(ctx, z):
    """per step, per robot: outer/inner hand a, b, y, horizontal distance, behind flag; arm-leg gap."""
    P = ctx.fk.poses(z["root_pos"], z["root_quat_xyzw"], z["dof_pos"], LINKS)[..., :3, 3]       # [T, 2, 4, 3]
    R = qmat(z["object_quat_xyzw"])                                                           # [T, 3, 3]
    loc = np.einsum("tji,trkj->trki", R, P - z["object_pos_w"][:, None, None, :])            # table-local, [T, 2, 4, 3]
    perp = np.array([-ctx.dhat[1], ctx.dhat[0]])
    out = {}
    for r in range(2):
        leg = LEG_XZ[r]
        for side in ("left", "right"):
            h = loc[:, r, LINKS.index(f"{side}_rubber_hand_link")]
            rel = h[:, [0, 2]] - leg
            tag = "outer" if side == OUTER[r] else "inner"
            out[(r, tag, "a")] = rel @ ctx.dhat
            out[(r, tag, "b")] = rel @ perp
            out[(r, tag, "y")] = h[:, 1]
            out[(r, tag, "dh")] = np.hypot(rel[:, 0], rel[:, 1])
        o = out
        out[(r, "behind")] = (o[(r, "outer", "a")] < 0) & (o[(r, "outer", "dh")] < BEHIND_RADIUS) & (o[(r, "outer", "y")] <= BEHIND_YMAX)
        gaps = []
        for side in ("left", "right"):
            e = loc[:, r, LINKS.index(f"{side}_elbow_link")]
            h = loc[:, r, LINKS.index(f"{side}_rubber_hand_link")]
            s = np.linspace(0, 1, NSAMP)[None, :, None]
            pts = e[:, None, :] + s * (h - e)[:, None, :]                                       # [T, S, 3]
            dh = np.hypot(pts[..., 0] - leg[0], pts[..., 2] - leg[1])
            dy = np.maximum(0, np.maximum(LEG_Y[0] - pts[..., 1], pts[..., 1] - LEG_Y[1]))
            gaps.append(np.hypot(dh, dy).min(axis=1) - LEG_HALF)
        out[(r, "gap")] = np.minimum(*gaps)
    return out


def separation(gap, t0):
    for t in range(t0, len(gap)):
        if gap[t] > CONTACT_GAP:
            hold = gap[t:t + SEP_HOLD]
            if (hold > CONTACT_GAP).all() and (len(hold) == SEP_HOLD or t + len(hold) == len(gap)):
                return t
    return None


def first(mask, t0):
    i = np.nonzero(mask[t0:])[0]
    return int(i[0]) + t0 if len(i) else None


def arm_dev(ctx, z, t_lo, t_hi):
    """mean |q - q_ref| (rad) per robot per arm joint over steps t_lo..t_hi (inclusive)."""
    if t_hi < t_lo:
        return None
    f = z["reference_frame"].astype(int)
    q = z["dof_pos"][t_lo:t_hi + 1][:, :, ctx.jidx]
    qr = ctx.ref_q[f[t_lo:t_hi + 1]][:, :, ctx.jidx]
    return np.abs(q - qr).mean(axis=0)                                                       # [2, 8]


def episodes():
    eps = []
    for r in csv.DictReader(open(A_CSV)):
        if r["experiment_group"] == "System1-main" and r["task"] == "pull" and r["perturbation_level_cm"] == "2":
            eps.append(dict(scheme="A", seed="", state_id=r["state_id"], is_nominal=r["is_nominal"] == "True", steps=int(r["executed_steps"]),
                            completed=r["completed_reference"] == "True", traj=r["trajectory_path"]))
    for r in csv.DictReader(open(V3)):
        if r["variant"] == "Full" and r["task"] == "pull" and r["perturbation_level_cm"] == "2" and r["training_seed"] in ("721", "722", "723"):
            eps.append(dict(scheme="D", seed=r["training_seed"], state_id=r["state_id"], is_nominal=r["is_nominal"] == "True", steps=int(r["executed_steps"]),
                            completed=r["completed_reference"] == "True", traj=r["trajectory_path"]))
    return eps


def load(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k].astype(np.float64) if z[k].dtype.kind == "f" else z[k] for k in
                ("root_pos", "root_quat_xyzw", "dof_pos", "object_pos_w", "object_quat_xyzw", "reference_frame")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    ctx = Ctx()
    eps = episodes()
    a_steps = {e["state_id"]: e["steps"] for e in eps if e["scheme"] == "A"}
    rows, ts = [], []
    items = [dict(scheme="reference", seed="", state_id="reference", is_nominal=True, steps=len(ctx.ref["reference_frame"]) - 1, completed=True,
                  traj=None)] + eps
    for e in items:
        z = ctx.ref if e["traj"] is None else load(e["traj"])
        if e["traj"] is not None and len(z["dof_pos"]) != e["steps"] + 1:
            raise RuntimeError(f"rows/steps {e['traj']}")
        g = geometry(ctx, z)
        T = len(z["dof_pos"]) - 1
        devP = arm_dev(ctx, z, 1, min(T_F - 1, T))
        devF = arm_dev(ctx, z, T_F, T) if T >= T_F else None
        devFm = None
        if e["scheme"] == "D":
            devFm = arm_dev(ctx, z, T_F, min(a_steps[e["state_id"]], T))
        for r in range(2):
            row = dict(scheme=e["scheme"], seed=e["seed"], state_id=e["state_id"], is_nominal=e["is_nominal"], executed_steps=e["steps"],
                       completed_reference=e["completed"], robot=r, reached_t_f=T >= T_F,
                       trajectory=e["traj"] or "reference", trajectory_sha256=sha(e["traj"]) if e["traj"] else ctx.meta["sha256"])
            for t, tag in ((0, "t0"), (T_F, "tf")):
                if t > T:
                    continue
                for h in ("outer", "inner"):
                    for q in ("a", "b", "y", "dh"):
                        row[f"{tag}_{h}_{q}_m"] = float(g[(r, h, q)][t])
                row[f"{tag}_behind"] = bool(g[(r, "behind")][t])
                row[f"{tag}_arm_leg_gap_m"] = float(g[(r, "gap")][t])
            if T >= T_F:
                row["separation_step"] = separation(g[(r, "gap")], T_F)
                row["outer_leaves_behind_step"] = first(g[(r, "outer", "a")] >= 0, T_F)
                row["behind_fraction_force_phase"] = float(g[(r, "behind")][T_F:].mean())
                row["contact_fraction_force_phase"] = float((g[(r, "gap")][T_F:] <= CONTACT_GAP).mean())
            for k, j in enumerate(ARM_JOINTS):
                row[f"dev_pre_{j}_deg"] = float(np.degrees(devP[r, k])) if devP is not None else None
                row[f"dev_force_{j}_deg"] = float(np.degrees(devF[r, k])) if devF is not None else None
                if devFm is not None:
                    row[f"dev_force_matchedA_{j}_deg"] = float(np.degrees(devFm[r, k]))
            rows.append(row)
            if e["is_nominal"] or e["scheme"] == "reference":
                f = z["reference_frame"].astype(int)
                for t in range(T + 1):
                    x = dict(scheme=e["scheme"], seed=e["seed"], robot=r, step=t, outer_a_m=float(g[(r, "outer", "a")][t]),
                             outer_y_m=float(g[(r, "outer", "y")][t]), outer_dh_m=float(g[(r, "outer", "dh")][t]),
                             inner_a_m=float(g[(r, "inner", "a")][t]), arm_leg_gap_m=float(g[(r, "gap")][t]), behind=bool(g[(r, "behind")][t]))
                    for k, j in enumerate(ARM_JOINTS):
                        x[f"dev_{j}_deg"] = float(np.degrees(abs(z["dof_pos"][t, r, ctx.jidx[k]] - ctx.ref_q[f[t], r, ctx.jidx[k]])))
                    ts.append(x)
    a.out.mkdir(parents=True, exist_ok=True)

    def wcsv(name, rr):
        fields = list(dict.fromkeys(k for x in rr for k in x))
        with (a.out / name).open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for x in rr:
                w.writerow({k: ("" if x.get(k) is None else x.get(k)) for k in fields})
        print("wrote", a.out / name, len(rr))
    wcsv("per_episode_robot.csv", rows)
    wcsv("nominal_timeseries.csv", ts)

    grp = defaultdict(list)
    for r in rows:
        if r["scheme"] == "reference" or not r["is_nominal"]:
            grp[(r["scheme"], r["seed"])].append(r)
    summ = []
    for (sch, sd), rs in sorted(grp.items()):
        reach = [r for r in rs if r["reached_t_f"]]
        x = dict(scheme=sch, seed=sd, n_episode_robots=len(rs), n_reached_t_f=len(reach),
                 t0_behind=sum(r["t0_behind"] for r in rs), tf_behind=sum(r["tf_behind"] for r in reach),
                 t0_gap_median_m=float(np.median([r["t0_arm_leg_gap_m"] for r in rs])),
                 tf_gap_median_m=float(np.median([r["tf_arm_leg_gap_m"] for r in reach])) if reach else None,
                 tf_outer_a_median_m=float(np.median([r["tf_outer_a_m"] for r in reach])) if reach else None,
                 tf_outer_y_median_m=float(np.median([r["tf_outer_y_m"] for r in reach])) if reach else None,
                 n_separated=sum(r.get("separation_step") is not None for r in reach),
                 separation_step_median=float(np.median([r["separation_step"] for r in reach if r.get("separation_step") is not None]))
                 if any(r.get("separation_step") is not None for r in reach) else None,
                 n_outer_leaves_behind=sum(r.get("outer_leaves_behind_step") is not None for r in reach),
                 outer_leaves_behind_step_median=float(np.median([r["outer_leaves_behind_step"] for r in reach if r.get("outer_leaves_behind_step") is not None]))
                 if any(r.get("outer_leaves_behind_step") is not None for r in reach) else None,
                 contact_fraction_force_phase_mean=float(np.mean([r["contact_fraction_force_phase"] for r in reach])) if reach else None,
                 executed_steps_median=float(np.median([r["executed_steps"] for r in rs])))
        for j in ARM_JOINTS:
            for ph in ("pre", "force", "force_matchedA"):
                v = [r.get(f"dev_{ph}_{j}_deg") for r in rs if r.get(f"dev_{ph}_{j}_deg") is not None]
                if v:
                    x[f"dev_{ph}_{j}_deg_mean"] = float(np.mean(v))
        summ.append(x)
    wcsv("summary.csv", summ)
    (a.out / "run.json").write_text(json.dumps(dict(argv=sys.argv, reference=ctx.meta, fk_check_vs_reference_bodies_max_m=ctx.fk_check_m,
                                                    pull_direction_table_local_xz=ctx.dhat.tolist(), t_f=T_F, contact_gap_m=CONTACT_GAP,
                                                    behind_radius_m=BEHIND_RADIUS, behind_ymax_m=BEHIND_YMAX, separation_hold_steps=SEP_HOLD,
                                                    n_episodes=len(eps)), indent=1) + "\n")


if __name__ == "__main__":
    main()
