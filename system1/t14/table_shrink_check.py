#!/usr/bin/env python3
"""T14 (Kevin 2026-11-24, item 2): Table at k < 1 - is the hovering mostly vertical, and are the legs / waist held at the
1.0x height by the prior? Verification only (no method change).

For each reference (Table, k = 0.8 / 0.9 / 1.0) and each robot, in the person's hand-contact frames:
  * error vector e = robot palm point - target (contact_error3d.py), in world axes: vertical part e_z, horizontal |e_xy|,
    share of vertical |e_z| / |e| (per hand, mean);
  * table: top-surface height of the tabletop (world z, mean over the contact frames) = object pose applied to the top face of
    the training tabletop box x k;
  * robot: pelvis (root) height, waist pitch, left / right hip pitch and knee (mean, deg), and the same of the nominal prior
    (the person's Stage-1 solution shifted horizontally, run_record), and the largest |q - q_prior| over the prior's 22 qpos
    in these frames (root horizontal and vertical in cm, joints in deg); elbow (mean, min; 0 = straight) and shoulder pitch;
  * robot palm height above the pelvis (mean of both hands) and the person's palm height above the person's pelvis (in the
    person-scaled source = the solve's human input), so the two bends can be compared.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "system1/t14"))
import contact_error3d as ce  # noqa: E402
import contact_gap as cg  # noqa: E402

TOP_LOCAL_Y = 0.2531295 + 0.0174189  # training tabletop box (desk001 x s_bar): centre y + half thickness, object frame


def check(path, solve_dir, k, robot, offs, human):
    z = np.load(path)
    pair, oq = z["robot_qpos"].astype(np.float64), z["object_qpos"].astype(np.float64)
    t15 = np.arange(min(len(human["flags"]), (len(pair) + 1) // 2))
    P, O = pair[2 * t15], oq[2 * t15]
    Ro = cg.cs.pose_mat(O[:, :3], np.concatenate([O[:, 4:7], O[:, 3:4]], -1))[:, :3, :3]
    pal = cg.robot_palms_local(robot, P, O, offs)
    s = ce.S_BAR["table"] * k
    e_w = np.einsum("tij,trhj->trhi", Ro, pal - s * human["target"][t15])
    top = O[:, 2] + (Ro @ np.array([0.0, TOP_LOCAL_Y * k, 0.0]))[:, 2]
    names = robot.dofs
    j = lambda n: 7 + names.index(n)
    res = {}
    for r in (0, 1):
        rec = json.loads((Path(solve_dir) / f"person{r + 1}/run_record.json").read_text())
        q1 = np.load(rec["stage1"])["qpos"][:, :36].astype(np.float64)
        q1[:, :2] += np.array(rec["palm"]["shift_xy_m"])
        sq = np.load(Path(solve_dir) / f"person{r + 1}/person{r + 1}_new_qpos.npz")
        hum = sq["human_joints"][2 * t15]  # the solve's (person-scaled) human input; 0 pelvis, 22/23 palms
        f = human["flags"][t15, r].any(1)
        Q, Qp = P[f, r], q1[2 * t15][f]
        hands = {}
        for h, side in enumerate(("left", "right")):
            fh = human["flags"][t15, r, h]
            ez, exy = e_w[fh, r, h, 2], np.linalg.norm(e_w[fh, r, h, :2], axis=1)
            hands[side] = dict(frames=int(fh.sum()), vertical_mean_cm=round(100 * float(ez.mean()), 2), horizontal_mean_cm=round(100 * float(exy.mean()), 2),
                               vertical_share=round(float(np.mean(np.abs(ez) / np.maximum(np.hypot(ez, exy), 1e-9))), 3))
        tl = robot.fk(P[f, r, :3], np.concatenate([P[f, r, 4:7], P[f, r, 3:4]], -1), P[f, r, 7:])
        palm_w = np.stack([tl[f"{sd}_rubber_hand_link"][:, :3, :3] @ offs[sd] + tl[f"{sd}_rubber_hand_link"][:, :3, 3] for sd in ("left", "right")], 1)
        dq = np.abs(Q[:, :22] - Qp[:, :22])
        res[f"r{r}"] = dict(contact_frames=int(f.sum()), hands=hands, table_top_z_m=round(float(top[f].mean()), 4),
                            pelvis_z_m=round(float(Q[:, 2].mean()), 4), prior_pelvis_z_m=round(float(Qp[:, 2].mean()), 4),
                            waist_pitch_deg=round(float(np.degrees(Q[:, j("waist_pitch_joint")]).mean()), 1), prior_waist_pitch_deg=round(float(np.degrees(Qp[:, j("waist_pitch_joint")]).mean()), 1),
                            hip_pitch_deg=[round(float(np.degrees(Q[:, j(f"{sd}_hip_pitch_joint")]).mean()), 1) for sd in ("left", "right")],
                            prior_hip_pitch_deg=[round(float(np.degrees(Qp[:, j(f"{sd}_hip_pitch_joint")]).mean()), 1) for sd in ("left", "right")],
                            knee_deg=[round(float(np.degrees(Q[:, j(f"{sd}_knee_joint")]).mean()), 1) for sd in ("left", "right")],
                            prior_knee_deg=[round(float(np.degrees(Qp[:, j(f"{sd}_knee_joint")]).mean()), 1) for sd in ("left", "right")],
                            max_dev_from_prior=dict(root_xy_cm=round(100 * float(np.hypot(dq[:, 0], dq[:, 1]).max()), 2), root_z_cm=round(100 * float(dq[:, 2].max()), 2),
                                                    joints_deg=round(float(np.degrees(dq[:, 7:22]).max()), 2)),
                            elbow_deg=[round(float(np.degrees(Q[:, j(f"{sd}_elbow_joint")]).mean()), 1) for sd in ("left", "right")],
                            elbow_min_deg=[round(float(np.degrees(Q[:, j(f"{sd}_elbow_joint")]).min()), 1) for sd in ("left", "right")],
                            shoulder_pitch_deg=[round(float(np.degrees(Q[:, j(f"{sd}_shoulder_pitch_joint")]).mean()), 1) for sd in ("left", "right")],
                            robot_palm_above_pelvis_m=round(float((palm_w[..., 2].mean(1) - Q[:, 2]).mean()), 4),
                            human_palm_above_pelvis_m=round(float((hum[f][:, 22:24, 2].mean(1) - hum[f][:, 0, 2]).mean()), 4),
                            robot_palm_above_table_top_m=round(float((palm_w[..., 2].mean(1) - top[f]).mean()), 4))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", action="append", required=True, help="name=pair npz@solve dir@k")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    robot = cg.cs.Robot("physx")
    offs = cg.palm_offsets()
    human = ce.human_core4d("table")
    out = {}
    for spec in a.ref:
        name, rest = spec.split("=", 1)
        p, d, k = rest.split("@")
        out[name] = dict(path=p, solve_dir=d, k=float(k), **check(p, d, float(k), robot, offs, human))
        print(name, json.dumps(out[name]), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    old = json.loads(a.out.read_text())["refs"] if a.out.exists() else {}
    a.out.write_text(json.dumps(dict(rules=__doc__, refs=old | out), indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
