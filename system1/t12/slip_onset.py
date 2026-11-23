#!/usr/bin/env python3
"""T12-v: when and in which reference phase does the table start to slide, for the original D Table seed-721 policy (paper D,
it12000) on the +/-2 states perturbed_000..002 in MuJoCo (T10 setting, all failed), with the same policy and states in Isaac
(d_final_5070 protocol) as the control. Read only, CPU, saved per-step states only (no simulation).

Rules (fixed in this docstring before the numbers were written into the return pack):
1. Hand-table distance: signed distance of the rubber-hand collision mesh (every k-th vertex, at most ~1500 points) to the
   table collision mesh (the five collision boxes of desk001_5kg_training.urdf), from the saved root pose and joint angles (URDF
   forward kinematics). In contact: distance >= -1.5 cm (within 1.5 cm or penetrating).
2. Hand sliding on the table: hand link origin in the table frame; speed = central difference (rows k-1, k+1) x 25 per s.
   A hand slides at row k if it is in contact at k and that speed > 0.1 m/s. "Hand slip onset" = first row >= 190 (start of
   the reference lift) that starts a run of >= 5 consecutive rows in which some hand slides.
3. Table sliding on the floor ahead of the reference: object planar speed minus reference object planar speed (central
   differences) > 0.2 m/s; "floor slide onset" = first row >= 190 that starts a run of >= 5 consecutive such rows.
4. Object position error vs the reference first > 0.1 m (evaluator alignment, row k vs reference frame k, as object_rmse).
5. MuJoCo vs Isaac (same policy, same state): first row where the object positions differ by > 5 cm and > 10 cm.
6. Table height: max object height minus the row-0 height (the reference lifts it by ~6 cm).
Phase: the reference phase of that row (results_system1/T10b_friction_margin/retrain/pilot_progress/references.json, the
T10b phase rule); time = row / 50 s.

    hssim_s1/bin/python -B system1/t12/slip_onset.py --out results_system1/T12_videos
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh
import yourdfpy
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[2]
ROBOT = ROOT / "src/holosoma/holosoma/data/robots/g1/main_mesh_collision_rubberhand.urdf"
OBJ = ROOT / "src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/core4d_smalltable5kg_A/desk001_5kg_training.urdf"
REFS = ROOT / "results_system1/T10b_friction_margin/retrain/pilot_progress/references.json"
SRC = {"isaac": ROOT / "logs_system1/post_train/d_curve/D_table_paper_T-07_seed721/it12000_pm2",
       "mujoco": ROOT / "logs_system1/t10/mj_eval/d_table_seed721_pm2"}
CKPT = "f163d11a9629edae5dd72732c4ddec08a43024533a47b349a099a3275a8d1c50"
STATES = ("perturbed_000", "perturbed_001", "perturbed_002")
HANDS = ("left_rubber_hand_link", "right_rubber_hand_link")
CONTACT_M, SLIDE_MS, FLOOR_MS, RUN, START = -0.015, 0.1, 0.2, 5, 190
ZH = dict(rotate="转动", lift="抬起", lower="放下", carry="举起中", push="贴地推行", still="静止")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def first_run(mask, start=START, run=RUN):
    for k in range(start, len(mask) - run + 1):
        if mask[k:k + run].all():
            return k
    return None


def first(mask):
    idx = np.flatnonzero(mask)
    return int(idx[0]) if idx.size else None


class Geometry:
    def __init__(self):
        self.u = yourdfpy.URDF.load(str(ROBOT), load_meshes=True, build_scene_graph=True, load_collision_meshes=True, build_collision_scene_graph=True)
        o = yourdfpy.URDF.load(str(OBJ), load_meshes=True, build_scene_graph=True, load_collision_meshes=True, build_collision_scene_graph=True)
        self.table = o.collision_scene.to_geometry()
        self.pq = trimesh.proximity.ProximityQuery(self.table)
        cs = self.u.collision_scene
        self.hv = {}
        for h in HANDS:
            T, g = cs.graph.get(frame_to=h + "_1", frame_from=h)
            v = trimesh.transform_points(cs.geometry[g].vertices, T)
            self.hv[h] = v[:: max(1, len(v) // 1500)]

    def hands(self, z, k):
        """hand link origins in the table frame (robot 0 L, R, robot 1 L, R) and their signed distances to the table (m)."""
        To = np.eye(4)
        To[:3, :3] = R.from_quat(z["object_quat_xyzw"][k]).as_matrix()
        To[:3, 3] = z["object_pos_w"][k]
        Toi = np.linalg.inv(To)
        P, D = [], []
        for r in range(2):
            self.u.update_cfg(z["dof_pos"][k, r])
            Tb = np.eye(4)
            Tb[:3, :3] = R.from_quat(z["root_quat_xyzw"][k, r]).as_matrix()
            Tb[:3, 3] = z["root_pos"][k, r]
            for h in HANDS:
                Th = Toi @ Tb @ self.u.get_transform(h)
                P.append(Th[:3, 3])
                D.append(float(self.pq.signed_distance(trimesh.transform_points(self.hv[h], Th)).max()))
        return np.array(P), np.array(D)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    ref = json.loads(REFS.read_text())["d_table_seed721_rand"]
    phase = [None] * ref["frames"]
    for name, _, s, e in ref["phase_segments"]:
        for k in range(s, e + 1):
            phase[k] = name
    geo = Geometry()
    rows, steps_out, traj = [], [], {}
    for sim, d in SRC.items():
        s = json.loads((d / "summary.json").read_text())
        assert s["checkpoint_sha256"] == CKPT
        for st in STATES:
            e = next(x for x in s["episodes"] if x["state_id"] == st)
            z = {k: v for k, v in np.load(e["trajectory"]).items()}
            n = e["steps"] + 1 if not e["completed_reference"] else e["steps"]  # executed rows (completed: the extra terminal row excluded)
            n = min(n, len(z["dof_pos"]))
            traj[(sim, st)] = (z, n, e)
    for (sim, st), (z, n, e) in traj.items():
        P, D = zip(*(geo.hands(z, k) for k in range(n)))
        P, D = np.array(P), np.array(D)
        contact = D >= CONTACT_M
        V = np.zeros(D.shape)
        V[1:-1] = np.linalg.norm(P[2:] - P[:-2], axis=-1) * 25
        sliding = (contact & (V > SLIDE_MS)).any(axis=1)
        ob, rf = z["object_pos_w"][:n], z["reference_object_pos_w"][:n]
        vo = np.zeros(n)
        vr = np.zeros(n)
        vo[1:-1] = np.linalg.norm(ob[2:, :2] - ob[:-2, :2], axis=-1) * 25
        vr[1:-1] = np.linalg.norm(rf[2:, :2] - rf[:-2, :2], axis=-1) * 25
        ahead = (vo - vr) > FLOOR_MS
        err = np.linalg.norm(ob - rf, axis=-1)
        zo, no, _ = traj[("isaac" if sim == "mujoco" else "mujoco", st)]
        m = min(n, no)
        sep = np.linalg.norm(ob[:m] - zo["object_pos_w"][:m], axis=-1)
        k_slip, k_floor, k_err = first_run(sliding), first_run(ahead), first(err > 0.1)
        k_sep5, k_sep10 = first(sep > 0.05), first(sep > 0.10)
        t = lambda k: None if k is None else round(k / 50, 2)  # noqa: E731
        ph = lambda k: None if k is None else ZH[phase[min(k, len(phase) - 1)]]  # noqa: E731
        rows.append(dict(state_id=st, simulator=sim, policy="original (paper D it12000)", completed=e["completed_reference"], executed_steps=e["steps"],
                         contact_fraction_r0_left=round(float(contact[START:, 0].mean()), 3), contact_fraction_r0_right=round(float(contact[START:, 1].mean()), 3),
                         contact_fraction_r1_left=round(float(contact[START:, 2].mean()), 3), contact_fraction_r1_right=round(float(contact[START:, 3].mean()), 3),
                         hand_slip_onset_row=k_slip, hand_slip_onset_s=t(k_slip), hand_slip_onset_phase=ph(k_slip),
                         r1_hand_disp_in_table_frame_190_to_end_cm=[round(100 * float(np.linalg.norm(P[-1, j] - P[START, j])), 1) for j in (2, 3)],
                         floor_slide_onset_row=k_floor, floor_slide_onset_s=t(k_floor), floor_slide_onset_phase=ph(k_floor),
                         object_error_gt_0p1m_row=k_err, object_error_gt_0p1m_s=t(k_err), object_error_gt_0p1m_phase=ph(k_err),
                         mujoco_vs_isaac_object_apart_5cm_row=k_sep5, apart_5cm_s=t(k_sep5), apart_5cm_phase=ph(k_sep5),
                         mujoco_vs_isaac_object_apart_10cm_row=k_sep10, apart_10cm_s=t(k_sep10), apart_10cm_phase=ph(k_sep10),
                         table_max_rise_cm=round(100 * float(ob[:, 2].max() - ob[0, 2]), 1), reference_max_rise_cm=round(100 * float(rf[:, 2].max() - rf[0, 2]), 1),
                         object_planar_speed_max_m_s=round(float(vo.max()), 2), reference_planar_speed_max_m_s=round(float(vr.max()), 2),
                         trajectory=e["trajectory"], trajectory_sha256=sha(e["trajectory"])))
        for k in range(n):
            steps_out.append(dict(state_id=st, simulator=sim, row=k, t_s=k / 50, phase=phase[min(k, len(phase) - 1)],
                                  **{f"dist_cm_{h}": round(100 * float(D[k, j]), 2) for j, h in enumerate(("r0L", "r0R", "r1L", "r1R"))},
                                  **{f"slide_speed_m_s_{h}": round(float(V[k, j]), 3) for j, h in enumerate(("r0L", "r0R", "r1L", "r1R"))},
                                  object_planar_speed_m_s=round(float(vo[k]), 3), reference_planar_speed_m_s=round(float(vr[k]), 3),
                                  object_error_m=round(float(err[k]), 4), object_z_m=round(float(ob[k, 2]), 4)))
        print(json.dumps({k: v for k, v in rows[-1].items() if k not in ("trajectory", "trajectory_sha256")}, ensure_ascii=False), flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (a.out / "slip_onset.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    keys = list(steps_out[0])
    with gzip.open(a.out / "slip_onset_per_step.csv.gz", "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(steps_out)
    (a.out / "slip_onset_provenance.json").write_text(json.dumps(dict(script_sha256=sha(__file__), robot_urdf_sha256=sha(ROBOT), object_urdf_sha256=sha(OBJ),
                                                                       references_json_sha256=sha(REFS), rules=dict(contact_m=CONTACT_M, slide_m_s=SLIDE_MS,
                                                                       floor_m_s=FLOOR_MS, run_rows=RUN, start_row=START)), indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
