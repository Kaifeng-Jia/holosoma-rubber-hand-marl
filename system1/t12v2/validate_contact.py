#!/usr/bin/env python3
"""T12-v2: checks of the fast contact geometry in contact_stats.py against the slow exact path T12 used.

1. Vectorized forward kinematics vs yourdfpy (random joint angles and root poses): max link-origin / rotation difference.
2. Object signed distance vs trimesh.proximity.signed_distance on the object's collision mesh, at points near the surface
   (surface samples moved along the normal by -4 .. +4 cm) and uniform in the grid box: max |difference| and the share of
   points classified differently at the -1.5 cm contact threshold.
3. T12 reproduction: D Table seed 721, +/-2 perturbed_000..002, Isaac and MuJoCo, per-hand contact fraction from row 190 (the
   T12 window) with the fast path vs results_system1/T12_videos/slip_onset.csv (T12, exact trimesh per vertex).

    OMP_NUM_THREADS=1 hssim_s1/bin/python -B system1/t12v2/validate_contact.py --out results_system1/T12_videos_v2/contact_validation.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
import contact_stats as cs  # noqa: E402

ROOT = cs.ROOT


def check_fk(robot, rng, n=40):
    worst_p = worst_r = 0.0
    lim = [(j.limit.lower, j.limit.upper) for j in (robot.u.joint_map[d] for d in robot.dofs)]
    for _ in range(n):
        q = np.array([rng.uniform(lo, hi) for lo, hi in lim])
        pos = rng.normal(0, 1, 3)
        quat = rng.normal(0, 1, 4)
        quat /= np.linalg.norm(quat)
        T = robot.fk(pos[None], quat[None], q[None])
        robot.u.update_cfg(dict(zip(robot.dofs, q)))
        Tb = cs.pose_mat(pos, quat)
        for link in robot.links:
            ref = Tb @ robot.u.get_transform(link)
            worst_p = max(worst_p, float(np.abs(T[link][0, :3, 3] - ref[:3, 3]).max()))
            worst_r = max(worst_r, float(np.abs(T[link][0, :3, :3] - ref[:3, :3]).max()))
    return dict(samples=n, max_position_diff_m=worst_p, max_rotation_entry_diff=worst_r)


def exact_sdf(obj, pts):
    if obj.hulls is not None:  # PhysX geometry: union of convex pieces
        return np.max([trimesh.proximity.signed_distance(h, pts) for h in obj.hulls], axis=0)
    return trimesh.proximity.signed_distance(obj.mesh, pts)


def check_sdf(obj, rng, n_surface=20000, n_uniform=20000):
    m = obj.mesh
    p, fi = trimesh.sample.sample_surface(m, n_surface, seed=1)
    p = p + m.face_normals[fi] * rng.uniform(-0.04, 0.04, (n_surface, 1))
    u = rng.uniform(obj.lo - cs.GRID_MARGIN, obj.hi + cs.GRID_MARGIN, (n_uniform, 3))
    out = {}
    for name, pts in (("near_surface", p), ("uniform_in_box", u)):
        exact = exact_sdf(obj, pts)
        fast = obj.sdf(pts)
        near = np.abs(exact - cs.CONTACT_M) < 0.05
        out[name] = dict(points=len(pts), max_abs_diff_m=float(np.abs(exact - fast)[near].max()), p99_abs_diff_m=float(np.quantile(np.abs(exact - fast)[near], 0.99)),
                         contact_class_mismatch_share=float(np.mean((exact >= cs.CONTACT_M) != (fast >= cs.CONTACT_M))),
                         mismatch_within_2mm_of_threshold=int(np.sum(((exact >= cs.CONTACT_M) != (fast >= cs.CONTACT_M)) & (np.abs(exact - cs.CONTACT_M) < 0.002))))
    return dict(kind=obj.kind, **out)


def check_t12(robot, obj):
    t12 = {(r["simulator"], r["state_id"]): r for r in csv.DictReader((ROOT / "results_system1/T12_videos/slip_onset.csv").open())}
    out = []
    for (sim, st), r in sorted(t12.items()):
        tr = Path(r["trajectory"])
        with np.load(tr) as z:
            a = {k: z[k] for k in z.files}
        n = int(r["executed_steps"]) + (0 if r["completed"] == "True" else 1)
        n = min(n, len(a["dof_pos"]))
        D = cs.link_distances(robot, obj, a["root_pos"][:n], a["root_quat_xyzw"][:n], a["dof_pos"][:n], a["object_pos_w"][:n], a["object_quat_xyzw"][:n])
        C = D[190:n, :, :2] >= cs.CONTACT_M
        row = dict(simulator=sim, state_id=st)
        for ri in range(2):
            for hi, side in enumerate(("left", "right")):
                k = f"contact_fraction_r{ri}_{side}"
                row[k] = dict(t12=float(r[k]), fast=round(float(C[:, ri, hi].mean()), 3))
        out.append(row)
    worst = max(abs(v["t12"] - v["fast"]) for row in out for k, v in row.items() if isinstance(v, dict))
    return dict(episodes=out, max_abs_diff=worst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--geometry", choices=("urdf", "physx"), default="urdf", help="physx: only the object distance checks and the hull-vs-mesh comparison")
    a = ap.parse_args()
    rng = np.random.default_rng(20261123)
    robot = cs.Robot(a.geometry)
    rec = dict(geometry=a.geometry)
    if a.geometry == "urdf":
        rec["fk"] = check_fk(robot, rng)
        print(json.dumps(rec["fk"]), flush=True)
    objs = {}
    for t in cs.TASKS:
        s0 = json.loads((next(r for r in cs.runs() if r["task"] == t)["eval_dir"] / "summary.json").read_text())
        objs[t] = cs.Object(cs.DATA / s0["object_urdf"], workers=16, geometry=a.geometry, task=t)
        rec[f"sdf_{t}"] = check_sdf(objs[t], rng)
        print(t, json.dumps(rec[f"sdf_{t}"]), flush=True)
        if a.geometry == "physx" and objs[t].hulls is not None:  # how far the PhysX pieces stick out of the URDF mesh
            mesh = cs.Object(cs.DATA / s0["object_urdf"])
            q, _ = trimesh.sample.sample_surface(objs[t].mesh, 30000, seed=0)
            d = mesh.sdf(q)
            rec[f"pieces_vs_urdf_mesh_{t}"] = dict(pieces=len(objs[t].hulls), share_of_piece_surface_outside_mesh_gt_1p5cm=float(np.mean(d < -0.015)),
                                                  share_gt_3cm=float(np.mean(d < -0.03)), max_outside_m=float(-d.min()))
    if a.geometry == "physx":  # robot: PhysX hull of each mesh link vs the URDF collision mesh
        ur = cs.Robot("urdf")
        out = {}
        for link in ("left_rubber_hand_link", "torso_link", "left_hip_yaw_link", "left_elbow_link"):
            m = ur.u.collision_scene
            node = [n for n in m.graph.nodes_geometry if m.graph.transforms.parents[n] == link][0]
            T, g = m.graph.get(frame_to=node, frame_from=link)
            um = m.geometry[g].copy().apply_transform(T)
            d = trimesh.proximity.signed_distance(um, robot.points[link][:5000])
            out[link] = dict(points=int(min(5000, len(robot.points[link]))), share_outside_urdf_mesh_gt_1cm=float(np.mean(d < -0.01)), max_outside_m=float(-d.min()))
        rec["robot_hulls_vs_urdf"] = out
    else:
        rec["t12_reproduction"] = check_t12(robot, objs["table"])
        print("t12 max abs diff", rec["t12_reproduction"]["max_abs_diff"], flush=True)
    rec["sample_points"] = {l: int(len(p)) for l, p in robot.points.items()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rec, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
