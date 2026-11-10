#!/usr/bin/env python3
"""T02 report item (Kevin 2026-11-10): how deep the reference hand meshes sit inside the object.

For every reference frame, robot and hand: the rubber_hand_link collision mesh vertices (every 2nd vertex, ~2 mm
spacing) are placed at the reference hand pose and expressed in the object frame. A vertex inside an object
collision part has depth = its distance to that part's surface (boxes: analytic; meshes: trimesh contains +
nearest surface point); vertex depth = max over the parts containing it. Hand-frame depth = max over vertices.
Reported per task and robot: number of hand-frames with depth > 0, max depth, mean depth over those hand-frames.

    hssim_s1/bin/python -B system1/scheme_b_embed_depth.py --out results_system1/T02/hand_embedding_depth.json
"""
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from system1.scheme_b_contacts import HANDS, OBJECT_URDF, TASKS, rpy_matrix, sha  # noqa: E402
from system1.scheme_b_contacts_v2 import ROBOT_URDF, hand_meshes, pose  # noqa: E402


def parts(urdf_path):
    import trimesh

    out = []
    link = ET.parse(urdf_path).getroot().findall("link")[0]
    for c in link.findall("collision"):
        o = c.find("origin")
        xyz = np.array([float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()])
        rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
        R = rpy_matrix(rpy)
        g = c.find("geometry")
        if g.find("box") is not None:
            out.append(("box", R, xyz, np.array([float(v) for v in g.find("box").get("size").split()]) / 2))
        else:
            me = g.find("mesh")
            m = trimesh.load(str(Path(urdf_path).parent / me.get("filename")), force="mesh")
            m.apply_scale([float(v) for v in me.get("scale", "1 1 1").split()])
            T = np.eye(4); T[:3, :3] = R; T[:3, 3] = xyz
            m.apply_transform(T)
            if not m.is_watertight:
                raise ValueError("mesh part not watertight")
            out.append(("mesh", m))
    return out


def depth(pts, prts):
    d = np.zeros(len(pts))
    for p in prts:
        if p[0] == "box":
            _, R, t, h = p
            loc = (pts - t) @ R
            inside = np.all(np.abs(loc) < h, axis=1)
            if inside.any():
                d[inside] = np.maximum(d[inside], np.min(h - np.abs(loc[inside]), axis=1))
        else:
            m = p[1]
            inside = m.contains(pts)
            if inside.any():
                _, dist, _ = m.nearest.on_surface(pts[inside])
                d[inside] = np.maximum(d[inside], dist)
    return d


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--out", type=Path, required=True)
    a = a.parse_args()
    from holosoma.envs.marl import PairedA1Reference, PairedMotionReference
    from holosoma.managers.command.terms.wbt import MotionLoader
    from system1.sonic.task_refs import PUSH_PROOF, reference_file, sim_names

    _, dofs = sim_names()
    hm = {h: np.asarray(v[0].vertices)[::2] for h, v in hand_meshes().items()}
    res = {}
    for task in TASKS:
        rel = str(reference_file(task))
        names = ["pelvis", *HANDS]
        if task == "push":
            sp = float(json.loads(PUSH_PROOF.read_text())["lateral_spacing_m"])
            ref = PairedA1Reference.from_motion_loader(MotionLoader(rel, names, dofs, device="cpu"), lateral_spacing_m=sp)
        else:
            ref = PairedMotionReference(rel, names, dofs, device="cpu")
        s = {k: v.numpy().astype(np.float64) for k, v in ref.sample(torch.arange(ref.num_frames)).items()}
        N = ref.num_frames
        urdf = ROOT / "src/holosoma" / OBJECT_URDF[task]
        prts = parts(urdf)
        dep = np.zeros((N, 2, 2))
        for k in range(N):
            Ti = np.linalg.inv(pose(s["object_pos_w"][k], s["object_quat_w"][k]))
            for r in range(2):
                for hi, h in enumerate(HANDS):
                    T = Ti @ pose(s["agent_body_pos_w"][k, r, 1 + hi], s["agent_body_quat_w"][k, r, 1 + hi])
                    pts = hm[h] @ T[:3, :3].T + T[:3, 3]
                    dep[k, r, hi] = depth(pts, prts).max()
        per_robot = []
        for r in range(2):
            x = dep[:, r].reshape(-1)
            e = x[x > 0]
            per_robot.append(dict(hand_frames=int(x.size), embedded_hand_frames=int(e.size), max_depth_cm=float(x.max() * 100),
                                  mean_depth_cm_over_embedded=float(e.mean() * 100) if e.size else None))
        allx = dep.reshape(-1); e = allx[allx > 0]
        res[task] = dict(frames=N, per_robot=per_robot, embedded_hand_frames=int(e.size), hand_frames=int(allx.size),
                         max_depth_cm=float(allx.max() * 100), mean_depth_cm_over_embedded=float(e.mean() * 100) if e.size else None,
                         object_urdf=OBJECT_URDF[task], object_urdf_sha256=sha(urdf))
        np.save(ROOT / f"logs_system1/T02/contacts_v2/{task}_embed_depth_m.npy", dep)
        print(task, res[task]["max_depth_cm"], res[task]["mean_depth_cm_over_embedded"], res[task]["embedded_hand_frames"], flush=True)
    a.out.write_text(json.dumps(dict(
        definition="rubber_hand_link collision mesh vertices (every 2nd) at the reference hand pose; depth of a vertex inside "
                   "an object collision part = distance to that part's surface; hand-frame depth = max over vertices",
        robot_urdf_sha256=sha(ROBOT_URDF), tasks=res), indent=1) + "\n")


if __name__ == "__main__":
    main()
