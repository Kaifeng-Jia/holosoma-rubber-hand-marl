#!/usr/bin/env python3
"""Scheme B1 offline contact precomputation, definition v2 (Kevin 2026-11-10, replaces v1 before any B evaluation).

For every reference frame, robot and hand:
  * the hand's collision geometry (rubber_hand_link collision mesh of our robot URDF, placed at the reference hand
    pose) and the object's collision geometry (object URDF boxes / meshes, placed at the reference object pose);
  * exact minimum distance between the two geometries and the pair of nearest points (python-fcl through
    trimesh.collision), expressed in the object frame;
  * contact = distance < 0.02 m;
  * n_o = unit vector from the nearest point on the hand to the nearest point on the object (object frame).
    If the geometries touch or overlap (distance <= 1e-6, nearest points undefined), n_o is the direction from the
    hand mesh centroid to its nearest object surface point, flipped if the centroid is inside the object; such
    frames are counted as `fallback_normal`.
Same definition for all five tasks. References come from the paper loaders (system1/sonic/task_refs.py).

    hssim_s1/bin/python -B system1/scheme_b_contacts_v2.py --out logs_system1/T02/contacts_v2
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
from system1.scheme_b_contacts import HANDS, OBJECT_URDF, TASKS, collision_mesh, sha  # noqa: E402

CONTACT_M = 0.02
ROBOT_URDF = ROOT / "src/holosoma/holosoma/data/robots/g1/main_mesh_collision_rubberhand.urdf"


def hand_meshes():
    import trimesh

    root = ET.parse(ROBOT_URDF).getroot()
    out = {}
    for h in HANDS:
        link = [l for l in root.findall("link") if l.get("name") == h][0]
        cols = link.findall("collision")
        if len(cols) != 1:
            raise ValueError(f"{h}: expected one collision geometry")
        o = cols[0].find("origin")
        if o is not None and (o.get("xyz", "0 0 0").split() != ["0", "0", "0"] or o.get("rpy", "0 0 0").split() != ["0", "0", "0"]):
            raise ValueError("non-identity collision origin not handled")
        fn = cols[0].find("geometry/mesh").get("filename")
        out[h] = (trimesh.load(str(ROBOT_URDF.parent / fn), force="mesh"), str(ROBOT_URDF.parent / fn))
    return out


def pose(p, q_xyzw):
    from scipy.spatial.transform import Rotation
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(q_xyzw).as_matrix()
    T[:3, 3] = p
    return T


def main():
    import trimesh

    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out = a.out.resolve()
    a.out.mkdir(parents=True, exist_ok=True)
    from holosoma.envs.marl import PairedA1Reference, PairedMotionReference
    from holosoma.managers.command.terms.wbt import MotionLoader
    from holosoma.utils.path import resolve_data_file_path
    from system1.sonic.task_refs import PUSH_PROOF, reference_file, sim_names

    _, dofs = sim_names()
    hands = hand_meshes()
    manifest = {}
    for task in TASKS:
        rel = str(reference_file(task))
        names = ["pelvis", *HANDS]
        if task == "push":
            spacing = float(json.loads(PUSH_PROOF.read_text())["lateral_spacing_m"])
            ref = PairedA1Reference.from_motion_loader(MotionLoader(rel, names, dofs, device="cpu"), lateral_spacing_m=spacing)
        else:
            ref = PairedMotionReference(rel, names, dofs, device="cpu")
        s = {k: v.numpy().astype(np.float64) for k, v in ref.sample(torch.arange(ref.num_frames)).items()}
        N = ref.num_frames
        urdf = ROOT / "src/holosoma" / OBJECT_URDF[task]
        parts = collision_mesh(urdf)
        cm = trimesh.collision.CollisionManager()
        for i, m in enumerate(parts):
            cm.add_object(f"part{i}", m)
        dist = np.zeros((N, 2, 2)); p_obj = np.zeros((N, 2, 2, 3)); p_hand = np.zeros((N, 2, 2, 3))
        n_o = np.zeros((N, 2, 2, 3)); fallback = np.zeros((N, 2, 2), bool)
        for k in range(N):
            To_inv = np.linalg.inv(pose(s["object_pos_w"][k], s["object_quat_w"][k]))
            for r in range(2):
                for hi, h in enumerate(HANDS):
                    Th = To_inv @ pose(s["agent_body_pos_w"][k, r, 1 + hi], s["agent_body_quat_w"][k, r, 1 + hi])
                    d, names_, data = cm.min_distance_single(hands[h][0], transform=Th, return_name=True, return_data=True)
                    po, ph = np.asarray(data.point(names_)), np.asarray(data.point("__external"))
                    dist[k, r, hi], p_obj[k, r, hi], p_hand[k, r, hi] = d, po, ph
                    v = po - ph
                    if d > 1e-6 and np.linalg.norm(v) > 1e-9:
                        n_o[k, r, hi] = v / np.linalg.norm(v)
                    else:
                        fallback[k, r, hi] = True
                        c = (Th @ np.r_[hands[h][0].centroid, 1.0])[:3]
                        best = None
                        for m in parts:
                            cp, cd, _ = m.nearest.on_surface(c[None])
                            if best is None or cd[0] < best[1]:
                                best = (cp[0], cd[0], m)
                        w = best[0] - c
                        inside = any(m.is_watertight and m.contains(c[None])[0] for m in parts)
                        w = -w if inside else w
                        n_o[k, r, hi] = w / max(np.linalg.norm(w), 1e-12)
        contact = dist < CONTACT_M
        out = a.out / f"{task}_contacts.npz"
        np.savez_compressed(out, dist=dist, p_obj_o=p_obj, p_hand_o=p_hand, n_o=n_o, contact=contact, fallback_normal=fallback,
                            hands=np.array(HANDS))
        manifest[task] = dict(file=str(out.relative_to(ROOT)), sha256=sha(out), frames=N, reference_file=rel,
                              reference_sha256=sha(resolve_data_file_path(rel)), object_urdf=OBJECT_URDF[task],
                              object_urdf_sha256=sha(urdf), contact_fraction=float(contact.mean()),
                              contact_fraction_per_robot=contact.mean(axis=(0, 2)).tolist(),
                              contact_fraction_per_robot_hand=contact.mean(0).tolist(),
                              contact_frames_any_hand_per_robot=contact.any(axis=2).sum(0).tolist(),
                              min_dist_m_per_robot_hand=dist.min(0).tolist(), fallback_normal_count=int(fallback.sum()))
        print(task, manifest[task]["contact_fraction_per_robot"], flush=True)
    (a.out / "manifest.json").write_text(json.dumps(dict(
        definition="v2: min distance between rubber_hand_link collision mesh and object collision geometry < 0.02 m; "
                   "normal = nearest-point pair direction (hand -> object), object frame",
        contact_threshold_m=CONTACT_M, hands=HANDS, hand_meshes={h: dict(file=v[1], sha256=sha(v[1])) for h, v in hands.items()},
        robot_urdf_sha256=sha(ROBOT_URDF), tasks=manifest), indent=1) + "\n")


if __name__ == "__main__":
    main()
