#!/usr/bin/env python3
"""Scheme B1 offline precomputation (T02 v2 rule 4, method of T02 v1 rule 1). Run once; results + sha256 saved.

For every reference frame, robot and hand (left/right_rubber_hand_link origin = the paper's hand-error point):
  * p_o: hand position in the object frame (reference object pose of that frame);
  * closest point on the object's collision geometry (URDF boxes / meshes, trimesh), distance;
  * n_o: unit vector in the object frame from the hand to that closest point (pointing into the object);
    if the reference hand is inside the geometry the direction is flipped so it still points inward;
  * contact = distance < 0.03 m (or inside).
References come from the paper environment's loaders (system1/sonic/task_refs.py; Push with the registered spacing).

    hssim_s1/bin/python -B system1/scheme_b_contacts.py --out logs_system1/T02/contacts
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TASKS = ("push", "pull", "chair", "bucket", "table")   # Kick not applicable (T02 v2)
HANDS = ("left_rubber_hand_link", "right_rubber_hand_link")
CONTACT_M = 0.03
OBJECT_URDF = dict(
    push="holosoma/data/motions/g1_29dof/whole_body_tracking/objects_widetable_plan5_training.urdf",
    pull="holosoma/data/motions/g1_29dof/whole_body_tracking/objects_widetable_plan5_pull_training.urdf",
    chair="holosoma/data/motions/g1_29dof/whole_body_tracking/core4d_chair021_20231020_074_a1/chair021_training.urdf",
    bucket="holosoma/data/motions/g1_29dof/whole_body_tracking/core4d_bucket003_20231020_071_a1/bucket003_training.urdf",
    table="holosoma/data/motions/g1_29dof/whole_body_tracking/core4d_smalltable5kg_A/desk001_5kg_training.urdf",
)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rpy_matrix(rpy):
    from scipy.spatial.transform import Rotation
    return Rotation.from_euler("xyz", rpy).as_matrix()


def collision_mesh(urdf_path):
    """Union of all <collision> geometries of the (single-link) object URDF, in the link frame."""
    import trimesh

    root = ET.parse(urdf_path).getroot()
    links = root.findall("link")
    if len(links) != 1:
        raise ValueError("expected a single-link object URDF")
    parts = []
    for c in links[0].findall("collision"):
        o = c.find("origin")
        xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
        rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
        T = np.eye(4); T[:3, :3] = rpy_matrix(rpy); T[:3, 3] = xyz
        g = c.find("geometry")
        if g.find("box") is not None:
            m = trimesh.creation.box(extents=[float(v) for v in g.find("box").get("size").split()])
        elif g.find("mesh") is not None:
            me = g.find("mesh")
            m = trimesh.load(str(Path(urdf_path).parent / me.get("filename")), force="mesh")
            sc = [float(v) for v in me.get("scale", "1 1 1").split()]
            m.apply_scale(sc)
        else:
            raise ValueError("unsupported collision geometry")
        m.apply_transform(T)
        parts.append(m)
    return parts


def closest(parts, pts):
    """Closest surface point over all parts; inside flag if the point is inside any watertight part."""
    best_d = np.full(len(pts), np.inf); best_c = np.zeros_like(pts); inside = np.zeros(len(pts), bool)
    for m in parts:
        c, d, _ = m.nearest.on_surface(pts)
        better = d < best_d
        best_d[better], best_c[better] = d[better], c[better]
        if m.is_watertight:
            inside |= m.contains(pts)
    return best_c, best_d, inside


def quat_xyzw_to_matrix(q):
    from scipy.spatial.transform import Rotation
    return Rotation.from_quat(q).as_matrix()


def main():
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
    manifest = {}
    for task in TASKS:
        rel = str(reference_file(task))
        path = resolve_data_file_path(rel)
        names = ["pelvis", *HANDS]
        if task == "push":
            spacing = float(json.loads(PUSH_PROOF.read_text())["lateral_spacing_m"])
            ref = PairedA1Reference.from_motion_loader(MotionLoader(rel, names, dofs, device="cpu"), lateral_spacing_m=spacing)
        else:
            ref = PairedMotionReference(rel, names, dofs, device="cpu")
        s = {k: v.numpy().astype(np.float64) for k, v in ref.sample(torch.arange(ref.num_frames)).items()}
        N = ref.num_frames
        hands_w = s["agent_body_pos_w"][:, :, 1:3]                     # [N, 2 robots, 2 hands, 3]
        R = quat_xyzw_to_matrix(s["object_quat_w"])                     # loader quats are xyzw
        p_o = np.einsum("nji,nrhj->nrhi", R, hands_w - s["object_pos_w"][:, None, None])  # R^T (p - t)
        urdf = ROOT / "src/holosoma" / OBJECT_URDF[task]
        parts = collision_mesh(urdf)
        flat = p_o.reshape(-1, 3)
        c, d, inside = closest(parts, flat)
        v = c - flat
        nrm = np.linalg.norm(v, axis=-1, keepdims=True)
        n_o = np.where(nrm > 1e-9, v / np.maximum(nrm, 1e-12), 0.0)
        n_o[inside] *= -1.0
        contact = (d < CONTACT_M) | inside
        out = a.out / f"{task}_contacts.npz"
        np.savez_compressed(out, p_o=p_o, closest_o=c.reshape(p_o.shape), dist=d.reshape(p_o.shape[:-1]),
                            inside=inside.reshape(p_o.shape[:-1]), n_o=n_o.reshape(p_o.shape),
                            contact=contact.reshape(p_o.shape[:-1]), hands=np.array(HANDS))
        manifest[task] = dict(file=str(out.relative_to(ROOT)), sha256=sha(out), frames=N, reference_file=rel,
                              reference_sha256=sha(path), object_urdf=OBJECT_URDF[task], object_urdf_sha256=sha(urdf),
                              contact_fraction=float(contact.mean()),
                              contact_fraction_per_robot_hand=contact.reshape(N, 2, 2).mean(0).tolist(),
                              inside_fraction=float(inside.mean()), all_parts_watertight=all(m.is_watertight for m in parts))
        print(task, manifest[task]["contact_fraction"], flush=True)
    (a.out / "manifest.json").write_text(json.dumps(dict(contact_threshold_m=CONTACT_M, hands=HANDS, tasks=manifest), indent=1) + "\n")


if __name__ == "__main__":
    main()
