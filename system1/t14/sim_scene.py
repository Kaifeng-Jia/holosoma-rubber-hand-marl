#!/usr/bin/env python3
"""T14 round 3 (section 8 item 3): make the retargeting scene's robot-object geometry the simulator's.

Cause found in round 3: the retargeting G1 model (src/holosoma_retargeting/.../models/g1/g1_29dof.{xml,urdf}) and the
simulator's G1 (src/holosoma/holosoma/data/robots/g1/main_mesh_collision_rubberhand.urdf, the robot of the T10 PhysX
dumps and of every evaluation) place the waist pivots differently:
  waist_roll_joint  z 0.035 (retarget) vs 0.044 (sim);  waist_pitch_joint z 0.019 vs 0;
  torso children (shoulders, head, logo, mid360) z -0.010 to compensate, so the two models agree only at zero waist roll
  and pitch. With the waist bent, every link above the waist is displaced (Table k = 1, robot 1, 2.87 s: 5.2 mm at the
  hands), so the retargeter's 1 mm non-penetration constraint, which holds in its own model at every frame, does not hold
  for the simulator's robot. In addition the retargeting model's robot-object collision set is not the simulator's
  (all-link MJCF hulls plus 1 mm thumb / pinky marker spheres that stick out of the rubber-hand hull by up to 1.45 cm).

simify_scene() rewrites a retargeting scene so that the pairs the non-penetration constraint sees are exactly the pairs
the evaluation measures:
  1. kinematics: every MJCF body that is the child of a simulator URDF joint gets that joint's origin (pos, quat);
     joint axes and limits are already identical (checked by the smoke test);
  2. robot-object pairs: the PhysX link shapes of the T10 dump (the evaluation's robot shapes, incl. the head) are added as
     mesh geoms (contype 0, conaffinity 4); the original robot collision geoms keep contype / conaffinity 1 and collide
     only with the ground (ground pairs unchanged);
  3. object: its collision geoms get contype 4, conaffinity 0; with object_pieces (Bin, Chair: the 16 PhysX convex pieces
     x k) the original collision geom is switched off (kept for mass) and the pieces are added instead.
Nothing else in the scene changes. Switch off (scenes not passed through simify_scene) = the existing behaviour.
"""
from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
SIM_URDF = ROOT / "src/holosoma/holosoma/data/robots/g1/main_mesh_collision_rubberhand.urdf"
ROBOT_BIT, OBJECT_BIT = 1, 4


def _urdf_joints(urdf):
    out = {}
    for j in ET.parse(urdf).getroot().iter("joint"):
        o = j.find("origin")
        xyz = np.array([float(v) for v in (o.get("xyz") if o is not None and o.get("xyz") else "0 0 0").split()])
        rpy = np.array([float(v) for v in (o.get("rpy") if o is not None and o.get("rpy") else "0 0 0").split()])
        out[j.find("child").get("link")] = dict(name=j.get("name"), parent=j.find("parent").get("link"), type=j.get("type"), xyz=xyz, rpy=rpy)
    return out


def _T(xyz, rpy):
    M = np.eye(4)
    M[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    M[:3, 3] = xyz
    return M


def _fmt(v):
    return " ".join(f"{x:.10g}" for x in v)


def simify_scene(scene_xml, out_xml, mesh_dir, object_body, object_pieces=None, dump_task="table", sim_urdf=SIM_URDF):
    sys.path.insert(0, str(ROOT / "system1/t12v2"))
    import contact_stats as cs
    mesh_dir = Path(mesh_dir).resolve()
    mesh_dir.mkdir(parents=True, exist_ok=True)
    tree = ET.parse(scene_xml)
    root = tree.getroot()
    asset = root.find("asset")
    world = root.find("worldbody")
    parent_of, bodies = {}, {}
    for b in root.iter("body"):
        bodies[b.get("name")] = b
        for c in b.findall("body"):
            parent_of[c.get("name")] = b.get("name")
    for c in world.findall("body"):
        parent_of[c.get("name")] = None
    rec = dict(scene_in=str(scene_xml), sim_urdf=str(sim_urdf), kinematics_changed={}, kinematics_skipped=[])
    # 1. kinematics
    J = _urdf_joints(sim_urdf)
    for child, j in J.items():
        if child not in bodies:
            continue
        if parent_of.get(child) != j["parent"]:
            rec["kinematics_skipped"].append(dict(body=child, mjcf_parent=parent_of.get(child), urdf_parent=j["parent"]))
            continue
        b = bodies[child]
        old_pos = np.array([float(v) for v in (b.get("pos") or "0 0 0").split()])
        old_quat = np.array([float(v) for v in (b.get("quat") or "1 0 0 0").split()])
        new_quat = Rotation.from_euler("xyz", j["rpy"]).as_quat()[[3, 0, 1, 2]]
        if np.dot(new_quat, old_quat) < 0:
            new_quat = -new_quat
        if np.abs(old_pos - j["xyz"]).max() > 1e-6 or np.abs(old_quat / np.linalg.norm(old_quat) - new_quat).max() > 1e-5:
            rec["kinematics_changed"][child] = dict(old_pos=old_pos.tolist(), new_pos=j["xyz"].tolist(), old_quat=old_quat.tolist(), new_quat=new_quat.round(10).tolist())
        # every matched body is written from the URDF (also removes the MJCF's 6-digit rounding of the quaternions)
        b.set("pos", _fmt(j["xyz"]))
        b.set("quat", _fmt(new_quat))
        for k in ("euler", "axisangle", "xyaxes", "zaxis"):
            b.attrib.pop(k, None)
    # robot bodies = the subtree of the pelvis
    robot_root = bodies["pelvis"]
    robot_bodies = {b.get("name"): b for b in robot_root.iter("body")}
    # 2a. original robot collision geoms: robot / ground bit only
    n_orig = 0
    for b in robot_bodies.values():
        for g in b.findall("geom"):
            ct, ca = int(g.get("contype", "1")), int(g.get("conaffinity", "1"))
            if ct == 0 and ca == 0:
                continue
            g.set("contype", str(ROBOT_BIT))
            g.set("conaffinity", str(ROBOT_BIT))
            n_orig += 1
    # 2b. PhysX link shapes for robot-object pairs
    shapes, scene_sha, conv_sha = cs.dump_shapes(dump_task, "/World/envs/env_0/Robot/")
    added = {}
    for i, (link, mesh) in enumerate(shapes):
        host, M = link, np.eye(4)
        while host not in robot_bodies:  # a simulator link without an MJCF body (head_link): attach to its parent
            j = J[host]
            assert j["type"] == "fixed", (host, j)
            M = _T(j["xyz"], j["rpy"]) @ M
            host = j["parent"]
        p = mesh_dir / f"physx_{link}_{i}.obj"
        mesh.export(p)
        ET.SubElement(asset, "mesh", name=f"physx_{link}_{i}", file=str(p))
        attrs = dict(name=f"physx_{link}_{i}", type="mesh", mesh=f"physx_{link}_{i}", contype="0", conaffinity=str(OBJECT_BIT),
                     density="0", group="3", rgba="1 0.5 0 0.3")
        if host != link:
            attrs.update(pos=_fmt(M[:3, 3]), quat=_fmt(Rotation.from_matrix(M[:3, :3]).as_quat()[[3, 0, 1, 2]]))
        ET.SubElement(robot_bodies[host], "geom", **attrs)
        added.setdefault(link, dict(host=host, pieces=0))["pieces"] += 1
    rec.update(robot_original_collision_geoms_ground_only=n_orig, physx_robot_shapes=added, physx_dump=dict(task=dump_task, scene_sha256=scene_sha, convexes_sha256=conv_sha))
    # 3. object
    ob = bodies[object_body]
    obj_col = [g for g in ob.findall("geom") if not (int(g.get("contype", "1")) == 0 and int(g.get("conaffinity", "1")) == 0)]
    assert obj_col, f"no collision geom on {object_body}"
    if object_pieces is None:
        for g in obj_col:
            g.set("contype", str(OBJECT_BIT))
            g.set("conaffinity", "0")
        rec["object"] = dict(body=object_body, collision="scene geoms", geoms=[g.get("name") for g in obj_col])
    else:
        for g in obj_col:
            g.set("contype", "0")
            g.set("conaffinity", "0")
        for i, m in enumerate(object_pieces):
            p = mesh_dir / f"object_piece_{i}.obj"
            m.export(p)
            ET.SubElement(asset, "mesh", name=f"object_piece_{i}", file=str(p))
            ET.SubElement(ob, "geom", name=f"{object_body}_piece_{i}", type="mesh", mesh=f"object_piece_{i}", contype=str(OBJECT_BIT), conaffinity="0",
                          density="0", group="3", rgba="0.3 0.5 0.8 0.15")
        rec["object"] = dict(body=object_body, collision=f"{len(object_pieces)} PhysX pieces", switched_off=[g.get("name") for g in obj_col])
    tree.write(out_xml)
    rec["scene_out"] = str(out_xml)
    return Path(out_xml), rec


def object_pieces(task, k):
    """The T10 PhysX convex pieces of a task's training object, x k about the object frame origin (= eval_core4d_hulls)."""
    sys.path.insert(0, str(ROOT / "system1/t12v2"))
    import contact_stats as cs
    dump = dict(table="table", bin="bucket", chair="chair")[task]
    out = []
    for _, mesh in cs.dump_shapes(dump, "/World/envs/env_0/Object/")[0]:
        m = mesh.copy()
        m.vertices = m.vertices * k
        out.append(m)
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--object-body", required=True)
    ap.add_argument("--pieces", default=None, help="task:k for PhysX object pieces")
    a = ap.parse_args()
    pcs = None if a.pieces is None else object_pieces(a.pieces.split(":")[0], float(a.pieces.split(":")[1]))
    _, r = simify_scene(a.scene, a.out, a.out.parent / "sim_meshes", a.object_body, pcs)
    print(json.dumps(r, indent=1))
