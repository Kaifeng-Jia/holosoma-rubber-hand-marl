#!/usr/bin/env python3
"""T14 step 1, prerequisite: the rubber-hand collision shape in the three places (read only, CPU).

(a) retargeting: MuJoCo mesh geom left/right_rubber_hand_link of src/holosoma_retargeting/.../models/g1/g1_29dof.xml
    (left/right_rubber_hand.obj; MuJoCo collides a mesh as its convex hull);
(b) T10 MuJoCo sim2sim scene: built in memory from the Isaac dump, i.e. the same hulls as (c);
(c) PhysX (Isaac): the convex hull PhysX cooked from main_mesh_collision_rubberhand.urdf's STL, exported in the T10 dump
    (logs_system1/t10/isaac/d_table/dump: scene.json + collision_convexes.npz), in the hand link frame (T12-v2 dump_shapes).
Also the raw STL's exact convex hull, to separate "STL vs OBJ" from "PhysX hull simplification".

All three are compared in the hand link frame (the URDF link and the MJCF body are the same frame: checked by comparing
the MJCF body pose chain with the URDF joint origins for the wrist-yaw -> rubber-hand step). Reported per shape: vertices,
volume, extents, oriented-box extents; between shapes: symmetric surface distance (mean / p95 / max over 20 000 surface
samples each way), and the palm point of _derive_palm_contact_point recomputed on each shape (support plane along the
hard-coded palm normal, bounding-box centre moved onto it).

    hssim_s1/bin/python -B system1/t14/hand_shapes.py --out results_system1/T14_retargeting/step1/hand_shapes.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
import yourdfpy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "system1/t12v2"))
import contact_stats as cs  # noqa: E402

RT = ROOT / "src/holosoma_retargeting/holosoma_retargeting"
MJCF = RT / "models/g1/g1_29dof.xml"
URDF = ROOT / "src/holosoma/holosoma/data/robots/g1/main_mesh_collision_rubberhand.urdf"
PALM_NORMAL = dict(left=np.array([-0.0751, -0.9954, -0.0594]), right=np.array([-0.0751, 0.9954, -0.0593]))  # interaction_mesh_retargeter.py:361-364


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def palm_point(vertices, side):
    """As InteractionMeshRetargeter._derive_palm_contact_point: bounding-box centre moved along the palm normal onto the
    support plane of the vertices."""
    n = PALM_NORMAL[side] / np.linalg.norm(PALM_NORMAL[side])
    v = np.asarray(vertices, float)
    c = 0.5 * (v.min(0) + v.max(0))
    support = float(np.max(v @ n))
    return c + (support - c @ n) * n


def mj_hand_mesh(side):
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(MJCF))
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_rubber_hand_link")
    mid = int(m.geom_dataid[g])
    a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
    V = m.mesh_vert[a:a + n].astype(np.float64)
    fa, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
    F = m.mesh_face[fa:fa + fn]
    # MuJoCo re-centres meshes; geom_pos / geom_quat put them back in the body frame
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, m.geom_quat[g])
    V = V @ R.reshape(3, 3).T + m.geom_pos[g]
    return trimesh.Trimesh(V, F, process=False)


def surf_dist(a, b, n=20000):
    pa, _ = trimesh.sample.sample_surface(a, n, seed=0)
    pb, _ = trimesh.sample.sample_surface(b, n, seed=1)
    da = np.abs(trimesh.proximity.signed_distance(b, pa))
    db = np.abs(trimesh.proximity.signed_distance(a, pb))
    d = np.concatenate([da, db])
    return dict(mean_mm=round(1e3 * float(d.mean()), 2), p95_mm=round(1e3 * float(np.quantile(d, 0.95)), 2), max_mm=round(1e3 * float(d.max()), 2))


def describe(m):
    h = m.convex_hull
    return dict(vertices=int(len(m.vertices)), hull_vertices=int(len(h.vertices)), hull_volume_cm3=round(1e6 * float(h.volume), 2),
                extents_cm=[round(100 * float(x), 2) for x in m.extents], obb_cm=[round(100 * float(x), 2) for x in np.sort(h.bounding_box_oriented.primitive.extents)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    u = yourdfpy.URDF.load(str(URDF), load_collision_meshes=True, build_collision_scene_graph=True)
    sc = u.collision_scene
    physx = dict(cs.dump_shapes("table", "/World/envs/env_0/Robot/")[0])
    rec = dict(mjcf=str(MJCF.relative_to(ROOT)), mjcf_sha256=sha(MJCF), urdf=str(URDF.relative_to(ROOT)), urdf_sha256=sha(URDF), hands={})
    for side in ("left", "right"):
        link = f"{side}_rubber_hand_link"
        node = [n for n in sc.graph.nodes_geometry if sc.graph.transforms.parents[n] == link][0]
        T, g = sc.graph.get(frame_to=node, frame_from=link)
        stl = sc.geometry[g].copy().apply_transform(T)
        shapes = dict(retarget_mjcf_obj=mj_hand_mesh(side), stl_exact_hull=stl.convex_hull, physx_cooked_hull=physx[link])
        r = dict(shapes={k: describe(v) for k, v in shapes.items()},
                 palm_point_m={k: [round(float(x), 6) for x in palm_point(v.vertices, side)] for k, v in shapes.items()},
                 distance={f"{x}__vs__{y}": surf_dist(shapes[x].convex_hull, shapes[y].convex_hull) for x, y in
                           (("retarget_mjcf_obj", "physx_cooked_hull"), ("stl_exact_hull", "physx_cooked_hull"), ("retarget_mjcf_obj", "stl_exact_hull"))})
        p = np.array(r["palm_point_m"]["physx_cooked_hull"]) - np.array(r["palm_point_m"]["retarget_mjcf_obj"])
        r["palm_point_shift_physx_minus_retarget_mm"] = [round(1e3 * float(x), 2) for x in p]
        rec["hands"][side] = r
        print(side, json.dumps(r), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rec, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
