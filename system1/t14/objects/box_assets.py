"""T14 step 7, layer 1: box approximation of an OMOMO captured object for the retargeting scene (robot side), and the
object's pose trajectory in the OMOMO_new (InterMimic) world of the retargeting pipeline.

Box = the oriented bounding box (trimesh bounding_box_oriented) of the captured mesh (captured_objects/<name>_cleaned_simplified.obj)
x the sequence's obj_scale (median over valid frames): the real captured size. (Axis-aligned boxes in the mesh frame do not
fit: largebox / smallbox / suitcase / trashcan are rotated in their mesh frames, axis-aligned box volume 2-3 x the oriented one.)
Box frame: origin = box centre; z = the box axis closest to the world vertical at the sequence's first frame (pointing up),
x = the next box axis in trimesh's order, y = z x x.
Pose in OMOMO_new: the captured pose (obj_rot, obj_trans, obj_scale of the original release) composed with the box frame, then
the rigid transform that maps the original release's SMPL-X body joints onto OMOMO_new's (single_shared.omomo_palms fit; the
same transform the person's palms get). Heights are not changed (the pipeline keeps the object's first-frame height).
Approximation quality (reported, not used): hull volume / box volume; box-surface coverage = share of 4000 uniform box-surface
samples within 2 cm of the captured mesh (obj_cache dense samples).

Assets written per sequence: <name>.urdf (one link <name>_link, one collision + visual box), <name>.obj (the box mesh, for the
interaction-mesh object points), g1_29dof_w_<name>.xml (the largebox retargeting scene with its object body replaced by a box
geom of the same contact parameters), box.json.
"""
from __future__ import annotations

import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
import omomo_motion as om  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
RT = ROOT / "src/holosoma_retargeting/holosoma_retargeting"
TEMPLATE = RT / "models/g1/g1_29dof_w_largebox.xml"
CACHE = ROOT / "logs_system1/t14/obj_cache"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def capture_mesh(name):
    return trimesh.load(om.OMOMO / f"captured_objects/{name}_cleaned_simplified.obj", force="mesh", process=True)


def box_frame(name, v):
    """-> dict(R_obb [3,3] (mesh axes -> box axes as columns), c_obb [3] (mesh units), extents_m [3], scale)."""
    mesh = capture_mesh(name)
    obb = mesh.bounding_box_oriented
    T, ext = np.asarray(obb.primitive.transform, float), np.asarray(obb.primitive.extents, float)
    s = np.asarray(v["obj_scale"], float)
    ok = np.isfinite(s) & (s > 0)
    scale = float(np.median(s[ok]))
    R0 = np.asarray(v["obj_rot"], float)[np.flatnonzero(ok)[0]]
    world = R0 @ T[:3, :3]  # box axes in the world at the first valid frame
    k = int(np.argmax(np.abs(world[2])))
    z = T[:3, k] * np.sign(world[2, k])
    rest = [i for i in range(3) if i != k]
    x = T[:3, rest[0]]
    y = np.cross(z, x)
    R = np.stack([x, y, z], 1)
    extents = np.array([ext[rest[0]], ext[rest[1]], ext[k]]) * scale
    return dict(R_obb=R, c_obb=T[:3, 3], extents_m=extents, scale=scale, mesh_sha256=sha(om.OMOMO / f"captured_objects/{name}_cleaned_simplified.obj"))


def mesh_in_box_frame(name, bf):
    m = capture_mesh(name)
    m.vertices = (np.asarray(m.vertices) - bf["c_obb"]) @ bf["R_obb"] * bf["scale"]
    return m


def quality(name, bf):
    from scipy.spatial import cKDTree
    m = mesh_in_box_frame(name, bf)
    box = trimesh.creation.box(extents=bf["extents_m"])
    pts, _ = trimesh.sample.sample_surface(box, 4000, seed=0)
    z = np.load(CACHE / f"omomo_{name}.npz")
    dense = (z["dense_p"].astype(np.float64) - bf["c_obb"]) @ bf["R_obb"] * bf["scale"]
    d, _ = cKDTree(dense).query(pts)
    return dict(hull_over_box_volume=round(float(m.convex_hull.volume / box.volume), 4), box_surface_within_2cm_share=round(float((d < 0.02).mean()), 4),
                box_surface_gap_p95_m=round(float(np.percentile(d, 95)), 4))


def poses_omomo_new(v, bf, fit):
    """[T, 7] (qw qx qy qz x y z) box poses in OMOMO_new; fit = single_shared.omomo_palms record (rotation, translation)."""
    from scipy.spatial.transform import Rotation
    s, R, t = np.asarray(v["obj_scale"], float), np.asarray(v["obj_rot"], float), np.asarray(v["obj_trans"], float)[:, :, 0]
    ok = np.isfinite(s) & (s > 0) & np.isfinite(R).all(axis=(1, 2)) & np.isfinite(t).all(axis=1)
    if not ok.all():
        good = np.flatnonzero(ok)
        near = good[np.abs(np.arange(len(s))[:, None] - good[None]).argmin(1)]
        s, R, t = s[near], R[near], t[near]
    Rf, tf = np.asarray(fit["rotation"], float), np.asarray(fit["translation"], float)
    Rw = Rf @ R @ bf["R_obb"]
    pw = (Rf @ (np.einsum("tij,j->ti", R, bf["c_obb"]) * s[:, None] + t).T).T + tf
    q = Rotation.from_matrix(Rw).as_quat()[:, [3, 0, 1, 2]]
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    return np.concatenate([q, pw], 1), int((~ok).sum())


def write_assets(name, bf, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    e = bf["extents_m"]
    size = " ".join(f"{x:.9g}" for x in e)
    urdf = out / f"{name}.urdf"
    urdf.write_text(f"""<?xml version="1.0" ?>
<robot name="{name}">
  <link name="{name}_link">
    <inertial><mass value="0.1"/><origin xyz="0 0 0"/><inertia ixx="0.002" ixy="0" ixz="0" iyy="0.002" iyz="0" izz="0.002"/></inertial>
    <visual name="{name}_box"><origin rpy="0 0 0" xyz="0 0 0"/><geometry><box size="{size}"/></geometry></visual>
    <collision name="{name}_box"><origin rpy="0 0 0" xyz="0 0 0"/><geometry><box size="{size}"/></geometry></collision>
  </link>
</robot>
""")
    mesh = out / f"{name}.obj"
    trimesh.creation.box(extents=e).export(mesh)
    tree = ET.parse(TEMPLATE)
    root = tree.getroot()
    comp = root.find("compiler")
    comp.set("meshdir", str((TEMPLATE.parent / comp.get("meshdir")).resolve()))
    asset = root.find("asset")
    for m in list(asset.findall("mesh")):
        if m.get("name") == "largebox_mesh":
            asset.remove(m)
    world = root.find("worldbody")
    body = next(b for b in world.findall("body") if b.get("name") == "largebox_link")
    body.set("name", f"{name}_link")
    g = body.find("geom")
    g.set("name", name)
    g.set("type", "box")
    del g.attrib["mesh"]
    g.set("size", " ".join(f"{x / 2:.9g}" for x in e))
    scene = out / f"g1_29dof_w_{name}.xml"
    tree.write(scene)
    rec = dict(object=name, extents_m=np.round(e, 6).tolist(), scale=bf["scale"], R_obb=np.round(bf["R_obb"], 9).tolist(), c_obb_mesh_units=np.round(bf["c_obb"], 9).tolist(),
               mesh_sha256=bf["mesh_sha256"], urdf=str(urdf), urdf_sha256=sha(urdf), mesh=str(mesh), mesh_box_sha256=sha(mesh), scene=str(scene), scene_sha256=sha(scene),
               template=str(TEMPLATE), template_sha256=sha(TEMPLATE), quality=quality(name, bf))
    (out / "box.json").write_text(json.dumps(rec, indent=1) + "\n")
    return rec
