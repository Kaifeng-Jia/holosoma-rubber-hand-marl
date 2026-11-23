#!/usr/bin/env python3
"""T14 stop 1, item 2: the actual size of each task's 1.0x object (read only, CPU).

1.0x = the object the current D training uses (summary.json object_urdf of the final evaluations, the same file as
docs/system1/codebase_map_zh.md section 4.1). For each of Push, Pull, Table, Bin, Chair:

* collision geometry of the training URDF (yourdfpy collision scene, as T12-v2 contact_stats.Object), extents along the
  object's own axes (URDF frame, Y = up for all five);
* the same geometry placed at reference frame 0 (paired runtime reference, loaded as the paper environment does,
  system1/sonic/task_refs.py): world axis-aligned extents (Z up) and the extent along the horizontal line between the two
  reference robots' pelvises at frame 0 (the "stretch axis" of T14 for the single-person tasks; informative for the
  two-person tasks), and perpendicular to it;
* where it comes from: for CORE4D, the raw CORE4D mesh, the two per-person human-to-robot scales, the shared scale in the
  current object and the ratio current / raw; for OMOMO, the source table (the 1.0x span vs the source width).

    hssim_s1/bin/python -B system1/t14/object_sizes.py --out results_system1/T14_retargeting/stop1
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
import yourdfpy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from system1.sonic.task_refs import load_task_reference  # noqa: E402

DATA = ROOT / "src/holosoma"
WBT = DATA / "holosoma/data/motions/g1_29dof/whole_body_tracking"
CORE4D = Path("/home/kevin/datasets/CORE4D-V1-full/CORE4D_Real/object_models")
URDF = dict(push="objects_widetable_plan5_training.urdf", pull="objects_widetable_plan5_pull_training.urdf",
            table="core4d_smalltable5kg_A/desk001_5kg_training.urdf", bucket="core4d_bucket003_20231020_071_a1/bucket003_training.urdf",
            chair="core4d_chair021_20231020_074_a1/chair021_training.urdf")
SOURCE = dict(push=("OMOMO", "sub6_largetable_033", "single person, role isometry (registered spacing 0.8780472 m)"),
              pull=("OMOMO", "sub3_largetable_010", "single person, mirrored role"),
              table=("CORE4D", "20231030/001 desk001", "two persons"),
              bucket=("CORE4D", "20231020/071 bucket003", "two persons"),
              chair=("CORE4D", "20231020/074 chair021", "two persons"))
MANIFEST = dict(table=WBT / "core4d_smalltable/source_manifest.json", bucket=WBT / "core4d_bucket003_20231020_071_a1/source_manifest.json",
                chair=WBT / "core4d_chair021_20231020_074_a1/source_manifest.json")
RAW = dict(table=CORE4D / "desk/desk001_m.obj", bucket=CORE4D / "bucket/bucket003_m.obj", chair=CORE4D / "chair/chair021_m.obj")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def quat_xyzw_to_R(q):
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def r4(x):
    return [round(float(v), 4) for v in np.atleast_1d(x)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rows, rec = [], {}
    for t in ("push", "pull", "table", "bucket", "chair"):
        u = WBT / URDF[t]
        mesh = yourdfpy.URDF.load(str(u), load_collision_meshes=True, build_collision_scene_graph=True).collision_scene.to_geometry()
        lo, hi = mesh.bounds
        ref, bodies, dofs, meta = load_task_reference(t)
        import torch
        s = ref.sample(torch.arange(1))
        pel = bodies.index("pelvis")
        p0 = s["agent_body_pos_w"][0, :, pel].numpy().astype(np.float64)  # [2, 3]
        op, oq = s["object_pos_w"][0].numpy().astype(np.float64), s["object_quat_w"][0].numpy().astype(np.float64)
        R = quat_xyzw_to_R(oq)
        V = mesh.vertices @ R.T + op
        wlo, whi = V.min(0), V.max(0)
        d = p0[1] - p0[0]
        d[2] = 0
        axis = d / np.linalg.norm(d)
        perp = np.array([-axis[1], axis[0], 0])
        along = V @ axis
        across = V @ perp
        local_axis_of_stretch = int(np.argmax(np.abs(R.T @ axis)))  # object axis closest to the robot-robot line
        r = dict(task=t, dataset=SOURCE[t][0], source=SOURCE[t][1], pairing=SOURCE[t][2], object_urdf=str(u.relative_to(DATA)), object_urdf_sha256=sha(u),
                 local_extent_xyz_m=r4(hi - lo), world_extent_xyz_m_frame0=r4(whi - wlo),
                 extent_along_robot_axis_m=round(float(along.max() - along.min()), 4), extent_across_robot_axis_m=round(float(across.max() - across.min()), 4),
                 height_m=round(float(whi[2] - wlo[2]), 4), top_height_above_ground_m_frame0=round(float(whi[2]), 4),
                 object_local_axis_closest_to_robot_axis="xyz"[local_axis_of_stretch], pelvis_separation_m_frame0=round(float(np.linalg.norm(d)), 4),
                 reference_file=meta["file"], reference_sha256=meta["sha256"])
        if t in MANIFEST:
            m = json.loads(MANIFEST[t].read_text())
            raw = trimesh.load(RAW[t], force="mesh")
            r.update(raw_mesh=str(RAW[t]), raw_mesh_sha256=sha(RAW[t]), raw_extent_xyz_m=r4(raw.extents),
                     person_scales=r4(m.get("source_human_to_robot_scales", m.get("human_to_robot_scales"))),
                     shared_object_scale_in_manifest=m.get("shared_object_scale"), baseline_kind=m.get("baseline_kind", m.get("scale_choice")),
                     current_over_raw_extent=r4((hi - lo) / raw.extents))
        rows.append(r)
        print(json.dumps(r), flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "object_sizes_1p0.json").write_text(json.dumps(rows, indent=1) + "\n")
    keys = ["task", "dataset", "source", "pairing", "extent_along_robot_axis_m", "extent_across_robot_axis_m", "height_m", "top_height_above_ground_m_frame0",
            "object_local_axis_closest_to_robot_axis", "local_extent_xyz_m", "world_extent_xyz_m_frame0", "pelvis_separation_m_frame0", "raw_extent_xyz_m",
            "person_scales", "shared_object_scale_in_manifest", "current_over_raw_extent", "baseline_kind", "object_urdf", "object_urdf_sha256", "raw_mesh_sha256"]
    with (a.out / "object_sizes_1p0.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (" ".join(map(str, r[k])) if isinstance(r.get(k), list) else r.get(k, "")) for k in keys})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
