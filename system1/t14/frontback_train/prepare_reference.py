#!/usr/bin/env python3
"""T14 front-back training task (Kevin 2026-11-26, after step 9g2), item 2.1-2.3: training assets of the front-back carry on the
Table Transport code path (CORE4D pair, experiment smalltable5kg_A, --t14-reference manifest), behind a manifest switch.

Reference: sub8_suitcase_020, level 1.0, frozen step-9g2 construction (exports_system1/T14_references/frontback_train/
sub8_suitcase_020_m1p00: the 9g2 pair copied byte for byte, its source_manifest.json, and the 50 Hz runtime made by
scripts/synthesize_core4d_smalltable_runtime_reference.py, the converter the Table reference used).
Object: the construction's lengthened box at level 1.0 (the solve's target URDF size), one box collision geom, link origin =
box centre; mass 5 kg (Kevin), uniform-density inertia of that box; material (static, dynamic, restitution) = Table Transport's
(0.5, 0.5, 0) through the experiment (not in the URDF, as for the table).
Everything else as prepare_table() of system1/t13v4/t14_reference.py: bucket interaction vectors from the new runtime (same
landmarks and mapping, object points = trimesh even surface sampling of the URDF box, 100 requested, seed 42), a promotion
manifest with the table's fields (values of this object / reference) and a t14_reference.json manifest with
"object_replacement" (the switch read by t14_reference.table_experiment: new object URDF and reference length; absent = as before).

    hsretargeting_s1/bin/python -B system1/t14/frontback_train/prepare_reference.py
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "src/holosoma"
MOTION = PACKAGE / "holosoma/data/motions/g1_29dof/whole_body_tracking"
OUT = MOTION / "system1_t14fb/table"
SRC = ROOT / "exports_system1/T14_references/frontback_train/sub8_suitcase_020_m1p00"
TABLE = MOTION / "system1_t14r6C/table"  # the Table Transport reference of d_table (fields mirrored)
MASS_KG = 5.0
MATERIAL = (0.5, 0.5, 0.0)
SCHEMA = "system1_t14_reference_v1"


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rel(p) -> str:
    return str(Path(p).resolve().relative_to(PACKAGE))


def _write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False,
                                     default=lambda x: x.tolist() if isinstance(x, (np.ndarray, np.generic)) else str(x)) + "\n")


def box_urdf(size, mass):
    a, b, c = (float(x) for x in size)
    ixx, iyy, izz = mass / 12 * (b * b + c * c), mass / 12 * (a * a + c * c), mass / 12 * (a * a + b * b)
    s = f"{a:.12g} {b:.12g} {c:.12g}"
    return f"""<?xml version='1.0' encoding='utf-8'?>
<robot name="frontback_box_5kg_training">
  <material name="frontback_box_material">
    <color rgba="0.76 0.80 0.86 1.0" />
  </material>
  <link name="box_link">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <mass value="{mass:.12g}" />
      <inertia ixx="{ixx:.12g}" ixy="0" ixz="0" iyy="{iyy:.12g}" iyz="0" izz="{izz:.12g}" />
    </inertial>
    <visual name="box_visual">
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <box size="{s}" />
      </geometry>
      <material name="frontback_box_material" />
    </visual>
    <collision name="box">
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <box size="{s}" />
      </geometry>
    </collision>
  </link>
</robot>
"""


def main():
    sys.path.insert(0, str(ROOT / "src/holosoma_retargeting"))
    from holosoma_retargeting.interaction_mesh_geometry import build_robot_landmarks, landmarks_world, load_box_object_mesh, sample_object_points
    from holosoma_retargeting.interaction_mesh_training_reference import map_landmarks_to_training, training_body_names
    from scipy.spatial.transform import Rotation

    smf = json.loads((SRC / "source_manifest.json").read_text())
    pair_src, runtime_src = SRC / "sub8_suitcase_020_m1p00_pair_fps30.npz", SRC / "core4d_pair_runtime_fps50.npz"
    rman = json.loads((SRC / "core4d_pair_runtime_fps50.manifest.json").read_text())
    if smf["pair_sha256"] != sha(pair_src) or rman["source_sha256"] != sha(pair_src) or rman["output_sha256"] != sha(runtime_src):
        raise ValueError("front-back pair / runtime changed since export")
    target = ROOT / smf["target_box_urdf"]
    if sha(target) != smf["target_box_urdf_sha256"]:
        raise ValueError("construction target box URDF changed")
    import xml.etree.ElementTree as ET
    boxes = ET.parse(target).getroot().findall(".//collision/geometry/box")
    if len(boxes) != 1:
        raise ValueError("target URDF must hold one box")
    size = [float(x) for x in boxes[0].get("size").split()]
    with np.load(pair_src, allow_pickle=False) as p:
        if float(p["m"]) != 1.0:
            raise ValueError("pair is not level 1.0")
    OUT.mkdir(parents=True, exist_ok=False)
    runtime, pair, urdf = OUT / "core4d_pair_runtime_fps50.npz", OUT / pair_src.name, OUT / "frontback_box5kg_training.urdf"
    shutil.copyfile(runtime_src, runtime)
    shutil.copyfile(pair_src, pair)
    shutil.copyfile(SRC / "source_manifest.json", OUT / "source_manifest.json")
    urdf.write_text(box_urdf(size, MASS_KG))
    robot_xml = ROOT / "src/holosoma_retargeting/holosoma_retargeting/models/g1/g1_29dof.xml"
    robot_urdf = PACKAGE / "holosoma/data/robots/g1/main_mesh_collision_rubberhand.urdf"
    specs, landmark_metadata = build_robot_landmarks(robot_xml)
    mapped, mapping_metadata = map_landmarks_to_training(specs, robot_xml, robot_urdf, training_body_names(PACKAGE / "holosoma/config_values/robot.py"))
    with np.load(runtime, allow_pickle=False) as z:
        T = int(z["agent_joint_pos"].shape[0])
        if z["agent_joint_pos"].shape != (T, 2, 29) or z["fps"].item() != 50:
            raise ValueError("Unexpected runtime layout")
        points = landmarks_world(z["agent_body_pos_w"], z["agent_body_quat_w"], z["body_names"], specs)
        mapped_points = landmarks_world(z["agent_body_pos_w"], z["agent_body_quat_w"], z["body_names"], mapped)
        rotation = Rotation.from_quat(z["object_quat_w"][:, [1, 2, 3, 0]]).as_matrix()
        object_pos = z["object_pos_w"].copy()
    error = float(np.linalg.norm(points - mapped_points, axis=-1).max())
    if points.shape != (T, 2, 19, 3) or error > 1e-6:
        raise ValueError("Fixed-chain landmark mapping mismatch")
    mesh, mesh_meta = load_box_object_mesh(urdf)
    object_points, sampling = sample_object_points(mesh, mesh_meta, 100, seed=42)
    objects = np.einsum("tij,nj->tni", rotation, object_points) + object_pos[:, None]
    metadata = {
        "version": "core4d_bucket_vectors_v1", "experiment_id": "smalltable5kg_A",
        "runtime_reference_sha256": sha(runtime), "source_pair_sha256": sha(pair),
        "object_urdf_sha256": sha(urdf), "training_robot_urdf_sha256": sha(robot_urdf),
        "num_frames": T, "num_agents": 2, "fps": 50, "num_body_points": 19,
        "actual_object_points": len(object_points), "sampling": sampling,
        "robot_landmarks": landmark_metadata, "point_mapping": mapping_metadata,
        "fixed_chain_reference_equivalence_max_m": error,
        "contact_target": "not_used_for_smalltable_A",
        "reference_contact_weights_meaning": "all-zero unused compatibility mask; not estimated contact labels; B forbidden",
        "coordinate_frame": "world relative vectors; object points in URDF link frame, not COM",
        "reward_reuse": "unchanged bucket A formulas and weights, front-back box geometry/reference",
        "system1_t14_reference": "T14 front-back, sub8_suitcase_020, level 1.0, frozen 9g2 construction (Kevin 2026-11-26)",
    }
    arrays = {
        "reference_body_points_world": points.astype(np.float32),
        "reference_object_points_world": objects.astype(np.float32),
        "object_points": object_points,
        "point_offsets": np.asarray([s["local_xyz"] for s in mapped], dtype=np.float32),
        "point_body_names": np.asarray([s["body"] for s in mapped]),
        "point_names": np.asarray([s["name"] for s in mapped]),
        "body_point_weights": np.asarray([1 / 3 if s["group"] in ("left_hand", "right_hand") else 1 for s in specs], dtype=np.float32),
        "reference_contact_weights": np.zeros((T, 2), dtype=np.float32),
        "fps": np.asarray(50, dtype=np.int64),
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True, default=lambda x: x.tolist() if isinstance(x, (np.ndarray, np.generic)) else str(x))),
    }
    with np.load(TABLE / "interaction_vectors_v1.npz", allow_pickle=False) as o:  # same layout as the Table artifact
        if set(o.files) != set(arrays):
            raise ValueError("artifact keys differ from the Table artifact")
        for k in ("point_offsets", "point_body_names", "point_names", "body_point_weights"):
            if not np.array_equal(o[k], arrays[k]):
                raise ValueError(f"robot-side array {k} differs from the Table artifact")
    artifact = OUT / "interaction_vectors_v1.npz"
    with artifact.open("xb") as f:
        np.savez_compressed(f, **arrays)
    old_prom = json.loads((TABLE / "training_asset_manifest.json").read_text())
    prom = dict(old_prom)
    for k in ("inertia_scale_from_20kg", "original_20kg_urdf_sha256", "physics_loading_check", "physics_loading_report_sha256", "system1_t14_reference"):
        prom.pop(k, None)
    prom.update(object_name="frontback_box", object_points=len(object_points), reference_frames=T, object_mass_kg=MASS_KG,
                object_material_static_dynamic_restitution=list(MATERIAL), object_inertia="uniform-density box",
                object_box_size_m=size, training_object_urdf_sha256=sha(urdf), source_pair_sha256=sha(pair),
                runtime_reference_sha256=sha(runtime), interaction_artifact_sha256=sha(artifact),
                fixed_chain_reference_equivalence_max_m=error, preparation_script_sha256=sha(__file__),
                physics_loading_check="pending: front-back smoke (system1/t14/frontback_train)",
                system1_t14_reference=dict(source="T14 front-back, sub8_suitcase_020, level 1.0, frozen 9g2 construction",
                                           runtime_source=str(runtime_src.relative_to(ROOT)),
                                           table_manifest_sha256=sha(TABLE / "training_asset_manifest.json")))
    _write_json(OUT / "training_asset_manifest.json", prom)
    files = dict(runtime_reference=runtime, source_pair=pair, object_urdf=urdf, interaction_artifact=artifact,
                 training_promotion=OUT / "training_asset_manifest.json", source_manifest=OUT / "source_manifest.json")
    manifest = dict(schema=SCHEMA, task="table", name="t14_frontback_sub8_suitcase_020_m1p00", parent_experiment="smalltable5kg_A",
                    decision="Kevin 2026-11-26: front-back training task = D on the Table Transport path; reference sub8_suitcase_020, "
                             "level 1.0, frozen 9g2 construction; object = the construction's lengthened box, 5 kg, Table Transport friction",
                    files={k: dict(path=rel(p), sha256=sha(p)) for k, p in files.items()},
                    object_replacement=dict(object_name="frontback_box", object_urdf=rel(urdf), object_urdf_sha256=sha(urdf),
                                            box_size_m=size, object_mass_kg=MASS_KG, material_static_dynamic_restitution=list(MATERIAL),
                                            reference_frames=T, reference_fps=50),
                    unchanged_vs_table=dict(robot_urdf=True, reward_formulas=True, fps=50, object_points_seed42=True,
                                            material=True, object_mass_kg=True, experiment="smalltable5kg_A"),
                    t14_solve_dirs=smf["solve_dirs"])
    _write_json(OUT / "t14_reference.json", manifest)
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
