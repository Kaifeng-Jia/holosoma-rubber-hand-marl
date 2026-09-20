#!/usr/bin/env python3
"""Prepare an isolated 5kg table and bucket-A vector-format data, offline only."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
MOTION = ROOT / "src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking"
OUTPUT = MOTION / "core4d_smalltable5kg_A"
OLD_URDF_SHA = "d4f166913ee6464dae1155428bdfe5169f63be94fbc8869535710bb6b672fc60"
RUNTIME_SHA = "582e76693f877c61b0b09ab3b584922f330aeb85cae6035f6b7cd2ae729ee153"
SOURCE_SHA = "d9d17e96f5f0b73aa76a1bf32f1ec50cfb7fa7fe78a847dd890882add4bbfd05"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    with Path(path).open("x") as f:
        json.dump(data, f, indent=2, sort_keys=True, allow_nan=False,
                  default=lambda x: x.tolist() if isinstance(x, (np.ndarray, np.generic)) else str(x))


def scale_mass(source, output):
    tree = ET.parse(source)
    original = copy.deepcopy(tree.getroot())
    link = tree.getroot().find("link")
    if len(tree.getroot().findall("link")) != 1 or len(link.findall("collision")) != 5:
        raise ValueError("Expected the frozen single-link five-box table")
    mass = link.find("inertial/mass")
    if float(mass.get("value")) != 20.0:
        raise ValueError("Expected original 20kg mass")
    mass.set("value", "5.0")
    inertia = link.find("inertial/inertia")
    for key in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz"):
        inertia.set(key, format(float(inertia.get(key)) * .25, ".15g"))
    # Compare every other element, including COM, geometry and collision origins.
    comparison = copy.deepcopy(tree.getroot())
    comparison.find("link").remove(comparison.find("link/inertial"))
    original.find("link").remove(original.find("link/inertial"))
    if ET.tostring(comparison) != ET.tostring(original):
        raise ValueError("Unexpected non-inertial change")
    ET.indent(tree, space="  ")
    with Path(output).open("xb") as f:
        tree.write(f, encoding="utf-8", xml_declaration=True)


def prepare():
    sys.path.insert(0, str(ROOT / "src/holosoma_retargeting"))
    from holosoma_retargeting.interaction_mesh_geometry import (
        build_robot_landmarks, landmarks_world, load_box_object_mesh, sample_object_points,
    )
    from holosoma_retargeting.interaction_mesh_training_reference import (
        map_landmarks_to_training, training_body_names,
    )
    old = MOTION / "objects_core4d_desk001_small_training.urdf"
    runtime = MOTION / "core4d_smalltable/core4d_pair_runtime_fps50.npz"
    if digest(old) != OLD_URDF_SHA or digest(runtime) != RUNTIME_SHA:
        raise ValueError("Frozen original smalltable inputs changed")
    OUTPUT.mkdir(exist_ok=False)
    urdf = OUTPUT / "desk001_5kg_training.urdf"
    scale_mass(old, urdf)
    shutil.copyfile(runtime, OUTPUT / runtime.name)
    robot_xml = ROOT / "src/holosoma_retargeting/holosoma_retargeting/models/g1/g1_29dof.xml"
    robot_urdf = ROOT / "src/holosoma/holosoma/data/robots/g1/main_mesh_collision_rubberhand.urdf"
    robot_config = ROOT / "src/holosoma/holosoma/config_values/robot.py"
    specs, landmark_metadata = build_robot_landmarks(robot_xml)
    mapped, mapping_metadata = map_landmarks_to_training(
        specs, robot_xml, robot_urdf, training_body_names(robot_config))
    with np.load(runtime, allow_pickle=False) as z:
        if z["agent_joint_pos"].shape != (687, 2, 29) or z["fps"].item() != 50:
            raise ValueError("Unexpected original reference layout")
        points = landmarks_world(z["agent_body_pos_w"], z["agent_body_quat_w"], z["body_names"], specs)
        mapped_points = landmarks_world(z["agent_body_pos_w"], z["agent_body_quat_w"], z["body_names"], mapped)
        rotation = Rotation.from_quat(z["object_quat_w"][:, [1, 2, 3, 0]]).as_matrix()
        object_pos = z["object_pos_w"].copy()
    error = float(np.linalg.norm(points - mapped_points, axis=-1).max())
    if points.shape != (687, 2, 19, 3) or error > 1e-6:
        raise ValueError("Fixed-chain landmark mapping mismatch")
    mesh, mesh_meta = load_box_object_mesh(urdf)
    object_points, sampling = sample_object_points(mesh, mesh_meta, 100, seed=42)
    objects = np.einsum("tij,nj->tni", rotation, object_points) + object_pos[:, None]
    metadata = {
        "version": "core4d_bucket_vectors_v1", "experiment_id": "smalltable5kg_A",
        "runtime_reference_sha256": RUNTIME_SHA, "source_pair_sha256": SOURCE_SHA,
        "object_urdf_sha256": digest(urdf), "training_robot_urdf_sha256": digest(robot_urdf),
        "num_frames": 687, "num_agents": 2, "fps": 50, "num_body_points": 19,
        "actual_object_points": len(object_points), "sampling": sampling,
        "robot_landmarks": landmark_metadata, "point_mapping": mapping_metadata,
        "fixed_chain_reference_equivalence_max_m": error,
        "contact_target": "not_used_for_smalltable_A",
        "reference_contact_weights_meaning": "all-zero unused compatibility mask; not estimated contact labels; B forbidden",
        "coordinate_frame": "world relative vectors; object points in URDF link frame, not COM",
        "reward_reuse": "unchanged bucket A formulas and weights, table-specific geometry/reference",
    }
    arrays = {
        "reference_body_points_world": points.astype(np.float32),
        "reference_object_points_world": objects.astype(np.float32),
        "object_points": object_points,
        "point_offsets": np.asarray([s["local_xyz"] for s in mapped], dtype=np.float32),
        "point_body_names": np.asarray([s["body"] for s in mapped]),
        "point_names": np.asarray([s["name"] for s in mapped]),
        "body_point_weights": np.asarray([1/3 if s["group"] in ("left_hand", "right_hand") else 1 for s in specs], dtype=np.float32),
        "reference_contact_weights": np.zeros((687, 2), dtype=np.float32),
        "fps": np.asarray(50, dtype=np.int64),
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True, default=lambda x: x.tolist() if isinstance(x, (np.ndarray, np.generic)) else str(x))),
    }
    for a in arrays.values():
        if np.issubdtype(a.dtype, np.number) and not np.isfinite(a).all():
            raise ValueError("Nonfinite prepared array")
    artifact = OUTPUT / "interaction_vectors_v1.npz"
    with artifact.open("xb") as f:
        np.savez_compressed(f, **arrays)
    promotion = {
        "experiment_id": "smalltable5kg_A", "object_name": "desk001",
        "source_pair_sha256": SOURCE_SHA, "runtime_reference_sha256": RUNTIME_SHA,
        "training_object_urdf_sha256": digest(urdf), "interaction_artifact_sha256": digest(artifact),
        "training_robot_urdf_sha256": digest(robot_urdf),
        "object_mass_kg": 5.0, "inertia_scale_from_20kg": .25,
        "object_material_static_dynamic_restitution": [.5, .5, 0.],
        "object_collider_type": "convex_hull", "reference_frames": 687, "reference_fps": 50,
        "physics_hz": 200, "control_hz": 50, "object_points": len(object_points),
        "contact_target": "not_used_for_smalltable_A", "allowed_reward_variant": "A",
        "training_ready": False, "physics_loading_check": "pending",
        "formal_training_authorized": False, "preparation_script_sha256": digest(__file__),
        "original_20kg_urdf_sha256": OLD_URDF_SHA,
        "fixed_chain_reference_equivalence_max_m": error,
    }
    write_json(OUTPUT / "training_asset_manifest.json", promotion)
    if digest(old) != OLD_URDF_SHA or digest(runtime) != RUNTIME_SHA:
        raise ValueError("Original inputs changed during preparation")
    print(json.dumps(promotion, indent=2))


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    prepare()
