#!/usr/bin/env python3
"""Prepare the reviewed bucket A1 assets offline, without simulation or training.

Contact weights are an explicit source-human geometric proxy, never force or
dataset contact ground truth. This script does not authorize formal training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh


SOURCE_PAIR_SHA256 = "ea1e73c80d98ac15602871a543bf1aa6ca221a7039fc8ba98b25b3b09cc094af"
SOURCE_CANONICAL_SHA256 = "1208deb13bf70daee2d42343c6728656cca8586ae46efc7a2bb718d514079083"
SOURCE_MESH_SHA256 = "2a95d824af8facbd363ce00eba0534cd7c341b51f9997e3b137990f5fb503bdf"
PREVIEW_REL = Path("logs/Core4DPreviews/bucket003_20231020_071_20260918")
FORMAT_VERSION = "core4d_bucket_vectors_v1"
PALM_JOINT_INDICES = np.asarray([[20, 25, 28, 31], [21, 40, 43, 46]])


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_hash(path: Path, expected: str) -> None:
    if sha256(path) != expected:
        raise ValueError(f"Reviewed input hash mismatch: {path}")


def json_default(value):
    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, default=json_default, allow_nan=False)
        stream.write("\n")


def intervals(mask: np.ndarray, fps: int) -> list[dict]:
    boundaries = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return [
        {"first_frame": int(first), "last_frame": int(stop - 1),
         "first_time_s": float(first / fps), "last_time_s": float((stop - 1) / fps)}
        for first, stop in zip(boundaries[::2], boundaries[1::2])
    ]


def contact_weights(distances: np.ndarray, inner: float, outer: float) -> np.ndarray:
    """Maximum over hands of the source-proximity soft band, independently per agent."""
    if not np.isfinite([inner, outer]).all() or not 0 <= inner < outer:
        raise ValueError("Require finite 0 <= contact-inner-m < contact-outer-m")
    return np.clip((outer - np.min(distances, axis=-1)) / (outer - inner), 0.0, 1.0)


def analyze_source_contacts(canonical: Path, mesh_path: Path) -> tuple[np.ndarray, dict]:
    """Unsigned distances from source-human palm skeleton centers to source mesh."""
    with np.load(canonical, allow_pickle=False) as archive:
        joints = archive["human_joints_full"].copy()
        # Canonical object layout is [wxyz, xyz], unlike compact qpos [xyz, wxyz].
        poses = archive["object_poses"].copy()
        fps = int(archive["fps"])
    if joints.shape != (150, 2, 127, 3) or poses.shape != (150, 7) or fps != 15:
        raise ValueError("Expected reviewed 150-frame, 15 Hz canonical source")
    centers = joints[:, :, PALM_JOINT_INDICES, :].mean(axis=3)
    rotation = Rotation.from_quat(poses[:, [1, 2, 3, 0]]).as_matrix()
    local = np.einsum("tji,tahj->tahi", rotation, centers - poses[:, None, None, 4:])
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    _, distance, _ = trimesh.proximity.closest_point(mesh, local.reshape(-1, 3))
    distances = distance.reshape(150, 2, 2)
    if not np.isfinite(distances).all() or np.any(distances < 0):
        raise ValueError("Source contact distances must be finite and nonnegative")
    report = {
        "semantic_kind": "source_human_geometric_proximity_proxy_not_contact_ground_truth",
        "source_canonical_sha256": sha256(canonical), "object_mesh_sha256": sha256(mesh_path),
        "source_frames": 150, "source_fps": 15,
        "palm_point_definition": "mean(wrist,index1,middle1,pinky1) of original unscaled SMPL-X joints",
        "palm_joint_indices_left_right": PALM_JOINT_INDICES.tolist(),
        "object_pose_layout": "source canonical [wxyz,xyz]; inverse rigid transform into object link frame",
        "distance_definition": "unsigned exact nearest triangle-surface distance; not hull distance",
        "agent_aggregation": "minimum distance over that person's two hands",
        "mesh_watertight": bool(mesh.is_watertight), "mesh_volume_m3": float(mesh.volume),
        "caveat": "Source joints and scanned mesh are not measured hand-surface contact, force, load sharing, or proof of a physically open bucket.",
        "distance_percentiles_m": {}, "threshold_sensitivity": {},
    }
    for agent in range(2):
        for hand, side in enumerate(("left", "right")):
            report["distance_percentiles_m"][f"agent{agent + 1}_{side}"] = dict(zip(
                ("min", "p10", "p25", "p50", "p75", "p90", "p95", "max"),
                np.percentile(distances[:, agent, hand], [0, 10, 25, 50, 75, 90, 95, 100]).tolist(),
            ))
    for threshold in (0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10):
        report["threshold_sensitivity"][str(threshold)] = {
            f"agent{agent + 1}": intervals(distances[:, agent].min(axis=-1) < threshold, fps)
            for agent in range(2)
        }
    return distances, report


def portable_training_urdf(source: Path, output: Path, mesh_name: str) -> dict:
    """Preserve full-size geometry/COM; explicitly normalize existing inertia to 1 kg."""
    tree = ET.parse(source)
    root = tree.getroot()
    links = root.findall("link")
    if len(links) != 1 or root.findall("joint"):
        raise ValueError("Expected a single rigid bucket link")
    link = links[0]
    inertial = link.find("inertial")
    if inertial is None:
        raise ValueError("Source bucket URDF must contain explicit mass/inertia")
    mass = inertial.find("mass")
    inertia = inertial.find("inertia")
    if mass is None or inertia is None:
        raise ValueError("Missing source mass or inertia")
    old_mass = float(mass.get("value", "nan"))
    if not np.isfinite(old_mass) or old_mass <= 0:
        raise ValueError("Invalid source mass")
    values = {name: float(inertia.get(name, "nan")) / old_mass
              for name in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")}
    tensor = np.asarray([[values["ixx"], values["ixy"], values["ixz"]],
                         [values["ixy"], values["iyy"], values["iyz"]],
                         [values["ixz"], values["iyz"], values["izz"]]])
    if not np.isfinite(tensor).all() or np.linalg.eigvalsh(tensor).min() <= 0:
        raise ValueError("Training inertia must be finite positive definite")
    mass.set("value", "1")
    for name, value in values.items():
        inertia.set(name, f"{value:.12g}")
    for kind in ("visual", "collision"):
        elements = link.findall(kind)
        if len(elements) != 1 or elements[0].find("geometry/mesh") is None:
            raise ValueError(f"Expected exactly one source bucket {kind} mesh")
        element = elements[0]
        mesh = element.find("geometry/mesh")
        if not np.array_equal(np.fromstring(mesh.get("scale", "1 1 1"), sep=" "), [1., 1., 1.]):
            raise ValueError("Reviewed bucket must remain at its original physical size")
        origin = element.find("origin")
        if origin is not None:
            for attr in ("xyz", "rpy"):
                if not np.array_equal(np.fromstring(origin.get(attr, "0 0 0"), sep=" "), np.zeros(3)):
                    raise ValueError("Reviewed bucket mesh must retain zero visual/collision origin")
        mesh.set("filename", mesh_name)
    ET.indent(tree, space="  ")
    with output.open("xb") as stream:
        tree.write(stream, encoding="utf-8", xml_declaration=True)
    return {"object_mass_kg": 1.0, "inertia_scale_from_source": 1.0 / old_mass,
            "inertia_tensor_kg_m2": tensor.tolist(),
            "inertia_basis": "source scanned-mesh uniform-volume inertia normalized to selected 1kg experiment load; not measured hollow-shell inertia"}


def prepare(repo: Path, output: Path | None, inner: float | None, outer: float | None,
            *, analyze_only: bool = False) -> dict:
    repo = repo.resolve()
    sys.path.insert(0, str(repo / "src/holosoma_retargeting"))
    sys.path.insert(0, str(repo / "scripts"))
    from holosoma_retargeting.core4d_pair_runtime_reference import build_core4d_pair_runtime_reference_file
    from holosoma_retargeting.interaction_mesh_geometry import build_robot_landmarks, landmarks_world, sample_object_points
    from holosoma_retargeting.interaction_mesh_training_reference import map_landmarks_to_training, training_body_names
    from audit_core4d_interaction_vectors import load_single_mesh

    preview = repo / PREVIEW_REL
    source_dir = preview / "a1_wrist_only"
    pair = source_dir / "core4d_pair_reference.npz"
    canonical = preview / "source_canonical.npz"
    source_manifest = source_dir / "manifest.json"
    source_urdf = source_dir / "assets/bucket003.urdf"
    manifest = json.loads(source_manifest.read_text())
    mesh_path = Path(manifest["object_mesh_path"])
    require_hash(pair, SOURCE_PAIR_SHA256)
    require_hash(canonical, SOURCE_CANONICAL_SHA256)
    require_hash(mesh_path, SOURCE_MESH_SHA256)
    require_hash(source_urdf, manifest["object_urdf_sha256"])
    if manifest["pair_sha256"] != SOURCE_PAIR_SHA256 or manifest["input_sha256"] != SOURCE_CANONICAL_SHA256:
        raise ValueError("Source manifest does not match reviewed A1 pair and canonical data")
    if manifest["object_mesh_sha256"] != SOURCE_MESH_SHA256:
        raise ValueError("Source manifest does not identify the reviewed bucket mesh")
    for element in ET.parse(source_urdf).getroot().findall(".//mesh"):
        source_mesh = (source_urdf.parent / element.get("filename", "")).resolve()
        require_hash(source_mesh, SOURCE_MESH_SHA256)
    with np.load(pair, allow_pickle=False) as archive:
        if str(archive["a1_solver_mode"].item()) != "wrist-only" or archive["robot_qpos"].shape != (299, 2, 36):
            raise ValueError("Expected the selected 299-frame wrist-only A1 reference")
    input_paths = {"source_pair": pair, "source_canonical": canonical,
                   "source_manifest": source_manifest, "source_urdf": source_urdf, "source_mesh": mesh_path}
    input_hashes = {name: sha256(path) for name, path in input_paths.items()}
    distances, contact_report = analyze_source_contacts(canonical, mesh_path)
    if analyze_only:
        return contact_report
    if output is None or inner is None or outer is None:
        raise ValueError("Asset preparation requires output-dir, contact-inner-m, and contact-outer-m")
    contact_weights(distances, inner, outer)  # Validate before writing anything.
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing output directory: {output}")
    output.mkdir(parents=True)
    compact = output / "core4d_pair_compact_fps30.npz"
    copied_manifest = output / "source_manifest.json"
    copied_canonical = output / "source_canonical.npz"
    copied_mesh = output / "bucket003_m.obj"
    for source, destination in ((pair, compact), (source_manifest, copied_manifest),
                                (canonical, copied_canonical), (mesh_path, copied_mesh)):
        shutil.copyfile(source, destination)
        if sha256(source) != sha256(destination):
            raise ValueError(f"Byte-preserving copy failed: {destination}")
    training_urdf = output / "bucket003_training.urdf"
    physics = portable_training_urdf(source_urdf, training_urdf, copied_mesh.name)
    runtime_path = output / "core4d_pair_runtime_fps50.npz"
    runtime = build_core4d_pair_runtime_reference_file(
        compact, runtime_path, target_fps=50, expected_source_sha256=SOURCE_PAIR_SHA256,
        source_manifest_path=copied_manifest,
    )
    if runtime.agent_joint_pos.shape != (497, 2, 29) or runtime.fps != 50:
        raise ValueError("Expected 497-frame/50Hz paired bucket runtime")
    runtime_hash = sha256(runtime_path)
    robot_xml = repo / "src/holosoma_retargeting/holosoma_retargeting/models/g1/g1_29dof.xml"
    robot_urdf = repo / "src/holosoma/holosoma/data/robots/g1/main_mesh_collision_rubberhand.urdf"
    robot_config = repo / "src/holosoma/holosoma/config_values/robot.py"
    specs, landmark_metadata = build_robot_landmarks(robot_xml)
    mapped, mapping_metadata = map_landmarks_to_training(
        specs, robot_xml, robot_urdf, training_body_names(robot_config),
    )
    points_world = landmarks_world(runtime.agent_body_pos_w, runtime.agent_body_quat_wxyz, runtime.body_names, specs)
    mapped_world = landmarks_world(runtime.agent_body_pos_w, runtime.agent_body_quat_wxyz, runtime.body_names, mapped)
    mapping_error = float(np.linalg.norm(points_world - mapped_world, axis=-1).max())
    if points_world.shape != (497, 2, 19, 3) or mapping_error > 1e-6:
        raise ValueError(f"Fixed-chain landmark mapping mismatch: {mapping_error}")
    mesh, mesh_metadata = load_single_mesh(training_urdf)
    object_points, sampling = sample_object_points(mesh, mesh_metadata, 100, seed=42)
    if object_points.shape != (89, 3):
        raise ValueError(f"Reviewed seed42 bucket sample count changed: {object_points.shape}")
    # The old audit shares only the object surface sampling, never its baseline robot points.
    old_audit = preview / "interaction_vector_audit/audit_arrays.npz"
    with np.load(old_audit, allow_pickle=False) as archive:
        if not np.array_equal(object_points, archive["object_points_local_m"]):
            raise ValueError("Bucket surface points differ from reviewed seed42 89-point sample")
    rotation = Rotation.from_quat(runtime.object_quat_wxyz[:, [1, 2, 3, 0]]).as_matrix()
    object_world = np.einsum("tij,nj->tni", rotation, object_points) + runtime.object_pos_w[:, None]
    prior = np.asarray([1. / 3. if spec["group"] in ("left_hand", "right_hand") else 1. for spec in specs])
    source_times = np.arange(150) / 15.
    runtime_times = np.arange(497) / 50.
    interpolated_distances = np.stack([
        np.interp(runtime_times, source_times, distances[:, agent, hand])
        for agent in range(2) for hand in range(2)
    ], axis=-1).reshape(497, 2, 2)
    weights = contact_weights(interpolated_distances, inner, outer)
    contact_settings = {
        "kind": contact_report["semantic_kind"], "inner_m": inner, "outer_m": outer,
        "formula": "clip((outer-min(left_distance,right_distance))/(outer-inner),0,1)",
        "time_mapping": "linear interpolate original 15Hz hand distances to uniform runtime 50Hz, then apply soft band",
        "source_canonical_sha256": SOURCE_CANONICAL_SHA256,
        "source_palm_definition": contact_report["palm_point_definition"],
        "source_palm_joint_indices": PALM_JOINT_INDICES.tolist(),
        "not_from_retargeted_robot_penetration": True, "dataset_contact_ground_truth": False,
        "threshold_status": "explicit_candidate_pending_user_review_before_formal_training",
        "runtime_positive_intervals": {f"agent{agent+1}": intervals(weights[:, agent] > 0, 50) for agent in range(2)},
        "runtime_unit_intervals": {f"agent{agent+1}": intervals(weights[:, agent] >= 1, 50) for agent in range(2)},
    }
    metadata = {
        "version": FORMAT_VERSION, "runtime_reference_sha256": runtime_hash,
        "source_pair_sha256": SOURCE_PAIR_SHA256, "source_canonical_sha256": SOURCE_CANONICAL_SHA256,
        "object_urdf_sha256": sha256(training_urdf), "object_mesh_sha256": SOURCE_MESH_SHA256,
        "object_points_sha256": hashlib.sha256(object_points.tobytes()).hexdigest(),
        "training_robot_urdf_sha256": sha256(robot_urdf), "reference_robot_xml_sha256": sha256(robot_xml),
        "robot_config_sha256": sha256(robot_config), "num_frames": 497, "fps": 50,
        "num_agents": 2, "num_body_points": 19, "num_object_points": 89,
        "body_point_weights_definition": "1 per original nonhand group; each hand's three landmarks each 1/3",
        "robot_landmarks": landmark_metadata, "point_mapping": mapping_metadata,
        "fixed_chain_reference_equivalence_max_m": mapping_error,
        "sampling": sampling, "point_groups": [spec["group"] for spec in specs],
        "coordinate_frame": "reference points world; object_points local to URDF link origin, not COM",
        "source_geometric_contact": contact_settings,
        "semantic_limit": "Geometry/proximity only, not contact truth, measured force, or load sharing",
    }
    artifact_path = output / "interaction_vectors_v1.npz"
    arrays = {
        "reference_body_points_world": points_world.astype(np.float32),
        "reference_object_points_world": object_world.astype(np.float32),
        # Preserve the audited sampler's exact float64 points and their hash.
        "object_points": object_points,
        "point_offsets": np.asarray([spec["local_xyz"] for spec in mapped], dtype=np.float32),
        "point_body_names": np.asarray([spec["body"] for spec in mapped]),
        "point_names": np.asarray([spec["name"] for spec in mapped]),
        "body_point_weights": prior.astype(np.float32),
        "reference_contact_weights": weights.astype(np.float32),
        "fps": np.asarray(50, dtype=np.int64),
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True, default=json_default, allow_nan=False)),
    }
    for key, value in arrays.items():
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError(f"Nonfinite artifact field: {key}")
    with artifact_path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    with (output / "source_contact_diagnostics.npz").open("xb") as stream:
        np.savez_compressed(stream, source_hand_surface_distances_m=distances,
                            runtime_hand_surface_distances_m=interpolated_distances,
                            runtime_contact_weights=weights, source_fps=np.asarray(15), runtime_fps=np.asarray(50))
    contact_report["selected_candidate_soft_band"] = contact_settings
    write_json(output / "source_contact_diagnostics.json", contact_report)
    runtime_report = {"artifact_kind": "core4d_pair_runtime_reference", "training_ready": False,
                      "source_sha256": SOURCE_PAIR_SHA256, "output_sha256": runtime_hash,
                      "source_manifest_sha256": sha256(copied_manifest), "frames": 497, "fps": 50,
                      "provenance": dict(runtime.provenance)}
    write_json(output / "core4d_pair_runtime_fps50.manifest.json", runtime_report)
    promotion = {
        "artifact_kind": "core4d_pair_training_asset_promotion", "experiment_id": "bucket003",
        "object_name": "bucket003", "source_pair_sha256": SOURCE_PAIR_SHA256,
        "source_canonical_sha256": SOURCE_CANONICAL_SHA256,
        "source_manifest_sha256": sha256(copied_manifest),
        "source_manifest_status_preserved": "diagnostic_preview_not_training_asset",
        "runtime_reference_sha256": runtime_hash, "training_object_urdf_sha256": sha256(training_urdf),
        "object_mesh_file": copied_mesh.name, "object_mesh_sha256": SOURCE_MESH_SHA256,
        "object_mass_kg": 1.0, "object_material_static_dynamic_restitution": [0.5, 0.5, 0.0],
        "object_collider_type": "convex_decomposition", "object_scale": 1.0,
        "physics_hz": 200, "control_hz": 50, "reference_frames": 497, "reference_fps": 50,
        "interaction_artifact_file": artifact_path.name, "interaction_artifact_sha256": sha256(artifact_path),
        "training_robot_urdf_sha256": sha256(robot_urdf), "source_geometric_contact": contact_settings,
        "mass_basis": "user_selected_experiment_load_not_dataset_measured_mass", **physics,
        "collision_scope": "source mesh preserved; planned Isaac convex_decomposition, hollow opening not physically verified",
        "physics_loading_check": "not_run", "training_ready": False,
        "formal_training_authorized": False,
        "preparation_scope": "offline_artifact_preparation_only_no_simulation_no_PPO_update",
        "source_input_hashes": input_hashes,
        "script_sha256": sha256(Path(__file__)),
    }
    if input_hashes != {name: sha256(path) for name, path in input_paths.items()}:
        raise ValueError("Input artifact changed during offline preparation")
    write_json(output / "training_asset_manifest.json", promotion)
    return {"output_dir": str(output), "runtime_reference_sha256": runtime_hash,
            "interaction_artifact_sha256": promotion["interaction_artifact_sha256"],
            "frames": 497, "fps": 50, "object_points": 89,
            "fixed_chain_reference_equivalence_max_m": mapping_error,
            "source_geometric_contact": contact_settings, "training_ready": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--contact-inner-m", type=float)
    parser.add_argument("--contact-outer-m", type=float)
    parser.add_argument("--analyze-contact-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.analyze_contact_only and (args.output_dir is None or args.contact_inner_m is None or args.contact_outer_m is None):
        parser.error("Preparation requires --output-dir, --contact-inner-m, --contact-outer-m")
    result = prepare(args.repo_root, args.output_dir, args.contact_inner_m, args.contact_outer_m,
                     analyze_only=args.analyze_contact_only)
    print(json.dumps(result, indent=2, sort_keys=True, default=json_default, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
