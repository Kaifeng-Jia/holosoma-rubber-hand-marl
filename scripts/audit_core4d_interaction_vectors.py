#!/usr/bin/env python3
"""Audit fixed-correspondence interaction vectors using FK and synthetic offsets.

No simulation, retargeting, optimization, training, or RewardManager integration.
Actual equals reference except for explicitly labelled synthetic point edits;
these edits are not executable robot motions or evidence of contact/support.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import torch
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "holosoma_retargeting"))
sys.path.insert(0, str(ROOT / "src" / "holosoma"))

from visualize_core4d_pair import (  # noqa: E402
    EXPECTED_ROBOT_JOINT_NAMES,
    load_pair_reference,
    resolve_object_urdf,
)
from holosoma_retargeting.interaction_mesh_geometry import (  # noqa: E402
    build_robot_landmarks,
    landmarks_world,
    sample_object_points,
)

DEFAULT_XML = ROOT / "src/holosoma_retargeting/holosoma_retargeting/models/g1/g1_29dof.xml"
VECTOR_MODULE = ROOT / "src/holosoma/holosoma/managers/reward/terms/interaction_vectors.py"
# Import the pure module without importing RewardManager or simulator packages.
_module_spec = importlib.util.spec_from_file_location("offline_interaction_vectors", VECTOR_MODULE)
_vector_module = importlib.util.module_from_spec(_module_spec)
_module_spec.loader.exec_module(_vector_module)
weighted_body_object_vector_error = _vector_module.weighted_body_object_vector_error


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_default(value):
    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def numbers(raw: str | None, default: str) -> np.ndarray:
    values = np.asarray([float(part) for part in (raw or default).split()])
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("URDF vectors must contain exactly three finite numbers")
    return values


def load_single_mesh(urdf_path: Path) -> tuple[trimesh.Trimesh, dict]:
    """Load the actual single collision mesh in the URDF link-origin frame.

    Deliberately reject compound/box geometry rather than silently substituting
    a table proxy. A sole visual mesh is allowed only when collisions are absent.
    """
    root = ET.parse(urdf_path).getroot()
    links = root.findall("link")
    if len(links) != 1 or root.findall("joint"):
        raise ValueError("Object audit supports a single link with no joints only")
    kind = "collision" if links[0].findall("collision") else "visual"
    elements = links[0].findall(kind)
    if len(elements) != 1:
        raise ValueError(f"Expected exactly one {kind} mesh; compound assets are unsupported")
    element = elements[0]
    geometry = element.find("geometry")
    if geometry is None or len(geometry) != 1 or geometry[0].tag != "mesh":
        raise ValueError("Expected one mesh geometry; boxes and other primitives are unsupported")
    mesh_element = geometry[0]
    filename = mesh_element.get("filename", "")
    if not filename or "://" in filename:
        raise ValueError("Mesh filename must be a local absolute or URDF-relative path")
    mesh_path = (urdf_path.parent / filename).resolve()
    mesh_hash = digest(mesh_path)
    scale = numbers(mesh_element.get("scale"), "1 1 1")
    if np.any(scale <= 0):
        raise ValueError("Mesh scale must be strictly positive")
    origin = element.find("origin")
    xyz = numbers(None if origin is None else origin.get("xyz"), "0 0 0")
    rpy = numbers(None if origin is None else origin.get("rpy"), "0 0 0")
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError("Expected a nonempty triangular mesh")
    mesh.vertices = (np.asarray(mesh.vertices) * scale) @ Rotation.from_euler("xyz", rpy).as_matrix().T + xyz
    if not np.isfinite(mesh.vertices).all() or not np.isfinite(mesh.area) or mesh.area <= 0:
        raise ValueError("Object sampling mesh is nonfinite or has zero surface area")
    dominant = np.argmax(np.abs(mesh.face_normals), axis=1)
    axis_labels = [
        ("+" if normal[axis] >= 0 else "-") + "xyz"[axis]
        for normal, axis in zip(mesh.face_normals, dominant)
    ]
    metadata = {
        "version": "single_link_single_mesh_urdf_v1",
        "urdf_path": urdf_path,
        "urdf_sha256": digest(urdf_path),
        "link_name": links[0].get("name"),
        "geometry_source": kind,
        "mesh_path": mesh_path,
        "mesh_sha256": mesh_hash,
        "scale": scale,
        "origin_xyz_m": xyz,
        "origin_rpy_rad": rpy,
        "coordinate_frame": "URDF link origin, metres; inertial/COM origin is not applied",
        "sampling_surface_area_m2": float(mesh.area),
        "vertices_faces_sha256": hashlib.sha256(mesh.vertices.tobytes() + mesh.faces.tobytes()).hexdigest(),
        "components": [{"name": element.get("name", kind)}],
        "face_component_ids": [0] * len(mesh.faces),
        "face_axis_labels": axis_labels,
        "face_axis_semantics": "dominant face-normal direction, not planar face/contact labels",
        "face_count": len(mesh.faces),
    }
    return mesh, metadata


def fk_landmarks(robot_qpos: np.ndarray, xml_path: Path, specs: list[dict]) -> np.ndarray:
    """Call mj_kinematics only; no dynamics or stepping functions are used."""
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(1, model.njnt)]
    if model.nq != 36 or names != list(EXPECTED_ROBOT_JOINT_NAMES):
        raise ValueError("G1 XML must match the paired-reference 29-DoF joint order")
    bodies = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "world" for b in range(model.nbody)]
    data = mujoco.MjData(model)
    positions = np.empty(robot_qpos.shape[:2] + (model.nbody, 3))
    quaternions = np.empty(robot_qpos.shape[:2] + (model.nbody, 4))
    for frame in range(len(robot_qpos)):
        for agent in range(2):
            data.qpos[:] = robot_qpos[frame, agent]
            mujoco.mj_kinematics(model, data)
            positions[frame, agent] = data.xpos
            quaternions[frame, agent] = data.xquat
    return landmarks_world(positions, quaternions, bodies, specs)


def world_object_points(points: np.ndarray, object_qpos: np.ndarray) -> np.ndarray:
    rotation = Rotation.from_quat(object_qpos[:, [4, 5, 6, 3]]).as_matrix()
    return np.einsum("tij,nj->tni", rotation, points) + object_qpos[:, None, :3]


def stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    return {"min": float(values.min()), "mean": float(values.mean()),
            "p50": float(np.median(values)), "p95": float(np.quantile(values, 0.95)),
            "max": float(values.max())}


def error_summary(error: np.ndarray) -> dict:
    return {
        "mse_m2": stats(error),
        "rms_m": stats(np.sqrt(np.maximum(error, 0))),
        "pooled_rms_m": float(np.sqrt(error.mean())),
        "mean_mse_m2_per_agent": error.mean(axis=0),
        "pooled_rms_m_per_agent": np.sqrt(error.mean(axis=0)),
        "team_mean_mse_m2": stats(error.mean(axis=1)),
    }


def hand_weights(weights: np.ndarray, specs: list[dict]) -> np.ndarray:
    return np.stack([
        weights[..., [i for i, item in enumerate(specs) if item["group"] == f"{side}_hand"]].sum(axis=-1)
        for side in ("left", "right")
    ], axis=-1)


def hand_weight_summary(weights: np.ndarray) -> dict:
    return {f"agent{agent + 1}_{side}": stats(weights[:, agent, hand])
            for agent in range(2) for hand, side in enumerate(("left", "right"))}


def score(actual, reference, actual_objects, reference_objects, prior, distance_floor_m):
    error, diagnostics = weighted_body_object_vector_error(
        torch.from_numpy(np.ascontiguousarray(actual)),
        torch.from_numpy(np.ascontiguousarray(reference)),
        torch.from_numpy(np.ascontiguousarray(actual_objects[:, None])),
        torch.from_numpy(np.ascontiguousarray(reference_objects[:, None])),
        body_point_weights=torch.from_numpy(prior), distance_floor_m=distance_floor_m,
    )
    result = {key: value.numpy() for key, value in diagnostics.items() if key not in ("actual_weights", "reference_weights")}
    result["actual_point_weight"] = diagnostics["actual_weights"].sum(dim=-1).numpy()
    result["reference_point_weight"] = diagnostics["reference_weights"].sum(dim=-1).numpy()
    if not torch.isfinite(error).all() or not all(np.isfinite(value).all() for value in result.values()):
        raise ValueError("Nonfinite vector error or diagnostics")
    return error.numpy(), result


def synthetic_states(reference: np.ndarray, objects: np.ndarray, specs: list[dict]):
    """Fixed point edits, explicitly not joint-feasible motions/contact labels."""
    groups = np.asarray([item["group"] for item in specs])
    hands = {side: np.flatnonzero(groups == f"{side}_hand") for side in ("left", "right")}
    for side, indices in hands.items():
        if len(indices) != 3:
            raise ValueError(f"Expected three fixed {side} hand landmarks")
    for distance in (0.02, 0.05, 0.10):
        for axis, axis_name in enumerate("xyz"):
            for agent in range(2):
                for side in ("left", "right", "both"):
                    indices = np.r_[hands["left"], hands["right"]] if side == "both" else hands[side]
                    actual = reference.copy()
                    actual[:, agent, indices, axis] += distance
                    yield f"agent{agent + 1}_{side}_hand_{axis_name}_{int(distance * 100)}cm", actual, {
                        "kind": "synthetic_hand_translation", "agent": agent + 1,
                        "hand": side, "world_axis": axis_name, "distance_m": distance,
                    }
        # Both hands on BOTH robots: a labelled diagnostic scale probe.
        actual = reference.copy()
        actual[:, :, hands["left"], 0] += distance
        actual[:, :, hands["right"], 1] += distance
        yield f"both_agents_both_hands_xy_{int(distance * 100)}cm", actual, {
            "kind": "synthetic_hand_translation", "agents": [1, 2],
            "left_world_axis": "x", "right_world_axis": "y", "distance_m": distance,
        }
    for agent in range(2):
        for side, indices in hands.items():
            origin = next(i for i in indices if specs[i]["kind"] == "omni_body_origin")
            for axis in "xyz":
                rotation = Rotation.from_euler(axis, 30, degrees=True).as_matrix()
                actual = reference.copy()
                center = reference[:, agent, origin, :][:, None, :]
                actual[:, agent, indices, :] = (reference[:, agent, indices, :] - center) @ rotation.T + center
                yield f"agent{agent + 1}_{side}_hand_about_wrist_{axis}_30deg", actual, {
                    "kind": "synthetic_rigid_hand_rotation_about_wrist_origin",
                    "agent": agent + 1, "hand": side, "world_axis": axis, "angle_deg": 30,
                }
    nonhand = np.flatnonzero(~np.isin(groups, ("left_hand", "right_hand")))
    distances = np.linalg.norm(reference[:, :, nonhand, None, :] - objects[:, None, None, :, :], axis=-1)
    closest = nonhand[distances.min(axis=-1).argmin(axis=-1)]
    for agent in range(2):
        for axis, axis_name in enumerate("xyz"):
            actual = reference.copy()
            actual[np.arange(len(reference)), agent, closest[:, agent], axis] += 0.05
            yield f"agent{agent + 1}_nearest_reference_nonhand_{axis_name}_5cm", actual, {
                "kind": "synthetic_nonhand_translation", "agent": agent + 1,
                "world_axis": axis_name, "distance_m": 0.05,
                "selection": "closest nonhand landmark to fixed sampled object points, reference only, each frame",
                "selected_landmark_indices_per_frame": closest[:, agent],
            }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--robot-xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--object-urdf", type=Path)
    parser.add_argument("--point-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--distance-floor-m", type=float, default=0.1,
                        help="Inverse-distance weight cap length, NOT a contact threshold")
    args = parser.parse_args(argv)
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        parser.error("Output directory exists; choose a new directory (no overwrite)")
    if args.point_count <= 0:
        parser.error("--point-count must be positive")
    if not np.isfinite(args.distance_floor_m) or args.distance_floor_m <= 0:
        parser.error("--distance-floor-m must be finite and positive")
    input_path, xml_path = args.input.expanduser().resolve(), args.robot_xml.expanduser().resolve()
    pair = load_pair_reference(input_path)
    urdf_path = resolve_object_urdf(input_path, pair.object_name, args.object_urdf)
    input_paths = {"paired_reference": input_path, "robot_xml": xml_path, "object_urdf": urdf_path,
                   "pure_vector_module": VECTOR_MODULE}
    input_hashes = {name: digest(path) for name, path in input_paths.items()}
    specs, robot_metadata = build_robot_landmarks(xml_path)
    mesh, object_metadata = load_single_mesh(urdf_path)
    input_paths["object_mesh"] = object_metadata["mesh_path"]
    input_hashes["object_mesh"] = object_metadata["mesh_sha256"]
    points, sampling = sample_object_points(mesh, object_metadata, args.point_count, args.seed)
    if not len(points):
        raise ValueError("Surface sampler returned no points")
    reference = fk_landmarks(pair.robot_qpos, xml_path, specs)
    objects = world_object_points(points, pair.object_qpos)
    # Each original Omni group has a prior mass of one: three hand points share it.
    groups = [item["group"] for item in specs]
    prior = np.asarray([1.0 / groups.count(group) for group in groups], dtype=np.float64)
    torch.set_num_threads(1)

    def compare(actual, actual_objects=objects, reference_body=reference, reference_objects=objects):
        return score(actual, reference_body, actual_objects, reference_objects, prior, args.distance_floor_m)

    identity, baseline = compare(reference)
    if np.max(np.abs(identity)) > 1e-20:
        raise ValueError("Identity/self-reference check failed")
    ref_hand_mass = hand_weights(baseline["reference_point_weight"], specs)
    arrays = {
        "object_points_local_m": points, "object_points_world_m": objects,
        "reference_landmarks_world_m": reference, "object_qpos": pair.object_qpos,
        "robot_qpos": pair.robot_qpos, "fps": np.asarray(pair.fps),
        "landmark_names": np.asarray([item["name"] for item in specs]),
        "landmark_groups": np.asarray(groups), "body_point_prior": prior,
        "reference_point_weights": baseline["reference_point_weight"],
        "reference_hand_weights": ref_hand_mass, "identity_error_m2": identity,
    }
    scenarios = {}
    invariance_error = 0.0
    scale_probe_error = None
    for name, actual, description in synthetic_states(reference, objects, specs):
        error, diagnostics = compare(actual)
        actual_hand_mass = hand_weights(diagnostics["actual_point_weight"], specs)
        scenarios[name] = {**description, **error_summary(error),
                           "actual_hand_weight": hand_weight_summary(actual_hand_mass)}
        arrays[f"error_m2__{name}"] = error
        arrays[f"actual_hand_weights__{name}"] = actual_hand_mass
        if name == "both_agents_both_hands_xy_5cm":
            scale_probe_error = error
            rotation = Rotation.from_euler("xyz", [0.21, -0.37, 0.72]).as_matrix()
            translation = np.asarray([1.7, -2.2, 0.4])
            transformed, _ = compare(
                actual @ rotation.T + translation, objects @ rotation.T + translation,
                reference @ rotation.T + translation, objects @ rotation.T + translation,
            )
            invariance_error = float(np.max(np.abs(error - transformed)))
            if invariance_error > 1e-12:
                raise ValueError("Joint actual/reference rigid-transform invariance failed")
    # Additional direction/correspondence probes: the reference stays fixed.
    origin = pair.object_qpos[:, None, :3]
    for angle in (10, 30):
        rotation = Rotation.from_euler("z", angle, degrees=True).as_matrix()
        turned_objects = (objects - origin) @ rotation.T + origin
        for kind in ("object_only", "body_and_object"):
            actual = reference if kind == "object_only" else (
                (reference - origin[:, None]) @ rotation.T + origin[:, None]
            )
            error, _ = compare(actual, turned_objects)
            name = f"{kind}_world_z_rotation_{angle}deg_reference_fixed"
            scenarios[name] = {"kind": "synthetic_rotation_with_fixed_reference",
                               "rotated": kind, "angle_deg": angle, **error_summary(error)}
            arrays[f"error_m2__{name}"] = error
    if scale_probe_error is None:
        raise ValueError("Missing diagnostic scale probe")
    team_probe_error = scale_probe_error.mean(axis=1)
    sigma_per_frame = np.sqrt(team_probe_error / np.log(2.0))
    # Chronological bins are not inferred/annotated contact or release phases.
    bins = {name: np.asarray(indices) for name, indices in zip(
        ("first_third", "middle_third", "last_third"), np.array_split(np.arange(len(reference)), 3)
    ) if len(indices)}
    report = {
        "scope": "offline_synthetic_geometry_diagnostic_only_no_physics_no_training",
        "input_paths": input_paths, "input_sha256": input_hashes,
        "object_name": pair.object_name, "frames": len(reference), "fps": pair.fps,
        "quaternion_order": "wxyz", "robot_geometry": robot_metadata, "landmarks": specs,
        "object_geometry": {key: value for key, value in object_metadata.items()
                            if key not in ("face_component_ids", "face_axis_labels")},
        "object_sampling": sampling,
        "metric": {
            "space": "world-coordinate body-to-object vectors; fixed corresponding object-local samples",
            "edge_error": "squared Euclidean norm BEFORE weighted summation; not scalar distances or Laplacian",
            "distance_floor_m": args.distance_floor_m,
            "distance_floor_semantics": "smooth salience weight cap, not a contact threshold",
            "weights": "mean of separately normalized actual/reference inverse-squared-distance weights",
            "normalization": "each frame and agent independently; team error is mean of two agents",
            "point_prior": prior,
            "group_sampling": "each three-point hand shares prior one, like each singleton Omni group",
            "actual_weight_limit": "reference branch preserves half its weighted error; actual weights remain state-dependent",
        },
        "reference_hand_weight": hand_weight_summary(ref_hand_mass),
        "reference_point_weight_mean_per_agent": baseline["reference_point_weight"].mean(axis=0),
        "chronological_thirds_not_contact_phases": {
            name: {"first_frame": int(indices[0]), "last_frame": int(indices[-1]),
                   "reference_hand_weight": hand_weight_summary(ref_hand_mass[indices]),
                   "identity_max_error_m2": float(identity[indices].max())}
            for name, indices in bins.items()
        },
        "checks": {
            "identity_max_error_m2": float(identity.max()),
            "joint_actual_and_reference_rigid_transform_max_error_m2": invariance_error,
            "release": "Self-reference is zero at EVERY frame, including any normal release; no release/contact labels inferred",
            "fk": "G1 XML mj_kinematics only; paired qpos order checked; no physics stepping",
        },
        "synthetic_scenarios": scenarios,
        "diagnostic_scale_only_not_training_selection": {
            "probe": "both_agents_both_hands_xy_5cm",
            "convention": "r=exp(-mean_agent(E_m2)/sigma_m^2)",
            "meaning": "Each candidate sigma makes THAT frame's team probe reward 0.5; no training sigma selected",
            "candidate_sigma_m_per_frame": stats(sigma_per_frame),
            "candidate_sigma_m_p05_p95": np.quantile(sigma_per_frame, [0.05, 0.95]),
            "candidate_from_mean_error_m": float(np.sqrt(team_probe_error.mean() / np.log(2.0))),
            "mean_error_candidate_caveat": "Makes exp(-mean(E)/sigma^2)=0.5, not mean(exp(-E/sigma^2))=0.5",
        },
        "limitations": [
            "All actual trajectories are reference or synthetic point edits; no measured policy rollout is evaluated",
            "Point edits need not be reachable, collision-free, dynamically feasible or contact-supporting",
            "Normals, forces, grasp support and prevention of nonhand pushing are not measured or guaranteed",
            "Fixed sample identities can penalize rotations of a symmetric object even when its surface occupancy is unchanged",
            "Common rotation of actual body+object alone is not invariant with a fixed world-oriented reference",
            "Hand vertices are geometric landmarks, not palm/contact labels",
            "Normal release has zero error when actual matches reference; normalized salience remains allocated per agent",
        ],
    }
    if {name: digest(path) for name, path in input_paths.items()} != input_hashes:
        raise ValueError("An input changed during the audit")
    report["inputs_unchanged"] = True
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "audit_arrays.npz", **arrays)
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8",
    )
    (output / "README.md").write_text(
        "# Offline interaction-vector audit\n\n"
        f"Input: `{input_path}` ({len(reference)} frames, {pair.fps} Hz, {pair.object_name}).\n\n"
        "G1 forward kinematics only. No simulation, training, retargeting or reward integration. "
        "All actual points equal reference except the labelled synthetic perturbations; these are not executable motions.\n\n"
        f"{len(specs)} robot landmarks per agent, {len(points)} fixed corresponding object samples "
        f"(requested {args.point_count}, seed {args.seed}). Three hand points share one sampling prior. "
        "The mesh is the actual URDF mesh with its scale/origin, not a box/table proxy.\n\n"
        "`summary.json` reports per-agent m² errors, RMS in metres, reference/actual hand-weight budgets, "
        "2/5/10 cm hand offsets along world axes, wrist rotations, and reference-selected nearest nonhand offsets. "
        "`audit_arrays.npz` stores point identities, poses, weights and scenario error curves. "
        "Object samples are never resampled separately for actual/reference.\n\n"
        f"Identity error: {float(identity.max()):.3g} m²; "
        f"common actual/reference rigid-transform discrepancy: {invariance_error:.3g} m². "
        "Self-reference stays zero through normal release; chronological thirds are not contact annotations.\n\n"
        "The 10 cm default distance floor caps proximity weighting; it is not a contact threshold. "
        "Weights include both actual and reference branches, so actual-state dependence remains. "
        "The reported 5 cm half-reward scales are diagnostic candidates only; no training sigma is chosen.\n\n"
        "This does not prove grasping, carrying, load sharing or preventing nonhand pushing. "
        "Fixed world vectors retain orientation constraints and object-symmetry correspondence ambiguities.\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output), "frames": len(reference), "object_points": len(points),
                      "synthetic_scenarios": len(scenarios), "identity_max_m2": float(identity.max()),
                      "rigid_transform_max_m2": invariance_error}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
