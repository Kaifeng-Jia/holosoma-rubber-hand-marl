"""Compile frozen Omni graph scores for training without altering references.

Offline only: runtime needs only fixed body offsets, reference points and Q.
The object frame is its URDF link origin, never its center of mass. Graph edges
are geometric relations, not contact or force labels.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from holosoma_retargeting.interaction_mesh_audit import (
    frozen_mesh, mesh_errors, points_in_object_frame,
)
from holosoma_retargeting.interaction_mesh_geometry import (
    build_robot_landmarks, landmarks_world, load_box_object_mesh, sample_object_points,
)


FORMAT_VERSION = "core4d_smalltable_interaction_mesh_v1"


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def training_body_names(config_path: str | Path) -> list[str]:
    """Read the existing G1 body contract without importing Isaac packages."""
    tree = ast.parse(Path(config_path).read_text())
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "g1_29dof"
            for target in statement.targets
        ):
            if isinstance(statement.value, ast.Call):
                for keyword in statement.value.keywords:
                    if keyword.arg == "body_names":
                        result = ast.literal_eval(keyword.value)
                        if len(result) != 32 or len(set(result)) != 32:
                            raise ValueError("Expected the existing 32-body G1 contract")
                        return result
    raise ValueError("Could not find g1_29dof.body_names")


def _origin(element: ET.Element | None) -> np.ndarray:
    transform = np.eye(4)
    if element is None:
        return transform
    xyz = np.fromstring(element.get("xyz", "0 0 0"), sep=" ")
    rpy = np.fromstring(element.get("rpy", "0 0 0"), sep=" ")
    if xyz.shape != (3,) or rpy.shape != (3,) or not np.isfinite(np.r_[xyz, rpy]).all():
        raise ValueError("Invalid URDF fixed transform")
    transform[:3, 3] = xyz
    transform[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    return transform


def urdf_body_transforms(urdf_path: str | Path, joints: dict[str, float]) -> dict[str, np.ndarray]:
    """Link-frame FK for validation, deliberately independent of MuJoCo FK."""
    root = ET.parse(urdf_path).getroot()
    children = {item.find("child").get("link") for item in root.findall("joint")}
    bases = {item.get("name") for item in root.findall("link")} - children
    if len(bases) != 1:
        raise ValueError("Expected a single URDF root")
    result = {bases.pop(): np.eye(4)}
    remaining = list(root.findall("joint"))
    while remaining:
        progressed = False
        for item in remaining[:]:
            parent, child = item.find("parent").get("link"), item.find("child").get("link")
            if parent not in result:
                continue
            transform = _origin(item.find("origin"))
            kind = item.get("type")
            if kind in ("revolute", "continuous"):
                axis_element = item.find("axis")
                axis = np.fromstring("1 0 0" if axis_element is None else axis_element.get("xyz"), sep=" ")
                axis /= np.linalg.norm(axis)
                motion = np.eye(4)
                motion[:3, :3] = Rotation.from_rotvec(axis * joints.get(item.get("name"), 0.0)).as_matrix()
                transform = transform @ motion
            elif kind != "fixed":
                raise ValueError(f"Unsupported joint type for validation: {kind}")
            result[child] = result[parent] @ transform
            remaining.remove(item)
            progressed = True
        if not progressed:
            raise ValueError("Disconnected/cyclic URDF")
    return result


def map_landmarks_to_training(
    specs: list[dict], xml_path: str | Path, urdf_path: str | Path, available: list[str],
) -> tuple[list[dict], dict]:
    """Move each point through fixed chains only, never across a moving joint.

    Existing training links (including rubber hands) use the training URDF.
    Reference-only virtual foot helpers use the reference XML fixed offsets.
    No main-link origin is substituted for an unavailable auxiliary point.
    """
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    urdf = ET.parse(urdf_path).getroot()
    links = {item.get("name") for item in urdf.findall("link")}
    parent_joint = {item.find("child").get("link"): item for item in urdf.findall("joint")}
    available_set = set(available)
    if not available_set <= links:
        raise ValueError("Online body contract contains a link absent from the training URDF")
    mapped, details = [], []
    for spec in specs:
        body = spec["body"]
        point = np.asarray(spec["local_xyz"], dtype=float)
        chain = []
        source = "training_urdf" if body in links else "reference_xml_virtual_fixed_point"
        while body not in available_set:
            if source == "training_urdf":
                joint = parent_joint.get(body)
                if joint is None or joint.get("type") != "fixed":
                    raise ValueError(f"Cannot map {spec['name']} through moving/unknown training joint at {body}")
                transform = _origin(joint.find("origin"))
                parent = joint.find("parent").get("link")
            else:
                body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
                if body_id <= 0 or model.body_jntnum[body_id]:
                    raise ValueError(f"Cannot map reference virtual point through moving joint: {body}")
                transform = np.eye(4)
                transform[:3, :3] = Rotation.from_quat(model.body_quat[body_id][[1, 2, 3, 0]]).as_matrix()
                transform[:3, 3] = model.body_pos[body_id]
                parent = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.body_parentid[body_id]))
            point = transform[:3, :3] @ point + transform[:3, 3]
            chain.append({"child": body, "parent": parent, "fixed_transform": transform.tolist()})
            body = parent
        mapped.append({**spec, "body": body, "local_xyz": point.tolist()})
        details.append({"name": spec["name"], "source_body": spec["body"], "online_body": body,
                        "source": source, "fixed_chain": chain, "online_offset": point.tolist()})
    return mapped, {"available_bodies": available, "points": details}


def grouped_q_matrix(laplacian: np.ndarray, adjacency: np.ndarray, groups: list[str]) -> np.ndarray:
    """Exactly fold the audit's grouped Laplacian rows into Q = K.T W K."""
    count = len(groups)
    labels = np.asarray(groups)
    unique = list(dict.fromkeys(groups))
    if laplacian.shape != adjacency.shape or laplacian.shape[0] <= count:
        raise ValueError("Laplacian and graph dimensions do not match body groups")
    weights = np.zeros(len(laplacian), dtype=float)
    for name in unique:
        indices = np.flatnonzero(labels == name)
        weights[indices] = 0.5 / (len(unique) * len(indices))
    active = np.flatnonzero(np.any(adjacency[count:, :count], axis=1)) + count
    if len(active):
        weights[active] = 0.5 / len(active)
    columns = laplacian[:, :count]
    return columns.T @ (weights[:, None] * columns)


def score_q(delta: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return np.einsum("...ic,...ij,...jc->...", delta, matrix, delta)


def compile_training_reference(
    reference_path: str | Path, object_urdf: str | Path, reference_xml: str | Path,
    training_urdf: str | Path, robot_config_path: str | Path,
) -> tuple[dict[str, np.ndarray], dict]:
    """Compile only from source assets; no checkpoint or replay is required."""
    paths = dict(runtime_reference=Path(reference_path), object_urdf=Path(object_urdf),
                 reference_robot_xml=Path(reference_xml), training_robot_urdf=Path(training_urdf),
                 robot_config=Path(robot_config_path))
    hashes = {name: sha256(path) for name, path in paths.items()}
    with np.load(reference_path, allow_pickle=False) as archive:
        ref = {key: archive[key].copy() for key in archive.files}
    fps = float(ref["fps"].item())
    names = ref["body_names"].tolist()
    specs, landmark_metadata = build_robot_landmarks(reference_xml)
    available = training_body_names(robot_config_path)
    mapped, mapping_metadata = map_landmarks_to_training(specs, reference_xml, training_urdf, available)
    world = landmarks_world(ref["agent_body_pos_w"], ref["agent_body_quat_w"], names, specs)
    object_pose = np.concatenate((ref["object_pos_w"], ref["object_quat_w"]), axis=-1)
    local = points_in_object_frame(world, object_pose)
    if local.shape != (687, 2, 19, 3) or fps != 50:
        raise ValueError(f"This v1 artifact expects the reviewed 687-frame, 50 Hz reference, got {local.shape}, {fps}")
    mapped_world = landmarks_world(ref["agent_body_pos_w"], ref["agent_body_quat_w"], names, mapped)
    mapping_error = float(np.max(np.linalg.norm(mapped_world - world, axis=-1)))
    if mapping_error > 1e-6:
        raise ValueError(f"Fixed-chain mapping changed the reviewed reference points by {mapping_error} metres")
    mesh, object_metadata = load_box_object_mesh(object_urdf)
    object_points, sampling = sample_object_points(mesh, object_metadata, 100, seed=42)
    if len(object_points) != 85:
        raise ValueError(f"Reviewed sampler returned 85 points; current sampler returned {len(object_points)}. Review before changing contract.")
    groups = [spec["group"] for spec in specs]
    matrices = np.empty(local.shape[:2] + (19, 19), dtype=np.float32)
    max_equivalence, min_eigenvalue = 0.0, float("inf")
    validation_frames = set(np.linspace(0, len(local) - 1, 21, dtype=int).tolist())
    cases_checked = 0
    rng = np.random.default_rng(721)
    for t in range(len(local)):
        for agent in range(2):
            points = local[t, agent]
            laplacian, adjacency = frozen_mesh(points, object_points)
            matrix = grouped_q_matrix(laplacian, adjacency, groups)
            matrices[t, agent] = matrix
            if t not in validation_frames:
                continue
            min_eigenvalue = min(min_eigenvalue, float(np.linalg.eigvalsh(matrix).min()))
            changes = [np.zeros_like(points), np.tile([0.03, 0, 0], (19, 1)), rng.normal(0, 0.05, points.shape)]
            hand = np.array([group == "left_hand" for group in groups])
            origin = points[[spec["name"] for spec in specs].index("L_Wrist")]
            rotated = points.copy()
            rotated[hand] = (points[hand] - origin) @ Rotation.from_euler("z", 90, degrees=True).as_matrix().T + origin
            changes.append(rotated - points)
            for delta in changes:
                original = mesh_errors(laplacian, adjacency, points, points + delta, object_points, groups)["grouped_mse_m2"]
                folded = float(score_q(delta, matrices[t, agent]))
                max_equivalence = max(max_equivalence, abs(original - folded))
                cases_checked += 1
    if max_equivalence > 1e-7 or min_eigenvalue < -1e-10:
        raise ValueError(f"Folded score validation failed: {max_equivalence=}, {min_eigenvalue=}")
    metadata = {
        "version": FORMAT_VERSION,
        **{name + "_sha256": value for name, value in hashes.items()},
        "object_points_sha256": hashlib.sha256(object_points.tobytes()).hexdigest(),
        "num_frames": len(local), "fps": fps, "num_agents": 2, "num_body_points": 19,
        "requested_object_points": 100, "actual_object_points": len(object_points), "seed": 42,
        "grouped_formula": "0.5 * mean_over_15_body_groups(mean_rows_squared_L_delta) + 0.5 * mean_over_reference_object_rows_adjacent_to_body(squared_L_delta)",
        "q_definition": "K=L[:, :19]; Q=K.T@diag(grouped_row_weights)@K; E=sum_xyz(delta.T@Q@delta)",
        "reference_definition": "Original runtime body poses and reviewed XML landmarks, unmodified; not training-URDF FK",
        "coordinate_frame": "respective object URDF link origin (not COM), metres",
        "point_groups": groups,
        "point_mapping": mapping_metadata,
        "robot_landmarks": landmark_metadata,
        "sampling": sampling,
        "semantic_limit": "Geometry only, not contact, force, load sharing or hand-only behavior",
    }
    arrays = {
        "reference_points_object": local.astype(np.float32),
        "q_matrices": matrices,
        "point_body_names": np.asarray([spec["body"] for spec in mapped]),
        "point_offsets": np.asarray([spec["local_xyz"] for spec in mapped], dtype=np.float32),
        "object_points": object_points,
        "point_names": np.asarray([spec["name"] for spec in specs]),
        "fps": np.asarray(fps),
        "metadata_json": np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True)),
    }
    after = {name: sha256(path) for name, path in paths.items()}
    if hashes != after:
        raise RuntimeError("An input source changed while compiling; no output should be written")
    diagnostics = {
        "metadata": metadata, "inputs_unchanged": True,
        "fixed_chain_reference_equivalence_max_m": mapping_error,
        "q_vs_full_laplacian_max_abs_m2": max_equivalence,
        "q_min_eigenvalue_float64": min_eigenvalue,
        "checked_frames": len(validation_frames), "checked_synthetic_cases": cases_checked,
        "q_float32_bytes": matrices.nbytes,
        "reference_float32_rounding_max_m": float(np.max(np.linalg.norm(local - arrays["reference_points_object"], axis=-1))),
    }
    return arrays, diagnostics


def validate_replay(artifact: dict[str, np.ndarray], reference_path: Path, replay_path: Path, xml_path: Path) -> dict:
    """Optional historical-replay comparison; does not affect compiled arrays."""
    from holosoma_retargeting.interaction_mesh_audit import load_pair_states

    specs, _ = build_robot_landmarks(xml_path)
    states = load_pair_states(reference_path, replay_path, xml_path, specs)
    error = states["actual_local"] - artifact["reference_points_object"]
    q_score = score_q(error, artifact["q_matrices"])
    differences = []
    for t in np.linspace(0, len(error) - 1, 21, dtype=int):
        for agent in range(2):
            ref = states["reference_local"][t, agent]
            laplacian, adjacency = frozen_mesh(ref, artifact["object_points"])
            old = mesh_errors(laplacian, adjacency, ref, states["actual_local"][t, agent],
                              artifact["object_points"], [spec["group"] for spec in specs])["grouped_mse_m2"]
            differences.append(abs(old - q_score[t, agent]))
    maximum = float(max(differences))
    if maximum > 1e-7:
        raise ValueError(f"Replay folded score differs from audit by {maximum} m^2")
    return {"path": str(replay_path.resolve()), "sha256": sha256(replay_path), "frames": len(error),
            "max_abs_score_difference_m2": maximum, "per_agent_mean_mse_m2": q_score.mean(axis=0).tolist(),
            "pair_mean_reward_sigma006": float(np.exp(-q_score.mean(axis=1) / 0.06**2).mean()),
            "note": "Recorded replay FK comparison only; no policy or new physics rollout"}
