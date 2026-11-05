"""Offline Omni interaction-mesh diagnostics; never used by training implicitly.

Reference and simulated nodes are expressed in their respective object frames.
Every simulated mesh uses the corresponding frozen reference topology. Graph
edges describe geometry, NOT contact labels or measured contributions to force.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from holosoma_retargeting.dual_pull_reference import G1_29DOF_JOINT_NAMES
from holosoma_retargeting.interaction_mesh_geometry import landmarks_world
from holosoma_retargeting.src.utils import (
    calculate_laplacian_matrix,
    create_interaction_mesh,
    get_adjacency_list,
)


def rotation_matrices(quaternions_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternions_wxyz, dtype=float)
    if q.shape[-1] != 4 or not np.isfinite(q).all():
        raise ValueError("Expected finite WXYZ quaternions")
    if not np.allclose(np.linalg.norm(q, axis=-1), 1.0, atol=1e-5, rtol=0):
        raise ValueError("Non-unit quaternion")
    return Rotation.from_quat(q.reshape(-1, 4)[:, [1, 2, 3, 0]]).as_matrix().reshape(
        q.shape[:-1] + (3, 3)
    )


def points_in_object_frame(points_w: np.ndarray, object_qpos: np.ndarray) -> np.ndarray:
    """Transform [T,A,N,3] points using [T,7] object-origin poses, not COM."""
    points = np.asarray(points_w, dtype=float)
    pose = np.asarray(object_qpos, dtype=float)
    if points.ndim != 4 or pose.shape != (len(points), 7):
        raise ValueError("Expected points [T,A,N,3] and object poses [T,7]")
    return np.einsum(
        "tanj,tjk->tank", points - pose[:, None, None, :3], rotation_matrices(pose[:, 3:])
    )


def frozen_mesh(reference_body: np.ndarray, object_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Use Omni's Delaunay adjacency and default uniform-neighbor Laplacian."""
    vertices, tetrahedra = create_interaction_mesh(np.vstack((reference_body, object_points)))
    adjacency_list = get_adjacency_list(tetrahedra, len(vertices))
    adjacency = np.zeros((len(vertices), len(vertices)), dtype=bool)
    for index, neighbors in enumerate(adjacency_list):
        adjacency[index, neighbors] = True
    if np.any(adjacency.sum(axis=1) == 0):
        raise ValueError("Reference Delaunay graph omitted an isolated/duplicate node")
    return calculate_laplacian_matrix(vertices, adjacency_list), adjacency


def mesh_errors(
    laplacian: np.ndarray,
    adjacency: np.ndarray,
    reference_body: np.ndarray,
    actual_body: np.ndarray,
    object_points: np.ndarray,
    groups: list[str],
) -> dict[str, np.ndarray | float]:
    """Record original sum and explicit proposed grouped normalization.

    Three points on each rigid hand share ONE body-group weight. Object rows
    adjacent to a body are retained; object-only zero rows are excluded from
    the object-group mean. The 50/50 mix is a diagnostic convention, not an
    approved reward hyperparameter. No actual-state proximity changes weights.
    """
    count = len(reference_body)
    if len(groups) != count or actual_body.shape != reference_body.shape:
        raise ValueError("Body point/group mismatch")
    delta = np.vstack((actual_body - reference_body, np.zeros_like(object_points)))
    residual = laplacian @ delta
    squared = np.sum(residual**2, axis=1)
    labels = np.asarray(groups)
    body_mean = float(np.mean([squared[:count][labels == name].mean() for name in dict.fromkeys(groups)]))
    active_object = np.any(adjacency[count:, :count], axis=1)
    object_mean = float(squared[count:][active_object].mean()) if active_object.any() else 0.0
    return {
        "node_residual_m": np.sqrt(squared),
        "body_group_mse_m2": body_mean,
        "active_object_mse_m2": object_mean,
        "grouped_mse_m2": 0.5 * (body_mean + object_mean),
        "omni_sum_m2": float(squared.sum()),
        "sum_per_body_node_m2": float(squared.sum() / count),
        "all_node_mean_m2": float(squared.mean()),
        "active_object_count": int(active_object.sum()),
    }


def load_pair_states(reference_path: Path, rollout_path: Path, xml_path: Path, specs: list[dict]) -> dict:
    """Recover recorded actual keypoints via FK; no policy or physics rollout."""
    with np.load(reference_path, allow_pickle=False) as archive:
        reference = {key: archive[key].copy() for key in archive.files}
    with np.load(rollout_path, allow_pickle=False) as archive:
        actual = {key: archive[key].copy() for key in archive.files}
    frames = len(reference["agent_joint_pos"])
    metadata = json.loads(str(actual["_metadata_json"].item()))
    fps = int(reference["fps"].item())
    if len(actual["dof_pos"]) != frames or metadata["fps"] != fps:
        raise ValueError("Reference/replay must have identical length and FPS; no silent alignment")
    if not metadata.get("result", {}).get("completed_reference", False):
        raise ValueError("This audit expects a complete from-frame-zero replay")
    joint_names = reference["joint_names"].tolist()
    if joint_names != list(G1_29DOF_JOINT_NAMES):
        raise ValueError("Unexpected reference joint order; actual replay uses canonical G1 order")
    names = reference["body_names"].tolist()
    pelvis = names.index("pelvis")
    reference_qpos = np.concatenate(
        (reference["agent_body_pos_w"][:, :, pelvis], reference["agent_body_quat_w"][:, :, pelvis],
         reference["agent_joint_pos"]), axis=-1,
    )
    actual_qpos = np.concatenate(
        (actual["root_pos"], actual["root_quat_xyzw"][..., [3, 0, 1, 2]], actual["dof_pos"]), axis=-1,
    )
    object_ref = np.concatenate((reference["object_pos_w"], reference["object_quat_w"]), axis=-1)
    object_act = np.concatenate((actual["object_pos_w"], actual["object_quat_xyzw"][:, [3, 0, 1, 2]]), axis=-1)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    if model.nq != 36:
        raise ValueError("Expected rubber-hand G1 with one floating base and 29 joints")
    model_joints = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(1, model.njnt)]
    order = [joint_names.index(name) for name in model_joints]
    model_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "world" for i in range(model.nbody)]
    data = mujoco.MjData(model)
    actual_pos = np.empty((frames, 2, model.nbody, 3))
    actual_quat = np.empty((frames, 2, model.nbody, 4))
    max_fk_error = 0.0
    max_landmark_rotation_error = 0.0
    for trajectory, check in ((reference_qpos, True), (actual_qpos, False)):
        for t in range(frames):
            for agent in range(2):
                q = trajectory[t, agent]
                data.qpos[:7] = q[:7]
                data.qpos[7:] = q[7:][order]
                mujoco.mj_kinematics(model, data)
                if check:
                    indexed = [names.index(name) for name in model_names]
                    ref_p = reference["agent_body_pos_w"][t, agent, indexed]
                    max_fk_error = max(max_fk_error, float(np.max(np.linalg.norm(data.xpos - ref_p, axis=-1))))
                    q_ref = reference["agent_body_quat_w"][t, agent, indexed]
                    max_landmark_rotation_error = max(
                        max_landmark_rotation_error,
                        float(np.max(1.0 - np.abs(np.sum(data.xquat * q_ref, axis=-1)))),
                    )
                else:
                    actual_pos[t, agent] = data.xpos
                    actual_quat[t, agent] = data.xquat
    if max_fk_error > 1e-4 or max_landmark_rotation_error > 1e-5:
        raise ValueError(f"FK/reference contract mismatch: position={max_fk_error}, rotation={max_landmark_rotation_error}")
    ref_points = landmarks_world(reference["agent_body_pos_w"], reference["agent_body_quat_w"], names, specs)
    act_points = landmarks_world(actual_pos, actual_quat, model_names, specs)
    return {
        "reference_robot_qpos": reference_qpos,
        "actual_robot_qpos": actual_qpos,
        "reference_object_qpos": object_ref,
        "actual_object_qpos": object_act,
        "reference_landmarks_w": ref_points,
        "actual_landmarks_w": act_points,
        "reference_local": points_in_object_frame(ref_points, object_ref),
        "actual_local": points_in_object_frame(act_points, object_act),
        "fps": fps,
        "metadata": metadata,
        "fk_max_position_error_m": max_fk_error,
        "fk_max_quaternion_dot_error": max_landmark_rotation_error,
    }


def analyze_budget(states: dict, points: np.ndarray, specs: list[dict]) -> tuple[dict, dict]:
    ref, actual = states["reference_local"], states["actual_local"]
    frames, agents, bodies, _ = ref.shape
    groups = [str(spec["group"]) for spec in specs]
    size = bodies + len(points)
    graphs = np.empty((frames, agents, size, size), dtype=np.uint8)
    residuals = np.empty((frames, agents, size), dtype=np.float32)
    error_keys = ("body_group_mse_m2", "active_object_mse_m2", "grouped_mse_m2", "omni_sum_m2", "all_node_mean_m2")
    curves = {key: np.empty((frames, agents)) for key in error_keys}
    active_counts = np.empty((frames, agents), dtype=int)
    cross_counts = np.empty((frames, agents), dtype=int)
    topology_changes = np.zeros((frames, agents), dtype=int)
    build_seconds, score_seconds = 0.0, 0.0
    identity_max = 0.0
    sensitivity: dict[str, list[float]] = {}
    check_frames = set(np.linspace(0, frames - 1, min(21, frames), dtype=int).tolist())
    for t in range(frames):
        for a in range(agents):
            start = time.perf_counter()
            lap, adjacency = frozen_mesh(ref[t, a], points)
            build_seconds += time.perf_counter() - start
            graphs[t, a] = adjacency
            if t:
                topology_changes[t, a] = int(np.count_nonzero(np.triu(adjacency != graphs[t-1, a], k=1)))
            cross_counts[t, a] = int(adjacency[:bodies, bodies:].sum())
            start = time.perf_counter()
            result = mesh_errors(lap, adjacency, ref[t, a], actual[t, a], points, groups)
            score_seconds += time.perf_counter() - start
            residuals[t, a] = result["node_residual_m"]
            active_counts[t, a] = result["active_object_count"]
            for key in error_keys:
                curves[key][t, a] = result[key]
            identity_max = max(identity_max, mesh_errors(lap, adjacency, ref[t, a], ref[t, a], points, groups)["grouped_mse_m2"])
            if t not in check_frames:
                continue
            for distance in (0.02, 0.05, 0.10):
                for kind in ("whole_robot", "left_hand", "left_foot"):
                    altered = ref[t, a].copy()
                    if kind == "whole_robot":
                        ids = np.arange(bodies)
                    elif kind == "left_hand":
                        ids = np.array([i for i, s in enumerate(specs) if s["group"] == "left_hand"])
                    else:
                        ids = np.array([i for i, s in enumerate(specs) if s["name"] in ("L_Ankle", "L_Foot")])
                    if not len(ids):
                        raise ValueError(f"Empty synthetic perturbation selection: {kind}")
                    altered[ids, 0] += distance
                    value = mesh_errors(lap, adjacency, ref[t, a], altered, points, groups)["grouped_mse_m2"]
                    sensitivity.setdefault(f"{kind}_shift_{distance:.2f}m", []).append(float(value))
            # Rigid hand rotation around its original Omni node leaves all 15
            # original points unchanged; only the two extra surface points move.
            ids = np.array([i for i, s in enumerate(specs) if s["group"] == "left_hand"])
            anchor = next(i for i, s in enumerate(specs) if s["name"] == "L_Wrist")
            altered = ref[t, a].copy()
            rot = Rotation.from_rotvec(np.array([0.0, 0.0, np.pi / 2])).as_matrix()
            altered[ids] = (altered[ids] - altered[anchor]) @ rot.T + altered[anchor]
            sensitivity.setdefault("left_hand_rotate_90deg_19points", []).append(float(
                mesh_errors(lap, adjacency, ref[t, a], altered, points, groups)["grouped_mse_m2"]
            ))
            lap15, adj15 = frozen_mesh(ref[t, a, :15], points)
            sensitivity.setdefault("left_hand_rotate_90deg_original15", []).append(float(
                mesh_errors(lap15, adj15, ref[t, a, :15], altered[:15], points, groups[:15])["grouped_mse_m2"]
            ))
    summary = {
        "actual_object_points": len(points),
        "identity_max_mse_m2": identity_max,
        "actual_grouped_rms_cm_per_agent": (100 * np.sqrt(curves["grouped_mse_m2"].mean(axis=0))).tolist(),
        "actual_grouped_rms_cm_pair": float(100 * np.sqrt(curves["grouped_mse_m2"].mean())),
        "actual_body_group_rms_cm_per_agent": (100 * np.sqrt(curves["body_group_mse_m2"].mean(axis=0))).tolist(),
        "actual_object_active_rms_cm_per_agent": (100 * np.sqrt(curves["active_object_mse_m2"].mean(axis=0))).tolist(),
        "cross_edges_mean_per_agent": cross_counts.mean(axis=0).tolist(),
        "active_object_rows_mean_per_agent": active_counts.mean(axis=0).tolist(),
        "topology_changed_edges_mean_per_step": topology_changes[1:].mean(axis=0).tolist(),
        "topology_changed_edges_max_per_step": topology_changes.max(axis=0).tolist(),
        "cpu_graph_build_ms_per_agent_frame": build_seconds * 1000 / (frames * agents),
        "cpu_score_ms_per_agent_frame": score_seconds * 1000 / (frames * agents),
        "sensitivity_mean_mse_m2": {key: float(np.mean(value)) for key, value in sensitivity.items()},
        "peak_frame_per_agent": np.argmax(curves["grouped_mse_m2"], axis=0).tolist(),
        "per_node_rms_cm_per_agent": (100 * np.sqrt(np.mean(residuals[:, :, :bodies] ** 2, axis=0))).tolist(),
    }
    arrays = {"adjacency": graphs, "residual": residuals, **curves}
    return summary, arrays
