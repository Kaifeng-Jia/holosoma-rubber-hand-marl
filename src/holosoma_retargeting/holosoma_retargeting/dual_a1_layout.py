"""Pure geometry helpers shared by dual-agent A1 inspection tools."""

from __future__ import annotations

from pathlib import Path

import numpy as np


G1_FLIP_SIGN_JOINT_NAMES = frozenset(
    {
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "left_ankle_roll_joint",
        "right_ankle_roll_joint",
        "waist_roll_joint",
        "waist_yaw_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "left_wrist_roll_joint",
        "left_wrist_yaw_joint",
        "right_wrist_roll_joint",
        "right_wrist_yaw_joint",
    }
)


def _normalized_quaternion_wxyz(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.shape[-1] != 4:
        raise ValueError(f"Expected quaternion final dimension 4, got {quaternion.shape}")
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    if np.any(norm <= 0.0):
        raise ValueError("Quaternion contains a zero-norm value")
    return quaternion / norm


def quaternion_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Convert one or more normalized wxyz quaternions to rotation matrices."""
    quat = _normalized_quaternion_wxyz(quaternion)
    w, x, y, z = np.moveaxis(quat, -1, 0)
    matrix = np.empty(quat.shape[:-1] + (3, 3), dtype=np.float64)
    matrix[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    matrix[..., 0, 1] = 2.0 * (x * y - z * w)
    matrix[..., 0, 2] = 2.0 * (x * z + y * w)
    matrix[..., 1, 0] = 2.0 * (x * y + z * w)
    matrix[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    matrix[..., 1, 2] = 2.0 * (y * z - x * w)
    matrix[..., 2, 0] = 2.0 * (x * z - y * w)
    matrix[..., 2, 1] = 2.0 * (y * z + x * w)
    matrix[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return matrix


def matrix_to_quaternion_wxyz(matrix: np.ndarray) -> np.ndarray:
    """Convert proper rotation matrices to canonicalized wxyz quaternions."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"Expected rotation matrices [..., 3, 3], got {matrix.shape}")
    flat = matrix.reshape(-1, 3, 3)
    output = np.empty((len(flat), 4), dtype=np.float64)
    for index, rotation in enumerate(flat):
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-8) or not np.isclose(
            np.linalg.det(rotation), 1.0, atol=1.0e-8
        ):
            raise ValueError("Matrix is not a proper rotation")
        trace = float(np.trace(rotation))
        if trace > 0.0:
            scale = 2.0 * np.sqrt(trace + 1.0)
            quat = np.array(
                [0.25 * scale, (rotation[2, 1] - rotation[1, 2]) / scale,
                 (rotation[0, 2] - rotation[2, 0]) / scale, (rotation[1, 0] - rotation[0, 1]) / scale]
            )
        else:
            diagonal_index = int(np.argmax(np.diag(rotation)))
            if diagonal_index == 0:
                scale = 2.0 * np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
                quat = np.array(
                    [(rotation[2, 1] - rotation[1, 2]) / scale, 0.25 * scale,
                     (rotation[0, 1] + rotation[1, 0]) / scale, (rotation[0, 2] + rotation[2, 0]) / scale]
                )
            elif diagonal_index == 1:
                scale = 2.0 * np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])
                quat = np.array(
                    [(rotation[0, 2] - rotation[2, 0]) / scale,
                     (rotation[0, 1] + rotation[1, 0]) / scale, 0.25 * scale,
                     (rotation[1, 2] + rotation[2, 1]) / scale]
                )
            else:
                scale = 2.0 * np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
                quat = np.array(
                    [(rotation[1, 0] - rotation[0, 1]) / scale,
                     (rotation[0, 2] + rotation[2, 0]) / scale,
                     (rotation[1, 2] + rotation[2, 1]) / scale, 0.25 * scale]
                )
        quat /= np.linalg.norm(quat)
        output[index] = quat if quat[0] >= 0.0 else -quat
    return output.reshape(matrix.shape[:-2] + (4,))


def load_a1_motion(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Load the robot, object, and timing channels used by the A1 layout."""
    with np.load(path, allow_pickle=False) as data:
        required = {"joint_pos", "object_pos_w", "object_quat_w", "fps"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"Motion file is missing required channels: {sorted(missing)}")

        joint_pos = np.asarray(data["joint_pos"], dtype=np.float64)
        object_pos = np.asarray(data["object_pos_w"], dtype=np.float64)
        object_quat = np.asarray(data["object_quat_w"], dtype=np.float64)
        fps = int(np.asarray(data["fps"]).reshape(-1)[0])

    if joint_pos.ndim != 2 or joint_pos.shape[1] != 36:
        raise ValueError(f"Expected joint_pos [T, 36], got {joint_pos.shape}")
    if object_pos.shape != (joint_pos.shape[0], 3):
        raise ValueError(f"Expected object_pos_w [T, 3], got {object_pos.shape}")
    if object_quat.shape != (joint_pos.shape[0], 4):
        raise ValueError(f"Expected object_quat_w [T, 4], got {object_quat.shape}")
    if fps <= 0:
        raise ValueError(f"FPS must be positive, got {fps}")

    return joint_pos, object_pos, object_quat, fps


def table_local_x_in_world(object_quat_wxyz: np.ndarray) -> np.ndarray:
    """Return the table local-X unit vector expressed in world coordinates."""
    return quaternion_wxyz_to_matrix(object_quat_wxyz)[..., :, 0]


def g1_sagittal_joint_map(joint_names: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build the established G1 left/right joint permutation and sign mask."""
    names = [str(name) for name in np.asarray(joint_names).tolist()]
    if len(names) != len(set(names)):
        raise ValueError("Joint names must be unique")
    name_to_index = {name: index for index, name in enumerate(names)}
    mapped_names = []
    for name in names:
        if name.startswith("left_"):
            mapped_names.append("right_" + name[len("left_") :])
        elif name.startswith("right_"):
            mapped_names.append("left_" + name[len("right_") :])
        else:
            mapped_names.append(name)
    missing = sorted(set(mapped_names).difference(name_to_index))
    if missing:
        raise ValueError(f"Joint mirror partners are missing: {missing}")
    index_map = np.asarray([name_to_index[name] for name in mapped_names], dtype=np.int64)
    sign_mask = np.asarray([-1.0 if name in G1_FLIP_SIGN_JOINT_NAMES else 1.0 for name in names])
    return index_map, sign_mask


def mirror_g1_robot_joint_positions_about_table(
    joint_pos: np.ndarray,
    joint_names: np.ndarray,
    object_pos_w: np.ndarray,
    object_quat_wxyz: np.ndarray,
) -> np.ndarray:
    """Mirror a G1 qpos trajectory across the table-local YZ plane.

    The table is not part of the returned state. Its pose defines the moving
    mirror plane. The robot's local sagittal reflection is the XZ plane, so the
    root orientation uses ``S_world @ R_root @ S_robot`` to remain a proper
    rotation after reflection.
    """
    joint_pos = np.asarray(joint_pos, dtype=np.float64)
    object_pos_w = np.asarray(object_pos_w, dtype=np.float64)
    object_quat_wxyz = np.asarray(object_quat_wxyz, dtype=np.float64)
    if joint_pos.ndim != 2 or joint_pos.shape[1] != 7 + len(joint_names):
        raise ValueError(f"Expected joint_pos [T, 7 + {len(joint_names)}], got {joint_pos.shape}")
    if object_pos_w.shape != (len(joint_pos), 3):
        raise ValueError(f"Expected object_pos_w [T, 3], got {object_pos_w.shape}")
    if object_quat_wxyz.shape != (len(joint_pos), 4):
        raise ValueError(f"Expected object_quat_wxyz [T, 4], got {object_quat_wxyz.shape}")

    table_rotation = quaternion_wxyz_to_matrix(object_quat_wxyz)
    table_reflection = np.einsum(
        "tij,jk,tlk->til",
        table_rotation,
        np.diag([-1.0, 1.0, 1.0]),
        table_rotation,
    )
    root_relative = joint_pos[:, :3] - object_pos_w
    mirrored_root_pos = object_pos_w + np.einsum("tij,tj->ti", table_reflection, root_relative)

    root_rotation = quaternion_wxyz_to_matrix(joint_pos[:, 3:7])
    robot_sagittal_reflection = np.diag([1.0, -1.0, 1.0])
    mirrored_root_rotation = np.einsum(
        "tij,tjk,kl->til",
        table_reflection,
        root_rotation,
        robot_sagittal_reflection,
    )
    mirrored_root_quat = matrix_to_quaternion_wxyz(mirrored_root_rotation)

    joint_index_map, sign_mask = g1_sagittal_joint_map(joint_names)
    mirrored_dof_pos = joint_pos[:, 7:][:, joint_index_map] * sign_mask
    return np.concatenate((mirrored_root_pos, mirrored_root_quat, mirrored_dof_pos), axis=-1)


def _quaternion_multiply_wxyz(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = np.moveaxis(first, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(second, -1, 0)
    return np.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        axis=-1,
    )


def _quaternion_to_rotvec_wxyz(quaternion: np.ndarray) -> np.ndarray:
    quat = _normalized_quaternion_wxyz(quaternion)
    quat = np.where(quat[..., :1] < 0.0, -quat, quat)
    vector_norm = np.linalg.norm(quat[..., 1:], axis=-1)
    angle = 2.0 * np.arctan2(vector_norm, np.clip(quat[..., 0], -1.0, 1.0))
    scale = np.divide(angle, vector_norm, out=np.full_like(angle, 2.0), where=vector_norm > 1.0e-12)
    return quat[..., 1:] * scale[..., None]


def robot_velocities_from_joint_positions(joint_pos: np.ndarray, fps: int) -> np.ndarray:
    """Recompute root/world and DOF velocities using the retargeting convention."""
    joint_pos = np.asarray(joint_pos, dtype=np.float64)
    if joint_pos.ndim != 2 or joint_pos.shape[1] < 8:
        raise ValueError(f"Expected joint_pos [T, >=8], got {joint_pos.shape}")
    if len(joint_pos) < 3:
        raise ValueError("At least three frames are required to compute velocities")
    if fps <= 0:
        raise ValueError(f"FPS must be positive, got {fps}")
    dt = 1.0 / float(fps)
    root_lin_vel = np.gradient(joint_pos[:, :3], dt, axis=0)
    dof_vel = np.gradient(joint_pos[:, 7:], dt, axis=0)
    root_quat = _normalized_quaternion_wxyz(joint_pos[:, 3:7])
    root_quat = np.array(root_quat, copy=True)
    for index in range(1, len(root_quat)):
        if np.dot(root_quat[index - 1], root_quat[index]) < 0.0:
            root_quat[index] *= -1.0
    relative = _quaternion_multiply_wxyz(
        root_quat[2:],
        root_quat[:-2] * np.array([1.0, -1.0, -1.0, -1.0]),
    )
    interior_ang_vel = _quaternion_to_rotvec_wxyz(relative) / (2.0 * dt)
    root_ang_vel = np.concatenate((interior_ang_vel[:1], interior_ang_vel, interior_ang_vel[-1:]), axis=0)
    return np.concatenate((root_lin_vel, root_ang_vel, dof_vel), axis=-1)


def shifted_robot_positions(
    root_pos: np.ndarray,
    object_quat_wxyz: np.ndarray,
    lateral_spacing: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Create symmetric A1 root positions around the original reference."""
    if lateral_spacing <= 0.0:
        raise ValueError(f"lateral_spacing must be positive, got {lateral_spacing}")
    half_offset = 0.5 * lateral_spacing * table_local_x_in_world(object_quat_wxyz)
    return root_pos - half_offset, root_pos + half_offset


def lateral_offset_trajectory(
    object_quat_wxyz: np.ndarray,
    lateral_spacing: float,
    observer_side: int,
    fps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one observer's world-frame lateral position and velocity offsets."""
    if observer_side not in (-1, 1):
        raise ValueError(f"observer_side must be -1 or +1, got {observer_side}")
    if lateral_spacing <= 0.0:
        raise ValueError(f"lateral_spacing must be positive, got {lateral_spacing}")
    if fps <= 0:
        raise ValueError(f"FPS must be positive, got {fps}")

    axis = table_local_x_in_world(object_quat_wxyz)
    if axis.ndim != 2:
        raise ValueError(f"Expected object quaternion trajectory [T, 4], got {object_quat_wxyz.shape}")
    offset = 0.5 * float(observer_side) * lateral_spacing * axis
    edge_order = 2 if len(offset) > 2 else 1
    offset_velocity = np.gradient(offset, 1.0 / float(fps), axis=0, edge_order=edge_order)
    return offset, offset_velocity


def object_centered_lateral_offset_trajectory(
    root_pos_w: np.ndarray,
    object_pos_w: np.ndarray,
    object_quat_wxyz: np.ndarray,
    lateral_spacing: float,
    observer_side: int,
    fps: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Shift the root so its mean table-local X is exactly one half-spacing.

    Only a constant table-local-X bias is removed. Frame-to-frame root motion
    relative to the object remains unchanged.
    """
    if observer_side not in (-1, 1):
        raise ValueError(f"observer_side must be -1 or +1, got {observer_side}")
    if lateral_spacing <= 0.0:
        raise ValueError(f"lateral_spacing must be positive, got {lateral_spacing}")
    if fps <= 0:
        raise ValueError(f"FPS must be positive, got {fps}")
    if root_pos_w.shape != object_pos_w.shape or root_pos_w.ndim != 2 or root_pos_w.shape[1] != 3:
        raise ValueError(
            f"root_pos_w and object_pos_w must both be [T, 3], got {root_pos_w.shape} and {object_pos_w.shape}"
        )
    if object_quat_wxyz.shape != (root_pos_w.shape[0], 4):
        raise ValueError(f"Expected object quaternion trajectory [T, 4], got {object_quat_wxyz.shape}")

    axis = table_local_x_in_world(object_quat_wxyz)
    source_local_x = np.sum((root_pos_w - object_pos_w) * axis, axis=-1)
    source_mean_local_x = float(np.mean(source_local_x))
    target_mean_local_x = 0.5 * float(observer_side) * lateral_spacing
    constant_correction = target_mean_local_x - source_mean_local_x
    offset = constant_correction * axis
    edge_order = 2 if len(offset) > 2 else 1
    offset_velocity = np.gradient(offset, 1.0 / float(fps), axis=0, edge_order=edge_order)
    return offset, offset_velocity, source_mean_local_x


def shifted_robot_motion_channels(
    *,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    body_pos_w: np.ndarray,
    body_lin_vel_w: np.ndarray,
    object_quat_wxyz: np.ndarray,
    lateral_spacing: float,
    observer_side: int,
    fps: int,
    object_pos_w: np.ndarray | None = None,
    object_centered_mean: bool = False,
) -> dict[str, np.ndarray]:
    """Shift all robot translation channels while leaving the object unchanged."""
    frame_count = joint_pos.shape[0]
    expected = {
        "joint_vel": joint_vel.shape[0],
        "body_pos_w": body_pos_w.shape[0],
        "body_lin_vel_w": body_lin_vel_w.shape[0],
        "object_quat_wxyz": object_quat_wxyz.shape[0],
    }
    mismatched = {name: count for name, count in expected.items() if count != frame_count}
    if mismatched:
        raise ValueError(f"Motion channel frame counts do not match joint_pos={frame_count}: {mismatched}")
    if joint_pos.ndim != 2 or joint_pos.shape[1] < 3:
        raise ValueError(f"Expected joint_pos [T, >=3], got {joint_pos.shape}")
    if joint_vel.ndim != 2 or joint_vel.shape[1] < 3:
        raise ValueError(f"Expected joint_vel [T, >=3], got {joint_vel.shape}")
    if body_pos_w.ndim != 3 or body_pos_w.shape[2] != 3:
        raise ValueError(f"Expected body_pos_w [T, B, 3], got {body_pos_w.shape}")
    if body_lin_vel_w.shape != body_pos_w.shape:
        raise ValueError(
            f"body_lin_vel_w must match body_pos_w shape {body_pos_w.shape}, got {body_lin_vel_w.shape}"
        )

    source_mean_local_x = None
    if object_centered_mean:
        if object_pos_w is None:
            raise ValueError("object_pos_w is required when object_centered_mean=True")
        offset, offset_velocity, source_mean_local_x = object_centered_lateral_offset_trajectory(
            joint_pos[:, :3],
            object_pos_w,
            object_quat_wxyz,
            lateral_spacing,
            observer_side,
            fps,
        )
    else:
        offset, offset_velocity = lateral_offset_trajectory(
            object_quat_wxyz,
            lateral_spacing,
            observer_side,
            fps,
        )
    shifted_joint_pos = np.array(joint_pos, copy=True)
    shifted_joint_pos[:, :3] += offset
    shifted_joint_vel = np.array(joint_vel, copy=True)
    shifted_joint_vel[:, :3] += offset_velocity
    shifted_body_pos = np.array(body_pos_w, copy=True)
    shifted_body_pos += offset[:, None, :]
    shifted_body_lin_vel = np.array(body_lin_vel_w, copy=True)
    shifted_body_lin_vel += offset_velocity[:, None, :]
    result = {
        "joint_pos": shifted_joint_pos,
        "joint_vel": shifted_joint_vel,
        "body_pos_w": shifted_body_pos,
        "body_lin_vel_w": shifted_body_lin_vel,
        "lateral_offset_w": offset,
        "lateral_offset_velocity_w": offset_velocity,
    }
    if source_mean_local_x is not None:
        result["source_mean_table_local_x_m"] = np.asarray(source_mean_local_x)
    return result
