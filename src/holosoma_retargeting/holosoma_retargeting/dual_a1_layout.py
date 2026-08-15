"""Pure geometry helpers shared by dual-agent A1 inspection tools."""

from __future__ import annotations

from pathlib import Path

import numpy as np


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
    quat = np.asarray(object_quat_wxyz, dtype=np.float64)
    if quat.shape[-1] != 4:
        raise ValueError(f"Expected quaternion final dimension 4, got {quat.shape}")
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    if np.any(norm <= 0.0):
        raise ValueError("Object quaternion contains a zero-norm value")
    quat = quat / norm
    w, x, y, z = np.moveaxis(quat, -1, 0)
    return np.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + z * w),
            2.0 * (x * z - y * w),
        ),
        axis=-1,
    )


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
    return {
        "joint_pos": shifted_joint_pos,
        "joint_vel": shifted_joint_vel,
        "body_pos_w": shifted_body_pos,
        "body_lin_vel_w": shifted_body_lin_vel,
        "lateral_offset_w": offset,
        "lateral_offset_velocity_w": offset_velocity,
    }
