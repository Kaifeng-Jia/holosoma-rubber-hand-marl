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
