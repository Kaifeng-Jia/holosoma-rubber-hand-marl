"""Offline synthesis of a symmetric two-robot Pull reference.

The accepted single-robot Pull rollout stores one 29-DoF G1 and one table in
MuJoCo qpos order::

    robot_xyz, robot_quat_wxyz, robot_joints[29],
    object_xyz, object_quat_wxyz

Pull uses table-local Z as the lateral axis.  The two robots are therefore
related by reflection across the table-local XY plane.  The shared table
trajectory is first symmetrized in the *initial* table frame: lateral
translation is averaged with its reflection and orientation is the geodesic
midpoint between the measured relative rotation and its reflected rotation.

This module is deliberately independent of the online Plan 5 command and the
Push A1 reference code.  It only creates an offline reference artifact.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from holosoma_retargeting.dual_a1_layout import (
    g1_sagittal_joint_map,
    matrix_to_quaternion_wxyz,
    quaternion_wxyz_to_matrix,
)


G1_29DOF_JOINT_NAMES = np.asarray(
    [
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ]
)

ROBOT_QPOS_DIM = 7 + len(G1_29DOF_JOINT_NAMES)
OBJECT_QPOS_DIM = 7
SINGLE_ROBOT_PULL_QPOS_DIM = ROBOT_QPOS_DIM + OBJECT_QPOS_DIM

# Difference between the original small-table leg coordinate and the new
# Z-wide-table leg coordinate.  It is opt-in so pure mirroring remains the
# default and callers must make the formal preview choice explicit.
FORMAL_PULL_LATERAL_LEG_OFFSET_M = 0.4390236

# Pull forward/backward is table-local X and local Y is vertical.  Thus local Z
# is the lateral direction and the collaboration mirror plane is local XY.
PULL_LOCAL_XY_REFLECTION = np.diag([1.0, 1.0, -1.0])

# A G1 sagittal mirror swaps left/right across the robot-local XZ plane.
G1_LOCAL_SAGITTAL_REFLECTION = np.diag([1.0, -1.0, 1.0])


@dataclass(frozen=True)
class DualPullReference:
    """Two robot qpos trajectories and one shared object qpos trajectory."""

    robot_a_qpos: np.ndarray
    robot_b_qpos: np.ndarray
    shared_object_qpos: np.ndarray
    fps: int
    lateral_leg_offset_m: float = 0.0

    def save(self, path: str | Path) -> Path:
        """Save the explicit offline reference contract as a compressed NPZ."""
        output = Path(path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            robot_a_qpos=self.robot_a_qpos,
            robot_b_qpos=self.robot_b_qpos,
            shared_object_qpos=self.shared_object_qpos,
            fps=np.asarray(self.fps, dtype=np.int64),
            joint_names=G1_29DOF_JOINT_NAMES,
            quaternion_convention=np.asarray("wxyz"),
            lateral_axis=np.asarray("table_local_z"),
            mirror_plane=np.asarray("table_local_xy"),
            lateral_leg_offset_m=np.asarray(self.lateral_leg_offset_m, dtype=np.float64),
        )
        return output

    def save_canonical_rollout(
        self,
        path: str | Path,
        *,
        provenance: Mapping[str, object] | None = None,
    ) -> Path:
        """Save the five canonical channels consumed by ``viser_dual_a1_player``.

        Internal qpos quaternions stay WXYZ.  Only this file boundary converts
        root and object orientations to the player's explicit XYZW channels.
        Extra scalar metadata is intentionally harmless because the player
        validates required channels without requiring an exact key set.
        """
        output = Path(path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        provenance_payload: dict[str, object] = {
            "generator": "holosoma_retargeting.dual_pull_reference",
            "source_quaternion_convention": "wxyz",
            "canonical_quaternion_convention": "xyzw",
            "lateral_axis": "table_local_z",
            "mirror_plane": "table_local_xy",
            "lateral_leg_offset_m": self.lateral_leg_offset_m,
        }
        if provenance is not None:
            provenance_payload.update(dict(provenance))
        root_wxyz = np.stack(
            (self.robot_a_qpos[:, 3:7], self.robot_b_qpos[:, 3:7]),
            axis=1,
        )
        np.savez_compressed(
            output,
            root_pos=np.stack(
                (self.robot_a_qpos[:, :3], self.robot_b_qpos[:, :3]),
                axis=1,
            ),
            root_quat_xyzw=root_wxyz[..., [1, 2, 3, 0]],
            dof_pos=np.stack(
                (self.robot_a_qpos[:, 7:], self.robot_b_qpos[:, 7:]),
                axis=1,
            ),
            object_pos_w=self.shared_object_qpos[:, :3],
            object_quat_xyzw=self.shared_object_qpos[:, [4, 5, 6, 3]],
            fps=np.asarray(self.fps, dtype=np.int64),
            provenance=np.asarray(json.dumps(provenance_payload, sort_keys=True)),
        )
        return output


def _continuous_quaternion_wxyz(
    quaternion: np.ndarray,
    *,
    first_reference: np.ndarray | None = None,
) -> np.ndarray:
    """Normalize a WXYZ quaternion trajectory and remove representation flips."""
    result = np.asarray(quaternion, dtype=np.float64).copy()
    if result.ndim != 2 or result.shape[1] != 4:
        raise ValueError(f"Expected quaternion trajectory [T, 4], got {result.shape}")
    norms = np.linalg.norm(result, axis=-1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 1.0e-12):
        raise ValueError("Quaternion trajectory contains a non-finite or zero-norm value")
    result /= norms
    if first_reference is not None:
        reference = np.asarray(first_reference, dtype=np.float64).copy()
        reference /= np.linalg.norm(reference)
        if np.dot(result[0], reference) < 0.0:
            result[0] *= -1.0
    for index in range(1, len(result)):
        if np.dot(result[index - 1], result[index]) < 0.0:
            result[index] *= -1.0
    return result


def _rotation_geodesic_midpoint(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return shortest-arc SO(3) midpoints for two rotation trajectories."""
    first_quat = matrix_to_quaternion_wxyz(first)
    second_quat = matrix_to_quaternion_wxyz(second)
    align = np.where(np.sum(first_quat * second_quat, axis=-1, keepdims=True) < 0.0, -1.0, 1.0)
    midpoint = first_quat + align * second_quat
    norms = np.linalg.norm(midpoint, axis=-1, keepdims=True)
    if np.any(norms <= 1.0e-12):
        raise ValueError("Object rotation and its mirror have an ambiguous geodesic midpoint")
    midpoint /= norms
    return quaternion_wxyz_to_matrix(midpoint)


def _symmetrize_object_qpos(object_qpos: np.ndarray) -> np.ndarray:
    """Remove table-local-Z asymmetry from one measured object trajectory."""
    object_pos = object_qpos[:, :3]
    object_quat = object_qpos[:, 3:7]
    object_rotation = quaternion_wxyz_to_matrix(object_quat)
    initial_pos = object_pos[0]
    initial_rotation = object_rotation[0]

    relative_pos = np.einsum(
        "ij,tj->ti",
        initial_rotation.T,
        object_pos - initial_pos,
    )
    mirrored_relative_pos = np.einsum("ij,tj->ti", PULL_LOCAL_XY_REFLECTION, relative_pos)
    symmetric_relative_pos = 0.5 * (relative_pos + mirrored_relative_pos)
    shared_pos = initial_pos + np.einsum(
        "ij,tj->ti",
        initial_rotation,
        symmetric_relative_pos,
    )

    relative_rotation = np.einsum("ij,tjk->tik", initial_rotation.T, object_rotation)
    mirrored_relative_rotation = np.einsum(
        "ij,tjk,kl->til",
        PULL_LOCAL_XY_REFLECTION,
        relative_rotation,
        PULL_LOCAL_XY_REFLECTION,
    )
    symmetric_relative_rotation = _rotation_geodesic_midpoint(
        relative_rotation,
        mirrored_relative_rotation,
    )
    shared_rotation = np.einsum("ij,tjk->tik", initial_rotation, symmetric_relative_rotation)
    shared_quat = _continuous_quaternion_wxyz(
        matrix_to_quaternion_wxyz(shared_rotation),
        first_reference=object_quat[0],
    )
    return np.concatenate((shared_pos, shared_quat), axis=-1)


def _transport_source_robot_to_shared_object(
    robot_qpos: np.ndarray,
    source_object_qpos: np.ndarray,
    shared_object_qpos: np.ndarray,
) -> np.ndarray:
    """Preserve the source robot pose in the table frame after symmetrization."""
    source_object_rotation = quaternion_wxyz_to_matrix(source_object_qpos[:, 3:7])
    shared_object_rotation = quaternion_wxyz_to_matrix(shared_object_qpos[:, 3:7])
    source_robot_rotation = quaternion_wxyz_to_matrix(robot_qpos[:, 3:7])
    rotation_delta = np.einsum(
        "tij,tkj->tik",
        shared_object_rotation,
        source_object_rotation,
    )
    source_relative_pos = robot_qpos[:, :3] - source_object_qpos[:, :3]
    robot_a_pos = shared_object_qpos[:, :3] + np.einsum(
        "tij,tj->ti",
        rotation_delta,
        source_relative_pos,
    )
    robot_a_rotation = np.einsum("tij,tjk->tik", rotation_delta, source_robot_rotation)
    robot_a_quat = _continuous_quaternion_wxyz(
        matrix_to_quaternion_wxyz(robot_a_rotation),
        first_reference=robot_qpos[0, 3:7],
    )
    return np.concatenate((robot_a_pos, robot_a_quat, robot_qpos[:, 7:]), axis=-1)


def _mirror_robot_across_shared_table(
    robot_a_qpos: np.ndarray,
    shared_object_qpos: np.ndarray,
    joint_names: np.ndarray,
) -> np.ndarray:
    """Create robot B by a physical G1 sagittal mirror across table local XY."""
    shared_pos = shared_object_qpos[:, :3]
    shared_rotation = quaternion_wxyz_to_matrix(shared_object_qpos[:, 3:7])
    robot_a_rotation = quaternion_wxyz_to_matrix(robot_a_qpos[:, 3:7])
    world_reflection = np.einsum(
        "tij,jk,tlk->til",
        shared_rotation,
        PULL_LOCAL_XY_REFLECTION,
        shared_rotation,
    )
    robot_b_pos = shared_pos + np.einsum(
        "tij,tj->ti",
        world_reflection,
        robot_a_qpos[:, :3] - shared_pos,
    )
    robot_b_rotation = np.einsum(
        "tij,tjk,kl->til",
        world_reflection,
        robot_a_rotation,
        G1_LOCAL_SAGITTAL_REFLECTION,
    )
    if not np.allclose(np.linalg.det(robot_b_rotation), 1.0, atol=1.0e-8):
        raise ValueError("Mirrored robot root orientation is not a proper rotation")
    robot_b_quat = _continuous_quaternion_wxyz(matrix_to_quaternion_wxyz(robot_b_rotation))
    index_map, sign_mask = g1_sagittal_joint_map(joint_names)
    robot_b_joints = robot_a_qpos[:, 7:][:, index_map] * sign_mask
    return np.concatenate((robot_b_pos, robot_b_quat, robot_b_joints), axis=-1)


def _offset_robots_to_wide_table_legs(
    robot_a_qpos: np.ndarray,
    robot_b_qpos: np.ndarray,
    shared_object_qpos: np.ndarray,
    lateral_leg_offset_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Move A/B equally outward along shared table local Z."""
    if lateral_leg_offset_m == 0.0:
        return robot_a_qpos, robot_b_qpos
    shared_rotation = quaternion_wxyz_to_matrix(shared_object_qpos[:, 3:7])
    local_z_w = shared_rotation[..., :, 2]
    robot_a = robot_a_qpos.copy()
    robot_b = robot_b_qpos.copy()
    robot_a[:, :3] -= lateral_leg_offset_m * local_z_w
    robot_b[:, :3] += lateral_leg_offset_m * local_z_w
    return robot_a, robot_b


def synthesize_dual_pull_reference(
    qpos: np.ndarray,
    *,
    fps: int,
    joint_names: np.ndarray = G1_29DOF_JOINT_NAMES,
    lateral_leg_offset_m: float = 0.0,
) -> DualPullReference:
    """Synthesize a symmetric two-agent Pull reference from one physical rollout."""
    qpos = np.asarray(qpos, dtype=np.float64)
    joint_names = np.asarray(joint_names)
    if qpos.ndim != 2 or qpos.shape[1] != SINGLE_ROBOT_PULL_QPOS_DIM:
        raise ValueError(
            "Expected single Pull qpos with shape "
            f"[T, {SINGLE_ROBOT_PULL_QPOS_DIM}], got {qpos.shape}"
        )
    if len(qpos) < 1 or not np.isfinite(qpos).all():
        raise ValueError("Pull qpos must contain at least one finite frame")
    if not isinstance(fps, (int, np.integer)) or int(fps) <= 0:
        raise ValueError(f"FPS must be a positive integer, got {fps!r}")
    lateral_leg_offset_m = float(lateral_leg_offset_m)
    if not np.isfinite(lateral_leg_offset_m) or lateral_leg_offset_m < 0.0:
        raise ValueError(
            "lateral_leg_offset_m must be finite and non-negative, "
            f"got {lateral_leg_offset_m!r}"
        )
    if joint_names.shape != (len(G1_29DOF_JOINT_NAMES),):
        raise ValueError(
            f"Expected {len(G1_29DOF_JOINT_NAMES)} G1 joint names, got {joint_names.shape}"
        )
    # Validate the left/right naming contract before doing any geometry work.
    g1_sagittal_joint_map(joint_names)

    robot_qpos = qpos[:, :ROBOT_QPOS_DIM]
    source_object_qpos = qpos[:, ROBOT_QPOS_DIM:]
    for name, quaternion in (
        ("robot", robot_qpos[:, 3:7]),
        ("object", source_object_qpos[:, 3:7]),
    ):
        norms = np.linalg.norm(quaternion, axis=-1)
        if np.any(~np.isfinite(norms)) or np.any(norms <= 1.0e-12):
            raise ValueError(f"{name} qpos contains a non-finite or zero-norm quaternion")

    shared_object_qpos = _symmetrize_object_qpos(source_object_qpos)
    robot_a_qpos = _transport_source_robot_to_shared_object(
        robot_qpos,
        source_object_qpos,
        shared_object_qpos,
    )
    robot_b_qpos = _mirror_robot_across_shared_table(
        robot_a_qpos,
        shared_object_qpos,
        joint_names,
    )
    robot_a_qpos, robot_b_qpos = _offset_robots_to_wide_table_legs(
        robot_a_qpos,
        robot_b_qpos,
        shared_object_qpos,
        lateral_leg_offset_m,
    )
    return DualPullReference(
        robot_a_qpos=robot_a_qpos,
        robot_b_qpos=robot_b_qpos,
        shared_object_qpos=shared_object_qpos,
        fps=int(fps),
        lateral_leg_offset_m=lateral_leg_offset_m,
    )


def synthesize_dual_pull_reference_file(
    source_path: str | Path,
    output_path: str | Path,
    *,
    lateral_leg_offset_m: float = 0.0,
) -> DualPullReference:
    """Load an attempt-style ``qpos`` NPZ, synthesize it, and save the result."""
    source = Path(source_path).expanduser().resolve()
    with np.load(source, allow_pickle=False) as data:
        missing = {"qpos", "fps"}.difference(data.files)
        if missing:
            raise ValueError(f"Pull rollout is missing required channels: {sorted(missing)}")
        qpos = np.asarray(data["qpos"])
        fps_value = np.asarray(data["fps"])
    if fps_value.size != 1:
        raise ValueError(f"Pull rollout FPS must be scalar, got shape {fps_value.shape}")
    fps_float = float(fps_value.reshape(()))
    if not np.isfinite(fps_float) or not fps_float.is_integer():
        raise ValueError(f"Pull rollout FPS must be a finite integer, got {fps_float!r}")
    result = synthesize_dual_pull_reference(
        qpos,
        fps=int(fps_float),
        lateral_leg_offset_m=lateral_leg_offset_m,
    )
    result.save(output_path)
    return result


__all__ = [
    "DualPullReference",
    "FORMAL_PULL_LATERAL_LEG_OFFSET_M",
    "G1_29DOF_JOINT_NAMES",
    "PULL_LOCAL_XY_REFLECTION",
    "synthesize_dual_pull_reference",
    "synthesize_dual_pull_reference_file",
]
