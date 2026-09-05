"""Build a planar-neutral two-robot table-leg tug-of-war preview.

The source can be the original single-robot square-table Pull motion or the
accepted cooperative Pull runtime reference.  Agent A is kept on its original
table leg and a second copy of A is rotated by 180 degrees around the world
vertical axis.  This places the two agents at diagonally opposite table legs.

Only the source table's planar translation and heading are removed.  Its
vertical motion and tilt are retained so that the robot remains on the world
floor while its frame-wise table-relative Pull pose stays unchanged.  This
preview contract must not be used as an object-tracking target for the
competitive physics task, where table motion is determined by contact forces.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from holosoma_retargeting.dual_a1_layout import (
    matrix_to_quaternion_wxyz,
    quaternion_wxyz_to_matrix,
)


ROBOT_DOF = 29
ROBOT_QPOS_DIM = 7 + ROBOT_DOF
WORLD_Z_HALF_TURN = np.diag([-1.0, -1.0, 1.0])
HEADING_NORM_EPS = 1.0e-8


def _continuous_quaternion_wxyz(quaternion: np.ndarray) -> np.ndarray:
    """Normalize a quaternion trajectory and remove representation flips."""
    result = np.asarray(quaternion, dtype=np.float64).copy()
    if result.ndim != 2 or result.shape[1] != 4:
        raise ValueError(f"Expected quaternion trajectory [T, 4], got {result.shape}")
    norms = np.linalg.norm(result, axis=-1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 1.0e-12):
        raise ValueError("Quaternion trajectory contains a non-finite or zero-norm value")
    result /= norms
    for frame in range(1, len(result)):
        if np.dot(result[frame - 1], result[frame]) < 0.0:
            result[frame] *= -1.0
    return result


def _table_heading_xy(table_rotation: np.ndarray) -> np.ndarray:
    """Return the unit world-XY projection of the table's local X axis."""
    heading = np.asarray(table_rotation, dtype=np.float64)[:, :2, 0].copy()
    norm = np.linalg.norm(heading, axis=-1, keepdims=True)
    if np.any(~np.isfinite(norm)) or np.any(norm <= HEADING_NORM_EPS):
        raise ValueError("Table local X axis has a degenerate world-XY projection")
    return heading / norm


def _world_z_heading_correction(table_rotation: np.ndarray) -> np.ndarray:
    """Build rotations that map every table heading onto its initial heading."""
    heading = _table_heading_xy(table_rotation)
    initial = heading[:1]
    cosine = np.sum(heading * initial, axis=-1)
    sine = heading[:, 0] * initial[:, 1] - heading[:, 1] * initial[:, 0]
    correction = np.zeros((len(table_rotation), 3, 3), dtype=np.float64)
    correction[:, 0, 0] = cosine
    correction[:, 0, 1] = -sine
    correction[:, 1, 0] = sine
    correction[:, 1, 1] = cosine
    correction[:, 2, 2] = 1.0
    return correction


def _remove_table_planar_motion(
    robot_qpos: np.ndarray,
    source_object_qpos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove table XY/heading motion without moving the robot through the floor."""
    source_rotation = quaternion_wxyz_to_matrix(source_object_qpos[:, 3:7])
    rotation_delta = _world_z_heading_correction(source_rotation)

    target_object_position = source_object_qpos[:, :3].copy()
    target_object_position[:, :2] = source_object_qpos[0, :2]
    target_object_rotation = np.einsum("tij,tjk->tik", rotation_delta, source_rotation)
    target_object_quaternion = _continuous_quaternion_wxyz(
        matrix_to_quaternion_wxyz(target_object_rotation)
    )
    target_object_qpos = np.concatenate(
        (target_object_position, target_object_quaternion), axis=-1
    )

    robot_rotation = quaternion_wxyz_to_matrix(robot_qpos[:, 3:7])
    target_robot_position = target_object_position + np.einsum(
        "tij,tj->ti",
        rotation_delta,
        robot_qpos[:, :3] - source_object_qpos[:, :3],
    )
    target_robot_rotation = np.einsum("tij,tjk->tik", rotation_delta, robot_rotation)
    target_robot_quaternion = _continuous_quaternion_wxyz(
        matrix_to_quaternion_wxyz(target_robot_rotation)
    )
    target_robot_qpos = np.concatenate(
        (target_robot_position, target_robot_quaternion, robot_qpos[:, 7:]), axis=-1
    )
    return target_robot_qpos, target_object_qpos


def _turn_robot_about_world_vertical(
    robot_qpos: np.ndarray,
    object_qpos: np.ndarray,
) -> np.ndarray:
    """Rotate a complete robot pose by pi around the world vertical axis."""
    turned_position = object_qpos[:, :3] + np.einsum(
        "ij,tj->ti",
        WORLD_Z_HALF_TURN,
        robot_qpos[:, :3] - object_qpos[:, :3],
    )
    robot_rotation = quaternion_wxyz_to_matrix(robot_qpos[:, 3:7])
    turned_rotation = np.einsum("ij,tjk->tik", WORLD_Z_HALF_TURN, robot_rotation)
    turned_quaternion = _continuous_quaternion_wxyz(
        matrix_to_quaternion_wxyz(turned_rotation)
    )
    # The opponent is a rigidly turned copy of A.  Its local Pull motion stays
    # unchanged, so rotating the root must not swap or negate any joint.
    return np.concatenate((turned_position, turned_quaternion, robot_qpos[:, 7:]), axis=-1)


@dataclass(frozen=True)
class TugOfWarReference:
    """Two opposing Pull priors and one planar-neutral preview table."""

    robot_a_qpos: np.ndarray
    robot_b_qpos: np.ndarray
    object_qpos: np.ndarray
    fps: int
    joint_names: np.ndarray

    def save_canonical_rollout(
        self,
        path: str | Path,
        *,
        provenance: Mapping[str, object] | None = None,
    ) -> Path:
        """Save channels accepted by ``viser_dual_a1_player``."""
        output = Path(path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        metadata: dict[str, object] = {
            "generator": "holosoma_retargeting.tug_of_war_reference",
            "schema_version": 2,
            "source_quaternion_convention": "wxyz",
            "canonical_quaternion_convention": "xyzw",
            "opponent_transform": "world_z_half_turn",
            "leg_pair": "diagonally_opposite",
            "opponent_source": "rotated_agent_a",
            "table_preview": "planar_neutralized_xy_heading_not_training_target",
        }
        if provenance is not None:
            metadata.update(dict(provenance))
        root_wxyz = np.stack((self.robot_a_qpos[:, 3:7], self.robot_b_qpos[:, 3:7]), axis=1)
        np.savez_compressed(
            output,
            root_pos=np.stack((self.robot_a_qpos[:, :3], self.robot_b_qpos[:, :3]), axis=1),
            root_quat_xyzw=root_wxyz[..., [1, 2, 3, 0]],
            dof_pos=np.stack((self.robot_a_qpos[:, 7:], self.robot_b_qpos[:, 7:]), axis=1),
            object_pos_w=self.object_qpos[:, :3],
            object_quat_xyzw=self.object_qpos[:, [4, 5, 6, 3]],
            fps=np.asarray(self.fps, dtype=np.int64),
            joint_names=self.joint_names,
            provenance=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
        return output


def synthesize_tug_of_war_reference(
    robot_qpos: np.ndarray,
    object_qpos: np.ndarray,
    *,
    fps: int,
    joint_names: np.ndarray,
) -> TugOfWarReference:
    """Create two opposing Pull priors at diagonally opposite table legs."""
    robot_qpos = np.asarray(robot_qpos, dtype=np.float64)
    object_qpos = np.asarray(object_qpos, dtype=np.float64)
    joint_names = np.asarray(joint_names)
    frames = len(object_qpos)
    expected_robot_shape = (frames, ROBOT_QPOS_DIM)
    if robot_qpos.shape != expected_robot_shape:
        raise ValueError(f"Expected robot trajectory to have shape {expected_robot_shape}, got {robot_qpos.shape}")
    if object_qpos.shape != (frames, 7):
        raise ValueError(f"Expected object_qpos [T, 7], got {object_qpos.shape}")
    if joint_names.shape != (ROBOT_DOF,):
        raise ValueError(f"Expected {ROBOT_DOF} joint names, got {joint_names.shape}")
    if frames < 1 or not all(
        np.isfinite(values).all() for values in (robot_qpos, object_qpos)
    ):
        raise ValueError("Reference trajectories must contain at least one finite frame")
    if not isinstance(fps, (int, np.integer)) or int(fps) <= 0:
        raise ValueError(f"FPS must be a positive integer, got {fps!r}")

    planar_a, planar_object = _remove_table_planar_motion(robot_qpos, object_qpos)
    opposing_b = _turn_robot_about_world_vertical(planar_a, planar_object)
    return TugOfWarReference(
        robot_a_qpos=planar_a,
        robot_b_qpos=opposing_b,
        object_qpos=planar_object,
        fps=int(fps),
        joint_names=joint_names.copy(),
    )


def _positive_integer_fps(fps_value: np.ndarray) -> int:
    if fps_value.size != 1:
        raise ValueError(f"FPS must be scalar, got shape {fps_value.shape}")
    fps_float = float(fps_value.reshape(()))
    if not np.isfinite(fps_float) or not fps_float.is_integer() or fps_float <= 0.0:
        raise ValueError(f"FPS must be a positive finite integer, got {fps_float!r}")
    return int(fps_float)


def load_single_pull_and_synthesize_tug_of_war_reference(
    path: str | Path,
) -> TugOfWarReference:
    """Load the original single-robot Pull motion and synthesize the preview."""
    source = Path(path).expanduser().resolve()
    with np.load(source, allow_pickle=False) as data:
        required = {"joint_pos", "object_pos_w", "object_quat_w", "fps", "joint_names"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"Single Pull reference is missing channels: {missing}")
        robot_qpos = np.asarray(data["joint_pos"], dtype=np.float64)
        object_pos = np.asarray(data["object_pos_w"], dtype=np.float64)
        object_quat = np.asarray(data["object_quat_w"], dtype=np.float64)
        joint_names = np.asarray(data["joint_names"])
        fps = _positive_integer_fps(np.asarray(data["fps"]))

    frames = len(robot_qpos)
    if robot_qpos.shape != (frames, ROBOT_QPOS_DIM):
        raise ValueError(f"Expected joint_pos [T, {ROBOT_QPOS_DIM}], got {robot_qpos.shape}")
    if object_pos.shape != (frames, 3) or object_quat.shape != (frames, 4):
        raise ValueError("Unexpected object pose channel shapes")
    object_qpos = np.concatenate((object_pos, object_quat), axis=-1)
    return synthesize_tug_of_war_reference(
        robot_qpos,
        object_qpos,
        fps=fps,
        joint_names=joint_names,
    )


def load_paired_pull_and_synthesize_tug_of_war_reference(
    path: str | Path,
) -> TugOfWarReference:
    """Load the accepted paired Pull runtime NPZ and synthesize the preview."""
    source = Path(path).expanduser().resolve()
    with np.load(source, allow_pickle=False) as data:
        required = {
            "agent_joint_pos",
            "agent_body_pos_w",
            "agent_body_quat_w",
            "object_pos_w",
            "object_quat_w",
            "fps",
            "joint_names",
            "body_names",
        }
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"Paired Pull runtime reference is missing channels: {missing}")
        joint_pos = np.asarray(data["agent_joint_pos"], dtype=np.float64)
        body_pos = np.asarray(data["agent_body_pos_w"], dtype=np.float64)
        body_quat = np.asarray(data["agent_body_quat_w"], dtype=np.float64)
        object_pos = np.asarray(data["object_pos_w"], dtype=np.float64)
        object_quat = np.asarray(data["object_quat_w"], dtype=np.float64)
        joint_names = np.asarray(data["joint_names"])
        body_names = np.asarray(data["body_names"])
        fps_value = np.asarray(data["fps"])

    fps = _positive_integer_fps(fps_value)
    pelvis_indices = np.flatnonzero(body_names == "pelvis")
    if pelvis_indices.shape != (1,):
        raise ValueError("Runtime reference must contain exactly one pelvis body")
    pelvis_index = int(pelvis_indices[0])
    frames = len(object_pos)
    if joint_pos.shape != (frames, 2, ROBOT_DOF):
        raise ValueError(f"Expected agent_joint_pos [T, 2, {ROBOT_DOF}], got {joint_pos.shape}")
    expected_body_prefix = (frames, 2, len(body_names))
    if body_pos.shape != expected_body_prefix + (3,):
        raise ValueError(f"Unexpected agent_body_pos_w shape: {body_pos.shape}")
    if body_quat.shape != expected_body_prefix + (4,):
        raise ValueError(f"Unexpected agent_body_quat_w shape: {body_quat.shape}")
    if object_pos.shape != (frames, 3) or object_quat.shape != (frames, 4):
        raise ValueError("Unexpected object pose channel shapes")

    robot_qpos = np.concatenate(
        (
            body_pos[:, :, pelvis_index],
            body_quat[:, :, pelvis_index],
            joint_pos,
        ),
        axis=-1,
    )
    object_qpos = np.concatenate((object_pos, object_quat), axis=-1)
    return synthesize_tug_of_war_reference(
        robot_qpos[:, 0],
        object_qpos,
        fps=fps,
        joint_names=joint_names,
    )


__all__ = [
    "TugOfWarReference",
    "load_paired_pull_and_synthesize_tug_of_war_reference",
    "load_single_pull_and_synthesize_tug_of_war_reference",
    "synthesize_tug_of_war_reference",
]
