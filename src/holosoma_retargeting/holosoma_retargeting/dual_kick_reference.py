"""Synthesize a paired cooperative-Kick reference from one physical rollout.

The frozen Kick prior is a learned single-robot rollout.  The reviewed default
constructs robot B as a full sagittal mirror of robot A across the table-local
XY plane: root position/orientation, left/right joint permutation, and joint
signs are all transformed together.  A legacy ``same_action`` layout remains
available only for explicit comparison with the rejected first preview.

The aggregate evaluation recorder does not reset ``episode_step`` after every
successful clip.  Attempt extraction consequently uses decreases/wraps in
``motion_time_step`` and validates the selected segment before synthesis.

All internal qpos quaternions are WXYZ.  The source evaluation and the
canonical ViSER artifact use explicitly named XYZW channels at their file
boundaries.
"""

from __future__ import annotations

import hashlib
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
from holosoma_retargeting.dual_pull_reference import (
    G1_29DOF_JOINT_NAMES,
    G1_LOCAL_SAGITTAL_REFLECTION,
)


DEFAULT_KICK_SOURCE_ATTEMPT = 9
DEFAULT_KICK_SOURCE_CONTACT_LOCAL_Z_M = -0.23525
DEFAULT_KICK_TARGET_CONTACT_HALF_SPAN_M = 0.6742736
DEFAULT_KICK_LAYOUT = "mirrored"
KICK_LAYOUTS = frozenset({"mirrored", "same_action"})

ROBOT_QPOS_DIM = 7 + len(G1_29DOF_JOINT_NAMES)
OBJECT_QPOS_DIM = 7

# Kick progress is table-local X, table-local Y is vertical, and the two
# cooperating contact lanes lie along table-local Z.
KICK_LOCAL_XY_REFLECTION = np.diag([1.0, 1.0, -1.0])


def _json_ready(value: object) -> object:
    """Convert common NumPy values to JSON-compatible Python values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        reference = np.asarray(first_reference, dtype=np.float64)
        reference /= np.linalg.norm(reference)
        if np.dot(result[0], reference) < 0.0:
            result[0] *= -1.0
    for frame in range(1, len(result)):
        if np.dot(result[frame - 1], result[frame]) < 0.0:
            result[frame] *= -1.0
    return result


def _rotation_geodesic_midpoint(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return shortest-arc SO(3) midpoints for two rotation trajectories."""
    first_quat = matrix_to_quaternion_wxyz(first)
    second_quat = matrix_to_quaternion_wxyz(second)
    alignment = np.where(
        np.sum(first_quat * second_quat, axis=-1, keepdims=True) < 0.0,
        -1.0,
        1.0,
    )
    midpoint = first_quat + alignment * second_quat
    norms = np.linalg.norm(midpoint, axis=-1, keepdims=True)
    if np.any(norms <= 1.0e-12):
        raise ValueError("Object rotation and its reflection have an ambiguous midpoint")
    midpoint /= norms
    return quaternion_wxyz_to_matrix(midpoint)


def _symmetrize_object_qpos(object_qpos: np.ndarray) -> np.ndarray:
    """Remove table-local-Z drift and relative yaw by reflection averaging."""
    object_pos = object_qpos[:, :3]
    object_quat = object_qpos[:, 3:7]
    object_rotation = quaternion_wxyz_to_matrix(object_quat)
    initial_pos = object_pos[0]
    initial_rotation = object_rotation[0]

    relative_pos = np.einsum("ij,tj->ti", initial_rotation.T, object_pos - initial_pos)
    mirrored_relative_pos = np.einsum(
        "ij,tj->ti",
        KICK_LOCAL_XY_REFLECTION,
        relative_pos,
    )
    symmetric_relative_pos = 0.5 * (relative_pos + mirrored_relative_pos)
    shared_pos = initial_pos + np.einsum("ij,tj->ti", initial_rotation, symmetric_relative_pos)

    relative_rotation = np.einsum("ij,tjk->tik", initial_rotation.T, object_rotation)
    mirrored_relative_rotation = np.einsum(
        "ij,tjk,kl->til",
        KICK_LOCAL_XY_REFLECTION,
        relative_rotation,
        KICK_LOCAL_XY_REFLECTION,
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


def _transport_robot_to_shared_object(
    robot_qpos: np.ndarray,
    source_object_qpos: np.ndarray,
    shared_object_qpos: np.ndarray,
) -> np.ndarray:
    """Preserve the learned robot pose in the table frame after symmetrization."""
    source_object_rotation = quaternion_wxyz_to_matrix(source_object_qpos[:, 3:7])
    shared_object_rotation = quaternion_wxyz_to_matrix(shared_object_qpos[:, 3:7])
    source_robot_rotation = quaternion_wxyz_to_matrix(robot_qpos[:, 3:7])
    rotation_delta = np.einsum(
        "tij,tkj->tik",
        shared_object_rotation,
        source_object_rotation,
    )
    relative_pos = robot_qpos[:, :3] - source_object_qpos[:, :3]
    transported_pos = shared_object_qpos[:, :3] + np.einsum(
        "tij,tj->ti",
        rotation_delta,
        relative_pos,
    )
    transported_rotation = np.einsum("tij,tjk->tik", rotation_delta, source_robot_rotation)
    transported_quat = _continuous_quaternion_wxyz(
        matrix_to_quaternion_wxyz(transported_rotation),
        first_reference=robot_qpos[0, 3:7],
    )
    return np.concatenate((transported_pos, transported_quat, robot_qpos[:, 7:]), axis=-1)


def _mirror_robot_across_shared_table(
    robot_a_qpos: np.ndarray,
    shared_object_qpos: np.ndarray,
) -> np.ndarray:
    """Return a physical G1 mirror across the shared table-local XY plane."""
    shared_pos = shared_object_qpos[:, :3]
    shared_rotation = quaternion_wxyz_to_matrix(shared_object_qpos[:, 3:7])
    robot_a_rotation = quaternion_wxyz_to_matrix(robot_a_qpos[:, 3:7])
    world_reflection = np.einsum(
        "tij,jk,tlk->til",
        shared_rotation,
        KICK_LOCAL_XY_REFLECTION,
        shared_rotation,
    )
    robot_b_pos = shared_pos + np.einsum(
        "tij,tj->ti",
        world_reflection,
        robot_a_qpos[:, :3] - shared_pos,
    )
    # Reflection on both sides keeps the mapped root orientation in SO(3).
    robot_b_rotation = np.einsum(
        "tij,tjk,kl->til",
        world_reflection,
        robot_a_rotation,
        G1_LOCAL_SAGITTAL_REFLECTION,
    )
    if not np.allclose(np.linalg.det(robot_b_rotation), 1.0, atol=1.0e-8):
        raise ValueError("Mirrored Kick root orientation is not a proper rotation")
    robot_b_quat = _continuous_quaternion_wxyz(
        matrix_to_quaternion_wxyz(robot_b_rotation)
    )
    index_map, sign_mask = g1_sagittal_joint_map(G1_29DOF_JOINT_NAMES)
    robot_b_joints = robot_a_qpos[:, 7:][:, index_map] * sign_mask
    return np.concatenate((robot_b_pos, robot_b_quat, robot_b_joints), axis=-1)


def _attempt_slices(motion_time_step: np.ndarray) -> tuple[slice, ...]:
    """Split an aggregate evaluation by motion-phase wraps, not episode_step."""
    steps = np.asarray(motion_time_step)
    if steps.ndim != 1 or len(steps) < 1:
        raise ValueError(f"motion_time_step must be a non-empty vector, got {steps.shape}")
    if not np.issubdtype(steps.dtype, np.integer):
        if not np.isfinite(steps).all() or not np.equal(steps, np.round(steps)).all():
            raise ValueError("motion_time_step must contain finite integer values")
        steps = steps.astype(np.int64)
    starts = np.concatenate((np.asarray([0]), np.flatnonzero(steps[1:] <= steps[:-1]) + 1))
    stops = np.concatenate((starts[1:], np.asarray([len(steps)])))
    slices = tuple(slice(int(start), int(stop)) for start, stop in zip(starts, stops, strict=True))
    for attempt_slice in slices:
        segment = steps[attempt_slice]
        if len(segment) > 1 and not np.equal(np.diff(segment), 1).all():
            raise ValueError(
                "motion_time_step is not contiguous inside attempt slice "
                f"[{attempt_slice.start}:{attempt_slice.stop}]"
            )
    return slices


@dataclass(frozen=True)
class DualKickReference:
    """Canonical qpos contract for two identical Kick priors and one table."""

    robot_a_qpos: np.ndarray
    robot_b_qpos: np.ndarray
    shared_object_qpos: np.ndarray
    fps: int
    layout: str
    source_attempt: int
    source_contact_local_z_m: float
    target_contact_half_span_m: float
    robot_a_lateral_offset_m: float
    robot_b_lateral_offset_m: float
    provenance: Mapping[str, object]

    def save(
        self,
        path: str | Path,
        *,
        provenance: Mapping[str, object] | None = None,
    ) -> Path:
        """Save the explicit A/B/shared-table qpos contract."""
        output = Path(path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = self._provenance_payload(provenance)
        np.savez_compressed(
            output,
            robot_a_qpos=self.robot_a_qpos,
            robot_b_qpos=self.robot_b_qpos,
            shared_object_qpos=self.shared_object_qpos,
            fps=np.asarray(self.fps, dtype=np.int64),
            joint_names=G1_29DOF_JOINT_NAMES,
            layout=np.asarray(self.layout),
            source_attempt=np.asarray(self.source_attempt, dtype=np.int64),
            source_contact_local_z_m=np.asarray(self.source_contact_local_z_m),
            target_contact_half_span_m=np.asarray(self.target_contact_half_span_m),
            robot_a_lateral_offset_m=np.asarray(self.robot_a_lateral_offset_m),
            robot_b_lateral_offset_m=np.asarray(self.robot_b_lateral_offset_m),
            quaternion_convention=np.asarray("wxyz"),
            lateral_axis=np.asarray("table_local_z"),
            provenance=np.asarray(json.dumps(_json_ready(payload), sort_keys=True)),
        )
        return output

    def save_canonical_rollout(
        self,
        path: str | Path,
        *,
        provenance: Mapping[str, object] | None = None,
    ) -> Path:
        """Save the seven-channel canonical artifact consumed by the ViSER player."""
        output = Path(path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        root_wxyz = np.stack((self.robot_a_qpos[:, 3:7], self.robot_b_qpos[:, 3:7]), axis=1)
        np.savez_compressed(
            output,
            root_pos=np.stack((self.robot_a_qpos[:, :3], self.robot_b_qpos[:, :3]), axis=1),
            root_quat_xyzw=root_wxyz[..., [1, 2, 3, 0]],
            dof_pos=np.stack((self.robot_a_qpos[:, 7:], self.robot_b_qpos[:, 7:]), axis=1),
            object_pos_w=self.shared_object_qpos[:, :3],
            object_quat_xyzw=self.shared_object_qpos[:, [4, 5, 6, 3]],
            fps=np.asarray(self.fps, dtype=np.int64),
            provenance=np.asarray(
                json.dumps(_json_ready(self._provenance_payload(provenance)), sort_keys=True)
            ),
        )
        return output

    def _provenance_payload(
        self,
        extra: Mapping[str, object] | None,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "generator": "holosoma_retargeting.dual_kick_reference",
            "layout": self.layout,
            "source_attempt": self.source_attempt,
            "source_contact_local_z_m": self.source_contact_local_z_m,
            "target_contact_half_span_m": self.target_contact_half_span_m,
            "robot_a_lateral_offset_m": self.robot_a_lateral_offset_m,
            "robot_b_lateral_offset_m": self.robot_b_lateral_offset_m,
            "agent_spacing_m": 2.0 * self.target_contact_half_span_m,
            "source_quaternion_convention": "xyzw",
            "qpos_quaternion_convention": "wxyz",
            "canonical_quaternion_convention": "xyzw",
            "lateral_axis": "table_local_z",
            "agent_joint_mirroring": self.layout == "mirrored",
            "mirror_plane": "table_local_xy" if self.layout == "mirrored" else None,
            "object_symmetrization": "initial_table_frame_local_z_reflection_geodesic_midpoint",
        }
        payload.update(dict(self.provenance))
        if extra is not None:
            payload.update(dict(extra))
        return payload


def _validate_motion_channels(
    root_pos: np.ndarray,
    root_quat_xyzw: np.ndarray,
    dof_pos: np.ndarray,
    object_pos_w: np.ndarray,
    object_quat_xyzw: np.ndarray,
    *,
    fps: int,
) -> None:
    frame_count = len(root_pos)
    expected = {
        "root_pos": (frame_count, 3),
        "root_quat_xyzw": (frame_count, 4),
        "dof_pos": (frame_count, len(G1_29DOF_JOINT_NAMES)),
        "object_pos_w": (frame_count, 3),
        "object_quat_xyzw": (frame_count, 4),
    }
    actual = {
        "root_pos": root_pos.shape,
        "root_quat_xyzw": root_quat_xyzw.shape,
        "dof_pos": dof_pos.shape,
        "object_pos_w": object_pos_w.shape,
        "object_quat_xyzw": object_quat_xyzw.shape,
    }
    if frame_count < 1:
        raise ValueError("Kick source must contain at least one frame")
    mismatched = {name: (actual[name], shape) for name, shape in expected.items() if actual[name] != shape}
    if mismatched:
        raise ValueError(f"Kick source channel shapes do not match: {mismatched}")
    if not all(np.isfinite(channel).all() for channel in (
        root_pos,
        root_quat_xyzw,
        dof_pos,
        object_pos_w,
        object_quat_xyzw,
    )):
        raise ValueError("Kick source channels must contain only finite values")
    if not isinstance(fps, (int, np.integer)) or int(fps) <= 0:
        raise ValueError(f"FPS must be a positive integer, got {fps!r}")


def synthesize_dual_kick_reference(
    root_pos: np.ndarray,
    root_quat_xyzw: np.ndarray,
    dof_pos: np.ndarray,
    object_pos_w: np.ndarray,
    object_quat_xyzw: np.ndarray,
    *,
    fps: int,
    layout: str = DEFAULT_KICK_LAYOUT,
    source_attempt: int = DEFAULT_KICK_SOURCE_ATTEMPT,
    source_contact_local_z_m: float = DEFAULT_KICK_SOURCE_CONTACT_LOCAL_Z_M,
    target_contact_half_span_m: float = DEFAULT_KICK_TARGET_CONTACT_HALF_SPAN_M,
    provenance: Mapping[str, object] | None = None,
) -> DualKickReference:
    """Build the paired Kick qpos trajectories from one selected rollout."""
    root_pos = np.asarray(root_pos, dtype=np.float64)
    root_quat_xyzw = np.asarray(root_quat_xyzw, dtype=np.float64)
    dof_pos = np.asarray(dof_pos, dtype=np.float64)
    object_pos_w = np.asarray(object_pos_w, dtype=np.float64)
    object_quat_xyzw = np.asarray(object_quat_xyzw, dtype=np.float64)
    _validate_motion_channels(
        root_pos,
        root_quat_xyzw,
        dof_pos,
        object_pos_w,
        object_quat_xyzw,
        fps=fps,
    )
    source_attempt = int(source_attempt)
    layout = str(layout)
    source_contact_local_z_m = float(source_contact_local_z_m)
    target_contact_half_span_m = float(target_contact_half_span_m)
    if not np.isfinite(source_contact_local_z_m):
        raise ValueError("source_contact_local_z_m must be finite")
    if not np.isfinite(target_contact_half_span_m) or target_contact_half_span_m <= 0.0:
        raise ValueError("target_contact_half_span_m must be finite and positive")
    if layout not in KICK_LAYOUTS:
        raise ValueError(f"layout must be one of {sorted(KICK_LAYOUTS)}, got {layout!r}")

    root_quat_wxyz = _continuous_quaternion_wxyz(root_quat_xyzw[:, [3, 0, 1, 2]])
    object_quat_wxyz = _continuous_quaternion_wxyz(object_quat_xyzw[:, [3, 0, 1, 2]])
    source_robot_qpos = np.concatenate((root_pos, root_quat_wxyz, dof_pos), axis=-1)
    source_object_qpos = np.concatenate((object_pos_w, object_quat_wxyz), axis=-1)
    shared_object_qpos = _symmetrize_object_qpos(source_object_qpos)
    transported_robot_qpos = _transport_robot_to_shared_object(
        source_robot_qpos,
        source_object_qpos,
        shared_object_qpos,
    )

    if layout == "mirrored":
        if source_contact_local_z_m >= 0.0:
            raise ValueError("mirrored layout requires a negative source contact local-Z")
        outward_offset = target_contact_half_span_m - abs(source_contact_local_z_m)
        if outward_offset < 0.0:
            raise ValueError(
                "target_contact_half_span_m must not be smaller than the source contact span"
            )
        robot_a_base = transported_robot_qpos
        robot_b_base = _mirror_robot_across_shared_table(
            transported_robot_qpos,
            shared_object_qpos,
        )
        robot_a_offset = -outward_offset
        robot_b_offset = outward_offset
    else:
        robot_a_base = transported_robot_qpos
        robot_b_base = transported_robot_qpos
        robot_a_offset = -target_contact_half_span_m - source_contact_local_z_m
        robot_b_offset = target_contact_half_span_m - source_contact_local_z_m
    shared_rotation = quaternion_wxyz_to_matrix(shared_object_qpos[:, 3:7])
    shared_local_z_w = shared_rotation[..., :, 2]
    robot_a_qpos = robot_a_base.copy()
    robot_b_qpos = robot_b_base.copy()
    robot_a_qpos[:, :3] += robot_a_offset * shared_local_z_w
    robot_b_qpos[:, :3] += robot_b_offset * shared_local_z_w

    return DualKickReference(
        robot_a_qpos=robot_a_qpos,
        robot_b_qpos=robot_b_qpos,
        shared_object_qpos=shared_object_qpos,
        fps=int(fps),
        layout=layout,
        source_attempt=source_attempt,
        source_contact_local_z_m=source_contact_local_z_m,
        target_contact_half_span_m=target_contact_half_span_m,
        robot_a_lateral_offset_m=robot_a_offset,
        robot_b_lateral_offset_m=robot_b_offset,
        provenance={} if provenance is None else dict(provenance),
    )


def synthesize_dual_kick_reference_file(
    source_path: str | Path,
    *,
    source_attempt: int = DEFAULT_KICK_SOURCE_ATTEMPT,
    layout: str = DEFAULT_KICK_LAYOUT,
    source_contact_local_z_m: float = DEFAULT_KICK_SOURCE_CONTACT_LOCAL_Z_M,
    target_contact_half_span_m: float = DEFAULT_KICK_TARGET_CONTACT_HALF_SPAN_M,
) -> DualKickReference:
    """Extract one attempt from an aggregate eval NPZ and synthesize the pair."""
    source = Path(source_path).expanduser().resolve()
    required = {
        "root_pos",
        "root_quat_xyzw",
        "dof_pos",
        "object_pos_w",
        "object_quat_xyzw",
        "motion_time_step",
        "terminated",
        "_metadata_json",
    }
    with np.load(source, allow_pickle=False) as data:
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"Kick aggregate is missing required channels: {sorted(missing)}")
        metadata_raw = np.asarray(data["_metadata_json"])
        if metadata_raw.size != 1:
            raise ValueError(f"Kick aggregate metadata must be scalar, got {metadata_raw.shape}")
        metadata = json.loads(str(metadata_raw.reshape(())))
        if not isinstance(metadata, dict):
            raise ValueError("Kick aggregate metadata must decode to an object")
        slices = _attempt_slices(np.asarray(data["motion_time_step"]))
        source_attempt = int(source_attempt)
        if source_attempt < 0 or source_attempt >= len(slices):
            raise ValueError(
                f"source_attempt {source_attempt} is outside available range [0, {len(slices) - 1}]"
            )
        selected = slices[source_attempt]
        selected_steps = np.asarray(data["motion_time_step"])[selected]
        terminated = np.asarray(data["terminated"], dtype=bool)[selected]
        channels = {
            name: np.asarray(data[name])[selected]
            for name in (
                "root_pos",
                "root_quat_xyzw",
                "dof_pos",
                "object_pos_w",
                "object_quat_xyzw",
            )
        }

    fps_value = metadata.get("fps", metadata.get("motion_fps"))
    if fps_value is None:
        raise ValueError("Kick aggregate metadata is missing fps/motion_fps")
    fps_float = float(fps_value)
    if not np.isfinite(fps_float) or not fps_float.is_integer() or fps_float <= 0.0:
        raise ValueError(f"Kick aggregate FPS must be a positive integer, got {fps_value!r}")
    joint_names = np.asarray(metadata.get("dof_names", []), dtype=str)
    if not np.array_equal(joint_names, G1_29DOF_JOINT_NAMES):
        raise ValueError("Kick aggregate dof_names do not match the frozen G1 29-DoF order")

    total_steps = metadata.get("motion_time_step_total")
    completed = (
        total_steps is not None
        and int(selected_steps[-1]) >= int(total_steps) - 1
        and not bool(terminated[-1])
    )
    source_provenance: dict[str, object] = {
        "source_path": source.name,
        "source_sha256": _sha256_file(source),
        "source_attempt": source_attempt,
        "source_attempt_start": int(selected.start),
        "source_attempt_end_exclusive": int(selected.stop),
        "source_attempt_end_inclusive": int(selected.stop - 1),
        "source_attempt_length": int(selected.stop - selected.start),
        "source_attempt_completed": bool(completed),
        "source_attempt_terminated": bool(terminated[-1]),
        "source_motion_start_step": int(selected_steps[0]),
        "source_motion_end_step": int(selected_steps[-1]),
    }
    return synthesize_dual_kick_reference(
        channels["root_pos"],
        channels["root_quat_xyzw"],
        channels["dof_pos"],
        channels["object_pos_w"],
        channels["object_quat_xyzw"],
        fps=int(fps_float),
        layout=layout,
        source_attempt=source_attempt,
        source_contact_local_z_m=source_contact_local_z_m,
        target_contact_half_span_m=target_contact_half_span_m,
        provenance=source_provenance,
    )


__all__ = [
    "DEFAULT_KICK_SOURCE_ATTEMPT",
    "DEFAULT_KICK_LAYOUT",
    "DEFAULT_KICK_SOURCE_CONTACT_LOCAL_Z_M",
    "DEFAULT_KICK_TARGET_CONTACT_HALF_SPAN_M",
    "DualKickReference",
    "G1_29DOF_JOINT_NAMES",
    "KICK_LOCAL_XY_REFLECTION",
    "KICK_LAYOUTS",
    "synthesize_dual_kick_reference",
    "synthesize_dual_kick_reference_file",
]
