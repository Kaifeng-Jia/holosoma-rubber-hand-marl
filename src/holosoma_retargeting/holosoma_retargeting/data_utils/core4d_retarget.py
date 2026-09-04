"""Prepare paired CORE4D data for two independent fixed-object solves."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from holosoma_retargeting.config_types.data_type import SMPLX_DEMO_JOINTS
from holosoma_retargeting.data_utils.core4d_adapter import Core4DPairSequence


CORE4D_A1_PALM_LANDMARKS = (
    ("L_Wrist", 20),
    ("L_Index1", 25),
    ("L_Middle1", 28),
    ("L_Pinky1", 31),
    ("R_Wrist", 21),
    ("R_Index1", 40),
    ("R_Middle1", 43),
    ("R_Pinky1", 46),
)


G1_HEIGHT_M = 1.32
_FOOT_INDICES = np.asarray(
    [SMPLX_DEMO_JOINTS.index("L_Foot"), SMPLX_DEMO_JOINTS.index("R_Foot")],
    dtype=np.int64,
)


@dataclass(frozen=True)
class Core4DPersonRetargetInput:
    """One person's scaled nominal source and the pair's shared target."""

    person_index: int
    human_height_m: float
    human_height_method: str
    human_to_robot_scale: float
    scale_anchor: np.ndarray
    ground_offset_m: float
    human_joints: np.ndarray
    human_joints_full: np.ndarray
    wrist_quat_xyzw: np.ndarray
    nominal_object_poses: np.ndarray
    physical_object_poses: np.ndarray


def build_core4d_a1_palm_orientation_targets(
    human_joints_full: np.ndarray,
    wrist_quat_xyzw: np.ndarray,
    *,
    max_calibration_error_deg: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Adapt one CORE4D person's SMPL-X hands to the existing A.1 input.

    CORE4D stores the official SMPL-X joint order but the baseline optimizer
    consumes only the first 22 body joints.  This adapter extracts only the
    eight landmarks needed to calibrate anatomical palm frames.  The A.1
    mathematics remains in the existing shared helper.
    """
    from holosoma_retargeting.src.utils import (
        build_pt_wrist_palm_orientation_targets,
    )

    full_joints = np.asarray(human_joints_full, dtype=np.float64)
    wrist_quaternions = np.asarray(wrist_quat_xyzw, dtype=np.float64)
    if full_joints.ndim != 3 or full_joints.shape[1:] != (127, 3):
        raise ValueError(
            "human_joints_full must have shape (T, 127, 3), got "
            f"{full_joints.shape}"
        )
    if wrist_quaternions.shape != (len(full_joints), 2, 4):
        raise ValueError(
            "wrist_quat_xyzw must have shape "
            f"({len(full_joints)}, 2, 4), got {wrist_quaternions.shape}"
        )
    if not np.isfinite(full_joints).all() or not np.isfinite(wrist_quaternions).all():
        raise ValueError("CORE4D A.1 inputs contain non-finite values")

    landmark_names = [name for name, _index in CORE4D_A1_PALM_LANDMARKS]
    landmark_indices = [index for _name, index in CORE4D_A1_PALM_LANDMARKS]
    landmarks = full_joints[:, landmark_indices].copy()
    return build_pt_wrist_palm_orientation_targets(
        landmarks,
        wrist_quaternions,
        landmark_names,
        max_calibration_error_deg=max_calibration_error_deg,
    )


def _validate_height(height_m: float, *, label: str) -> float:
    value = float(height_m)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{label} must be a positive finite value, got {height_m!r}")
    return value


def _scale_points_about_ground_anchor(
    points: np.ndarray,
    *,
    anchor: np.ndarray,
    scale: float,
    ground_offset_m: float,
) -> np.ndarray:
    result = np.asarray(points, dtype=np.float64).copy()
    if result.ndim != 3 or result.shape[-1] != 3:
        raise ValueError(f"points must have shape (T, J, 3), got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError("points contain non-finite values")
    result[..., 2] -= ground_offset_m
    result = anchor + scale * (result - anchor)
    return result


def prepare_core4d_person_retarget_input(
    sequence: Core4DPairSequence,
    person_index: int,
    *,
    robot_height_m: float = G1_HEIGHT_M,
    human_height_override_m: float | None = None,
    mat_height_m: float = 0.1,
) -> Core4DPersonRetargetInput:
    """Build one person's nominal scene without modifying the shared object.

    Scaling is performed about the first physical object's ground projection,
    not the arbitrary dataset world origin.  The temporary nominal object and
    the selected human use the same person-specific scale.  The returned
    physical object poses are an untouched copy of the canonical pair track.
    """
    if person_index not in (0, 1):
        raise ValueError(f"person_index must be 0 or 1, got {person_index}")
    robot_height_m = _validate_height(robot_height_m, label="robot_height_m")
    if not np.isfinite(mat_height_m) or mat_height_m < 0.0:
        raise ValueError(f"mat_height_m must be finite and non-negative, got {mat_height_m!r}")

    if human_height_override_m is None:
        heights = np.asarray(sequence.human_heights, dtype=np.float64)
        if heights.shape != (2,):
            raise ValueError(f"sequence.human_heights must have shape (2,), got {heights.shape}")
        human_height_m = _validate_height(
            heights[person_index],
            label=f"human_heights[{person_index}]",
        )
        height_method = str(sequence.height_method)
    else:
        human_height_m = _validate_height(
            human_height_override_m,
            label="human_height_override_m",
        )
        height_method = "manual_override"
    scale = robot_height_m / human_height_m

    physical_object_poses = np.asarray(sequence.object_poses, dtype=np.float64).copy()
    if physical_object_poses.ndim != 2 or physical_object_poses.shape[1] != 7:
        raise ValueError(
            "sequence.object_poses must have shape (T, 7), got "
            f"{physical_object_poses.shape}"
        )
    if not np.isfinite(physical_object_poses).all():
        raise ValueError("sequence.object_poses contains non-finite values")
    anchor = np.asarray(
        [physical_object_poses[0, 4], physical_object_poses[0, 5], 0.0],
        dtype=np.float64,
    )

    source_body = np.asarray(sequence.human_joints[:, person_index], dtype=np.float64)
    source_full = np.asarray(sequence.human_joints_full[:, person_index], dtype=np.float64)
    ground_offset_m = float(source_body[:, _FOOT_INDICES, 2].min())
    if ground_offset_m >= mat_height_m:
        ground_offset_m -= mat_height_m
    human_joints = _scale_points_about_ground_anchor(
        source_body,
        anchor=anchor,
        scale=scale,
        ground_offset_m=ground_offset_m,
    )
    human_joints_full = _scale_points_about_ground_anchor(
        source_full,
        anchor=anchor,
        scale=scale,
        ground_offset_m=ground_offset_m,
    )

    nominal_object_poses = physical_object_poses.copy()
    nominal_object_poses[:, 4:] = anchor + scale * (
        nominal_object_poses[:, 4:] - anchor
    )
    wrist_quat_xyzw = np.asarray(
        sequence.wrist_quat_xyzw[:, person_index],
        dtype=np.float64,
    ).copy()

    return Core4DPersonRetargetInput(
        person_index=person_index,
        human_height_m=human_height_m,
        human_height_method=height_method,
        human_to_robot_scale=scale,
        scale_anchor=anchor,
        ground_offset_m=ground_offset_m,
        human_joints=human_joints,
        human_joints_full=human_joints_full,
        wrist_quat_xyzw=wrist_quat_xyzw,
        nominal_object_poses=nominal_object_poses,
        physical_object_poses=physical_object_poses,
    )


def _physical_object_qpos(object_poses: np.ndarray) -> np.ndarray:
    poses = np.asarray(object_poses, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 7:
        raise ValueError(f"object_poses must have shape (T, 7), got {poses.shape}")
    return poses[:, [4, 5, 6, 0, 1, 2, 3]]


def require_shared_physical_object_qpos(
    qpos_sequences: Sequence[np.ndarray],
    physical_object_poses: np.ndarray,
) -> np.ndarray:
    """Require exact shared object qpos and return its single canonical copy."""
    if len(qpos_sequences) != 2:
        raise ValueError(f"Expected exactly two qpos sequences, got {len(qpos_sequences)}")
    expected = _physical_object_qpos(physical_object_poses)
    for person_index, qpos in enumerate(qpos_sequences):
        values = np.asarray(qpos, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] != len(expected) or values.shape[1] < 7:
            raise ValueError(
                f"person {person_index + 1} qpos must have shape (T, >=7), got {values.shape}"
            )
        if not np.array_equal(values[:, -7:], expected):
            max_error = float(np.max(np.abs(values[:, -7:] - expected)))
            raise ValueError(
                f"person {person_index + 1} object qpos diverged from the shared physical "
                f"trajectory (max abs error {max_error:.9g})"
            )
    if not np.array_equal(
        np.asarray(qpos_sequences[0])[:, -7:],
        np.asarray(qpos_sequences[1])[:, -7:],
    ):
        raise ValueError("The two retarget results contain different object qpos")
    return expected.copy()


def save_core4d_pair_reference(
    path: str | Path,
    *,
    qpos_sequences: Sequence[np.ndarray],
    sequence: Core4DPairSequence,
    person_inputs: Sequence[Core4DPersonRetargetInput],
) -> Path:
    """Save two robot references and one shared object reference without pickle."""
    if len(person_inputs) != 2:
        raise ValueError(f"Expected exactly two person inputs, got {len(person_inputs)}")
    shared_object_qpos = require_shared_physical_object_qpos(
        qpos_sequences,
        sequence.object_poses,
    )
    robot_qpos = np.stack(
        [np.asarray(qpos, dtype=np.float64)[:, :-7] for qpos in qpos_sequences],
        axis=1,
    )
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    provenance = {
        "source_sequence": sequence.provenance.get("source_sequence"),
        "person_order": ["person1", "person2"],
        "height_methods": [value.human_height_method for value in person_inputs],
        "scale_anchor_policy": "initial_object_origin_projected_to_ground",
        "object_scale": 1.0,
        "shared_object_qpos_exact": True,
    }
    np.savez_compressed(
        output,
        robot_qpos=robot_qpos,
        object_qpos=shared_object_qpos,
        fps=np.asarray(sequence.fps, dtype=np.int64),
        human_heights=np.asarray(
            [value.human_height_m for value in person_inputs],
            dtype=np.float64,
        ),
        human_to_robot_scales=np.asarray(
            [value.human_to_robot_scale for value in person_inputs],
            dtype=np.float64,
        ),
        scale_anchor=np.asarray(person_inputs[0].scale_anchor, dtype=np.float64),
        object_name=np.asarray(sequence.object_name),
        object_mesh_path=np.asarray(sequence.object_mesh_path),
        provenance_json=np.asarray(json.dumps(provenance, sort_keys=True)),
    )
    return output


__all__ = [
    "CORE4D_A1_PALM_LANDMARKS",
    "G1_HEIGHT_M",
    "Core4DPersonRetargetInput",
    "build_core4d_a1_palm_orientation_targets",
    "prepare_core4d_person_retarget_input",
    "require_shared_physical_object_qpos",
    "save_core4d_pair_reference",
]
