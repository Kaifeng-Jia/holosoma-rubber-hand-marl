"""Analyze opt-in Stage-1B table-vs-robot contact audit recordings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _yaw_xyzw(quaternion: np.ndarray) -> np.ndarray:
    quaternion = quaternion / np.maximum(
        np.linalg.norm(quaternion, axis=-1, keepdims=True),
        1.0e-12,
    )
    x, y, z, w = np.moveaxis(quaternion, -1, 0)
    return np.arctan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _rotate_xyzw(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    quaternion = quaternion / np.maximum(
        np.linalg.norm(quaternion, axis=-1, keepdims=True),
        1.0e-12,
    )
    xyz = quaternion[..., :3]
    w = quaternion[..., 3:4]
    twice_cross = 2.0 * np.cross(xyz, vector)
    return vector + w * twice_cross + np.cross(xyz, twice_cross)


def _attempts(
    motion_time_step: np.ndarray,
    done: np.ndarray,
    total_frames: int,
) -> list[dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    start = 0
    end_frame_threshold = max(total_frames - 2, 0)
    for index in range(len(motion_time_step) - 1):
        wrapped = motion_time_step[index + 1] <= motion_time_step[index]
        reset = bool(done[index])
        if not (wrapped or reset):
            continue
        reached_end = int(np.max(motion_time_step[start : index + 1])) >= end_frame_threshold
        attempts.append(
            {
                "start": start,
                "end": index,
                "completed": bool(
                    wrapped
                    and reached_end
                    and not np.any(done[start : index + 1])
                ),
            }
        )
        start = index + 1
    return attempts


def _sum_filter_forces(values: np.ndarray) -> np.ndarray:
    """Sum sensor-body/filter axes while preserving time/history and xyz."""
    if values.ndim < 3 or values.shape[-1] != 3:
        raise ValueError(f"Expected filtered contact forces ending in xyz, got {values.shape}")
    if values.ndim == 3:
        return np.sum(values, axis=1)
    if values.ndim == 4:
        return np.sum(values, axis=(1, 2))
    if values.ndim == 5:
        return np.sum(values, axis=(2, 3))
    raise ValueError(f"Unsupported filtered contact force shape: {values.shape}")


def _contact_yaw_moment(
    force_matrix_w: np.ndarray,
    contact_pos_w: np.ndarray,
    com_pos_w: np.ndarray,
) -> np.ndarray:
    if force_matrix_w.shape != contact_pos_w.shape:
        raise ValueError(
            "Filtered force and contact-position matrices must have identical shapes, "
            f"got {force_matrix_w.shape} and {contact_pos_w.shape}"
        )
    lever = contact_pos_w - com_pos_w[:, None, None, :]
    moment = np.cross(lever, force_matrix_w)
    return np.nansum(moment[..., 2], axis=(1, 2))


def _median(values: list[float]) -> float | None:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    return float(np.median(finite)) if finite.size else None


def _filter_body_names(metadata: dict[str, Any]) -> list[str]:
    contact_metadata = metadata.get("object_robot_contact")
    if not isinstance(contact_metadata, dict):
        raise ValueError("Recording metadata is missing object_robot_contact")
    expressions = contact_metadata.get("filter_prim_paths_expr")
    if not isinstance(expressions, list) or not expressions:
        raise ValueError("object_robot_contact metadata is missing filter_prim_paths_expr")
    return [str(expression).rsplit("/", 1)[-1] for expression in expressions]


def _per_filter_positive_propulsive_impulse(
    force_history_w: np.ndarray,
    direction: np.ndarray,
    sim_dt: float,
) -> np.ndarray:
    if force_history_w.ndim != 5 or force_history_w.shape[-1] != 3:
        raise ValueError(
            "Expected filtered force history shaped [time, history, sensor_body, filter, xyz], "
            f"got {force_history_w.shape}"
        )
    projected = np.einsum("thbfi,i->thbf", force_history_w, direction)
    return np.sum(np.maximum(projected, 0.0), axis=(0, 1, 2)) * sim_dt


def analyze_recording(
    recording: dict[str, np.ndarray],
    metadata: dict[str, Any],
    *,
    endpoint_yaw_limit_deg: float = 15.0,
    transient_yaw_warning_deg: float = 30.0,
    hand_dominance_fraction: float = 0.5,
    max_attempts: int | None = None,
) -> dict[str, Any]:
    required = {
        "motion_time_step",
        "done",
        "ref_object_pos_w",
        "object_pos_w",
        "ref_object_quat_xyzw",
        "object_quat_xyzw",
        "object_robot_contact_force_matrix_w",
        "object_robot_contact_force_matrix_history_w",
        "object_robot_contact_pos_w",
        "object_hand_contact_force_matrix_w",
        "object_hand_contact_force_matrix_history_w",
        "object_hand_contact_pos_w",
    }
    missing = sorted(required.difference(recording))
    if missing:
        raise ValueError(f"Recording is missing audit channels: {', '.join(missing)}")
    if "object_physics" not in metadata:
        raise ValueError("Recording metadata is missing object_physics")

    sample_count = len(recording["motion_time_step"])
    inconsistent = sorted(name for name in required if len(recording[name]) != sample_count)
    if inconsistent:
        raise ValueError(f"Audit channels with inconsistent lengths: {', '.join(inconsistent)}")

    dt = float(metadata["dt"])
    sim_dt = float(metadata["sim_dt"])
    attempts = _attempts(
        np.asarray(recording["motion_time_step"], dtype=np.int64),
        np.asarray(recording["done"], dtype=bool),
        int(metadata["motion_time_step_total"]),
    )
    if max_attempts is not None:
        attempts = attempts[:max_attempts]

    total_force = _sum_filter_forces(recording["object_robot_contact_force_matrix_w"])
    hand_force = _sum_filter_forces(recording["object_hand_contact_force_matrix_w"])
    nonhand_force = total_force - hand_force
    total_force_history = _sum_filter_forces(
        recording["object_robot_contact_force_matrix_history_w"]
    )
    hand_force_history = _sum_filter_forces(
        recording["object_hand_contact_force_matrix_history_w"]
    )
    nonhand_force_history = total_force_history - hand_force_history

    physics = metadata["object_physics"]
    robot_contact_body_names = _filter_body_names(metadata)
    robot_force_history = recording["object_robot_contact_force_matrix_history_w"]
    if robot_force_history.shape[-2] != len(robot_contact_body_names):
        raise ValueError(
            "Robot contact filter metadata does not match force history: "
            f"{len(robot_contact_body_names)} names vs {robot_force_history.shape[-2]} filters"
        )
    com_pose_b = np.asarray(physics["com_pose_b"], dtype=np.float64).reshape(-1, 7)[0]
    com_offset_b = np.broadcast_to(com_pose_b[:3], recording["object_pos_w"].shape)
    com_pos_w = recording["object_pos_w"] + _rotate_xyzw(
        recording["object_quat_xyzw"],
        com_offset_b,
    )
    total_yaw_moment = _contact_yaw_moment(
        recording["object_robot_contact_force_matrix_w"],
        recording["object_robot_contact_pos_w"],
        com_pos_w,
    )
    hand_yaw_moment = _contact_yaw_moment(
        recording["object_hand_contact_force_matrix_w"],
        recording["object_hand_contact_pos_w"],
        com_pos_w,
    )
    nonhand_yaw_moment = total_yaw_moment - hand_yaw_moment

    yaw_error = np.arctan2(
        np.sin(_yaw_xyzw(recording["object_quat_xyzw"]) - _yaw_xyzw(recording["ref_object_quat_xyzw"])),
        np.cos(_yaw_xyzw(recording["object_quat_xyzw"]) - _yaw_xyzw(recording["ref_object_quat_xyzw"])),
    )

    attempt_details: list[dict[str, Any]] = []
    endpoint_yaw_values: list[float] = []
    transient_yaw_values: list[float] = []
    hand_propulsion_fractions: list[float] = []
    for attempt_index, attempt in enumerate(attempts):
        start = int(attempt["start"])
        end = int(attempt["end"])
        attempt_slice = slice(start, end + 1)
        reference_displacement = (
            recording["ref_object_pos_w"][end] - recording["ref_object_pos_w"][start]
        )
        reference_norm = float(np.linalg.norm(reference_displacement))
        direction = (
            reference_displacement / reference_norm
            if reference_norm > 1.0e-9
            else np.zeros(3, dtype=np.float64)
        )

        def _positive_propulsive_impulse(force_history: np.ndarray) -> float:
            projected = np.einsum("thi,i->th", force_history[attempt_slice], direction)
            return float(np.sum(np.maximum(projected, 0.0)) * sim_dt)

        total_propulsive_impulse = _positive_propulsive_impulse(total_force_history)
        hand_propulsive_impulse = _positive_propulsive_impulse(hand_force_history)
        nonhand_propulsive_impulse = _positive_propulsive_impulse(nonhand_force_history)
        attributable_propulsive_impulse = hand_propulsive_impulse + nonhand_propulsive_impulse
        per_body_propulsive_impulse = _per_filter_positive_propulsive_impulse(
            robot_force_history[attempt_slice],
            direction,
            sim_dt,
        )
        top_body_indices = np.argsort(per_body_propulsive_impulse)[::-1]
        top_propulsive_contact_bodies = [
            {
                "body_name": robot_contact_body_names[int(body_index)],
                "positive_propulsive_impulse_n_s": float(per_body_propulsive_impulse[body_index]),
            }
            for body_index in top_body_indices[:5]
            if per_body_propulsive_impulse[body_index] > 1.0e-9
        ]
        hand_fraction = (
            hand_propulsive_impulse / attributable_propulsive_impulse
            if attributable_propulsive_impulse > 1.0e-9
            else 0.0
        )
        endpoint_yaw_deg = float(np.degrees(abs(yaw_error[end])))
        transient_yaw_deg = float(np.degrees(np.max(abs(yaw_error[attempt_slice]))))
        endpoint_yaw_values.append(endpoint_yaw_deg)
        transient_yaw_values.append(transient_yaw_deg)
        hand_propulsion_fractions.append(hand_fraction)
        attempt_details.append(
            {
                "attempt": attempt_index,
                **attempt,
                "length_steps": end - start + 1,
                "actual_table_displacement_m": float(
                    np.linalg.norm(recording["object_pos_w"][end] - recording["object_pos_w"][start])
                ),
                "reference_table_displacement_m": reference_norm,
                "endpoint_abs_yaw_error_deg": endpoint_yaw_deg,
                "transient_max_abs_yaw_error_deg": transient_yaw_deg,
                "endpoint_yaw_pass": endpoint_yaw_deg <= endpoint_yaw_limit_deg,
                "transient_yaw_warning": transient_yaw_deg > transient_yaw_warning_deg,
                "total_positive_propulsive_impulse_n_s": total_propulsive_impulse,
                "hand_positive_propulsive_impulse_n_s": hand_propulsive_impulse,
                "nonhand_positive_propulsive_impulse_n_s": nonhand_propulsive_impulse,
                "attributable_positive_propulsive_impulse_n_s": attributable_propulsive_impulse,
                "hand_propulsion_fraction": hand_fraction,
                "rubber_hand_propulsion_dominant": hand_fraction >= hand_dominance_fraction,
                "top_propulsive_contact_bodies": top_propulsive_contact_bodies,
                "total_yaw_moment_impulse_n_m_s": float(
                    np.sum(total_yaw_moment[attempt_slice]) * dt
                ),
                "hand_yaw_moment_impulse_n_m_s": float(
                    np.sum(hand_yaw_moment[attempt_slice]) * dt
                ),
                "nonhand_yaw_moment_impulse_n_m_s": float(
                    np.sum(nonhand_yaw_moment[attempt_slice]) * dt
                ),
            }
        )

    return {
        "schema": "holosoma.stage1b_contact_audit.v1",
        "recording": {
            "samples": sample_count,
            "dt_s": dt,
            "sim_dt_s": sim_dt,
            "attempts": len(attempts),
            "completed": sum(bool(attempt["completed"]) for attempt in attempts),
        },
        "contract": {
            "endpoint_yaw_limit_deg": endpoint_yaw_limit_deg,
            "transient_yaw_warning_deg": transient_yaw_warning_deg,
            "hand_dominance_fraction": hand_dominance_fraction,
            "incidental_nonhand_contact_allowed": True,
        },
        "object_physics": physics,
        "summary": {
            "median_endpoint_abs_yaw_error_deg": _median(endpoint_yaw_values),
            "median_transient_max_abs_yaw_error_deg": _median(transient_yaw_values),
            "endpoint_yaw_pass_rate": (
                float(np.mean(np.asarray(endpoint_yaw_values) <= endpoint_yaw_limit_deg))
                if endpoint_yaw_values
                else None
            ),
            "transient_yaw_warning_rate": (
                float(np.mean(np.asarray(transient_yaw_values) > transient_yaw_warning_deg))
                if transient_yaw_values
                else None
            ),
            "median_hand_propulsion_fraction": _median(hand_propulsion_fractions),
            "rubber_hand_dominant_attempt_rate": (
                float(np.mean(np.asarray(hand_propulsion_fractions) >= hand_dominance_fraction))
                if hand_propulsion_fractions
                else None
            ),
        },
        "attempt_details": attempt_details,
        "limitations": [
            "Filtered ContactSensor forces are PhysX aggregate forces, not a trajectory optimizer proof.",
            "Contact positions are aggregate points at the recorded sensor update, not full substep histories.",
            "A failed policy rollout cannot by itself prove that a reference is mathematically infeasible.",
        ],
    }


def load_recording(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as data:
        recording = {name: data[name] for name in data.files if name != "_metadata_json"}
        metadata = json.loads(str(data["_metadata_json"].item()))
    return recording, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--endpoint-yaw-limit-deg", type=float, default=15.0)
    parser.add_argument("--transient-yaw-warning-deg", type=float, default=30.0)
    parser.add_argument("--hand-dominance-fraction", type=float, default=0.5)
    parser.add_argument("--max-attempts", type=int)
    args = parser.parse_args()
    recording, metadata = load_recording(args.recording)
    result = analyze_recording(
        recording,
        metadata,
        endpoint_yaw_limit_deg=args.endpoint_yaw_limit_deg,
        transient_yaw_warning_deg=args.transient_yaw_warning_deg,
        hand_dominance_fraction=args.hand_dominance_fraction,
        max_attempts=args.max_attempts,
    )
    output = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n")
    print(output)


if __name__ == "__main__":
    main()
