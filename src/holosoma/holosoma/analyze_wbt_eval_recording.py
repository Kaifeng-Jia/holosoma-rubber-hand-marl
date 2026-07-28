"""Summarize a recorded WBT rollout into physics-based evaluation metrics.

The input is produced by :class:`EvalRecordingCallback`.  This module uses
only NumPy so that recordings can be analyzed without starting Isaac Sim.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def quaternion_angle_error(q_ref: np.ndarray, q_actual: np.ndarray) -> np.ndarray:
    """Return the shortest xyzw quaternion angle error in radians."""
    q_ref = q_ref / np.maximum(np.linalg.norm(q_ref, axis=-1, keepdims=True), 1.0e-12)
    q_actual = q_actual / np.maximum(np.linalg.norm(q_actual, axis=-1, keepdims=True), 1.0e-12)
    dot = np.abs(np.sum(q_ref * q_actual, axis=-1))
    return 2.0 * np.arccos(np.clip(dot, 0.0, 1.0))


def _rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def _mean_or_none(values: list[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def _find_attempts(
    motion_time_step: np.ndarray,
    done: np.ndarray,
    motion_time_step_total: int,
) -> tuple[list[dict[str, Any]], int]:
    """Split a looping WBT rollout into closed motion attempts.

    MotionCommand loops clips internally without emitting ``done``.  A
    successful attempt is therefore a high-frame-to-low-frame wrap with no
    termination.  A reset caused by bad tracking also wraps, but the
    transition carries ``done=True``.
    """
    time_step: np.ndarray = np.asarray(motion_time_step, dtype=np.int64).reshape(-1)
    done = np.asarray(done, dtype=bool).reshape(-1)
    if time_step.shape != done.shape:
        raise ValueError("motion_time_step and done must have the same number of samples")

    attempts: list[dict[str, Any]] = []
    start = 0
    end_frame_threshold = max(motion_time_step_total - 2, 0)
    for index in range(len(time_step) - 1):
        wrapped = time_step[index + 1] <= time_step[index]
        reset = bool(done[index])
        if not (wrapped or reset):
            continue

        reached_end = int(np.max(time_step[start : index + 1])) >= end_frame_threshold
        attempts.append(
            {
                "start": start,
                "end": index,
                "length_steps": index - start + 1,
                "completed": bool(wrapped and reached_end and not np.any(done[start : index + 1])),
                "terminated": bool(np.any(done[start : index + 1])),
                "reached_end": bool(reached_end),
            }
        )
        start = index + 1

    trailing_steps = len(time_step) - start
    return attempts, trailing_steps


def _contact_magnitudes(
    recording: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> tuple[np.ndarray, list[str]]:
    if "contact_sensor_forces_history_w" in recording:
        history = recording["contact_sensor_forces_history_w"]
        return (
            np.max(np.linalg.norm(history, axis=-1), axis=1),
            list(metadata["contact_sensor_body_names"]),
        )
    if "contact_forces_history_w" in recording:
        history = recording["contact_forces_history_w"]
        return np.max(np.linalg.norm(history, axis=-1), axis=1), list(metadata["body_names"])
    return np.linalg.norm(recording["contact_forces_w"], axis=-1), list(metadata["body_names"])


def _body_indices(body_names: list[str], names: tuple[str, ...]) -> list[int]:
    return [body_names.index(name) for name in names if name in body_names]


def _preferred_body_index(body_names: list[str], names: tuple[str, ...]) -> list[int]:
    for name in names:
        if name in body_names:
            return [body_names.index(name)]
    return []


def summarize_recording(
    recording: dict[str, np.ndarray],
    metadata: dict[str, Any],
    *,
    contact_force_threshold_n: float = 5.0,
    min_table_displacement_m: float = 0.01,
    object_motion_speed_threshold_m_s: float = 0.02,
    fall_height_drop_m: float = 0.30,
    fall_tilt_deg: float = 60.0,
    max_attempts: int | None = None,
) -> dict[str, Any]:
    """Compute the frozen Step-1 metrics for one recorded environment."""
    required = {
        "motion_time_step",
        "done",
        "timeout",
        "ref_joint_pos",
        "pre_dof_pos",
        "ref_body_pos_w",
        "pre_tracked_body_pos_w",
        "ref_body_quat_xyzw",
        "pre_tracked_body_quat_xyzw",
        "pre_root_pos",
        "pre_root_quat_xyzw",
        "contact_forces_w",
        "actions",
        "torques",
    }
    missing = sorted(required.difference(recording))
    if missing:
        raise ValueError(f"Recording is missing required channels: {', '.join(missing)}")

    sample_count = int(len(recording["motion_time_step"]))
    inconsistent = sorted(name for name in required if len(recording[name]) != sample_count)
    if inconsistent:
        raise ValueError(f"Channels with inconsistent sample counts: {', '.join(inconsistent)}")

    motion_time_step_total = int(metadata["motion_time_step_total"])
    dt = float(metadata["dt"])
    done: np.ndarray = recording["done"].astype(bool).reshape(-1)
    timeout: np.ndarray = recording["timeout"].astype(bool).reshape(-1)
    detected_attempts, trailing_steps = _find_attempts(
        recording["motion_time_step"],
        done,
        motion_time_step_total,
    )
    attempts = detected_attempts[:max_attempts] if max_attempts is not None else detected_attempts
    evaluated_sample_count = int(attempts[-1]["end"]) + 1 if attempts else sample_count
    evaluated_slice = slice(0, evaluated_sample_count)

    joint_error = (
        recording["pre_dof_pos"][evaluated_slice] - recording["ref_joint_pos"][evaluated_slice]
    )
    body_position_error = (
        recording["pre_tracked_body_pos_w"][evaluated_slice]
        - recording["ref_body_pos_w"][evaluated_slice]
    )
    body_orientation_error = quaternion_angle_error(
        recording["ref_body_quat_xyzw"][evaluated_slice],
        recording["pre_tracked_body_quat_xyzw"][evaluated_slice],
    )

    contact_magnitude, contact_body_names = _contact_magnitudes(recording, metadata)
    left_hand_indices = _preferred_body_index(
        contact_body_names,
        ("left_rubber_hand_link", "left_wrist_yaw_link"),
    )
    right_hand_indices = _preferred_body_index(
        contact_body_names,
        ("right_rubber_hand_link", "right_wrist_yaw_link"),
    )
    allowed_contact_indices = set(
        _body_indices(
            contact_body_names,
            (
                "left_foot_contact_point",
                "right_foot_contact_point",
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_rubber_hand_link",
                "right_rubber_hand_link",
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
            ),
        )
    )
    invalid_contact_indices = [
        index for index in range(len(contact_body_names)) if index not in allowed_contact_indices
    ]

    def _any_contact(indices: list[int]) -> np.ndarray:
        if not indices:
            return np.zeros(sample_count, dtype=bool)
        return np.any(contact_magnitude[:, indices] > contact_force_threshold_n, axis=1)

    left_contact = _any_contact(left_hand_indices)
    right_contact = _any_contact(right_hand_indices)
    either_hand_contact = left_contact | right_contact
    invalid_contact = _any_contact(invalid_contact_indices)

    root_height_reference = (
        recording["ref_root_pos_w"][:, 2]
        if "ref_root_pos_w" in recording
        else np.full(sample_count, recording["pre_root_pos"][0, 2])
    )
    root_height_drop = root_height_reference - recording["pre_root_pos"][:, 2]
    root_quat = recording["pre_root_quat_xyzw"]
    root_up_z = 1.0 - 2.0 * (np.square(root_quat[:, 0]) + np.square(root_quat[:, 1]))
    fall_proxy = (root_height_drop > fall_height_drop_m) | (root_up_z < math.cos(math.radians(fall_tilt_deg)))

    completed_count = sum(attempt["completed"] for attempt in attempts)
    terminated_count = sum(attempt["terminated"] for attempt in attempts)
    timeout_count = 0
    attempt_details: list[dict[str, Any]] = []
    contact_successes = 0
    interaction_proxy_successes = 0
    correct_displacement_count = 0
    displacement_evaluable_count = 0
    direction_cosines: list[float] = []
    actual_displacements: list[float] = []
    reference_displacements: list[float] = []
    first_contact_times: list[float] = []

    has_object = all(
        name in recording
        for name in (
            "ref_object_pos_w",
            "ref_object_quat_xyzw",
            "object_pos_w",
            "object_quat_xyzw",
        )
    )
    hand_object_motion_overlap: np.ndarray = np.zeros(sample_count, dtype=bool)
    if has_object and "object_lin_vel_w" in recording:
        object_is_moving = (
            np.linalg.norm(recording["object_lin_vel_w"], axis=-1) > object_motion_speed_threshold_m_s
        )
        hand_object_motion_overlap = either_hand_contact & object_is_moving

    for attempt_index, attempt in enumerate(attempts):
        start = int(attempt["start"])
        end = int(attempt["end"])
        attempt_slice = slice(start, end + 1)
        contact_indices = np.flatnonzero(either_hand_contact[attempt_slice])
        contact_success = bool(contact_indices.size)
        contact_successes += int(contact_success)
        first_contact_s = float(contact_indices[0] * dt) if contact_success else None
        if first_contact_s is not None:
            first_contact_times.append(first_contact_s)

        attempt_timeout = bool(np.any(timeout[attempt_slice]))
        timeout_count += int(attempt_timeout)
        details: dict[str, Any] = {
            "attempt": attempt_index,
            **attempt,
            "duration_s": float(attempt["length_steps"] * dt),
            "contact_success": contact_success,
            "first_contact_s": first_contact_s,
            "hand_contact_fraction": float(np.mean(either_hand_contact[attempt_slice])),
            "hand_object_motion_overlap_fraction": float(
                np.mean(hand_object_motion_overlap[attempt_slice])
            ),
            "invalid_contact_fraction": float(np.mean(invalid_contact[attempt_slice])),
            "fall_proxy": bool(np.any(fall_proxy[attempt_slice])),
            "timeout": attempt_timeout,
        }

        if has_object:
            actual_displacement = recording["object_pos_w"][end] - recording["object_pos_w"][start]
            reference_displacement = (
                recording["ref_object_pos_w"][end] - recording["ref_object_pos_w"][start]
            )
            actual_norm = float(np.linalg.norm(actual_displacement))
            reference_norm = float(np.linalg.norm(reference_displacement))
            actual_displacements.append(actual_norm)
            reference_displacements.append(reference_norm)
            direction_cosine = None
            correct_direction = False
            if reference_norm >= min_table_displacement_m:
                displacement_evaluable_count += 1
                if actual_norm >= min_table_displacement_m:
                    direction_cosine = float(
                        np.dot(actual_displacement, reference_displacement) / (actual_norm * reference_norm)
                    )
                    direction_cosines.append(direction_cosine)
                    correct_direction = direction_cosine > 0.0
                correct_displacement_count += int(correct_direction)

            details.update(
                {
                    "actual_table_displacement_m": actual_norm,
                    "reference_table_displacement_m": reference_norm,
                    "table_displacement_direction_cosine": direction_cosine,
                    "correct_table_displacement_direction": correct_direction,
                }
            )
            interaction_proxy_success = bool(
                np.any(hand_object_motion_overlap[attempt_slice]) and correct_direction
            )
            interaction_proxy_successes += int(interaction_proxy_success)
            details["interaction_proxy_success"] = interaction_proxy_success
        attempt_details.append(details)

    effort_limits: np.ndarray = np.asarray(metadata["effort_limits"], dtype=np.float64)
    torque_utilization = np.abs(recording["torques"][evaluated_slice]) / np.maximum(
        effort_limits,
        1.0e-12,
    )
    action_delta = np.diff(recording["actions"][evaluated_slice], axis=0)

    summary: dict[str, Any] = {
        "schema": "holosoma.wbt_physics_eval.v1",
        "recording": {
            "sample_count": sample_count,
            "evaluated_sample_count": evaluated_sample_count,
            "dt_s": dt,
            "duration_s": evaluated_sample_count * dt,
            "motion_frames": motion_time_step_total,
            "motion_fps": int(metadata["motion_fps"]),
            "trailing_unclosed_steps": trailing_steps,
        },
        "attempts": {
            "detected_closed_total": len(detected_attempts),
            "closed": len(attempts),
            "completed": completed_count,
            "completion_rate": completed_count / len(attempts) if attempts else None,
            "early_terminated": terminated_count - timeout_count,
            "early_termination_rate": (
                (terminated_count - timeout_count) / len(attempts) if attempts else None
            ),
            "timed_out": timeout_count,
            "mean_length_steps": (
                float(np.mean([attempt["length_steps"] for attempt in attempts])) if attempts else None
            ),
        },
        "tracking": {
            "joint_position_rmse_rad": _rmse(joint_error),
            "joint_position_mae_rad": float(np.mean(np.abs(joint_error))),
            "key_body_position_rmse_m": _rmse(body_position_error),
            "key_body_orientation_mean_rad": float(np.mean(body_orientation_error)),
            "key_body_orientation_mean_deg": float(np.degrees(np.mean(body_orientation_error))),
        },
        "contact": {
            "force_threshold_n": contact_force_threshold_n,
            "proxy_semantics": metadata.get("contact_force_semantics"),
            "left_contact_bodies": [contact_body_names[index] for index in left_hand_indices],
            "right_contact_bodies": [contact_body_names[index] for index in right_hand_indices],
            "attempt_contact_success_rate": contact_successes / len(attempts) if attempts else None,
            "interaction_proxy_success_rate": (
                interaction_proxy_successes / len(attempts) if attempts and has_object else None
            ),
            "object_motion_speed_threshold_m_s": object_motion_speed_threshold_m_s,
            "mean_first_contact_s": _mean_or_none(first_contact_times),
            "left_hand_step_fraction": float(np.mean(left_contact[evaluated_slice])),
            "right_hand_step_fraction": float(np.mean(right_contact[evaluated_slice])),
            "either_hand_step_fraction": float(np.mean(either_hand_contact[evaluated_slice])),
            "invalid_contact_step_fraction": float(np.mean(invalid_contact[evaluated_slice])),
            "left_hand_peak_force_n": (
                float(np.max(contact_magnitude[evaluated_slice, left_hand_indices]))
                if left_hand_indices
                else None
            ),
            "right_hand_peak_force_n": (
                float(np.max(contact_magnitude[evaluated_slice, right_hand_indices]))
                if right_hand_indices
                else None
            ),
        },
        "stability": {
            "fall_proxy_rate_per_attempt": (
                float(np.mean([attempt["fall_proxy"] for attempt in attempt_details]))
                if attempt_details
                else None
            ),
            "fall_height_drop_threshold_m": fall_height_drop_m,
            "fall_tilt_threshold_deg": fall_tilt_deg,
            "minimum_root_height_m": float(np.min(recording["pre_root_pos"][evaluated_slice, 2])),
            "peak_torque_utilization": float(np.max(torque_utilization)),
            "mean_torque_utilization": float(np.mean(torque_utilization)),
            "action_delta_rmse": _rmse(action_delta) if len(action_delta) else 0.0,
        },
        "attempt_details": attempt_details,
    }

    if has_object:
        object_position_error = (
            recording["object_pos_w"][evaluated_slice]
            - recording["ref_object_pos_w"][evaluated_slice]
        )
        object_orientation_error = quaternion_angle_error(
            recording["ref_object_quat_xyzw"][evaluated_slice],
            recording["object_quat_xyzw"][evaluated_slice],
        )
        summary["object"] = {
            "trajectory_position_rmse_m": _rmse(object_position_error),
            "trajectory_orientation_mean_rad": float(np.mean(object_orientation_error)),
            "trajectory_orientation_mean_deg": float(np.degrees(np.mean(object_orientation_error))),
            "mean_actual_displacement_m": _mean_or_none(actual_displacements),
            "mean_reference_displacement_m": _mean_or_none(reference_displacements),
            "direction_evaluable_attempts": displacement_evaluable_count,
            "correct_displacement_direction_rate": (
                correct_displacement_count / displacement_evaluable_count
                if displacement_evaluable_count
                else None
            ),
            "mean_displacement_direction_cosine": _mean_or_none(direction_cosines),
            "minimum_displacement_m": min_table_displacement_m,
        }

    return summary


def load_recording(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as data:
        recording = {name: data[name] for name in data.files if name != "_metadata_json"}
        metadata = json.loads(str(data["_metadata_json"].item()))
    return recording, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path, help="NPZ produced by EvalRecordingCallback")
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    parser.add_argument("--contact-force-threshold-n", type=float, default=5.0)
    parser.add_argument("--min-table-displacement-m", type=float, default=0.01)
    parser.add_argument("--object-motion-speed-threshold-m-s", type=float, default=0.02)
    parser.add_argument("--fall-height-drop-m", type=float, default=0.30)
    parser.add_argument("--fall-tilt-deg", type=float, default=60.0)
    parser.add_argument("--max-attempts", type=int)
    args = parser.parse_args()

    recording, metadata = load_recording(args.recording)
    summary = summarize_recording(
        recording,
        metadata,
        contact_force_threshold_n=args.contact_force_threshold_n,
        min_table_displacement_m=args.min_table_displacement_m,
        object_motion_speed_threshold_m_s=args.object_motion_speed_threshold_m_s,
        fall_height_drop_m=args.fall_height_drop_m,
        fall_tilt_deg=args.fall_tilt_deg,
        max_attempts=args.max_attempts,
    )
    output = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n")
    print(output)


if __name__ == "__main__":
    main()
