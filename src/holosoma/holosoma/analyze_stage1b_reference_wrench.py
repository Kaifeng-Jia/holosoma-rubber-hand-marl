#!/usr/bin/env python3
"""Screen side-A1 references for planar table-wrench feasibility.

This is an optimistic, table-side necessary-condition check.  It asks whether
the rubber-hand contacts present in the exact-reference collision replay can
produce the reference table's planar force and yaw moment while passive sliding
friction acts at the four table legs.  It does not prove that the floating-base
G1 can realize the resulting forces within balance and actuator limits.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linprog

from holosoma_retargeting.dual_a1_layout import quaternion_wxyz_to_matrix


GRAVITY_M_S2 = 9.81
WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)
RUBBER_HAND_BODIES = ("left_rubber_hand_link", "right_rubber_hand_link")
DEFAULT_MASSES_KG = (1.1, 2.6, 4.1)
DEFAULT_HAND_FRICTIONS = (0.15, 0.3, 0.5, 0.6, 0.8)
DEFAULT_GROUND_FRICTION = 0.5
DEFAULT_REFERENCE_MASS_KG = 2.6
DEFAULT_REFERENCE_INERTIA_DIAGONAL = np.array(
    [0.07644915580749512, 0.566853940486908, 0.5135635733604431],
    dtype=np.float64,
)
DEFAULT_COM_LOCAL_M = np.array([0.0, 0.015111745335161686, 0.0], dtype=np.float64)
DEFAULT_TABLE_HALF_EXTENTS_M = np.array([0.7, 0.023725, 0.2609764], dtype=np.float64)
DEFAULT_TABLETOP_CENTER_Y_M = 0.036275
DEFAULT_FOOT_POINTS_LOCAL_M = np.array(
    [
        [-0.6742736, -0.368, -0.23525],
        [-0.6742736, -0.368, 0.23525],
        [0.6742736, -0.368, -0.23525],
        [0.6742736, -0.368, 0.23525],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class PlanarWrenchSolution:
    """One linear-program solution in world coordinates."""

    feasible: bool
    status: int
    message: str
    hand_forces_w_n: np.ndarray
    ground_normal_forces_n: np.ndarray
    objective_n: float | None
    yaw_moment_residual_nm: float | None


def _horizontal_unit(vector: np.ndarray, *, fallback: np.ndarray | None = None) -> np.ndarray:
    horizontal = np.asarray(vector, dtype=np.float64).copy()
    horizontal[2] = 0.0
    norm = float(np.linalg.norm(horizontal))
    if norm > 1.0e-10:
        return horizontal / norm
    if fallback is not None:
        return _horizontal_unit(fallback)
    raise ValueError("Cannot normalize a zero horizontal vector")


def solve_planar_contact_wrench(
    *,
    hand_points_w_m: np.ndarray,
    table_com_w_m: np.ndarray,
    push_normal_w: np.ndarray,
    lateral_tangent_w: np.ndarray,
    foot_points_w_m: np.ndarray,
    foot_slip_velocities_w_m_s: np.ndarray,
    fallback_slip_velocity_w_m_s: np.ndarray,
    required_planar_force_w_n: np.ndarray,
    required_yaw_moment_nm: float,
    table_weight_n: float,
    hand_friction: float,
    ground_friction: float,
    allow_yaw_moment_slack: bool = False,
) -> PlanarWrenchSolution:
    """Minimize total rubber-hand normal force under planar wrench balance.

    Each hand can push only along ``push_normal_w`` and can exert one lateral
    tangential component bounded by ``|f_t| <= mu_hand * f_n``.  Each table leg
    contributes a non-negative normal load and sliding friction opposite its
    horizontal slip direction.  The four normal loads must sum to table weight.
    """
    hand_points = np.asarray(hand_points_w_m, dtype=np.float64)
    foot_points = np.asarray(foot_points_w_m, dtype=np.float64)
    foot_slips = np.asarray(foot_slip_velocities_w_m_s, dtype=np.float64)
    if hand_points.ndim != 2 or hand_points.shape[1] != 3:
        raise ValueError(f"hand_points_w_m must be [H, 3], got {hand_points.shape}")
    if foot_points.shape != foot_slips.shape or foot_points.ndim != 2 or foot_points.shape[1] != 3:
        raise ValueError("foot points and slip velocities must have matching [F, 3] shapes")
    if len(hand_points) == 0:
        return PlanarWrenchSolution(
            False,
            2,
            "no active rubber-hand contact",
            np.empty((0, 3)),
            np.empty(0),
            None,
            None,
        )
    if table_weight_n <= 0.0:
        raise ValueError("table_weight_n must be positive")
    if hand_friction < 0.0 or ground_friction < 0.0:
        raise ValueError("friction coefficients must be non-negative")

    normal = _horizontal_unit(push_normal_w)
    tangent_candidate = _horizontal_unit(lateral_tangent_w)
    tangent = tangent_candidate - float(np.dot(tangent_candidate, normal)) * normal
    tangent = _horizontal_unit(tangent)

    hand_count = len(hand_points)
    foot_count = len(foot_points)
    physical_variable_count = 2 * hand_count + foot_count
    variable_count = physical_variable_count + (2 if allow_yaw_moment_slack else 0)
    objective = np.zeros(variable_count, dtype=np.float64)
    objective[0 : 2 * hand_count : 2] = 1.0
    equality = np.zeros((4, variable_count), dtype=np.float64)
    target = np.array(
        [
            float(np.dot(required_planar_force_w_n, tangent)),
            float(np.dot(required_planar_force_w_n, normal)),
            float(required_yaw_moment_nm),
            float(table_weight_n),
        ],
        dtype=np.float64,
    )

    for hand_index, point in enumerate(hand_points):
        normal_column = 2 * hand_index
        tangent_column = normal_column + 1
        radius = point - table_com_w_m
        normal_yaw = float(np.cross(radius, normal)[2])
        tangent_yaw = float(np.cross(radius, tangent)[2])
        equality[:, normal_column] = [float(np.dot(normal, tangent)), 1.0, normal_yaw, 0.0]
        equality[:, tangent_column] = [1.0, float(np.dot(tangent, normal)), tangent_yaw, 0.0]

    for foot_index, (point, slip_velocity) in enumerate(zip(foot_points, foot_slips, strict=True)):
        column = 2 * hand_count + foot_index
        slip_direction = _horizontal_unit(slip_velocity, fallback=fallback_slip_velocity_w_m_s)
        friction_per_normal = -float(ground_friction) * slip_direction
        radius = point - table_com_w_m
        equality[:, column] = [
            float(np.dot(friction_per_normal, tangent)),
            float(np.dot(friction_per_normal, normal)),
            float(np.cross(radius, friction_per_normal)[2]),
            1.0,
        ]

    inequality = np.zeros((2 * hand_count, variable_count), dtype=np.float64)
    for hand_index in range(hand_count):
        normal_column = 2 * hand_index
        tangent_column = normal_column + 1
        inequality[2 * hand_index, normal_column] = -float(hand_friction)
        inequality[2 * hand_index, tangent_column] = 1.0
        inequality[2 * hand_index + 1, normal_column] = -float(hand_friction)
        inequality[2 * hand_index + 1, tangent_column] = -1.0

    bounds = []
    for _ in range(hand_count):
        bounds.extend(((0.0, None), (None, None)))
    bounds.extend((0.0, None) for _ in range(foot_count))
    inequality_target = np.zeros(len(inequality), dtype=np.float64)
    if allow_yaw_moment_slack:
        positive_slack_column = physical_variable_count
        negative_slack_column = physical_variable_count + 1
        equality[2, positive_slack_column] = -1.0
        equality[2, negative_slack_column] = 1.0
        bounds.extend(((0.0, None), (0.0, None)))
        slack_objective = np.zeros(variable_count, dtype=np.float64)
        slack_objective[positive_slack_column] = 1.0
        slack_objective[negative_slack_column] = 1.0
        slack_result = linprog(
            slack_objective,
            A_ub=inequality,
            b_ub=inequality_target,
            A_eq=equality,
            b_eq=target,
            bounds=bounds,
            method="highs",
        )
        if not slack_result.success:
            return PlanarWrenchSolution(
                False,
                int(slack_result.status),
                str(slack_result.message),
                np.empty((0, 3)),
                np.empty(0),
                None,
                None,
            )
        slack_bound = np.zeros(variable_count, dtype=np.float64)
        slack_bound[positive_slack_column] = 1.0
        slack_bound[negative_slack_column] = 1.0
        inequality = np.concatenate((inequality, slack_bound[None, :]), axis=0)
        inequality_target = np.concatenate(
            (inequality_target, np.array([float(slack_result.fun) + 1.0e-8]))
        )

    result = linprog(
        objective,
        A_ub=inequality,
        b_ub=inequality_target,
        A_eq=equality,
        b_eq=target,
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        return PlanarWrenchSolution(
            False,
            int(result.status),
            str(result.message),
            np.empty((0, 3)),
            np.empty(0),
            None,
            None,
        )

    hand_forces = np.empty((hand_count, 3), dtype=np.float64)
    for hand_index in range(hand_count):
        normal_force = result.x[2 * hand_index]
        tangent_force = result.x[2 * hand_index + 1]
        hand_forces[hand_index] = normal_force * normal + tangent_force * tangent
    yaw_residual = 0.0
    if allow_yaw_moment_slack:
        yaw_residual = float(result.x[positive_slack_column] - result.x[negative_slack_column])
    return PlanarWrenchSolution(
        True,
        int(result.status),
        str(result.message),
        hand_forces,
        np.asarray(result.x[2 * hand_count : physical_variable_count], dtype=np.float64),
        float(result.fun),
        yaw_residual,
    )


def _load_motion(path: Path) -> dict[str, np.ndarray | int | list[str]]:
    with np.load(path, allow_pickle=False) as data:
        required = {
            "fps",
            "body_names",
            "body_pos_w",
            "object_pos_w",
            "object_quat_w",
            "object_lin_vel_w",
            "object_ang_vel_w",
        }
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"{path} is missing required channels: {sorted(missing)}")
        motion: dict[str, np.ndarray | int | list[str]] = {
            "fps": int(np.asarray(data["fps"]).reshape(-1)[0]),
            "body_names": [str(name) for name in data["body_names"].tolist()],
        }
        for name in required.difference({"fps", "body_names"}):
            motion[name] = np.asarray(data[name], dtype=np.float64)
    frame_count = len(np.asarray(motion["object_pos_w"]))
    if frame_count < 3:
        raise ValueError(f"{path} must contain at least three frames")
    if int(motion["fps"]) <= 0:
        raise ValueError(f"{path} has invalid fps {motion['fps']}")
    return motion


def _active_hand_frames(collision_report: dict[str, Any], side_key: str) -> dict[int, tuple[str, ...]]:
    side = collision_report["sides"][side_key]
    active: dict[int, set[str]] = {}
    for contact in side["rubber_hand_contacts"]:
        body = str(contact["body"])
        if body not in RUBBER_HAND_BODIES:
            raise ValueError(f"Unexpected rubber-hand body in collision report: {body}")
        active.setdefault(int(contact["frame"]), set()).add(body)
    return {frame: tuple(sorted(bodies)) for frame, bodies in active.items()}


def _project_hand_points_to_push_face(
    hand_origins_w_m: np.ndarray,
    object_pos_w_m: np.ndarray,
    object_rotation_w: np.ndarray,
) -> np.ndarray:
    local = (object_rotation_w.T @ (hand_origins_w_m - object_pos_w_m).T).T
    local[:, 0] = np.clip(local[:, 0], -DEFAULT_TABLE_HALF_EXTENTS_M[0], DEFAULT_TABLE_HALF_EXTENTS_M[0])
    tabletop_y_min = DEFAULT_TABLETOP_CENTER_Y_M - DEFAULT_TABLE_HALF_EXTENTS_M[1]
    tabletop_y_max = DEFAULT_TABLETOP_CENTER_Y_M + DEFAULT_TABLE_HALF_EXTENTS_M[1]
    local[:, 1] = np.clip(local[:, 1], tabletop_y_min, tabletop_y_max)
    local[:, 2] = -DEFAULT_TABLE_HALF_EXTENTS_M[2]
    return object_pos_w_m + (object_rotation_w @ local.T).T


def _summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"median": None, "p95": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95.0)),
        "max": float(np.max(array)),
    }


def _contiguous_ranges(frames: list[int]) -> list[list[int]]:
    if not frames:
        return []
    ranges: list[list[int]] = []
    start = previous = frames[0]
    for frame in frames[1:]:
        if frame != previous + 1:
            ranges.append([start, previous])
            start = frame
        previous = frame
    ranges.append([start, previous])
    return ranges


def analyze_side(
    *,
    motion: dict[str, np.ndarray | int | list[str]],
    active_hands_by_frame: dict[int, tuple[str, ...]],
    mass_kg: float,
    hand_friction: float,
    ground_friction: float,
    speed_threshold_m_s: float,
) -> dict[str, Any]:
    fps = int(motion["fps"])
    dt = 1.0 / fps
    body_names = list(motion["body_names"])
    body_pos = np.asarray(motion["body_pos_w"])
    object_pos = np.asarray(motion["object_pos_w"])
    object_quat = np.asarray(motion["object_quat_w"])
    object_ang_vel = np.asarray(motion["object_ang_vel_w"])
    rotations = quaternion_wxyz_to_matrix(object_quat)
    com_positions = object_pos + np.einsum("tij,j->ti", rotations, DEFAULT_COM_LOCAL_M)
    com_velocities = np.gradient(com_positions, dt, axis=0, edge_order=2)
    com_accelerations = np.gradient(com_velocities, dt, axis=0, edge_order=2)
    angular_accelerations = np.gradient(object_ang_vel, dt, axis=0, edge_order=2)
    inertia_body = DEFAULT_REFERENCE_INERTIA_DIAGONAL * (mass_kg / DEFAULT_REFERENCE_MASS_KG)
    inertia_world = np.einsum("tij,j,tkj->tik", rotations, inertia_body, rotations)
    angular_momenta = np.einsum("tij,tj->ti", inertia_world, object_ang_vel)
    required_moments = np.einsum("tij,tj->ti", inertia_world, angular_accelerations) + np.cross(
        object_ang_vel,
        angular_momenta,
    )
    planar_speeds = np.linalg.norm(com_velocities[:, :2], axis=1)
    hand_body_indexes = {name: body_names.index(name) for name in RUBBER_HAND_BODIES}

    evaluated_frames: list[int] = []
    transition_frames: list[int] = []
    strict_feasible_frames: list[int] = []
    strict_failed_frames: list[int] = []
    relaxed_feasible_frames: list[int] = []
    unresolved_frames: list[int] = []
    failed_status_counts: dict[str, int] = {}
    failed_messages: set[str] = set()
    total_normal_forces: list[float] = []
    peak_individual_forces: list[float] = []
    peak_friction_utilizations: list[float] = []
    absolute_yaw_moment_relaxations: list[float] = []
    absolute_yaw_acceleration_relaxations: list[float] = []
    signed_yaw_acceleration_relaxation = np.zeros(len(object_pos), dtype=np.float64)

    for frame, active_bodies in sorted(active_hands_by_frame.items()):
        if planar_speeds[frame] <= speed_threshold_m_s:
            transition_frames.append(frame)
            continue
        evaluated_frames.append(frame)
        rotation = rotations[frame]
        push_normal = rotation[:, 2]
        lateral_tangent = rotation[:, 0]
        hand_origins = body_pos[frame, [hand_body_indexes[name] for name in active_bodies]]
        hand_points = _project_hand_points_to_push_face(hand_origins, object_pos[frame], rotation)
        foot_points = object_pos[frame] + (rotation @ DEFAULT_FOOT_POINTS_LOCAL_M.T).T
        foot_radii = foot_points - com_positions[frame]
        foot_slips = com_velocities[frame] + np.cross(object_ang_vel[frame], foot_radii)
        required_force = mass_kg * com_accelerations[frame]
        solution = solve_planar_contact_wrench(
            hand_points_w_m=hand_points,
            table_com_w_m=com_positions[frame],
            push_normal_w=push_normal,
            lateral_tangent_w=lateral_tangent,
            foot_points_w_m=foot_points,
            foot_slip_velocities_w_m_s=foot_slips,
            fallback_slip_velocity_w_m_s=com_velocities[frame],
            required_planar_force_w_n=required_force,
            required_yaw_moment_nm=float(required_moments[frame, 2]),
            table_weight_n=mass_kg * GRAVITY_M_S2,
            hand_friction=hand_friction,
            ground_friction=ground_friction,
        )
        if not solution.feasible:
            strict_failed_frames.append(frame)
            solution = solve_planar_contact_wrench(
                hand_points_w_m=hand_points,
                table_com_w_m=com_positions[frame],
                push_normal_w=push_normal,
                lateral_tangent_w=lateral_tangent,
                foot_points_w_m=foot_points,
                foot_slip_velocities_w_m_s=foot_slips,
                fallback_slip_velocity_w_m_s=com_velocities[frame],
                required_planar_force_w_n=required_force,
                required_yaw_moment_nm=float(required_moments[frame, 2]),
                table_weight_n=mass_kg * GRAVITY_M_S2,
                hand_friction=hand_friction,
                ground_friction=ground_friction,
                allow_yaw_moment_slack=True,
            )
        else:
            strict_feasible_frames.append(frame)
        if not solution.feasible:
            unresolved_frames.append(frame)
            status_key = str(solution.status)
            failed_status_counts[status_key] = failed_status_counts.get(status_key, 0) + 1
            failed_messages.add(solution.message)
            continue
        relaxed_feasible_frames.append(frame)
        yaw_relaxation = abs(float(solution.yaw_moment_residual_nm))
        absolute_yaw_moment_relaxations.append(yaw_relaxation)
        yaw_inertia = float(inertia_world[frame, 2, 2])
        absolute_yaw_acceleration_relaxations.append(yaw_relaxation / yaw_inertia)
        signed_yaw_acceleration_relaxation[frame] = float(solution.yaw_moment_residual_nm) / yaw_inertia
        normal_components = solution.hand_forces_w_n @ _horizontal_unit(push_normal)
        tangent_components = solution.hand_forces_w_n @ _horizontal_unit(lateral_tangent)
        total_normal_forces.append(float(np.sum(normal_components)))
        peak_individual_forces.append(float(np.max(np.linalg.norm(solution.hand_forces_w_n, axis=1))))
        denominator = np.maximum(hand_friction * normal_components, 1.0e-12)
        peak_friction_utilizations.append(float(np.max(np.abs(tangent_components) / denominator)))

    yaw_rate_delta = np.cumsum(signed_yaw_acceleration_relaxation) * dt
    yaw_delta = np.cumsum(yaw_rate_delta) * dt
    reference_yaw = np.unwrap(np.arctan2(rotations[:, 1, 0], rotations[:, 0, 0]))
    linearized_yaw = reference_yaw + yaw_delta
    abs_yaw_delta_deg = np.degrees(np.abs(yaw_delta))

    def first_limit_crossing(limit_deg: float) -> dict[str, float | int] | None:
        crossings = np.flatnonzero(abs_yaw_delta_deg >= limit_deg)
        if not len(crossings):
            return None
        frame = int(crossings[0])
        return {"frame": frame, "time_s": frame * dt}

    crossed_validity_limit = bool(np.any(abs_yaw_delta_deg >= 15.0))

    return {
        "mass_kg": mass_kg,
        "hand_friction": hand_friction,
        "ground_friction": ground_friction,
        "contact_frame_count": len(active_hands_by_frame),
        "transition_frame_count": len(transition_frames),
        "transition_frames": transition_frames,
        "evaluated_sliding_frame_count": len(evaluated_frames),
        "strict_yaw_feasible_frame_count": len(strict_feasible_frames),
        "strict_yaw_feasible_fraction": (
            len(strict_feasible_frames) / len(evaluated_frames) if evaluated_frames else None
        ),
        "strict_yaw_failed_frame_ranges": _contiguous_ranges(strict_failed_frames),
        "natural_yaw_relaxed_feasible_frame_count": len(relaxed_feasible_frames),
        "natural_yaw_relaxed_feasible_fraction": (
            len(relaxed_feasible_frames) / len(evaluated_frames) if evaluated_frames else None
        ),
        "all_evaluated_frames_feasible_with_natural_yaw": (
            bool(evaluated_frames) and len(relaxed_feasible_frames) == len(evaluated_frames)
        ),
        "unresolved_frame_ranges": _contiguous_ranges(unresolved_frames),
        "failed_status_counts": failed_status_counts,
        "failed_messages": sorted(failed_messages),
        "minimum_abs_yaw_moment_relaxation_nm": _summary(absolute_yaw_moment_relaxations),
        "equivalent_abs_yaw_acceleration_relaxation_rad_s2": _summary(
            absolute_yaw_acceleration_relaxations
        ),
        "linearized_natural_yaw_estimate": {
            "scope": (
                "first-order integration around the frozen reference; transition and unresolved "
                "frames use zero relaxation; not a physics-rollout verdict"
            ),
            "validity_limit_deg": 15.0,
            "first_abs_delta_15deg": first_limit_crossing(15.0),
            "first_abs_delta_30deg": first_limit_crossing(30.0),
            "remained_within_validity_limit": not crossed_validity_limit,
            "endpoint_delta_from_reference_deg_if_within_limit": (
                float(np.degrees(yaw_delta[-1])) if not crossed_validity_limit else None
            ),
            "max_abs_delta_from_reference_deg_if_within_limit": (
                float(np.max(abs_yaw_delta_deg)) if not crossed_validity_limit else None
            ),
            "reference_endpoint_change_deg": float(np.degrees(reference_yaw[-1] - reference_yaw[0])),
            "linearized_endpoint_change_deg_if_within_limit": (
                float(np.degrees(linearized_yaw[-1] - linearized_yaw[0]))
                if not crossed_validity_limit
                else None
            ),
        },
        "minimum_total_hand_normal_force_n": _summary(total_normal_forces),
        "peak_individual_hand_force_n": _summary(peak_individual_forces),
        "peak_hand_friction_utilization": _summary(peak_friction_utilizations),
    }


def analyze(
    *,
    left_motion_path: Path,
    right_motion_path: Path,
    collision_report_path: Path,
    masses_kg: tuple[float, ...],
    hand_frictions: tuple[float, ...],
    ground_friction: float,
    speed_threshold_m_s: float,
) -> dict[str, Any]:
    left_motion = _load_motion(left_motion_path)
    right_motion = _load_motion(right_motion_path)
    if int(left_motion["fps"]) != int(right_motion["fps"]):
        raise ValueError("Left and right reference FPS values differ")
    for channel in ("object_pos_w", "object_quat_w", "object_lin_vel_w", "object_ang_vel_w"):
        if not np.allclose(np.asarray(left_motion[channel]), np.asarray(right_motion[channel]), atol=1.0e-9):
            raise ValueError(f"Left and right references differ in shared object channel {channel}")
    collision_report = json.loads(collision_report_path.read_text())
    if int(collision_report["frames_evaluated"]) != len(np.asarray(left_motion["object_pos_w"])):
        raise ValueError("Collision report and motion frame counts differ")

    sides = {
        "left": (left_motion, _active_hand_frames(collision_report, "desired_0")),
        "right": (right_motion, _active_hand_frames(collision_report, "desired_1")),
    }
    cases: dict[str, list[dict[str, Any]]] = {"left": [], "right": []}
    for side_name, (motion, active_hands) in sides.items():
        for mass_kg in masses_kg:
            for hand_friction in hand_frictions:
                cases[side_name].append(
                    analyze_side(
                        motion=motion,
                        active_hands_by_frame=active_hands,
                        mass_kg=mass_kg,
                        hand_friction=hand_friction,
                        ground_friction=ground_friction,
                        speed_threshold_m_s=speed_threshold_m_s,
                    )
                )

    return {
        "schema": "holosoma.stage1b_reference_wrench.v1",
        "scope": (
            "optimistic planar table-side necessary-condition screen; "
            "not a whole-body balance or actuator-feasibility verdict"
        ),
        "inputs": {
            "left_motion_npz": str(left_motion_path.resolve()),
            "right_motion_npz": str(right_motion_path.resolve()),
            "collision_report_json": str(collision_report_path.resolve()),
            "fps": int(left_motion["fps"]),
        },
        "contract": {
            "masses_kg": list(masses_kg),
            "hand_friction_scan": list(hand_frictions),
            "ground_friction": ground_friction,
            "sliding_speed_threshold_m_s": speed_threshold_m_s,
            "wrist_contact_counts_as_rubber_hand_propulsion": False,
            "ground_friction_is_passive_and_opposes_reference_foot_slip": True,
            "table_inertia_scales_linearly_from_mass_kg": DEFAULT_REFERENCE_MASS_KG,
            "reference_inertia_diagonal_kg_m2": DEFAULT_REFERENCE_INERTIA_DIAGONAL.tolist(),
            "com_local_m": DEFAULT_COM_LOCAL_M.tolist(),
        },
        "cases": cases,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-motion-npz", type=Path, required=True)
    parser.add_argument("--right-motion-npz", type=Path, required=True)
    parser.add_argument("--collision-report", type=Path, required=True)
    parser.add_argument("--masses-kg", type=float, nargs="+", default=list(DEFAULT_MASSES_KG))
    parser.add_argument("--hand-frictions", type=float, nargs="+", default=list(DEFAULT_HAND_FRICTIONS))
    parser.add_argument("--ground-friction", type=float, default=DEFAULT_GROUND_FRICTION)
    parser.add_argument("--speed-threshold-m-s", type=float, default=0.02)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _print_case_summary(report: dict[str, Any]) -> None:
    for side, cases in report["cases"].items():
        for case in cases:
            evaluated = case["evaluated_sliding_frame_count"]
            crossing = case["linearized_natural_yaw_estimate"]["first_abs_delta_15deg"]
            crossing_text = "none" if crossing is None else f"{crossing['time_s']:.2f}s"
            print(
                "[stage1b-reference-wrench] "
                f"side={side} mass={case['mass_kg']:.1f}kg mu_hand={case['hand_friction']:.2f} "
                f"strict={case['strict_yaw_feasible_frame_count']}/{evaluated} "
                f"natural_yaw={case['natural_yaw_relaxed_feasible_frame_count']}/{evaluated} "
                f"first_15deg={crossing_text}"
            )


def main() -> int:
    args = parse_args()
    if any(mass <= 0.0 for mass in args.masses_kg):
        raise ValueError("All table masses must be positive")
    if any(friction < 0.0 for friction in args.hand_frictions) or args.ground_friction < 0.0:
        raise ValueError("Friction coefficients must be non-negative")
    if args.speed_threshold_m_s < 0.0:
        raise ValueError("speed threshold must be non-negative")
    report = analyze(
        left_motion_path=args.left_motion_npz,
        right_motion_path=args.right_motion_npz,
        collision_report_path=args.collision_report,
        masses_kg=tuple(args.masses_kg),
        hand_frictions=tuple(args.hand_frictions),
        ground_friction=args.ground_friction,
        speed_threshold_m_s=args.speed_threshold_m_s,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _print_case_summary(report)
    print(f"[stage1b-reference-wrench] report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
