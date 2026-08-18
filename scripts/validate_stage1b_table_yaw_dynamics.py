#!/usr/bin/env python3
"""Run a minimal table-only forward-dynamics check for side-A1 references.

The tool derives an open-loop rubber-hand force schedule from the frozen
reference, then lets a planar rigid table translate and yaw freely.  It does
not contain a robot, policy, reward, controller, or hidden stabilizing force.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "holosoma"))
sys.path.insert(0, str(REPO_ROOT / "src" / "holosoma_retargeting"))

from holosoma.analyze_stage1b_reference_wrench import (  # noqa: E402
    DEFAULT_COM_LOCAL_M,
    DEFAULT_FOOT_POINTS_LOCAL_M,
    DEFAULT_REFERENCE_INERTIA_DIAGONAL,
    DEFAULT_REFERENCE_MASS_KG,
    GRAVITY_M_S2,
RUBBER_HAND_BODIES,
    _active_hand_frames,
    _load_motion,
    _project_hand_points_to_push_face,
    solve_planar_contact_wrench,
)
from holosoma_retargeting.dual_a1_layout import quaternion_wxyz_to_matrix  # noqa: E402


DYNAMICS_SUBSTEPS = 10


def _planar_axes(yaw: float) -> tuple[np.ndarray, np.ndarray]:
    lateral = np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float64)
    push = np.array([np.sin(yaw), -np.cos(yaw)], dtype=np.float64)
    return lateral, push


def _first_crossing(
    values: np.ndarray,
    limit: float,
    dt: float,
    *,
    frame_offset: int,
) -> dict[str, float | int] | None:
    indexes = np.flatnonzero(np.abs(values) >= limit)
    if not len(indexes):
        return None
    relative_frame = int(indexes[0])
    return {
        "frame": frame_offset + relative_frame,
        "time_from_rollout_start_s": relative_frame * dt,
    }


def _derive_force_schedule(
    *,
    motion: dict[str, np.ndarray | int | list[str]],
    active_hands_by_frame: dict[int, tuple[str, ...]],
    mass_kg: float,
    hand_friction: float,
    ground_friction: float,
    speed_threshold_m_s: float,
) -> tuple[list[dict[str, Any] | None], dict[str, Any]]:
    fps = int(motion["fps"])
    dt = 1.0 / fps
    body_names = list(motion["body_names"])
    body_pos = np.asarray(motion["body_pos_w"])
    object_pos = np.asarray(motion["object_pos_w"])
    rotations = quaternion_wxyz_to_matrix(np.asarray(motion["object_quat_w"]))
    object_ang_vel = np.asarray(motion["object_ang_vel_w"])
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
    hand_indexes = {name: body_names.index(name) for name in RUBBER_HAND_BODIES}
    schedule: list[dict[str, Any] | None] = [None] * len(object_pos)
    transition_frames: list[int] = []
    unresolved_frames: list[int] = []
    strict_solution_count = 0
    relaxed_solution_count = 0

    for frame, active_bodies in sorted(active_hands_by_frame.items()):
        if np.linalg.norm(com_velocities[frame, :2]) <= speed_threshold_m_s:
            transition_frames.append(frame)
            continue
        rotation = rotations[frame]
        push_w = rotation[:, 2]
        lateral_w = rotation[:, 0]
        push_xy = push_w.copy()
        push_xy[2] = 0.0
        push_xy /= np.linalg.norm(push_xy)
        lateral_xy = lateral_w.copy()
        lateral_xy[2] = 0.0
        lateral_xy -= np.dot(lateral_xy, push_xy) * push_xy
        lateral_xy /= np.linalg.norm(lateral_xy)
        hand_origins = body_pos[frame, [hand_indexes[name] for name in active_bodies]]
        hand_points = _project_hand_points_to_push_face(hand_origins, object_pos[frame], rotation)
        foot_points = object_pos[frame] + (rotation @ DEFAULT_FOOT_POINTS_LOCAL_M.T).T
        foot_radii = foot_points - com_positions[frame]
        foot_slips = com_velocities[frame] + np.cross(object_ang_vel[frame], foot_radii)
        common = {
            "hand_points_w_m": hand_points,
            "table_com_w_m": com_positions[frame],
            "push_normal_w": push_w,
            "lateral_tangent_w": lateral_w,
            "foot_points_w_m": foot_points,
            "foot_slip_velocities_w_m_s": foot_slips,
            "fallback_slip_velocity_w_m_s": com_velocities[frame],
            "required_planar_force_w_n": mass_kg * com_accelerations[frame],
            "required_yaw_moment_nm": float(required_moments[frame, 2]),
            "table_weight_n": mass_kg * GRAVITY_M_S2,
            "hand_friction": hand_friction,
            "ground_friction": ground_friction,
        }
        solution = solve_planar_contact_wrench(**common)
        used_yaw_relaxation = False
        if solution.feasible:
            strict_solution_count += 1
        else:
            solution = solve_planar_contact_wrench(**common, allow_yaw_moment_slack=True)
            used_yaw_relaxation = True
        if not solution.feasible:
            unresolved_frames.append(frame)
            continue
        relaxed_solution_count += int(used_yaw_relaxation)
        hand_points_local = (rotation.T @ (hand_points - object_pos[frame]).T).T
        hand_forces_local = np.stack(
            (
                solution.hand_forces_w_n @ lateral_xy,
                solution.hand_forces_w_n @ push_xy,
            ),
            axis=-1,
        )
        schedule[frame] = {
            "hand_points_local_xz": hand_points_local[:, [0, 2]],
            "hand_forces_local_xz": hand_forces_local,
            "ground_normal_forces_n": solution.ground_normal_forces_n,
            "used_yaw_relaxation": used_yaw_relaxation,
        }

    return schedule, {
        "strict_solution_count": strict_solution_count,
        "relaxed_solution_count": relaxed_solution_count,
        "transition_frames": transition_frames,
        "unresolved_frames": unresolved_frames,
    }


def _simulate(
    *,
    motion: dict[str, np.ndarray | int | list[str]],
    schedule: list[dict[str, Any] | None],
    mass_kg: float,
    ground_friction: float,
) -> dict[str, Any]:
    fps = int(motion["fps"])
    dt = 1.0 / fps
    object_pos = np.asarray(motion["object_pos_w"])
    rotations = quaternion_wxyz_to_matrix(np.asarray(motion["object_quat_w"]))
    reference_com = object_pos + np.einsum("tij,j->ti", rotations, DEFAULT_COM_LOCAL_M)
    reference_com_velocity = np.gradient(reference_com, dt, axis=0, edge_order=2)
    reference_yaw = np.unwrap(np.arctan2(rotations[:, 1, 0], rotations[:, 0, 0]))
    yaw_inertia = float(DEFAULT_REFERENCE_INERTIA_DIAGONAL[1] * mass_kg / DEFAULT_REFERENCE_MASS_KG)
    foot_points_local_xz = DEFAULT_FOOT_POINTS_LOCAL_M[:, [0, 2]]
    active_frames = [frame for frame, entry in enumerate(schedule) if entry is not None]
    if not active_frames:
        raise ValueError("Force schedule contains no solvable sliding-contact frame")
    start_frame = active_frames[0]

    position = reference_com[start_frame, :2].copy()
    velocity = reference_com_velocity[start_frame, :2].copy()
    yaw = float(reference_yaw[start_frame])
    yaw_rate = float(np.asarray(motion["object_ang_vel_w"])[start_frame, 2])
    positions = np.empty((len(schedule), 2), dtype=np.float64)
    yaws = np.empty(len(schedule), dtype=np.float64)
    positions[:start_frame] = np.nan
    yaws[:start_frame] = np.nan
    positions[start_frame] = position
    yaws[start_frame] = yaw

    substep_dt = dt / DYNAMICS_SUBSTEPS
    for frame in range(start_frame, len(schedule) - 1):
        entry = schedule[frame]
        ground_normals = (
            np.asarray(entry["ground_normal_forces_n"], dtype=np.float64)
            if entry is not None
            else np.full(4, mass_kg * GRAVITY_M_S2 / 4.0, dtype=np.float64)
        )
        for _ in range(DYNAMICS_SUBSTEPS):
            lateral, push = _planar_axes(yaw)
            net_force = np.zeros(2, dtype=np.float64)
            net_yaw_moment = 0.0
            if entry is not None:
                for point_local, force_local in zip(
                    entry["hand_points_local_xz"],
                    entry["hand_forces_local_xz"],
                    strict=True,
                ):
                    radius = point_local[0] * lateral + point_local[1] * push
                    force = force_local[0] * lateral + force_local[1] * push
                    net_force += force
                    net_yaw_moment += float(radius[0] * force[1] - radius[1] * force[0])

            for point_local, normal_force in zip(
                foot_points_local_xz,
                ground_normals,
                strict=True,
            ):
                radius = point_local[0] * lateral + point_local[1] * push
                slip_velocity = velocity + yaw_rate * np.array([-radius[1], radius[0]])
                slip_speed = float(np.linalg.norm(slip_velocity))
                if slip_speed <= 1.0e-8:
                    continue
                supported_mass = float(normal_force) / GRAVITY_M_S2
                friction_magnitude = min(
                    ground_friction * float(normal_force),
                    supported_mass * slip_speed / substep_dt,
                )
                friction = -friction_magnitude * slip_velocity / slip_speed
                net_force += friction
                net_yaw_moment += float(radius[0] * friction[1] - radius[1] * friction[0])

            velocity += (net_force / mass_kg) * substep_dt
            yaw_rate += (net_yaw_moment / yaw_inertia) * substep_dt
            position += velocity * substep_dt
            yaw += yaw_rate * substep_dt
        positions[frame + 1] = position
        yaws[frame + 1] = yaw

    evaluated_slice = slice(start_frame, len(schedule))
    position_error = positions[evaluated_slice] - reference_com[evaluated_slice, :2]
    simulated_yaw = np.unwrap(yaws[evaluated_slice])
    yaw_error = simulated_yaw - reference_yaw[evaluated_slice]
    return {
        "rollout_start_frame": start_frame,
        "reference_displacement_m": (reference_com[-1, :2] - reference_com[start_frame, :2]).tolist(),
        "simulated_displacement_m": (positions[-1] - positions[start_frame]).tolist(),
        "endpoint_position_error_m": float(np.linalg.norm(position_error[-1])),
        "max_position_error_m": float(np.max(np.linalg.norm(position_error, axis=1))),
        "reference_endpoint_yaw_change_deg": float(
            np.degrees(reference_yaw[-1] - reference_yaw[start_frame])
        ),
        "simulated_endpoint_yaw_change_deg": float(
            np.degrees(yaws[-1] - yaws[start_frame])
        ),
        "endpoint_yaw_error_deg": float(np.degrees(yaw_error[-1])),
        "max_abs_yaw_error_deg": float(np.degrees(np.max(np.abs(yaw_error)))),
        "first_abs_yaw_error_15deg": _first_crossing(
            yaw_error,
            np.radians(15.0),
            dt,
            frame_offset=start_frame,
        ),
        "first_abs_yaw_error_30deg": _first_crossing(
            yaw_error,
            np.radians(30.0),
            dt,
            frame_offset=start_frame,
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    motions = {
        "left": _load_motion(args.left_motion_npz),
        "right": _load_motion(args.right_motion_npz),
    }
    if int(motions["left"]["fps"]) != int(motions["right"]["fps"]):
        raise ValueError("Left and right reference FPS values differ")
    for channel in ("object_pos_w", "object_quat_w", "object_lin_vel_w", "object_ang_vel_w"):
        if not np.allclose(np.asarray(motions["left"][channel]), np.asarray(motions["right"][channel])):
            raise ValueError(f"Left and right references differ in shared object channel {channel}")
    collision_report = json.loads(args.collision_report.read_text())
    if int(collision_report["frames_evaluated"]) != len(np.asarray(motions["left"]["object_pos_w"])):
        raise ValueError("Collision report and motion frame counts differ")
    contacts = {
        "left": _active_hand_frames(collision_report, "desired_0"),
        "right": _active_hand_frames(collision_report, "desired_1"),
    }
    sides: dict[str, Any] = {}
    for side in ("left", "right"):
        schedule, schedule_summary = _derive_force_schedule(
            motion=motions[side],
            active_hands_by_frame=contacts[side],
            mass_kg=args.mass_kg,
            hand_friction=args.hand_friction,
            ground_friction=args.ground_friction,
            speed_threshold_m_s=args.speed_threshold_m_s,
        )
        sides[side] = {
            "force_schedule": schedule_summary,
            "rollout": _simulate(
                motion=motions[side],
                schedule=schedule,
                mass_kg=args.mass_kg,
                ground_friction=args.ground_friction,
            ),
        }
    return {
        "schema": "holosoma.stage1b_table_yaw_dynamics.v1",
        "scope": "minimal open-loop planar table dynamics; no robot, policy, reward, or stabilizer",
        "parameters": {
            "mass_kg": args.mass_kg,
            "hand_friction": args.hand_friction,
            "ground_friction": args.ground_friction,
            "speed_threshold_m_s": args.speed_threshold_m_s,
            "dynamics_substeps_per_reference_frame": DYNAMICS_SUBSTEPS,
            "friction_impulse_does_not_reverse_local_slip_in_one_substep": True,
            "static_friction_not_modeled": True,
        },
        "inputs": {
            "left_motion_npz": str(args.left_motion_npz.resolve()),
            "right_motion_npz": str(args.right_motion_npz.resolve()),
            "collision_report": str(args.collision_report.resolve()),
        },
        "sides": sides,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-motion-npz", type=Path, required=True)
    parser.add_argument("--right-motion-npz", type=Path, required=True)
    parser.add_argument("--collision-report", type=Path, required=True)
    parser.add_argument("--mass-kg", type=float, default=2.6)
    parser.add_argument("--hand-friction", type=float, default=0.5)
    parser.add_argument("--ground-friction", type=float, default=0.5)
    parser.add_argument("--speed-threshold-m-s", type=float, default=0.02)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mass_kg <= 0.0:
        raise ValueError("mass-kg must be positive")
    if args.hand_friction < 0.0 or args.ground_friction < 0.0:
        raise ValueError("friction coefficients must be non-negative")
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    for side, result in report["sides"].items():
        rollout = result["rollout"]
        print(
            "[stage1b-table-yaw] "
            f"side={side} position_error={rollout['endpoint_position_error_m']:.3f}m "
            f"yaw_error={rollout['endpoint_yaw_error_deg']:.1f}deg "
            f"max_abs_yaw_error={rollout['max_abs_yaw_error_deg']:.1f}deg "
            f"first_15deg={rollout['first_abs_yaw_error_15deg']}"
        )
    print(f"[stage1b-table-yaw] report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
