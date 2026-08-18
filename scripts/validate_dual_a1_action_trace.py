#!/usr/bin/env python3
"""Replay two frozen-A1 action traces against one physical wide table.

This is a no-training mechanics preflight, not an online policy evaluation and
not MARL. It extracts one complete motion-phase-aligned episode from each of
the existing frozen-A1 Stage-1B recordings, places two physical rubber-hand G1
robots around one shared table, and replays their recorded PD position targets.

The diagnostic asks one narrow question: do the two frozen action traces show
useful shared-table translation and opposite-side yaw-moment cancellation? A
failure is not evidence that jointly trained multi-agent control is infeasible,
because the traces were produced independently and are replayed open loop.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "holosoma"))
sys.path.insert(0, str(REPO_ROOT / "src" / "holosoma_retargeting"))

RECORDING_DIR = REPO_ROOT / "logs" / "WholeBodyTracking" / "stage1b_a1_retention_v1"
DEFAULT_LEFT_RECORDING = RECORDING_DIR / "formal_wide_left_trajectory.npz"
DEFAULT_RIGHT_RECORDING = RECORDING_DIR / "formal_wide_right_trajectory.npz"
DEFAULT_ROBOT_URDF = (
    REPO_ROOT / "src" / "holosoma" / "holosoma" / "data" / "robots" / "g1"
    / "main_mesh_collision_rubberhand.urdf"
)
DEFAULT_TABLE_URDF = (
    REPO_ROOT / "src" / "holosoma" / "holosoma" / "data" / "motions"
    / "g1_29dof" / "whole_body_tracking" / "objects_widetable_a1_retention.urdf"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-recording", type=Path, default=DEFAULT_LEFT_RECORDING)
    parser.add_argument("--right-recording", type=Path, default=DEFAULT_RIGHT_RECORDING)
    parser.add_argument("--robot-urdf", type=Path, default=DEFAULT_ROBOT_URDF)
    parser.add_argument("--table-urdf", type=Path, default=DEFAULT_TABLE_URDF)
    parser.add_argument("--table-mass-kg", type=float, default=2.6)
    parser.add_argument("--friction", type=float, default=0.5)
    parser.add_argument("--contact-threshold-n", type=float, default=5.0)
    parser.add_argument("--fall-height-m", type=float, default=0.45)
    parser.add_argument(
        "--report", type=Path, default=Path("/tmp/dual_a1_action_trace_report.json")
    )

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


ARGS = parse_args()

from isaaclab.app import AppLauncher  # noqa: E402


APP_LAUNCHER = AppLauncher(ARGS)
SIMULATION_APP = APP_LAUNCHER.app

import isaaclab.sim as sim_utils  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.actuators import IdealPDActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg  # noqa: E402
from isaaclab.sensors import ContactSensor, ContactSensorCfg  # noqa: E402

from holosoma.config_values import robot as robot_values  # noqa: E402
from holosoma.simulator.isaacsim.isaacsim_articulation_cfg import ARTICULATION_CFG  # noqa: E402
from holosoma.utils.inference_helpers import get_control_gains_from_config  # noqa: E402


PHYSICS_DT = 1.0 / 200.0
CONTROL_DECIMATION = 4
CONTROL_DT = PHYSICS_DT * CONTROL_DECIMATION
ROBOT_NAMES = ("left", "right")
RUBBER_HAND_BODIES = {"left_rubber_hand_link", "right_rubber_hand_link"}


def _metadata(recording: np.lib.npyio.NpzFile) -> dict[str, Any]:
    if "_metadata_json" not in recording:
        raise ValueError("recording is missing _metadata_json")
    return json.loads(str(recording["_metadata_json"]))


def _episode_slices(motion_time_step: np.ndarray) -> list[slice]:
    phase = np.asarray(motion_time_step, dtype=np.int64).reshape(-1)
    if phase.size == 0:
        raise ValueError("recording contains no motion frames")
    starts = np.r_[0, np.flatnonzero(phase[1:] <= phase[:-1]) + 1]
    ends = np.r_[starts[1:], phase.size]
    return [slice(int(start), int(end)) for start, end in zip(starts, ends)]


def _select_first_complete_episode(
    recording: np.lib.npyio.NpzFile,
    motion_frames: int,
) -> tuple[slice, np.ndarray]:
    phase = np.asarray(recording["motion_time_step"], dtype=np.int64).reshape(-1)
    for episode_slice in _episode_slices(phase):
        episode_phase = phase[episode_slice]
        if int(episode_phase.max()) >= motion_frames - 1:
            return episode_slice, episode_phase
    maxima = [int(phase[part].max()) for part in _episode_slices(phase)]
    raise ValueError(f"recording has no complete episode; maximum phases: {maxima}")


def _load_trace(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    recording = np.load(path, allow_pickle=True)
    required = {
        "motion_time_step", "dof_pos_target", "pre_dof_pos", "pre_dof_vel",
        "pre_root_pos", "pre_root_quat_xyzw", "ref_object_pos_w",
        "ref_object_quat_xyzw",
    }
    missing = sorted(required.difference(recording.files))
    if missing:
        raise ValueError(f"{path} is missing required channels: {missing}")

    metadata = _metadata(recording)
    motion_frames = int(metadata["motion_time_step_total"])
    episode_slice, phase = _select_first_complete_episode(recording, motion_frames)
    indices = np.arange(len(recording["motion_time_step"]))[episode_slice]
    first = int(indices[0])
    return {
        "path": str(path.resolve()),
        "metadata": metadata,
        "episode_start": first,
        "episode_end": int(indices[-1]),
        "phase": phase.copy(),
        "dof_pos_target": np.asarray(recording["dof_pos_target"][episode_slice], dtype=np.float32),
        "initial_dof_pos": np.asarray(recording["pre_dof_pos"][first], dtype=np.float32),
        "initial_dof_vel": np.asarray(recording["pre_dof_vel"][first], dtype=np.float32),
        "initial_root_pos": np.asarray(recording["pre_root_pos"][first], dtype=np.float32),
        "initial_root_quat_xyzw": np.asarray(recording["pre_root_quat_xyzw"][first], dtype=np.float32),
        "ref_object_pos_w": np.asarray(recording["ref_object_pos_w"][episode_slice], dtype=np.float32),
        "ref_object_quat_xyzw": np.asarray(
            recording["ref_object_quat_xyzw"][episode_slice], dtype=np.float32
        ),
    }


def _robot_spawn_cfg(force_conversion: bool) -> sim_utils.UrdfFileCfg:
    asset = robot_values.g1_29dof_w_object.asset
    return sim_utils.UrdfFileCfg(
        usd_dir="/tmp/holosoma_dual_a1_action_trace_usd",
        asset_path=str(ARGS.robot_urdf.resolve()),
        fix_base=asset.fix_base_link,
        merge_fixed_joints=asset.collapse_fixed_joints,
        replace_cylinders_with_capsules=asset.replace_cylinder_with_capsule,
        force_usd_conversion=force_conversion,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0),
            target_type="none",
        ),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=asset.linear_damping,
            angular_damping=asset.angular_damping,
            max_linear_velocity=asset.max_linear_velocity,
            max_angular_velocity=asset.max_angular_velocity,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    )


def _actuators() -> dict[str, IdealPDActuatorCfg]:
    config = robot_values.g1_29dof_w_object
    return {
        joint_name: IdealPDActuatorCfg(
            joint_names_expr=[joint_name],
            effort_limit=config.dof_effort_limit_list[index],
            velocity_limit=config.dof_vel_limit_list[index],
            stiffness=0,
            damping=0,
            armature=config.dof_armature_list[index],
            friction=config.dof_joint_friction_list[index],
        )
        for index, joint_name in enumerate(config.dof_names)
    }


def _make_robot(name: str, force_conversion: bool) -> Articulation:
    cfg = ARTICULATION_CFG.replace(
        prim_path=f"/World/{name}/Robot",
        spawn=_robot_spawn_cfg(force_conversion),
        init_state=ARTICULATION_CFG.InitialStateCfg(pos=(0.0, 0.0, 0.8)),
        actuators=_actuators(),
    )
    return Articulation(cfg)


def _make_table() -> RigidObject:
    cfg = RigidObjectCfg(
        prim_path="/World/table",
        spawn=sim_utils.UrdfFileCfg(
            usd_dir="/tmp/holosoma_dual_a1_action_trace_usd",
            asset_path=str(ARGS.table_urdf.resolve()),
            fix_base=False,
            replace_cylinders_with_capsules=True,
            force_usd_conversion=True,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.01,
                angular_damping=0.01,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
    )
    return RigidObject(cfg)


def _make_table_contact_sensor(filter_paths: list[str]) -> ContactSensor:
    return ContactSensor(
        ContactSensorCfg(
            prim_path="/World/table/.*",
            filter_prim_paths_expr=filter_paths,
            history_length=1,
            update_period=PHYSICS_DT,
            track_air_time=False,
            force_threshold=ARGS.contact_threshold_n,
            debug_vis=False,
        )
    )


def _xyzw_to_wxyz(quat: np.ndarray) -> np.ndarray:
    return np.asarray(quat, dtype=np.float32)[[3, 0, 1, 2]]


def _yaw_from_wxyz(quat: np.ndarray) -> float:
    w, x, y, z = (float(value) for value in quat)
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _set_robot_initial_state(robot: Articulation, trace: dict[str, Any], joint_ids: list[int]) -> None:
    root_pose = np.concatenate(
        (trace["initial_root_pos"], _xyzw_to_wxyz(trace["initial_root_quat_xyzw"]))
    )
    robot.write_root_pose_to_sim(
        torch.as_tensor(root_pose, dtype=torch.float32, device=robot.device).unsqueeze(0)
    )
    robot.write_root_velocity_to_sim(torch.zeros((1, 6), dtype=torch.float32, device=robot.device))
    joint_pos = torch.as_tensor(
        trace["initial_dof_pos"], dtype=torch.float32, device=robot.device
    ).unsqueeze(0)
    joint_vel = torch.as_tensor(
        trace["initial_dof_vel"], dtype=torch.float32, device=robot.device
    ).unsqueeze(0)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=joint_ids)


def _set_table_physics(table: RigidObject) -> dict[str, Any]:
    env_ids = torch.tensor([0], dtype=torch.int32, device="cpu")
    original_masses = table.root_physx_view.get_masses().clone()
    original_inertias = table.root_physx_view.get_inertias().clone()
    original_mass = float(original_masses.reshape(-1)[0].item())
    scale = ARGS.table_mass_kg / original_mass
    table.root_physx_view.set_masses(original_masses * scale, env_ids)
    table.root_physx_view.set_inertias(original_inertias * scale, env_ids)

    material = table.root_physx_view.get_material_properties().clone()
    material[..., 0] = ARGS.friction
    material[..., 1] = ARGS.friction
    material[..., 2] = 0.0
    table.root_physx_view.set_material_properties(material, env_ids)
    return {
        "urdf_mass_kg": original_mass,
        "runtime_mass_kg": float(table.root_physx_view.get_masses().reshape(-1)[0].item()),
        "inertia_scale": scale,
        "material_properties": table.root_physx_view.get_material_properties().detach().cpu().tolist(),
    }


def run_preflight() -> dict[str, Any]:
    for path in (ARGS.robot_urdf, ARGS.table_urdf):
        if not path.is_file():
            raise FileNotFoundError(path)
    if ARGS.table_mass_kg <= 0.0:
        raise ValueError("table-mass-kg must be positive")
    if ARGS.friction < 0.0:
        raise ValueError("friction must be non-negative")

    traces = {
        "left": _load_trace(ARGS.left_recording),
        "right": _load_trace(ARGS.right_recording),
    }
    if not np.array_equal(traces["left"]["phase"], traces["right"]["phase"]):
        raise ValueError("left and right complete episodes do not share an identical phase sequence")
    for name, trace in traces.items():
        if trace["metadata"].get("dof_names") != robot_values.g1_29dof_w_object.dof_names:
            raise ValueError(f"{name} recording joint order does not match the active G1 config")

    left_ref_pos = traces["left"]["ref_object_pos_w"]
    right_ref_pos = traces["right"]["ref_object_pos_w"]
    left_ref_quat = traces["left"]["ref_object_quat_xyzw"]
    right_ref_quat = traces["right"]["ref_object_quat_xyzw"]
    if not np.allclose(left_ref_pos, right_ref_pos, atol=1.0e-5) or not np.allclose(
        left_ref_quat, right_ref_quat, atol=1.0e-5
    ):
        raise ValueError("left and right recordings do not share one object reference")

    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(
            dt=PHYSICS_DT,
            render_interval=CONTROL_DECIMATION,
            gravity=(0.0, 0.0, -9.81),
            device=ARGS.device,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            physx=sim_utils.PhysxCfg(
                solver_type=1,
                max_position_iteration_count=8,
                max_velocity_iteration_count=4,
                bounce_threshold_velocity=0.5,
                gpu_max_rigid_patch_count=10 * 2**15,
            ),
        )
    )
    ground_cfg = sim_utils.GroundPlaneCfg(
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )
    )
    ground_cfg.func("/World/ground", ground_cfg)
    robots = {
        "left": _make_robot("left", force_conversion=True),
        "right": _make_robot("right", force_conversion=False),
    }
    table = _make_table()
    sensor_body_names = traces["left"]["metadata"].get("contact_sensor_body_names")
    if not sensor_body_names or not RUBBER_HAND_BODIES.issubset(sensor_body_names):
        raise ValueError("recording metadata lacks the rubber-hand contact-body contract")
    contact_filters = [
        (name, body_name, f"/World/{name}/Robot/{body_name}")
        for name in ROBOT_NAMES
        for body_name in sensor_body_names
    ]
    table_contact_sensor = _make_table_contact_sensor(
        [filter_path for _, _, filter_path in contact_filters]
    )
    sim.reset()

    joint_ids: dict[str, list[int]] = {}
    for name, robot in robots.items():
        ids, names = robot.find_joints(robot_values.g1_29dof_w_object.dof_names, preserve_order=True)
        if names != robot_values.g1_29dof_w_object.dof_names:
            raise RuntimeError(f"{name} robot joint order mismatch: {names}")
        joint_ids[name] = ids
        _set_robot_initial_state(robot, traces[name], ids)

    table_physics = _set_table_physics(table)
    initial_table_pose = np.concatenate((left_ref_pos[0], _xyzw_to_wxyz(left_ref_quat[0])))
    table.write_root_pose_to_sim(
        torch.as_tensor(initial_table_pose, dtype=torch.float32, device=table.device).unsqueeze(0)
    )
    table.write_root_velocity_to_sim(torch.zeros((1, 6), dtype=torch.float32, device=table.device))

    kp_values, kd_values = get_control_gains_from_config(robot_values.g1_29dof_w_object)
    kp = torch.tensor(kp_values, dtype=torch.float32, device=table.device).unsqueeze(0)
    kd = torch.tensor(kd_values, dtype=torch.float32, device=table.device).unsqueeze(0)
    effort = torch.tensor(
        robot_values.g1_29dof_w_object.dof_effort_limit_list,
        dtype=torch.float32,
        device=table.device,
    ).unsqueeze(0)

    contact_counts = {name: Counter() for name in ROBOT_NAMES}
    max_contact_force = {name: 0.0 for name in ROBOT_NAMES}
    min_root_height = {name: float(traces[name]["initial_root_pos"][2]) for name in ROBOT_NAMES}
    table_positions: list[np.ndarray] = []
    table_yaws: list[float] = []

    for control_index in range(len(traces["left"]["phase"])):
        targets = {
            name: torch.as_tensor(
                traces[name]["dof_pos_target"][control_index],
                dtype=torch.float32,
                device=robots[name].device,
            ).unsqueeze(0)
            for name in ROBOT_NAMES
        }
        for _ in range(CONTROL_DECIMATION):
            for name, robot in robots.items():
                q = robot.data.joint_pos[:, joint_ids[name]]
                qd = robot.data.joint_vel[:, joint_ids[name]]
                torque = torch.clamp(kp * (targets[name] - q) - kd * qd, -effort, effort)
                robot.set_joint_effort_target(torque, joint_ids=joint_ids[name])
                robot.write_data_to_sim()
            table.write_data_to_sim()
            sim.step(render=False)
            for robot in robots.values():
                robot.update(PHYSICS_DT)
            table.update(PHYSICS_DT)
            table_contact_sensor.update(PHYSICS_DT, force_recompute=True)

        force_matrix = table_contact_sensor.data.force_matrix_w
        if force_matrix is None:
            raise RuntimeError("filtered table-contact force matrix is unavailable")
        if force_matrix.shape[1] != 1 or force_matrix.shape[2] != len(contact_filters):
            raise RuntimeError(
                "unexpected table-contact matrix shape: "
                f"{tuple(force_matrix.shape)} for {len(contact_filters)} filters"
            )
        force_by_filter = torch.linalg.vector_norm(force_matrix[0, 0], dim=-1)
        for filter_index in torch.nonzero(
            force_by_filter > ARGS.contact_threshold_n, as_tuple=False
        ).flatten():
            index = int(filter_index.item())
            name, body_name, _ = contact_filters[index]
            force = float(force_by_filter[index].item())
            contact_counts[name][body_name] += 1
            max_contact_force[name] = max(max_contact_force[name], force)
        for name in ROBOT_NAMES:
            min_root_height[name] = min(
                min_root_height[name], float(robots[name].data.root_pos_w[0, 2].item())
            )
        table_positions.append(table.data.root_pos_w[0].detach().cpu().numpy().copy())
        table_yaws.append(_yaw_from_wxyz(table.data.root_quat_w[0].detach().cpu().numpy()))

    table_positions_np = np.asarray(table_positions)
    initial_position = np.asarray(initial_table_pose[:3])
    reference_displacement = left_ref_pos[-1] - left_ref_pos[0]
    actual_displacement = table_positions_np[-1] - initial_position
    reference_distance = float(np.linalg.norm(reference_displacement[:2]))
    actual_distance = float(np.linalg.norm(actual_displacement[:2]))
    direction_cosine = float(
        np.dot(reference_displacement[:2], actual_displacement[:2])
        / max(reference_distance * actual_distance, 1.0e-8)
    )
    initial_yaw = _yaw_from_wxyz(initial_table_pose[3:7])
    reference_final_yaw = _yaw_from_wxyz(_xyzw_to_wxyz(left_ref_quat[-1]))
    final_yaw_delta = _wrap_angle(table_yaws[-1] - initial_yaw)
    reference_yaw_delta = _wrap_angle(reference_final_yaw - initial_yaw)

    side_reports: dict[str, Any] = {}
    for name in ROBOT_NAMES:
        hand_steps = sum(
            count for body, count in contact_counts[name].items() if body in RUBBER_HAND_BODIES
        )
        other_steps = sum(
            count for body, count in contact_counts[name].items() if body not in RUBBER_HAND_BODIES
        )
        side_reports[name] = {
            "recording": traces[name]["path"],
            "episode_start_index": traces[name]["episode_start"],
            "episode_end_index": traces[name]["episode_end"],
            "phase_first": int(traces[name]["phase"][0]),
            "phase_last": int(traces[name]["phase"][-1]),
            "minimum_root_height_m": min_root_height[name],
            "fell_below_height_threshold": min_root_height[name] < ARGS.fall_height_m,
            "rubber_hand_body_contact_counts": {
                body: int(count) for body, count in sorted(contact_counts[name].items())
                if body in RUBBER_HAND_BODIES
            },
            "other_body_contact_counts": {
                body: int(count) for body, count in sorted(contact_counts[name].items())
                if body not in RUBBER_HAND_BODIES
            },
            "rubber_hand_contact_body_steps": int(hand_steps),
            "other_contact_body_steps": int(other_steps),
            "max_filtered_table_contact_force_n": max_contact_force[name],
        }

    return {
        "purpose": "no-training dual physical action-trace replay; not online policy evaluation or MARL",
        "interpretation_limit": (
            "The two traces were recorded independently. Failure rejects only this open-loop copied-A1 "
            "preflight, not jointly trained multi-agent control."
        ),
        "physics_dt_s": PHYSICS_DT,
        "control_dt_s": CONTROL_DT,
        "control_steps": len(traces["left"]["phase"]),
        "gravity_enabled": True,
        "robot_urdf": str(ARGS.robot_urdf.resolve()),
        "table_urdf": str(ARGS.table_urdf.resolve()),
        "table_physics": table_physics,
        "contact_threshold_n": ARGS.contact_threshold_n,
        "table_motion": {
            "reference_planar_displacement_m": reference_distance,
            "actual_planar_displacement_m": actual_distance,
            "planar_direction_cosine": direction_cosine,
            "reference_yaw_delta_deg": float(np.degrees(reference_yaw_delta)),
            "actual_yaw_delta_deg": float(np.degrees(final_yaw_delta)),
            "endpoint_planar_error_m": float(
                np.linalg.norm(table_positions_np[-1, :2] - left_ref_pos[-1, :2])
            ),
            "maximum_absolute_yaw_delta_deg": float(
                np.degrees(max(abs(_wrap_angle(yaw - initial_yaw)) for yaw in table_yaws))
            ),
        },
        "robots": side_reports,
    }


def _close_simulation_app() -> None:
    try:
        import omni.usd

        context_class = omni.usd.get_context().__class__

        def noop_close_stage(self, *args, **kwargs):  # noqa: ANN001, ARG001
            return True

        context_class.close_stage = noop_close_stage
    except Exception as error:  # pragma: no cover
        print(f"[dual-a1-action-trace] close_stage workaround unavailable: {error}")

    try:
        sim_context = sim_utils.SimulationContext.instance()
        if sim_context is not None:
            sim_context._disable_app_control_on_stop_handle = True
    except Exception as error:  # pragma: no cover
        print(f"[dual-a1-action-trace] stop-callback workaround unavailable: {error}")

    SIMULATION_APP.close(wait_for_replicator=False)


def main() -> int:
    try:
        report = run_preflight()
        ARGS.report.parent.mkdir(parents=True, exist_ok=True)
        ARGS.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, indent=2, sort_keys=True))
        print(f"[dual-a1-action-trace] report: {ARGS.report}")
        return 0
    finally:
        _close_simulation_app()


if __name__ == "__main__":
    raise SystemExit(main())
