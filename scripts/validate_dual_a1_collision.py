#!/usr/bin/env python3
"""Run a geometry-only Isaac Sim collision gate for the dual-A1 reset pose.

This tool deliberately disables gravity and does not apply control.  Two A1
robots are placed around the wide table, while two identical control copies are
placed far away.  Excess contact force or drift in the desired pair relative to
the controls reveals reset-time interpenetration without conflating it with
gravity, ground contact, or policy behavior.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "holosoma"))
sys.path.insert(0, str(REPO_ROOT / "src" / "holosoma_retargeting"))

DEFAULT_MOTION = (
    REPO_ROOT
    / "src"
    / "holosoma"
    / "holosoma"
    / "data"
    / "motions"
    / "g1_29dof"
    / "whole_body_tracking"
    / "rubber_hand_largetable_v1"
    / "a1"
    / "sub6_largetable_033_a1_mj_fps50_w_obj.npz"
)
DEFAULT_ROBOT_URDF = (
    REPO_ROOT
    / "src"
    / "holosoma"
    / "holosoma"
    / "data"
    / "robots"
    / "g1"
    / "main_mesh_collision_rubberhand.urdf"
)
DEFAULT_TABLE_URDF = (
    REPO_ROOT
    / "src"
    / "holosoma"
    / "holosoma"
    / "data"
    / "motions"
    / "g1_29dof"
    / "whole_body_tracking"
    / "objects_widetable.urdf"
)


def parse_args() -> argparse.Namespace:
    """Parse preflight and Isaac Sim launcher arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-npz", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--robot-urdf", type=Path, default=DEFAULT_ROBOT_URDF)
    parser.add_argument("--table-urdf", type=Path, default=DEFAULT_TABLE_URDF)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--lateral-spacing", type=float, default=0.8)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--force-threshold-n", type=float, default=5.0)
    parser.add_argument("--drift-threshold-m", type=float, default=5.0e-4)
    parser.add_argument("--control-offset-m", type=float, default=4.0)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("/tmp/dual_a1_collision_report.json"),
    )

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


ARGS = parse_args()

from isaaclab.app import AppLauncher  # noqa: E402


APP_LAUNCHER = AppLauncher(ARGS)
SIMULATION_APP = APP_LAUNCHER.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import IdealPDActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg  # noqa: E402
from isaaclab.sensors import ContactSensor, ContactSensorCfg  # noqa: E402

from holosoma.config_values import robot as robot_values  # noqa: E402
from holosoma.simulator.isaacsim.isaacsim_articulation_cfg import ARTICULATION_CFG  # noqa: E402
from holosoma_retargeting.dual_a1_layout import load_a1_motion, shifted_robot_positions  # noqa: E402


DT = 1.0 / 200.0
ROBOT_NAMES = ("desired_0", "desired_1", "control_0", "control_1")
REQUIRED_RUBBER_HAND_BODIES = {"left_rubber_hand_link", "right_rubber_hand_link"}


def _robot_spawn_cfg(force_conversion: bool) -> sim_utils.UrdfFileCfg:
    robot_cfg = robot_values.g1_29dof_w_object
    asset = robot_cfg.asset
    return sim_utils.UrdfFileCfg(
        usd_dir="/tmp/holosoma_dual_a1_collision_usd",
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
            disable_gravity=True,
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
    robot_cfg = robot_values.g1_29dof_w_object
    return {
        joint_name: IdealPDActuatorCfg(
            joint_names_expr=[joint_name],
            effort_limit=robot_cfg.dof_effort_limit_list[index],
            velocity_limit=robot_cfg.dof_vel_limit_list[index],
            stiffness=0,
            damping=0,
            armature=robot_cfg.dof_armature_list[index],
            friction=robot_cfg.dof_joint_friction_list[index],
        )
        for index, joint_name in enumerate(robot_cfg.dof_names)
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
            usd_dir="/tmp/holosoma_dual_a1_collision_usd",
            asset_path=str(ARGS.table_urdf.resolve()),
            fix_base=False,
            replace_cylinders_with_capsules=True,
            force_usd_conversion=True,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
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


def _make_sensor(name: str) -> ContactSensor:
    return ContactSensor(
        ContactSensorCfg(
            prim_path=f"/World/{name}/Robot/.*",
            history_length=1,
            update_period=DT,
            track_air_time=False,
            force_threshold=ARGS.force_threshold_n,
            debug_vis=False,
        )
    )


def _validate_inputs(joint_pos: np.ndarray) -> int:
    for path in (ARGS.motion_npz, ARGS.robot_urdf, ARGS.table_urdf):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not 0 <= ARGS.frame_index < joint_pos.shape[0]:
        raise ValueError(f"frame-index {ARGS.frame_index} outside [0, {joint_pos.shape[0] - 1}]")
    if ARGS.steps <= 0:
        raise ValueError("steps must be positive")
    if ARGS.control_offset_m <= ARGS.lateral_spacing:
        raise ValueError("control-offset-m must exceed lateral-spacing")
    return ARGS.frame_index


def _set_robot_state(robot: Articulation, root_pose: np.ndarray, joint_pose: np.ndarray) -> None:
    robot_cfg = robot_values.g1_29dof_w_object
    joint_ids, joint_names = robot.find_joints(robot_cfg.dof_names, preserve_order=True)
    if joint_names != robot_cfg.dof_names:
        raise RuntimeError(f"Robot joint order mismatch: {joint_names}")
    root = torch.as_tensor(root_pose, dtype=torch.float32, device=robot.device).unsqueeze(0)
    joints = torch.as_tensor(joint_pose, dtype=torch.float32, device=robot.device).unsqueeze(0)
    robot.write_root_pose_to_sim(root)
    robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=robot.device))
    robot.write_joint_state_to_sim(joints, torch.zeros_like(joints), joint_ids=joint_ids)


def _max_force(sensor: ContactSensor) -> tuple[float, str]:
    force_norm = torch.linalg.vector_norm(sensor.data.net_forces_w[0], dim=-1)
    index = int(torch.argmax(force_norm).item())
    return float(force_norm[index].item()), sensor.body_names[index]


def run_preflight() -> tuple[dict[str, object], bool]:
    joint_pos, object_pos, object_quat, _ = load_a1_motion(ARGS.motion_npz)
    frame = _validate_inputs(joint_pos)

    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(
            dt=DT,
            render_interval=1,
            gravity=(0.0, 0.0, 0.0),
            device=ARGS.device,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            physx=sim_utils.PhysxCfg(
                solver_type=1,
                max_position_iteration_count=8,
                max_velocity_iteration_count=4,
                bounce_threshold_velocity=0.5,
            ),
        )
    )

    robots = {
        name: _make_robot(name, force_conversion=index == 0)
        for index, name in enumerate(ROBOT_NAMES)
    }
    table = _make_table()
    sensors = {name: _make_sensor(name) for name in ROBOT_NAMES}
    sim.reset()

    robot_body_contract = {
        name: sorted(REQUIRED_RUBBER_HAND_BODIES.intersection(robot.body_names))
        for name, robot in robots.items()
    }
    sensor_body_contract = {
        name: sorted(REQUIRED_RUBBER_HAND_BODIES.intersection(sensor.body_names))
        for name, sensor in sensors.items()
    }
    rubber_hand_asset_passed = all(
        set(robot_body_contract[name]) == REQUIRED_RUBBER_HAND_BODIES
        and set(sensor_body_contract[name]) == REQUIRED_RUBBER_HAND_BODIES
        for name in ROBOT_NAMES
    )

    qpos = joint_pos[frame]
    desired_0, desired_1 = shifted_robot_positions(
        qpos[:3],
        object_quat[frame],
        ARGS.lateral_spacing,
    )
    roots = {
        "desired_0": desired_0,
        "desired_1": desired_1,
        "control_0": desired_0 + np.array([ARGS.control_offset_m, 0.0, 0.0]),
        "control_1": desired_1 + np.array([-ARGS.control_offset_m, 0.0, 0.0]),
    }
    initial_roots = {name: value.copy() for name, value in roots.items()}
    for name, robot in robots.items():
        _set_robot_state(robot, np.concatenate((roots[name], qpos[3:7])), qpos[7:])
    table_pose = torch.as_tensor(
        np.concatenate((object_pos[frame], object_quat[frame])),
        dtype=torch.float32,
        device=table.device,
    ).unsqueeze(0)
    table.write_root_pose_to_sim(table_pose)
    table.write_root_velocity_to_sim(torch.zeros((1, 6), device=table.device))

    max_force = {name: 0.0 for name in ROBOT_NAMES}
    max_force_body = {name: "" for name in ROBOT_NAMES}
    max_drift = {name: 0.0 for name in ROBOT_NAMES}
    initial_table_pos = object_pos[frame].copy()
    max_table_drift = 0.0

    for _ in range(ARGS.steps):
        for robot in robots.values():
            robot.write_data_to_sim()
        table.write_data_to_sim()
        sim.step(render=False)
        for robot in robots.values():
            robot.update(DT)
        table.update(DT)
        for name, sensor in sensors.items():
            sensor.update(DT, force_recompute=True)
            force, body = _max_force(sensor)
            if force > max_force[name]:
                max_force[name] = force
                max_force_body[name] = body
            current_root = robots[name].data.root_pos_w[0].detach().cpu().numpy()
            max_drift[name] = max(
                max_drift[name],
                float(np.linalg.norm(current_root - initial_roots[name])),
            )
        current_table = table.data.root_pos_w[0].detach().cpu().numpy()
        max_table_drift = max(max_table_drift, float(np.linalg.norm(current_table - initial_table_pos)))

    comparisons = {}
    passed = rubber_hand_asset_passed
    for index in range(2):
        desired = f"desired_{index}"
        control = f"control_{index}"
        force_excess = max(0.0, max_force[desired] - max_force[control])
        drift_excess = max(0.0, max_drift[desired] - max_drift[control])
        pair_passed = force_excess <= ARGS.force_threshold_n and drift_excess <= ARGS.drift_threshold_m
        passed &= pair_passed
        comparisons[desired] = {
            "control": control,
            "force_excess_n": force_excess,
            "drift_excess_m": drift_excess,
            "passed": pair_passed,
        }
    passed &= max_table_drift <= ARGS.drift_threshold_m

    report: dict[str, object] = {
        "purpose": "geometry-only reset collision gate; not training or physical-feasibility evidence",
        "passed": passed,
        "frame_index": frame,
        "steps": ARGS.steps,
        "dt_s": DT,
        "gravity_enabled": False,
        "rubber_hand_asset_passed": rubber_hand_asset_passed,
        "robot_rubber_hand_bodies": robot_body_contract,
        "sensor_rubber_hand_bodies": sensor_body_contract,
        "lateral_spacing_m": ARGS.lateral_spacing,
        "robot_urdf": str(ARGS.robot_urdf.resolve()),
        "table_urdf": str(ARGS.table_urdf.resolve()),
        "thresholds": {
            "force_excess_n": ARGS.force_threshold_n,
            "drift_excess_m": ARGS.drift_threshold_m,
        },
        "max_force_n": max_force,
        "max_force_body": max_force_body,
        "max_root_drift_m": max_drift,
        "max_table_drift_m": max_table_drift,
        "comparisons": comparisons,
    }
    return report, passed


def _close_simulation_app() -> None:
    """Apply the project's known Isaac Sim headless-shutdown workarounds."""
    try:
        import omni.usd

        context_class = omni.usd.get_context().__class__

        def noop_close_stage(self, *args, **kwargs):  # noqa: ANN001, ARG001
            return True

        context_class.close_stage = noop_close_stage
    except Exception as error:  # pragma: no cover - defensive shutdown path
        print(f"[dual-a1-collision] close_stage workaround unavailable: {error}")

    try:
        sim_context = sim_utils.SimulationContext.instance()
        if sim_context is not None:
            sim_context._disable_app_control_on_stop_handle = True
    except Exception as error:  # pragma: no cover - defensive shutdown path
        print(f"[dual-a1-collision] stop-callback workaround unavailable: {error}")

    SIMULATION_APP.close(wait_for_replicator=False)


def main() -> int:
    try:
        report, passed = run_preflight()
        ARGS.report.parent.mkdir(parents=True, exist_ok=True)
        ARGS.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, indent=2, sort_keys=True))
        print(f"[dual-a1-collision] report: {ARGS.report}")
        return 0 if passed else 1
    finally:
        _close_simulation_app()


if __name__ == "__main__":
    raise SystemExit(main())
