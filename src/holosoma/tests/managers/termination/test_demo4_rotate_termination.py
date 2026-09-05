"""Success, horizon, fall, and broad table-safety contracts for Demo 4."""

import math
from types import SimpleNamespace

import torch

from holosoma.managers.command.terms.demo4_rotate import Demo4RotateMotionCommand
from holosoma.managers.termination.terms.demo4_rotate import (
    any_robot_clearly_fallen,
    reference_horizon_reached,
    table_physical_safety_exceeded,
    yaw_goal_reached,
)


class _CommandManager:
    def __init__(self, command):
        self.command = command

    def get_state(self, name):
        return self.command if name == "paired_motion_command" else None


def _upright_table_quaternion(angle: float) -> torch.Tensor:
    half_yaw = angle / 2.0
    half_roll = math.pi / 4.0
    sy = math.sin(half_yaw)
    cy = math.cos(half_yaw)
    sr = math.sin(half_roll)
    cr = math.cos(half_roll)
    return torch.tensor([[cy * sr, sy * sr, sy * cr, cy * cr]])


def _set_table_yaw(env, angle: float) -> None:
    env.simulator.all_root_states[:, 3:7] = _upright_table_quaternion(angle)
    env.command_manager.command.update_yaw_progress()


def _env():
    command = object.__new__(Demo4RotateMotionCommand)
    command.goal_yaw_radians = math.pi / 2.0
    command.maximum_success_tilt_radians = math.radians(60.0)
    command.reset_object_heading_w = torch.tensor([[1.0, 0.0]])
    command.last_object_heading_w = torch.tensor([[1.0, 0.0]])
    command.unwrapped_yaw_progress_radians = torch.zeros(1)
    command.reset_object_pos_w = torch.tensor([[0.0, 0.0, 0.368]])
    command.object_indices_in_simulator = torch.tensor([0])
    command.ref_body_index = 0
    command.reference = SimpleNamespace(num_frames=316)
    object_states = torch.zeros(1, 13)
    object_states[:, 2] = 0.368
    # Pull-table URDF uses local Y as vertical: +90 degrees around local X
    # maps that axis to world Z while keeping local X as its planar heading.
    object_states[:, 3] = math.sqrt(0.5)
    object_states[:, 6] = math.sqrt(0.5)
    body_pos = torch.tensor([[[[0.0, 0.0, 0.8]], [[0.0, 0.0, 0.8]]]])
    body_quat = torch.zeros(1, 2, 1, 4)
    body_quat[..., 3] = 1.0
    env = SimpleNamespace(
        num_envs=1,
        num_agents=2,
        device="cpu",
        episode_length_buf=torch.tensor([315]),
        simulator=SimpleNamespace(
            all_root_states=object_states,
            agent_rigid_body_pos=body_pos,
            agent_rigid_body_rot=body_quat,
        ),
    )
    command.env = env
    env.command_manager = _CommandManager(command)
    return env


def test_positive_goal_is_success_and_negative_goal_is_not() -> None:
    env = _env()
    _set_table_yaw(env, math.pi / 2.0)
    assert yaw_goal_reached(env).item()

    negative_env = _env()
    _set_table_yaw(negative_env, -math.pi / 2.0)
    assert not yaw_goal_reached(negative_env).item()


def test_goal_requires_safe_tilt_and_negative_wrap_is_not_success() -> None:
    tilted = _env()
    tilted.command_manager.command.unwrapped_yaw_progress_radians[:] = math.pi / 2.0
    tilted.simulator.all_root_states[:, 3:7] = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    assert not yaw_goal_reached(tilted).item()

    wrapped = _env()
    _set_table_yaw(wrapped, math.radians(-170.0))
    _set_table_yaw(wrapped, math.radians(170.0))
    assert wrapped.command_manager.command.signed_yaw_progress_radians.item() < -math.pi
    assert not yaw_goal_reached(wrapped).item()


def test_horizon_is_exactly_316_frames() -> None:
    env = _env()
    assert not reference_horizon_reached(env).item()
    env.episode_length_buf[:] = 316
    assert reference_horizon_reached(env).item()


def test_robot_and_table_safety_are_loose_physical_guards() -> None:
    env = _env()
    assert not any_robot_clearly_fallen(
        env, minimum_ref_body_height=0.25, maximum_gravity_z=-0.2
    ).item()
    assert not table_physical_safety_exceeded(
        env,
        maximum_tilt_degrees=60.0,
        maximum_xy_drift_m=3.0,
        minimum_height_m=0.05,
        maximum_height_m=1.5,
    ).item()

    env.simulator.all_root_states[:, 0] = 3.1
    assert table_physical_safety_exceeded(
        env,
        maximum_tilt_degrees=60.0,
        maximum_xy_drift_m=3.0,
        minimum_height_m=0.05,
        maximum_height_m=1.5,
    ).item()

    env.simulator.all_root_states[:, 0] = 0.0
    env.simulator.agent_rigid_body_pos[0, 1, 0, 2] = 0.1
    assert any_robot_clearly_fallen(
        env, minimum_ref_body_height=0.25, maximum_gravity_z=-0.2
    ).item()
