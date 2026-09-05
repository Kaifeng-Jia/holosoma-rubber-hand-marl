"""Integrated yaw-potential and first-success bonus contracts."""

import math
from types import SimpleNamespace

import torch

from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg
from holosoma.managers.command.terms.demo4_rotate import Demo4RotateMotionCommand
from holosoma.managers.reward.manager import RewardManager


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
    command.object_indices_in_simulator = torch.tensor([0])
    simulator = SimpleNamespace(all_root_states=torch.zeros(1, 13))
    simulator.all_root_states[:, 3:7] = _upright_table_quaternion(0.0)
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        dt=0.02,
        max_episode_length_s=6.32,
        simulator=simulator,
        logger=None,
    )
    command.env = env
    env.command_manager = _CommandManager(command)
    return env


def test_complete_positive_yaw_contributes_ten_plus_five_once() -> None:
    cfg = RewardManagerCfg(
        terms={
            "progress": RewardTermCfg(
                func=(
                    "holosoma.managers.reward.terms.demo4_rotate:"
                    "SignedYawPotentialDelta"
                ),
                weight=10.0,
            ),
            "success": RewardTermCfg(
                func=(
                    "holosoma.managers.reward.terms.demo4_rotate:"
                    "FirstYawGoalBonus"
                ),
                weight=5.0,
            ),
        }
    )
    env = _env()
    manager = RewardManager(cfg, env, "cpu")

    torch.testing.assert_close(manager.compute(env.dt), torch.tensor([0.0]))
    _set_table_yaw(env, math.pi / 4.0)
    torch.testing.assert_close(manager.compute(env.dt), torch.tensor([5.0]), atol=1e-6, rtol=0.0)
    _set_table_yaw(env, math.pi / 2.0)
    torch.testing.assert_close(manager.compute(env.dt), torch.tensor([10.0]), atol=1e-6, rtol=0.0)
    torch.testing.assert_close(manager.compute(env.dt), torch.tensor([0.0]))


def test_reversing_yaw_loses_potential_without_repeating_bonus() -> None:
    cfg = RewardManagerCfg(
        terms={
            "progress": RewardTermCfg(
                func=(
                    "holosoma.managers.reward.terms.demo4_rotate:"
                    "SignedYawPotentialDelta"
                ),
                weight=10.0,
            ),
            "success": RewardTermCfg(
                func=(
                    "holosoma.managers.reward.terms.demo4_rotate:"
                    "FirstYawGoalBonus"
                ),
                weight=5.0,
            ),
        }
    )
    env = _env()
    manager = RewardManager(cfg, env, "cpu")
    _set_table_yaw(env, math.pi / 2.0)
    torch.testing.assert_close(manager.compute(env.dt), torch.tensor([15.0]), atol=1e-6, rtol=0.0)
    _set_table_yaw(env, math.pi / 4.0)
    torch.testing.assert_close(manager.compute(env.dt), torch.tensor([-5.0]), atol=1e-6, rtol=0.0)


def test_negative_rotation_cannot_wrap_into_positive_success() -> None:
    cfg = RewardManagerCfg(
        terms={
            "progress": RewardTermCfg(
                func=(
                    "holosoma.managers.reward.terms.demo4_rotate:"
                    "SignedYawPotentialDelta"
                ),
                weight=10.0,
            ),
            "success": RewardTermCfg(
                func=(
                    "holosoma.managers.reward.terms.demo4_rotate:"
                    "FirstYawGoalBonus"
                ),
                weight=5.0,
            ),
        }
    )
    env = _env()
    manager = RewardManager(cfg, env, "cpu")

    _set_table_yaw(env, math.radians(-170.0))
    torch.testing.assert_close(
        manager.compute(env.dt),
        torch.tensor([-10.0]),
        atol=1.0e-6,
        rtol=0.0,
    )
    _set_table_yaw(env, math.radians(170.0))
    torch.testing.assert_close(manager.compute(env.dt), torch.tensor([0.0]))
    assert env.command_manager.command.signed_yaw_progress_radians.item() < -math.pi


def test_unsafe_terminal_suppresses_success_bonus() -> None:
    cfg = RewardManagerCfg(
        terms={
            "progress": RewardTermCfg(
                func=(
                    "holosoma.managers.reward.terms.demo4_rotate:"
                    "SignedYawPotentialDelta"
                ),
                weight=10.0,
            ),
            "success": RewardTermCfg(
                func=(
                    "holosoma.managers.reward.terms.demo4_rotate:"
                    "FirstYawGoalBonus"
                ),
                weight=5.0,
            ),
        }
    )
    env = _env()
    _set_table_yaw(env, math.pi / 2.0)
    env.termination_manager = SimpleNamespace(
        term_results={
            "yaw_goal_success": torch.tensor([False]),
            "clear_robot_fall": torch.tensor([True]),
            "table_physical_safety": torch.tensor([False]),
        }
    )
    manager = RewardManager(cfg, env, "cpu")

    torch.testing.assert_close(manager.compute(env.dt), torch.tensor([0.0]))
