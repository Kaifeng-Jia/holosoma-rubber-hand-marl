"""Non-looping, reset-only table command contract for Demo 4."""

import math

from types import SimpleNamespace
from unittest.mock import patch

import torch

from holosoma.managers.command.terms.demo4_rotate import Demo4RotateMotionCommand
from holosoma.managers.command.terms.marl import PairedA1MotionCommand


def _yaw_quaternion(angle: float) -> torch.Tensor:
    return torch.tensor([[0.0, 0.0, math.sin(angle / 2.0), math.cos(angle / 2.0)]])


def test_demo4_command_clamps_at_last_frame_without_looping() -> None:
    command = object.__new__(Demo4RotateMotionCommand)
    command.time_steps = torch.tensor([0, 314, 315], dtype=torch.long)
    command.reference = SimpleNamespace(num_frames=316)

    command.step()
    torch.testing.assert_close(command.time_steps, torch.tensor([1, 315, 315]))
    command.step()
    torch.testing.assert_close(command.time_steps, torch.tensor([2, 315, 315]))


def test_table_write_counter_changes_only_on_reference_reset_write() -> None:
    command = object.__new__(Demo4RotateMotionCommand)
    command.time_steps = torch.tensor([0, 0], dtype=torch.long)
    command.reference = SimpleNamespace(num_frames=316)
    command.table_reset_write_count = torch.zeros(2, dtype=torch.long)
    env_ids = torch.tensor([1], dtype=torch.long)

    with patch.object(PairedA1MotionCommand, "_write_reference_state") as base_write:
        command._write_reference_state(env_ids)
        base_write.assert_called_once_with(env_ids)

    torch.testing.assert_close(command.table_reset_write_count, torch.tensor([0, 1]))
    command.step()
    command.step()
    torch.testing.assert_close(command.table_reset_write_count, torch.tensor([0, 1]))


def test_yaw_progress_is_unwrapped_across_negative_pi_boundary() -> None:
    command = object.__new__(Demo4RotateMotionCommand)
    command.last_object_heading_w = torch.tensor([[1.0, 0.0]])
    command.unwrapped_yaw_progress_radians = torch.zeros(1)
    command.object_indices_in_simulator = torch.tensor([0])
    simulator = SimpleNamespace(all_root_states=torch.zeros(1, 13))
    simulator.all_root_states[:, 6] = 1.0
    command.env = SimpleNamespace(simulator=simulator)

    simulator.all_root_states[:, 3:7] = _yaw_quaternion(math.radians(-170.0))
    command.update_yaw_progress()
    simulator.all_root_states[:, 3:7] = _yaw_quaternion(math.radians(170.0))
    command.update_yaw_progress()

    torch.testing.assert_close(
        command.signed_yaw_progress_radians,
        torch.tensor([math.radians(-190.0)]),
        atol=1.0e-6,
        rtol=0.0,
    )
