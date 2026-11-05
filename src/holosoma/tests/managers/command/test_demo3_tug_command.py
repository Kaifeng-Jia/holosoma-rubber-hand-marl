"""Non-looping Demo 3 command contract."""

from types import SimpleNamespace

import torch

from holosoma.managers.command.terms.demo3_tug import Demo3TugMotionCommand


def test_demo3_command_clamps_at_last_frame_instead_of_resetting() -> None:
    command = object.__new__(Demo3TugMotionCommand)
    command.time_steps = torch.tensor([0, 315, 316], dtype=torch.long)
    command.reference = SimpleNamespace(num_frames=317)

    command.step()
    torch.testing.assert_close(command.time_steps, torch.tensor([1, 316, 316]))
    command.step()
    torch.testing.assert_close(command.time_steps, torch.tensor([2, 316, 316]))


def test_signed_progress_is_opposite_for_the_same_table_displacement() -> None:
    command = object.__new__(Demo3TugMotionCommand)
    command.agent_pull_axis_w = torch.tensor([[[1.0, 0.0], [-1.0, 0.0]]])
    command.reset_object_pos_w = torch.tensor([[0.2, -0.1, 0.0]])
    command.object_indices_in_simulator = torch.tensor([0])
    command.env = SimpleNamespace(
        simulator=SimpleNamespace(
            all_root_states=torch.tensor(
                [[0.5, -0.1, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
            )
        )
    )

    torch.testing.assert_close(
        command.signed_object_progress,
        torch.tensor([[0.3, -0.3]]),
    )
