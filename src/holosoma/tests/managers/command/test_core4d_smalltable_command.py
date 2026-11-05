"""Runtime and non-looping command contracts for CORE4D small-table."""

from types import SimpleNamespace

import numpy as np
import torch

from holosoma.config_values.marl.g1.core4d_smalltable_command import (
    CORE4D_SMALLTABLE_REFERENCE_FPS,
    CORE4D_SMALLTABLE_REFERENCE_FRAMES,
    CORE4D_SMALLTABLE_RUNTIME_REFERENCE_FILE,
)
from holosoma.envs.marl.core4d_smalltable_reference import (
    Core4DSmallTableReference,
)
from holosoma.managers.command.terms.core4d_smalltable import (
    Core4DSmallTableMotionCommand,
    object_com_velocity_to_origin_velocity,
    object_origin_velocity_to_com_velocity,
)
from holosoma.managers.termination.terms.core4d_smalltable import (
    reference_horizon_reached,
)
from holosoma.utils.path import resolve_data_file_path


def test_core4d_runtime_loads_complete_velocity_contract() -> None:
    path = resolve_data_file_path(CORE4D_SMALLTABLE_RUNTIME_REFERENCE_FILE)
    with np.load(path, allow_pickle=False) as data:
        body_names = data["body_names"].astype(str).tolist()
        joint_names = data["joint_names"].astype(str).tolist()
    reference = Core4DSmallTableReference(
        CORE4D_SMALLTABLE_RUNTIME_REFERENCE_FILE,
        body_names,
        joint_names,
        device="cpu",
    )

    assert reference.num_frames == CORE4D_SMALLTABLE_REFERENCE_FRAMES
    assert reference.fps == CORE4D_SMALLTABLE_REFERENCE_FPS
    assert reference.object_ang_vel_w.shape == (
        CORE4D_SMALLTABLE_REFERENCE_FRAMES,
        3,
    )
    assert torch.isfinite(reference.object_ang_vel_w).all()
    sample = reference.sample(torch.tensor([0, reference.num_frames - 1]))
    assert sample["object_ang_vel_w"].shape == (2, 3)


def test_core4d_command_clamps_and_horizon_terminates_without_looping() -> None:
    command = object.__new__(Core4DSmallTableMotionCommand)
    command.time_steps = torch.tensor([0, 685, 686], dtype=torch.long)
    command.reference = SimpleNamespace(num_frames=687)

    command.step()
    torch.testing.assert_close(command.time_steps, torch.tensor([1, 686, 686]))
    command.step()
    torch.testing.assert_close(command.time_steps, torch.tensor([2, 686, 686]))

    env = SimpleNamespace(
        command_manager=SimpleNamespace(
            get_state=lambda name: command if name == "paired_motion_command" else None
        )
    )
    torch.testing.assert_close(
        reference_horizon_reached(env),
        torch.tensor([False, True, True]),
    )


def test_object_origin_and_com_velocity_conversion_round_trips() -> None:
    origin_velocity = torch.tensor([[0.4, -0.2, 0.1]])
    angular_velocity = torch.tensor([[0.0, 0.0, 2.0]])
    object_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    com_position_b = torch.tensor([[0.0, 0.126415, 0.0]])

    com_velocity = object_origin_velocity_to_com_velocity(
        origin_velocity,
        angular_velocity,
        object_quat,
        com_position_b,
    )
    torch.testing.assert_close(
        com_velocity,
        torch.tensor([[0.14717, -0.2, 0.1]]),
        rtol=0.0,
        atol=1.0e-6,
    )
    torch.testing.assert_close(
        object_com_velocity_to_origin_velocity(
            com_velocity,
            angular_velocity,
            object_quat,
            com_position_b,
        ),
        origin_velocity,
        rtol=0.0,
        atol=1.0e-7,
    )
