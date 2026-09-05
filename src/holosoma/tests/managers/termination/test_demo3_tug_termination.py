"""Demo 3 terminates on horizon or unmistakable robot failure only."""

from types import SimpleNamespace

import torch

from holosoma.managers.command.terms.demo3_tug import Demo3TugMotionCommand
from holosoma.managers.termination.terms.demo3_tug import (
    any_robot_clearly_fallen,
    reference_horizon_reached,
)


class _CommandManager:
    def __init__(self, command):
        self.command = command

    def get_state(self, name):
        return self.command if name == "paired_motion_command" else None


def _env():
    command = object.__new__(Demo3TugMotionCommand)
    command.ref_body_index = 0
    command.reference = SimpleNamespace(num_frames=317)
    position = torch.tensor([[[0.0, 0.0, 0.8], [0.0, 0.0, 0.8]]])
    orientation = torch.zeros(1, 2, 1, 4)
    orientation[..., 3] = 1.0
    return SimpleNamespace(
        num_envs=1,
        num_agents=2,
        episode_length_buf=torch.tensor([316]),
        simulator=SimpleNamespace(
            agent_rigid_body_pos=position[:, :, None],
            agent_rigid_body_rot=orientation,
        ),
        command_manager=_CommandManager(command),
    )


def test_reference_horizon_is_a_shared_timeout() -> None:
    env = _env()
    assert not reference_horizon_reached(env).item()
    env.episode_length_buf[:] = 317
    assert reference_horizon_reached(env).item()


def test_table_deviation_is_not_part_of_clear_robot_fall() -> None:
    env = _env()
    assert not any_robot_clearly_fallen(
        env, minimum_ref_body_height=0.25, maximum_gravity_z=-0.2
    ).item()
    env.simulator.agent_rigid_body_pos[0, 1, 0, 2] = 0.1
    assert any_robot_clearly_fallen(
        env, minimum_ref_body_height=0.25, maximum_gravity_z=-0.2
    ).item()
