"""Actor/table/team-critic observation contracts for Demo 4."""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch

from holosoma.config_values.marl.g1.demo4_observation import (
    DEMO4_ACTOR_OBS_DIM,
    DEMO4_CRITIC_OBS_DIM,
    DEMO4_TABLE_OBS_DIM,
    g1_29dof_demo4_rotate_observation,
)
from holosoma.managers.observation.manager import ObservationManager
from holosoma.managers.observation.terms.demo4_rotate import (
    table_linear_velocity_b,
    table_relative_position_b,
    table_relative_yaw_sin_cos,
)


class _CommandManager:
    def __init__(self, command):
        self.command = command

    def get_state(self, name):
        return self.command if name == "paired_motion_command" else None


def _yaw_quaternion(angle: float) -> torch.Tensor:
    return torch.tensor([0.0, 0.0, math.sin(angle / 2.0), math.cos(angle / 2.0)])


def _make_env(num_envs: int = 1):
    agents = 2
    dofs = 29
    bodies = 14
    roots = torch.zeros(num_envs, agents, 13)
    roots[..., 6] = 1.0
    roots[:, 1, 0] = 0.8
    body_pos = torch.zeros(num_envs, agents, bodies, 3)
    body_quat = torch.zeros(num_envs, agents, bodies, 4)
    body_quat[..., 3] = 1.0
    object_states = torch.zeros(num_envs, 13)
    object_states[..., 6] = 1.0
    simulator = SimpleNamespace(
        agent_root_states=roots,
        agent_dof_pos=torch.zeros(num_envs, agents, dofs),
        agent_dof_vel=torch.zeros(num_envs, agents, dofs),
        agent_rigid_body_pos=body_pos,
        agent_rigid_body_rot=body_quat,
        all_root_states=object_states,
    )
    command = SimpleNamespace(
        command=torch.zeros(num_envs, agents, dofs * 2),
        ref_body_index=0,
        agent_ref_pos_w=body_pos[:, :, 0].clone(),
        agent_ref_quat_w=body_quat[:, :, 0].clone(),
        simulator_agent_body_pos_w=body_pos,
        simulator_agent_body_quat_w=body_quat,
        simulator_object_pos_w=object_states[:, :3],
        simulator_object_quat_w=object_states[:, 3:7],
        object_pos_w=torch.zeros(num_envs, 3),
        object_quat_w=torch.tensor([0.0, 0.0, 0.0, 1.0]).repeat(num_envs, 1),
        object_lin_vel_w=torch.zeros(num_envs, 3),
        object_indices_in_simulator=torch.arange(num_envs),
        time_steps=torch.zeros(num_envs, dtype=torch.long),
        reference=SimpleNamespace(num_frames=316),
    )
    return SimpleNamespace(
        num_envs=num_envs,
        num_agents=agents,
        num_dof=dofs,
        device="cpu",
        simulator=simulator,
        default_dof_pos=torch.zeros(num_envs, dofs),
        action_manager=SimpleNamespace(action=torch.zeros(num_envs, agents * dofs)),
        command_manager=_CommandManager(command),
        logger=None,
    )


def _table_obs(env) -> torch.Tensor:
    return torch.cat(
        (
            table_relative_position_b(env),
            table_linear_velocity_b(env),
            table_relative_yaw_sin_cos(env),
        ),
        dim=-1,
    )


def test_demo4_actor_has_164_channels_and_plan5_team_critic() -> None:
    env = _make_env(num_envs=3)
    observations = ObservationManager(
        g1_29dof_demo4_rotate_observation, env, "cpu"
    ).compute()
    actor_input = torch.cat(
        (
            observations["actor_obs"],
            observations["teammate_obs"],
            observations["table_obs"],
        ),
        dim=-1,
    )

    assert observations["actor_obs"].shape == (3, 2, 154)
    assert observations["teammate_obs"].shape == (3, 2, 4)
    assert observations["table_obs"].shape == (3, 2, DEMO4_TABLE_OBS_DIM)
    assert actor_input.shape == (3, 2, DEMO4_ACTOR_OBS_DIM)
    assert observations["critic_obs"].shape == (3, DEMO4_CRITIC_OBS_DIM)
    assert "yaw_rate" not in g1_29dof_demo4_rotate_observation.groups["table_obs"].terms


def test_demo4_table_actor_channels_use_actual_state_only() -> None:
    env = _make_env()
    env.simulator.agent_root_states[0, 0, 3:7] = _yaw_quaternion(math.pi / 2.0)
    env.simulator.all_root_states[0, :3] = torch.tensor([1.0, 0.0, 0.0])
    env.simulator.all_root_states[0, 3:7] = _yaw_quaternion(0.0)
    env.simulator.all_root_states[0, 7:10] = torch.tensor([0.0, 1.0, 0.0])
    before = _table_obs(env)

    env.simulator.all_root_states[0, 10:13] = torch.tensor([7.0, -8.0, 9.0])
    env.command_manager.command.object_pos_w[:] = 100.0
    env.command_manager.command.object_quat_w[:] = _yaw_quaternion(-1.7)
    after = _table_obs(env)

    torch.testing.assert_close(after, before, atol=0.0, rtol=0.0)
    torch.testing.assert_close(before[0, 0, :2], torch.tensor([0.0, -1.0]), atol=1e-6, rtol=0.0)
