"""Object-state observation contracts for the competitive square-table Demo 3."""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch

from holosoma.config_values.marl.g1.demo3_observation import (
    DEMO3_ACTOR_OBS_DIM,
    DEMO3_TABLE_OBS_DIM,
    g1_29dof_demo3_observation,
)
from holosoma.config_values.marl.g1.observation import g1_29dof_plan5_observation
from holosoma.managers.observation.manager import ObservationManager
from holosoma.managers.observation.terms.demo3_tug import (
    table_linear_velocity_b,
    table_relative_position_b,
    table_relative_yaw_sin_cos,
)
from holosoma.managers.observation.terms.demo3_tug import (
    _per_agent_state,
    ego_ordered_critic_obs,
)


class FakeCommandManager:
    def __init__(self, command):
        self.command = command

    def get_state(self, name):
        return self.command if name == "paired_motion_command" else None


def _yaw_quaternion(angle: float) -> torch.Tensor:
    return torch.tensor([0.0, 0.0, math.sin(angle / 2.0), math.cos(angle / 2.0)])


def _make_env(num_envs: int = 1):
    num_agents = 2
    num_dof = 29
    num_bodies = 14
    root_states = torch.zeros(num_envs, num_agents, 13)
    root_states[..., 6] = 1.0
    root_states[:, 1, 0] = 0.8
    body_pos = torch.zeros(num_envs, num_agents, num_bodies, 3)
    body_quat = torch.zeros(num_envs, num_agents, num_bodies, 4)
    body_quat[..., 3] = 1.0
    all_root_states = torch.zeros(num_envs, 13)
    all_root_states[..., 6] = 1.0
    simulator = SimpleNamespace(
        agent_root_states=root_states,
        agent_dof_pos=torch.zeros(num_envs, num_agents, num_dof),
        agent_dof_vel=torch.zeros(num_envs, num_agents, num_dof),
        agent_rigid_body_pos=body_pos,
        agent_rigid_body_rot=body_quat,
        all_root_states=all_root_states,
    )
    command = SimpleNamespace(
        command=torch.zeros(num_envs, num_agents, num_dof * 2),
        ref_body_index=0,
        agent_ref_pos_w=body_pos[:, :, 0].clone(),
        agent_ref_quat_w=body_quat[:, :, 0].clone(),
        simulator_agent_body_pos_w=body_pos,
        simulator_agent_body_quat_w=body_quat,
        simulator_object_pos_w=all_root_states[:, :3],
        simulator_object_quat_w=all_root_states[:, 3:7],
        object_pos_w=torch.zeros(num_envs, 3),
        object_quat_w=torch.tensor([0.0, 0.0, 0.0, 1.0]).repeat(num_envs, 1),
        object_lin_vel_w=torch.zeros(num_envs, 3),
        object_indices_in_simulator=torch.arange(num_envs),
        time_steps=torch.zeros(num_envs, dtype=torch.long),
        reference=SimpleNamespace(num_frames=317),
    )
    return SimpleNamespace(
        num_envs=num_envs,
        num_agents=num_agents,
        num_dof=num_dof,
        device="cpu",
        simulator=simulator,
        default_dof_pos=torch.zeros(num_envs, num_dof),
        action_manager=SimpleNamespace(action=torch.zeros(num_envs, num_agents * num_dof)),
        command_manager=FakeCommandManager(command),
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


def test_demo3_adds_only_six_table_channels_to_the_actor_contract() -> None:
    env = _make_env(num_envs=3)
    observations = ObservationManager(g1_29dof_demo3_observation, env, "cpu").compute()
    combined = torch.cat(
        (observations["actor_obs"], observations["teammate_obs"], observations["table_obs"]),
        dim=-1,
    )

    assert list(g1_29dof_demo3_observation.groups) == [
        "actor_obs",
        "teammate_obs",
        "table_obs",
        "critic_obs",
    ]
    assert list(g1_29dof_plan5_observation.groups) == [
        "actor_obs",
        "teammate_obs",
        "critic_obs",
    ]
    assert observations["actor_obs"].shape == (3, 2, 154)
    assert observations["teammate_obs"].shape == (3, 2, 4)
    assert observations["table_obs"].shape == (3, 2, DEMO3_TABLE_OBS_DIM)
    assert observations["critic_obs"].shape == (3, 2, 527)
    assert combined.shape == (3, 2, DEMO3_ACTOR_OBS_DIM)
    assert "yaw_rate" not in g1_29dof_demo3_observation.groups["table_obs"].terms


def test_demo3_critic_is_ego_ordered_without_changing_total_dimension() -> None:
    env = _make_env(num_envs=2)
    env.simulator.agent_dof_pos[:, 0] = 1.0
    env.simulator.agent_dof_pos[:, 1] = 2.0
    per_agent = _per_agent_state(env)
    critic = ego_ordered_critic_obs(env)

    assert per_agent.shape == (2, 2, 228)
    assert critic.shape == (2, 2, 527)
    torch.testing.assert_close(critic[:, 0, :228], per_agent[:, 0])
    torch.testing.assert_close(critic[:, 0, 228:456], per_agent[:, 1])
    torch.testing.assert_close(critic[:, 1, :228], per_agent[:, 1])
    torch.testing.assert_close(critic[:, 1, 228:456], per_agent[:, 0])


def test_table_state_uses_each_observers_heading_frame() -> None:
    env = _make_env()
    env.simulator.agent_root_states[0, 0, 3:7] = _yaw_quaternion(math.pi / 2.0)
    env.simulator.all_root_states[0, :3] = torch.tensor([1.0, 0.0, 0.0])
    env.simulator.all_root_states[0, 3:7] = _yaw_quaternion(0.0)
    env.simulator.all_root_states[0, 7:10] = torch.tensor([0.0, 1.0, 0.0])

    position = table_relative_position_b(env)
    velocity = table_linear_velocity_b(env)
    yaw = table_relative_yaw_sin_cos(env)

    torch.testing.assert_close(position[0, 0], torch.tensor([0.0, -1.0]), atol=1e-6, rtol=0.0)
    torch.testing.assert_close(position[0, 1], torch.tensor([0.2, 0.0]), atol=1e-6, rtol=0.0)
    torch.testing.assert_close(velocity[0, 0], torch.tensor([1.0, 0.0]), atol=1e-6, rtol=0.0)
    torch.testing.assert_close(velocity[0, 1], torch.tensor([0.0, 1.0]), atol=1e-6, rtol=0.0)
    torch.testing.assert_close(yaw[0, 0], torch.tensor([-1.0, 0.0]), atol=1e-6, rtol=0.0)
    torch.testing.assert_close(yaw[0, 1], torch.tensor([0.0, 1.0]), atol=1e-6, rtol=0.0)


def test_table_observation_is_global_se2_invariant() -> None:
    env = _make_env()
    roots = env.simulator.agent_root_states[0]
    roots[0, :3] = torch.tensor([-0.3, 0.2, 0.0])
    roots[1, :3] = torch.tensor([0.7, -0.1, 0.0])
    roots[0, 3:7] = _yaw_quaternion(0.3)
    roots[1, 3:7] = _yaw_quaternion(-0.5)
    env.simulator.all_root_states[0, :3] = torch.tensor([0.2, 0.7, 0.0])
    env.simulator.all_root_states[0, 3:7] = _yaw_quaternion(0.9)
    env.simulator.all_root_states[0, 7:10] = torch.tensor([0.4, -0.2, 0.0])
    expected = _table_obs(env)

    angle = 1.2
    rotation = torch.tensor(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    translation = torch.tensor([1.1, -0.4])
    roots[:, :2] = roots[:, :2] @ rotation.T + translation
    roots[0, 3:7] = _yaw_quaternion(0.3 + angle)
    roots[1, 3:7] = _yaw_quaternion(-0.5 + angle)
    table_position = env.simulator.all_root_states[0, :2].clone()
    env.simulator.all_root_states[0, :2] = table_position @ rotation.T + translation
    env.simulator.all_root_states[0, 3:7] = _yaw_quaternion(0.9 + angle)
    table_velocity = env.simulator.all_root_states[0, 7:9].clone()
    env.simulator.all_root_states[0, 7:9] = table_velocity @ rotation.T

    torch.testing.assert_close(_table_obs(env), expected, atol=1e-6, rtol=0.0)


def test_yaw_rate_and_reference_object_pose_do_not_enter_table_observation() -> None:
    env = _make_env()
    env.simulator.all_root_states[0, :3] = torch.tensor([0.4, -0.2, 0.0])
    env.simulator.all_root_states[0, 3:7] = _yaw_quaternion(math.pi - 1.0e-5)
    before = _table_obs(env)

    env.simulator.all_root_states[0, 10:13] = torch.tensor([7.0, -8.0, 9.0])
    env.command_manager.command.object_pos_w[:] = 100.0
    env.command_manager.command.object_quat_w[:] = _yaw_quaternion(-1.7)
    after = _table_obs(env)

    torch.testing.assert_close(after, before, atol=0.0, rtol=0.0)
