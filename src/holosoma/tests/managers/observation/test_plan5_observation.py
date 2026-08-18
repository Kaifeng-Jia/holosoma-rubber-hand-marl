"""Shape and coordinate contracts for Plan 5 per-agent actor observations."""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch

from holosoma.config_values.marl.g1.observation import g1_29dof_plan5_actor_observation
from holosoma.config_values.wbt.g1.observation import (
    actor_obs_shared,
    teammate_obs_marl_compat,
)
from holosoma.managers.observation.manager import ObservationManager
from holosoma.managers.observation.terms.marl import (
    real_teammate_relative_position_b,
    real_teammate_relative_velocity_b,
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
    root_states = torch.zeros(num_envs, num_agents, 13)
    root_states[..., 6] = 1.0
    root_states[:, 1, 0] = 0.8

    body_pos = torch.zeros(num_envs, num_agents, 1, 3)
    body_quat = torch.zeros(num_envs, num_agents, 1, 4)
    body_quat[..., 3] = 1.0
    simulator = SimpleNamespace(
        agent_root_states=root_states,
        agent_dof_pos=torch.zeros(num_envs, num_agents, num_dof),
        agent_dof_vel=torch.zeros(num_envs, num_agents, num_dof),
        agent_rigid_body_pos=body_pos,
        agent_rigid_body_rot=body_quat,
    )
    command = SimpleNamespace(
        command=torch.zeros(num_envs, num_agents, num_dof * 2),
        ref_body_index=0,
        agent_ref_pos_w=body_pos[:, :, 0].clone(),
        agent_ref_quat_w=body_quat[:, :, 0].clone(),
    )
    env = SimpleNamespace(
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
    return env


def test_real_teammate_state_uses_each_observers_heading_frame() -> None:
    env = _make_env()
    env.simulator.agent_root_states[0, 0, 3:7] = _yaw_quaternion(math.pi / 2.0)
    env.simulator.agent_root_states[0, 1, 7:10] = torch.tensor([0.0, 1.0, 0.0])

    position = real_teammate_relative_position_b(env)
    velocity = real_teammate_relative_velocity_b(env)

    torch.testing.assert_close(position[0, 0], torch.tensor([0.0, -0.8]), atol=1e-6, rtol=0.0)
    torch.testing.assert_close(position[0, 1], torch.tensor([-0.8, 0.0]), atol=1e-6, rtol=0.0)
    torch.testing.assert_close(velocity[0, 0], torch.tensor([1.0, 0.0]), atol=1e-6, rtol=0.0)
    torch.testing.assert_close(velocity[0, 1], torch.tensor([0.0, -1.0]), atol=1e-6, rtol=0.0)


def test_plan5_observation_preserves_154_plus_4_actor_contract() -> None:
    env = _make_env(num_envs=3)
    manager = ObservationManager(g1_29dof_plan5_actor_observation, env, "cpu")

    observations = manager.compute()
    combined = torch.cat((observations["actor_obs"], observations["teammate_obs"]), dim=-1)

    assert observations["actor_obs"].shape == (3, 2, 154)
    assert observations["teammate_obs"].shape == (3, 2, 4)
    assert combined.shape == (3, 2, 158)
    assert torch.isfinite(combined).all()
    torch.testing.assert_close(
        observations["teammate_obs"][:, 0, :2],
        -observations["teammate_obs"][:, 1, :2],
    )


def test_plan5_terms_only_replace_single_agent_data_providers() -> None:
    plan5_actor = g1_29dof_plan5_actor_observation.groups["actor_obs"]
    plan5_teammate = g1_29dof_plan5_actor_observation.groups["teammate_obs"]

    assert list(plan5_actor.terms) == list(actor_obs_shared.terms)
    assert list(plan5_teammate.terms) == list(teammate_obs_marl_compat.terms)
    for plan5_group, source_group in (
        (plan5_actor, actor_obs_shared),
        (plan5_teammate, teammate_obs_marl_compat),
    ):
        assert plan5_group.concatenate == source_group.concatenate
        assert plan5_group.enable_noise == source_group.enable_noise
        assert plan5_group.history_length == source_group.history_length
        for name, source_term in source_group.terms.items():
            plan5_term = plan5_group.terms[name]
            assert plan5_term.func != source_term.func
            assert plan5_term.params == source_term.params
            assert plan5_term.scale == source_term.scale
            assert plan5_term.noise == source_term.noise
            assert plan5_term.clip == source_term.clip
