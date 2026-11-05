"""Observation contracts for the CORE4D paired small-table demo."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from holosoma.agents.mappo.core4d_smalltable_initialization import (
    CORE4D_SMALLTABLE_ACTOR_OBS_DIM,
)
from holosoma.config_values.marl.g1.core4d_smalltable_observation import (
    CORE4D_SMALLTABLE_CRITIC_OBS_DIM,
    g1_29dof_core4d_smalltable_evaluation_observation,
    g1_29dof_core4d_smalltable_observation,
)
from holosoma.config_values.marl.g1.observation import g1_29dof_plan5_observation
from holosoma.managers.observation.manager import ObservationManager
from holosoma.managers.observation.terms.marl import (
    centralized_shared_object_tracking,
)


class _CommandManager:
    def __init__(self, command):
        self.command = command

    def get_state(self, name):
        return self.command if name == "paired_motion_command" else None


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
    object_states = torch.zeros(num_envs, 13)
    object_states[..., 6] = 1.0
    simulator = SimpleNamespace(
        agent_root_states=root_states,
        agent_dof_pos=torch.zeros(num_envs, num_agents, num_dof),
        agent_dof_vel=torch.zeros(num_envs, num_agents, num_dof),
        agent_rigid_body_pos=body_pos,
        agent_rigid_body_rot=body_quat,
        all_root_states=object_states,
    )
    command = SimpleNamespace(
        command=torch.zeros(num_envs, num_agents, num_dof * 2),
        ref_body_index=0,
        agent_ref_pos_w=body_pos[:, :, 0].clone(),
        agent_ref_quat_w=body_quat[:, :, 0].clone(),
        simulator_agent_body_pos_w=body_pos,
        simulator_agent_body_quat_w=body_quat,
        simulator_object_pos_w=object_states[:, :3],
        simulator_object_quat_w=object_states[:, 3:7],
        simulator_object_lin_vel_w=object_states[:, 7:10],
        object_pos_w=torch.zeros(num_envs, 3),
        object_quat_w=torch.tensor([0.0, 0.0, 0.0, 1.0]).repeat(num_envs, 1),
        object_lin_vel_w=torch.zeros(num_envs, 3),
        object_indices_in_simulator=torch.arange(num_envs),
        time_steps=torch.zeros(num_envs, dtype=torch.long),
        reference=SimpleNamespace(num_frames=687),
    )
    return SimpleNamespace(
        num_envs=num_envs,
        num_agents=num_agents,
        num_dof=num_dof,
        device="cpu",
        simulator=simulator,
        default_dof_pos=torch.zeros(num_envs, num_dof),
        action_manager=SimpleNamespace(action=torch.zeros(num_envs, num_agents * num_dof)),
        command_manager=_CommandManager(command),
        logger=None,
    )


def test_smalltable_actor_contract_matches_plan5_154_plus_teammate_4() -> None:
    env = _make_env(num_envs=3)
    observations = ObservationManager(
        g1_29dof_core4d_smalltable_observation, env, "cpu"
    ).compute()
    actor_input = torch.cat(
        (
            observations["actor_obs"],
            observations["teammate_obs"],
        ),
        dim=-1,
    )

    assert list(g1_29dof_core4d_smalltable_observation.groups) == [
        "actor_obs",
        "teammate_obs",
        "critic_obs",
    ]
    assert list(g1_29dof_plan5_observation.groups) == [
        "actor_obs",
        "teammate_obs",
        "critic_obs",
    ]
    assert observations["actor_obs"].shape == (3, 2, 154)
    assert observations["teammate_obs"].shape == (3, 2, 4)
    assert observations["critic_obs"].shape == (3, CORE4D_SMALLTABLE_CRITIC_OBS_DIM)
    assert actor_input.shape == (3, 2, CORE4D_SMALLTABLE_ACTOR_OBS_DIM)
    assert torch.isfinite(actor_input).all()


def test_evaluation_disables_only_actor_training_noise() -> None:
    training = g1_29dof_core4d_smalltable_observation.groups
    evaluation = g1_29dof_core4d_smalltable_evaluation_observation.groups

    assert training["actor_obs"].enable_noise is True
    assert evaluation["actor_obs"].enable_noise is False
    assert evaluation["actor_obs"].terms == training["actor_obs"].terms
    assert evaluation["teammate_obs"] == training["teammate_obs"]
    assert evaluation["critic_obs"] == training["critic_obs"]


def test_critic_uses_reference_compatible_origin_velocity() -> None:
    env = _make_env()
    command = env.command_manager.command
    env.simulator.all_root_states[0, 7:10] = torch.tensor([9.0, 8.0, 7.0])
    command.simulator_object_lin_vel_w = torch.tensor([[0.4, -0.2, 0.1]])
    command.object_lin_vel_w[:] = torch.tensor([0.1, -0.1, 0.0])

    tracking = centralized_shared_object_tracking(env)
    torch.testing.assert_close(
        tracking[0, -3:],
        torch.tensor([0.3, -0.1, 0.1]),
        rtol=0.0,
        atol=1.0e-7,
    )
