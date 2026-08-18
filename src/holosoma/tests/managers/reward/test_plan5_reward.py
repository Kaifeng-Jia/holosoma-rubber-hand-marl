from __future__ import annotations

from types import SimpleNamespace

import torch

from holosoma.config_values.marl.g1.reward import g1_29dof_plan5_push_reward
from holosoma.config_values.wbt.g1.reward import g1_29dof_wbt_reward_w_object
from holosoma.managers.reward.terms import marl
from tests.managers.command.test_paired_a1_command import make_command


def make_reward_env(tmp_path):
    command, env = make_command(tmp_path)
    command.reset(None)
    env.num_agents = 2
    env.num_dof = 2
    env.command_manager = SimpleNamespace(
        get_state=lambda name: command if name == "paired_motion_command" else None
    )
    env.action_manager = SimpleNamespace(
        action=torch.zeros(env.num_envs, env.num_agents * env.num_dof),
        prev_action=torch.zeros(env.num_envs, env.num_agents * env.num_dof),
    )
    env.simulator.body_names = ["pelvis", "torso_link"]
    env.simulator.hard_dof_pos_limits = torch.tensor([[-1.0, 1.0], [-2.0, 2.0]])
    env.simulator.agent_rigid_body_pos.copy_(command.agent_body_pos_w)
    env.simulator.agent_rigid_body_rot.copy_(command.agent_body_quat_w)
    env.simulator.agent_rigid_body_vel = command.agent_body_lin_vel_w.clone()
    env.simulator.agent_rigid_body_ang_vel = command.agent_body_ang_vel_w.clone()
    env.simulator.agent_contact_forces_history = torch.zeros(2, 2, 3, 2, 3)
    return command, env


def test_plan5_reward_preserves_original_a1_weights_and_parameters():
    original = g1_29dof_wbt_reward_w_object.terms
    paired = g1_29dof_plan5_push_reward.terms

    assert list(paired) == list(original)
    for name in original:
        assert paired[name].weight == original[name].weight
        assert paired[name].params == original[name].params


def test_per_agent_tracking_rewards_are_averaged(tmp_path):
    command, env = make_reward_env(tmp_path)
    perfect = marl.motion_global_ref_position_error_exp(env, sigma=0.3)
    torch.testing.assert_close(perfect, torch.ones(2))

    env.simulator.agent_rigid_body_pos[0, 1, command.ref_body_index, 2] += 0.3
    reward = marl.motion_global_ref_position_error_exp(env, sigma=0.3)

    expected_agent_1 = torch.exp(torch.tensor(-1.0))
    torch.testing.assert_close(reward[0], (1.0 + expected_agent_1) / 2.0)
    torch.testing.assert_close(reward[1], torch.tensor(1.0))


def test_all_body_tracking_terms_equal_one_for_matching_state(tmp_path):
    _, env = make_reward_env(tmp_path)

    terms = [
        marl.motion_global_ref_orientation_error_exp(env, sigma=0.4),
        marl.motion_relative_body_position_error_exp(env, sigma=0.3),
        marl.motion_relative_body_orientation_error_exp(env, sigma=0.4),
        marl.motion_global_body_lin_vel(env, sigma=1.0),
        marl.motion_global_body_ang_vel(env, sigma=3.14),
    ]

    for reward in terms:
        torch.testing.assert_close(reward, torch.ones(env.num_envs))


def test_object_reward_is_computed_once_per_shared_environment(tmp_path):
    command, env = make_reward_env(tmp_path)

    reward = marl.object_global_ref_position_error_exp(env, sigma=0.3)

    assert reward.shape == (env.num_envs,)
    torch.testing.assert_close(reward, torch.ones(env.num_envs))


def test_incidental_contact_penalty_is_meaned_not_used_as_a_gate(tmp_path):
    _, env = make_reward_env(tmp_path)
    cfg = g1_29dof_plan5_push_reward.terms["undesired_contacts"]
    term = marl.UndesiredContacts(cfg, env)
    env.simulator.agent_contact_forces_history[0, 1, 0, 0, 2] = 2.0

    penalty = term(env)

    torch.testing.assert_close(penalty, torch.tensor([0.5, 0.0]))


def test_regularizers_average_the_two_agent_costs(tmp_path):
    _, env = make_reward_env(tmp_path)
    env.action_manager.action[0, :2] = 1.0

    action_rate = marl.penalty_action_rate(env)

    torch.testing.assert_close(action_rate, torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(marl.limits_dof_pos(env, soft_dof_pos_limit=0.9), torch.zeros(2))
