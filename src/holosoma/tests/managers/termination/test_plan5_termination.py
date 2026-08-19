from __future__ import annotations

from types import SimpleNamespace

import torch

from holosoma.config_types.termination import TerminationTermCfg
from holosoma.managers.termination.terms.marl import JointBadTrackingZOnly
from tests.managers.command.test_paired_a1_command import make_command


def make_termination(tmp_path):
    command, env = make_command(tmp_path)
    command.reset(None)
    env.command_manager = SimpleNamespace(
        get_state=lambda name: command if name == "paired_motion_command" else None
    )
    env.simulator.agent_rigid_body_pos.copy_(command.agent_body_pos_w)
    env.simulator.agent_rigid_body_rot.copy_(command.agent_body_quat_w)
    cfg = TerminationTermCfg(
        func="unused",
        params={
            "bad_ref_pos_threshold": 0.5,
            "bad_ref_ori_threshold": 0.8,
            "bad_motion_body_pos_threshold": 0.25,
            "minimum_ref_body_height": 0.4,
            "body_names_to_track": ["pelvis", "torso_link"],
            "bad_motion_body_pos_body_names": ["pelvis", "torso_link"],
            "bad_object_pos_threshold": 0.25,
            "bad_object_ori_threshold": 0.8,
        },
    )
    return command, env, JointBadTrackingZOnly(cfg, env)


def test_matching_joint_state_does_not_terminate(tmp_path):
    _, env, term = make_termination(tmp_path)

    torch.testing.assert_close(term(env), torch.tensor([False, False]))


def test_reference_height_deviation_is_diagnostic_only(tmp_path):
    command, env, term = make_termination(tmp_path)
    env.simulator.agent_rigid_body_pos[0, 1, command.ref_body_index, 2] += 0.6

    torch.testing.assert_close(term(env), torch.tensor([False, False]))
    torch.testing.assert_close(
        term.last_diagnostics["bad_robot_ref_height_by_agent"],
        torch.tensor([[False, True], [False, False]]),
    )
    torch.testing.assert_close(term.last_diagnostics["bad_robot"], torch.tensor([False, False]))
    torch.testing.assert_close(
        term.last_diagnostics["robot_ref_height_error_m_by_agent"],
        torch.tensor([[0.0, 0.6], [0.0, 0.0]]),
    )
    torch.testing.assert_close(
        term.last_diagnostics["robot_ref_height_reference_m_by_agent"],
        command.agent_ref_pos_w[..., 2],
    )
    torch.testing.assert_close(
        term.last_diagnostics["robot_ref_height_actual_m_by_agent"],
        env.simulator.agent_rigid_body_pos[:, :, command.ref_body_index, 2],
    )
    torch.testing.assert_close(
        term.last_diagnostics["robot_body_height_error_m_by_agent"],
        torch.tensor(
            [
                [[0.0, 0.0], [0.0, 0.6]],
                [[0.0, 0.0], [0.0, 0.0]],
            ]
        ),
    )
    torch.testing.assert_close(
        term.last_diagnostics["robot_max_body_height_error_m_by_agent"],
        torch.tensor([[0.0, 0.6], [0.0, 0.0]]),
    )
    torch.testing.assert_close(
        term.last_diagnostics["bad_object_position"],
        torch.tensor([False, False]),
    )


def test_low_reference_body_height_of_either_robot_resets_its_environment(tmp_path):
    command, env, term = make_termination(tmp_path)
    env.simulator.agent_rigid_body_pos[0, 1, command.ref_body_index, 2] = 0.39

    torch.testing.assert_close(term(env), torch.tensor([True, False]))
    torch.testing.assert_close(
        term.last_diagnostics["bad_robot_low_height_by_agent"],
        torch.tensor([[False, True], [False, False]]),
    )
    torch.testing.assert_close(term.last_diagnostics["bad_robot"], torch.tensor([True, False]))


def test_shared_object_error_jointly_resets_the_environment(tmp_path):
    command, env, term = make_termination(tmp_path)
    env.simulator.all_root_states[command.object_indices_in_simulator[1], 0] += 0.3

    torch.testing.assert_close(term(env), torch.tensor([False, True]))
    torch.testing.assert_close(
        term.last_diagnostics["bad_object_position"],
        torch.tensor([False, True]),
    )
    torch.testing.assert_close(
        term.last_diagnostics["reference_frame"],
        command.time_steps,
    )
    torch.testing.assert_close(
        term.last_diagnostics["actual_object_position"],
        command.simulator_object_pos_w,
    )
    torch.testing.assert_close(
        term.last_diagnostics["object_position_error_m"],
        torch.tensor([0.0, 0.3]),
    )
    torch.testing.assert_close(
        term.last_diagnostics["object_orientation_error_rad"],
        torch.zeros(2),
    )


def test_termination_has_no_contact_condition(tmp_path):
    _, env, term = make_termination(tmp_path)
    env.simulator.agent_contact_forces_history = torch.full((2, 2, 3, 2, 3), 100.0)

    torch.testing.assert_close(term(env), torch.tensor([False, False]))
