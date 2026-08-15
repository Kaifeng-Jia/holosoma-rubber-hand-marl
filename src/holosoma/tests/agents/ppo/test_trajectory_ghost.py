"""Unit tests for the trajectory-consistent Stage 1B ghost geometry."""

from __future__ import annotations

import math

import pytest
import torch

from holosoma.envs.wbt.wbt_marl_compat_manager import (
    LeftSpacing070TrajectoryGhostTeammateWholeBodyTrackingManager,
    RightSpacing070TrajectoryGhostTeammateWholeBodyTrackingManager,
    _compute_stage1b_termination_diagnostics,
    _finite_difference_by_clip,
    _mirror_joint_tensor,
    _mirror_rotation_6d_xz,
    _mirror_teammate_observation,
    _mirror_wbt_actor_observation,
    _table_contact_moment_proxy,
    _table_local_x_in_world_xyzw,
    _yaw_from_xyzw,
)


def _test_joint_symmetry() -> tuple[torch.Tensor, torch.Tensor]:
    index_map = torch.arange(29)
    for first in range(0, 28, 2):
        index_map[first] = first + 1
        index_map[first + 1] = first
    sign_mask = torch.ones(29)
    sign_mask[0:28:2] = -1.0
    sign_mask[1:28:2] = -1.0
    return index_map, sign_mask


def test_table_local_x_xyzw_rotates_with_yaw() -> None:
    half_angle = math.pi / 4.0
    quaternion_xyzw = torch.tensor([[0.0, 0.0, math.sin(half_angle), math.cos(half_angle)]])
    axis = _table_local_x_in_world_xyzw(quaternion_xyzw)
    torch.testing.assert_close(axis, torch.tensor([[0.0, 1.0, 0.0]]), atol=1.0e-6, rtol=0.0)


def test_finite_difference_does_not_cross_motion_boundaries() -> None:
    values = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [100.0, 0.0],
            [102.0, 0.0],
        ]
    )
    derivative = _finite_difference_by_clip(
        values,
        torch.tensor([0, 3]),
        torch.tensor([3, 5]),
        dt=1.0,
    )
    torch.testing.assert_close(derivative[:, 0], torch.tensor([1.0, 1.0, 1.0, 2.0, 2.0]))


def test_finite_difference_uses_second_order_clip_endpoints() -> None:
    values = torch.tensor([[0.0], [1.0], [4.0], [9.0]])
    derivative = _finite_difference_by_clip(values, torch.tensor([0]), torch.tensor([4]), dt=1.0)
    torch.testing.assert_close(derivative[:, 0], torch.tensor([0.0, 2.0, 4.0, 6.0]))


def test_ghost_geometry_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match=r"\[N, 4\]"):
        _table_local_x_in_world_xyzw(torch.zeros(4))
    with pytest.raises(ValueError, match="positive"):
        _finite_difference_by_clip(torch.zeros((2, 2)), torch.tensor([0]), torch.tensor([2]), dt=0.0)


def test_spacing_070_candidate_classes_preserve_side_contract() -> None:
    assert LeftSpacing070TrajectoryGhostTeammateWholeBodyTrackingManager.observer_side == -1
    assert RightSpacing070TrajectoryGhostTeammateWholeBodyTrackingManager.observer_side == 1
    assert LeftSpacing070TrajectoryGhostTeammateWholeBodyTrackingManager.lateral_spacing_m == pytest.approx(0.7)
    assert RightSpacing070TrajectoryGhostTeammateWholeBodyTrackingManager.lateral_spacing_m == pytest.approx(0.7)


def test_yaw_and_table_contact_moment_proxy_sign() -> None:
    half_angle = math.pi / 4.0
    yaw = _yaw_from_xyzw(
        torch.tensor([[0.0, 0.0, math.sin(half_angle), math.cos(half_angle)]])
    )
    torch.testing.assert_close(yaw, torch.tensor([math.pi / 2.0]), atol=1.0e-6, rtol=0.0)

    hand_pos = torch.tensor([[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]])
    force_on_hand = torch.tensor([[[0.0, -2.0, 0.0], [0.0, -3.0, 0.0]]])
    table_force, yaw_moment = _table_contact_moment_proxy(hand_pos, force_on_hand, torch.zeros((1, 3)))
    torch.testing.assert_close(table_force, -force_on_hand)
    torch.testing.assert_close(yaw_moment, torch.tensor([[2.0, -3.0]]))


def test_stage1b_diagnostics_separate_exact_bad_tracking_causes() -> None:
    class DummyMotion:
        has_object = True

    class DummyCommand:
        motion = DummyMotion()
        ref_pos_w = torch.tensor([[0.0, 0.0, 0.8], [0.0, 0.0, 0.8]])
        robot_ref_pos_w = torch.tensor([[0.0, 0.0, 0.8], [0.0, 0.0, 0.2]])
        ref_quat_w = torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]])
        robot_ref_quat_w = ref_quat_w.clone()
        body_pos_relative_w = torch.zeros((2, 2, 3))
        robot_body_pos_w = torch.zeros((2, 2, 3))
        robot_body_quat_w = ref_quat_w.unsqueeze(1).expand(-1, 2, -1).clone()
        object_pos_w = torch.zeros((2, 3))
        simulator_object_pos_w = torch.tensor([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]])
        object_quat_w = ref_quat_w.clone()
        simulator_object_quat_w = ref_quat_w.clone()

    class DummyCommandManager:
        @staticmethod
        def get_state(name: str):
            assert name == "motion_command"
            return DummyCommand()

    class DummyBadTracking:
        bad_ref_pos_threshold = 0.5
        bad_ref_ori_threshold = 0.8
        bad_motion_body_pos_threshold = 0.25
        bad_object_pos_threshold = 0.25
        bad_object_ori_threshold = 0.8
        bad_motion_body_pos_body_indexes = torch.tensor([0, 1])
        bad_motion_body_pos_body_names = ["left_wrist", "right_wrist"]

    class DummyTerminationManager:
        _term_instances = {"bad_tracking": DummyBadTracking()}
        terminated = torch.tensor([False, True])

    class DummyContactData:
        pos_w = torch.zeros((2, 2, 3))
        net_forces_w = torch.zeros((2, 2, 3))

    class DummyContactSensor:
        data = DummyContactData()

    class DummySimulator:
        contact_sensor = DummyContactSensor()

    class DummyEnv:
        num_envs = 2
        device = "cpu"
        command_manager = DummyCommandManager()
        termination_manager = DummyTerminationManager()
        simulator = DummySimulator()
        stage1b_rubber_hand_sensor_indexes = torch.tensor([0, 1])
        stage1b_wrist_tracked_indexes = torch.tensor([0, 1])
        stage1b_hand_origin_offset_b = torch.tensor([[0.0415, 0.003, 0.0], [0.0415, -0.003, 0.0]])
        time_out_buf = torch.tensor([False, False])
        reset_buf = torch.tensor([False, True])

    diagnostics = _compute_stage1b_termination_diagnostics(DummyEnv())

    torch.testing.assert_close(diagnostics["stage1b_ref_pos_z_error_m"], torch.tensor([0.0, 0.6]))
    torch.testing.assert_close(diagnostics["stage1b_object_pos_error_m"], torch.tensor([0.0, 0.3]))
    assert diagnostics["stage1b_bad_ref_pos"].tolist() == [False, True]
    assert diagnostics["stage1b_bad_object_pos"].tolist() == [False, True]
    assert diagnostics["stage1b_bad_tracking"].tolist() == [False, True]
    assert diagnostics["stage1b_reset"].tolist() == [False, True]


def test_joint_and_teammate_mirrors_are_involutions() -> None:
    index_map, sign_mask = _test_joint_symmetry()
    joints = torch.arange(58, dtype=torch.float).reshape(2, 29)
    mirrored_joints = _mirror_joint_tensor(joints, index_map, sign_mask)
    torch.testing.assert_close(_mirror_joint_tensor(mirrored_joints, index_map, sign_mask), joints)

    teammate = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    mirrored_teammate = _mirror_teammate_observation(teammate)
    torch.testing.assert_close(mirrored_teammate, torch.tensor([[1.0, -2.0, 3.0, -4.0]]))
    torch.testing.assert_close(_mirror_teammate_observation(mirrored_teammate), teammate)


def test_rotation_6d_mirror_is_an_involution() -> None:
    angle = 0.37
    cosine = math.cos(angle)
    sine = math.sin(angle)
    rotation = torch.tensor(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotation_6d = rotation[..., :2].reshape(1, 6)
    mirrored = _mirror_rotation_6d_xz(rotation_6d)
    expected_rotation = torch.tensor(
        [
            [cosine, sine, 0.0],
            [-sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    expected = expected_rotation[..., :2].reshape(1, 6)
    torch.testing.assert_close(mirrored, expected)
    torch.testing.assert_close(_mirror_rotation_6d_xz(mirrored), rotation_6d)


def test_complete_wbt_actor_observation_mirror_preserves_layout_and_is_involutive() -> None:
    index_map, sign_mask = _test_joint_symmetry()
    observation = torch.arange(2 * 154, dtype=torch.float).reshape(2, 154) / 100.0
    # Use valid two-row rotation encodings for the final six values.
    observation[:, 148:154] = torch.tensor([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
    mirrored = _mirror_wbt_actor_observation(observation, index_map, sign_mask)
    recovered = _mirror_wbt_actor_observation(mirrored, index_map, sign_mask)
    assert mirrored.shape == observation.shape
    torch.testing.assert_close(
        mirrored[:, 29:32],
        observation[:, 29:32] * torch.tensor([-1.0, 1.0, -1.0]),
    )
    torch.testing.assert_close(recovered, observation)
