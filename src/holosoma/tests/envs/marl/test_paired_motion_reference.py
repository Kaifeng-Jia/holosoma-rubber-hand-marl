from __future__ import annotations

import numpy as np
import pytest
import torch

from holosoma.envs.marl.paired_motion_reference import PairedMotionReference


def write_paired_reference(path, *, include_object_velocity: bool = True) -> None:
    frames = 4
    joint_names = np.array(["joint_b", "joint_a"])
    body_names = np.array(["torso", "pelvis"])

    joint_pos = np.zeros((frames, 2, 2), dtype=np.float32)
    joint_pos[:, 0] = np.array([20.0, 10.0])
    joint_pos[:, 1] = np.array([40.0, 30.0])
    joint_vel = joint_pos + 0.5

    body_pos = np.zeros((frames, 2, 2, 3), dtype=np.float32)
    body_pos[:, 0, 0, 0] = 2.0  # agent 0 torso in stored order
    body_pos[:, 0, 1, 0] = -0.6  # agent 0 pelvis
    body_pos[:, 1, 0, 0] = 3.0  # agent 1 torso
    body_pos[:, 1, 1, 0] = 0.7  # agent 1 pelvis
    body_lin_vel = body_pos + 1.0
    body_ang_vel = body_pos + 2.0

    body_quat = np.zeros((frames, 2, 2, 4), dtype=np.float32)
    body_quat[..., 0] = 1.0
    body_quat[:, 0, 1] = np.array([0.5, 0.5, -0.5, -0.5], dtype=np.float32)

    object_pos = np.zeros((frames, 3), dtype=np.float32)
    object_pos[:, 1] = np.arange(frames, dtype=np.float32)
    object_quat = np.zeros((frames, 4), dtype=np.float32)
    object_quat[:] = np.array([0.5, -0.5, 0.5, -0.5], dtype=np.float32)

    fields = {
        "fps": np.asarray(50),
        "joint_names": joint_names,
        "body_names": body_names,
        "agent_joint_pos": joint_pos,
        "agent_joint_vel": joint_vel,
        "agent_body_pos_w": body_pos,
        "agent_body_quat_w": body_quat,
        "agent_body_lin_vel_w": body_lin_vel,
        "agent_body_ang_vel_w": body_ang_vel,
        "object_pos_w": object_pos,
        "object_quat_w": object_quat,
        "provenance": np.asarray("test"),
    }
    if include_object_velocity:
        fields["object_lin_vel_w"] = np.ones((frames, 3), dtype=np.float32)
    np.savez(path, **fields)


def test_load_reorders_simulator_names_and_converts_wxyz(tmp_path):
    reference_path = tmp_path / "explicit_pull_pair.npz"
    write_paired_reference(reference_path)

    reference = PairedMotionReference(
        str(reference_path),
        robot_body_names=["pelvis", "torso"],
        robot_joint_names=["joint_a", "joint_b"],
    )

    assert reference.fps == 50
    assert reference.time_step_total == 4
    assert reference.num_frames == 4
    assert reference.has_object
    assert reference.motion_files == [str(reference_path)]
    assert reference.motion_names == ["explicit_pull_pair"]
    torch.testing.assert_close(
        reference.agent_joint_pos[0],
        torch.tensor([[10.0, 20.0], [30.0, 40.0]]),
    )
    torch.testing.assert_close(
        reference.agent_body_pos_w[0, :, 0, 0], torch.tensor([-0.6, 0.7])
    )
    torch.testing.assert_close(
        reference.agent_body_quat_w[0, 0, 0],
        torch.tensor([0.5, -0.5, -0.5, 0.5]),
    )
    torch.testing.assert_close(
        reference.object_quat_w[0], torch.tensor([-0.5, 0.5, -0.5, 0.5])
    )

    sample = reference.sample(torch.tensor([1, 3]))
    assert set(sample) == set(PairedMotionReference._SAMPLE_KEYS)
    assert sample["agent_body_pos_w"].shape == (2, 2, 2, 3)
    torch.testing.assert_close(sample["object_pos_w"][:, 1], torch.tensor([1.0, 3.0]))


def test_load_fails_closed_when_runtime_channel_is_missing(tmp_path):
    reference_path = tmp_path / "incomplete_pair.npz"
    write_paired_reference(reference_path, include_object_velocity=False)

    with pytest.raises(ValueError, match="object_lin_vel_w"):
        PairedMotionReference(
            str(reference_path),
            robot_body_names=["pelvis", "torso"],
            robot_joint_names=["joint_a", "joint_b"],
        )


def test_load_fails_closed_when_simulator_name_is_missing(tmp_path):
    reference_path = tmp_path / "wrong_names.npz"
    write_paired_reference(reference_path)

    with pytest.raises(ValueError, match="missing simulator body names"):
        PairedMotionReference(
            str(reference_path),
            robot_body_names=["pelvis", "rubber_hand_link"],
            robot_joint_names=["joint_a", "joint_b"],
        )


def test_load_allows_aliases_to_repeat_one_stored_body(tmp_path):
    reference_path = tmp_path / "repeated_body_alias.npz"
    write_paired_reference(reference_path)

    reference = PairedMotionReference(
        str(reference_path),
        robot_body_names=["pelvis", "torso", "torso"],
        robot_joint_names=["joint_a", "joint_b"],
    )

    assert reference.agent_body_pos_w.shape == (4, 2, 3, 3)
    torch.testing.assert_close(
        reference.agent_body_pos_w[:, :, 1], reference.agent_body_pos_w[:, :, 2]
    )
    torch.testing.assert_close(
        reference.agent_body_quat_w[:, :, 1], reference.agent_body_quat_w[:, :, 2]
    )


def test_load_still_rejects_repeated_requested_joint_names(tmp_path):
    reference_path = tmp_path / "repeated_joint_request.npz"
    write_paired_reference(reference_path)

    with pytest.raises(ValueError, match="simulator joint names must be unique"):
        PairedMotionReference(
            str(reference_path),
            robot_body_names=["pelvis", "torso"],
            robot_joint_names=["joint_a", "joint_a"],
        )
