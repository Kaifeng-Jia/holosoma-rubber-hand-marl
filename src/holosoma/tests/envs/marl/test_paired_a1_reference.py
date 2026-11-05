from __future__ import annotations

import pytest
import torch

from holosoma.envs.marl import PairedA1Reference


def _reference() -> PairedA1Reference:
    frames = 5
    angles = torch.linspace(0.0, 0.4, frames)
    object_quat = torch.stack(
        (
            torch.zeros(frames),
            torch.zeros(frames),
            torch.sin(angles / 2.0),
            torch.cos(angles / 2.0),
        ),
        dim=-1,
    )
    body_pos = torch.zeros((frames, 3, 3))
    body_pos[:, :, 1] = torch.linspace(0.0, 0.2, frames).view(-1, 1)
    body_quat = torch.zeros((frames, 3, 4))
    body_quat[..., 3] = 1.0
    return PairedA1Reference(
        joint_pos=torch.zeros((frames, 29)),
        joint_vel=torch.zeros((frames, 29)),
        body_pos_w=body_pos,
        body_quat_w=body_quat,
        body_lin_vel_w=torch.zeros_like(body_pos),
        body_ang_vel_w=torch.zeros_like(body_pos),
        object_pos_w=torch.zeros((frames, 3)),
        object_quat_w=object_quat,
        object_lin_vel_w=torch.zeros((frames, 3)),
        fps=50,
        lateral_spacing_m=0.8,
    )


def test_paired_roots_are_symmetric_with_constant_spacing() -> None:
    reference = _reference()
    roots = reference.agent_body_pos_w[:, :, 0]
    source_root = roots.mean(dim=1)
    expected_source_root = torch.zeros((5, 3))
    expected_source_root[:, 1] = torch.linspace(0.0, 0.2, 5)

    torch.testing.assert_close(source_root, expected_source_root)
    torch.testing.assert_close(torch.linalg.vector_norm(roots[:, 1] - roots[:, 0], dim=-1), torch.full((5,), 0.8))


def test_lateral_axis_rotates_with_shared_table_orientation() -> None:
    reference = _reference()
    first_offset = reference.lateral_offset_w[0, 1]
    final_offset = reference.lateral_offset_w[-1, 1]

    torch.testing.assert_close(first_offset, torch.tensor([0.4, 0.0, 0.0]))
    torch.testing.assert_close(
        final_offset,
        torch.tensor([0.4 * torch.cos(torch.tensor(0.4)), 0.4 * torch.sin(torch.tensor(0.4)), 0.0]),
    )


def test_robot_joint_motion_is_shared_but_object_is_not_duplicated() -> None:
    reference = _reference()

    assert reference.agent_joint_pos.shape == (5, 2, 29)
    assert reference.object_pos_w.shape == (5, 3)
    torch.testing.assert_close(reference.agent_joint_pos[:, 0], reference.agent_joint_pos[:, 1])


def test_phase_sampling_keeps_agents_and_object_synchronized() -> None:
    reference = _reference()
    sample = reference.sample(torch.tensor([1, 4]))

    assert sample["agent_body_pos_w"].shape == (2, 2, 3, 3)
    torch.testing.assert_close(sample["object_quat_w"], reference.object_quat_w[[1, 4]])
    torch.testing.assert_close(sample["agent_joint_pos"], reference.agent_joint_pos[[1, 4]])


def test_invalid_spacing_and_phase_are_rejected() -> None:
    reference = _reference()
    with pytest.raises(IndexError, match="out-of-range"):
        reference.sample(torch.tensor([5]))

    with pytest.raises(ValueError, match="positive"):
        PairedA1Reference(
            joint_pos=torch.zeros((3, 1)),
            joint_vel=torch.zeros((3, 1)),
            body_pos_w=torch.zeros((3, 1, 3)),
            body_quat_w=torch.tensor([[[0.0, 0.0, 0.0, 1.0]]] * 3),
            body_lin_vel_w=torch.zeros((3, 1, 3)),
            body_ang_vel_w=torch.zeros((3, 1, 3)),
            object_pos_w=torch.zeros((3, 3)),
            object_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 3),
            object_lin_vel_w=torch.zeros((3, 3)),
            fps=50,
            lateral_spacing_m=0.0,
        )
