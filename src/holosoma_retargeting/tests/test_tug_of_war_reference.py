from __future__ import annotations

import json

import numpy as np
import pytest

from holosoma_retargeting.dual_a1_layout import (
    matrix_to_quaternion_wxyz,
    quaternion_wxyz_to_matrix,
)
from holosoma_retargeting.tug_of_war_reference import (
    load_single_pull_and_synthesize_tug_of_war_reference,
    synthesize_tug_of_war_reference,
)


def _qpos(position: np.ndarray, rotation: np.ndarray, joints: np.ndarray) -> np.ndarray:
    return np.concatenate((position, matrix_to_quaternion_wxyz(rotation), joints), axis=-1)


def _fixture(frames: int = 4):
    yaw = np.linspace(np.deg2rad(31.0), np.deg2rad(47.0), frames)
    tilt = np.linspace(np.deg2rad(-2.0), np.deg2rad(7.0), frames)
    table_rotation = []
    table_base = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    for yaw_frame, tilt_frame in zip(yaw, tilt, strict=True):
        world_yaw = np.array(
            [
                [np.cos(yaw_frame), -np.sin(yaw_frame), 0.0],
                [np.sin(yaw_frame), np.cos(yaw_frame), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        local_tilt = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(tilt_frame), -np.sin(tilt_frame)],
                [0.0, np.sin(tilt_frame), np.cos(tilt_frame)],
            ]
        )
        table_rotation.append(world_yaw @ table_base @ local_tilt)
    table_rotation = np.asarray(table_rotation)
    object_position = np.stack(
        (np.linspace(0.0, 0.3, frames), np.linspace(0.1, 0.2, frames), np.ones(frames) * 0.4),
        axis=-1,
    )
    object_qpos = np.concatenate(
        (object_position, matrix_to_quaternion_wxyz(table_rotation)),
        axis=-1,
    )
    pull_progress = np.linspace(0.0, 0.15, frames)
    radial = np.array([-0.5, 0.0, -0.6])
    radial /= np.linalg.norm(radial)
    local_a = np.repeat(np.array([[-0.5, 0.4, -0.6]]), frames, axis=0)
    local_a += pull_progress[:, None] * radial
    local_b = np.repeat(np.array([[-0.5, 0.4, 0.6]]), frames, axis=0)
    robot_rotation = np.repeat(np.eye(3)[None], frames, axis=0)
    joints_a = np.repeat(np.linspace(-0.2, 0.2, 29)[None], frames, axis=0)
    joints_b = np.repeat(np.linspace(0.3, -0.3, 29)[None], frames, axis=0)
    robot_a = _qpos(
        object_position + np.einsum("tij,tj->ti", table_rotation, local_a),
        robot_rotation,
        joints_a,
    )
    robot_b = _qpos(
        object_position + np.einsum("tij,tj->ti", table_rotation, local_b),
        robot_rotation,
        joints_b,
    )
    return robot_a, robot_b, object_qpos, joints_b


def test_opponents_are_at_diagonally_opposite_table_legs():
    robot_a, _, object_qpos, _ = _fixture()
    result = synthesize_tug_of_war_reference(
        robot_a,
        object_qpos,
        fps=50,
        joint_names=np.asarray([f"joint_{index}" for index in range(29)]),
    )
    relative_a = result.robot_a_qpos[:, :3] - result.object_qpos[:, :3]
    relative_b = result.robot_b_qpos[:, :3] - result.object_qpos[:, :3]
    np.testing.assert_allclose(relative_a[:, :2], -relative_b[:, :2], atol=1.0e-12)
    np.testing.assert_allclose(relative_a[:, 2], relative_b[:, 2], atol=1.0e-12)
    np.testing.assert_allclose(result.robot_b_qpos[:, 7:], result.robot_a_qpos[:, 7:])


def test_planar_neutralization_preserves_floor_and_table_relative_pull_pose():
    robot_a, _, object_qpos, _ = _fixture()
    result = synthesize_tug_of_war_reference(
        robot_a,
        object_qpos,
        fps=50,
        joint_names=np.asarray([f"joint_{index}" for index in range(29)]),
    )
    source_object_rotation = quaternion_wxyz_to_matrix(object_qpos[:, 3:7])
    target_object_rotation = quaternion_wxyz_to_matrix(result.object_qpos[:, 3:7])
    source_robot_rotation = quaternion_wxyz_to_matrix(robot_a[:, 3:7])
    target_robot_rotation = quaternion_wxyz_to_matrix(result.robot_a_qpos[:, 3:7])

    np.testing.assert_allclose(result.object_qpos[:, :2] - result.object_qpos[:1, :2], 0.0)
    np.testing.assert_allclose(result.object_qpos[:, 2], object_qpos[:, 2])
    np.testing.assert_allclose(result.robot_a_qpos[:, 2], robot_a[:, 2])
    np.testing.assert_allclose(
        target_object_rotation[:, 2], source_object_rotation[:, 2], atol=1.0e-12
    )
    np.testing.assert_allclose(
        target_robot_rotation[:, 2], source_robot_rotation[:, 2], atol=1.0e-12
    )
    np.testing.assert_allclose(result.robot_a_qpos[:, 7:], robot_a[:, 7:])

    source_relative_position = np.einsum(
        "tji,tj->ti", source_object_rotation, robot_a[:, :3] - object_qpos[:, :3]
    )
    target_relative_position = np.einsum(
        "tji,tj->ti",
        target_object_rotation,
        result.robot_a_qpos[:, :3] - result.object_qpos[:, :3],
    )
    source_relative_rotation = np.einsum(
        "tji,tjk->tik", source_object_rotation, source_robot_rotation
    )
    target_relative_rotation = np.einsum(
        "tji,tjk->tik", target_object_rotation, target_robot_rotation
    )
    np.testing.assert_allclose(target_relative_position, source_relative_position, atol=1.0e-12)
    np.testing.assert_allclose(target_relative_rotation, source_relative_rotation, atol=1.0e-12)

    heading = target_object_rotation[:, :2, 0]
    heading /= np.linalg.norm(heading, axis=-1, keepdims=True)
    np.testing.assert_allclose(heading - heading[:1], 0.0, atol=1.0e-12)


def test_pull_progress_is_preserved_for_both_opponents():
    robot_a, _, object_qpos, _ = _fixture()
    result = synthesize_tug_of_war_reference(
        robot_a,
        object_qpos,
        fps=50,
        joint_names=np.asarray([f"joint_{index}" for index in range(29)]),
    )
    relative_a = result.robot_a_qpos[:, :2] - result.object_qpos[:, :2]
    relative_b = result.robot_b_qpos[:, :2] - result.object_qpos[:, :2]
    initial_direction_a = relative_a[0] / np.linalg.norm(relative_a[0])
    initial_direction_b = relative_b[0] / np.linalg.norm(relative_b[0])
    progress_a = (relative_a[-1] - relative_a[0]) @ initial_direction_a
    progress_b = (relative_b[-1] - relative_b[0]) @ initial_direction_b
    assert progress_a > 0.0
    np.testing.assert_allclose(progress_b, progress_a, atol=1.0e-12)


def test_opponent_root_is_half_turned_about_table_vertical():
    robot_a, _, object_qpos, _ = _fixture()
    result = synthesize_tug_of_war_reference(
        robot_a,
        object_qpos,
        fps=50,
        joint_names=np.asarray([f"joint_{index}" for index in range(29)]),
    )
    expected_half_turn = np.diag([-1.0, -1.0, 1.0])
    actual_a_rotation = quaternion_wxyz_to_matrix(result.robot_a_qpos[:, 3:7])
    actual_b_rotation = quaternion_wxyz_to_matrix(result.robot_b_qpos[:, 3:7])
    np.testing.assert_allclose(actual_b_rotation, expected_half_turn @ actual_a_rotation, atol=1.0e-12)


def test_canonical_rollout_schema_and_metadata(tmp_path):
    robot_a, _, object_qpos, _ = _fixture()
    result = synthesize_tug_of_war_reference(
        robot_a,
        object_qpos,
        fps=50,
        joint_names=np.asarray([f"joint_{index}" for index in range(29)]),
    )
    output = result.save_canonical_rollout(tmp_path / "tug.npz", provenance={"source": "fixture"})
    with np.load(output, allow_pickle=False) as data:
        assert data["root_pos"].shape == (4, 2, 3)
        assert data["root_quat_xyzw"].shape == (4, 2, 4)
        assert data["dof_pos"].shape == (4, 2, 29)
        assert data["object_pos_w"].shape == (4, 3)
        assert data["object_quat_xyzw"].shape == (4, 4)
        metadata = json.loads(str(data["provenance"].item()))
    assert metadata["source"] == "fixture"
    assert metadata["schema_version"] == 2
    assert metadata["opponent_transform"] == "world_z_half_turn"
    assert metadata["table_preview"] == "planar_neutralized_xy_heading_not_training_target"


def test_rejects_degenerate_table_heading():
    robot_a, _, object_qpos, _ = _fixture()
    rotation = np.repeat(np.eye(3)[None], len(object_qpos), axis=0)
    rotation[:, :, 0] = np.array([0.0, 0.0, 1.0])
    rotation[:, :, 1] = np.array([1.0, 0.0, 0.0])
    rotation[:, :, 2] = np.array([0.0, 1.0, 0.0])
    object_qpos[:, 3:7] = matrix_to_quaternion_wxyz(rotation)
    with pytest.raises(ValueError, match="degenerate world-XY projection"):
        synthesize_tug_of_war_reference(
            robot_a,
            object_qpos,
            fps=50,
            joint_names=np.asarray([f"joint_{index}" for index in range(29)]),
        )


def test_loads_single_pull_motion_schema(tmp_path):
    robot_a, _, object_qpos, _ = _fixture()
    joint_names = np.asarray([f"joint_{index}" for index in range(29)])
    source = tmp_path / "single_pull.npz"
    np.savez_compressed(
        source,
        joint_pos=robot_a,
        object_pos_w=object_qpos[:, :3],
        object_quat_w=object_qpos[:, 3:7],
        joint_names=joint_names,
        fps=np.asarray([50], dtype=np.int64),
    )
    result = load_single_pull_and_synthesize_tug_of_war_reference(source)
    assert result.robot_a_qpos.shape == (4, 36)
    assert result.robot_b_qpos.shape == (4, 36)
    assert result.fps == 50
    np.testing.assert_array_equal(result.joint_names, joint_names)


def test_rejects_invalid_shapes():
    robot_a, _, object_qpos, _ = _fixture()
    with pytest.raises(ValueError, match="robot trajectory"):
        synthesize_tug_of_war_reference(
            robot_a[:, :-1],
            object_qpos,
            fps=50,
            joint_names=np.asarray([f"joint_{index}" for index in range(29)]),
        )
