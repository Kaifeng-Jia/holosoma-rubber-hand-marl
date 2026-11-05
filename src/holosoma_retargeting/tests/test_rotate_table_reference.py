from __future__ import annotations

import json

import numpy as np
import pytest

from holosoma_retargeting.dual_a1_layout import (
    matrix_to_quaternion_wxyz,
    quaternion_wxyz_to_matrix,
)
from holosoma_retargeting.rotate_table_reference import (
    EXPECTED_OBJECT_URDF,
    EXPECTED_ROBOT_URDF,
    RECTANGULAR_TABLE_ORIGIN_HEIGHT_M,
    WORLD_Z_HALF_TURN,
    load_physical_pull_rollout_and_synthesize_rotate_table_reference,
    synthesize_rotate_table_reference,
)


def _fixture(frames: int = 7):
    phase = np.linspace(0.0, 1.0, frames)
    object_position = np.zeros((frames, 3), dtype=np.float64)
    object_position[:, 0] = 0.3 * phase
    object_position[:, 1] = -0.1 * phase
    object_position[:, 2] = 0.36 + 0.01 * phase
    object_rotation = np.repeat(np.eye(3)[None], frames, axis=0)
    object_qpos = np.concatenate(
        (object_position, matrix_to_quaternion_wxyz(object_rotation)), axis=-1
    )

    robot_position = np.zeros((frames, 3), dtype=np.float64)
    robot_position[:, 0] = 0.5
    robot_position[:, 1] = -0.4 - 0.2 * phase
    robot_position[:, 2] = 0.78
    robot_rotation = np.repeat(np.eye(3)[None], frames, axis=0)
    joints = np.repeat(np.linspace(-0.2, 0.2, 29)[None], frames, axis=0)
    joints += 0.05 * phase[:, None]
    robot_qpos = np.concatenate(
        (robot_position, matrix_to_quaternion_wxyz(robot_rotation), joints), axis=-1
    )
    joint_names = np.asarray([f"joint_{index}" for index in range(29)])
    return robot_qpos, object_qpos, joint_names


def _write_physical_pull(
    path,
    *,
    include_contact: bool,
    force_y: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    robot, object_qpos, joint_names = _fixture()
    frames = len(robot)
    root_quaternion_xyzw = robot[:, 3:7][:, [1, 2, 3, 0]]
    object_quaternion_xyzw = object_qpos[:, 3:7][:, [1, 2, 3, 0]]
    metadata = {
        "fps": 50,
        "dt": 0.02,
        "num_agents": 2,
        "dof_names": joint_names.tolist(),
        "object_physics": {
            "com_pose_b": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        },
        "object_robot_contact": {
            "force_semantics": (
                "World-frame contact force on the object sensor body from the "
                "filtered robot bodies."
            )
        },
    }
    fields = {
        "root_pos": np.stack((robot[:, :3], robot[:, :3] + 1.0), axis=1),
        "root_quat_xyzw": np.stack(
            (root_quaternion_xyzw, root_quaternion_xyzw), axis=1
        ),
        "dof_pos": np.stack((robot[:, 7:], robot[:, 7:] + 0.1), axis=1),
        "object_pos_w": object_qpos[:, :3],
        "object_quat_xyzw": object_quaternion_xyzw,
        "_metadata_json": np.asarray(json.dumps(metadata)),
    }
    if include_contact:
        contact_position = np.zeros((frames, 1, 2, 3), dtype=np.float64)
        contact_position[:, 0, 0, 0] = 1.0
        contact_force = np.zeros_like(contact_position)
        contact_force[:, 0, 0, 1] = force_y
        fields["object_robot_contact_pos_w"] = contact_position
        fields["object_robot_contact_force_matrix_w"] = contact_force
    np.savez_compressed(path, **fields)
    return robot, object_qpos, joint_names


def test_table_is_constant_upright_and_not_a_yaw_trajectory():
    robot, object_qpos, joint_names = _fixture()
    result = synthesize_rotate_table_reference(
        robot,
        object_qpos,
        fps=50,
        joint_names=joint_names,
    )

    np.testing.assert_allclose(
        result.object_qpos - result.object_qpos[:1], 0.0, atol=1.0e-12
    )
    np.testing.assert_allclose(
        result.object_qpos[:, 2], RECTANGULAR_TABLE_ORIGIN_HEIGHT_M
    )
    rotation = quaternion_wxyz_to_matrix(result.object_qpos[:, 3:7])
    np.testing.assert_allclose(
        rotation[..., :, 1] - np.array([0.0, 0.0, 1.0]),
        0.0,
        atol=1.0e-12,
    )
    assert result.signed_target_yaw_degrees == pytest.approx(-90.0)


def test_agent_a_is_source_motion_and_agent_b_is_strict_half_turn():
    robot, object_qpos, joint_names = _fixture()
    robot_before = robot.copy()
    object_before = object_qpos.copy()
    result = synthesize_rotate_table_reference(
        robot,
        object_qpos,
        fps=50,
        joint_names=joint_names,
    )

    np.testing.assert_allclose(result.robot_a_qpos, robot, atol=1.0e-12)
    expected_b_position = result.object_qpos[:, :3] + np.einsum(
        "ij,tj->ti",
        WORLD_Z_HALF_TURN,
        result.robot_a_qpos[:, :3] - result.object_qpos[:, :3],
    )
    np.testing.assert_allclose(result.robot_b_qpos[:, :3], expected_b_position)
    rotation_a = quaternion_wxyz_to_matrix(result.robot_a_qpos[:, 3:7])
    rotation_b = quaternion_wxyz_to_matrix(result.robot_b_qpos[:, 3:7])
    np.testing.assert_allclose(
        rotation_b,
        np.einsum("ij,tjk->tik", WORLD_Z_HALF_TURN, rotation_a),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(result.robot_b_qpos[:, 7:], robot[:, 7:])
    np.testing.assert_array_equal(robot, robot_before)
    np.testing.assert_array_equal(object_qpos, object_before)
    assert np.sign(result.pull_moment_proxy[0]) == np.sign(
        result.pull_moment_proxy[1]
    )


def test_canonical_rollout_marks_table_as_reset_only(tmp_path):
    robot, object_qpos, joint_names = _fixture()
    result = synthesize_rotate_table_reference(
        robot,
        object_qpos,
        fps=50,
        joint_names=joint_names,
    )
    output = result.save_canonical_rollout(
        tmp_path / "rotate.npz", provenance={"source": "fixture"}
    )

    with np.load(output, allow_pickle=False) as data:
        assert data["root_pos"].shape == (7, 2, 3)
        assert data["root_quat_xyzw"].shape == (7, 2, 4)
        assert data["dof_pos"].shape == (7, 2, 29)
        np.testing.assert_allclose(
            data["object_pos_w"] - data["object_pos_w"][:1], 0.0
        )
        np.testing.assert_allclose(
            data["object_quat_xyzw"] - data["object_quat_xyzw"][:1], 0.0
        )
        metadata = json.loads(str(data["provenance"].item()))

    assert metadata["source"] == "fixture"
    assert metadata["agent_b_source"] == "world_z_half_turn_copy_of_agent_a"
    assert metadata["expected_robot_urdf"] == EXPECTED_ROBOT_URDF
    assert metadata["expected_object_urdf"] == EXPECTED_OBJECT_URDF
    assert metadata["table_preview"] == "constant_upright_pose_no_animation"
    assert (
        metadata["table_channel_role"]
        == "reset_and_schema_only_not_tracking_target"
    )
    assert metadata["post_reset_table_motion"] == "physics_contact_only"


def test_physical_loader_copies_agent_zero_without_scene_rotation(tmp_path):
    source = tmp_path / "physical_pull.npz"
    robot, _, _ = _write_physical_pull(source, include_contact=False)

    with pytest.raises(ValueError, match="no object-contact channels"):
        load_physical_pull_rollout_and_synthesize_rotate_table_reference(source)
    result = load_physical_pull_rollout_and_synthesize_rotate_table_reference(
        source,
        rotation_direction=1.0,
    )

    np.testing.assert_allclose(result.robot_a_qpos, robot, atol=1.0e-12)
    np.testing.assert_allclose(result.robot_a_qpos[:, 7:], result.robot_b_qpos[:, 7:])
    np.testing.assert_allclose(
        result.object_qpos - result.object_qpos[:1], 0.0, atol=1.0e-12
    )
    assert result.signed_target_yaw_degrees == pytest.approx(90.0)
    assert result.rotation_direction_source == "explicit_rotation_direction"
    assert result.source_contact_yaw_angular_impulse_nms is None


@pytest.mark.parametrize("force_y, expected_yaw", [(1.0, 90.0), (-1.0, -90.0)])
def test_contact_impulse_sets_target_direction_but_does_not_move_table(
    tmp_path,
    force_y,
    expected_yaw,
):
    source = tmp_path / f"physical_contact_{force_y:+.0f}.npz"
    _write_physical_pull(source, include_contact=True, force_y=force_y)

    result = load_physical_pull_rollout_and_synthesize_rotate_table_reference(source)

    assert result.signed_target_yaw_degrees == pytest.approx(expected_yaw)
    assert (
        result.rotation_direction_source
        == "object_robot_contact_yaw_angular_impulse"
    )
    assert np.sign(result.source_contact_yaw_angular_impulse_nms) == np.sign(force_y)
    np.testing.assert_allclose(
        result.object_qpos - result.object_qpos[:1], 0.0, atol=1.0e-12
    )


@pytest.mark.parametrize("target", [0.0, -1.0, 181.0, np.nan])
def test_rejects_invalid_target_yaw(target):
    robot, object_qpos, joint_names = _fixture()
    with pytest.raises(ValueError, match="target_yaw_degrees"):
        synthesize_rotate_table_reference(
            robot,
            object_qpos,
            fps=50,
            joint_names=joint_names,
            target_yaw_degrees=target,
        )
