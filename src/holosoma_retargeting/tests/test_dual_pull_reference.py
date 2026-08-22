from __future__ import annotations

import json

import numpy as np
import pytest

from holosoma_retargeting.dual_a1_layout import (
    g1_sagittal_joint_map,
    quaternion_wxyz_to_matrix,
)
from holosoma_retargeting.dual_pull_reference import (
    FORMAL_PULL_LATERAL_LEG_OFFSET_M,
    G1_29DOF_JOINT_NAMES,
    G1_LOCAL_SAGITTAL_REFLECTION,
    PULL_LOCAL_XY_REFLECTION,
    synthesize_dual_pull_reference,
    synthesize_dual_pull_reference_file,
)


def _rotation(axis: str, angle: float) -> np.ndarray:
    cosine = np.cos(angle)
    sine = np.sin(angle)
    if axis == "x":
        return np.array([[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]])
    if axis == "y":
        return np.array([[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]])
    if axis == "z":
        return np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])
    raise ValueError(axis)


def _matrix_to_wxyz(rotation: np.ndarray) -> np.ndarray:
    # The tests only need the positive-trace branch for their bounded angles.
    scale = 2.0 * np.sqrt(1.0 + np.trace(rotation))
    return np.array(
        [
            0.25 * scale,
            (rotation[2, 1] - rotation[1, 2]) / scale,
            (rotation[0, 2] - rotation[2, 0]) / scale,
            (rotation[1, 0] - rotation[0, 1]) / scale,
        ]
    )


def _single_pull_qpos(frames: int = 4) -> np.ndarray:
    initial_table_rotation = _rotation("z", 0.35) @ _rotation("x", -0.2)
    robot_relative_rotation = _rotation("y", -0.25) @ _rotation("x", 0.1)
    initial_table_pos = np.array([0.3, -0.2, 0.4])
    robot_relative_pos = np.array([0.45, 0.55, -0.3])
    qpos = np.zeros((frames, 43), dtype=np.float64)
    for frame in range(frames):
        fraction = frame / max(frames - 1, 1)
        relative_table_rotation = (
            _rotation("x", 0.18 * fraction)
            @ _rotation("y", -0.24 * fraction)
            @ _rotation("z", 0.12 * fraction)
        )
        table_rotation = initial_table_rotation @ relative_table_rotation
        table_local_translation = np.array([0.8 * fraction, 0.04 * fraction, 0.3 * fraction])
        table_pos = initial_table_pos + initial_table_rotation @ table_local_translation
        robot_rotation = table_rotation @ robot_relative_rotation
        robot_pos = table_pos + table_rotation @ robot_relative_pos

        qpos[frame, :3] = robot_pos
        qpos[frame, 3:7] = _matrix_to_wxyz(robot_rotation)
        qpos[frame, 7:36] = np.arange(29) / 100.0 + fraction
        qpos[frame, 36:39] = table_pos
        qpos[frame, 39:43] = _matrix_to_wxyz(table_rotation)
    return qpos


def _rotation_angle(rotation: np.ndarray) -> np.ndarray:
    cosine = np.clip((np.trace(rotation, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    return np.arccos(cosine)


def test_shared_object_is_symmetric_in_initial_table_coordinates() -> None:
    qpos = _single_pull_qpos()
    result = synthesize_dual_pull_reference(qpos, fps=50)
    source_object = qpos[:, 36:43]
    source_rotation = quaternion_wxyz_to_matrix(source_object[:, 3:7])
    shared_rotation = quaternion_wxyz_to_matrix(result.shared_object_qpos[:, 3:7])
    initial_rotation = source_rotation[0]

    shared_local_translation = np.einsum(
        "ij,tj->ti",
        initial_rotation.T,
        result.shared_object_qpos[:, :3] - source_object[0, :3],
    )
    np.testing.assert_allclose(shared_local_translation[:, 2], 0.0, atol=1.0e-12)
    # Pull symmetrization must preserve local-X progress rather than treating the
    # Push A1 local-X axis as lateral.
    assert shared_local_translation[-1, 0] == pytest.approx(0.8)

    shared_relative_rotation = np.einsum("ij,tjk->tik", initial_rotation.T, shared_rotation)
    reflected_shared_rotation = np.einsum(
        "ij,tjk,kl->til",
        PULL_LOCAL_XY_REFLECTION,
        shared_relative_rotation,
        PULL_LOCAL_XY_REFLECTION,
    )
    np.testing.assert_allclose(shared_relative_rotation, reflected_shared_rotation, atol=1.0e-12)

    source_relative_rotation = np.einsum("ij,tjk->tik", initial_rotation.T, source_rotation)
    mirrored_source_rotation = np.einsum(
        "ij,tjk,kl->til",
        PULL_LOCAL_XY_REFLECTION,
        source_relative_rotation,
        PULL_LOCAL_XY_REFLECTION,
    )
    first_distance = _rotation_angle(
        np.einsum("tji,tjk->tik", source_relative_rotation, shared_relative_rotation)
    )
    second_distance = _rotation_angle(
        np.einsum("tji,tjk->tik", mirrored_source_rotation, shared_relative_rotation)
    )
    np.testing.assert_allclose(first_distance, second_distance, atol=1.0e-8)


def test_robot_a_keeps_table_relative_pose_and_robot_b_is_exact_mirror() -> None:
    qpos = _single_pull_qpos()
    result = synthesize_dual_pull_reference(qpos, fps=50)
    source_object = qpos[:, 36:43]
    source_object_rotation = quaternion_wxyz_to_matrix(source_object[:, 3:7])
    source_robot_rotation = quaternion_wxyz_to_matrix(qpos[:, 3:7])
    shared_rotation = quaternion_wxyz_to_matrix(result.shared_object_qpos[:, 3:7])
    robot_a_rotation = quaternion_wxyz_to_matrix(result.robot_a_qpos[:, 3:7])
    robot_b_rotation = quaternion_wxyz_to_matrix(result.robot_b_qpos[:, 3:7])

    source_relative_pos = np.einsum(
        "tji,tj->ti",
        source_object_rotation,
        qpos[:, :3] - source_object[:, :3],
    )
    robot_a_relative_pos = np.einsum(
        "tji,tj->ti",
        shared_rotation,
        result.robot_a_qpos[:, :3] - result.shared_object_qpos[:, :3],
    )
    np.testing.assert_allclose(robot_a_relative_pos, source_relative_pos, atol=1.0e-12)
    np.testing.assert_allclose(
        np.einsum("tji,tjk->tik", shared_rotation, robot_a_rotation),
        np.einsum("tji,tjk->tik", source_object_rotation, source_robot_rotation),
        atol=1.0e-12,
    )

    world_reflection = np.einsum(
        "tij,jk,tlk->til",
        shared_rotation,
        PULL_LOCAL_XY_REFLECTION,
        shared_rotation,
    )
    expected_b_pos = result.shared_object_qpos[:, :3] + np.einsum(
        "tij,tj->ti",
        world_reflection,
        result.robot_a_qpos[:, :3] - result.shared_object_qpos[:, :3],
    )
    expected_b_rotation = np.einsum(
        "tij,tjk,kl->til",
        world_reflection,
        robot_a_rotation,
        G1_LOCAL_SAGITTAL_REFLECTION,
    )
    np.testing.assert_allclose(result.robot_b_qpos[:, :3], expected_b_pos, atol=1.0e-12)
    np.testing.assert_allclose(robot_b_rotation, expected_b_rotation, atol=1.0e-12)

    index_map, sign_mask = g1_sagittal_joint_map(G1_29DOF_JOINT_NAMES)
    np.testing.assert_allclose(
        result.robot_b_qpos[:, 7:],
        result.robot_a_qpos[:, 7:][:, index_map] * sign_mask,
    )


def test_optional_leg_offset_moves_agents_equally_outward_along_local_z() -> None:
    qpos = _single_pull_qpos()
    pure = synthesize_dual_pull_reference(qpos, fps=50)
    offset = synthesize_dual_pull_reference(
        qpos,
        fps=50,
        lateral_leg_offset_m=FORMAL_PULL_LATERAL_LEG_OFFSET_M,
    )
    shared_rotation = quaternion_wxyz_to_matrix(pure.shared_object_qpos[:, 3:7])
    local_z_w = shared_rotation[..., :, 2]

    np.testing.assert_allclose(offset.shared_object_qpos, pure.shared_object_qpos)
    np.testing.assert_allclose(
        offset.robot_a_qpos[:, :3] - pure.robot_a_qpos[:, :3],
        -FORMAL_PULL_LATERAL_LEG_OFFSET_M * local_z_w,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        offset.robot_b_qpos[:, :3] - pure.robot_b_qpos[:, :3],
        FORMAL_PULL_LATERAL_LEG_OFFSET_M * local_z_w,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(offset.robot_a_qpos[:, 3:], pure.robot_a_qpos[:, 3:])
    np.testing.assert_allclose(offset.robot_b_qpos[:, 3:], pure.robot_b_qpos[:, 3:])


def test_quaternions_are_unit_wxyz_and_file_contract_round_trips(tmp_path) -> None:
    source = tmp_path / "attempt08.npz"
    output = tmp_path / "dual_pull.npz"
    qpos = _single_pull_qpos()
    np.savez(source, qpos=qpos, fps=np.asarray(50))

    result = synthesize_dual_pull_reference_file(source, output)

    assert result.fps == 50
    for quaternion in (
        result.robot_a_qpos[:, 3:7],
        result.robot_b_qpos[:, 3:7],
        result.shared_object_qpos[:, 3:7],
    ):
        np.testing.assert_allclose(np.linalg.norm(quaternion, axis=-1), 1.0, atol=1.0e-12)
        assert np.all(np.sum(quaternion[:-1] * quaternion[1:], axis=-1) >= 0.0)
    # WXYZ identity has its scalar in column zero; this also catches an XYZW
    # interpretation in the first-frame shared-table path.
    np.testing.assert_allclose(
        quaternion_wxyz_to_matrix(result.shared_object_qpos[:1, 3:7]),
        quaternion_wxyz_to_matrix(qpos[:1, 39:43]),
        atol=1.0e-12,
    )

    with np.load(output, allow_pickle=False) as saved:
        assert set(saved.files) == {
            "robot_a_qpos",
            "robot_b_qpos",
            "shared_object_qpos",
            "fps",
            "joint_names",
            "quaternion_convention",
            "lateral_axis",
            "mirror_plane",
            "lateral_leg_offset_m",
        }
        assert saved["robot_a_qpos"].shape == (4, 36)
        assert saved["robot_b_qpos"].shape == (4, 36)
        assert saved["shared_object_qpos"].shape == (4, 7)
        assert saved["quaternion_convention"].item() == "wxyz"
        assert saved["lateral_axis"].item() == "table_local_z"
        assert saved["mirror_plane"].item() == "table_local_xy"


def test_canonical_rollout_matches_viser_five_channel_contract(tmp_path) -> None:
    output = tmp_path / "dual_pull_viser.npz"
    result = synthesize_dual_pull_reference(
        _single_pull_qpos(),
        fps=50,
        lateral_leg_offset_m=FORMAL_PULL_LATERAL_LEG_OFFSET_M,
    )

    result.save_canonical_rollout(
        output,
        provenance={"source": "attempt08.npz", "source_attempt": 8},
    )

    with np.load(output, allow_pickle=False) as saved:
        assert set(saved.files) == {
            "root_pos",
            "root_quat_xyzw",
            "dof_pos",
            "object_pos_w",
            "object_quat_xyzw",
            "fps",
            "provenance",
        }
        assert saved["root_pos"].shape == (4, 2, 3)
        assert saved["root_quat_xyzw"].shape == (4, 2, 4)
        assert saved["dof_pos"].shape == (4, 2, 29)
        assert saved["object_pos_w"].shape == (4, 3)
        assert saved["object_quat_xyzw"].shape == (4, 4)
        np.testing.assert_allclose(saved["root_pos"][:, 0], result.robot_a_qpos[:, :3])
        np.testing.assert_allclose(saved["root_pos"][:, 1], result.robot_b_qpos[:, :3])
        np.testing.assert_allclose(
            saved["root_quat_xyzw"][:, 0],
            result.robot_a_qpos[:, [4, 5, 6, 3]],
        )
        np.testing.assert_allclose(
            saved["object_quat_xyzw"],
            result.shared_object_qpos[:, [4, 5, 6, 3]],
        )
        provenance = json.loads(saved["provenance"].item())
        assert provenance["source"] == "attempt08.npz"
        assert provenance["source_attempt"] == 8
        assert provenance["lateral_leg_offset_m"] == FORMAL_PULL_LATERAL_LEG_OFFSET_M


@pytest.mark.parametrize(
    ("qpos", "fps", "message"),
    [
        (np.zeros((3, 42)), 50, "shape"),
        (np.zeros((3, 43)), 0, "positive integer"),
        (np.full((3, 43), np.nan), 50, "finite"),
    ],
)
def test_invalid_source_fails_closed(qpos, fps, message) -> None:
    with pytest.raises(ValueError, match=message):
        synthesize_dual_pull_reference(qpos, fps=fps)


@pytest.mark.parametrize("offset", [-0.1, np.nan, np.inf])
def test_invalid_lateral_leg_offset_fails_closed(offset) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        synthesize_dual_pull_reference(
            _single_pull_qpos(),
            fps=50,
            lateral_leg_offset_m=offset,
        )
