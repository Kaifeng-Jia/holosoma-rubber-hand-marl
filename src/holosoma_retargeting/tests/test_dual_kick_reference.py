from __future__ import annotations

import json

import numpy as np
import pytest

from holosoma_retargeting.dual_a1_layout import (
    g1_sagittal_joint_map,
    matrix_to_quaternion_wxyz,
    quaternion_wxyz_to_matrix,
)
from holosoma_retargeting.dual_kick_reference import (
    DEFAULT_KICK_SOURCE_CONTACT_LOCAL_Z_M,
    DEFAULT_KICK_TARGET_CONTACT_HALF_SPAN_M,
    G1_29DOF_JOINT_NAMES,
    KICK_LOCAL_XY_REFLECTION,
    synthesize_dual_kick_reference,
    synthesize_dual_kick_reference_file,
)
from holosoma_retargeting.dual_pull_reference import G1_LOCAL_SAGITTAL_REFLECTION


def _rotation_y(angle: float) -> np.ndarray:
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return np.asarray(
        [
            [cosine, 0.0, sine],
            [0.0, 1.0, 0.0],
            [-sine, 0.0, cosine],
        ]
    )


def _single_kick_channels(frames: int = 4) -> dict[str, np.ndarray]:
    root_pos = np.empty((frames, 3), dtype=np.float64)
    root_quat_xyzw = np.empty((frames, 4), dtype=np.float64)
    object_pos = np.empty((frames, 3), dtype=np.float64)
    object_quat_xyzw = np.empty((frames, 4), dtype=np.float64)
    dof_pos = np.empty((frames, 29), dtype=np.float64)
    robot_relative_pos = np.asarray(
        [-0.4, 0.38, DEFAULT_KICK_SOURCE_CONTACT_LOCAL_Z_M]
    )

    for frame in range(frames):
        fraction = frame / max(frames - 1, 1)
        object_rotation = _rotation_y(0.3 * fraction)
        object_pos[frame] = [0.3 * fraction, 0.36, 0.12 * fraction]
        root_pos[frame] = object_pos[frame] + object_rotation @ robot_relative_pos
        object_wxyz = matrix_to_quaternion_wxyz(object_rotation)
        root_wxyz = matrix_to_quaternion_wxyz(object_rotation)
        object_quat_xyzw[frame] = object_wxyz[[1, 2, 3, 0]]
        root_quat_xyzw[frame] = root_wxyz[[1, 2, 3, 0]]
        dof_pos[frame] = np.arange(29, dtype=np.float64) / 100.0 + fraction

    return {
        "root_pos": root_pos,
        "root_quat_xyzw": root_quat_xyzw,
        "dof_pos": dof_pos,
        "object_pos_w": object_pos,
        "object_quat_xyzw": object_quat_xyzw,
    }


def test_pair_keeps_same_action_and_places_contact_lanes_symmetrically() -> None:
    channels = _single_kick_channels()
    result = synthesize_dual_kick_reference(**channels, fps=50, layout="same_action")
    shared_rotation = quaternion_wxyz_to_matrix(result.shared_object_qpos[:, 3:7])

    a_local = np.einsum(
        "tji,tj->ti",
        shared_rotation,
        result.robot_a_qpos[:, :3] - result.shared_object_qpos[:, :3],
    )
    b_local = np.einsum(
        "tji,tj->ti",
        shared_rotation,
        result.robot_b_qpos[:, :3] - result.shared_object_qpos[:, :3],
    )
    np.testing.assert_allclose(
        a_local[:, 2],
        -DEFAULT_KICK_TARGET_CONTACT_HALF_SPAN_M,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        b_local[:, 2],
        DEFAULT_KICK_TARGET_CONTACT_HALF_SPAN_M,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        b_local[:, 2] - a_local[:, 2],
        2.0 * DEFAULT_KICK_TARGET_CONTACT_HALF_SPAN_M,
        atol=1.0e-12,
    )

    # Version 1 deliberately copies the accepted learned leg-contact action;
    # no left/right joint mirroring is introduced at this stage.
    np.testing.assert_allclose(result.robot_a_qpos[:, 3:], result.robot_b_qpos[:, 3:])
    np.testing.assert_allclose(result.robot_a_qpos[:, 7:], channels["dof_pos"])
    assert result.robot_a_lateral_offset_m == pytest.approx(-0.4390236)
    assert result.robot_b_lateral_offset_m == pytest.approx(0.9095236)


def test_default_pair_fully_mirrors_robot_b_across_shared_table() -> None:
    channels = _single_kick_channels()
    result = synthesize_dual_kick_reference(**channels, fps=50)
    assert result.layout == "mirrored"

    shared_pos = result.shared_object_qpos[:, :3]
    shared_rotation = quaternion_wxyz_to_matrix(result.shared_object_qpos[:, 3:7])
    world_reflection = np.einsum(
        "tij,jk,tlk->til",
        shared_rotation,
        KICK_LOCAL_XY_REFLECTION,
        shared_rotation,
    )
    expected_b_pos = shared_pos + np.einsum(
        "tij,tj->ti",
        world_reflection,
        result.robot_a_qpos[:, :3] - shared_pos,
    )
    np.testing.assert_allclose(result.robot_b_qpos[:, :3], expected_b_pos, atol=1.0e-12)

    robot_a_rotation = quaternion_wxyz_to_matrix(result.robot_a_qpos[:, 3:7])
    robot_b_rotation = quaternion_wxyz_to_matrix(result.robot_b_qpos[:, 3:7])
    expected_b_rotation = np.einsum(
        "tij,tjk,kl->til",
        world_reflection,
        robot_a_rotation,
        G1_LOCAL_SAGITTAL_REFLECTION,
    )
    np.testing.assert_allclose(robot_b_rotation, expected_b_rotation, atol=1.0e-12)
    np.testing.assert_allclose(np.linalg.det(robot_b_rotation), 1.0, atol=1.0e-12)

    index_map, sign_mask = g1_sagittal_joint_map(G1_29DOF_JOINT_NAMES)
    np.testing.assert_allclose(
        result.robot_b_qpos[:, 7:],
        result.robot_a_qpos[:, 7:][:, index_map] * sign_mask,
        atol=1.0e-12,
    )
    a_local = np.einsum(
        "tji,tj->ti",
        shared_rotation,
        result.robot_a_qpos[:, :3] - shared_pos,
    )
    b_local = np.einsum(
        "tji,tj->ti",
        shared_rotation,
        result.robot_b_qpos[:, :3] - shared_pos,
    )
    np.testing.assert_allclose(a_local[:, 2], -DEFAULT_KICK_TARGET_CONTACT_HALF_SPAN_M)
    np.testing.assert_allclose(b_local[:, 2], DEFAULT_KICK_TARGET_CONTACT_HALF_SPAN_M)
    assert result.robot_a_lateral_offset_m == pytest.approx(-0.4390236)
    assert result.robot_b_lateral_offset_m == pytest.approx(0.4390236)


def test_shared_table_removes_lateral_drift_and_vertical_yaw() -> None:
    channels = _single_kick_channels()
    result = synthesize_dual_kick_reference(**channels, fps=50)
    shared_rotation = quaternion_wxyz_to_matrix(result.shared_object_qpos[:, 3:7])
    initial_rotation = shared_rotation[0]
    local_displacement = np.einsum(
        "ij,tj->ti",
        initial_rotation.T,
        result.shared_object_qpos[:, :3] - result.shared_object_qpos[0, :3],
    )
    np.testing.assert_allclose(local_displacement[:, 2], 0.0, atol=1.0e-12)
    assert local_displacement[-1, 0] > 0.0

    relative_rotation = np.einsum("ij,tjk->tik", initial_rotation.T, shared_rotation)
    reflected = np.einsum(
        "ij,tjk,kl->til",
        KICK_LOCAL_XY_REFLECTION,
        relative_rotation,
        KICK_LOCAL_XY_REFLECTION,
    )
    np.testing.assert_allclose(relative_rotation, reflected, atol=1.0e-12)


def test_aggregate_attempt_extraction_and_canonical_file_contract(tmp_path) -> None:
    channels = _single_kick_channels(frames=3)
    aggregate = tmp_path / "kick_eval.npz"
    output = tmp_path / "dual_kick_viser.npz"
    metadata = {
        "fps": 50,
        "motion_time_step_total": 3,
        "dof_names": G1_29DOF_JOINT_NAMES.tolist(),
    }
    repeated = {name: np.concatenate((value, value), axis=0) for name, value in channels.items()}
    np.savez_compressed(
        aggregate,
        **repeated,
        motion_time_step=np.asarray([0, 1, 2, 0, 1, 2]),
        terminated=np.zeros(6, dtype=bool),
        _metadata_json=np.asarray(json.dumps(metadata)),
    )

    result = synthesize_dual_kick_reference_file(aggregate, source_attempt=1)
    assert result.robot_a_qpos.shape == (3, 36)
    assert result.provenance["source_attempt_start"] == 3
    assert result.provenance["source_attempt_end_exclusive"] == 6
    assert result.provenance["source_attempt_end_inclusive"] == 5
    assert result.provenance["source_attempt_completed"] is True
    assert result.provenance["source_attempt_terminated"] is False

    result.save_canonical_rollout(output, provenance={"review_status": "candidate"})
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
        assert saved["root_pos"].shape == (3, 2, 3)
        assert saved["root_quat_xyzw"].shape == (3, 2, 4)
        assert saved["dof_pos"].shape == (3, 2, 29)
        np.testing.assert_allclose(
            np.linalg.norm(saved["root_quat_xyzw"], axis=-1),
            1.0,
            atol=1.0e-12,
        )
        provenance = json.loads(saved["provenance"].item())
        assert provenance["source_attempt"] == 1
        assert provenance["layout"] == "mirrored"
        assert provenance["agent_joint_mirroring"] is True
        assert provenance["mirror_plane"] == "table_local_xy"
        assert provenance["review_status"] == "candidate"


def test_invalid_attempt_and_geometry_fail_closed(tmp_path) -> None:
    channels = _single_kick_channels(frames=3)
    aggregate = tmp_path / "kick_eval.npz"
    metadata = {
        "fps": 50,
        "motion_time_step_total": 3,
        "dof_names": G1_29DOF_JOINT_NAMES.tolist(),
    }
    np.savez_compressed(
        aggregate,
        **channels,
        motion_time_step=np.asarray([0, 1, 2]),
        terminated=np.zeros(3, dtype=bool),
        _metadata_json=np.asarray(json.dumps(metadata)),
    )
    with pytest.raises(ValueError, match="outside available range"):
        synthesize_dual_kick_reference_file(aggregate, source_attempt=4)
    with pytest.raises(ValueError, match="finite and positive"):
        synthesize_dual_kick_reference(**channels, fps=50, target_contact_half_span_m=0.0)
    with pytest.raises(ValueError, match="layout must be one of"):
        synthesize_dual_kick_reference(**channels, fps=50, layout="unknown")
