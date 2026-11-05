from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

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
)
from holosoma_retargeting.dual_pull_runtime_reference import (
    DEFAULT_RUBBER_HAND_G1_XML,
    build_dual_pull_runtime_reference,
    build_dual_pull_runtime_reference_file,
    sha256_file,
)


def _yaw_quaternion_wxyz(angle: float) -> np.ndarray:
    return np.asarray([np.cos(0.5 * angle), 0.0, 0.0, np.sin(0.5 * angle)])


def _canonical_reference(frames: int = 7, fps: int = 50):
    qpos = np.zeros((frames, 43), dtype=np.float64)
    for frame in range(frames):
        time = frame / fps
        object_pos = np.asarray([0.35 * time, -0.1, 0.42])
        object_quat = _yaw_quaternion_wxyz(0.4 * time)
        object_rotation = quaternion_wxyz_to_matrix(object_quat)
        robot_relative_pos = np.asarray([0.45, 0.25, -0.3])
        robot_pos = object_pos + object_rotation @ robot_relative_pos
        robot_quat = _yaw_quaternion_wxyz(0.4 * time + 0.2)

        qpos[frame, :3] = robot_pos
        qpos[frame, 3:7] = robot_quat
        qpos[frame, 7:36] = 0.02 * np.arange(29) + 0.1 * time
        qpos[frame, 36:39] = object_pos
        qpos[frame, 39:43] = object_quat
    return synthesize_dual_pull_reference(
        qpos,
        fps=fps,
        lateral_leg_offset_m=FORMAL_PULL_LATERAL_LEG_OFFSET_M,
    )


def _save_canonical_qpos(path, reference) -> None:
    reference.save(path)


def test_runtime_reference_shapes_names_finiteness_and_unit_quaternions() -> None:
    canonical = _canonical_reference()
    result = build_dual_pull_runtime_reference(canonical)
    frames = len(canonical.robot_a_qpos)

    assert result.agent_joint_pos.shape == (frames, 2, 29)
    assert result.agent_joint_vel.shape == (frames, 2, 29)
    assert result.agent_body_pos_w.shape == (frames, 2, 51, 3)
    assert result.agent_body_quat_wxyz.shape == (frames, 2, 51, 4)
    assert result.agent_body_lin_vel_w.shape == (frames, 2, 51, 3)
    assert result.agent_body_ang_vel_w.shape == (frames, 2, 51, 3)
    assert result.object_pos_w.shape == (frames, 3)
    assert result.object_quat_wxyz.shape == (frames, 4)
    assert result.object_lin_vel_w.shape == (frames, 3)
    assert result.object_ang_vel_w.shape == (frames, 3)
    np.testing.assert_array_equal(result.joint_names, G1_29DOF_JOINT_NAMES)
    assert result.body_names[0] == "world"
    assert "left_rubber_hand_link" in result.body_names
    assert "right_rubber_hand_link" in result.body_names

    numeric_channels = (
        result.agent_joint_pos,
        result.agent_joint_vel,
        result.agent_body_pos_w,
        result.agent_body_quat_wxyz,
        result.agent_body_lin_vel_w,
        result.agent_body_ang_vel_w,
        result.object_pos_w,
        result.object_quat_wxyz,
        result.object_lin_vel_w,
        result.object_ang_vel_w,
    )
    assert all(np.isfinite(channel).all() for channel in numeric_channels)
    np.testing.assert_allclose(
        np.linalg.norm(result.agent_body_quat_wxyz, axis=-1),
        1.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(np.linalg.norm(result.object_quat_wxyz, axis=-1), 1.0, atol=1.0e-12)


def test_forward_kinematics_round_trip_matches_fixed_rubber_hand_model() -> None:
    canonical = _canonical_reference()
    result = build_dual_pull_runtime_reference(canonical)
    model = mujoco.MjModel.from_xml_path(str(DEFAULT_RUBBER_HAND_G1_XML))
    data = mujoco.MjData(model)

    for agent, source_qpos in enumerate((canonical.robot_a_qpos, canonical.robot_b_qpos)):
        for frame in (0, len(source_qpos) // 2, len(source_qpos) - 1):
            data.qpos[:] = source_qpos[frame]
            mujoco.mj_forward(model, data)
            np.testing.assert_allclose(result.agent_body_pos_w[frame, agent], data.xpos, atol=1.0e-12)
            # q and -q represent the same orientation; compare rotation matrices.
            np.testing.assert_allclose(
                quaternion_wxyz_to_matrix(result.agent_body_quat_wxyz[frame, agent]),
                quaternion_wxyz_to_matrix(data.xquat),
                atol=1.0e-12,
            )


def test_mirrored_agents_and_shared_object_survive_full_expansion() -> None:
    canonical = _canonical_reference()
    result = build_dual_pull_runtime_reference(canonical)
    table_rotation = quaternion_wxyz_to_matrix(result.object_quat_wxyz)
    root_a_pos = canonical.robot_a_qpos[:, :3]
    root_b_pos = canonical.robot_b_qpos[:, :3]
    root_a_rotation = quaternion_wxyz_to_matrix(canonical.robot_a_qpos[:, 3:7])
    root_b_rotation = quaternion_wxyz_to_matrix(canonical.robot_b_qpos[:, 3:7])
    world_reflection = np.einsum(
        "tij,jk,tlk->til",
        table_rotation,
        PULL_LOCAL_XY_REFLECTION,
        table_rotation,
    )
    expected_b_pos = result.object_pos_w + np.einsum(
        "tij,tj->ti",
        world_reflection,
        root_a_pos - result.object_pos_w,
    )
    expected_b_rotation = np.einsum(
        "tij,tjk,kl->til",
        world_reflection,
        root_a_rotation,
        G1_LOCAL_SAGITTAL_REFLECTION,
    )
    np.testing.assert_allclose(root_b_pos, expected_b_pos, atol=1.0e-12)
    np.testing.assert_allclose(root_b_rotation, expected_b_rotation, atol=1.0e-12)

    index_map, sign_mask = g1_sagittal_joint_map(result.joint_names)
    np.testing.assert_allclose(
        result.agent_joint_pos[:, 1],
        result.agent_joint_pos[:, 0][:, index_map] * sign_mask,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(result.object_pos_w, canonical.shared_object_qpos[:, :3])
    np.testing.assert_allclose(result.object_quat_wxyz, canonical.shared_object_qpos[:, 3:7])


def test_world_velocities_match_known_motion_and_body_position_difference() -> None:
    canonical = _canonical_reference(frames=9, fps=50)
    result = build_dual_pull_runtime_reference(canonical)
    pelvis_index = int(np.flatnonzero(result.body_names == "pelvis")[0])
    dt = 1.0 / result.fps

    expected_object_linear = np.tile(np.asarray([0.35, 0.0, 0.0]), (9, 1))
    expected_object_angular = np.tile(np.asarray([0.0, 0.0, 0.4]), (9, 1))
    np.testing.assert_allclose(result.object_lin_vel_w, expected_object_linear, atol=1.0e-12)
    np.testing.assert_allclose(result.object_ang_vel_w, expected_object_angular, atol=1.0e-12)

    root_position_gradient = np.gradient(
        result.agent_body_pos_w[:, :, pelvis_index],
        dt,
        axis=0,
        # MuJoCo qvel uses central differences internally and one-sided
        # differences at the two sequence boundaries.
        edge_order=1,
    )
    np.testing.assert_allclose(
        result.agent_body_lin_vel_w[:, :, pelvis_index],
        root_position_gradient,
        atol=1.0e-12,
    )
    # Both pelvis orientations have a constant world-Z yaw rate.  This checks
    # that body angular velocity was not accidentally saved in a local frame.
    np.testing.assert_allclose(
        result.agent_body_ang_vel_w[:, :, pelvis_index, 2],
        0.4,
        atol=1.0e-10,
    )


def test_file_contract_records_exact_source_and_model_hashes(tmp_path) -> None:
    source = tmp_path / "approved_dual_pull_qpos.npz"
    output = tmp_path / "dual_pull_runtime.npz"
    canonical = _canonical_reference()
    _save_canonical_qpos(source, canonical)
    expected_source_hash = hashlib.sha256(source.read_bytes()).hexdigest()

    result = build_dual_pull_runtime_reference_file(source, output)
    assert result.provenance["source_sha256"] == expected_source_hash
    assert result.provenance["model_sha256"] == sha256_file(DEFAULT_RUBBER_HAND_G1_XML)

    with np.load(output, allow_pickle=False) as saved:
        assert set(saved.files) == {
            "agent_joint_pos",
            "agent_joint_vel",
            "agent_body_pos_w",
            "agent_body_quat_w",
            "agent_body_lin_vel_w",
            "agent_body_ang_vel_w",
            "object_pos_w",
            "object_quat_w",
            "object_lin_vel_w",
            "object_ang_vel_w",
            "fps",
            "joint_names",
            "body_names",
            "provenance",
        }
        provenance = json.loads(saved["provenance"].item())
        assert provenance["source_sha256"] == expected_source_hash
        assert provenance["model_sha256"] == sha256_file(DEFAULT_RUBBER_HAND_G1_XML)
        assert provenance["source_path"] == source.name
        assert not Path(provenance["source_path"]).is_absolute()
        assert provenance["model_path"].endswith("models/g1/g1_29dof.xml")
        assert not Path(provenance["model_path"]).is_absolute()
        assert provenance["quaternion_convention"] == "wxyz"
        assert provenance["velocity_frame"] == "world"
        assert provenance["schema_version"] == 1
        np.testing.assert_array_equal(saved["joint_names"], result.joint_names)
        np.testing.assert_array_equal(saved["body_names"], result.body_names)
        np.testing.assert_allclose(saved["agent_body_quat_w"], result.agent_body_quat_wxyz)
        np.testing.assert_allclose(saved["object_quat_w"], result.object_quat_wxyz)
