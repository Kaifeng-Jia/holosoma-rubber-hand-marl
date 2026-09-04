from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from holosoma_retargeting.config_types.data_type import SMPLX_DEMO_JOINTS
from holosoma_retargeting.data_utils.core4d_adapter import (
    CORE4D_HEIGHT_METHOD,
    Core4DPairSequence,
)
from holosoma_retargeting.data_utils.core4d_retarget import (
    CORE4D_A1_PALM_LANDMARKS,
    G1_HEIGHT_M,
    build_core4d_a1_palm_orientation_targets,
    prepare_core4d_person_retarget_input,
    require_shared_physical_object_qpos,
    save_core4d_pair_reference,
)


def test_build_core4d_a1_targets_uses_official_smplx_palm_landmarks() -> None:
    pytest.importorskip("smplx")
    assert CORE4D_A1_PALM_LANDMARKS == (
        ("L_Wrist", 20),
        ("L_Index1", 25),
        ("L_Middle1", 28),
        ("L_Pinky1", 31),
        ("R_Wrist", 21),
        ("R_Index1", 40),
        ("R_Middle1", 43),
        ("R_Pinky1", 46),
    )
    frames = 2
    full_joints = np.zeros((frames, 127, 3), dtype=np.float64)
    left_points = {
        20: np.asarray([0.0, 0.0, 0.0]),
        25: np.asarray([0.0, 1.0, 0.0]),
        28: np.asarray([1.0, 0.0, 0.0]),
        31: np.asarray([0.0, -1.0, 0.0]),
    }
    right_points = {
        21: np.asarray([0.0, 0.0, 0.0]),
        40: np.asarray([0.0, -1.0, 0.0]),
        43: np.asarray([1.0, 0.0, 0.0]),
        46: np.asarray([0.0, 1.0, 0.0]),
    }
    quarter_turn = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    for index, point in {**left_points, **right_points}.items():
        full_joints[0, index] = point
        full_joints[1, index] = quarter_turn @ point
    wrist_quat_xyzw = np.zeros((frames, 2, 4), dtype=np.float64)
    wrist_quat_xyzw[0, :, 3] = 1.0
    wrist_quat_xyzw[1, :, 2:] = np.sqrt(0.5)

    targets, calibration_errors = build_core4d_a1_palm_orientation_targets(
        full_joints,
        wrist_quat_xyzw,
    )

    np.testing.assert_allclose(
        targets[0],
        np.repeat(np.eye(3)[None], 2, axis=0),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        targets[1],
        np.repeat(quarter_turn[None], 2, axis=0),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(calibration_errors, 0.0, atol=1.0e-12)


def _pair_sequence(frames: int = 4) -> Core4DPairSequence:
    body_joint_count = len(SMPLX_DEMO_JOINTS)
    human_joints = np.empty((frames, 2, body_joint_count, 3), dtype=np.float64)
    for frame in range(frames):
        for person in range(2):
            for joint in range(body_joint_count):
                human_joints[frame, person, joint] = [
                    2.0 + 0.05 * frame + 0.1 * person + 0.01 * joint,
                    -3.0 - 0.02 * frame + 0.05 * person - 0.005 * joint,
                    0.7 + 0.01 * joint,
                ]

    foot_indices = [
        SMPLX_DEMO_JOINTS.index("L_Foot"),
        SMPLX_DEMO_JOINTS.index("R_Foot"),
    ]
    human_joints[:, 0, foot_indices, 2] = 0.02
    human_joints[:, 1, foot_indices, 2] = 0.03

    human_joints_full = np.zeros((frames, 2, 127, 3), dtype=np.float64)
    human_joints_full[:, :, :body_joint_count] = human_joints
    human_joints_full[:, :, body_joint_count:, 0] = 2.4
    human_joints_full[:, :, body_joint_count:, 1] = -2.7
    human_joints_full[:, :, body_joint_count:, 2] = 1.0

    wrist_quat_xyzw = np.zeros((frames, 2, 2, 4), dtype=np.float64)
    wrist_quat_xyzw[..., 3] = 1.0
    object_poses = np.zeros((frames, 7), dtype=np.float64)
    object_poses[:, 0] = 1.0
    object_poses[:, 4:] = np.column_stack(
        (
            2.0 + 0.1 * np.arange(frames),
            -3.0 + 0.04 * np.arange(frames),
            0.4 + 0.01 * np.arange(frames),
        )
    )

    return Core4DPairSequence(
        human_joints=human_joints,
        human_joints_full=human_joints_full,
        betas=np.asarray([[0.0] * 10, [0.1] * 10], dtype=np.float64),
        human_heights=np.asarray([1.8, 1.65], dtype=np.float64),
        height_method=CORE4D_HEIGHT_METHOD,
        wrist_quat_xyzw=wrist_quat_xyzw,
        object_poses=object_poses,
        fps=30,
        joint_names=np.asarray(SMPLX_DEMO_JOINTS),
        aligned_frame_ids=np.arange(frames, dtype=np.int64)[:, None],
        object_name="desk001",
        object_mesh_path="/fixtures/desk/desk001_m.obj",
        provenance={
            "source_sequence": "fixture/sequence",
            "person_order": ["person1", "person2"],
        },
    )


def _physical_object_qpos(sequence: Core4DPairSequence) -> np.ndarray:
    return sequence.object_poses[:, [4, 5, 6, 0, 1, 2, 3]]


def _qpos_pair(
    sequence: Core4DPairSequence,
    *,
    robot_qpos_width: int = 36,
) -> list[np.ndarray]:
    shared_object_qpos = _physical_object_qpos(sequence)
    frames = len(sequence.object_poses)
    person1_robot = np.arange(frames * robot_qpos_width, dtype=np.float64).reshape(
        frames,
        robot_qpos_width,
    )
    person2_robot = -person1_robot - 1.0
    return [
        np.concatenate((person1_robot, shared_object_qpos), axis=1),
        np.concatenate((person2_robot, shared_object_qpos), axis=1),
    ]


def test_prepare_person_inputs_use_individual_scales_shared_anchor_and_physical_object() -> None:
    sequence = _pair_sequence()
    array_fields = (
        "human_joints",
        "human_joints_full",
        "betas",
        "human_heights",
        "wrist_quat_xyzw",
        "object_poses",
        "joint_names",
        "aligned_frame_ids",
    )
    snapshots = {name: np.asarray(getattr(sequence, name)).copy() for name in array_fields}
    provenance_snapshot = json.loads(json.dumps(dict(sequence.provenance)))

    person_inputs = [
        prepare_core4d_person_retarget_input(sequence, person_index)
        for person_index in range(2)
    ]

    expected_scales = G1_HEIGHT_M / sequence.human_heights
    np.testing.assert_allclose(
        [value.human_to_robot_scale for value in person_inputs],
        expected_scales,
    )
    assert person_inputs[0].human_to_robot_scale != person_inputs[1].human_to_robot_scale
    expected_anchor = np.asarray(
        [sequence.object_poses[0, 4], sequence.object_poses[0, 5], 0.0]
    )
    for person_index, value in enumerate(person_inputs):
        np.testing.assert_array_equal(value.scale_anchor, expected_anchor)
        np.testing.assert_array_equal(value.physical_object_poses, sequence.object_poses)
        assert not np.shares_memory(value.physical_object_poses, sequence.object_poses)
        assert value.human_height_m == sequence.human_heights[person_index]
        assert value.human_height_method == CORE4D_HEIGHT_METHOD

        source_body = sequence.human_joints[:, person_index].copy()
        source_body[..., 2] -= value.ground_offset_m
        expected_human_joints = expected_anchor + value.human_to_robot_scale * (
            source_body - expected_anchor
        )
        np.testing.assert_allclose(value.human_joints, expected_human_joints)

        expected_nominal_positions = expected_anchor + value.human_to_robot_scale * (
            sequence.object_poses[:, 4:] - expected_anchor
        )
        np.testing.assert_allclose(
            value.nominal_object_poses[:, 4:],
            expected_nominal_positions,
        )
        np.testing.assert_array_equal(
            value.nominal_object_poses[:, :4],
            sequence.object_poses[:, :4],
        )

    np.testing.assert_array_equal(
        person_inputs[0].physical_object_poses,
        person_inputs[1].physical_object_poses,
    )
    for name, snapshot in snapshots.items():
        np.testing.assert_array_equal(getattr(sequence, name), snapshot)
    assert sequence.provenance == provenance_snapshot


def test_require_shared_physical_object_qpos_is_exact_and_does_not_mutate_inputs() -> None:
    sequence = _pair_sequence()
    qpos_sequences = _qpos_pair(sequence)
    snapshots = [value.copy() for value in qpos_sequences]

    shared = require_shared_physical_object_qpos(qpos_sequences, sequence.object_poses)

    np.testing.assert_array_equal(shared, _physical_object_qpos(sequence))
    assert not np.shares_memory(shared, qpos_sequences[0])
    for value, snapshot in zip(qpos_sequences, snapshots, strict=True):
        np.testing.assert_array_equal(value, snapshot)

    divergent = [value.copy() for value in qpos_sequences]
    divergent[1][1, -7] += 1.0e-12
    with pytest.raises(ValueError, match="diverged from the shared physical trajectory"):
        require_shared_physical_object_qpos(divergent, sequence.object_poses)


def test_save_core4d_pair_reference_has_pickle_free_schema_and_shapes(
    tmp_path: Path,
) -> None:
    sequence = _pair_sequence()
    person_inputs = [
        prepare_core4d_person_retarget_input(sequence, person_index)
        for person_index in range(2)
    ]
    qpos_sequences = _qpos_pair(sequence)
    output = tmp_path / "paired_reference.npz"

    result = save_core4d_pair_reference(
        output,
        qpos_sequences=qpos_sequences,
        sequence=sequence,
        person_inputs=person_inputs,
    )

    assert result == output.resolve()
    with np.load(result, allow_pickle=False) as archive:
        assert set(archive.files) == {
            "robot_qpos",
            "object_qpos",
            "fps",
            "human_heights",
            "human_to_robot_scales",
            "scale_anchor",
            "object_name",
            "object_mesh_path",
            "provenance_json",
        }
        assert all(archive[name].dtype != object for name in archive.files)
        assert archive["robot_qpos"].shape == (4, 2, 36)
        assert archive["object_qpos"].shape == (4, 7)
        np.testing.assert_array_equal(
            archive["robot_qpos"][:, 0],
            qpos_sequences[0][:, :-7],
        )
        np.testing.assert_array_equal(
            archive["robot_qpos"][:, 1],
            qpos_sequences[1][:, :-7],
        )
        np.testing.assert_array_equal(
            archive["object_qpos"],
            _physical_object_qpos(sequence),
        )
        np.testing.assert_allclose(archive["human_heights"], [1.8, 1.65])
        np.testing.assert_allclose(
            archive["human_to_robot_scales"],
            G1_HEIGHT_M / sequence.human_heights,
        )
        np.testing.assert_array_equal(archive["scale_anchor"], [2.0, -3.0, 0.0])
        assert int(archive["fps"].reshape(())) == 30
        assert str(archive["object_name"].reshape(())) == "desk001"
        assert str(archive["object_mesh_path"].reshape(())) == sequence.object_mesh_path
        provenance = json.loads(str(archive["provenance_json"].reshape(())))
        assert provenance == {
            "height_methods": [CORE4D_HEIGHT_METHOD, CORE4D_HEIGHT_METHOD],
            "object_scale": 1.0,
            "person_order": ["person1", "person2"],
            "scale_anchor_policy": "initial_object_origin_projected_to_ground",
            "shared_object_qpos_exact": True,
            "source_sequence": "fixture/sequence",
        }
