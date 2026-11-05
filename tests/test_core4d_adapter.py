from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from holosoma_retargeting.data_utils.core4d_adapter import (
    CORE4D_HEIGHT_METHOD,
    CORE4D_TO_OMNI_ROTATION,
    load_canonical_core4d_sequence,
    load_core4d_sequence,
    resample_core4d_pair_sequence,
)


SMPLX_BODY_JOINT_NAMES = [
    "Pelvis",
    "L_Hip",
    "R_Hip",
    "Spine1",
    "L_Knee",
    "R_Knee",
    "Spine2",
    "L_Ankle",
    "R_Ankle",
    "Spine3",
    "L_Foot",
    "R_Foot",
    "Neck",
    "L_Collar",
    "R_Collar",
    "Head",
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow",
    "R_Elbow",
    "L_Wrist",
    "R_Wrist",
]
REPO_ROOT = Path(__file__).resolve().parents[1]
PREPARE_SCRIPT = REPO_ROOT / "scripts" / "prepare_core4d_sequence.py"


def _load_prepare_script():
    spec = importlib.util.spec_from_file_location("prepare_core4d_sequence", PREPARE_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _person_payload(
    frames: int,
    *,
    root_y_angle: float = 0.0,
    raw_vertex_height: float = 1.8,
) -> dict[str, np.ndarray]:
    joints = np.empty((frames, 127, 3), dtype=np.float32)
    joint_index = np.arange(127, dtype=np.float32)
    for frame in range(frames):
        joints[frame, :, 0] = 0.1 * frame + 0.01 * joint_index
        joints[frame, :, 1] = 1.0 + 0.001 * joint_index
        joints[frame, :, 2] = 2.0 + 0.002 * joint_index

    global_orient = np.zeros((frames, 3), dtype=np.float32)
    global_orient[:, 1] = root_y_angle
    vertices = np.zeros((frames, 10475, 3), dtype=np.float32)
    frame_heights = np.linspace(
        0.9 * raw_vertex_height,
        raw_vertex_height,
        frames,
        dtype=np.float32,
    )
    vertices[:, 1, 1] = frame_heights
    # A much larger source-Z extent proves that fallback height uses source Y.
    vertices[:, 2, 2] = 50.0
    return {
        "betas": np.zeros((frames, 10), dtype=np.float32),
        "global_orient": global_orient,
        "transl": np.zeros((frames, 3), dtype=np.float32),
        "body_pose": np.zeros((frames, 21, 3), dtype=np.float32),
        "left_hand_pose": np.zeros((frames, 12), dtype=np.float32),
        "right_hand_pose": np.zeros((frames, 12), dtype=np.float32),
        "joints": joints,
        "vertices": vertices,
    }


def _write_official_person_file(path: Path, payload: dict[str, np.ndarray]) -> None:
    # CORE4D's official files place one Python dictionary under ``arr_0``.
    np.savez(path, arr_0=np.asarray(payload, dtype=object))


def _write_sequence(
    root: Path,
    *,
    person_frames: int = 3,
    object_frames: int | None = None,
    object_rotation: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    root.mkdir(parents=True)
    person1 = _person_payload(
        person_frames,
        root_y_angle=np.pi / 2.0,
        raw_vertex_height=1.8,
    )
    person2 = _person_payload(person_frames, raw_vertex_height=1.65)
    _write_official_person_file(root / "person1_poses.npz", person1)
    _write_official_person_file(root / "person2_poses.npz", person2)

    object_frame_count = person_frames if object_frames is None else object_frames
    object_poses = np.repeat(np.eye(4, dtype=np.float64)[None], object_frame_count, axis=0)
    object_poses[:, :3, 3] = np.column_stack(
        (
            np.arange(object_frame_count, dtype=np.float64),
            np.full(object_frame_count, 2.0),
            np.full(object_frame_count, 3.0),
        )
    )
    if object_rotation is not None:
        object_poses[:, :3, :3] = object_rotation
    np.save(root / "smooth_objposes.npy", object_poses)

    (root / "object_metadata.json").write_text(
        json.dumps({"obj_name": "table001"}),
        encoding="utf-8",
    )
    (root / "aligned_frame_ids.txt").write_text(
        "\n".join(str(frame) for frame in range(person_frames)) + "\n",
        encoding="utf-8",
    )
    mesh_path = root.parent / "object_models" / "table" / "table001_m.obj"
    mesh_path.parent.mkdir(parents=True, exist_ok=True)
    mesh_path.write_text("# synthetic CORE4D object fixture\n", encoding="utf-8")
    return person1, person2, object_poses


def _load_official_fixture(sequence_dir: Path):
    return load_core4d_sequence(
        sequence_dir,
        sequence_dir.parent / "object_models",
    )


def _assert_same_canonical_sequence(first, second) -> None:
    assert first.fps == second.fps
    assert first.height_method == second.height_method
    assert first.object_name == second.object_name
    assert first.object_mesh_path == second.object_mesh_path
    assert first.provenance == second.provenance
    assert list(first.joint_names) == list(second.joint_names)
    np.testing.assert_array_equal(first.human_joints, second.human_joints)
    np.testing.assert_array_equal(first.human_joints_full, second.human_joints_full)
    np.testing.assert_array_equal(first.betas, second.betas)
    np.testing.assert_array_equal(first.human_heights, second.human_heights)
    np.testing.assert_array_equal(first.wrist_quat_xyzw, second.wrist_quat_xyzw)
    np.testing.assert_array_equal(first.object_poses, second.object_poses)
    np.testing.assert_array_equal(first.aligned_frame_ids, second.aligned_frame_ids)


def test_load_core4d_sequence_converts_y_up_to_z_up_and_wrist_orientations(tmp_path: Path) -> None:
    person1, person2, source_object_poses = _write_sequence(tmp_path / "sequence")

    sequence = _load_official_fixture(tmp_path / "sequence")

    expected_axis_conversion = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ]
    )
    np.testing.assert_allclose(CORE4D_TO_OMNI_ROTATION, expected_axis_conversion)
    assert np.linalg.det(CORE4D_TO_OMNI_ROTATION) == pytest.approx(1.0)

    source_joints = np.stack((person1["joints"], person2["joints"]), axis=1)
    expected_full_joints = np.einsum(
        "ij,tpnj->tpni",
        expected_axis_conversion,
        source_joints,
    )
    assert sequence.human_joints.shape == (3, 2, 22, 3)
    assert sequence.human_joints_full.shape == (3, 2, 127, 3)
    assert sequence.betas.shape == (2, 10)
    np.testing.assert_allclose(sequence.human_heights, [1.8, 1.65])
    assert sequence.height_method == CORE4D_HEIGHT_METHOD
    assert sequence.provenance["human_height_source_axis"] == "Y"
    np.testing.assert_allclose(sequence.human_joints_full, expected_full_joints)
    np.testing.assert_allclose(sequence.human_joints, expected_full_joints[:, :, :22])
    assert list(sequence.joint_names) == SMPLX_BODY_JOINT_NAMES

    assert sequence.wrist_quat_xyzw.shape == (3, 2, 2, 4)
    np.testing.assert_allclose(
        np.linalg.norm(sequence.wrist_quat_xyzw, axis=-1),
        1.0,
        atol=1.0e-7,
    )
    source_root_rotation = Rotation.from_rotvec(person1["global_orient"][0]).as_matrix()
    # SMPL-X rotations map a body-local frame into the source world.  Changing
    # only the world basis therefore left-multiplies the rotation; it does not
    # conjugate it as one would for a world-to-world linear operator.
    expected_root_rotation = expected_axis_conversion @ source_root_rotation
    expected_person1_wrist_xyzw = Rotation.from_matrix(expected_root_rotation).as_quat()
    for wrist_index in range(2):
        np.testing.assert_allclose(
            np.abs(np.dot(sequence.wrist_quat_xyzw[0, 0, wrist_index], expected_person1_wrist_xyzw)),
            1.0,
            atol=1.0e-7,
        )
    expected_person2_wrist_xyzw = Rotation.from_matrix(expected_axis_conversion).as_quat()
    for wrist_index in range(2):
        np.testing.assert_allclose(
            np.abs(np.sum(sequence.wrist_quat_xyzw[:, 1, wrist_index] * expected_person2_wrist_xyzw, axis=-1)),
            1.0,
            atol=1.0e-7,
        )

    expected_object_positions = np.einsum(
        "ij,tj->ti",
        expected_axis_conversion,
        source_object_poses[:, :3, 3],
    )
    assert sequence.object_poses.shape == (3, 7)
    expected_object_xyzw = Rotation.from_matrix(expected_axis_conversion).as_quat()
    expected_object_wxyz = expected_object_xyzw[[3, 0, 1, 2]]
    np.testing.assert_allclose(
        sequence.object_poses[:, :4],
        np.broadcast_to(expected_object_wxyz, (3, 4)),
    )
    np.testing.assert_allclose(sequence.object_poses[:, 4:], expected_object_positions)
    assert sequence.fps == 15


def test_load_core4d_sequence_resolves_mixed_case_object_name(tmp_path: Path) -> None:
    sequence_dir = tmp_path / "sequence"
    _write_sequence(sequence_dir)
    (sequence_dir / "object_metadata.json").write_text(
        json.dumps({"obj_name": "Box001"}),
        encoding="utf-8",
    )
    mesh_path = tmp_path / "object_models" / "box" / "box001_m.obj"
    mesh_path.parent.mkdir(parents=True, exist_ok=True)
    mesh_path.write_text("# mixed-case CORE4D object fixture\n", encoding="utf-8")

    sequence = _load_official_fixture(sequence_dir)

    assert sequence.object_name == "Box001"
    assert Path(sequence.object_mesh_path) == mesh_path.resolve()


def test_load_core4d_sequence_rejects_nonpositive_raw_vertex_height(tmp_path: Path) -> None:
    sequence_dir = tmp_path / "sequence"
    _write_sequence(sequence_dir)
    invalid_person = _person_payload(3)
    invalid_person["vertices"].fill(0.0)
    _write_official_person_file(sequence_dir / "person1_poses.npz", invalid_person)

    with pytest.raises(ValueError, match="vertical extent must be positive"):
        _load_official_fixture(sequence_dir)


def test_load_core4d_sequence_rejects_mismatched_frame_counts(tmp_path: Path) -> None:
    sequence_dir = tmp_path / "sequence"
    _write_sequence(sequence_dir, person_frames=3, object_frames=2)

    with pytest.raises(ValueError, match="frame|Frame|length"):
        _load_official_fixture(sequence_dir)


def test_load_core4d_sequence_rejects_invalid_object_rotation(tmp_path: Path) -> None:
    invalid_rotation = np.eye(3)
    invalid_rotation[0, 0] = 2.0
    sequence_dir = tmp_path / "sequence"
    _write_sequence(sequence_dir, object_rotation=invalid_rotation)

    with pytest.raises(ValueError, match=r"rotation|orthonormal|SO\(3\)"):
        _load_official_fixture(sequence_dir)


def test_canonical_core4d_npz_round_trip_is_pickle_free(tmp_path: Path) -> None:
    sequence_dir = tmp_path / "sequence"
    _write_sequence(sequence_dir)
    sequence = _load_official_fixture(sequence_dir)
    output = tmp_path / "canonical_core4d.npz"

    sequence.save(output)

    with np.load(output, allow_pickle=False) as saved:
        required_keys = {
            "human_joints",
            "human_joints_full",
            "betas",
            "human_heights",
            "height_method",
            "wrist_quat_xyzw",
            "object_poses",
            "fps",
            "joint_names",
        }
        assert required_keys.issubset(saved.files)
        assert all(saved[key].dtype != object for key in saved.files)

    reloaded = load_canonical_core4d_sequence(output)
    _assert_same_canonical_sequence(sequence, reloaded)


def test_resample_pair_15_to_30_preserves_duration_and_inputs(tmp_path: Path) -> None:
    sequence_dir = tmp_path / "sequence"
    _write_sequence(sequence_dir)
    template = _load_official_fixture(sequence_dir)
    frames = 197
    source_times = np.arange(frames, dtype=np.float64) / 15.0

    human_joints_full = np.broadcast_to(
        template.human_joints_full[0],
        (frames,) + template.human_joints_full.shape[1:],
    ).copy()
    human_joints_full += source_times[:, None, None, None] * np.asarray(
        [0.3, -0.2, 0.1]
    )
    human_joints = human_joints_full[:, :, :22].copy()

    wrist_xyzw = Rotation.from_euler("z", (0.25 * source_times)[:, None]).as_quat()
    wrist_xyzw = np.broadcast_to(wrist_xyzw[:, None, None, :], (frames, 2, 2, 4)).copy()
    object_xyzw = Rotation.from_euler("z", (-0.4 * source_times)[:, None]).as_quat()
    object_positions = np.column_stack(
        (0.5 * source_times, -0.25 * source_times, 0.4 + 0.05 * source_times)
    )
    object_poses = np.concatenate(
        (object_xyzw[:, [3, 0, 1, 2]], object_positions),
        axis=1,
    )
    source = replace(
        template,
        human_joints=human_joints,
        human_joints_full=human_joints_full,
        wrist_quat_xyzw=wrist_xyzw,
        object_poses=object_poses,
        fps=15,
        aligned_frame_ids=np.arange(frames, dtype=np.int64)[:, None],
    )
    snapshots = {
        "human_joints": source.human_joints.copy(),
        "human_joints_full": source.human_joints_full.copy(),
        "betas": source.betas.copy(),
        "human_heights": source.human_heights.copy(),
        "wrist_quat_xyzw": source.wrist_quat_xyzw.copy(),
        "object_poses": source.object_poses.copy(),
        "aligned_frame_ids": source.aligned_frame_ids.copy(),
        "provenance": json.loads(json.dumps(dict(source.provenance))),
    }

    result = resample_core4d_pair_sequence(source, 30)

    assert len(result.human_joints) == 393
    assert result.fps == 30
    assert (len(result.human_joints) - 1) / result.fps == pytest.approx(
        (len(source.human_joints) - 1) / source.fps
    )
    np.testing.assert_allclose(result.human_joints[::2], source.human_joints)
    np.testing.assert_allclose(result.human_joints_full[::2], source.human_joints_full)
    np.testing.assert_allclose(
        result.human_joints[1],
        0.5 * (source.human_joints[0] + source.human_joints[1]),
    )
    np.testing.assert_allclose(result.object_poses[::2, 4:], source.object_poses[:, 4:])
    np.testing.assert_allclose(
        np.abs(np.sum(result.wrist_quat_xyzw[::2] * source.wrist_quat_xyzw, axis=-1)),
        1.0,
        atol=1.0e-7,
    )
    result_object_xyzw = result.object_poses[:, [1, 2, 3, 0]]
    np.testing.assert_allclose(
        np.abs(np.sum(result_object_xyzw[::2] * object_xyzw, axis=-1)),
        1.0,
        atol=1.0e-7,
    )
    expected_mid_wrist = Rotation.from_euler("z", 0.25 / 30.0).as_quat()
    np.testing.assert_allclose(
        np.abs(np.sum(result.wrist_quat_xyzw[1, 0, 0] * expected_mid_wrist)),
        1.0,
        atol=1.0e-7,
    )
    np.testing.assert_array_equal(result.betas, source.betas)
    np.testing.assert_array_equal(result.human_heights, source.human_heights)
    assert result.height_method == source.height_method
    assert result.object_name == source.object_name
    assert result.object_mesh_path == source.object_mesh_path
    np.testing.assert_array_equal(result.joint_names, source.joint_names)
    assert result.provenance == source.provenance
    assert not np.shares_memory(result.betas, source.betas)
    assert not np.shares_memory(result.human_heights, source.human_heights)

    for name, expected in snapshots.items():
        if name == "provenance":
            assert source.provenance == expected
        else:
            np.testing.assert_array_equal(getattr(source, name), expected)


def test_prepare_cli_writes_and_verifies_one_safe_artifact(tmp_path: Path, capsys) -> None:
    sequence_dir = tmp_path / "sequence"
    _write_sequence(sequence_dir)
    output = tmp_path / "prepared.npz"
    module = _load_prepare_script()

    assert module.main(
        [
            "--sequence-dir",
            str(sequence_dir),
            "--object-model-root",
            str(tmp_path / "object_models"),
            "--output",
            str(output),
            "--output-fps",
            "30",
        ]
    ) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["pickle_free_output_verified"] is True
    assert report["source_frames"] == 3
    assert report["source_fps"] == 15
    assert report["frames"] == 5
    assert report["fps"] == 30
    assert report["resampled"] is True
    assert report["duration_seconds"] == pytest.approx(2.0 / 15.0)
    assert report["human_joints_shape"] == [5, 2, 22, 3]
    assert report["betas_shape"] == [2, 10]
    assert report["human_heights"] == pytest.approx([1.8, 1.65])
    assert report["height_method"] == CORE4D_HEIGHT_METHOD
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        module.main(
            [
                "--sequence-dir",
                str(sequence_dir),
                "--object-model-root",
                str(tmp_path / "object_models"),
                "--output",
                str(output),
            ]
        )
