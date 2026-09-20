"""Raw display-only CORE4D preview tests; no viewer, physics or training starts."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from holosoma_retargeting.data_utils.core4d_adapter import (
    load_core4d_sequence,
)


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "visualize_core4d_source.py"


@pytest.fixture(scope="module")
def viewer():
    spec = importlib.util.spec_from_file_location("visualize_core4d_source_tested", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _person(frames: int, shift: float = 0.0) -> dict[str, np.ndarray]:
    """Small genuine official pickle fixture (two 10475-vertex frames by default)."""
    joints = np.zeros((frames, 127, 3), dtype=np.float32)
    joints[..., 0] = np.arange(127, dtype=np.float32)[None] * .01 + shift
    joints[..., 1] = 1.0
    vertices = np.zeros((frames, 10475, 3), dtype=np.float32)
    vertices[:, 1, 1] = 1.7
    return {
        "betas": np.zeros((frames, 10), dtype=np.float32),
        "joints": joints,
        "vertices": vertices,
        "global_orient": np.zeros((frames, 3), dtype=np.float32),
        "body_pose": np.zeros((frames, 21, 3), dtype=np.float32),
    }


def _fixture(
    tmp_path: Path,
    *,
    standard: bool = True,
    transforms: np.ndarray | None = None,
    person_frames: tuple[int, int] = (2, 2),
) -> tuple[Path, Path, np.ndarray]:
    root = tmp_path / "CORE4D_Real"
    sequence = root / "human_object_motions" / "20231002" / "035" if standard else tmp_path / "custom_sequence"
    sequence.mkdir(parents=True)
    model_root = root / "object_models"
    mesh_path = model_root / "board" / "board020_m.obj"
    mesh_path.parent.mkdir(parents=True)
    mesh_path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    for index, count in enumerate(person_frames, start=1):
        np.savez(sequence / f"person{index}_poses.npz", arr_0=np.asarray(_person(count, .2 * index), dtype=object))
    if transforms is None:
        transforms = np.repeat(np.eye(4)[None], 2, axis=0)
        transforms[:, :3, 3] = [[1, 2, 3], [2, 3, 4]]
    np.save(sequence / "smooth_objposes.npy", transforms)
    (sequence / "object_metadata.json").write_text(json.dumps({"obj_name": "Board020"}), encoding="utf-8")
    (sequence / "aligned_frame_ids.txt").write_text("0\n1\n", encoding="utf-8")
    return sequence, model_root, transforms


@pytest.mark.parametrize("argv", [
    [],
    ["--input", "a.npz", "--sequence-dir", "/raw"],
    ["--input", "a.npz", "--object-model-root", "/models"],
])
def test_cli_rejects_missing_or_ambiguous_modes(viewer, argv):
    with pytest.raises(SystemExit) as error:
        viewer.parse_args(argv)
    assert error.value.code == 2


def test_cli_accepts_distinct_raw_and_canonical_modes(viewer):
    raw = viewer.parse_args(["--sequence-dir", "/raw", "--validate-only"])
    assert raw.input is None and raw.sequence_dir == Path("/raw")
    assert raw.validate_only
    canonical = viewer.parse_args(["--input", "/reference.npz"])
    assert canonical.sequence_dir is None and canonical.input == Path("/reference.npz")


def test_raw_loader_infers_models_and_preserves_identity_and_fps(viewer, tmp_path):
    sequence_dir, model_root, transforms = _fixture(tmp_path)
    files_before = {path.relative_to(tmp_path) for path in tmp_path.rglob("*") if path.is_file()}
    loaded = viewer.load_raw_core4d_sequence(sequence_dir)
    assert loaded.fps == 15
    assert loaded.human_joints.shape == (2, 2, 22, 3)
    assert loaded.human_joints_full.shape == (2, 2, 127, 3)
    assert loaded.wrist_quat_xyzw.shape == (2, 2, 2, 4)
    assert loaded.object_name == "Board020"
    assert Path(loaded.object_mesh_path) == model_root / "board" / "board020_m.obj"
    np.testing.assert_array_equal(loaded.raw_object_transforms, transforms)
    # Original source Y=1 maps to target Z=1, not a second coordinate rotation.
    np.testing.assert_allclose(loaded.human_joints[..., 2], 1.0)
    assert files_before == {path.relative_to(tmp_path) for path in tmp_path.rglob("*") if path.is_file()}
    assert not hasattr(loaded, "save")
    assert not hasattr(loaded, "object_poses")


def test_nonstandard_directory_requires_explicit_mesh_root(viewer, tmp_path):
    sequence_dir, model_root, _ = _fixture(tmp_path, standard=False)
    with pytest.raises(ValueError, match="Cannot infer object models"):
        viewer.load_raw_core4d_sequence(sequence_dir)
    assert viewer.load_raw_core4d_sequence(sequence_dir, model_root).object_name == "Board020"


@pytest.mark.parametrize("mutation, message", [
    ("wrong_shape", "shape"),
    ("empty", "shape"),
    ("nan", "non-finite"),
    ("inf", "non-finite"),
    ("bottom_row", "homogeneous"),
])
def test_raw_loader_keeps_necessary_matrix_checks(viewer, tmp_path, mutation, message):
    transforms = np.repeat(np.eye(4)[None], 2, axis=0)
    if mutation == "wrong_shape":
        transforms = transforms[:, :3, :3]
    elif mutation == "empty":
        transforms = transforms[:0]
    elif mutation == "nan":
        transforms[0, 0, 0] = np.nan
    elif mutation == "inf":
        transforms[0, 0, 3] = np.inf
    elif mutation == "bottom_row":
        transforms[1, 3, 0] = .01
    sequence_dir, _, _ = _fixture(tmp_path, transforms=transforms)
    with pytest.raises(ValueError, match=message):
        viewer.load_raw_core4d_sequence(sequence_dir)


@pytest.mark.parametrize("person_frames", [(1, 2), (2, 1)])
def test_raw_loader_rejects_person_object_frame_mismatch(viewer, tmp_path, person_frames):
    sequence_dir, _, _ = _fixture(tmp_path, person_frames=person_frames)
    with pytest.raises(ValueError, match="identical frame counts"):
        viewer.load_raw_core4d_sequence(sequence_dir)


def test_raw_affine_vertices_are_not_repaired_or_double_transformed(viewer, tmp_path):
    transforms = np.repeat(np.eye(4)[None], 2, axis=0)
    transforms[0, :3, :3] = [[1.03, .2, 0], [0, .98, .1], [0, 0, 1.01]]
    transforms[1, :3, :3] = [[1, 0, .3], [.1, 1, 0], [0, .1, 1]]
    transforms[:, :3, 3] = [[1, 2, 3], [-2, 1, .5]]
    sequence_dir, model_root, _ = _fixture(tmp_path, transforms=transforms)
    original_bytes = (sequence_dir / "smooth_objposes.npy").read_bytes()
    loaded = viewer.load_raw_core4d_sequence(sequence_dir)
    mesh = viewer.load_object_mesh(loaded.object_mesh_path)
    for frame in range(2):
        actual = viewer.object_vertices_world(loaded, mesh, frame)
        expected_raw = mesh.vertices @ transforms[frame, :3, :3].T + transforms[frame, :3, 3]
        expected_z_up = expected_raw[:, [0, 2, 1]].copy()
        expected_z_up[:, 1] *= -1
        np.testing.assert_allclose(actual, expected_z_up, atol=1e-12)
    np.testing.assert_array_equal(loaded.raw_object_transforms, transforms)
    assert (sequence_dir / "smooth_objposes.npy").read_bytes() == original_bytes
    # Preview acceptance must not silently relax the training adapter.
    with pytest.raises(ValueError, match="invalid rotation"):
        load_core4d_sequence(sequence_dir, model_root)
    diagnostics = viewer.raw_rotation_diagnostics(transforms)
    assert diagnostics["non_rigid_frame_indices"] == [0, 1]
    assert diagnostics["max_rotation_orthogonality_error"] > 0


def test_legal_rigid_raw_matches_existing_canonical_world_vertices(viewer, tmp_path):
    transforms = np.repeat(np.eye(4)[None], 2, axis=0)
    transforms[:, :3, :3] = Rotation.from_euler("xyz", [[.2, .4, -.3], [-.1, .3, .5]]).as_matrix()
    transforms[:, :3, 3] = [[1, 2, 3], [-2, 1, .5]]
    sequence_dir, model_root, _ = _fixture(tmp_path, transforms=transforms)
    raw = viewer.load_raw_core4d_sequence(sequence_dir)
    canonical = load_core4d_sequence(sequence_dir, model_root)
    mesh = viewer.load_object_mesh(raw.object_mesh_path)
    np.testing.assert_allclose(raw.human_joints, canonical.human_joints)
    np.testing.assert_allclose(raw.wrist_quat_xyzw, canonical.wrist_quat_xyzw)
    for frame in range(2):
        pose = canonical.object_poses[frame]
        rotation = Rotation.from_quat(pose[:4][[1, 2, 3, 0]]).as_matrix()
        canonical_vertices = mesh.vertices @ rotation.T + pose[4:]
        np.testing.assert_allclose(viewer.object_vertices_world(raw, mesh, frame), canonical_vertices, atol=1e-12)
    assert viewer.raw_rotation_diagnostics(transforms)["non_rigid_frame_indices"] == []
    assert viewer.validation_summary(canonical, mesh)["pickle_free_input_verified"] is True


def test_raw_summary_is_display_only_not_pickle_free_training_asset(viewer, tmp_path):
    sequence_dir, _, _ = _fixture(tmp_path)
    raw = viewer.load_raw_core4d_sequence(sequence_dir)
    summary = viewer.validation_summary(raw, viewer.load_object_mesh(raw.object_mesh_path))
    assert summary["input_mode"] == "raw_preview"
    assert summary["pickle_free_input_verified"] is False
    assert summary["preview_only"] is True
    assert summary["training_asset_exported"] is False
    assert summary["raw_object_transforms_shape"] == [2, 4, 4]
    assert "object_poses_shape" not in summary


def test_validate_only_never_starts_server(viewer, tmp_path, monkeypatch, capsys):
    sequence_dir, _, _ = _fixture(tmp_path)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT_PATH), "--sequence-dir", str(sequence_dir), "--validate-only"])
    def refuse_server(*args, **kwargs):
        raise AssertionError("validate-only must not create a viewer or server")
    monkeypatch.setattr(viewer, "run_viewer", refuse_server)
    monkeypatch.setattr(viewer.viser, "ViserServer", refuse_server)
    viewer.main()
    output = json.loads(capsys.readouterr().out)
    assert output["input_mode"] == "raw_preview"
    assert output["frames"] == 2
