from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from holosoma_retargeting.core4d_pair_runtime_reference import (
    DEFAULT_RUBBER_HAND_G1_XML,
    Core4DCompactPairReference,
    build_core4d_pair_runtime_reference,
    build_core4d_pair_runtime_reference_file,
    resample_core4d_pair_reference,
    sha256_file,
)


def _yaw_wxyz(angle: np.ndarray) -> np.ndarray:
    angle = np.asarray(angle, dtype=np.float64)
    result = np.zeros(angle.shape + (4,), dtype=np.float64)
    result[..., 0] = np.cos(0.5 * angle)
    result[..., 3] = np.sin(0.5 * angle)
    return result


def _compact(frames: int = 413, fps: int = 30) -> Core4DCompactPairReference:
    time = np.arange(frames, dtype=np.float64) / fps
    robot_qpos = np.zeros((frames, 2, 36), dtype=np.float64)
    for agent in range(2):
        robot_qpos[:, agent, :3] = np.stack(
            (0.2 * time, np.full(frames, 0.4 * agent), 0.8 + 0.01 * time),
            axis=-1,
        )
        robot_qpos[:, agent, 3:7] = _yaw_wxyz((0.1 + 0.05 * agent) * time)
        robot_qpos[:, agent, 7:] = time[:, None] * (0.01 + 0.001 * agent)
    object_qpos = np.zeros((frames, 7), dtype=np.float64)
    object_qpos[:, :3] = np.stack((0.3 * time, -0.1 * time, 0.4 + 0.02 * time), axis=-1)
    object_qpos[:, 3:7] = _yaw_wxyz(0.2 * time)
    return Core4DCompactPairReference(
        robot_qpos=robot_qpos,
        object_qpos=object_qpos,
        fps=fps,
        object_name="desk001",
        shared_object_scale=0.74,
    )


def _save_compact(path: Path, reference: Core4DCompactPairReference) -> None:
    np.savez_compressed(
        path,
        robot_qpos=reference.robot_qpos,
        object_qpos=reference.object_qpos,
        fps=np.asarray(reference.fps, dtype=np.int64),
        object_name=np.asarray(reference.object_name),
        shared_object_scale=np.asarray(reference.shared_object_scale),
        provenance_json=np.asarray(
            json.dumps({"kind": "diagnostic_preview_not_training_asset"})
        ),
    )


def test_413_frames_at_30_hz_become_uniform_687_frames_at_50_hz() -> None:
    source = _compact()
    result = resample_core4d_pair_reference(source, target_fps=50)

    assert result.robot_qpos.shape == (687, 2, 36)
    assert result.object_qpos.shape == (687, 7)
    assert result.target_frames == 687
    assert result.sampled_duration_seconds == pytest.approx(13.72)
    assert result.source_duration_seconds == pytest.approx(412 / 30)
    assert result.omitted_source_tail_seconds == pytest.approx(1 / 75)
    np.testing.assert_array_equal(result.robot_qpos[0], source.robot_qpos[0])
    np.testing.assert_allclose(result.object_qpos[-1, :3], [4.116, -1.372, 0.6744])
    np.testing.assert_allclose(
        np.linalg.norm(result.robot_qpos[:, :, 3:7], axis=-1),
        1.0,
        atol=1.0e-12,
    )
    assert np.all(np.sum(result.robot_qpos[1:, :, 3:7] * result.robot_qpos[:-1, :, 3:7], axis=-1) > 0.0)


def test_runtime_reuses_fk_and_emits_world_object_angular_velocity() -> None:
    source = _compact(frames=7)
    result = build_core4d_pair_runtime_reference(source, target_fps=50)

    assert result.agent_joint_pos.shape == (11, 2, 29)
    assert result.agent_joint_vel.shape == (11, 2, 29)
    assert result.agent_body_pos_w.shape == (11, 2, 51, 3)
    assert result.object_ang_vel_w.shape == (11, 3)
    expected_object_lin_vel_w = np.tile(np.asarray([0.3, -0.1, 0.02]), (11, 1))
    expected_object_ang_vel_w = np.tile(np.asarray([0.0, 0.0, 0.2]), (11, 1))
    np.testing.assert_allclose(
        result.object_lin_vel_w,
        expected_object_lin_vel_w,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        result.object_ang_vel_w,
        expected_object_ang_vel_w,
        atol=1.0e-10,
    )
    assert result.fps == 50
    assert result.provenance["episode_loop"] is False
    assert result.provenance["terminal_behavior"].startswith("stop_at_last_reference_frame")
    assert "left_rubber_hand_link" in result.body_names


def test_file_contract_records_hashes_timing_and_non_looping(tmp_path: Path) -> None:
    source = tmp_path / "compact.npz"
    source_manifest = tmp_path / "source_manifest.json"
    output = tmp_path / "runtime.npz"
    compact = _compact(frames=7)
    _save_compact(source, compact)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    source_manifest.write_text(json.dumps({"pair_sha256": source_hash}), encoding="utf-8")

    result = build_core4d_pair_runtime_reference_file(
        source,
        output,
        expected_source_sha256=source_hash,
        source_manifest_path=source_manifest,
    )
    assert result.provenance["source_sha256"] == source_hash
    assert result.provenance["source_manifest_sha256"] == sha256_file(source_manifest)
    assert result.provenance["model_sha256"] == sha256_file(DEFAULT_RUBBER_HAND_G1_XML)
    assert result.provenance["episode_loop"] is False

    with np.load(output, allow_pickle=False) as saved:
        assert "object_ang_vel_w" in saved.files
        assert int(saved["fps"]) == 50
        provenance = json.loads(saved["provenance"].item())
        assert provenance["source_frames"] == 7
        assert provenance["target_frames"] == 11
        assert provenance["episode_loop"] is False
        assert provenance["quaternion_interpolation"] == "SLERP"


def test_wrong_hash_or_manifest_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "compact.npz"
    output = tmp_path / "runtime.npz"
    _save_compact(source, _compact(frames=7))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        build_core4d_pair_runtime_reference_file(
            source,
            output,
            expected_source_sha256="0" * 64,
        )

    source_hash = sha256_file(source)
    manifest = tmp_path / "source_manifest.json"
    manifest.write_text(json.dumps({"pair_sha256": "f" * 64}), encoding="utf-8")
    with pytest.raises(ValueError, match="pair_sha256"):
        build_core4d_pair_runtime_reference_file(
            source,
            output,
            expected_source_sha256=source_hash,
            source_manifest_path=manifest,
        )
