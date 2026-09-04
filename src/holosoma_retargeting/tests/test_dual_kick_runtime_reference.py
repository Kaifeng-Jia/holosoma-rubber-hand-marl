from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from holosoma_retargeting.dual_kick_runtime_reference import (
    ACCEPTED_MIRRORED_KICK_VISER_SHA256,
    DEFAULT_RUBBER_HAND_G1_XML,
    build_dual_kick_runtime_reference_file,
    sha256_file,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
ACCEPTED_SOURCE = (
    REPO_ROOT
    / "src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/"
    "rubber_hand_largetable_v1/kick/plan5_attempt09_dual_kick_mirrored_viser.npz"
)


def test_accepted_source_hash_and_full_runtime_contract(tmp_path) -> None:
    assert sha256_file(ACCEPTED_SOURCE) == ACCEPTED_MIRRORED_KICK_VISER_SHA256
    output = tmp_path / "kick_runtime.npz"
    result = build_dual_kick_runtime_reference_file(ACCEPTED_SOURCE, output)

    assert result.agent_joint_pos.shape == (298, 2, 29)
    assert result.agent_joint_vel.shape == (298, 2, 29)
    assert result.agent_body_pos_w.shape == (298, 2, 51, 3)
    assert result.agent_body_quat_wxyz.shape == (298, 2, 51, 4)
    assert result.object_pos_w.shape == (298, 3)
    assert result.object_quat_wxyz.shape == (298, 4)
    assert result.fps == 50
    assert "left_rubber_hand_link" in result.body_names
    assert "right_rubber_hand_link" in result.body_names
    assert result.provenance["generator"] == "holosoma_retargeting.dual_kick_runtime_reference"
    assert result.provenance["source_layout"] == "mirrored"
    assert result.provenance["source_sha256"] == ACCEPTED_MIRRORED_KICK_VISER_SHA256
    assert "lateral_leg_offset_m" not in result.provenance
    assert not any("pull" in str(value).lower() for value in result.provenance.values())

    with np.load(ACCEPTED_SOURCE, allow_pickle=False) as source, np.load(
        output, allow_pickle=False
    ) as saved:
        np.testing.assert_allclose(saved["agent_joint_pos"], source["dof_pos"])
        np.testing.assert_allclose(saved["object_pos_w"], source["object_pos_w"])
        np.testing.assert_allclose(
            saved["object_quat_w"], source["object_quat_xyzw"][:, [3, 0, 1, 2]]
        )
        provenance = json.loads(saved["provenance"].item())
        assert provenance["model_sha256"] == sha256_file(DEFAULT_RUBBER_HAND_G1_XML)
        assert set(saved.files) == {
            "agent_joint_pos", "agent_joint_vel", "agent_body_pos_w",
            "agent_body_quat_w", "agent_body_lin_vel_w", "agent_body_ang_vel_w",
            "object_pos_w", "object_quat_w", "object_lin_vel_w", "object_ang_vel_w",
            "fps", "joint_names", "body_names", "provenance",
        }


def test_tampered_or_wrong_layout_source_is_rejected(tmp_path) -> None:
    tampered = tmp_path / "tampered.npz"
    with np.load(ACCEPTED_SOURCE, allow_pickle=False) as source:
        payload = {name: np.asarray(source[name]).copy() for name in source.files}
    payload["root_pos"][0, 0, 0] += 1.0e-6
    np.savez_compressed(tampered, **payload)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        build_dual_kick_runtime_reference_file(tampered, tmp_path / "runtime.npz")


def test_byte_identical_copy_is_accepted(tmp_path) -> None:
    copied = tmp_path / "accepted_copy.npz"
    shutil.copyfile(ACCEPTED_SOURCE, copied)
    assert hashlib.sha256(copied.read_bytes()).hexdigest() == ACCEPTED_MIRRORED_KICK_VISER_SHA256
    result = build_dual_kick_runtime_reference_file(copied, tmp_path / "runtime.npz")
    assert result.provenance["source_path"] == copied.name
