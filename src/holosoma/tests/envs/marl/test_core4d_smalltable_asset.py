"""Frozen-data and portable-asset contract for the CORE4D small-table demo."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest


MOTION_DIR = (
    Path(__file__).resolve().parents[3]
    / "holosoma"
    / "data"
    / "motions"
    / "g1_29dof"
    / "whole_body_tracking"
)
CORE4D_DIR = MOTION_DIR / "core4d_smalltable"
COMPACT_REFERENCE = CORE4D_DIR / "20231030_001_desk001_move_compact_fps30.npz"
SOURCE_MANIFEST = CORE4D_DIR / "source_manifest.json"
TRAINING_ASSET_MANIFEST = CORE4D_DIR / "training_asset_manifest.json"
RUNTIME_MANIFEST = CORE4D_DIR / "core4d_pair_runtime_fps50.manifest.json"
TRAINING_URDF = MOTION_DIR / "objects_core4d_desk001_small_training.urdf"
SOURCE_SHA256 = "d9d17e96f5f0b73aa76a1bf32f1ec50cfb7fa7fe78a847dd890882add4bbfd05"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_compact_reference_is_the_reviewed_artifact() -> None:
    assert _sha256(COMPACT_REFERENCE) == SOURCE_SHA256
    with np.load(COMPACT_REFERENCE, allow_pickle=False) as data:
        assert set(data.files) >= {"fps", "robot_qpos", "object_qpos"}
        assert float(np.asarray(data["fps"]).reshape(-1)[0]) == 30.0
        assert data["robot_qpos"].shape == (413, 2, 36)
        assert data["object_qpos"].shape == (413, 7)
        assert np.isfinite(data["robot_qpos"]).all()
        assert np.isfinite(data["object_qpos"]).all()


def test_source_manifest_keeps_preview_and_scale_provenance() -> None:
    manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["pair_sha256"] == SOURCE_SHA256
    assert manifest["shared_object_scale"] == pytest.approx(0.7412283856190455)
    assert manifest["training_ready"] is False


def test_reviewed_promotion_preserves_source_history_and_binds_training_assets() -> None:
    promotion = json.loads(TRAINING_ASSET_MANIFEST.read_text(encoding="utf-8"))
    runtime = json.loads(RUNTIME_MANIFEST.read_text(encoding="utf-8"))

    assert promotion["training_ready"] is True
    assert promotion["source_pair_sha256"] == SOURCE_SHA256
    assert promotion["source_manifest_status_preserved"] == (
        "diagnostic_preview_not_training_asset"
    )
    assert promotion["training_object_urdf_sha256"] == _sha256(TRAINING_URDF)
    assert runtime["training_ready"] is True
    assert runtime["training_promotion_sha256"] == _sha256(TRAINING_ASSET_MANIFEST)


def test_training_urdf_is_portable_and_has_explicit_physics() -> None:
    root = ET.parse(TRAINING_URDF).getroot()
    link = root.find("./link")
    assert link is not None
    assert root.findall(".//mesh") == []
    assert len(link.findall("./visual")) == 5
    assert len(link.findall("./collision")) == 5

    inertial = link.find("./inertial")
    assert inertial is not None
    assert float(inertial.find("./mass").attrib["value"]) == 20.0
    assert inertial.find("./origin").attrib["xyz"] == "0 0.126415000089 0"
    inertia = inertial.find("./inertia").attrib
    diagonal = np.asarray([float(inertia[name]) for name in ("ixx", "iyy", "izz")])
    np.testing.assert_allclose(
        diagonal,
        [1.508353334770, 0.964473631150, 1.070457149412],
        rtol=0.0,
        atol=1.0e-12,
    )
    assert np.all(diagonal > 0.0)
    assert diagonal[0] < diagonal[1] + diagonal[2]
    assert diagonal[1] < diagonal[0] + diagonal[2]
    assert diagonal[2] < diagonal[0] + diagonal[1]


def test_training_urdf_preserves_the_reviewed_five_box_proxy() -> None:
    root = ET.parse(TRAINING_URDF).getroot()
    collisions = {
        element.attrib["name"]: (
            element.find("./origin").attrib["xyz"],
            element.find("./geometry/box").attrib["size"],
        )
        for element in root.findall("./link/collision")
    }
    assert collisions == {
        "desk001_tabletop": (
            "0 0.253129493689 0",
            "0.443699311632 0.0348377341241 0.592241480110",
        ),
        "desk001_vertical_support_negative_z": (
            "0 -0.0037061419281 -0.214956231830",
            "0.0444737031371 0.485504592580 0.0555921289214",
        ),
        "desk001_vertical_support_positive_z": (
            "0 -0.0037061419281 0.214956231830",
            "0.0444737031371 0.485504592580 0.0555921289214",
        ),
        "desk001_bottom_foot_negative_z": (
            "0 -0.252017651110 -0.214956231830",
            "0.407675612090 0.037061419281 0.0741228385619",
        ),
        "desk001_bottom_foot_positive_z": (
            "0 -0.252017651110 0.214956231830",
            "0.407675612090 0.037061419281 0.0741228385619",
        ),
    }
