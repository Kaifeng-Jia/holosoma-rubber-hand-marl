from __future__ import annotations

import importlib.util
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "export_core4d_small_table_pair.py"
)


def _load_script():
    spec = importlib.util.spec_from_file_location("export_core4d_small_table_pair", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_source_run(path: Path, *, method: str = "two-stage") -> tuple[np.ndarray, np.ndarray]:
    frames = 3
    path.mkdir()
    (path / "manifest.json").write_text(json.dumps({"method": method}), encoding="utf-8")
    (path / "person1").mkdir()
    (path / "person2").mkdir()
    (path / "assets").mkdir()

    qpos1 = np.zeros((frames, 43), dtype=np.float64)
    qpos2 = np.zeros((frames, 43), dtype=np.float64)
    qpos1[:, 3] = 1.0
    qpos2[:, 3] = 1.0
    qpos1[:, 36:39] = np.asarray([[0.0, 0.0, 0.4], [0.1, 0.0, 0.4], [0.2, 0.0, 0.4]])
    qpos2[:, 36:39] = qpos1[:, 36:39] + np.asarray([0.02, -0.04, 0.06])
    qpos1[:, 39] = 1.0
    qpos2[:, 39] = 1.0
    for person, qpos in ((1, qpos1), (2, qpos2)):
        np.savez_compressed(
            path / f"person{person}" / f"person{person}_nominal_scaled.npz",
            qpos=qpos,
            fps=np.asarray(30, dtype=np.int64),
        )

    source_robot_qpos = np.zeros((frames, 2, 36), dtype=np.float64)
    source_robot_qpos[..., 3] = 1.0
    source_object_qpos = np.zeros((frames, 7), dtype=np.float64)
    source_object_qpos[:, 3] = 1.0
    np.savez_compressed(
        path / "core4d_pair_reference.npz",
        robot_qpos=source_robot_qpos,
        object_qpos=source_object_qpos,
        fps=np.asarray(30, dtype=np.int64),
        human_heights=np.asarray([1.76, 1.80]),
        human_to_robot_scales=np.asarray([0.75, 0.73]),
        scale_anchor=np.asarray([0.5, 0.3, 0.0]),
        object_name=np.asarray("desk001"),
        object_mesh_path=np.asarray("/fixtures/desk001_m.obj"),
    )
    (path / "assets" / "desk001.urdf").write_text(
        """\
<robot name="desk001">
  <link name="desk001_link">
    <inertial>
      <origin xyz="0 0.2 0"/>
      <mass value="1"/>
      <inertia ixx="2" ixy="0" ixz="0" iyy="3" iyz="0" izz="4"/>
    </inertial>
    <visual><geometry><mesh filename="desk001_m.obj" scale="1 1 1"/></geometry></visual>
    <collision><origin xyz="0 0.4 0"/><geometry><box size="0.6 0.05 0.8"/></geometry></collision>
  </link>
</robot>
""",
        encoding="utf-8",
    )
    return qpos1, qpos2


def test_export_reproduces_shared_small_table_contract(tmp_path: Path) -> None:
    module = _load_script()
    source = tmp_path / "source"
    output = tmp_path / "output"
    qpos1, qpos2 = _write_source_run(source)

    pair_path = module.export_small_table_pair(source, output)

    with np.load(pair_path, allow_pickle=False) as archive:
        assert archive["robot_qpos"].shape == (3, 2, 36)
        np.testing.assert_array_equal(archive["robot_qpos"][:, 0], qpos1[:, :36])
        np.testing.assert_array_equal(archive["robot_qpos"][:, 1], qpos2[:, :36])
        np.testing.assert_allclose(
            archive["object_qpos"][:, :3],
            0.5 * (qpos1[:, 36:39] + qpos2[:, 36:39]),
        )
        np.testing.assert_array_equal(archive["object_qpos"][:, 3:], qpos1[:, 39:43])
        assert float(archive["shared_object_scale"]) == pytest.approx(0.74)
        assert archive["provenance_json"].dtype.kind == "U"

    root = ET.parse(output / "assets" / "desk001.urdf").getroot()
    assert root.find(".//mass").get("value") == "1"
    assert root.find(".//visual/geometry/mesh").get("scale") == "0.74 0.74 0.74"
    assert root.find(".//collision/origin").get("xyz") == "0 0.296 0"
    assert root.find(".//box").get("size") == "0.444 0.037 0.592"
    assert float(root.find(".//inertia").get("ixx")) == pytest.approx(2.0 * 0.74**2)

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["training_ready"] is False
    assert manifest["shared_object_scale"] == pytest.approx(0.74)
    assert manifest["maximum_nominal_object_translation_disagreement_m"] == pytest.approx(
        np.linalg.norm([0.02, -0.04, 0.06])
    )
    assert manifest["pair_sha256"]
    assert manifest["object_urdf_sha256"]


def test_export_rejects_different_nominal_object_rotations(tmp_path: Path) -> None:
    module = _load_script()
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_source_run(source)
    second = source / "person2" / "person2_nominal_scaled.npz"
    with np.load(second, allow_pickle=False) as archive:
        qpos = np.asarray(archive["qpos"]).copy()
        fps = np.asarray(archive["fps"]).copy()
    qpos[1, 39:43] = np.asarray([0.0, 0.0, 0.0, 1.0])
    np.savez_compressed(second, qpos=qpos, fps=fps)

    with pytest.raises(ValueError, match="nominal object rotations differ"):
        module.export_small_table_pair(source, output)
    assert not output.exists()


def test_export_requires_two_stage_source_and_empty_output(tmp_path: Path) -> None:
    module = _load_script()
    omni_source = tmp_path / "omni"
    _write_source_run(omni_source, method="omni")
    with pytest.raises(ValueError, match="--method two-stage"):
        module.export_small_table_pair(omni_source, tmp_path / "unused")

    source = tmp_path / "source"
    _write_source_run(source)
    output = tmp_path / "output"
    output.mkdir()
    (output / "keep.txt").write_text("do not overwrite", encoding="utf-8")
    with pytest.raises(FileExistsError, match="not empty"):
        module.export_small_table_pair(source, output)
