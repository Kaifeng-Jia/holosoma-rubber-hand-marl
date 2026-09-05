from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "synthesize_plan5_kick_reference.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("synthesize_plan5_kick_reference", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class _FakeKickReference:
    robot_a_qpos: np.ndarray = field(
        default_factory=lambda: np.tile(
            np.asarray([0.0, 0.0, -0.6742736, 1.0, 0.0, 0.0, 0.0] + [0.0] * 29),
            (3, 1),
        )
    )
    robot_b_qpos: np.ndarray = field(
        default_factory=lambda: np.tile(
            np.asarray([0.0, 0.0, 0.6742736, 1.0, 0.0, 0.0, 0.0] + [0.0] * 29),
            (3, 1),
        )
    )
    shared_object_qpos: np.ndarray = field(
        default_factory=lambda: np.asarray(
            [
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.2, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ]
        )
    )
    fps: int = 50
    layout: str = "mirrored"
    source_attempt: int = 9
    source_contact_local_z_m: float = -0.23525
    target_contact_half_span_m: float = 0.6742736
    robot_a_lateral_offset_m: float = -0.4390236
    robot_b_lateral_offset_m: float = 0.4390236
    provenance: dict[str, object] = field(
        default_factory=lambda: {
            "source_attempt_start": 2418,
            "source_attempt_end_exclusive": 2716,
            "source_attempt_length": 298,
            "source_attempt_completed": True,
        }
    )

    def save(self, path: Path) -> Path:
        np.savez_compressed(
            path,
            robot_a_qpos=self.robot_a_qpos,
            robot_b_qpos=self.robot_b_qpos,
            shared_object_qpos=self.shared_object_qpos,
            fps=np.asarray(self.fps),
        )
        return path

    def save_canonical_rollout(self, path: Path, *, provenance=None) -> Path:
        np.savez_compressed(
            path,
            root_pos=np.stack((self.robot_a_qpos[:, :3], self.robot_b_qpos[:, :3]), axis=1),
            provenance=np.asarray(json.dumps(provenance, sort_keys=True)),
        )
        return path


def test_cli_writes_manifest_and_uses_frozen_defaults(tmp_path, monkeypatch, capsys) -> None:
    module = _load_script_module()
    source = tmp_path / "kick_eval.npz"
    source.write_bytes(b"aggregate-eval-fixture")
    output = tmp_path / "dual_kick.npz"
    qpos_output = tmp_path / "dual_kick_qpos.npz"
    calls = []

    def _fake_synthesize(
        path,
        *,
        source_attempt,
        layout,
        source_contact_local_z_m,
        target_contact_half_span_m,
    ):
        calls.append(
            (
                path,
                source_attempt,
                layout,
                source_contact_local_z_m,
                target_contact_half_span_m,
            )
        )
        return _FakeKickReference()

    monkeypatch.setattr(module, "synthesize_dual_kick_reference_file", _fake_synthesize)
    assert module.main(
        [
            "--source",
            str(source),
            "--output",
            str(output),
            "--qpos-output",
            str(qpos_output),
        ]
    ) == 0

    assert calls == [
        (
            source.resolve(),
            9,
            "mirrored",
            pytest.approx(-0.23525),
            pytest.approx(0.6742736),
        )
    ]
    manifest_path = output.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    assert manifest["attempt"] == {
        "completed": True,
        "end_exclusive": 2716,
        "end_inclusive": 2715,
        "index": 9,
        "indexing": "zero_based",
        "length_steps": 298,
        "start": 2418,
    }
    assert manifest["frames"] == 3
    assert manifest["fps"] == 50
    assert manifest["layout"] == "mirrored"
    assert manifest["agent_joint_mirroring"] is True
    assert manifest["mirror_plane"] == "table_local_xy"
    geometry = manifest["geometry"]
    assert geometry["layout"] == "mirrored"
    assert geometry["robot_a_lateral_offset_m"] == pytest.approx(-0.4390236)
    assert geometry["robot_b_lateral_offset_m"] == pytest.approx(0.4390236)
    assert geometry["target_robot_spacing_m"] == pytest.approx(1.3485472)
    assert manifest["artifacts"]["canonical_rollout"]["sha256"] == module._sha256(output)
    assert manifest["artifacts"]["qpos_contract"]["sha256"] == module._sha256(qpos_output)
    report = json.loads(capsys.readouterr().out)
    assert report["manifest"]["sha256"] == module._sha256(manifest_path)


def test_cli_forwards_explicit_same_action_layout(tmp_path, monkeypatch) -> None:
    module = _load_script_module()
    source = tmp_path / "kick_eval.npz"
    source.write_bytes(b"aggregate-eval-fixture")
    output = tmp_path / "dual_kick_same_action.npz"
    calls = []

    def _fake_synthesize(path, **kwargs):
        calls.append((path, kwargs))
        return _FakeKickReference(
            layout="same_action",
            robot_b_lateral_offset_m=0.9095236,
        )

    monkeypatch.setattr(module, "synthesize_dual_kick_reference_file", _fake_synthesize)
    assert module.main(
        [
            "--source",
            str(source),
            "--output",
            str(output),
            "--layout",
            "same_action",
        ]
    ) == 0
    assert calls[0][1]["layout"] == "same_action"
    manifest = json.loads(output.with_suffix(".manifest.json").read_text())
    assert manifest["layout"] == "same_action"
    assert manifest["agent_joint_mirroring"] is False
    assert manifest["mirror_plane"] is None
    assert manifest["geometry"]["robot_b_lateral_offset_m"] == pytest.approx(0.9095236)


def test_cli_refuses_to_overwrite_without_force(tmp_path, monkeypatch) -> None:
    module = _load_script_module()
    source = tmp_path / "kick_eval.npz"
    source.write_bytes(b"aggregate-eval-fixture")
    output = tmp_path / "dual_kick.npz"
    output.write_bytes(b"keep-me")
    called = False

    def _unexpected_synthesize(*args, **kwargs):
        nonlocal called
        called = True
        return _FakeKickReference()

    monkeypatch.setattr(module, "synthesize_dual_kick_reference_file", _unexpected_synthesize)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        module.main(["--source", str(source), "--output", str(output)])
    assert output.read_bytes() == b"keep-me"
    assert not called
