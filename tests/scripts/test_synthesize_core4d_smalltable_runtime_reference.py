from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/synthesize_core4d_smalltable_runtime_reference.py"


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "synthesize_core4d_smalltable_runtime_reference",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cli_refuses_overwrite_before_build(monkeypatch, tmp_path: Path) -> None:
    module = _load_script()
    source = tmp_path / "source.npz"
    output = tmp_path / "runtime.npz"
    source.write_bytes(b"source")
    output.write_bytes(b"existing")
    monkeypatch.setattr(
        module,
        "build_core4d_pair_runtime_reference_file",
        lambda *_args, **_kwargs: pytest.fail("builder must not run"),
    )
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        module.main(["--source", str(source), "--output", str(output)])


def test_cli_writes_manifest_with_hashes_and_non_loop_contract(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    module = _load_script()
    source = tmp_path / "source.npz"
    output = tmp_path / "runtime.npz"
    manifest = tmp_path / "runtime.manifest.json"
    source.write_bytes(b"source")

    provenance = {
        "source_frames": 413,
        "source_fps": 30,
        "source_duration_seconds": 412 / 30,
        "sampled_duration_seconds": 13.72,
        "omitted_source_tail_seconds": 1 / 75,
        "uniform_runtime_timestep_seconds": 0.02,
        "terminal_behavior": "stop_at_last_reference_frame_then_environment_reset",
    }

    def fake_build(_source, target, **_kwargs):
        Path(target).write_bytes(b"runtime")
        return SimpleNamespace(
            agent_joint_pos=SimpleNamespace(shape=(687, 2, 29)),
            agent_body_pos_w=SimpleNamespace(shape=(687, 2, 51, 3)),
            fps=50,
            provenance=provenance,
        )

    monkeypatch.setattr(module, "build_core4d_pair_runtime_reference_file", fake_build)
    monkeypatch.setattr(module, "sha256_file", lambda _path: "hash")
    assert module.main(
        [
            "--source",
            str(source),
            "--output",
            str(output),
            "--manifest",
            str(manifest),
        ]
    ) == 0

    report = json.loads(manifest.read_text(encoding="utf-8"))
    assert report["frames"] == 687
    assert report["fps"] == 50
    assert report["episode_loop"] is False
    assert report["source_sha256"] == "hash"
    assert report["output_sha256"] == "hash"
    assert report["training_ready"] is False
    assert report["training_promotion"] is None
    assert '"episode_loop": false' in capsys.readouterr().out
