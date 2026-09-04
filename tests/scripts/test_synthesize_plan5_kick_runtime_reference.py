from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/synthesize_plan5_kick_runtime_reference.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("synthesize_plan5_kick_runtime_reference", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_refuses_overwrite_before_build(monkeypatch, tmp_path) -> None:
    module = _load_script()
    source = tmp_path / "source.npz"
    output = tmp_path / "runtime.npz"
    source.write_bytes(b"source")
    output.write_bytes(b"existing")
    monkeypatch.setattr(
        module,
        "build_dual_kick_runtime_reference_file",
        lambda *_args, **_kwargs: pytest.fail("builder must not run"),
    )
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        module.main(["--source", str(source), "--output", str(output)])


def test_cli_reports_runtime_contract(monkeypatch, tmp_path, capsys) -> None:
    module = _load_script()
    source = tmp_path / "source.npz"
    output = tmp_path / "runtime.npz"
    source.write_bytes(b"source")

    def fake_build(_source, target):
        Path(target).write_bytes(b"runtime")
        return SimpleNamespace(
            agent_joint_pos=SimpleNamespace(shape=(298, 2, 29)),
            agent_body_pos_w=SimpleNamespace(shape=(298, 2, 51, 3)),
            fps=50,
        )

    monkeypatch.setattr(module, "build_dual_kick_runtime_reference_file", fake_build)
    monkeypatch.setattr(module, "sha256_file", lambda path: "source" if Path(path) == source else "hash")
    assert module.main(["--source", str(source), "--output", str(output)]) == 0
    report = capsys.readouterr().out
    assert '"frames": 298' in report
    assert '"layout": "mirrored"' in report
