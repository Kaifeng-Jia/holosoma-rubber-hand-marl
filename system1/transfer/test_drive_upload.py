"""Smoke test for drive_upload.py (no network, no rclone): the refusal rules, manifest rows and the CSV merge.

    python3 -m pytest -q system1/transfer/test_drive_upload.py
"""
import csv
import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import drive_upload as du  # noqa: E402


def _tree(tmp_path):
    (tmp_path / "exports_system1/T").mkdir(parents=True)
    (tmp_path / "results_system1/T").mkdir(parents=True)
    (tmp_path / "logs_system1").mkdir()
    files = {"exports_system1/T/a.mp4": b"video", "results_system1/T/manifest.json": b"{}", "exports_system1/T/model_1.pt": b"w",
             "logs_system1/x.mp4": b"v", "exports_system1/T/states.npz": b"d", "results_system1/T/a.mp4": b"other"}
    for k, v in files.items():
        (tmp_path / k).write_bytes(v)
    return tmp_path


def test_check_allowed(tmp_path):
    r = _tree(tmp_path)
    ok = du.check_allowed([r / "exports_system1/T/a.mp4", r / "results_system1/T/manifest.json"], root=r)
    assert [p.name for p in ok] == ["a.mp4", "manifest.json"]
    for bad in ("exports_system1/T/model_1.pt", "exports_system1/T/states.npz", "logs_system1/x.mp4", "exports_system1/T/missing.mp4"):
        with pytest.raises(ValueError):
            du.check_allowed([r / bad], root=r)
    with pytest.raises(ValueError, match="two files named"):
        du.check_allowed([r / "exports_system1/T/a.mp4", r / "results_system1/T/a.mp4"], root=r)


def test_manifest_rows_and_merge(tmp_path):
    r = _tree(tmp_path)
    rows = du.manifest_rows(du.check_allowed([r / "exports_system1/T/a.mp4"], root=r), "T")
    assert rows == [dict(file_name="a.mp4", bytes=5, sha256=hashlib.sha256(b"video").hexdigest(), drive_path="gdrive:DuetMimic_transfer/T/a.mp4")]
    out = r / "results_system1/T/drive_manifest.csv"
    du.merge_manifest(out, rows)
    du.merge_manifest(out, du.manifest_rows(du.check_allowed([r / "results_system1/T/manifest.json"], root=r), "T"))
    (r / "exports_system1/T/a.mp4").write_bytes(b"video2")
    du.merge_manifest(out, du.manifest_rows(du.check_allowed([r / "exports_system1/T/a.mp4"], root=r), "T"))
    with out.open(newline="") as f:
        got = list(csv.DictReader(f))
    assert [g["file_name"] for g in got] == ["a.mp4", "manifest.json"]
    assert got[0]["bytes"] == "6" and got[0]["sha256"] == hashlib.sha256(b"video2").hexdigest()
    assert list(got[0]) == list(du.FIELDS)
