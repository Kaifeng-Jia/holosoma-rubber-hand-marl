#!/usr/bin/env python3
"""Upload our own result files (videos, images, small results) to Google Drive and record them (task D0, steps 4-5).

    python3 system1/transfer/drive_upload.py --task T12_videos exports_system1/T12_videos/a.mp4 results_system1/T12_videos/manifest.json

For each source directory: `rclone copy --checksum --files-from <list>` into gdrive:DuetMimic_transfer/<task>/, then
`rclone check --checksum --files-from <list>` on the same files; then the Drive-side sha256 and size of every file are compared
with the local ones. Only after all of that passes, results_system1/<task>/drive_manifest.csv is written (file_name, bytes,
sha256, drive_path; rows of earlier uploads to the same task are kept, a re-uploaded name is replaced). Commands, return codes
and the rclone version go to logs_system1/drive/<task>/<time>/record.json.

Refuses: files outside exports_system1/ and results_system1/, checkpoint / weight / dataset extensions, two files with the
same name. The Drive folder is never shared by this script.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
REMOTE = "gdrive:DuetMimic_transfer"
ALLOWED_DIRS = ("exports_system1", "results_system1")
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".ckpt", ".onnx", ".safetensors", ".bin", ".engine", ".trt", ".plan", ".h5", ".pkl", ".npz", ".npy", ".hdf5"}
FIELDS = ("file_name", "bytes", "sha256", "drive_path")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()


def check_allowed(paths, root=ROOT):
    """resolved paths; raises ValueError for anything this script must not upload."""
    out, names = [], {}
    for p in paths:
        p = Path(p).resolve()
        rel = p.relative_to(root) if p.is_relative_to(root) else None
        if rel is None or rel.parts[0] not in ALLOWED_DIRS:
            raise ValueError(f"not under {ALLOWED_DIRS}: {p}")
        if p.suffix.lower() in FORBIDDEN_SUFFIXES:
            raise ValueError(f"checkpoint / weight / dataset type is not uploaded: {p}")
        if not p.is_file():
            raise ValueError(f"not a file: {p}")
        if p.name in names:
            raise ValueError(f"two files named {p.name}: {names[p.name]} and {p}")
        names[p.name] = p
        out.append(p)
    return out


def manifest_rows(paths, task):
    return [dict(file_name=p.name, bytes=p.stat().st_size, sha256=sha256(p), drive_path=f"{REMOTE}/{task}/{p.name}") for p in paths]


def merge_manifest(csv_path: Path, rows):
    old = {}
    if csv_path.exists():
        with csv_path.open(newline="") as f:
            old = {r["file_name"]: r for r in csv.DictReader(f)}
    old.update({r["file_name"]: {k: str(r[k]) for k in FIELDS} for r in rows})
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(old[k] for k in sorted(old))
    return [old[k] for k in sorted(old)]


def run(cmd, rec):
    p = subprocess.run(cmd, capture_output=True, text=True)
    rec["commands"].append(dict(argv=cmd, returncode=p.returncode, stderr_tail=[x for x in p.stderr.splitlines() if "client_id" not in x][-6:]))
    if p.returncode != 0:
        raise RuntimeError(f"failed ({p.returncode}): {' '.join(cmd)}\n{p.stderr[-2000:]}")
    return p.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="Drive subfolder, same name as results_system1/<task>/")
    ap.add_argument("--manifest", type=Path, default=None, help="manifest csv (default results_system1/<task>/drive_manifest.csv; "
                    "T15: one manifest results_system1/T15_figures/drive_manifest.csv for the group folders T15_figures/<group>)")
    ap.add_argument("files", nargs="+", type=Path)
    a = ap.parse_args()
    files = check_allowed(a.files)
    ny = ZoneInfo("America/New_York")
    log = ROOT / "logs_system1/drive" / a.task / datetime.now(ny).strftime("%Y%m%d_%H%M%S")
    log.mkdir(parents=True)
    dest = f"{REMOTE}/{a.task}"
    rec = dict(task=a.task, dest=dest, started=datetime.now(ny).isoformat(timespec="seconds"), rclone_version=None, commands=[])
    rec["rclone_version"] = run(["rclone", "version"], rec).splitlines()[0]
    rows = manifest_rows(files, a.task)
    groups = {}
    for p in files:
        groups.setdefault(p.parent, []).append(p.name)
    for i, (src, names) in enumerate(groups.items()):
        lst = log / f"files_from_{i}.txt"
        lst.write_text("".join(n + "\n" for n in names))
        run(["rclone", "copy", "--checksum", "--files-from", str(lst), str(src), dest, "--log-file", str(log / f"copy_{i}.log"), "--log-level", "INFO"], rec)
        run(["rclone", "check", "--checksum", "--files-from", str(lst), str(src), dest, "--combined", str(log / f"check_{i}.combined.txt")], rec)
    drive_sha = {n: h for h, n in (x.split(None, 1) for x in run(["rclone", "hashsum", "sha256", dest], rec).splitlines() if x.strip())}
    drive_size = {e["Name"]: e["Size"] for e in json.loads(run(["rclone", "lsjson", dest], rec))}
    bad = [r["file_name"] for r in rows if drive_sha.get(r["file_name"]) != r["sha256"] or drive_size.get(r["file_name"]) != r["bytes"]]
    rec.update(files=rows, source_dirs={str(k.relative_to(ROOT)): v for k, v in groups.items()}, drive_sha256_and_size_match=not bad, mismatched=bad)
    if not bad:
        merge_manifest(a.manifest.resolve() if a.manifest else ROOT / "results_system1" / a.task / "drive_manifest.csv", rows)
    rec["ended"] = datetime.now(ny).isoformat(timespec="seconds")
    (log / "record.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(dict(dest=dest, files=len(rows), drive_sha256_and_size_match=not bad, mismatched=bad, log=str(log.relative_to(ROOT))), indent=1))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
