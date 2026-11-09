#!/usr/bin/env python3
"""Sequential queue for the T04 D-part (end-to-end DuetMimic) mass-sweep trainings.

Each segment's exact argv, start/end time (America/New_York), exit code, git HEAD,
`git status --porcelain` and the sha256 of its parent checkpoint are written next to
the run as ``<label>.launch.json`` / ``<label>.exit.json``. The queue stops at the
first failing segment and never deletes anything. Already-finished segments
(an exit record with returncode 0) are skipped, so the queue can be restarted.

Configs mirror the paper/V3 runs; only mass, seed and directory differ:
* Table Pulling: V3 Full-rerun (prior init): 50 critic-only + 8000 joint, episode telemetry.
* Table Transport: paper T-07 seed721: fresh 12000, bucket_A_no_rel_no_height.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
PY = "/home/kevin/.holosoma_deps/miniconda3/envs/hssim_s1/bin/python"
OUT = ROOT / "logs_system1" / "T04" / "D"
NY = ZoneInfo("America/New_York")
SEED = 721
# Separate URDF->USD conversion cache (converted_rank<LOCAL_RANK>) from concurrent SONIC runs;
# device is still cuda:0 (WORLD_SIZE=1). Same practice as the V3 cloud runs.
CHILD_ENV = dict(os.environ, LOCAL_RANK="21", WORLD_SIZE="1", CUDA_VISIBLE_DEVICES="0")


def now():
    return datetime.now(NY).isoformat(timespec="seconds")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True).stdout


def segments(order):
    segs = []
    for task, mass in order:
        if task == "pull":
            tag = f"pull_m{mass:g}kg_seed{SEED}".replace(".", "p")
            common = [
                "scripts/train_plan5_push.py", "--skill", "pull", "--num-envs", "2048", "--steps-per-env", "24",
                "--seed", str(SEED), "--save-interval", "1000", "--episode-telemetry",
                "--actor-learning-rate", "0.00001", "--critic-learning-rate", "0.001",
                "--object-mass-kg", f"{mass:g}",
            ]
            c50 = OUT / f"{tag}_critic50"
            segs.append(dict(label=f"{tag}_critic50", out=c50, parent=None,
                             argv=common + ["--iterations", "50", "--output-dir", str(c50), "--critic-only"]))
            j = OUT / f"{tag}_joint8000"
            segs.append(dict(label=f"{tag}_joint8000", out=j, parent=c50 / "model_00050.pt",
                             argv=common + ["--iterations", "8000", "--output-dir", str(j), "--resume", str(c50 / "model_00050.pt")]))
        elif task == "table":
            tag = f"table_m{mass:g}kg_seed{SEED}".replace(".", "p")
            o = OUT / f"{tag}_fresh12000"
            segs.append(dict(label=f"{tag}_fresh12000", out=o, parent=None, argv=[
                "scripts/train_core4d_smalltable.py", "--experiment", "smalltable5kg_A",
                "--reward-variant", "bucket_A_no_rel_no_height", "--num-envs", "2048", "--steps-per-env", "24",
                "--seed", str(SEED), "--iterations", "12000", "--save-interval", "2000",
                "--output-dir", str(o), "--object-mass-kg", f"{mass:g}"]))
        else:
            raise ValueError(task)
    return segs


def main():
    order = [tuple((t, float(m))) for t, m in (x.split(":") for x in sys.argv[1:])]
    OUT.mkdir(parents=True, exist_ok=True)
    ops = OUT / "ops"
    ops.mkdir(exist_ok=True)
    for seg in segments(order):
        exit_path = ops / f"{seg['label']}.exit.json"
        if exit_path.exists() and json.loads(exit_path.read_text()).get("returncode") == 0:
            continue
        if seg["out"].exists():
            print(f"{seg['label']}: output dir exists without a successful exit record; stopping", flush=True)
            return 2
        argv = ["taskset", "-c", "0-7", PY, "-B", *seg["argv"]]
        launch = dict(label=seg["label"], argv=argv, cwd=str(ROOT), started=now(), git_head=git("rev-parse", "HEAD").strip(),
                      git_status=git("status", "--porcelain"),
                      parent_checkpoint=None if seg["parent"] is None else str(seg["parent"]),
                      parent_sha256=None if seg["parent"] is None else sha(seg["parent"]),
                      env={k: CHILD_ENV.get(k) for k in ("CUDA_VISIBLE_DEVICES", "LOCAL_RANK", "WORLD_SIZE")})
        (ops / f"{seg['label']}.launch.json").write_text(json.dumps(launch, indent=1))
        print(f"{now()} start {seg['label']}", flush=True)
        with (ops / f"{seg['label']}.stdout.log").open("w") as log:
            rc = subprocess.run(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=CHILD_ENV).returncode
        exit_path.write_text(json.dumps(dict(label=seg["label"], returncode=rc, ended=now()), indent=1))
        print(f"{now()} end {seg['label']} rc={rc}", flush=True)
        if rc != 0:
            return rc
        time.sleep(5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
