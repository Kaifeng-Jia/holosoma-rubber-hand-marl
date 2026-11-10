#!/usr/bin/env python3
"""Start the T04-D + T03 training queues on the cloud: one run_train_queue.py per GPU, detached.

  source $S1_ROOT/s1_env.sh && python system1/cloud/start_queues.py --gpus 0,1,2,3,4,5,6,7,8

Runs are assigned longest-first to the GPU with the least total iterations (one run per GPU when
there are at least as many GPUs as runs). CPU cores are split evenly between the queues; each
queue gets its own LOCAL_RANK (conversion cache). Writes <out>/assignment.json and starts nothing
twice (refuses if a queue log for that GPU already exists). --dry-run prints the plan only.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
RUNS = ["d_pull:40", "d_table:10", "d_table:20", "d_bucket:2", "d_bucket:4", "d_bucket:8",  # T04-D
        "c_pull:20", "c_table:5", "c_bucket:1"]                                              # T03
ITERS = dict(pull=8050, table=12000, bucket=12000)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpus", required=True, help="comma-separated GPU indices")
    p.add_argument("--runs", default=",".join(RUNS))
    p.add_argument("--out", type=Path, default=None, help="default $S1_ROOT/logs/train")
    p.add_argument("--python", default=os.environ.get("S1_PY"))
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    gpus = a.gpus.split(",")
    runs = a.runs.split(",")
    out = a.out or Path(os.environ["S1_ROOT"]) / "logs" / "train"
    load = {g: 0 for g in gpus}
    plan = {g: [] for g in gpus}
    for r in sorted(runs, key=lambda r: -ITERS[r.split("_", 1)[1].split(":")[0]]):
        g = min(gpus, key=lambda g: (load[g], gpus.index(g)))
        plan[g].append(r)
        load[g] += ITERS[r.split("_", 1)[1].split(":")[0]]
    ncpu = os.cpu_count() or 1
    per = max(1, ncpu // len(gpus))
    cpus = {g: f"{i * per}-{(i + 1) * per - 1}" for i, g in enumerate(gpus)}
    assignment = dict(created=datetime.now(ZoneInfo("America/New_York")).isoformat(timespec="seconds"), out=str(out),
                      cpu_count=ncpu, queues=[dict(gpu=g, runs=plan[g], iterations=load[g], cpus=cpus[g], local_rank=str(200 + i))
                                              for i, g in enumerate(gpus) if plan[g]])
    print(json.dumps(assignment, indent=1))
    if a.dry_run:
        return 0
    out.mkdir(parents=True, exist_ok=True)
    for q in assignment["queues"]:
        log = out / f"queue_gpu{q['gpu']}.log"
        if log.exists():
            raise FileExistsError(f"{log} exists; not starting a second queue on GPU {q['gpu']}")
    (out / "assignment.json").write_text(json.dumps(assignment, indent=1))
    for q in assignment["queues"]:
        log = out / f"queue_gpu{q['gpu']}.log"
        argv = [a.python, "-B", str(ROOT / "system1/cloud/run_train_queue.py"), *q["runs"], "--out", str(out), "--python", a.python,
                "--gpu", q["gpu"], "--local-rank", q["local_rank"], "--cpus", q["cpus"]]
        with log.open("w") as fh:
            subprocess.Popen(argv, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        print(f"started GPU {q['gpu']}: {' '.join(q['runs'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
