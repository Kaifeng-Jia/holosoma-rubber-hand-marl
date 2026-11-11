#!/usr/bin/env python3
"""Run evaluator jobs on the cloud in fixed GPU slots (T03 C v2: the freed slots evaluate first).

jobs.json: list of {"name", "argv": [script, args...], "env": {optional extra env}}; each job runs as
  CUDA_VISIBLE_DEVICES=<slot gpu> LOCAL_RANK=<unique> $S1_PY -B <argv>   in the repo root,
writing <out>/ops/<name>.launch.json (argv, time, git HEAD/status, GPU) and <name>.exit.json (returncode, completed =
summary.json complete). A job whose output summary.json is already complete is skipped. Never deletes anything.

    $S1_PY system1/cloud/run_eval_jobs.py --jobs jobs.json --out $S1_ROOT/logs/evals --gpu-slots 1,3,4
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
NY = ZoneInfo("America/New_York")


def now():
    return datetime.now(NY).isoformat(timespec="seconds")


def out_dir(job):
    a = job["argv"]
    return Path(a[a.index("--output-dir") + 1])


def complete(job):
    s = out_dir(job) / "summary.json"
    try:
        return bool(json.loads(s.read_text()).get("complete"))
    except (OSError, json.JSONDecodeError):
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jobs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--gpu-slots", required=True, help="comma list, one entry per concurrent job (a GPU may repeat)")
    p.add_argument("--python", default=os.environ.get("S1_PY"))
    p.add_argument("--rank-base", type=int, default=500)
    a = p.parse_args()
    ops = a.out / "ops"
    ops.mkdir(parents=True, exist_ok=True)
    jobs = [j for j in json.loads(a.jobs.read_text())]
    slots = a.gpu_slots.split(",")
    running = {}  # slot index -> (job, proc, log)
    pending = [j for j in jobs if not complete(j)]
    for j in jobs:
        if complete(j):
            print(now(), "skip (complete)", j["name"], flush=True)
    git = lambda *x: subprocess.run(["git", "-C", str(ROOT), *x], capture_output=True, text=True).stdout  # noqa: E731
    while pending or running:
        for i, (job, proc, log) in list(running.items()):
            if proc.poll() is not None:
                log.close()
                ok = complete(job)
                (ops / f"{job['name']}.exit.json").write_text(json.dumps(dict(name=job["name"], returncode=proc.returncode, completed=ok,
                                                                            ended=now()), indent=1))
                print(now(), "end", job["name"], "rc", proc.returncode, "complete", ok, flush=True)
                del running[i]
        free = [i for i in range(len(slots)) if i not in running]
        while pending and free:
            i = free.pop(0)
            job = pending.pop(0)
            gpu = slots[i]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, LOCAL_RANK=str(a.rank_base + i), WORLD_SIZE="1", **job.get("env", {}))
            argv = [a.python, "-B", *job["argv"]]
            smi = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total", "--format=csv,noheader"],
                                 capture_output=True, text=True).stdout
            (ops / f"{job['name']}.launch.json").write_text(json.dumps(dict(
                name=job["name"], argv=argv, env={k: env[k] for k in ("CUDA_VISIBLE_DEVICES", "LOCAL_RANK")} | job.get("env", {}),
                started=now(), git_head=git("rev-parse", "HEAD").strip(), git_status=git("status", "--porcelain"),
                nvidia_smi=smi, meta=job.get("meta")), indent=1))
            log = (ops / f"{job['name']}.stdout.log").open("w")
            proc = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            running[i] = (job, proc, log)
            print(now(), "start", job["name"], "gpu", gpu, flush=True)
            time.sleep(20)  # stagger Isaac start-ups
        time.sleep(15)
    print(now(), "all done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
