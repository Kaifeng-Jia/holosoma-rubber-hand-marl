#!/usr/bin/env python3
"""T03 (Kevin 2026-11-13 22:1x): evaluate on the cloud 4090s the main-curve points listed in curves_end.json that the
crashed scheduler left out (group "main" only; the mass-sweep points are not evaluated, Kevin's decision), then exit.

Every job is the scheduler's own job (system1/cloud/c2_scheduler.c2_eval_jobs with the arguments the scheduler ran with:
--pm8-last --curves-on-4090 --mass-curves-last), launched exactly as the scheduler launches it: same argv
([python, -B, *job argv], cwd = the repo), same environment (CUDA_VISIBLE_DEVICES, LOCAL_RANK, WORLD_SIZE=1), same
records in <evals>/ops2 (<name>.launch.json with pid / git / meta, <name>.stdout.log, <name>.exit.json), same claim
handling (laptop-claimed final-checkpoint +/-2: claim stays the laptop's, cloud_rerun.json added, as --curves-on-4090
does), an earlier output directory renamed <dir>.attemptK. Jobs are spread over the GPUs (at most --per-gpu each),
launches 30 s apart (Isaac start-up). Nothing else is started or stopped.

    source /root/autodl-tmp/system1/s1_env.sh
    $S1_PY -B system1/cloud/run_curve_points.py --curves-end <evals>/ops2/curves_end.json --gpus 0,1,2,3,4 --per-gpu 6
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c2_scheduler as s  # noqa: E402
from claims import claim_local_fs, safe  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves-end", type=Path, default=None)
    ap.add_argument("--names", default=None, help="comma-separated job names (c2_curve/... or d_curve/...) instead of --curves-end "
                    "(Kevin 2026-11-13: the 6 D Bin seed-721 points that were never in the cloud D catalog)")
    ap.add_argument("--rank-base", type=int, default=900, help="LOCAL_RANK = rank-base + k (a second launch uses another base)")
    ap.add_argument("--train", type=Path, default=Path("/root/autodl-tmp/system1/logs/train"))
    ap.add_argument("--gpus", default="0,1,2,3,4")
    ap.add_argument("--per-gpu", type=int, default=6)
    ap.add_argument("--python", default=os.environ.get("S1_PY"))
    ap.add_argument("--group", default="main")
    ap.add_argument("--poll", type=float, default=60.0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not a.python or not Path(a.python).exists():
        raise SystemExit(f"--python / S1_PY not set or missing: {a.python!r} (source s1_env.sh)")
    if a.names:
        want = [n for n in a.names.split(",") if n]
    else:
        ce = json.loads(a.curves_end.read_text())
        want = [x["name"] for x in ce["not_evaluated"] if x["group"] == a.group]
    jobs = {j["name"]: j for j in s.c2_eval_jobs(a.train, True, True, True) + s.d_curve_jobs(True)}
    missing = [n for n in want if n not in jobs]
    if missing:
        raise SystemExit(f"not in the scheduler's job list: {missing}")
    todo = [jobs[n] for n in want if not s.summary_complete(s.EVALS / n)]
    todo.sort(key=lambda j: (-s.est_minutes(j, s.eval_minutes(s.EVALS / "ops2")), j["name"]))  # longest first
    gpus = [int(g) for g in a.gpus.split(",")]
    if len(todo) > len(gpus) * a.per_gpu:
        raise SystemExit(f"{len(todo)} jobs > {len(gpus)} GPUs x {a.per_gpu}")
    ops = s.EVALS / "ops2"
    print(s.now(), f"{len(want)} listed, {len(todo)} to evaluate", flush=True)
    if a.dry_run:
        for j in todo:
            print(j["name"], " ".join(j["argv"]))
        return 0
    git = lambda *x: subprocess.run(["git", "-C", str(s.ROOT), *x], capture_output=True, text=True).stdout  # noqa: E731
    running = {}
    for k, job in enumerate(todo):
        g = gpus[k % len(gpus)]
        name = job["name"]
        if job.get("sha256"):  # D checkpoints carry the catalog sha256 (as the scheduler checks)
            import hashlib
            ck = job["argv"][job["argv"].index("--checkpoint") + 1]
            if hashlib.sha256(Path(ck).read_bytes()).hexdigest() != job["sha256"]:
                print(s.now(), "checkpoint sha256 mismatch, skip", name, flush=True)
                continue
        if not claim_local_fs(name, "cloud", dict(gpu="NVIDIA GeForce RTX 4090")):
            if not job["meta"].get("ignore_claims"):
                print(s.now(), "claimed by another owner, skip", name, flush=True)
                continue
            rr = s.EVALS / "claims" / safe(name) / "cloud_rerun.json"
            if not rr.exists():
                rr.write_text(json.dumps(dict(time=s.now(), gpu="NVIDIA GeForce RTX 4090",
                                              reason="Kevin 2026-11-12: curve points on the cloud 4090"), indent=1))
        out = s.EVALS / name
        if out.exists():
            n = 1
            while out.with_name(f"{out.name}.attempt{n}").exists():
                n += 1
            out.rename(out.with_name(f"{out.name}.attempt{n}"))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), LOCAL_RANK=str(a.rank_base + k), WORLD_SIZE="1")
        argv = [a.python, "-B", *job["argv"]]
        lj = ops / f"{safe(name)}.launch.json"
        if lj.exists():  # the crashed scheduler's record (pid None) is kept, renamed
            n = 1
            while lj.with_name(f"{lj.name}.attempt{n}").exists():
                n += 1
            lj.rename(lj.with_name(f"{lj.name}.attempt{n}"))
        lj.write_text(json.dumps(dict(
            name=name, argv=argv, gpu=g, started=s.now(), env={x: env[x] for x in ("CUDA_VISIBLE_DEVICES", "LOCAL_RANK")}, pid=None,
            git_head=git("rev-parse", "HEAD").strip(), git_status=git("status", "--porcelain"), meta=job["meta"],
            launcher="system1/cloud/run_curve_points.py (Kevin 2026-11-13: " + ("--names" if a.names else "the main-curve points of curves_end.json") + ")"), indent=1))
        log = (ops / f"{safe(name)}.stdout.log").open("w")
        proc = subprocess.Popen(argv, cwd=s.ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        lj.write_text(json.dumps(dict(json.loads(lj.read_text()), pid=proc.pid), indent=1))
        running[name] = (proc, log)
        print(s.now(), "start", name, "gpu", g, "pid", proc.pid, flush=True)
        time.sleep(30)
    while running:
        for name, (proc, log) in list(running.items()):
            if proc.poll() is not None:
                log.close()
                ok = s.summary_complete(s.EVALS / name)
                (ops / f"{safe(name)}.exit.json").write_text(json.dumps(dict(name=name, returncode=proc.returncode, completed=ok,
                                                                             ended=s.now()), indent=1))
                print(s.now(), "end", name, "rc", proc.returncode, "complete", ok, flush=True)
                del running[name]
        time.sleep(a.poll)
    left = [n for n in want if not s.summary_complete(s.EVALS / n)]
    print(s.now(), "all done; not complete:", left, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
