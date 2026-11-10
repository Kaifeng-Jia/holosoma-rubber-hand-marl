#!/usr/bin/env python3
"""Training queue for the cloud (T04-D and T03) — one queue per GPU.

Run ids (seed 721 unless --seed):
  d_pull:<kg>    D, Table Pulling, V3 Full-rerun config (prior init, 50 critic-only + 8000, telemetry)
  d_table:<kg>   D, Table Transport, paper T-07 seed721 config (fresh 12000)
  d_bucket:<kg>  D, Bin Transport, paper B-RZ0 seed721 config (fresh 12000)
  c_pull:<kg>    C, Table Pulling, V3-25 config (from scratch, 50 + 8000) + --sonic-residual
  c_table:<kg>   C, Table Transport, T-07 config + --sonic-residual
  c_bucket:<kg>  C, Bin Transport, B-RZ0 config + --sonic-residual
A mass equal to the task's original mass adds no --object-mass-kg flag.

Every segment writes <label>.launch.json (argv, time, git HEAD/status, GPU, parent sha256)
and <label>.exit.json (returncode, completed = status.json passed). Never deletes; skips
completed segments; stops the queue at the first failure.

--preflight K: run each segment's exact command until K iterations are logged, then stop it
(SIGTERM) — used to produce run_config diffs and to measure seconds per iteration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
NY = ZoneInfo("America/New_York")
ORIGINAL_MASS = {"pull": 20.0, "table": 5.0, "bucket": 1.0}
C_TF32 = True  # T03: TF32 may be on in training (evaluation turns it off)


def now():
    return datetime.now(NY).isoformat(timespec="seconds")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def segments(run, seed, out):
    scheme, rest = run.split("_", 1)
    task, mass = rest.split(":")
    mass = float(mass)
    tag = f"{scheme}_{task}_m{mass:g}kg_seed{seed}".replace(".", "p")
    mass_flag = [] if mass == ORIGINAL_MASS[task] else ["--object-mass-kg", f"{mass:g}"]
    c_flags = ["--sonic-residual"] + (["--sonic-tf32"] if C_TF32 else []) if scheme == "c" else []
    if task == "pull":
        base = ["scripts/train_plan5_push.py", "--skill", "pull", "--num-envs", "2048", "--steps-per-env", "24",
                "--seed", str(seed), "--save-interval", "1000", "--episode-telemetry",
                "--actor-learning-rate", "0.00001", "--critic-learning-rate", "0.001", *mass_flag, *c_flags]
        if scheme == "c":
            base.append("--from-scratch")
        c50, j = out / f"{tag}_critic50", out / f"{tag}_joint8000"
        return [dict(label=c50.name, out=c50, parent=None, argv=base + ["--iterations", "50", "--output-dir", str(c50), "--critic-only"]),
                dict(label=j.name, out=j, parent=c50 / "model_00050.pt",
                     argv=base + ["--iterations", "8000", "--output-dir", str(j), "--resume", str(c50 / "model_00050.pt")])]
    exp = dict(table="smalltable5kg_A", bucket="bucket003")[task]
    o = out / f"{tag}_fresh12000"
    return [dict(label=o.name, out=o, parent=None, argv=[
        "scripts/train_core4d_smalltable.py", "--experiment", exp, "--reward-variant", "bucket_A_no_rel_no_height",
        "--num-envs", "2048", "--steps-per-env", "24", "--seed", str(seed), "--iterations", "12000", "--save-interval",
        "1000" if scheme == "c" else "2000", "--output-dir", str(o), *mass_flag, *c_flags])]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--python", required=True)
    p.add_argument("--gpu", default="0")
    p.add_argument("--local-rank", default="21", help="conversion-cache id (LOCAL_RANK); unique per queue")
    p.add_argument("--seed", type=int, default=721)
    p.add_argument("--cpus", default=None)
    p.add_argument("--preflight", type=int, default=0)
    a = p.parse_args()
    out = a.out if a.out.is_absolute() else ROOT / a.out
    ops = out / "ops"
    ops.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpu, LOCAL_RANK=a.local_rank, WORLD_SIZE="1")
    for run in a.runs:
        for seg in segments(run, a.seed, out):
            ex = ops / f"{seg['label']}.exit.json"
            if ex.exists() and json.loads(ex.read_text()).get("completed"):
                continue
            if seg["out"].exists() and not a.preflight:
                print(f"{seg['label']}: output exists without a completed exit record; stopping", flush=True)
                return 2
            if seg["parent"] is not None and not Path(seg["parent"]).exists():
                if a.preflight:
                    print(f"{seg['label']}: parent checkpoint missing in preflight; skipped", flush=True)
                    continue
                return 3
            argv = [a.python, "-B", *seg["argv"]]
            if a.cpus:
                argv = ["taskset", "-c", a.cpus, *argv]
            git = lambda *x: subprocess.run(["git", "-C", str(ROOT), *x], capture_output=True, text=True).stdout  # noqa: E731
            smi = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free", "--format=csv,noheader"],
                                 capture_output=True, text=True).stdout
            launch = dict(label=seg["label"], argv=argv, started=now(), preflight_iterations=a.preflight or None,
                          git_head=git("rev-parse", "HEAD").strip(), git_status=git("status", "--porcelain"), nvidia_smi=smi,
                          env={k: env[k] for k in ("CUDA_VISIBLE_DEVICES", "LOCAL_RANK", "WORLD_SIZE")},
                          parent_checkpoint=None if seg["parent"] is None else str(seg["parent"]),
                          parent_sha256=None if seg["parent"] is None else sha(seg["parent"]))
            (ops / f"{seg['label']}.launch.json").write_text(json.dumps(launch, indent=1))
            print(f"{now()} start {seg['label']}", flush=True)
            with (ops / f"{seg['label']}.stdout.log").open("w") as log:
                proc = subprocess.Popen(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
                if a.preflight:
                    metrics = seg["out"] / "metrics.jsonl"
                    while proc.poll() is None:
                        if metrics.exists() and sum(1 for _ in metrics.open()) >= a.preflight:
                            proc.send_signal(signal.SIGTERM)
                            break
                        time.sleep(5)
                rc = proc.wait()
            status = seg["out"] / "status.json"
            completed = (not a.preflight) and rc == 0 and status.exists() and bool(json.loads(status.read_text()).get("passed"))
            ex.write_text(json.dumps(dict(label=seg["label"], returncode=rc, completed=completed, preflight=bool(a.preflight),
                                          ended=now()), indent=1))
            print(f"{now()} end {seg['label']} rc={rc} completed={completed}", flush=True)
            if not a.preflight and not completed:
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
