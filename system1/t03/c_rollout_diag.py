#!/usr/bin/env python3
"""Replay scheme C training rollouts locally from saved checkpoints (Kevin 2026-11-11: C does not learn; cloud runs
untouched). For each checkpoint: the cloud training command (same script, task flags, --sonic-residual --sonic-tf32),
except --resume <checkpoint> --iterations 1 --num-envs 256 --steps-per-env 240 and a scratch output dir: one 240-step
stochastic rollout with that checkpoint's policy (the update after it only touches the scratch dir). SYSTEM1_C_DIAG makes
SonicResidualEnv record dq, the SONIC fall rule, the reference check and the PD check (system1/c_diag.py).

    hssim_s1/bin/python -B system1/t03/c_rollout_diag.py [--only pull]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from system1.t03.post_train import evaluator_count, git, now  # noqa: E402

PY = "/home/kevin/.holosoma_deps/miniconda3/envs/hssim_s1/bin/python"
OUT = ROOT / "logs_system1/T03/c_diag"
P = ROOT / "exports_system1/train_partial"
NUM_ENVS, STEPS = 256, 240
TASKS = {
    "pull": ("c_pull_m20kg_seed721_joint8000", ["00050", "01050", "02050", "03050", "04050"],
             ["scripts/train_plan5_push.py", "--skill", "pull", "--seed", "721", "--episode-telemetry", "--actor-learning-rate", "0.00001",
              "--critic-learning-rate", "0.001", "--sonic-residual", "--sonic-tf32", "--from-scratch"]),
    "table": ("c_table_m5kg_seed721_fresh12000", ["00000", "01000", "02000", "03000", "04000"],
              ["scripts/train_core4d_smalltable.py", "--experiment", "smalltable5kg_A", "--reward-variant", "bucket_A_no_rel_no_height",
               "--seed", "721", "--sonic-residual", "--sonic-tf32"]),
    "bucket": ("c_bucket_m1kg_seed721_fresh12000", ["00000", "01000", "02000", "03000", "04000"],
               ["scripts/train_core4d_smalltable.py", "--experiment", "bucket003", "--reward-variant", "bucket_A_no_rel_no_height",
                "--seed", "721", "--sonic-residual", "--sonic-tf32"]),
}


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--its", nargs="*", default=None)
    p.add_argument("--slots", type=int, default=2)
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for task, (label, its, base) in TASKS.items():
        if a.only and task not in a.only:
            continue
        for it in its:
            if a.its and it not in a.its:
                continue
            name = f"{task}_it{it}"
            res = OUT / f"{name}.json"
            if res.exists() and json.loads(res.read_text()).get("steps") == STEPS:
                print(now(), "skip (done)", name, flush=True)
                continue
            ck = P / label / f"model_{it}.pt"
            run_dir = OUT / "runs" / name
            k = 1
            while run_dir.exists():  # earlier attempt: keep it, renamed
                run_dir = OUT / "runs" / f"{name}.attempt{k}"
                k += 1
            argv = ["taskset", "-c", "24-31", PY, "-B", *base, "--num-envs", str(NUM_ENVS), "--steps-per-env", str(STEPS), "--iterations", "1",
                    "--resume", str(ck), "--output-dir", str(run_dir)]
            while evaluator_count() >= a.slots:
                time.sleep(20)
            env = dict(os.environ, SYSTEM1_C_DIAG=str(OUT / f"{name}.npz"), LOCAL_RANK="64", WORLD_SIZE="1")
            rec = dict(name=name, argv=argv, started=now(), checkpoint=str(ck), checkpoint_sha256=sha(ck),
                       git_head=git("rev-parse", "HEAD").strip(), git_status=git("status", "--porcelain"), git_diff=git("diff", "HEAD"),
                       gpu=subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())
            (OUT / f"{name}.launch.json").write_text(json.dumps(rec, indent=1))
            with open(OUT / f"{name}.stdout.log", "w") as log:
                rc = subprocess.run(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
            done = res.exists() and json.loads(res.read_text()).get("steps") == STEPS
            (OUT / f"{name}.exit.json").write_text(json.dumps(dict(name=name, returncode=rc, completed=done, ended=now()), indent=1))
            print(now(), "end", name, "rc", rc, "complete", done, flush=True)


if __name__ == "__main__":
    main()
