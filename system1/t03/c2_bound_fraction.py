#!/usr/bin/env python3
"""T03 C v2 check 3d (Kevin 2026-11-12): per joint, the fraction of C-in actions at the bound (|a| >= 0.99) at training
iterations 1000 / 2000 / 3000 (Pull: the saved 1050 / 2050 / 3050) of the three C-in main runs.

Method (same replay as the pre-training check, system1/t03/c2_replay_check.py): the production training command of the run
(its cloud launch record) with --resume <checkpoint> --num-envs 256 --steps-per-env 240 --iterations 1 and a scratch output
dir, SYSTEM1_C_DIAG on (system1/c_diag.py). That is one stochastic training rollout from the training reset distribution with
the checkpoint's policy and noise; c_diag stores |a| (the sampled action the learner stores, before the C-in clip to [-1, 1])
per step / env / robot / joint as float16 (spacing 2^-11 = 0.00049 just below 1, so a value within 0.0005 of 0.99 may fall on
either side). Reported per joint over all 240 x 256 x 2 samples: fraction |a| >= 0.99, fraction |a| > 1 (the training
diagnostic's frac_clipped, per joint) and mean |a|. Runs alone on the laptop GPU (pipeline paused; a replay uses 3.7 GB).

    hssim_s1/bin/python -B system1/t03/c2_bound_fraction.py --out results_system1/T03/c2_checks/d_bound_fraction
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from system1.t03.c_rollout_diag import PY, sha  # noqa: E402
from system1.t03.post_train import evaluator_count, git, now  # noqa: E402

EXP = ROOT / "exports_system1/train_partial"
SCRATCH = ROOT / "logs_system1/T03/c2_bound_fraction"
RUNS = {"cin_pull_m20kg_seed721_joint8000": (1050, 2050, 3050),
        "cin_table_m5kg_seed721_fresh12000": (1000, 2000, 3000),
        "cin_bucket_m1kg_seed721_fresh12000": (1000, 2000, 3000)}
NUM_ENVS, STEPS = 256, 240
REPLACE = {"--num-envs": str(NUM_ENVS), "--steps-per-env": str(STEPS), "--iterations": "1"}


def replay_argv(label, ck, out_dir):
    rec = json.loads((EXP / label / f"{label}.launch.json").read_text())
    src = rec["argv"][1:]  # drop the cloud interpreter
    argv, i = [], 0
    while i < len(src):
        t = src[i]
        if t in REPLACE:
            argv += [t, REPLACE[t]]
            i += 2
        elif t in ("--output-dir", "--resume"):
            i += 2
        else:
            argv.append(t)
            i += 1
    return ["taskset", "-c", "24-31", PY, *argv, "--resume", str(ck), "--output-dir", str(out_dir)], rec["argv"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    rows, recs = [], []
    for label, its in RUNS.items():
        for it in its:
            ck = EXP / label / f"model_{it:05d}.pt"
            name = f"{label}_it{it:05d}"
            run_dir = SCRATCH / "runs" / name
            k = 1
            while run_dir.exists():
                run_dir = SCRATCH / "runs" / f"{name}.attempt{k}"
                k += 1
            argv, prod = replay_argv(label, ck, run_dir)
            while evaluator_count() > 0:  # the GPU alone
                time.sleep(20)
            npz = SCRATCH / f"{name}.npz"
            env = dict(os.environ, SYSTEM1_C_DIAG=str(npz), LOCAL_RANK="64", WORLD_SIZE="1")
            rec = dict(name=name, label=label, iteration=it, argv=argv, production_argv=prod, started=now(), checkpoint=str(ck),
                       checkpoint_sha256=sha(ck), git_head=git("rev-parse", "HEAD").strip(), git_status=git("status", "--porcelain"),
                       gpu=subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())
            with open(SCRATCH / f"{name}.stdout.log", "w") as log:
                proc = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                rec["pid"] = proc.pid
                (SCRATCH / f"{name}.launch.json").write_text(json.dumps(rec, indent=1))
                rc = proc.wait()
            rec.update(returncode=rc, ended=now())
            j = json.loads(npz.with_suffix(".json").read_text()) if npz.with_suffix(".json").exists() else {}
            rec["diag_summary"] = j
            ok = rc == 0 and j.get("steps") == STEPS and j.get("kind") == "in"
            rec["ok"] = bool(ok)
            recs.append(rec)
            print(now(), name, "rc", rc, "steps", j.get("steps"), flush=True)
            if not ok:
                continue
            z = np.load(npz)
            aa = z["a_abs"].astype(np.float32)          # [steps, E, A, 29]
            names = [str(x) for x in z["joint_names"]]
            flat = aa.reshape(-1, aa.shape[-1])
            for j_, n in enumerate(names):
                v = flat[:, j_]
                rows.append(dict(run=label, task=label.split("_")[1], iteration=it, joint=n, n_samples=int(v.size),
                                 frac_abs_a_ge_0p99=float((v >= 0.99).mean()), frac_abs_a_gt_1=float((v > 1.0).mean()),
                                 mean_abs_a=float(v.mean()), checkpoint_sha256=rec["checkpoint_sha256"]))
            rows.append(dict(run=label, task=label.split("_")[1], iteration=it, joint="ALL", n_samples=int(flat.size),
                             frac_abs_a_ge_0p99=float((flat >= 0.99).mean()), frac_abs_a_gt_1=float((flat > 1.0).mean()),
                             mean_abs_a=float(flat.mean()), checkpoint_sha256=rec["checkpoint_sha256"]))
            with (a.out / "per_joint.csv").open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
            (a.out / "replays.json").write_text(json.dumps(recs, indent=1))
    (a.out / "replays.json").write_text(json.dumps(recs, indent=1))
    print("done", all(r["ok"] for r in recs), flush=True)
    return 0 if all(r["ok"] for r in recs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
