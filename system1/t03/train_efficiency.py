#!/usr/bin/env python3
"""T03 training efficiency (preregistered, docs/system1/T03_schemeC_marl_sonic_zh.md "训练速度"): completion rate on the frozen
+/-2 states (50 perturbed states, deterministic, TF32 off) of every saved checkpoint, C and D, all curve points from the
cloud RTX 4090 (Kevin 2026-11-12). CPU only, read-only.

Sources:
  * curve points: exports_system1/cloud_sync/logs/evals/{c2_curve,d_curve}/<run>/it<k>_pm2/summary.json (cloud sync);
  * training logs: exports_system1/cloud_sync/logs/train/<run>/metrics.jsonl (C, T04 retrained D on the same cloud 4090s);
    D paper runs: the run directories of the checkpoints in logs_system1/post_train/d_curve_catalog.json.
Every run trains with 2048 envs x 24 steps per iteration (run_config), so env steps = iteration x 49152.

Metrics per run: first evaluated checkpoint with completion >= 50 % and >= 80 % (iteration and env steps; "not reached"),
area under the completion-iteration curve (trapezoid over the evaluated checkpoints, divided by the iteration span = mean
completion in %, over the run's own span), final completion. A curve point that is missing or incomplete is listed and left
out (no interpolation), and the metrics of that run say how many points are missing.
Wall clock: median seconds per iteration (iterations 101..last) and total hours from metrics.jsonl elapsed_seconds; GPU model
from the launch record (C, T04 D) - the paper D runs do not record it.
Auxiliary reward curves (every 50 iterations): reward_mean (all terms) and, for Table / Bin, Bucket/raw_reward_mean (the
object-interaction tracking term); Pull logs a per-term snapshot at the end of each rollout (reward_terms_raw_mean), from
which the body + object tracking terms are summed.

    python3 system1/t03/train_efficiency.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SYNC = ROOT / "exports_system1/cloud_sync/logs"
OUT = ROOT / "results_system1/T03/train_efficiency"
STEPS_PER_IT = 2048 * 24
CAT = json.loads((ROOT / "logs_system1/post_train/d_curve_catalog.json").read_text())["runs"]
C_RUN = dict(pull="{k}_pull_m20kg_seed{s}_joint8000", table="{k}_table_m5kg_seed{s}_fresh12000", bucket="{k}_bucket_m1kg_seed{s}_fresh12000")
RUNS = []  # (task, scheme, seed, run label, curve kind)
for task in ("pull", "table", "bucket"):
    for s in (721, 722, 723):
        RUNS.append((task, "C-in", s, C_RUN[task].format(k="cin", s=s), "c2_curve"))
    RUNS.append((task, "C-out", 721, C_RUN[task].format(k="cout", s=721), "c2_curve"))
    for label, r in CAT.items():
        if r["case"] == task:
            scheme = "D (from scratch, V3-25-27)" if "_scratch_" in label else ("D (paper, single-person prior)" if task == "pull" else "D (paper)")
            RUNS.append((task, scheme, int(r["seed"]), label, "d_curve"))
T04_D = ["d_pull_m40kg_seed721_joint8000", "d_table_m10kg_seed721_fresh12000", "d_table_m20kg_seed721_fresh12000",
         "d_bucket_m2kg_seed721_fresh12000", "d_bucket_m4kg_seed721_fresh12000", "d_bucket_m8kg_seed721_fresh12000"]
PULL_TRACK = ("motion_global_ref_position_error_exp", "motion_global_ref_orientation_error_exp", "motion_relative_body_position_error_exp",
              "motion_relative_body_orientation_error_exp", "motion_global_body_lin_vel", "motion_global_body_ang_vel",
              "object_global_ref_position_error_exp", "object_global_ref_orientation_error_exp")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def expected_iterations(label, kind):
    if kind == "d_curve":
        return sorted(int(i) for i in CAT[label]["checkpoints"])
    d = SYNC / "train" / label
    return sorted(int(p.stem.split("_")[1]) for p in d.glob("model_*.pt"))


def curve(label, kind):
    pts = []
    for it in expected_iterations(label, kind):
        d = SYNC / "evals" / kind / label / f"it{it:05d}_pm2"
        s = json.loads((d / "summary.json").read_text()) if (d / "summary.json").exists() else None
        if not s or not s.get("complete") or len(s["episodes"]) != 51:
            pts.append(dict(iteration=it, status="missing" if not s else "incomplete", eval_dir=str(d.relative_to(ROOT))))
            continue
        per = [e for e in s["episodes"] if e["kind"] != "nominal"]
        assert len(per) == 50
        n = sum(bool(e["completed_reference"]) for e in per)
        pts.append(dict(iteration=it, status="complete", n_completed=n, completion_pct=2 * n, gpu=s.get("gpu_name"),
                        checkpoint_sha256=s.get("checkpoint_sha256"), states_sha256=s.get("states_sha256"),
                        summary_sha256=sha(d / "summary.json"), eval_dir=str(d.relative_to(ROOT))))
    return pts


def reach(pts, thr):
    for p in pts:
        if p["status"] == "complete" and p["completion_pct"] >= thr:
            return p["iteration"]
    return None


def auc(pts):
    c = [p for p in pts if p["status"] == "complete"]
    if len(c) < 2:
        return None
    a = sum((c[i + 1]["iteration"] - c[i]["iteration"]) * (c[i]["completion_pct"] + c[i + 1]["completion_pct"]) / 2 for i in range(len(c) - 1))
    return a / (c[-1]["iteration"] - c[0]["iteration"])


def metrics_paths(label, kind):
    if kind == "d_curve":
        dirs = []
        for it in sorted(CAT[label]["checkpoints"], key=int):
            p = Path(CAT[label]["checkpoints"][it]["path"]).parent
            if p not in dirs:
                dirs.append(p)
        return [p / "metrics.jsonl" for p in dirs if (p / "metrics.jsonl").exists()]
    return [SYNC / "train" / label / "metrics.jsonl"]


def load_metrics(paths):
    rows = {}
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            rows[int(r["iteration"])] = r
    return rows


def gpu_of(label):
    f = SYNC / "train" / "ops" / f"{label}.launch.json"
    if not f.exists():
        return None
    L = json.loads(f.read_text())
    env = L.get("env")
    env = json.loads(env.replace("'", '"')) if isinstance(env, str) else (env or {})
    idx = env.get("CUDA_VISIBLE_DEVICES")
    for line in str(L.get("nvidia_smi", "")).splitlines():
        f0 = [x.strip() for x in line.split(",")]
        if idx is not None and f0 and f0[0] == str(idx):
            return f0[1]
    m = re.search(r"NVIDIA [^,\n]+", str(L.get("nvidia_smi", "")))
    return m.group(0) if m else None


def wall(label, kind, rows):
    el = [rows[i]["elapsed_seconds"] for i in sorted(rows) if i > 100 and rows[i].get("elapsed_seconds") is not None]
    tot = sum(r["elapsed_seconds"] for r in rows.values() if r.get("elapsed_seconds") is not None)
    return dict(iterations_logged=len(rows), s_per_it_median_it101plus=statistics.median(el) if el else None,
                train_hours_total=tot / 3600 if tot else None, gpu_model=gpu_of(label) if kind == "c2_curve" or label.startswith("d_") else None)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pts_rows, run_rows, wall_rows, rew_rows = [], [], [], []
    for task, scheme, seed, label, kind in RUNS:
        pts = curve(label, kind)
        for p in pts:
            pts_rows.append(dict(task=task, scheme=scheme, seed=seed, run=label, env_steps=p["iteration"] * STEPS_PER_IT, **p))
        miss = [p["iteration"] for p in pts if p["status"] != "complete"]
        gpus = sorted({p.get("gpu") for p in pts if p["status"] == "complete"})
        r50, r80 = reach(pts, 50), reach(pts, 80)
        comp = [p for p in pts if p["status"] == "complete"]
        run_rows.append(dict(task=task, scheme=scheme, seed=seed, run=label, n_points_expected=len(pts), n_points_missing=len(miss),
                             missing_iterations=";".join(map(str, miss)),
                             first_iteration_ge_50=r50 if r50 is not None else "not reached",
                             env_steps_ge_50=r50 * STEPS_PER_IT if r50 is not None else "not reached",
                             first_iteration_ge_80=r80 if r80 is not None else "not reached",
                             env_steps_ge_80=r80 * STEPS_PER_IT if r80 is not None else "not reached",
                             auc_mean_completion_pct=auc(pts), first_point=comp[0]["iteration"] if comp else None,
                             last_point=comp[-1]["iteration"] if comp else None, last_point_completion_pct=comp[-1]["completion_pct"] if comp else None,
                             eval_gpus=";".join(str(g) for g in gpus), all_points_on_4090=all("4090" in str(g) for g in gpus) if gpus else None))
        rows = load_metrics(metrics_paths(label, kind))
        if rows:
            wall_rows.append(dict(task=task, scheme=scheme, seed=seed, run=label, **wall(label, kind, rows),
                                  metrics=";".join(str(p) for p in metrics_paths(label, kind))))
            for i in sorted(rows):
                if i % 50 and i != max(rows):
                    continue
                r = rows[i]
                x = dict(task=task, scheme=scheme, seed=seed, run=label, iteration=i, reward_mean=r.get("reward_mean"),
                         bucket_raw_reward_mean=r.get("Bucket/raw_reward_mean"))
                if isinstance(r.get("reward_terms_raw_mean"), dict):
                    x["pull_tracking_terms_snapshot_sum"] = sum(r["reward_terms_raw_mean"].get(k, 0.0) for k in PULL_TRACK)
                rew_rows.append(x)
    for label in T04_D:  # same cloud 4090s as C, trained concurrently (T04 retrained D, seed 721)
        rows = load_metrics([SYNC / "train" / label / "metrics.jsonl"])
        task = label.split("_")[1]
        wall_rows.append(dict(task=task, scheme="D (T04 retrained, same cloud 4090s as C)", seed=721, run=label, **wall(label, "c2_curve", rows),
                              metrics=str(SYNC / "train" / label / "metrics.jsonl")))
    # per-scheme summary over seeds
    summ = []
    for task in ("pull", "table", "bucket"):
        for scheme in sorted({r["scheme"] for r in run_rows if r["task"] == task}):
            rr = [r for r in run_rows if r["task"] == task and r["scheme"] == scheme]
            def agg(key):
                v = [r[key] for r in rr if isinstance(r[key], (int, float))]
                return dict(per_seed=";".join(str(r[key]) for r in rr), mean=statistics.mean(v) if len(v) == len(rr) and v else None,
                            n_reached=len(v))
            a50, a80, au = agg("first_iteration_ge_50"), agg("first_iteration_ge_80"), agg("auc_mean_completion_pct")
            summ.append(dict(task=task, scheme=scheme, seeds=";".join(str(r["seed"]) for r in rr), n_runs=len(rr),
                             first_iteration_ge_50_per_seed=a50["per_seed"], first_iteration_ge_50_mean=a50["mean"], n_reached_50=a50["n_reached"],
                             first_iteration_ge_80_per_seed=a80["per_seed"], first_iteration_ge_80_mean=a80["mean"], n_reached_80=a80["n_reached"],
                             auc_per_seed=";".join(f"{r['auc_mean_completion_pct']:.1f}" if r["auc_mean_completion_pct"] is not None else "" for r in rr),
                             auc_mean=au["mean"], points_missing=sum(r["n_points_missing"] for r in rr)))
    for fn, rows in (("curve_points.csv", pts_rows), ("per_run.csv", run_rows), ("scheme_summary.csv", summ), ("wallclock.csv", wall_rows),
                     ("reward_curves.csv", rew_rows)):
        keys = []
        for r in rows:
            keys += [k for k in r if k not in keys]
        with (OUT / fn).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    for r in run_rows:
        print(r["task"], r["scheme"][:12], r["seed"], "miss", r["n_points_missing"], "50%:", r["first_iteration_ge_50"], "80%:", r["first_iteration_ge_80"],
              "auc", None if r["auc_mean_completion_pct"] is None else round(r["auc_mean_completion_pct"], 1), "last", r["last_point"], r["last_point_completion_pct"],
              "4090", r["all_points_on_4090"])


if __name__ == "__main__":
    raise SystemExit(main())
