#!/usr/bin/env python3
"""T07 step 3: speed / memory / CPU of the D-with-hands (learn) Bin training benchmark runs, from their own records
(metrics.jsonl per iteration, the 10 s monitor file, launch.json). CPU only, read-only.

    python3 system1/t07/bench_summary.py --bench exports_system1/cloud_sync/logs/t07_bench --out results_system1/T07/benchmark

Per run: iterations done, seconds per iteration (median and mean over iterations 11..last, the first 10 are start-up),
environment steps per second (num_envs x 24 steps per env / s_it), peak GPU memory of the process and of the whole GPU,
CPU cores used (monitor v2: per-interval delta of utime+stime / wall time, mean and p90 over the same window;
monitor v1 of the first run: ps %cpu = lifetime average, last value / 100).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
from pathlib import Path

STEPS_PER_ENV = 24
RUNS = [  # name, setting (what shares the GPU), GPU
    ("bench_learn_bin_e2048_it100", "first run: GPU 0 shared with one curve evaluation (until it ended)", 0),
    ("bench2_excl_gpu1_seed721", "alone on GPU 1", 1),
    ("bench2_3pergpu_gpu2_seed721", "3 x 2048 launched on GPU 2; 1 OOM at start-up -> 2 runs on the GPU", 2),
    ("bench2_3pergpu_gpu2_seed722", "3 x 2048 launched on GPU 2 (this run: CUDA OOM at start-up)", 2),
    ("bench2_3pergpu_gpu2_seed723", "3 x 2048 launched on GPU 2; 1 OOM at start-up -> 2 runs on the GPU", 2),
    ("bench3_3pergpu_e1024_gpu3_seed721", "3 x 1024 on GPU 3", 3),
    ("bench3_3pergpu_e1024_gpu3_seed722", "3 x 1024 on GPU 3", 3),
    ("bench3_3pergpu_e1024_gpu3_seed723", "3 x 1024 on GPU 3", 3),
]


def sha(p: Path):
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def mib(s):
    m = re.search(r"(\d+)\s*MiB", s or "")
    return int(m.group(1)) if m else None


def monitor(path: Path):
    rows = []
    for line in path.read_text().splitlines() if path.exists() else []:
        if "cores=" in line:  # v2: "<t> pid_mem=9452MiB cores=2.41 gpu1=9465 MiB, 15 %"
            t = line.split()[0]
            pm = re.search(r"pid_mem=(\d+)MiB", line)
            c = re.search(r"cores=([0-9.]+|nan)", line)
            g = re.search(r"gpu\d+=(\d+) MiB, (\d+) %", line)
            rows.append(dict(t=int(t), pid_mib=int(pm.group(1)) if pm else None,
                             cores=float(c.group(1)) if c and c.group(1) != "nan" else None,
                             gpu_mib=int(g.group(1)) if g else None, gpu_util=int(g.group(2)) if g else None, v=2))
        else:  # v1: "<t> <pid MiB> MiB cpu <ps %cpu> gpu0 <MiB> MiB, <util> %"
            m = re.match(r"(\d+) (\d+) MiB cpu ([0-9.]+) gpu\d+ (\d+) MiB, (\d+) %", line)
            if m:
                rows.append(dict(t=int(m.group(1)), pid_mib=int(m.group(2)), ps_cpu=float(m.group(3)), gpu_mib=int(m.group(4)),
                                 gpu_util=int(m.group(5)), v=1))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    out_rows = []
    for name, setting, gpu in RUNS:
        d = a.bench / name
        launch = json.loads((a.bench / f"{name}.launch.json").read_text()) if (a.bench / f"{name}.launch.json").exists() else {}
        cfg = json.loads((d / "run_config.json").read_text()) if (d / "run_config.json").exists() else {}
        n_env = launch.get("num_envs") or cfg.get("num_envs") or 2048
        met = [json.loads(x) for x in (d / "metrics.jsonl").read_text().splitlines()] if (d / "metrics.jsonl").exists() else []
        el = [(r["iteration"], r["elapsed_seconds"]) for r in met if "elapsed_seconds" in r]
        win = [s for i, s in el if i > 10]
        mon = monitor(a.bench / f"{name}.monitor.txt")
        cores = [r["cores"] for r in mon if r.get("cores") is not None]
        if cores:
            cores = cores[min(len(cores) - 1, 6):]  # drop the first minute (start-up)
        log = a.bench / f"{name}.log"
        oom = bool(log.exists() and "OutOfMemoryError" in log.read_text(errors="replace"))
        status = json.loads((d / "status.json").read_text()) if (d / "status.json").exists() else {}
        out_rows.append(dict(
            run=name, setting=setting, gpu=gpu, num_envs=n_env, steps_per_env=STEPS_PER_ENV, seed=launch.get("seed", 721),
            iterations_done=len(el), passed=status.get("passed"), cuda_oom=oom,
            s_per_it_median_it11plus=round(statistics.median(win), 2) if win else None,
            s_per_it_mean_it11plus=round(statistics.mean(win), 2) if win else None,
            env_steps_per_s=round(n_env * STEPS_PER_ENV / statistics.median(win)) if win else None,
            process_gpu_mib_max=max((r["pid_mib"] for r in mon if r.get("pid_mib")), default=None),
            gpu_total_mib_max=max((r["gpu_mib"] for r in mon if r.get("gpu_mib")), default=None),
            gpu_util_mean_pct=round(statistics.mean(r["gpu_util"] for r in mon), 1) if mon else None,
            cpu_cores_mean=round(statistics.mean(cores), 2) if cores else None,
            cpu_cores_p90=round(sorted(cores)[int(0.9 * (len(cores) - 1))], 2) if cores else None,
            cpu_ps_lifetime_avg_cores=round(mon[-1]["ps_cpu"] / 100, 2) if mon and mon[-1].get("v") == 1 else None,
            started=launch.get("started"), metrics_sha256=sha(d / "metrics.jsonl"), monitor_sha256=sha(a.bench / f"{name}.monitor.txt")))
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / "benchmark.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    for r in out_rows:
        print(r["run"], r["iterations_done"], r["s_per_it_median_it11plus"], r["env_steps_per_s"], r["process_gpu_mib_max"],
              r["gpu_total_mib_max"], r["cpu_cores_mean"], r["cpu_ps_lifetime_avg_cores"], r["cuda_oom"])


if __name__ == "__main__":
    raise SystemExit(main())
