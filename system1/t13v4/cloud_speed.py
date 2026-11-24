#!/usr/bin/env python3
"""T13 v4 section 4 B + cloud_speed_test_plan.md: environment count and thread setting on the first cloud card.

D-Table main-stage command line from `cloud_runs.py --dry-run` (identical flags), --iterations 100, one process at a time:
  1. t2048_default   2048 envs, PyTorch default threads (OMP_NUM_THREADS / MKL_NUM_THREADS removed, no --torch-threads)
  2. t2048_threadsN  2048 envs, OMP_NUM_THREADS = MKL_NUM_THREADS = N, --torch-threads N; N = the instance's vCPU allocation
                     (cgroup cpu.max quota / period; `nproc` is recorded too: in an AutoDL container it shows the host's CPUs)
  3. t4096_<faster>  4096 envs with the faster thread setting of 1 / 2 (median s/it after iteration 3)
Every 5 s: GPU utilisation and memory (nvidia-smi), CPU use of the training process tree (/proc utime + stime), metrics lines.
Means are taken over the samples after iteration 3. PhysX overflow warnings: log lines matching OVERFLOW. Decision (section 4 B):
4096 when (a) its s/it < 1.5 x the 2048 s/it with the same thread setting, (b) no overflow warning, (c) it ran to the end with
peak GPU memory below the card's total; else 2048. Writes <out>/speed_summary.json; nothing is launched afterwards.

    S1_PY=<hssim python> python3 system1/t13v4/cloud_speed.py --out <dir>
"""
from __future__ import annotations

import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "system1/t13v4"))
from cloud_smoke import metrics_summary, overflow_lines, sha  # noqa: E402

TICK = os.sysconf("SC_CLK_TCK")


def vcpu_quota():
    try:
        q, p = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        return None if q == "max" else int(int(q) // int(p))
    except Exception:
        return None


def tree_ticks(root):
    kids, ticks = {}, {}
    for p in Path("/proc").iterdir():
        if p.name.isdigit():
            try:
                f = (p / "stat").read_text().rsplit(")", 1)[1].split()
                kids.setdefault(int(f[1]), []).append(int(p.name))
                ticks[int(p.name)] = int(f[11]) + int(f[12])
            except Exception:
                pass
    tot, todo = 0, [root]
    while todo:
        x = todo.pop()
        tot += ticks.get(x, 0)
        todo += kids.get(x, [])
    return tot


def gpu():
    s = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True).stdout.strip().splitlines()[0]
    return [float(x) for x in s.split(",")]


def one(tag, envs, threads, out, py):
    argv_plan = [sys.executable, str(ROOT / "system1/t13v4/cloud_runs.py"), "--run", "d_table", "--num-envs", str(envs), "--out", str(out / "_plan"),
                 "--python", py, "--dry-run"] + (["--torch-threads", str(threads)] if threads else [])
    plan = json.loads(subprocess.run(argv_plan, capture_output=True, text=True, check=True).stdout)
    argv = list(plan["stages"][-1]["argv"])
    argv[argv.index("--iterations") + 1] = "100"
    argv[argv.index("--output-dir") + 1] = str(out / tag)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", LOCAL_RANK="400", WORLD_SIZE="1")
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        env.pop(k, None)
        if threads:
            env[k] = str(threads)
    cpustat0 = Path("/sys/fs/cgroup/cpu.stat").read_text() if Path("/sys/fs/cgroup/cpu.stat").exists() else ""
    t0 = time.time()
    log = out / f"{tag}.log"
    with open(log, "w") as f:
        p = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
        samples, last = [], (time.time(), tree_ticks(p.pid))
        while p.poll() is None:
            time.sleep(5)
            now, tk = time.time(), tree_ticks(p.pid)
            m = out / tag / "metrics.jsonl"
            n = len(m.read_text().splitlines()) if m.exists() else 0
            try:
                g = gpu()
            except Exception:
                g = [None, None, None]
            samples.append(dict(t=round(now - t0, 1), iteration_lines=n, cpu_percent=round(100 * (tk - last[1]) / TICK / (now - last[0]), 1),
                                gpu_util=g[0], gpu_mem_mib=g[1], gpu_mem_total_mib=g[2]))
            last = (now, tk)
    rc = p.wait()
    after = [s for s in samples if s["iteration_lines"] > 3]
    mean = lambda k: round(statistics.mean(s[k] for s in after if s[k] is not None), 1) if after else None  # noqa: E731
    cks = sorted((out / tag).glob("model_*.pt"))
    rc_cfg = out / tag / "run_config.json"
    r = dict(tag=tag, num_envs=envs, threads=threads, argv=argv, env_threads={k: env.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
             returncode=rc, wall_seconds=round(time.time() - t0, 1), **metrics_summary(out / tag / "metrics.jsonl"),
             gpu_util_mean=mean("gpu_util"), cpu_percent_mean=mean("cpu_percent"),
             gpu_mem_peak_mib=max((s["gpu_mem_mib"] for s in samples if s["gpu_mem_mib"] is not None), default=None),
             gpu_mem_total_mib=samples[-1]["gpu_mem_total_mib"] if samples else None, overflow_lines=overflow_lines(log),
             torch_threads_record=(json.loads(rc_cfg.read_text()).get("torch_threads") if rc_cfg.exists() else None),
             checkpoint_sha256=sha(cks[-1]) if cks else None, cgroup_cpu_stat_before=cpustat0,
             cgroup_cpu_stat_after=Path("/sys/fs/cgroup/cpu.stat").read_text() if cpustat0 else "")
    (out / f"{tag}.samples.json").write_text(json.dumps(samples) + "\n")
    (out / f"{tag}.json").write_text(json.dumps(r, indent=1) + "\n")
    print(json.dumps({k: r[k] for k in ("tag", "returncode", "s_per_it_median_after_3", "gpu_util_mean", "cpu_percent_mean", "gpu_mem_peak_mib")}
                     | dict(overflow=len(r["overflow_lines"]))), flush=True)
    return r


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--python", default=os.environ.get("S1_PY"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    n = vcpu_quota()
    lscpu = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
    info = dict(cpu_model=next((x.split(":", 1)[1].strip() for x in lscpu.splitlines() if x.startswith("Model name")), None),
                nproc=int(subprocess.run(["nproc"], capture_output=True, text=True).stdout.strip()), vcpu_quota=n,
                gpu=subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip(),
                git_head=subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
                git_status=subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True, text=True).stdout)
    runs = [one("t2048_default", 2048, None, a.out, a.python), one(f"t2048_threads{n}", 2048, n, a.out, a.python)]
    ok = [r for r in runs if r["returncode"] == 0 and r["s_per_it_median_after_3"]]
    faster = min(ok, key=lambda r: r["s_per_it_median_after_3"]) if ok else runs[0]
    r4 = one(f"t4096_{'default' if faster['threads'] is None else 'threads%d' % n}", 4096, faster["threads"], a.out, a.python)
    runs.append(r4)
    s2, s4 = faster["s_per_it_median_after_3"], r4["s_per_it_median_after_3"]
    cond = dict(ratio_4096_over_2048=round(s4 / s2, 3) if s2 and s4 else None,
                a_ratio_below_1p5=bool(s2 and s4 and s4 < 1.5 * s2),
                b_no_overflow_warning=not (r4["overflow_lines"] or faster["overflow_lines"]),
                c_memory_enough=bool(r4["returncode"] == 0 and r4["gpu_mem_peak_mib"] and r4["gpu_mem_total_mib"] and r4["gpu_mem_peak_mib"] < r4["gpu_mem_total_mib"]))
    summary = dict(info=info, runs=runs, thread_setting=dict(faster=faster["tag"], threads=faster["threads"]), conditions=cond,
                   num_envs=4096 if all(v for k, v in cond.items() if k != "ratio_4096_over_2048") else 2048,
                   slower_than_6s_at_2048=bool(s2 and s2 > 6.0))
    (a.out / "speed_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(dict(thread_setting=summary["thread_setting"], conditions=cond, num_envs=summary["num_envs"]), indent=1), flush=True)
    print("SPEED_DONE", flush=True)


if __name__ == "__main__":
    main()
