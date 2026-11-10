#!/usr/bin/env python3
"""Cloud training monitor / C mass-sweep scheduler (T04 C part).

  status : per run directory under <train>: last iteration, median s/it over the last 50 iterations, non-finite values
  c-mass : wait until the three T03 C runs (c_pull:20 joint8000 segment, c_table:5, c_bucket:1) have all passed
           iteration 1000 with no non-finite metric, then start the six T04 C runs one by one on GPUs that have fewer
           than --per-gpu active run queues (Kevin 2026-11-10: "有卡空出来就上"); checks every minute.

    source $S1_ROOT/s1_env.sh && $S1_PY system1/cloud/train_monitor.py status
    nohup $S1_PY system1/cloud/train_monitor.py c-mass --gpus 0,1,2,3,4 > logs/train/c_mass_scheduler.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
NY = ZoneInfo("America/New_York")
C_MAIN = {"c_pull_m20kg_seed721_joint8000", "c_table_m5kg_seed721_fresh12000", "c_bucket_m1kg_seed721_fresh12000"}
C_MASS = ["c_pull:40", "c_table:10", "c_table:20", "c_bucket:2", "c_bucket:4", "c_bucket:8"]


def now():
    return datetime.now(NY).isoformat(timespec="seconds")


def finite(v):
    return not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))


def run_status(d: Path):
    f = d / "metrics.jsonl"
    if not f.exists():
        return dict(run=d.name, iteration=None)
    rows, bad = [], 0
    with f.open() as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # partially written last line
            rows.append(r)
            bad += sum(not finite(v) for v in r.values() if isinstance(v, float))
    if not rows:
        return dict(run=d.name, iteration=None)
    el = sorted(r["elapsed_seconds"] for r in rows[-50:] if "elapsed_seconds" in r)
    return dict(run=d.name, iteration=rows[-1].get("iteration"), median_s_per_it_last50=el[len(el) // 2] if el else None,
                n_nonfinite_values=bad, n_rows=len(rows))


def all_status(train: Path):
    return [run_status(d) for d in sorted(train.iterdir())
            if d.is_dir() and ((d / "run_config.json").exists() or (d / "metrics.jsonl").exists())]


def active_per_gpu():
    out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    cnt = {}
    for line in out.splitlines():
        if "run_train_queue.py" in line and "python" in line:
            m = re.search(r"--gpu (\S+)", line)
            if m:
                cnt[m.group(1)] = cnt.get(m.group(1), 0) + 1
    return cnt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("status", "c-mass"))
    p.add_argument("--train", type=Path, default=Path(os.environ.get("S1_ROOT", "/root/autodl-tmp/system1")) / "logs/train")
    p.add_argument("--gpus", default="0,1,2,3,4")
    p.add_argument("--per-gpu", type=int, default=2)
    a = p.parse_args()
    if a.mode == "status":
        print(json.dumps(dict(time=now(), runs=all_status(a.train), active_per_gpu=active_per_gpu()), indent=1))
        return 0
    pending = list(C_MASS)
    started = {}
    gate_passed_at = None
    while pending:
        st = {s["run"]: s for s in all_status(a.train)}
        if gate_passed_at is None:
            ok = all(n in st and (st[n]["iteration"] or 0) >= 1000 and st[n].get("n_nonfinite_values", 1) == 0 for n in C_MAIN)
            bad = [n for n in C_MAIN if n in st and st[n].get("n_nonfinite_values", 0) > 0]
            print(now(), "gate", ok, {n: (st.get(n) or {}).get("iteration") for n in C_MAIN}, "nonfinite", bad, flush=True)
            if bad:
                print(now(), "non-finite metrics in a C main run: not starting the C mass sweep", flush=True)
                return 2
            if ok:
                gate_passed_at = now()
                print(now(), "gate passed", flush=True)
        if gate_passed_at is not None:
            act = active_per_gpu()
            free = [g for g in a.gpus.split(",") if act.get(g, 0) < a.per_gpu]
            if free:
                g = min(free, key=lambda g: (act.get(g, 0), a.gpus.split(",").index(g)))
                run = pending.pop(0)
                cmd = [sys.executable, "-B", str(ROOT / "system1/cloud/start_queues.py"), "--assign", f"{run}@{g}"]
                r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
                started[run] = dict(gpu=g, time=now(), rc=r.returncode, active_before=act)
                print(now(), "started", run, "on GPU", g, "rc", r.returncode, r.stdout[-300:], r.stderr[-300:], flush=True)
                (a.train / "c_mass_started.json").write_text(json.dumps(dict(gate_passed_at=gate_passed_at, started=started), indent=1))
                time.sleep(90)  # let the new queue appear in ps before the next decision
                continue
        time.sleep(60)
    print(now(), "all C mass runs started", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
