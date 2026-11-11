#!/usr/bin/env python3
"""T03 C v2 scheduler on the cloud: the preregistered iteration-1000 early stop and the iteration-2000 choice of the
C kind for the T04 C mass sweep (T03 "提前停止、质量扫描（事先定）"; Kevin 2026-11-11: act without waiting for a reply).

Runs (seed 721): cin_/cout_ x pull (20 kg, critic50 + joint8000), table (5 kg), bucket (1 kg). Iteration numbers are the
logged ones; for Pull, 1-50 are the critic-only segment and 51.. the joint segment.
Metrics: mean_step_reward = metrics `reward_mean`; mean episode length = `Episode/mean_length_last100_steps` (Pull,
--episode-telemetry) or `Episode/reset_length_mean_steps` (Table / Bin), the same fields as the v1 diagnosis.

* Iteration 1000, per kind: when all 3 runs of the kind have logged iteration 1000, a run counts as progressing if its mean
  over iterations 901-1000 of episode length OR of reward_mean is higher than its own mean over 1-100. If none of the 3
  progresses, the kind's 3 runs are stopped (system1/cloud/stop_runs.py; data kept).
* Iteration 2000: when every kind not stopped has all 3 runs at iteration 2000: per task, the kind with the higher mean
  reward_mean over 1901-2000 wins the task; the kind winning more tasks does the C mass sweep (6 runs: same config, only
  the mass: pull 40, table 10 / 20, bucket 2 / 4 / 8). Only one kind left: that one. None left: no C mass sweep.
* Mass sweep launch: longest expected first (iterations x median seconds per iteration of the kind's main run of the
  task over its last 200 logged iterations), each on a GPU with fewer than --per-gpu compute processes (nvidia-smi),
  one at a time, 120 s apart.
Every decision with its numbers goes to <train>/c2_decisions.json and c2_scheduler.log; nothing is deleted.

    nohup $S1_PY -B system1/cloud/c2_scheduler.py --train $S1_ROOT/logs/train > $S1_ROOT/logs/train/c2_scheduler.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
NY = ZoneInfo("America/New_York")
KINDS = ("cin", "cout")
TASKS = {"pull": "pull:20", "table": "table:5", "bucket": "bucket:1"}
LABEL = {"pull": "{k}_pull_m20kg_seed721", "table": "{k}_table_m5kg_seed721_fresh12000", "bucket": "{k}_bucket_m1kg_seed721_fresh12000"}
LEN_KEY = {"pull": "Episode/mean_length_last100_steps", "table": "Episode/reset_length_mean_steps", "bucket": "Episode/reset_length_mean_steps"}
MASS = [("pull", "pull:40", 8050), ("table", "table:10", 12000), ("table", "table:20", 12000),
        ("bucket", "bucket:2", 12000), ("bucket", "bucket:4", 12000), ("bucket", "bucket:8", 12000)]


def now():
    return datetime.now(NY).isoformat(timespec="seconds")


def metrics(train, kind, task):
    base = LABEL[task].format(k=kind)
    files = [train / f"{base}_critic50" / "metrics.jsonl", train / f"{base}_joint8000" / "metrics.jsonl"] if task == "pull" \
        else [train / base / "metrics.jsonl"]
    rows = {}
    for f in files:
        if f.exists():
            for line in f.read_text().splitlines():
                try:
                    j = json.loads(line)
                except json.JSONDecodeError:
                    continue  # partial last line
                rows[int(j["iteration"])] = j
    return rows


def wmean(rows, lo, hi, key):
    v = [rows[i][key] for i in range(lo, hi + 1) if i in rows and isinstance(rows[i].get(key), (int, float))]
    return (sum(v) / len(v) if v else None), len(v)


def gpu_load():
    q = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], capture_output=True, text=True).stdout
    idx = {u.strip(): int(i) for i, u in (l.split(",") for l in q.strip().splitlines())}
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"], capture_output=True, text=True).stdout
    load = {i: 0 for i in idx.values()}
    for l in apps.strip().splitlines():
        if "," in l:
            load[idx[l.split(",")[1].strip()]] += 1
    return load


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--per-gpu", type=int, default=3)
    p.add_argument("--python", default=os.environ.get("S1_PY"))
    p.add_argument("--poll", type=int, default=120)
    a = p.parse_args()
    dec_path = a.train / "c2_decisions.json"
    dec = json.loads(dec_path.read_text()) if dec_path.exists() else dict(early_stop={}, choice=None, mass_started={})
    save = lambda: dec_path.write_text(json.dumps(dec, indent=1))  # noqa: E731
    print(now(), "c2 scheduler start; state", json.dumps(dec)[:300], flush=True)
    while True:
        # iteration 1000
        for k in KINDS:
            if k in dec["early_stop"]:
                continue
            m = {t: metrics(a.train, k, t) for t in TASKS}
            if not all(max(r, default=0) >= 1000 for r in m.values()):
                continue
            per = {}
            for t, r in m.items():
                e_len, n1 = wmean(r, 1, 100, LEN_KEY[t])
                l_len, n2 = wmean(r, 901, 1000, LEN_KEY[t])
                e_rew, n3 = wmean(r, 1, 100, "reward_mean")
                l_rew, n4 = wmean(r, 901, 1000, "reward_mean")
                prog = (e_len is not None and l_len is not None and l_len > e_len) or (e_rew is not None and l_rew is not None and l_rew > e_rew)
                per[t] = dict(len_1_100=e_len, len_901_1000=l_len, reward_1_100=e_rew, reward_901_1000=l_rew,
                              n=[n1, n2, n3, n4], length_key=LEN_KEY[t], progressing=bool(prog))
            stop = not any(x["progressing"] for x in per.values())
            dec["early_stop"][k] = dict(decided=now(), runs=per, stop=stop)
            save()
            print(now(), "iteration-1000 check", k, json.dumps(dec["early_stop"][k]), flush=True)
            if stop:
                r = subprocess.run(["python3", str(ROOT / "system1/cloud/stop_runs.py"), "--reason",
                                    f"T03 C v2 iteration-1000 rule: no {k} run progressed", *[f"{k}_{v}" for v in TASKS.values()]],
                                   capture_output=True, text=True)
                dec["early_stop"][k]["stop_output"] = (r.stdout + r.stderr)[-3000:]
                save()
                print(now(), "stopped", k, r.stdout[-1500:], flush=True)
        # iteration 2000
        if dec["choice"] is None and all(k in dec["early_stop"] for k in KINDS):
            alive = [k for k in KINDS if not dec["early_stop"][k]["stop"]]
            ready = all(max(metrics(a.train, k, t), default=0) >= 2000 for k in alive for t in TASKS)
            if not alive:
                dec["choice"] = dict(decided=now(), kind=None, reason="both kinds stopped at iteration 1000: no C mass sweep")
                save()
                print(now(), "no C mass sweep", flush=True)
            elif ready:
                per, wins = {}, {k: 0 for k in alive}
                for t in TASKS:
                    v = {k: wmean(metrics(a.train, k, t), 1901, 2000, "reward_mean") for k in alive}
                    per[t] = {k: dict(reward_1901_2000=x[0], n=x[1]) for k, x in v.items()}
                    if len(alive) == 2:
                        w = max(alive, key=lambda k: v[k][0] if v[k][0] is not None else float("-inf"))
                        per[t]["winner"] = w
                        wins[w] += 1
                kind = alive[0] if len(alive) == 1 else max(alive, key=lambda k: wins[k])
                dec["choice"] = dict(decided=now(), kind=kind, alive=alive, per_task=per, wins=wins)
                save()
                print(now(), "iteration-2000 choice", json.dumps(dec["choice"]), flush=True)
        # mass sweep launches
        if dec["choice"] and dec["choice"].get("kind"):
            kind = dec["choice"]["kind"]
            todo = []
            for task, run, its in MASS:
                rid = f"{kind}_{run}"
                if rid in dec["mass_started"]:
                    continue
                r = metrics(a.train, kind, task)
                last = sorted(r)[-200:]
                spi = statistics.median([r[i]["elapsed_seconds"] for i in last if "elapsed_seconds" in r[i]]) if last else 0.0
                todo.append((its * spi, rid, its, spi))
            todo.sort(reverse=True)
            if not todo:
                print(now(), "all C mass runs started; scheduler exits", flush=True)
                return 0
            load = gpu_load()
            free = sorted((n, g) for g, n in load.items() if n < a.per_gpu)
            if free:
                est, rid, its, spi = todo[0]
                g = free[0][1]
                n_q = len(dec["mass_started"])
                cmd = [a.python, "-B", str(ROOT / "system1/cloud/run_train_queue.py"), rid, "--out", str(a.train), "--python", a.python,
                       "--gpu", str(g), "--local-rank", str(600 + n_q)]
                log = a.train / f"queue_{rid.replace(':', '_')}_seed721.log"
                subprocess.Popen(["nohup", *cmd], stdout=log.open("w"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                 cwd=ROOT, start_new_session=True)
                dec["mass_started"][rid] = dict(time=now(), gpu=g, gpu_load_before=load, expected_seconds=est,
                                                sec_per_it_main=spi, iterations=its, cmd=cmd)
                save()
                print(now(), "started", rid, "on GPU", g, "load", load, flush=True)
                time.sleep(120)
                continue
        time.sleep(a.poll)


if __name__ == "__main__":
    raise SystemExit(main())
