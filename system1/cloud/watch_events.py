#!/usr/bin/env python3
"""Local watcher for the cloud training round: exits (printing the event) at the first of
  * a final-segment exit record appears for one of the 15 runs (completed or failed), or any segment fails;
  * a GPU has no training queue left (Kevin 2026-11-10: report when a card is completely free);
  * a non-finite value appears in any run's metrics;
  * the cloud has been unreachable for --max-failures consecutive polls.
Events already present at start are listed in --seen (JSON) and ignored, so the watcher can be restarted.

    python system1/cloud/watch_events.py --seen logs_system1/post_train/seen_events.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from system1.t03.post_train import RUNS, SSH  # noqa: E402

NY = ZoneInfo("America/New_York")
REMOTE = r'''
import json, glob, os, re, subprocess
T = "/root/autodl-tmp/system1/logs/train"
ex = {}
for f in glob.glob(T + "/ops/*.exit.json"):
    try:
        ex[os.path.basename(f)[:-10]] = bool(json.load(open(f)).get("completed"))
    except Exception:
        pass
ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
gpu = {str(g): 0 for g in range(5)}
for l in ps.splitlines():
    m = re.search(r"run_train_queue.py.*--gpu (\d)", l)
    if m and "python" in l:
        gpu[m.group(1)] += 1
bad = []
for f in glob.glob(T + "/*/metrics.jsonl"):
    with open(f, "rb") as fh:
        fh.seek(max(0, os.path.getsize(f) - 200000))
        txt = fh.read().decode(errors="replace")
    if re.search(r"\b(NaN|Infinity|-Infinity)\b", txt):
        bad.append(f.split("/")[-2])
print(json.dumps(dict(exits=ex, gpu=gpu, nonfinite=bad)))
'''


def now():
    return datetime.now(NY).isoformat(timespec="seconds")


def poll():
    r = subprocess.run(SSH + ["python3 -"], input=REMOTE, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-300:])
    return json.loads(r.stdout.strip().splitlines()[-1])


def events(st):
    ev = set()
    for label, ok in st["exits"].items():
        if label in RUNS:
            ev.add(f"run_{'completed' if ok else 'failed'}:{label}")
        elif not ok:
            ev.add(f"segment_failed:{label}")
    for g, n in st["gpu"].items():
        if n == 0:
            ev.add(f"gpu_idle:{g}")
    for r in st["nonfinite"]:
        ev.add(f"nonfinite:{r}")
    if all(st["exits"].get(r) for r in RUNS):
        ev.add("all_runs_completed")
    return ev


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seen", type=Path, required=True)
    p.add_argument("--interval", type=int, default=600)
    p.add_argument("--max-failures", type=int, default=6)
    a = p.parse_args()
    seen = set(json.loads(a.seen.read_text())) if a.seen.exists() else set()
    fails = 0
    while True:
        try:
            st = poll()
            fails = 0
        except Exception as e:
            fails += 1
            print(now(), "poll failed", fails, repr(e)[:200], flush=True)
            if fails >= a.max_failures:
                print(json.dumps(dict(time=now(), new_events=["cloud_unreachable"])), flush=True)
                return 3
            time.sleep(a.interval)
            continue
        new = sorted(events(st) - seen)
        if new:
            a.seen.parent.mkdir(parents=True, exist_ok=True)
            a.seen.write_text(json.dumps(sorted(seen | set(new)), indent=1))
            print(json.dumps(dict(time=now(), new_events=new, gpu=st["gpu"]), indent=1), flush=True)
            return 0
        time.sleep(a.interval)


if __name__ == "__main__":
    raise SystemExit(main())
