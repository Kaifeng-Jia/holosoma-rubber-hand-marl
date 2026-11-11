#!/usr/bin/env python3
"""Stop named cloud training runs (process only; logs and checkpoints are kept, nothing is deleted).

For every queue argument given (e.g. cin_table:5), takes the run_train_queue.py PID recorded at launch (never searched
by name; Kevin 2026-11-11): from --pid-file (JSON {run: pid}, default logs/train/c2_pids.json) or the queue's own
ops/queue_pid_<run>_seed<seed>.json. It checks /proc/<pid>/cmdline is run_train_queue.py with that run, finds the training
child by parent PID, records the last logged iteration and the newest checkpoint (with sha256) in ops/<label>.stopped.json, sends SIGTERM to the
child, waits up to --wait s, then SIGKILL. The queue process then writes its own <label>.exit.json (completed = false) and ends.
Runs as python3 on the cloud (stdlib only), e.g.  ssh ... python3 - --reason "..." c_table:5 c_bucket:1 < stop_runs.py

T03 C v2 (Kevin 2026-11-11 14:41): stop the C v1 runs; D runs untouched. (That first stop, 14:43, still located the
queues by matching their command line; the PID rule applies from 2026-11-11 16:05.)
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import signal
import subprocess
import time
from datetime import datetime, timedelta, timezone

T = "/root/autodl-tmp/system1/logs/train"


def now():
    # America/New_York without zoneinfo data dependence: the cloud image has tzdata; fall back to -05:00
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/New_York")).isoformat(timespec="seconds")
    except Exception:
        return datetime.now(timezone(timedelta(hours=-5))).isoformat(timespec="seconds")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def procs():
    out = subprocess.run(["ps", "-eo", "pid,ppid,args"], capture_output=True, text=True).stdout.splitlines()[1:]
    rows = []
    for ln in out:
        pid, ppid, args = ln.strip().split(None, 2)
        rows.append((int(pid), int(ppid), args))
    return rows


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--reason", required=True)
    p.add_argument("--wait", type=int, default=60)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--pid-file", default=os.path.join(T, "c2_pids.json"))
    p.add_argument("--seed", type=int, default=721)
    a = p.parse_args()
    ps = procs()
    recorded = json.load(open(a.pid_file)) if os.path.exists(a.pid_file) else {}
    plan = []
    for run in a.runs:
        qpid = recorded.get(run, {}).get("pid") if isinstance(recorded.get(run), dict) else recorded.get(run)
        own = os.path.join(T, "ops", f"queue_pid_{run.replace(':', '_')}_seed{a.seed}.json")
        if qpid is None and os.path.exists(own):
            qpid = json.load(open(own))["pid"]
        if qpid is None:
            print(json.dumps(dict(run=run, error="no recorded PID; not stopped")), flush=True)
            continue
        try:
            cmd = open(f"/proc/{qpid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            print(json.dumps(dict(run=run, pid=qpid, error="recorded PID not running")), flush=True)
            continue
        if "run_train_queue.py" not in cmd or not re.search(r"run_train_queue\.py\s+" + re.escape(run) + r"(\s|$)", cmd):
            print(json.dumps(dict(run=run, pid=qpid, error="recorded PID is not this run's queue", cmdline=cmd[:200])), flush=True)
            continue
        kids = [r for r in ps if r[1] == qpid and ("train_plan5_push.py" in r[2] or "train_core4d_smalltable.py" in r[2])]
        if len(kids) != 1:
            print(json.dumps(dict(run=run, error=f"{len(kids)} training children")), flush=True)
            continue
        cpid, _, cargs = kids[0]
        out_dir = re.search(r"--output-dir\s+(\S+)", cargs).group(1)
        label = os.path.basename(out_dir.rstrip("/"))
        cks = sorted(glob.glob(os.path.join(out_dir, "model_*.pt")))
        last_it = None
        mf = os.path.join(out_dir, "metrics.jsonl")
        if os.path.exists(mf):
            with open(mf, "rb") as fh:
                fh.seek(max(0, os.path.getsize(mf) - 20000))
                lines = [ln for ln in fh.read().decode(errors="replace").splitlines() if ln.strip().startswith("{")]
            if lines:
                try:
                    last_it = json.loads(lines[-1]).get("iteration")
                except json.JSONDecodeError:
                    last_it = None
        plan.append(dict(run=run, label=label, queue_pid=qpid, child_pid=cpid, output_dir=out_dir, last_logged_iteration=last_it,
                         newest_checkpoint=cks[-1] if cks else None, newest_checkpoint_sha256=sha(cks[-1]) if cks else None,
                         n_checkpoints=len(cks), child_args=cargs))
    for x in plan:
        print(json.dumps({k: v for k, v in x.items() if k != "child_args"}), flush=True)
    if a.dry_run:
        return 0
    for x in plan:
        x.update(reason=a.reason, sigterm_at=now())
        os.kill(x["child_pid"], signal.SIGTERM)
    t0 = time.time()
    while time.time() - t0 < a.wait and any(alive(x["child_pid"]) for x in plan):
        time.sleep(2)
    for x in plan:
        if alive(x["child_pid"]):
            os.kill(x["child_pid"], signal.SIGKILL)
            x["sigkill_at"] = now()
    time.sleep(10)
    for x in plan:
        x["child_alive_after"] = alive(x["child_pid"])
        x["queue_alive_after"] = alive(x["queue_pid"])
        ex = os.path.join(T, "ops", x["label"] + ".exit.json")
        x["exit_record"] = json.load(open(ex)) if os.path.exists(ex) else None
        with open(os.path.join(T, "ops", x["label"] + ".stopped.json"), "w") as f:
            json.dump(x, f, indent=1)
        print(json.dumps({k: x[k] for k in ("label", "last_logged_iteration", "sigterm_at", "child_alive_after", "queue_alive_after")}
                         | {"sigkill": "sigkill_at" in x, "exit_completed": (x["exit_record"] or {}).get("completed")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
