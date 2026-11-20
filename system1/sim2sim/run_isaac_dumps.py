#!/usr/bin/env python3
"""T10 sim2sim, Isaac side: re-run the nominal episode (first state of the frozen +/-2 state file) of D and C-in, Table and Bin,
seed 721, with exactly the argv of the laptop-5070 final evaluations plus ``--limit 1 --sim2sim-dump <dir>``: the evaluator
writes the PhysX scene export and a per-step record (system1/sim2sim/isaac_dump.py). Two at a time on the 5070 (taskset
16-23 / 24-31 as the finals). Records <name>.launch.json / .exit.json next to each output.

    python3 system1/sim2sim/run_isaac_dumps.py [--only d_table,cin_table]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "logs_system1/t10/isaac"
LAUNCH = dict(
    d_table=ROOT / "logs_system1/post_train/d_curve/D_table_paper_T-07_seed721/it12000_pm2.launch.json",
    cin_table=ROOT / "logs_system1/post_train/final/cin_table_m5kg_seed721_fresh12000_pm2.launch.json",
    d_bucket=ROOT / "logs_system1/post_train/d_curve/D_bucket_paper_B-RZ0_seed721/it12000_pm2.launch.json",
    cin_bucket=ROOT / "logs_system1/post_train/final/cin_bucket_m1kg_seed721_fresh12000_pm2.launch.json",
)
SLOTS = ("16-23", "24-31")
NY = ZoneInfo("America/New_York")


def now():
    return datetime.now(NY).isoformat(timespec="seconds")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def argv_for(name):
    L = json.loads(LAUNCH[name].read_text())
    argv = list(L["argv"])
    if argv[0] == "taskset":
        argv = argv[3:]
    i = argv.index("--output-dir")
    argv[i + 1] = str(OUT / name / "eval")
    i = argv.index("--model-id")
    argv[i + 1] = "t10_isaac_dump__" + name
    return argv + ["--limit", "1", "--sim2sim-dump", str(OUT / name / "dump")], L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=",".join(LAUNCH))
    a = ap.parse_args()
    todo = [n for n in a.only.split(",") if n]
    git = lambda *x: subprocess.run(["git", "-C", str(ROOT), *x], capture_output=True, text=True).stdout  # noqa: E731
    running = {}
    while todo or running:
        for slot in SLOTS:
            if slot in running or not todo:
                continue
            name = todo.pop(0)
            argv, src = argv_for(name)
            (OUT / name).mkdir(parents=True, exist_ok=True)
            ck = argv[argv.index("--checkpoint") + 1]
            st = argv[argv.index("--states") + 1]
            rec = dict(name=name, argv=["taskset", "-c", slot, *argv], started=now(), source_launch=str(LAUNCH[name]), checkpoint_sha256=sha(ck),
                       states_sha256=sha(st), git_head=git("rev-parse", "HEAD").strip(), git_status=git("status", "--porcelain"), git_diff=git("diff"))
            log = (OUT / name / "stdout.log").open("w")
            p = subprocess.Popen(rec["argv"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            rec["pid"] = p.pid
            (OUT / name / "launch.json").write_text(json.dumps(rec, indent=1))
            running[slot] = (name, p, log)
            print(now(), "start", name, "pid", p.pid, flush=True)
            time.sleep(20)
        for slot, (name, p, log) in list(running.items()):
            if p.poll() is not None:
                log.close()
                ok = (OUT / name / "eval" / "summary.json").exists()
                (OUT / name / "exit.json").write_text(json.dumps(dict(name=name, returncode=p.returncode, summary=ok, ended=now()), indent=1))
                print(now(), "end", name, "rc", p.returncode, "summary", ok, flush=True)
                del running[slot]
        time.sleep(10)
    print(now(), "all done", flush=True)


if __name__ == "__main__":
    main()
