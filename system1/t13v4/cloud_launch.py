#!/usr/bin/env python3
"""T13 v4 round 1: launch the three cloud runs automatically once the section 4 B speed test and the cloud smokes are done
(local, in tmux; no Claude session needed). Reads logs_system1/t13v4/cloud/hosts.json (local, not committed).

Waits (poll 60 s) for: <speed host>:speed4b/speed_summary.json, smoke.json of d_push and d_table on the d_push host, smoke.json
of d_table_pa on its host. Launches nothing when a smoke failed, the speed test did not finish, or 2048 ran slower than 6 s/it
(section 4 B: report to Kevin). Otherwise, on each run's host (its own GPU 0):
    source s1_env.sh; [OMP_NUM_THREADS = MKL_NUM_THREADS = T]; nohup python3 system1/t13v4/cloud_runs.py --run R --num-envs N
        --gpu 0 --out <runs> [--torch-threads T] &      (N and T from speed_summary.json, the same for all runs)
with the runner PID in <runs>/R.runner.pid. Then runs status_hourly.py once and installs its hourly cron line (if absent).
Record: logs_system1/t13v4/cloud/launch_record.json.

    python3 system1/t13v4/cloud_launch.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
HOSTS = ROOT / "logs_system1/t13v4/cloud/hosts.json"
REC = ROOT / "logs_system1/t13v4/cloud/launch_record.json"
S1 = "/root/autodl-tmp/system1"
L = f"{S1}/logs/t13v4"
RUNS = f"{L}/runs"
NY = ZoneInfo("America/New_York")
CRON = "0 * * * * cd /home/kevin/duet-system1 && /usr/bin/python3 system1/t13v4/status_hourly.py >> logs_system1/t13v4/cloud/status_cron.log 2>&1"


def now():
    return datetime.now(NY).isoformat(timespec="seconds")


def ssh(spec, cmd, timeout=120):
    p = subprocess.run([*spec, "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", cmd], capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout


def read_json(spec, path):
    try:
        rc, out = ssh(spec, f"cat {path} 2>/dev/null")
        return json.loads(out) if rc == 0 and out.strip() else None
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="check once and print the launch commands, launch nothing")
    a = ap.parse_args()
    runs = json.loads(HOSTS.read_text())["runs"]
    speed_host = runs["d_table"]["ssh"]
    need = {"speed": (speed_host, f"{L}/speed4b/speed_summary.json"),
            "smoke_d_push": (runs["d_push"]["ssh"], f"{L}/smoke_cloud/d_push/smoke.json"),
            "smoke_d_table": (runs["d_push"]["ssh"], f"{L}/smoke_cloud/d_table/smoke.json"),
            "smoke_d_table_pa": (runs["d_table_pa"]["ssh"], f"{L}/smoke_cloud/d_table_pa/smoke.json")}
    rec = dict(started=now(), waits=[])
    while True:
        got = {k: read_json(*v) for k, v in need.items()}
        missing = [k for k, v in got.items() if v is None]
        if not missing or a.dry_run:
            break
        print(f"{now()} waiting for {missing}", flush=True)
        time.sleep(60)
    sp = got["speed"] or {}
    problems = [k for k in ("smoke_d_push", "smoke_d_table", "smoke_d_table_pa") if not (got[k] or {}).get("passed")]
    if not sp.get("num_envs"):
        problems.append("speed test without a decision")
    if sp.get("slower_than_6s_at_2048"):
        problems.append("2048 slower than 6 s/it (section 4 B: report to Kevin, change machines)")
    n = sp.get("num_envs")
    t = (sp.get("thread_setting") or {}).get("threads")
    rec.update(speed_decision=dict(num_envs=n, threads=t, conditions=sp.get("conditions"), thread_setting=sp.get("thread_setting")),
               smokes={k: (got[k] or {}).get("passed") for k in need if k.startswith("smoke")}, problems=problems)
    cmds = {}
    for run, h in runs.items():
        env = f"export OMP_NUM_THREADS={t} MKL_NUM_THREADS={t}; " if t else ""
        tt = f" --torch-threads {t}" if t else ""
        cmds[run] = (f"source {S1}/s1_env.sh; cd {S1}/duet-system1; mkdir -p {RUNS}; {env}"
                     f"nohup python3 system1/t13v4/cloud_runs.py --run {run} --num-envs {n} --gpu 0 --out {RUNS}{tt} > {RUNS}/{run}.runner.log 2>&1 < /dev/null &\n"
                     f"P=$!\necho $P > {RUNS}/{run}.runner.pid\nsleep 5; kill -0 $P && echo RUNNER_ALIVE $P; git -C {S1}/duet-system1 rev-parse HEAD")
    rec["commands"] = cmds
    if a.dry_run or problems:
        print(json.dumps(rec, indent=1))
        if problems and not a.dry_run:
            REC.write_text(json.dumps(rec, indent=1) + "\n")
            print("NOT_LAUNCHED", flush=True)
        return 1 if problems else 0
    rec["launched"] = {}
    for run, h in runs.items():
        rc, out = ssh(h["ssh"], f"bash -c {shlex.quote(cmds[run])}")
        rec["launched"][run] = dict(machine=h.get("machine"), time=now(), returncode=rc, output=out.strip())
        print(run, rc, out.strip(), flush=True)
    REC.write_text(json.dumps(rec, indent=1) + "\n")
    time.sleep(600)  # first metrics lines before the first status
    s = subprocess.run(["/usr/bin/python3", str(ROOT / "system1/t13v4/status_hourly.py")], cwd=ROOT, capture_output=True, text=True)
    print("status_hourly", s.returncode, s.stdout[-2000:], s.stderr[-500:], flush=True)
    cur = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
    lines = cur.stdout.splitlines() if cur.returncode == 0 else []
    if CRON not in lines:
        subprocess.run(["crontab", "-"], input="\n".join(lines + [CRON]) + "\n", text=True, check=True)
    print("cron:", subprocess.run(["crontab", "-l"], capture_output=True, text=True).stdout, flush=True)
    print("LAUNCH_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
