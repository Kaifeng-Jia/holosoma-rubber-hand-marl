#!/usr/bin/env python3
"""T13 v4 section 6: cron watcher (every 10 min, no Claude session needed). For every run of the local hosts file that has no
launch marker yet (logs_system1/t13v4/post_run/<run>/launched.json), read the cloud's decision.json over ssh (read only); when
it exists and records an end time, write the marker and start system1/t13v4/post_run.py for that run in its own tmux session
(post_<run>). Nothing on the cloud is changed. Unreachable hosts are skipped and tried again next time.

    python3 system1/t13v4/post_run_watch.py          # cron: */10 * * * *
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
HOSTS = ROOT / "logs_system1/t13v4/cloud/hosts.json"
TMUX = "/home/kevin/.holosoma_deps/miniconda3/envs/tools_s1/bin/tmux"


def main():
    now = datetime.now(ZoneInfo("America/New_York")).isoformat(timespec="seconds")
    for run, v in json.loads(HOSTS.read_text())["runs"].items():
        d = ROOT / "logs_system1/t13v4/post_run" / run
        if (d / "launched.json").exists():
            continue
        try:
            p = subprocess.run([*v["ssh"][:-1], "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", v["ssh"][-1], f"cat {v['dir']}/decision.json"],
                               capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            print(now, run, "unreachable (timeout)")
            continue
        if p.returncode != 0 or not p.stdout.strip():
            print(now, run, "no decision yet" if p.returncode == 1 else f"ssh rc {p.returncode}")
            continue
        dec = json.loads(p.stdout)
        if not dec.get("ended"):
            print(now, run, "decision without end time; waiting")
            continue
        if str(dec.get("status", "")).startswith("stopped: resume check"):  # T13 section 9: "对不上先停" - no evaluation; queue_s9.py reports
            if not (d / "resume_check_failed.json").exists():
                d.mkdir(parents=True, exist_ok=True)
                (d / "resume_check_failed.json").write_text(json.dumps(dict(run=run, seen=now, cloud_decision=dec), indent=1) + "\n")
            print(now, run, "stopped by the section 9 resume check; no post_run")
            continue
        d.mkdir(parents=True, exist_ok=True)
        (d / "launched.json").write_text(json.dumps(dict(run=run, launched=now, cloud_decision=dec), indent=1) + "\n")
        # cron's PATH lacks ~/.local/bin (rclone for the Drive upload); set it explicitly for the pipeline
        cmd = (f"export PATH=/home/kevin/.local/bin:/usr/local/bin:/usr/bin:/bin; cd {ROOT} && "
               f"python3 system1/t13v4/post_run.py --run {run} > {d}/post_run.stdout.log 2>&1")
        subprocess.run([TMUX, "new-session", "-d", "-s", f"post_{run}", cmd], check=True)
        print(now, run, "ended", dec.get("ended"), "-> post_run started")


if __name__ == "__main__":
    main()
