#!/usr/bin/env python3
"""T13 v4 round 2 (task book section 8 item 2): bring up C-out-Table / C-out-Push on the two cloud cards Kevin opens.
Local, run in tmux; reads logs_system1/t13v4/cloud/hosts.json["round2"] = {run: {machine, ssh: [...], dir}} (local, never
committed; the same format as "runs"). Per host, in order, stopping that host at the first failed check (recorded):

  1. code: git fetch / checkout system1-sonic / pull --ff-only in <S1>/duet-system1; the cloud HEAD must equal the local HEAD;
  2. SONIC weights: system1.sonic_controller.DEFAULT_CHECKPOINT exists on the cloud and its sha256 equals the laptop's
     (nothing is copied: a missing file is reported to Kevin);
  3. GPU idle (nvidia-smi: no compute process);
  4. smoke: system1/t13v4/cloud_smoke.py --run R --num-envs 4096 --iterations 10 (nohup; polled until smoke.json);
  5. launch (only with --launch and a passed smoke): nohup cloud_runs.py --run R --num-envs 4096 --gpu 0 (runner PID file), as
     round 1 (system1/t13v4/cloud_launch.py).
After both runs are launched: hosts.json["runs"] (round 1) is kept as hosts.json["round1"] and hosts.json["runs"] becomes the
round-2 runs, so that status_hourly.py and post_run_watch.py follow round 2; their cron lines are installed (if absent).
Record: logs_system1/t13v4/cloud/round2_launch.json.

    python3 system1/t13v4/round2_launch.py            # checks + smokes only
    python3 system1/t13v4/round2_launch.py --launch   # checks + smokes, then launch the runs whose smoke passed
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
HOSTS = ROOT / "logs_system1/t13v4/cloud/hosts.json"
REC = ROOT / "logs_system1/t13v4/cloud/round2_launch.json"
S1 = "/root/autodl-tmp/system1"
L = f"{S1}/logs/t13v4"
RUNS = f"{L}/runs"
NY = ZoneInfo("America/New_York")
CRON = ["0 * * * * cd /home/kevin/duet-system1 && /usr/bin/python3 system1/t13v4/status_hourly.py >> logs_system1/t13v4/cloud/status_cron.log 2>&1",
        "*/10 * * * * cd /home/kevin/duet-system1 && /usr/bin/python3 system1/t13v4/post_run_watch.py >> logs_system1/t13v4/post_run/watch_cron.log 2>&1"]
ROUND2 = ("c_out_table", "c_out_push")
lock = threading.Lock()


def now():
    return datetime.now(NY).isoformat(timespec="seconds")


def ssh(spec, cmd, timeout=300):
    p = subprocess.run([*spec[:-1], "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", spec[-1], f"bash -lc {shlex.quote(cmd)}"],
                       capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout.strip(), p.stderr.strip()[-500:]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def one(run, h, rec, local_head, sonic_sha, launch):
    r = rec["runs"][run] = dict(machine=h.get("machine"), started=now(), steps=[])
    save = lambda: (REC.write_text(json.dumps(rec, indent=1) + "\n"))  # noqa: E731

    def step(name, ok, **kw):
        with lock:
            r["steps"].append(dict(name=name, ok=bool(ok), time=now(), **kw))
            if not ok:
                r["stopped"] = name
            save()
        return ok
    env = f"source {S1}/s1_env.sh; cd {S1}/duet-system1"
    pull = "git fetch -q origin && git checkout -q system1-sonic && git pull -q --ff-only origin system1-sonic"
    rc, out, err = ssh(h["ssh"], f"{env}; ({pull}) || (sleep 10; git -c http.version=HTTP/1.1 fetch -q origin && git checkout -q system1-sonic && "
                                 f"git -c http.version=HTTP/1.1 pull -q --ff-only origin system1-sonic); git rev-parse HEAD")  # one HTTP/1.1 retry (try 1: curl 16 HTTP2)
    head = out.splitlines()[-1] if out else ""
    if not step("code", rc == 0 and head == local_head, cloud_head=head, local_head=local_head, rc=rc, err=err):
        return
    py = ("import hashlib,sys;sys.path.insert(0,'.');from system1 import sonic_controller as sc;p=sc.DEFAULT_CHECKPOINT;"
          "print(p);print(hashlib.sha256(open(p,'rb').read()).hexdigest() if p.exists() else 'MISSING')")
    rc, out, err = ssh(h["ssh"], f"{env}; $S1_PY -c {shlex.quote(py)}")  # the cloud's hssim_s1 (system python3 has no torch)
    lines = out.splitlines()
    if not step("sonic_weights", rc == 0 and len(lines) >= 2 and lines[-1] == sonic_sha, path=lines[0] if lines else None,
                cloud_sha256=lines[-1] if lines else None, local_sha256=sonic_sha, rc=rc, err=err):
        return
    rc, out, err = ssh(h["ssh"], "nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l; nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader")
    lines = out.splitlines()
    if not step("gpu_idle", rc == 0 and lines and lines[0].strip() == "0", nvidia=lines[1:], rc=rc):
        return
    sm_dir = f"{L}/smoke_cloud_r2"
    rc, out, err = ssh(h["ssh"], f"{env}; mkdir -p {sm_dir}; nohup python3 system1/t13v4/cloud_smoke.py --run {run} --num-envs 4096 --iterations 10 "
                                 f"--out {sm_dir} > {sm_dir}/{run}.log 2>&1 < /dev/null & echo $! > {sm_dir}/{run}.pid; cat {sm_dir}/{run}.pid")
    if not step("smoke_started", rc == 0, pid=out, err=err):
        return
    sm = None
    for _ in range(240):  # up to 2 h
        time.sleep(30)
        rc, out, _ = ssh(h["ssh"], f"cat {sm_dir}/{run}/smoke.json 2>/dev/null; kill -0 $(cat {sm_dir}/{run}.pid) 2>/dev/null && echo ALIVE")
        if out.startswith("{"):
            sm = json.loads(out[:out.rindex("}") + 1])
            break
        if "ALIVE" not in out:
            break
    if sm is None:
        _, tail, _ = ssh(h["ssh"], f"tail -20 {sm_dir}/{run}.log")
        step("smoke", False, note="no smoke.json (smoke process ended or 2 h passed)", log_tail=tail)
        return
    st = [dict(name=s["name"], rc=s["returncode"], seconds=s["seconds"], s_per_it=s["s_per_it_median_after_3"], overflow=len(s["overflow_lines"]),
               nonfinite=s["nonfinite_keys"]) for s in sm["stages"]]
    if not step("smoke", sm.get("passed"), stages=st, validation={k: v for k, v in (sm.get("validation") or {}).items() if k != "argv"}):
        return
    if not launch:
        return
    cmd = (f"{env}; mkdir -p {RUNS}; nohup python3 system1/t13v4/cloud_runs.py --run {run} --num-envs 4096 --gpu 0 --out {RUNS} "
           f"> {RUNS}/{run}.runner.log 2>&1 < /dev/null &\nP=$!\necho $P > {RUNS}/{run}.runner.pid\nsleep 5; kill -0 $P && echo RUNNER_ALIVE $P")
    rc, out, err = ssh(h["ssh"], cmd)
    step("launch", rc == 0 and "RUNNER_ALIVE" in out, output=out, err=err, command=cmd.replace(env, "<env>"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--launch", action="store_true")
    a = ap.parse_args()
    hosts = json.loads(HOSTS.read_text())
    r2 = hosts.get("round2") or {}
    if sorted(r2) != sorted(ROUND2):
        raise SystemExit(f"hosts.json round2 must name {ROUND2}, has {sorted(r2)}")
    local_head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    src = (ROOT / "system1/sonic_controller.py").read_text()  # the laptop's SONIC root (default of S1_SONIC_ROOT), without importing torch
    ck = Path(next(x for x in src.splitlines() if x.startswith("SONIC_ROOT")).split('"')[-2]) / "sonic_release/last.pt"
    rec = dict(started=now(), local_head=local_head, local_sonic_checkpoint=str(ck), local_sonic_sha256=sha(ck), launch=a.launch, runs={})
    ts = [threading.Thread(target=one, args=(run, r2[run], rec, local_head, rec["local_sonic_sha256"], a.launch)) for run in ROUND2]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    launched = [run for run, v in rec["runs"].items() if any(s["name"] == "launch" and s["ok"] for s in v["steps"])]
    rec.update(ended=now(), launched=launched)
    if a.launch and sorted(launched) == sorted(ROUND2):
        hosts = json.loads(HOSTS.read_text())
        hosts.setdefault("round1", hosts.get("runs"))
        hosts["runs"] = r2
        HOSTS.write_text(json.dumps(hosts, indent=1) + "\n")
        cur = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
        lines = cur.stdout.splitlines() if cur.returncode == 0 else []
        subprocess.run(["crontab", "-"], input="\n".join(lines + [c for c in CRON if c not in lines]) + "\n", text=True, check=True)
        rec["hosts_switched_to_round2"] = True
        rec["crontab"] = subprocess.run(["crontab", "-l"], capture_output=True, text=True).stdout
    REC.write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k != "crontab"}, indent=1)[-3000:])
    return 0 if (not a.launch or sorted(launched) == sorted(ROUND2)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
