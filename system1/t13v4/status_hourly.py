#!/usr/bin/env python3
"""T13 v4 section 6: hourly status of the cloud runs (local, unattended; cron, no Claude session needed).

For every run in the local hosts file (logs_system1/t13v4/cloud/hosts.json, never committed: it holds the ssh commands):
  {"runs": {"d_table": {"ssh": ["ssh", "-p", "PORT", "root@HOST"], "dir": "/root/autodl-tmp/system1/logs/t13v4/d_table"}, ...}}
a read-only collector runs on the cloud machine over ssh (python3 from stdin) and returns: last iteration, s/it (median of the last
50), the latest validation completions, state / decision of cloud_runs.py, whether the training PID is alive, non-finite metrics,
the per-robot ContactV1 diagnostics (mean of the last 50 iterations), GPU memory / utilisation. The table goes to
results_system1/T13_runs/status.md (one screen); it is committed and pushed to system1-sonic only when it changed (flock against
concurrent git use). Unreachable hosts are written as such; nothing on the cloud is ever changed.

    python3 system1/t13v4/status_hourly.py [--no-git]           # cron: 0 * * * *
    python3 system1/t13v4/status_hourly.py --local DIR --no-git  # test against a local run directory
"""
from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
HOSTS = ROOT / "logs_system1/t13v4/cloud/hosts.json"
OUT = ROOT / "results_system1/T13_runs/status.md"
NY = ZoneInfo("America/New_York")

COLLECT = r'''
import json, math, os, statistics, subprocess, sys
d = sys.argv[1]
r = dict(dir=d, exists=os.path.isdir(d))
def js(p):
    try:
        return json.load(open(os.path.join(d, p)))
    except Exception:
        return None
r["launch"] = {k: v for k, v in (js("launch.json") or {}).items() if k in ("run", "num_envs", "rules", "warmup_iterations", "started", "git_head")}
r["state"] = js("state.json")
r["decision"] = js("decision.json")
r["curve"] = js("validation_curve.json")
mdir = None
for sub in ("train_resume1", "train", "train_warmup"):
    if os.path.exists(os.path.join(d, sub, "metrics.jsonl")):
        mdir = sub
        break
r["metrics_dir"] = mdir
if mdir:
    lines = open(os.path.join(d, mdir, "metrics.jsonl")).read().splitlines()[-50:]
    ms = [json.loads(x) for x in lines if x.strip()]
    if ms:
        r["iteration"] = ms[-1].get("iteration")
        el = [m.get("elapsed_seconds") for m in ms if isinstance(m.get("elapsed_seconds"), (int, float))]
        r["s_per_it"] = statistics.median(el) if el else None
        bad = sorted({k for m in ms for k, v in m.items() if isinstance(v, float) and not math.isfinite(v)})
        r["nonfinite_keys_last50"] = bad
        ck = {}
        for m in ms:
            for k, v in m.items():
                if k.startswith("ContactV1/") and isinstance(v, (int, float)):
                    ck.setdefault(k, []).append(v)
        r["contact_v1_mean_last50"] = {k: sum(v) / len(v) for k, v in ck.items()}
        r["reward_mean_last"] = ms[-1].get("reward_mean")
st = r["state"] or {}
pid = st.get("train_pid") if not (st.get("resumed") or {}).get("pid") else st["resumed"]["pid"]
try:
    os.kill(int(pid), 0)
    r["train_alive"] = True
except PermissionError:
    r["train_alive"] = True
except Exception:
    r["train_alive"] = False if pid else None
try:
    r["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=20).stdout.strip()
except Exception as e:
    r["gpu"] = repr(e)
print(json.dumps(r))
'''


def collect(ssh, d):
    if ssh is None:
        p = subprocess.run([sys.executable, "-c", COLLECT, d], capture_output=True, text=True, timeout=120)
    else:
        p = subprocess.run([*ssh, "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "python3", "-", d], input=COLLECT,
                           capture_output=True, text=True, timeout=180)
    if p.returncode != 0:
        return dict(error=(p.stderr or p.stdout).strip()[-300:])
    return json.loads(p.stdout.strip().splitlines()[-1])


def row(name, r):
    if "error" in r:
        return f"| {name} | 连不上：{r['error'][:80]} | | | | | |"
    rules = (r.get("launch") or {}).get("rules") or {}
    warm = (r.get("launch") or {}).get("warmup_iterations") or 0
    it = r.get("iteration")
    counted = None if it is None else (it - warm if r.get("metrics_dir") != "train_warmup" else 0)
    eta = "—"
    if counted is not None and r.get("s_per_it") and rules.get("max_it") and not r.get("decision"):
        sec = max(rules["max_it"] - counted, 0) * r["s_per_it"]
        eta = datetime.fromtimestamp(datetime.now().timestamp() + sec, NY).strftime("%m-%d %H:%M")
    curve = r.get("curve") or {}
    last_val = f"{max(curve, key=int)}: {curve[max(curve, key=int)]}" if curve else "—"
    st = r.get("state") or {}
    flags = []
    if r.get("nonfinite_keys_last50"):
        flags.append("非有限值 " + ",".join(r["nonfinite_keys_last50"][:3]))
    if st.get("crashes"):
        flags.append(f"崩溃 {len(st['crashes'])} 次")
    if st.get("stopped"):
        flags.append("停止规则已触发")
    if r.get("decision"):
        flags.append("已结束：" + str(r["decision"].get("status")))
    elif r.get("train_alive") is False:
        flags.append("训练进程不在")
    c = r.get("contact_v1_mean_last50") or {}
    f = lambda k: "—" if c.get(k) is None else f"{100 * c[k]:.0f}%"  # noqa: E731
    contact = f"{f('ContactV1/r0_force_over_1n_in_should')} / {f('ContactV1/r1_force_over_1n_in_should')}"
    sp = "—" if not r.get("s_per_it") else f"{r['s_per_it']:.2f}"
    return f"| {name} | {counted if counted is not None else '—'} / {rules.get('max_it', '—')} | {last_val} | {sp} | {eta} | {contact} | {'；'.join(flags) or '无'} |"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--local", default=None, help="test: one local run directory instead of the hosts file")
    ap.add_argument("--no-git", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    runs = {Path(a.local).name: dict(ssh=None, dir=a.local)} if a.local else json.loads(HOSTS.read_text())["runs"]
    res = {name: collect(v.get("ssh"), v["dir"]) for name, v in runs.items()}
    now = datetime.now(NY).strftime("%Y-%m-%d %H:%M")
    lines = [f"# T13 训练状态（{now} 纽约时间，每小时自动更新）", "",
             "| run | 迭代（预热后计数）/ 上限 | 最近一次验证（轮: 完成数） | s/轮 | 预计结束 | 应接触时手力 > 1 N（r0 / r1，近 50 轮） | 异常 |",
             "|---|---|---|---|---|---|---|"]
    lines += [row(n, r) for n, r in res.items()]
    gpus = {n: r.get("gpu") for n, r in res.items() if "error" not in r}
    lines += ["", "GPU：" + "；".join(f"{n} {g}" for n, g in gpus.items())]
    text = "\n".join(lines) + "\n"
    a.out.parent.mkdir(parents=True, exist_ok=True)
    old = a.out.read_text() if a.out.exists() else ""
    strip = lambda t: "\n".join(t.splitlines()[1:])  # noqa: E731  (the timestamp line alone is not a change)
    (a.out.parent / "status_raw.json").write_text(json.dumps(dict(time=now, runs=res), indent=1) + "\n")
    if strip(old) == strip(text):
        print("unchanged")
        return 0
    a.out.write_text(text)
    print(text)
    if a.no_git:
        return 0
    with open(ROOT / ".git/s1_status.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        g = lambda *x: subprocess.run(["git", "-C", str(ROOT), *x], capture_output=True, text=True)  # noqa: E731
        if g("rev-parse", "--abbrev-ref", "HEAD").stdout.strip() != "system1-sonic":
            return 1
        g("add", str(a.out.relative_to(ROOT)), str((a.out.parent / "status_raw.json").relative_to(ROOT)))
        c = g("commit", "-m", f"system1(T13 runs): hourly status {now}\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>",
              "--", str(a.out.relative_to(ROOT)), str((a.out.parent / "status_raw.json").relative_to(ROOT)))
        if c.returncode == 0:
            g("pull", "--no-edit", "--no-rebase")
            g("push", "origin", "system1-sonic")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
