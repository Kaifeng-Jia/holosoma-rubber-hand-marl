#!/usr/bin/env python3
"""Cloud events while the laptop was off (Kevin 2026-11-12: Ubuntu off ~4 h; report afterwards what happened on the cloud).

The watcher (c2_watch.py) only sees the last ~3 MB of each metrics.jsonl and the latest diagnostic windows, so after a
long gap it can miss spikes and divergence signs. This script scans the full metrics of every C v2 run on the cloud.

    # before the shutdown: record the iteration of every run
    hssim_s1/bin/python -B system1/cloud/catchup_events.py baseline --out logs_system1/post_train/shutdown_baseline.json
    # after the restart: everything after the baseline
    hssim_s1/bin/python -B system1/cloud/catchup_events.py report --baseline logs_system1/post_train/shutdown_baseline.json

Report: per run, iterations logged since the baseline; spikes (reward_min < -1e3 or value_loss > 1e3, same rule as the
watcher) with iteration and values; divergence signs (watcher rule: action std rose in each of the last 5 diagnostic
windows and the 100-iteration mean reward is below that of the window 500 iterations earlier) at diagnostic windows after
the baseline, per 1000-iteration block; non-finite values; 1000-iteration milestones crossed; training exits; scheduler
log lines (evaluation starts / ends, deadline, extension, end) and decisions (all_training_done, curves_extension,
curves_end) after the baseline time. Read-only on the cloud.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from system1.t03.post_train import SSH  # noqa: E402

NY = ZoneInfo("America/New_York")
REMOTE = r'''
import json, glob, os, re, sys
base = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {"runs": {}, "time": None}
T = "/root/autodl-tmp/system1/logs/train"
out = {"runs": {}, "exits": [], "scheduler_log": [], "decisions": {}}
for d in sorted(glob.glob(T + "/cin_*") + glob.glob(T + "/cout_*")):
    f = d + "/metrics.jsonl"
    if not os.path.exists(f):
        continue
    name = os.path.basename(d)
    b = base["runs"].get(name, -1)
    rew, diag, spikes, nonfinite, last = {}, [], [], [], 0
    for l in open(f, errors="replace"):
        try:
            j = json.loads(l)
        except Exception:
            continue
        it = j.get("iteration")
        if it is None:
            continue
        last = max(last, it)
        if re.search(r"\b(NaN|Infinity|-Infinity)\b", l) and it > b:
            nonfinite.append(it)
        r, rmin, vl = j.get("reward_mean"), j.get("reward_min"), j.get("value_loss")
        if isinstance(r, (int, float)):
            rew[it] = r
        if it > b and ((isinstance(rmin, (int, float)) and rmin < -1e3) or (isinstance(vl, (int, float)) and vl > 1e3)):
            spikes.append(dict(it=it, reward_min=rmin, reward_mean=r, value_loss=vl, kl=j.get("kl")))
        x = j.get("system1_c_train_diag")
        if x:
            diag.append(dict(it=it, std=x["action_std_mean"], clipped=x["frac_clipped"], fall=x["sonic_fall_rate_per_robot_step"]))
    div = {}
    for i in range(5, len(diag)):
        if diag[i]["it"] <= b:
            continue
        if all(diag[k]["std"] > diag[k - 1]["std"] for k in range(i - 4, i + 1)):
            lt = diag[i]["it"]
            nw = [rew[k] for k in range(lt - 99, lt + 1) if k in rew]
            ow = [rew[k] for k in range(lt - 599, lt - 499) if k in rew]
            if nw and ow and sum(nw) / len(nw) < sum(ow) / len(ow):
                div.setdefault(lt // 1000, dict(it=lt, std=diag[i]["std"], reward_100=sum(nw) / len(nw), reward_100_500_earlier=sum(ow) / len(ow),
                                                clipped=diag[i]["clipped"], fall=diag[i]["fall"]))
    dl = diag[-1] if diag else {}
    w = [rew[k] for k in range(last - 99, last + 1) if k in rew]
    out["runs"][name] = dict(baseline=b, last=last, spikes=spikes, divergence_blocks=div, nonfinite=nonfinite[:20],
                             milestones=[k for k in range(1000, last + 1, 1000) if k > b], last_diag=dl,
                             reward_100=sum(w) / len(w) if w else None)
for f in sorted(glob.glob(T + "/ops/c*.exit.json")):
    try:
        j = json.load(open(f))
    except Exception:
        continue
    if base["time"] is None or (j.get("ended") or "") > base["time"]:
        out["exits"].append(dict(label=os.path.basename(f)[:-10], completed=j.get("completed"), ended=j.get("ended")))
if os.path.exists(T + "/c2_scheduler.log"):
    for l in open(T + "/c2_scheduler.log", errors="replace"):
        if base["time"] is None or l[:25] > base["time"]:
            out["scheduler_log"].append(l.rstrip()[:300])
dec = json.load(open(T + "/c2_decisions.json")) if os.path.exists(T + "/c2_decisions.json") else {}
out["decisions"] = {k: dec.get(k) for k in ("all_training_done", "curves_extension", "curves_end", "mass_stopped_at_deadline")}
out["seed_started"] = dec.get("seed_started")
print(json.dumps(out))
'''


def remote(base):
    arg = json.dumps(base).replace("'", "")
    r = subprocess.run(SSH + [f"python3 - '{arg}'"], input=REMOTE, capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-2000:])
    return json.loads(r.stdout)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("baseline", "report"))
    p.add_argument("--out", type=Path)
    p.add_argument("--baseline", type=Path)
    a = p.parse_args()
    if a.mode == "baseline":
        st = remote({"runs": {}, "time": None})
        b = dict(time=datetime.now(NY).isoformat(timespec="seconds"), runs={k: v["last"] for k, v in st["runs"].items()})
        a.out.write_text(json.dumps(b, indent=1))
        print(json.dumps(b, indent=1))
        return 0
    base = json.loads(a.baseline.read_text())
    st = remote(base)
    print("baseline", base["time"], "now", datetime.now(NY).isoformat(timespec="seconds"))
    for name, r in st["runs"].items():
        flags = []
        if r["spikes"]:
            flags.append(f"spikes {[(s['it'], s['reward_min'], s['value_loss']) for s in r['spikes']]}")
        if r["divergence_blocks"]:
            flags.append(f"divergence_sign blocks {sorted(int(k) for k in r['divergence_blocks'])}")
        if r["nonfinite"]:
            flags.append(f"NONFINITE {r['nonfinite']}")
        d = r["last_diag"]
        print(f"{name}: {r['baseline']} -> {r['last']}  milestones {r['milestones']}  std {d.get('std')}  clipped {d.get('clipped')}  "
              f"fall {d.get('fall')}  reward_100 {r['reward_100']}  " + "; ".join(flags))
    print("exits", json.dumps(st["exits"]))
    print("decisions", json.dumps(st["decisions"])[:2000])
    print("scheduler log lines", len(st["scheduler_log"]))
    for l in st["scheduler_log"]:
        print("  ", l)
    (ROOT / "logs_system1/post_train/catchup_report.json").write_text(json.dumps(st, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
