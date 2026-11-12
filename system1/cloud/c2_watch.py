#!/usr/bin/env python3
"""Local watcher for the T03 C v2 round. Exits (printing the new events as JSON) at the first of:
  * a C v2 run (cin_/cout_ main runs, and C v2 mass runs once started) logs its first iteration >= 100, or crosses a
    multiple of 1000 (Kevin 2026-11-11: report after launch with s/it and the iteration-100 diagnostics, then every 1000);
  * c2_decisions.json changes (iteration-1000 early-stop decisions, iteration-2000 choice, mass runs started);
  * a C v2 segment writes an exit record; a non-finite value appears in a C v2 metrics file;
  * divergence sign (report only, never stops anything): action std rose in each of the last 5 diagnostic windows and
    the last window's mean reward_mean is below the window 500 iterations earlier; at most once per run per 1000-iteration block;
  * the cloud evaluation job list has finished (logs/evals/ops: one exit record per launch record, at least one);
  * numeric spike (report only): an iteration with reward_min < -1e3 or value_loss > 1e3 (cin_table iteration 295 had
    reward_min -3.6e12; C v1 Pull / Table 20 kg had them too); at most one event per run per 1000-iteration block;
  * the cloud is unreachable for --max-failures polls.
Already-seen events are listed in --seen and ignored, so it can be restarted.
Quiet events (C mass sweep launches and the iteration-100 / every-1000 events of mass-sweep runs) do not end the watcher: they are
kept in --quiet-pending and printed with the next event that does (2026-11-11: batch the six mass launches).

    python system1/cloud/c2_watch.py --seen logs_system1/post_train/c2_seen.json
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from system1.t03.post_train import SSH  # noqa: E402

NY = ZoneInfo("America/New_York")
REMOTE = r'''
import json, glob, os, re
T = "/root/autodl-tmp/system1/logs/train"
E = "/root/autodl-tmp/system1/logs/evals/ops"
runs = {}
for d in sorted(glob.glob(T + "/cin_*") + glob.glob(T + "/cout_*")):
    f = d + "/metrics.jsonl"
    if not os.path.exists(f):
        continue
    with open(f, "rb") as fh:
        fh.seek(max(0, os.path.getsize(f) - 3000000))
        lines = fh.read().decode(errors="replace").splitlines()[1:]
    its, diag, bad, spikes = [], [], False, []
    for l in lines:
        if re.search(r"\b(NaN|Infinity|-Infinity)\b", l):
            bad = True
        try:
            j = json.loads(l)
        except Exception:
            continue
        its.append((j["iteration"], j.get("reward_mean"), j.get("elapsed_seconds")))
        rmin, vl = j.get("reward_min"), j.get("value_loss")
        if (isinstance(rmin, (int, float)) and rmin < -1e3) or (isinstance(vl, (int, float)) and vl > 1e3):
            spikes.append([j["iteration"], rmin, j.get("reward_mean"), vl])
        if "system1_c_train_diag" in j:
            x = j["system1_c_train_diag"]
            diag.append(dict(it=j["iteration"], std=x["action_std_mean"], corr_mean=x["correction_abs_mean_rad"],
                             corr_p95=x["correction_abs_p95_rad"], corr_max=x["correction_abs_max_rad"],
                             clipped=x["frac_clipped"], a_gt1=x["frac_abs_a_gt_1"], fall=x["sonic_fall_rate_per_robot_step"],
                             term=[x["finished_terminated"], x["terminated_object_position"], x["terminated_object_orientation"],
                                   x["terminated_robot_torso"], x["finished_timeout"]]))
    last = its[-1][0] if its else 0
    w = [r for i, r, s in its if i > last - 100 and isinstance(r, (int, float))]
    sp = sorted(s for i, r, s in its[-200:] if isinstance(s, (int, float)))
    runs[os.path.basename(d)] = dict(last=last, nonfinite=bad, diag=diag[-8:], spikes=spikes[-50:], rew100=sum(w) / len(w) if w else None,
                                     rew_by_it={i: r for i, r, s in its[-700:]}, sec_it_median=sp[len(sp) // 2] if sp else None)
exits = {}
for f in glob.glob(T + "/ops/cin_*.exit.json") + glob.glob(T + "/ops/cout_*.exit.json"):
    try:
        exits[os.path.basename(f)[:-10]] = json.load(open(f)).get("completed")
    except Exception:
        pass
dec = json.load(open(T + "/c2_decisions.json")) if os.path.exists(T + "/c2_decisions.json") else None
nl, ne = len(glob.glob(E + "/*.launch.json")), len(glob.glob(E + "/*.exit.json"))
slots = json.load(open(T + "/c2_slots.json")) if os.path.exists(T + "/c2_slots.json") else None
print(json.dumps(dict(runs=runs, exits=exits, decisions=dec, evals=[nl, ne], slots=slots)))
'''


def now():
    return datetime.now(NY).isoformat(timespec="seconds")


def poll():
    r = subprocess.run(SSH + ["python3 -"], input=REMOTE, capture_output=True, text=True, timeout=180)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-300:])
    return json.loads(r.stdout.strip().splitlines()[-1])


def events(st):
    ev = set()
    for name, r in st["runs"].items():
        if r["last"] >= 100:
            ev.add(f"it100:{name}")
        for k in range(1000, r["last"] + 1, 1000):
            ev.add(f"it{k}:{name}")
        if r["nonfinite"]:
            ev.add(f"nonfinite:{name}")
        for it, *_ in r.get("spikes") or []:
            ev.add(f"spike:{name}:block{it // 1000}")
        d = r["diag"]
        if len(d) >= 6 and all(d[i]["std"] > d[i - 1]["std"] for i in range(len(d) - 5, len(d))):
            rb = {int(k): v for k, v in r["rew_by_it"].items() if isinstance(v, (int, float))}
            lt = d[-1]["it"]
            now_w = [rb[i] for i in range(lt - 99, lt + 1) if i in rb]
            old_w = [rb[i] for i in range(lt - 599, lt - 499) if i in rb]
            if now_w and old_w and sum(now_w) / len(now_w) < sum(old_w) / len(old_w):
                ev.add(f"divergence_sign:{name}:block{lt // 1000}")  # at most once per run per 1000-iteration block
    for label, ok in st["exits"].items():
        ev.add(f"exit:{label}:{'completed' if ok else 'not_completed'}")
    dec = st["decisions"] or {}
    for k, v in (dec.get("early_stop") or {}).items():
        ev.add(f"early_stop_decision:{k}:{'stop' if v.get('stop') else 'continue'}")
    if dec.get("choice"):
        ev.add(f"choice:{dec['choice'].get('kind')}")
    for rid in (dec.get("mass_started") or {}):
        ev.add(f"mass_started:{rid}")
    nl, ne = st["evals"]
    if nl and nl == ne:
        ev.add(f"evals_done:{ne}")
    sl = st.get("slots")
    if sl:  # Kevin 2026-11-11: report when the queue is empty while a GPU has a free slot (at most once per hour)
        q = sl["queue"]
        free = sum(v["free_slots"] for v in sl["gpus"].values() if not v.get("held_by_first_eval_list"))
        if not q["mass_pending"] and q["evals_pending"] == 0 and free > 0:
            ev.add(f"queue_empty:{sl['time'][:13]}")
    return ev


MAIN_RUNS = {f"{k}_{t}_seed721_{s}" for k in ("cin", "cout") for t, s in (("pull_m20kg", "joint8000"), ("pull_m20kg", "critic50"),
                                                                           ("table_m5kg", "fresh12000"), ("bucket_m1kg", "fresh12000"))}


def quiet(e):
    return (e.startswith("mass_started:") or (re.match(r"it\d+:", e) is not None and e.split(":", 1)[1] not in MAIN_RUNS)
            or (e.startswith("exit:") and e.endswith("_critic50:completed")))  # Pull critic-only segment ending as planned


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seen", type=Path, required=True)
    p.add_argument("--quiet-pending", type=Path, default=None, help="default: <seen dir>/c2_quiet_pending.json")
    p.add_argument("--interval", type=int, default=300)
    p.add_argument("--max-failures", type=int, default=6)
    a = p.parse_args()
    seen = set(json.loads(a.seen.read_text())) if a.seen.exists() else set()
    fails = 0
    while True:
        try:
            st = poll()
            fails = 0
        except Exception as e:  # noqa: BLE001
            fails += 1
            print(now(), "poll failed", fails, repr(e)[:200], flush=True)
            if fails >= a.max_failures:
                print(json.dumps(dict(time=now(), new_events=["cloud_unreachable"])), flush=True)
                return 3
            time.sleep(a.interval)
            continue
        new = sorted(events(st) - seen)
        qp = a.quiet_pending or a.seen.parent / "c2_quiet_pending.json"
        pend = json.loads(qp.read_text()) if qp.exists() else []
        loud = [e for e in new if not quiet(e)]
        if new and not loud:
            seen |= set(new)
            a.seen.write_text(json.dumps(sorted(seen), indent=1))
            qp.write_text(json.dumps(pend + [dict(time=now(), event=e) for e in new], indent=1))
            print(now(), "quiet", new, flush=True)
        elif new:
            new = [x["time"] + " " + x["event"] for x in pend] + new
            qp.write_text("[]")
            a.seen.parent.mkdir(parents=True, exist_ok=True)
            a.seen.write_text(json.dumps(sorted(seen | {e.split(" ")[-1] for e in new}), indent=1))
            brief = {n: dict(last=r["last"], sec_it=r["sec_it_median"], rew100=r["rew100"], diag=r["diag"][-1:] if r["diag"] else None,
                             spikes=r.get("spikes"))
                     for n, r in st["runs"].items()}
            print(json.dumps(dict(time=now(), new_events=new, runs=brief, evals=st["evals"], slots=st.get("slots")), indent=1), flush=True)
            return 0
        time.sleep(a.interval)


if __name__ == "__main__":
    raise SystemExit(main())
