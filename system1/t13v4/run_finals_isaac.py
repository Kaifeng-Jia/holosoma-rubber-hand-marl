#!/usr/bin/env python3
"""T13 v4 section 5 item 2, Isaac side: final evaluation of a finished run on the laptop 5070 (d_final_5070 protocol).

The argv is the run's own last training-time validation argv (validation/it*.launch.json, written by cloud_runs.py on the cloud:
same evaluator, --case, --controller policy, --obs-v1 --critic-v1, reference, placement proof), with only the interpreter
(local hssim_s1), --checkpoint (the copied-back final checkpoint, decision.json final_checkpoint_iteration), --states (the frozen
+/-2 / +/-8 test states of system1/t13v4/states_t14r6C, sha256 recorded), --model-id and --output-dir replaced.
Deterministic actions, randomization off, observation noise off (the evaluator forces noise off for --obs-v1).
Two processes at a time (taskset 16-23 / 24-31, as the T10b finals); the second starts --stagger seconds after the first
(two evaluators starting together raced on Isaac's URDF -> USD conversion cache, 2026-11-25 06:45). Records <name>.launch.json /
.exit.json (return code plus episodes written vs states in the file: the evaluator can return 0 after an Isaac fatal error);
an existing output directory is never reused.

    python3 system1/t13v4/run_finals_isaac.py --run d_push --out logs_system1/t13v4/finals
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
NY = ZoneInfo("America/New_York")
PY = "/home/kevin/.holosoma_deps/miniconda3/envs/hssim_s1/bin/python"
SLOTS = ("16-23", "24-31")
STATES = ROOT / "system1/t13v4/states_t14r6C"
STATES_RUN = dict(d_frontback=ROOT / "system1/t14/frontback_train/states", d_frontback_smoke=ROOT / "system1/t14/frontback_train/states")
CASE = {"d_push": "push", "d_table": "table", "d_table_pa": "table", "c_out_table": "table", "c_out_push": "push"}  # round 2: c_out_*
CASE.update(d_frontback="table", d_frontback_smoke="table")  # T14 front-back training task (Kevin 2026-11-26); _smoke: pre-training T10 gate
CASE.update(d_push_to10k="push", d_table_to10k="table", c_out_push_to10k="push")  # T13 section 9 continued runs (Kevin 2026-11-26 fixed budget, cloud_runs CONT)


def now():
    return datetime.now(NY).isoformat(timespec="seconds")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def replace(argv, flag, value):
    i = argv.index(flag)
    argv[i + 1] = str(value)


def build(run, level, out, slot, noise="off"):
    rd = ROOT / "logs_system1/t13v4/runs" / run
    dec = json.loads((rd / "decision.json").read_text())
    ck = rd / "train" / f"model_{dec['final_checkpoint_iteration']:05d}.pt"
    val = sorted(rd.glob("validation/it*.launch.json"))[-1]
    argv = list(json.loads(val.read_text())["argv"])
    assert argv[2] == "scripts/evaluate_paper_supplement.py" and "--obs-v1" in argv, argv
    argv[0] = PY
    states = STATES_RUN.get(run, STATES) / f"{CASE[run]}_pm{level}_states.json"
    name = f"{run}_pm{level}" + ("_noise_on" if noise == "on" else "")
    replace(argv, "--checkpoint", ck)
    replace(argv, "--states", states)
    replace(argv, "--model-id", f"t13v4_final__{name}")
    replace(argv, "--output-dir", out / name)
    if noise == "on":  # section 5 item 2, second group: observation-v1 training noise on (evaluator --obs-v1-noise on), all else equal
        argv += ["--obs-v1-noise", "on"]
    argv = ["taskset", "-c", slot] + argv
    rec = dict(name=name, run=run, level=level, kind=f"final_isaac_noise_{noise}", protocol=f"d_final_5070 (deterministic, randomization off, noise {noise})",
               source_validation_launch=str(val.relative_to(ROOT)), checkpoint=str(ck.relative_to(ROOT)), checkpoint_sha256=sha(ck),
               counted_iteration=dec.get("final_checkpoint_counted_iteration"), states=str(states.relative_to(ROOT)), states_sha256=sha(states),
               git_head=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(), argv=argv)
    return name, argv, rec


def launch(name, argv, rec, out):
    d = out / name
    assert not d.exists(), f"output exists: {d}"
    rec["started"] = now()
    (out / f"{name}.launch.json").write_text(json.dumps(rec, indent=1) + "\n")
    with open(out / f"{name}.stdout.log", "w") as f:
        p = subprocess.Popen(argv, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        rec["pid"] = p.pid
        (out / f"{name}.launch.json").write_text(json.dumps(rec, indent=1) + "\n")
        rc = p.wait()
    ep = d / "episodes.json"
    n_ep = len(json.loads(ep.read_text())) if ep.exists() else 0
    n_st = len(json.loads((ROOT / rec["states"]).read_text())["states"])
    ok = rc == 0 and n_ep == n_st and (d / "summary.json").exists()
    (out / f"{name}.exit.json").write_text(json.dumps(dict(name=name, rc=rc, episodes=n_ep, states=n_st, summary=(d / "summary.json").exists(),
                                                           complete=ok, ended=now()), indent=1) + "\n")
    return 0 if ok else (rc or 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, choices=sorted(CASE))
    ap.add_argument("--out", type=Path, default=ROOT / "logs_system1/t13v4/finals")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--levels", default="2,8")
    ap.add_argument("--stagger", type=float, default=90.0)
    ap.add_argument("--noise", choices=("off", "on"), default="off")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    jobs = [build(a.run, int(lv), a.out, SLOTS[0 if int(lv) == 2 else 1], a.noise) for lv in a.levels.split(",")]  # +/-2 on 16-23, +/-8 on 24-31
    if a.dry_run:
        for j in jobs:
            print(json.dumps(j[2], indent=1))
        return 0
    rcs = [None] * len(jobs)
    ts = [threading.Thread(target=lambda i=i, j=j: rcs.__setitem__(i, launch(*j, a.out))) for i, j in enumerate(jobs)]
    for i, t in enumerate(ts):
        if i:
            time.sleep(a.stagger)
        t.start()
    for t in ts:
        t.join()
    print("FINALS_ISAAC_DONE", a.run, rcs, flush=True)
    return 0 if all(r == 0 for r in rcs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
