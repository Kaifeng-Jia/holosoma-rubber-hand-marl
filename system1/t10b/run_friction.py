#!/usr/bin/env python3
"""T10b A runner: friction-margin evaluations on the laptop 5070 - the +/-2 final evaluation of every D (Push, Pull, Kick, Chair,
Bin, Table) and C-in (Pull, Table, Bin) policy, seeds 721-723, re-run with exactly its original argv plus --hand-friction F
(rubber-hand material static = dynamic = F; F = 0.75, 0.5). Order (task book): Bin, Table, Pull, Chair, Push, Kick.
Concurrency is read every loop from <out>/slots.txt (1 or 2; taskset 16-23 / 24-31 as the finals), so the GPU can be shared.
Records <name>.launch.json / .exit.json; an existing output directory is never reused.

    python3 system1/t10b/run_friction.py --out logs_system1/t10b/friction
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
NY = ZoneInfo("America/New_York")
SLOTS = ("16-23", "24-31")
LEVELS = (0.75, 0.5)
ORDER = ("bucket", "table", "pull", "chair", "push", "kick")


def now():
    return datetime.now(NY).isoformat(timespec="seconds")


def final_launch(scheme, task, seed):
    """The launch record of the laptop-5070 +/-2 final evaluation of this policy."""
    if scheme == "D" and task in ("push", "kick", "chair"):
        d = ROOT / "logs_system1/post_train/d_final_5070" / f"D_{task}_paper_seed{seed}"
        f = sorted(d.glob("it*_pm2.launch.json"))
        assert len(f) == 1, f
        return f[0]
    rows = list(csv.DictReader((ROOT / "results_system1/T03/c2_main/per_seed_summary.csv").open()))
    r = next(x for x in rows if x["variant"] == scheme and x["task"] == task and int(x["training_seed"]) == seed and x["perturbation_level_cm"] == "2")
    d = ROOT / r["eval_dir"]
    return d.with_name(d.name + ".launch.json")


def jobs(levels=LEVELS):
    out = []
    for task in ORDER:
        for scheme in ("D", "C-in"):
            if scheme == "C-in" and task not in ("pull", "table", "bucket"):
                continue
            for seed in (721, 722, 723):
                for f in levels:
                    out.append((task, scheme, seed, f))
    return out


def run_name(task, scheme, seed, f):
    return f"{scheme.replace('-', '').lower()}_{task}_seed{seed}_hf{int(round(f * 100)):03d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", default="", help="comma-separated run names to (re)run, e.g. cin_table_seed721_hf075 (default: all)")
    ap.add_argument("--levels", default="", help="comma-separated hand frictions instead of 0.75,0.5 (e.g. 1.0 for the reproducibility check of the finals; default: unchanged)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    ctl = a.out / "slots.txt"
    if not ctl.exists():
        ctl.write_text("1\n")
    git = lambda *x: subprocess.run(["git", "-C", str(ROOT), *x], capture_output=True, text=True).stdout  # noqa: E731
    levels = tuple(float(x) for x in a.levels.split(",")) if a.levels else LEVELS
    todo = [j for j in jobs(levels) if not a.only or run_name(*j) in a.only.split(",")]
    if a.only:
        assert len(todo) == len(a.only.split(",")), (a.only, todo)
    running = {}
    while todo or running:
        n = max(1, min(2, int(ctl.read_text().strip() or 1)))
        for slot in SLOTS[:n]:
            if slot in running or not todo:
                continue
            task, scheme, seed, f = todo.pop(0)
            name = run_name(task, scheme, seed, f)
            src = final_launch(scheme, task, seed)
            argv = list(json.loads(src.read_text())["argv"])
            if argv[0] == "taskset":
                argv = argv[3:]
            argv[argv.index("--output-dir") + 1] = str(a.out / name)
            argv[argv.index("--model-id") + 1] = "t10b_friction__" + name
            argv = ["taskset", "-c", slot, *argv, "--hand-friction", str(f)]
            rec = dict(name=name, task=task, scheme=scheme, seed=seed, hand_friction=f, source_final_launch=str(src.relative_to(ROOT)), argv=argv,
                       started=now(), git_head=git("rev-parse", "HEAD").strip(), git_status=git("status", "--porcelain"), git_diff=git("diff"))
            log = (a.out / f"{name}.stdout.log").open("w")
            p = subprocess.Popen(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            rec["pid"] = p.pid
            (a.out / f"{name}.launch.json").write_text(json.dumps(rec, indent=1))
            running[slot] = (name, p, log)
            print(now(), "start", name, "slot", slot, "pid", p.pid, flush=True)
            time.sleep(20)
        for slot, (name, p, log) in list(running.items()):
            if p.poll() is not None:
                log.close()
                s = a.out / name / "summary.json"
                ok = s.exists() and json.loads(s.read_text()).get("complete", False)
                (a.out / f"{name}.exit.json").write_text(json.dumps(dict(name=name, returncode=p.returncode, complete=ok, ended=now()), indent=1))
                print(now(), "end", name, "rc", p.returncode, "complete", ok, flush=True)
                del running[slot]
        time.sleep(15)
    print(now(), "all done", flush=True)


if __name__ == "__main__":
    main()
