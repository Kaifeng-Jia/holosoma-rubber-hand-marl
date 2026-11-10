#!/usr/bin/env python3
"""T02 v2 scheme B evaluation queue: tasks x variants x {+/-2, +/-8}, same frozen state files and evaluator as
scheme A (system1/scheme_a_queue.STATES, sha256 checked), --controller sonic_planner --b-variant V.
Launch/exit records with git head/status/diff; skips completed runs; never deletes.

    python system1/t02/run_b_queue.py --out logs_system1/T02/B --variants B0,B1-0,B1-2,B1-4
    python system1/t02/run_b_queue.py --out logs_system1/T02/B_mass --variants B0,B1-2 --mass pull:5,pull:10,...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from system1.scheme_a_queue import PUSH_PROOF, PY, STATES  # noqa: E402

NY = ZoneInfo("America/New_York")
TASKS = "push,pull,chair,bucket,table"   # Kick not applicable (T02 v2)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def git(*x):
    return subprocess.run(["git", "-C", str(ROOT), *x], capture_output=True, text=True).stdout


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--variants", default="B0,B1-0,B1-2,B1-4")
    p.add_argument("--cases", default=TASKS)
    p.add_argument("--levels", default="2,8")
    p.add_argument("--mass", default=None, help="T04 levels case:kg,... (default: original mass)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--cpus", default=None)
    p.add_argument("--limit", type=int, default=0, help="smoke test only")
    a = p.parse_args()
    out = a.out if a.out.is_absolute() else ROOT / a.out
    (out / "ops").mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, LOCAL_RANK=os.environ.get("LOCAL_RANK", "71"), WORLD_SIZE="1")
    jobs = []
    if a.mass:
        for cm in a.mass.split(","):
            c, m = cm.split(":")
            jobs += [(c, float(m), v) for v in a.variants.split(",")]
    else:
        jobs = [(c, None, v) for v in a.variants.split(",") for c in a.cases.split(",")]
    rc_all = 0
    for lvl in (int(x) for x in a.levels.split(",")):
        for case, mass, var in jobs:
            states, expect = STATES[(case, lvl)]
            if sha(states) != expect:
                raise RuntimeError(f"frozen state hash mismatch: {states}")
            tag = f"{case}_{var}" + ("" if mass is None else f"_m{mass:g}kg".replace(".", "p"))
            label = f"{tag}_pm{lvl}"
            ex = out / "ops" / f"{label}.exit.json"
            if ex.exists() and json.loads(ex.read_text()).get("completed"):
                continue
            argv = [PY, "-B", "scripts/evaluate_paper_supplement.py", "--case", case, "--controller", "sonic_planner",
                    "--b-variant", var, "--device", a.device, "--states", str(states), "--model-id", f"SchemeB_{label}",
                    "--output-dir", str(out / label)]
            if case == "push":
                argv += ["--push-placement-proof", str(PUSH_PROOF)]
            if mass is not None:
                argv += ["--object-mass-kg", f"{mass:g}"]
            if a.limit:
                argv += ["--limit", str(a.limit)]
            if a.cpus:
                argv = ["taskset", "-c", a.cpus] + argv
            launch = dict(label=label, argv=argv, started=datetime.now(NY).isoformat(timespec="seconds"), variant=var,
                          mass_kg=mass, states=str(states), states_sha256=expect, git_head=git("rev-parse", "HEAD").strip(),
                          git_status=git("status", "--porcelain"), git_diff=git("diff", "HEAD"))
            (out / "ops" / f"{label}.launch.json").write_text(json.dumps(launch, indent=1))
            with (out / "ops" / f"{label}.stdout.log").open("w") as log:
                rc = subprocess.run(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
            summ = out / label / "summary.json"
            completed = rc == 0 and summ.exists() and bool(json.loads(summ.read_text()).get("complete"))
            ex.write_text(json.dumps(dict(label=label, returncode=rc, completed=completed,
                                          ended=datetime.now(NY).isoformat(timespec="seconds")), indent=1))
            print(label, "rc", rc, "completed", completed, flush=True)
            if not completed:
                rc_all = 1
    return rc_all


if __name__ == "__main__":
    raise SystemExit(main())
