#!/usr/bin/env python3
"""T10 step 3 runner: every MuJoCo closed-loop evaluation of the task book (D: Table / Bin x seeds 721-723 x +/-2 / +/-8, paper setting;
C-in: the same x PD (i) isaac and (ii) official), at most --jobs at a time, each single-threaded on its own core (taskset). Records
<name>.launch.json (argv, git HEAD / status / diff) and <name>.exit.json next to each output; an existing output is never reused.

    python3 system1/sim2sim/run_mj_evals.py --out logs_system1/t10/mj_eval --jobs 16
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
PY = "/home/kevin/.holosoma_deps/miniconda3/envs/sim2sim_s1/bin/python"
NY = ZoneInfo("America/New_York")


def now():
    return datetime.now(NY).isoformat(timespec="seconds")


def jobs():
    out = []
    for task in ("table", "bucket"):
        for seed in (721, 722, 723):
            for lvl in (2, 8):
                out.append(("D", task, seed, lvl, "isaac"))
                out.append(("C-in", task, seed, lvl, "isaac"))
                out.append(("C-in", task, seed, lvl, "official"))
    out.sort(key=lambda j: (j[0] == "D", j))  # C-in (slower) first
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--first-core", type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    git = lambda *x: subprocess.run(["git", "-C", str(ROOT), *x], capture_output=True, text=True).stdout  # noqa: E731
    todo = jobs()
    free = list(range(a.first_core, a.first_core + a.jobs))
    running = {}
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    while todo or running:
        while todo and free:
            scheme, task, seed, lvl, pd = todo.pop(0)
            core = free.pop(0)
            name = f"{scheme.replace('-', '').lower()}_{task}_seed{seed}_pm{lvl}" + ("" if scheme == "D" else f"_pd{pd}")
            argv = ["taskset", "-c", str(core), PY, "-B", "system1/sim2sim/mj_eval.py", "--task", task, "--scheme", scheme, "--seed", str(seed),
                    "--level", str(lvl), "--pd", pd, "--out", str(a.out / name)]
            rec = dict(name=name, argv=argv, started=now(), git_head=git("rev-parse", "HEAD").strip(), git_status=git("status", "--porcelain"),
                       git_diff=git("diff"))
            log = (a.out / f"{name}.stdout.log").open("w")
            p = subprocess.Popen(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
            rec["pid"] = p.pid
            (a.out / f"{name}.launch.json").write_text(json.dumps(rec, indent=1))
            running[name] = (p, log, core)
            print(now(), "start", name, "core", core, "pid", p.pid, flush=True)
        for name, (p, log, core) in list(running.items()):
            if p.poll() is not None:
                log.close()
                s = a.out / name / "summary.json"
                ok = s.exists() and json.loads(s.read_text()).get("complete", False)
                (a.out / f"{name}.exit.json").write_text(json.dumps(dict(name=name, returncode=p.returncode, complete=ok, ended=now()), indent=1))
                print(now(), "end", name, "rc", p.returncode, "complete", ok, flush=True)
                free.append(core)
                del running[name]
        time.sleep(5)
    print(now(), "all done", flush=True)


if __name__ == "__main__":
    main()
