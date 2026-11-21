#!/usr/bin/env python3
"""T10b B step 5, MuJoCo side: closed-loop evaluation of a pilot's last checkpoint with the T10 setting and protocol
(system1/sim2sim/mj_eval.py: the T10 Isaac scene export of the original seed-721 policy, the same frozen states, +/-2 and +/-8;
C-in with PD (i) isaac and (ii) official as T10), only the actor ONNX / checkpoint replaced (mj_eval --onnx / --checkpoint).
Optional --repro: the original policy's +/-2 (C-in: PD isaac) run again on the current code, to check the T10 numbers
the pilot is compared with. Each job single-threaded on its own core (taskset). Records <name>.launch.json / .exit.json.

    python3 system1/t10b/run_mj_finals.py --run cin_bucket_seed721_rand --checkpoint exports_system1/t10b_pilot/cin_bucket_seed721_rand/model_12000.pt \
        --onnx exports_system1/t10b/onnx/cin_bucket_seed721_rand.onnx --out logs_system1/t10b/mj_finals --repro
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
RUNS = {"cin_bucket_seed721_rand": ("C-in", "bucket", 721), "d_table_seed721_rand": ("D", "table", 721)}


def now():
    return datetime.now(NY).isoformat(timespec="seconds")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=tuple(RUNS), required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--first-core", type=int, default=0)
    ap.add_argument("--repro", action="store_true")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    scheme, task, seed = RUNS[a.run]
    pds = ("isaac",) if scheme == "D" else ("isaac", "official")
    todo = []
    for lvl in (2, 8):
        for pd in pds:
            name = f"{a.run}_pm{lvl}" + ("" if scheme == "D" else f"_pd{pd}")
            extra = ["--onnx", str(a.onnx.resolve()), "--checkpoint", str(a.checkpoint.resolve())]
            todo.append((name, lvl, pd, extra))
    if a.repro:
        todo.append((f"original_{scheme.replace('-', '').lower()}_{task}_seed{seed}_pm2" + ("" if scheme == "D" else "_pdisaac") + "_repro", 2, "isaac", []))
    for name, *_ in todo:
        if (a.out / name).exists():
            raise FileExistsError(a.out / name)
    git = lambda *x: subprocess.run(["git", "-C", str(ROOT), *x], capture_output=True, text=True).stdout  # noqa: E731
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    running = {}
    for i, (name, lvl, pd, extra) in enumerate(todo):
        core = a.first_core + i
        argv = ["taskset", "-c", str(core), PY, "-B", "system1/sim2sim/mj_eval.py", "--task", task, "--scheme", scheme, "--seed", str(seed),
                "--level", str(lvl), "--pd", pd, "--out", str(a.out / name), *extra]
        rec = dict(name=name, run=a.run, argv=argv, started=now(), git_head=git("rev-parse", "HEAD").strip(), git_status=git("status", "--porcelain"),
                   git_diff=git("diff"))
        log = (a.out / f"{name}.stdout.log").open("w")
        p = subprocess.Popen(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
        rec["pid"] = p.pid
        (a.out / f"{name}.launch.json").write_text(json.dumps(rec, indent=1))
        running[name] = (p, log)
        print(now(), "start", name, "core", core, "pid", p.pid, flush=True)
    while running:
        for name, (p, log) in list(running.items()):
            if p.poll() is not None:
                log.close()
                s = a.out / name / "summary.json"
                ok = s.exists() and json.loads(s.read_text()).get("complete", False)
                (a.out / f"{name}.exit.json").write_text(json.dumps(dict(name=name, returncode=p.returncode, complete=ok, ended=now()), indent=1))
                print(now(), "end", name, "rc", p.returncode, "complete", ok, flush=True)
                del running[name]
        time.sleep(5)
    print(now(), "all done", flush=True)


if __name__ == "__main__":
    main()
