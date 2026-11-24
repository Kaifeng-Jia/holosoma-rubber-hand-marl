#!/usr/bin/env python3
"""T14 round 3 orchestration (section 8 items 1-2), run inside tmux after run_r3.sh was started:
  1. wait for the round-3 solves (logs_system1/t14/r3/run.log ALL_DONE);
  2. evaluate all five tasks (finish scripts, STAGE=eval, at most 3 at once);
  3. step 2 where the round-3 pooled contact retention is below 0.99: CORE4D levels (both persons), Push / Pull the single
     solve (its levels are re-made from it); solves via run_r3_step2.sh (6 at once);
  4. meanwhile render the round-3 grids and videos (STAGE=render, 2 at once);
  5. evaluate and render the step-2 levels.
Every decision is written to logs_system1/t14/r3/pipeline.json.

    python3 system1/t14/r3_pipeline.py
"""
from __future__ import annotations

import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
S = ROOT / "logs_system1/t14/r3"
OUT = ROOT / "results_system1/T14_retargeting/step3"
CORE4D, SINGLE = ("table", "bin", "chair"), ("push", "pull")
KS = ("0.8", "0.9", "1.0", "1.1", "1.2")
THRESH = 0.99
log = dict(started=time.strftime("%F %T"))


def save():
    (S / "pipeline.json").write_text(json.dumps(log, indent=1) + "\n")


def sh(cmd, logfile, env=None):
    import os
    e = dict(os.environ, **(env or {}))
    with open(logfile, "w") as f:
        r = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=e)
    return r.returncode


def finish(task, stage, tag="", ks=()):
    script = "system1/t14/finish_r3.sh" if task in CORE4D else "system1/t14/finish_r3_single.sh"
    args = ["bash", script, task, tag] + list(ks)
    rc = sh(args, S / f"fin_{task}{tag}_{stage}.log", env=dict(STAGE=stage))
    log.setdefault("finish", []).append(dict(task=task, tag=tag, stage=stage, rc=rc, at=time.strftime("%T")))
    save()
    return rc


def main():
    while "ALL_DONE" not in (S / "run.log").read_text():
        time.sleep(15)
    log["solves_done"] = time.strftime("%T")
    save()
    with ThreadPoolExecutor(3) as ex:
        list(ex.map(lambda t: finish(t, "eval"), CORE4D + SINGLE))
    step2 = []
    for t in CORE4D:
        refs = json.loads((OUT / t / "eval_r3.json").read_text())["refs"]
        for k in KS:
            r = refs.get(f"r3_k{k}")
            if r is None:  # a person's solve failed: no round-3 level; not a retention decision
                log.setdefault("no_level", []).append(f"{t}:{k}")
                continue
            if r["contact_retention"]["pooled"] < THRESH:
                step2.append(f"{t}:{k}")
    for t in SINGLE:
        ev = json.loads((OUT / t / "eval_r3.json").read_text())
        lv = [r for n, r in ev["refs"].items() if n.startswith("r3_m")]
        if any(r["contact_retention"]["pooled"] < THRESH for r in lv):
            step2.append(f"{t}:single")
    log["step2"] = step2
    save()
    with ThreadPoolExecutor(3) as ex:
        fut = ex.submit(sh, ["bash", "system1/t14/run_r3_step2.sh"] + step2, S / "a10_queue.log", dict(P="6")) if step2 else None
        renders = ThreadPoolExecutor(2)
        list(renders.map(lambda t: finish(t, "render"), CORE4D + SINGLE))
        renders.shutdown()
        if fut is not None:
            log["step2_queue_rc"] = fut.result()
            save()
    for t in CORE4D:
        ks = [x.split(":")[1] for x in step2 if x.startswith(t + ":")]
        if ks:
            finish(t, "all", "_a10", ks)
    for t in SINGLE:
        if f"{t}:single" in step2:
            finish(t, "all", "_a10")
    log["done"] = time.strftime("%T")
    save()
    print("PIPELINE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
