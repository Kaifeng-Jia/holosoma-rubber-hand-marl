#!/usr/bin/env python3
"""T13 v4 section 4 B: cloud smoke of one run with the exact cloud_runs.py command lines.

The plan comes from `cloud_runs.py --dry-run` (same flags, same environment variables); the warm-up stage (d_push) runs as
planned, the main stage is shortened to --iterations N, then the last checkpoint is validated once with the cloud_runs.py
validation command (validation states, deterministic, noise off). Writes <out>/<run>/smoke.json: return codes, seconds, s/it
(median after iteration 3), non-finite metric keys, PhysX overflow lines, checkpoint sha256, validation completeness.

    S1_PY=<hssim python> python3 system1/t13v4/cloud_smoke.py --run d_push --num-envs 2048 --out <dir> [--iterations 10]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OVERFLOW = re.compile(r"overflow|increase.*(buffer|capacity|size)|exceed.*(buffer|capacity|stack)|collision stack|lost.*patch|physx (error|warning)", re.I)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def metrics_summary(path):
    ms = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()] if Path(path).exists() else []
    el = [m["elapsed_seconds"] for m in ms if isinstance(m.get("elapsed_seconds"), (int, float)) and m.get("iteration", 0) > 3]
    bad = sorted({k for m in ms for k, v in m.items() if isinstance(v, float) and not math.isfinite(v)})
    return dict(iterations=len(ms), s_per_it_median_after_3=statistics.median(el) if el else None, nonfinite_keys=bad)


def overflow_lines(log):
    return [x.strip()[:300] for x in Path(log).read_text(errors="replace").splitlines() if OVERFLOW.search(x)] if Path(log).exists() else []


def plan_of(a):
    argv = [sys.executable, str(ROOT / "system1/t13v4/cloud_runs.py"), "--run", a.run, "--num-envs", str(a.num_envs), "--gpu", a.gpu,
            "--out", str(a.out), "--python", a.python, "--dry-run"]
    if a.torch_threads:
        argv += ["--torch-threads", str(a.torch_threads)]
    return json.loads(subprocess.run(argv, capture_output=True, text=True, check=True).stdout)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--num-envs", type=int, required=True)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--python", default=os.environ.get("S1_PY"))
    ap.add_argument("--torch-threads", type=int, default=None)
    ap.add_argument("--iterations", type=int, default=10, help="main-stage iterations (the warm-up keeps its planned length)")
    a = ap.parse_args()
    plan = plan_of(a)
    d = a.out / a.run
    d.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpu, LOCAL_RANK=str(400 + int(a.gpu)), WORLD_SIZE="1")
    rec = dict(run=a.run, num_envs=a.num_envs, main_iterations=a.iterations, plan=plan, stages=[],
               git_head=subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
               git_status=subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True, text=True).stdout)
    ok = True
    for st in plan["stages"]:
        argv = list(st["argv"])
        if st["name"] == "train":
            argv[argv.index("--iterations") + 1] = str(a.iterations)
        out_dir = Path(argv[argv.index("--output-dir") + 1])
        log = d / f"{st['name']}.stdout.log"
        t0 = time.time()
        with open(log, "w") as f:
            rc = subprocess.run(argv, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
        cks = sorted(out_dir.glob("model_*.pt"))
        s = dict(name=st["name"], argv=argv, returncode=rc, seconds=round(time.time() - t0, 1), overflow_lines=overflow_lines(log),
                 last_checkpoint=str(cks[-1]) if cks else None, last_checkpoint_sha256=sha(cks[-1]) if cks else None,
                 **metrics_summary(out_dir / "metrics.jsonl"))
        rec["stages"].append(s)
        if rc != 0 or not cks or s["nonfinite_keys"]:
            ok = False
            break
    if ok:
        ck = rec["stages"][-1]["last_checkpoint"]
        out = d / "validation"
        argv = [a.python, "-B", "scripts/evaluate_paper_supplement.py", "--case", plan["task"], "--controller", "policy", *plan["validation_flags"],
                "--checkpoint", ck, "--device", "cuda:0", "--states", plan["validation_states"], "--model-id", f"t13v4_cloud_smoke__{a.run}",
                "--output-dir", str(out)]
        t0 = time.time()
        with open(d / "validation.stdout.log", "w") as f:
            rc = subprocess.run(argv, cwd=ROOT, env=dict(env, LOCAL_RANK=str(int(env["LOCAL_RANK"]) + 50)), stdout=f, stderr=subprocess.STDOUT).returncode
        summ = out / "summary.json"
        sm = json.loads(summ.read_text()) if summ.exists() else {}
        eps = sm.get("episodes", [])
        rec["validation"] = dict(argv=argv, returncode=rc, seconds=round(time.time() - t0, 1), complete=sm.get("complete"), episodes=len(eps),
                                 completed_non_nominal=sum(1 for e in eps if e.get("kind") != "nominal" and e.get("completed_reference")),
                                 states_sha256=sm.get("states_sha256"), overflow_lines=overflow_lines(d / "validation.stdout.log"))
        ok = rc == 0 and sm.get("complete") is True
    rec["passed"] = ok
    (d / "smoke.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(dict(run=a.run, passed=ok, stages=[(s["name"], s["returncode"], s["seconds"], s["s_per_it_median_after_3"], len(s["overflow_lines"]))
                                                         for s in rec["stages"]], validation=rec.get("validation", {}).get("complete"))))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
