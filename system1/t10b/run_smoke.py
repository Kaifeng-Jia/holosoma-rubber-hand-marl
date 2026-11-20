#!/usr/bin/env python3
"""T10b B smoke test: 50 iterations of each randomised training configuration on the laptop 5070 (one after the other, shared
with the friction evaluations): D Table seed 721 (= paper D Table command + --system1-randomization) and C-in Bin seed 721
(= T03 C-in Bin command + --system1-randomization). 512 environments (memory of the 8 GB 5070 next to one evaluation); everything
else as the cloud commands (results_system1/T10b_friction_margin/retrain/configs). Checks: exit 0, 50 iterations logged, no
non-finite value in metrics.jsonl, randomisation samples written. Records launch / exit / check JSON.

    python3 system1/t10b/run_smoke.py --out logs_system1/t10b/smoke
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
PY = "/home/kevin/.holosoma_deps/miniconda3/envs/hssim_s1/bin/python"
NY = ZoneInfo("America/New_York")
COMMON = ["--reward-variant", "bucket_A_no_rel_no_height", "--steps-per-env", "24", "--seed", "721", "--save-interval", "50", "--system1-randomization"]
RUNS = dict(
    d_table_seed721=["--experiment", "smalltable5kg_A"],
    cin_bucket_seed721=["--experiment", "bucket003", "--sonic-residual", "--sonic-tf32", "--sonic-residual-kind", "in", "--c-train-diag"],
)


def now():
    return datetime.now(NY).isoformat(timespec="seconds")


def finite(x):
    if isinstance(x, (int, float)):
        return math.isfinite(x)
    if isinstance(x, dict):
        return all(finite(v) for v in x.values())
    if isinstance(x, list):
        return all(finite(v) for v in x)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--num-envs", type=int, default=512)
    ap.add_argument("--iterations", type=int, default=50)
    ap.add_argument("--only", default=",".join(RUNS))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    git = lambda *x: subprocess.run(["git", "-C", str(ROOT), *x], capture_output=True, text=True).stdout  # noqa: E731
    for k, (name, extra) in enumerate((n, RUNS[n]) for n in a.only.split(",")):
        out = a.out / name
        argv = ["taskset", "-c", "24-31", PY, "-B", "scripts/train_core4d_smalltable.py", *extra, *COMMON, "--num-envs", str(a.num_envs),
                "--iterations", str(a.iterations), "--output-dir", str(out)]
        env = dict(os.environ, LOCAL_RANK=str(310 + k), WORLD_SIZE="1")
        rec = dict(name=name, argv=argv, started=now(), env=dict(LOCAL_RANK=env["LOCAL_RANK"]), git_head=git("rev-parse", "HEAD").strip(),
                   git_status=git("status", "--porcelain"), git_diff=git("diff"))
        (a.out / f"{name}.launch.json").write_text(json.dumps(rec, indent=1))
        with (a.out / f"{name}.stdout.log").open("w") as log:
            rc = subprocess.run(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
        rows = [json.loads(x) for x in (out / "metrics.jsonl").read_text().splitlines() if x.strip()] if (out / "metrics.jsonl").exists() else []
        samples = json.loads((out / "randomization_samples.json").read_text()) if (out / "randomization_samples.json").exists() else {}
        chk = dict(name=name, returncode=rc, iterations_logged=len(rows), all_finite=all(finite(r) for r in rows), ended=now(),
                   sample_keys=sorted(samples))
        # recipe v2 (docs/system1/T10b_dr_ranges_zh.md): every randomised quantity read back and inside its range
        rc_ok = samples.get("startup_range_check", {}).get("all_ok", False)
        fr = samples.get("final_readback", {})
        chk.update(startup_range_check_all_ok=rc_ok, final_pd_ok=fr.get("pd_ok", False), final_push_ok=fr.get("push_ok", False),
                   passed=rc == 0 and len(rows) == a.iterations and chk["all_finite"] and bool(samples) and rc_ok
                   and fr.get("pd_ok", False) and fr.get("push_ok", False))
        (a.out / f"{name}.exit.json").write_text(json.dumps(chk, indent=1))
        print(json.dumps(chk), flush=True)


if __name__ == "__main__":
    main()
