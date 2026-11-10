#!/usr/bin/env python3
"""T04 direct evaluation of the original-mass paper D policy (seed 721) at other mass levels, no retraining.

Covers both T04 items: the lighter levels ("未重训") and the "额外评测" at every non-original level.
Same evaluator and protocol as the paper/V3 D evaluation (scripts/evaluate_paper_supplement.py
--controller policy) with --object-mass-kg <m> --checkpoint-mass original. Same frozen state files
as scheme A (system1/scheme_a_queue.STATES, sha256 checked). Launch/exit records per run; skips
completed runs; never deletes.

    python system1/t04/eval_d_direct.py --out logs_system1/T04/D_direct
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
from system1.scheme_a_queue import PY, STATES  # noqa: E402

NY = ZoneInfo("America/New_York")
B = Path("/home/kevin/holosoma-core4d-base/logs")
# paper main D, seed 721 (V3 closeout "Full" rows; sha256 = V3 checkpoint_sha256 prefix)
CHECKPOINTS = {
    "pull": (Path("/home/kevin/holosoma-rubber-hand-marl/logs/Plan5Pull/pull_20kg_full8000_from_critic50_seed721_env2048/model_08050.pt"),
             "Pull-baseline_seed721", "727630e9cec6"),
    "table": (B / "Core4DSmallTableA/A_no_rel_no_height_5kg_fresh12000_env2048_seed721_cloud_20260920/model_12000.pt",
              "T-07_seed721", "f163d11a9629"),
    "bucket": (B / "Core4DBucket/A_no_rel_no_height_fresh12000_env2048_seed721_cloud_20260919/model_12000.pt",
               "B-RZ0_seed721", "7ba039fbb608"),
}
ORIG = dict(pull=20.0, table=5.0, bucket=1.0)
RUNS = "pull:5,pull:10,pull:40,table:2.5,table:10,table:20,bucket:2,bucket:4,bucket:8"


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git(*x):
    return subprocess.run(["git", "-C", str(ROOT), *x], capture_output=True, text=True).stdout


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--runs", default=RUNS)
    p.add_argument("--levels", default="2,8")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--cpus", default=None)
    p.add_argument("--limit", type=int, default=0, help="smoke test only")
    a = p.parse_args()
    out = a.out if a.out.is_absolute() else ROOT / a.out
    (out / "ops").mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, LOCAL_RANK=os.environ.get("LOCAL_RANK", "61"), WORLD_SIZE="1")
    ck_sha = {}
    for case, (ck, _, prefix) in CHECKPOINTS.items():
        ck_sha[case] = sha(ck)
        if not ck_sha[case].startswith(prefix):
            raise RuntimeError(f"checkpoint sha mismatch {ck}")
    rc_all = 0
    for run in a.runs.split(","):
        case, m = run.split(":")
        m = float(m)
        if m == ORIG[case]:
            raise ValueError("original mass is the paper result; not re-evaluated here")
        tag = f"{case}_m{m:g}kg".replace(".", "p")
        for lvl in (int(x) for x in a.levels.split(",")):
            states, expect = STATES[(case, lvl)]
            if sha(states) != expect:
                raise RuntimeError(f"frozen state hash mismatch: {states}")
            label = f"{tag}_pm{lvl}"
            ex = out / "ops" / f"{label}.exit.json"
            summ = out / label / "summary.json"
            if ex.exists() and json.loads(ex.read_text()).get("completed"):
                continue
            ck, mid, _ = CHECKPOINTS[case]
            argv = [PY, "-B", "scripts/evaluate_paper_supplement.py", "--case", case, "--controller", "policy",
                    "--checkpoint", str(ck), "--device", a.device, "--states", str(states),
                    "--model-id", f"{mid}_m{m:g}kg_pm{lvl}_noretrain", "--output-dir", str(out / label),
                    "--object-mass-kg", f"{m:g}", "--checkpoint-mass", "original"]
            if a.limit:
                argv += ["--limit", str(a.limit)]
            if a.cpus:
                argv = ["taskset", "-c", a.cpus] + argv
            launch = dict(label=label, argv=argv, started=datetime.now(NY).isoformat(timespec="seconds"),
                          checkpoint=str(ck), checkpoint_sha256=ck_sha[case], model_id=mid, mass_kg=m,
                          original_mass_kg=ORIG[case], retrained=False, states=str(states), states_sha256=expect,
                          git_head=git("rev-parse", "HEAD").strip(), git_status=git("status", "--porcelain"),
                          git_diff=git("diff", "HEAD"))
            (out / "ops" / f"{label}.launch.json").write_text(json.dumps(launch, indent=1))
            with (out / "ops" / f"{label}.stdout.log").open("w") as log:
                rc = subprocess.run(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
            # evaluator hard-exits 0 even after an exception: success is a complete summary.json
            completed = rc == 0 and summ.exists() and bool(json.loads(summ.read_text()).get("complete"))
            ex.write_text(json.dumps(dict(label=label, returncode=rc, completed=completed,
                                          ended=datetime.now(NY).isoformat(timespec="seconds")), indent=1))
            print(label, "rc", rc, "completed", completed, flush=True)
            if not completed:
                rc_all = 1
    return rc_all


if __name__ == "__main__":
    raise SystemExit(main())
