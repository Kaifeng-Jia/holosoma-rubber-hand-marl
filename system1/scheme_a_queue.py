#!/usr/bin/env python3
"""Scheme A evaluation queue: 6 tasks x {+/-2, +/-8} frozen state sets, SONIC controller.

Uses scripts/evaluate_paper_supplement.py --controller sonic with the same frozen state
files as the paper/V3 evaluations (sha256 checked against the values in
docs/system1/codebase_map_zh.md). Each run writes launch/exit records. Never deletes;
skips runs whose exit record has returncode 0.

    python system1/scheme_a_queue.py --device cpu --out logs_system1/T01/schemeA_cpu [--cpus 8-31]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
PY = "/home/kevin/.holosoma_deps/miniconda3/envs/hssim_s1/bin/python"
B = Path("/home/kevin/holosoma-core4d-base/logs")
U = B / "PaperSupplement20260929/unified_evaluation_20261003"
X = B / "V3_20261102/cloud_backup_20261104/raw/v3_extended_eval_20261104_ops/states"
STATES = {
    ("push", 2): (B / "ExperimentsR2_evaluation_20261026_retry01/U5_push_registered/push_states.json", "1799ea1eec1a3ff097d46e08e6492694d6455490ccf9f2de26586e079fef895d"),
    ("pull", 2): (U / "pull_states.json", "473c1fc0563228a770393c2ccd78f3b051df258a408e6481d44c12fa2450c572"),
    ("kick", 2): (B / "ExperimentsR2_evaluation_20261026_retry01/U6_kick/kick_states.json", "c1ac57c62345363c85f96f98363d778898a23e1f35a2596fbcd33a68367527c4"),
    ("chair", 2): (U / "chair_states.json", "5bb8ff9f1cb7adb72b784c0d8475779cf68099aba551901114a2f78625960bc3"),
    ("bucket", 2): (U / "bucket_states.json", "631026d39f436ceca4008e2f5e684fff16102fba8895fafd23ae3639b575d71f"),
    ("table", 2): (U / "table_states.json", "389e8da5fb818572a48cbba049254585311079439265f5ab4cfc1523a185dac1"),
    ("push", 8): (X / "push_pm8.json", "a58b2fae20f488c082f1ac07c7d552cc85053e6d5bc8360a60df2a723d26c266"),
    ("pull", 8): (X / "pull_pm8.json", "3c2a60c7982db92b59b1f3343103e711409a6c756faaf83b86016b8163c8d5b2"),
    ("kick", 8): (X / "kick_pm8.json", "dba247caf71d5a80213f5eb96dd8c7a32ba0ff75d8fe2dd0f18411753c779319"),
    ("chair", 8): (X / "chair_pm8.json", "f9a144ed0738975fef4453047c48382cee383b7daaa680e4cf43b468e8ea5af7"),
    ("bucket", 8): (X / "bucket_pm8.json", "3bae05a1b1d716d97c7025386dc3d7f832a7203a665d33f4f9979b3adb6e6ff0"),
    ("table", 8): (X / "table_pm8.json", "6a626f48c14598a646d40d35694e32b5be198ea09491b7f3557ab9c1fe02fcad"),
}
PUSH_PROOF = ROOT / "system1/evidence/preflight_push_registered.json"
NY = ZoneInfo("America/New_York")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cpus", default=None, help="taskset core list, e.g. 8-31")
    p.add_argument("--cases", default="push,pull,kick,chair,bucket,table")
    p.add_argument("--levels", default="2,8")
    p.add_argument("--extra", default="", help='extra evaluator args as one string, e.g. --extra="--object-mass-kg 10"')
    a = p.parse_args()
    out = a.out if a.out.is_absolute() else ROOT / a.out
    (out / "ops").mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, LOCAL_RANK=os.environ.get("LOCAL_RANK", "41"), WORLD_SIZE="1")
    rc_all = 0
    for lvl in (int(x) for x in a.levels.split(",")):
        for case in a.cases.split(","):
            states, expect = STATES[(case, lvl)]
            if sha(states) != expect:
                raise RuntimeError(f"frozen state hash mismatch: {states}")
            label = f"{case}_pm{lvl}"
            ex = out / "ops" / f"{label}.exit.json"
            summ0 = out / label / "summary.json"
            if ex.exists() and (json.loads(ex.read_text()).get("completed")
                                or (summ0.exists() and json.loads(summ0.read_text()).get("complete"))):
                continue
            argv = [PY, "-B", "scripts/evaluate_paper_supplement.py", "--case", case, "--controller", "sonic",
                    "--device", a.device, "--states", str(states), "--model-id", f"SchemeA_{label}",
                    "--output-dir", str(out / label), *a.extra.split()]
            if case == "push":
                argv += ["--push-placement-proof", str(PUSH_PROOF)]
            if a.cpus:
                argv = ["taskset", "-c", a.cpus] + argv
            launch = dict(label=label, argv=argv, started=datetime.now(NY).isoformat(timespec="seconds"),
                          states=str(states), states_sha256=expect,
                          git_head=subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
                          git_status=subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True, text=True).stdout,
                          git_diff=subprocess.run(["git", "-C", str(ROOT), "diff", "HEAD"], capture_output=True, text=True).stdout)
            (out / "ops" / f"{label}.launch.json").write_text(json.dumps(launch, indent=1))
            with (out / "ops" / f"{label}.stdout.log").open("w") as log:
                rc = subprocess.run(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
            # The evaluator hard-exits with 0 from close_simulation_app even after an exception,
            # so success is a complete summary.json, not the return code alone.
            summ = out / label / "summary.json"
            completed = rc == 0 and summ.exists() and bool(json.loads(summ.read_text()).get("complete"))
            ex.write_text(json.dumps(dict(label=label, returncode=rc, completed=completed,
                                          ended=datetime.now(NY).isoformat(timespec="seconds")), indent=1))
            print(f"{label} rc={rc} completed={completed}", flush=True)
            rc_all = rc_all or (0 if completed else 1)
    return rc_all


if __name__ == "__main__":
    raise SystemExit(main())
