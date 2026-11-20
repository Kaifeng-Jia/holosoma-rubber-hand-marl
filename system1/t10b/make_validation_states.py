#!/usr/bin/env python3
"""T10b B: validation initial states for the pilot retraining (checkpoint monitoring only; the frozen test states and MuJoCo results
are never used to pick a checkpoint or decide when to stop). Same generator as the frozen +/-2 test files
(scripts/paper_evaluation_utils.make_states: nominal + 50 states, independent uniform per-robot world XY within +/-2 cm and yaw
within +/-2 deg), another seed (20261120 instead of 241003). Checked: no validation state equals any state of the frozen +/-2 and
+/-8 test files of the task. Writes system1/t10b/validation_states/<task>_validation_states.json and prints the sha256.

    python3 system1/t10b/make_validation_states.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from paper_evaluation_utils import make_states  # noqa: E402

SEED = 20261120
TEST = dict(table=["/home/kevin/holosoma-core4d-base/logs/PaperSupplement20260929/unified_evaluation_20261003/table_states.json",
                   "/home/kevin/holosoma-core4d-base/logs/V3_20261102/cloud_backup_20261104/raw/v3_extended_eval_20261104_ops/states/table_pm8.json"],
            bucket=["/home/kevin/holosoma-core4d-base/logs/PaperSupplement20260929/unified_evaluation_20261003/bucket_states.json",
                    "/home/kevin/holosoma-core4d-base/logs/V3_20261102/cloud_backup_20261104/raw/v3_extended_eval_20261104_ops/states/bucket_pm8.json"])


def key(s):
    return json.dumps([s["robot_xy_m"], s["robot_yaw_rad"]])


def main():
    out = ROOT / "system1/t10b/validation_states"
    out.mkdir(parents=True, exist_ok=True)
    for task, tests in TEST.items():
        rec = make_states(task, seed=SEED)
        rec["purpose"] = "T10b pilot retraining: validation states for checkpoint monitoring only (not the frozen test states)"
        test_keys = set()
        for t in tests:
            test_keys |= {key(s) for s in json.loads(Path(t).read_text())["states"] if s["kind"] != "nominal"}
        val = [s for s in rec["states"] if s["kind"] != "nominal"]
        dup = [s["state_id"] for s in val if key(s) in test_keys]
        if dup or len(val) != 50:
            raise SystemExit(f"{task}: duplicates with the test states {dup} or wrong count {len(val)}")
        p = out / f"{task}_validation_states.json"
        p.write_text(json.dumps(rec, indent=1) + "\n")
        print(task, p.relative_to(ROOT), hashlib.sha256(p.read_bytes()).hexdigest(), "perturbed", len(val), "duplicates with test files", len(dup))


if __name__ == "__main__":
    main()
