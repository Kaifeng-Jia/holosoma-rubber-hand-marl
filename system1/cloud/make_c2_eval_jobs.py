#!/usr/bin/env python3
"""Job list for system1/cloud/run_eval_jobs.py (T03 C v2, the freed cloud slots evaluate first). Runs on the cloud.

1. C v1 at stop, +/-2 (50 frozen states + nominal), TF32 off: the newest checkpoint of each of the 9 v1 runs (8 stopped
   2026-11-11 14:43 EST, ops/<label>.stopped.json; c_pull_m20kg joint8000 completed at 8050), at the run's own mass.
2. Constant-dq check (diagnostic, not a scheme): Pull +/-2 with scheme A's evaluation path and a constant dq on SONIC's input
   reference: hip_pitch (both legs) and waist_pitch toward more trunk flexion / lower hands, each at 0.5x and 1.0x of its
   C-in bound scale_i (frozen paper-D action scale); other joints 0. Direction: waist_pitch +, hip_pitch - (hip flexion;
   G1 crouch default hip_pitch -0.312 with knee +0.669; torso = pelvis * Ry(waist_pitch), pitch = waist_pitch).
   Read as: both joints together at 0.5x, both together at 1.0x; plus dq = 0 on the same GPU.

    $S1_PY system1/cloud/make_c2_eval_jobs.py --train $S1_ROOT/logs/train --out $S1_ROOT/logs/evals > jobs.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

V1 = ["c_pull_m20kg_seed721_joint8000", "c_table_m5kg_seed721_fresh12000", "c_bucket_m1kg_seed721_fresh12000",
      "c_pull_m40kg_seed721_joint8000", "c_table_m10kg_seed721_fresh12000", "c_table_m20kg_seed721_fresh12000",
      "c_bucket_m2kg_seed721_fresh12000", "c_bucket_m4kg_seed721_fresh12000", "c_bucket_m8kg_seed721_fresh12000"]
ORIG = dict(pull=20.0, table=5.0, bucket=1.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--states", type=Path, default=Path("/root/autodl-tmp/system1/inputs/states"))
    p.add_argument("--scales-from", type=Path, required=True, help="a C v2 run_config.json (system1_scheme_c joint_names / frozen_action_scales)")
    a = p.parse_args()
    jobs = []
    for label in V1:
        case = label.split("_")[1]
        mass = float(re.search(r"_m([\d.p]+)kg", label).group(1).replace("p", "."))
        cks = sorted((a.train / label).glob("model_*.pt"))
        ck = cks[-1]
        it = int(ck.stem.split("_")[1])
        name = f"c1stop_{label}_it{it:05d}_pm2"
        argv = ["scripts/evaluate_paper_supplement.py", "--case", case, "--controller", "sonic_residual", "--checkpoint", str(ck),
                "--states", str(a.states / f"{case}_states.json"), "--model-id", name, "--output-dir", str(a.out / "c1_stop" / name)]
        if mass != ORIG[case]:
            argv += ["--object-mass-kg", f"{mass:g}"]
        stopped = a.train / "ops" / f"{label}.stopped.json"
        jobs.append(dict(name=name, argv=argv, meta=dict(kind="c1_stop", run=label, case=case, mass_kg=mass, iteration=it,
                                                          stopped_record=str(stopped) if stopped.exists() else None)))
    sc = json.loads(a.scales_from.read_text())["system1_scheme_c"]
    scale = dict(zip(sc["joint_names"], sc["frozen_action_scales"]))
    joints = {"left_hip_pitch_joint": -1.0, "right_hip_pitch_joint": -1.0, "waist_pitch_joint": +1.0}
    for tag, f in (("dq0", 0.0), ("hipwaist_0p5", 0.5), ("hipwaist_1p0", 1.0)):
        name = f"constdq_pull_{tag}_pm2"
        argv = ["scripts/evaluate_paper_supplement.py", "--case", "pull", "--controller", "sonic", "--states", str(a.states / "pull_states.json"),
                "--model-id", name, "--output-dir", str(a.out / "const_dq" / name)]
        dq = {j: s * f * scale[j] for j, s in joints.items()}
        if f:
            argv += ["--sonic-const-dq", ",".join(f"{j}={v!r}" for j, v in dq.items())]
        jobs.append(dict(name=name, argv=argv, meta=dict(kind="const_dq", factor=f, dq_rad=dq, scale_rad={j: scale[j] for j in joints})))
    print(json.dumps(jobs, indent=1))


if __name__ == "__main__":
    main()
