#!/usr/bin/env python3
"""T03 C v2 pre-training check with the local replay script ("用本机复现脚本核对"): for C-in and C-out on Pull / Table / Bin,
one stochastic training rollout (the training script with --resume <smoke checkpoint> --iterations 1, 256 envs x 240 steps,
scratch output dir) with SYSTEM1_C_DIAG on (system1/c_diag.py, kind-aware):
  * C-in: max |dq_i| / scale_i <= 1; the joint reference SONIC receives is within the joint limits; it equals
    clamp(reference + dq) on every frame (reference check 0);
  * C-out: the window SONIC receives is the unmodified reference (reference check 0); max |q* - (SONIC output + dq)| = 0;
  * both: PD target = q* (0).
Checkpoints: the cloud smoke runs (logs/c2_smoke), fetched to exports_system1/c2_smoke/ with sha256 checked; Pull uses the
joint-segment smoke model_00010.pt (actor updated 10 times), Table / Bin the smoke model_00000.pt.

    hssim_s1/bin/python -B system1/t03/c2_replay_check.py [--fetch]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from system1.t03.c_rollout_diag import PY, sha  # noqa: E402
from system1.t03.post_train import SCP, evaluator_count, git, now, ssh  # noqa: E402

OUT = ROOT / "logs_system1/T03/c2_replay"
EXP = ROOT / "exports_system1/c2_smoke"
REMOTE = "/root/autodl-tmp/system1/logs/c2_smoke"
NUM_ENVS, STEPS = 256, 240
BASE = {
    "pull": ["scripts/train_plan5_push.py", "--skill", "pull", "--seed", "721", "--episode-telemetry", "--actor-learning-rate", "0.00001",
             "--critic-learning-rate", "0.001", "--sonic-residual", "--sonic-tf32", "--from-scratch"],
    "table": ["scripts/train_core4d_smalltable.py", "--experiment", "smalltable5kg_A", "--reward-variant", "bucket_A_no_rel_no_height",
              "--seed", "721", "--sonic-residual", "--sonic-tf32"],
    "bucket": ["scripts/train_core4d_smalltable.py", "--experiment", "bucket003", "--reward-variant", "bucket_A_no_rel_no_height",
               "--seed", "721", "--sonic-residual", "--sonic-tf32"],
}


def ckpt(kind, task):
    if task == "pull":
        return f"{kind}_pull_joint_smoke", "model_00010.pt"
    return f"{kind}_{task}_m{dict(table=5, bucket=1)[task]}kg_seed721_fresh12000", "model_00000.pt"


def fetch():
    for kind in ("cin", "cout"):
        for task in BASE:
            d, f = ckpt(kind, task)
            dst = EXP / d / f
            dst.parent.mkdir(parents=True, exist_ok=True)
            rc, out = ssh(f"sha256sum {REMOTE}/{d}/{f}")
            h = out.split()[0]
            if not (dst.exists() and sha(dst) == h):
                subprocess.run(SCP + [f"root@connect.nmb1.seetacloud.com:{REMOTE}/{d}/{f}", str(dst)], check=True, timeout=600)
            assert sha(dst) == h, dst
            (dst.parent / "SHA256SUMS.remote").write_text(f"{h}  {f}\n")
            print(d, f, h[:16], flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fetch", action="store_true")
    p.add_argument("--slots", type=int, default=1)
    a = p.parse_args()
    if a.fetch:
        fetch()
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    for kind in ("cin", "cout"):
        for task, base in BASE.items():
            name = f"{kind}_{task}"
            d, f = ckpt(kind, task)
            ck = EXP / d / f
            run_dir = OUT / "runs" / name
            k = 1
            while run_dir.exists():
                run_dir = OUT / "runs" / f"{name}.attempt{k}"
                k += 1
            argv = ["taskset", "-c", "24-31", PY, "-B", *base, "--sonic-residual-kind", kind[1:], "--num-envs", str(NUM_ENVS),
                    "--steps-per-env", str(STEPS), "--iterations", "1", "--resume", str(ck), "--output-dir", str(run_dir)]
            while evaluator_count() >= a.slots:
                time.sleep(20)
            env = dict(os.environ, SYSTEM1_C_DIAG=str(OUT / f"{name}.npz"), LOCAL_RANK="64", WORLD_SIZE="1")
            rec = dict(name=name, argv=argv, started=now(), checkpoint=str(ck), checkpoint_sha256=sha(ck),
                       git_head=git("rev-parse", "HEAD").strip(), git_status=git("status", "--porcelain"),
                       gpu=subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())
            (OUT / f"{name}.launch.json").write_text(json.dumps(rec, indent=1))
            with open(OUT / f"{name}.stdout.log", "w") as log:
                rc = subprocess.run(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
            j = json.loads((OUT / f"{name}.json").read_text()) if (OUT / f"{name}.json").exists() else {}
            ok = j.get("steps") == STEPS and j.get("reference_check_max_abs") == 0.0 and j.get("pd_target_minus_qstar_max_abs") == 0.0
            if kind == "cin":
                ok = ok and j.get("c_in_max_dq_over_scale") is not None and j["c_in_max_dq_over_scale"] <= 1.0 + 1e-6 \
                    and j.get("c_in_max_reference_joint_limit_violation_rad") == 0.0
            else:
                ok = ok and j.get("c_out_max_qstar_minus_sonic_output_plus_dq") == 0.0
            res[name] = dict(returncode=rc, passed=bool(ok), checkpoint=str(ck), checkpoint_sha256=rec["checkpoint_sha256"],
                             **{k: j.get(k) for k in ("steps", "reference_check_max_abs", "pd_target_minus_qstar_max_abs", "c_in_max_dq_over_scale",
                                                      "c_in_max_reference_joint_limit_violation_rad",
                                                      "c_out_max_qstar_minus_sonic_output_plus_dq", "to_sonic_is_permutation")})
            print(now(), name, json.dumps(res[name]), flush=True)
    (OUT / "c2_replay_check.json").write_text(json.dumps(dict(checked=now(), passed=all(r["passed"] for r in res.values()), runs=res), indent=1))
    print("passed", all(r["passed"] for r in res.values()))


if __name__ == "__main__":
    main()
