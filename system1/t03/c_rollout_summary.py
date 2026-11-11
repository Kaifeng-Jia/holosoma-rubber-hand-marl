#!/usr/bin/env python3
"""Summarise the scheme C rollout replays (system1/t03/c_rollout_diag.py, SYSTEM1_C_DIAG records).

Per checkpoint (= iteration segment) and joint: mean and 95th percentile of |dq| = |scale_D * a| over all recorded
env steps / envs / robots; the noise-only lower bounds from the checkpoint's action std (|mu + sigma*eps| has mean
>= sigma*sqrt(2/pi) and 95th percentile >= 1.96*sigma for any mean mu); the SONIC fall-rule trigger rate per robot-step
by criterion and the share of finished episodes in which it fired; the reference and PD checks.

    hssim_s1/bin/python -B system1/t03/c_rollout_summary.py --out results_system1/T03/c_learning_diag_20261111
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from system1.t03.c_rollout_diag import OUT, P, STEPS, TASKS, sha  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    dq_rows, fall_rows, checks = [], [], []
    for task, (label, its, _) in TASKS.items():
        for it in its:
            name = f"{task}_it{it}"
            js, npz = OUT / f"{name}.json", OUT / f"{name}.npz"
            if not js.exists() or json.loads(js.read_text()).get("steps") != STEPS:
                fall_rows.append(dict(task=task, iteration=int(it), status="not_done"))
                continue
            j = json.loads(js.read_text())
            z = np.load(npz)
            ck = P / label / f"model_{it}.pt"
            sigma = torch.load(ck, map_location="cpu", weights_only=False)["actor_model_state_dict"]["std"].numpy().astype(np.float64)
            scale = z["scale_d"].astype(np.float64)[:29]
            dq = z["dq_abs"].astype(np.float32)                                     # [T, E, A, 29]
            aa = z["a_abs"].astype(np.float32)
            flat = dq.reshape(-1, 29)
            mean, p95 = flat.mean(0), np.percentile(flat, 95, axis=0)
            names = [str(x) for x in z["joint_names"]][:29]
            for k in range(29):
                dq_rows.append(dict(task=task, iteration=int(it), joint=names[k], mean_abs_dq_rad=float(mean[k]), p95_abs_dq_rad=float(p95[k]),
                                    action_std=float(sigma[k]), action_scale=float(scale[k]),
                                    noise_only_mean_lower_bound_rad=float(sigma[k] * scale[k] * math.sqrt(2 / math.pi)),
                                    noise_only_p95_lower_bound_rad=float(1.96 * sigma[k] * scale[k]),
                                    mean_abs_action=float(aa.reshape(-1, 29)[:, k].mean())))
            c = j["sonic_fall_counts"]
            ep = z["episode_fell_any"]
            fall_rows.append(dict(task=task, iteration=int(it), status="done", num_envs=j["num_envs"], steps=j["steps"],
                                  robot_steps=c["robot_steps"], fall_rate=c["fall"] / c["robot_steps"],
                                  pelvis_height_rate=c["pelvis_height"] / c["robot_steps"], orientation_rate=c["orientation"] / c["robot_steps"],
                                  wrist_ankle_height_rate=c["wrist_ankle_height"] / c["robot_steps"],
                                  finished_episodes=int(len(ep)), episodes_fell_any_robot=float(ep.any(axis=1).mean()) if len(ep) else None,
                                  finished_episode_length_mean=float(z["episode_length"].mean()) if len(ep) else None,
                                  mean_abs_dq_all_joints_rad=float(flat.mean()), p95_abs_dq_all_joints_rad=float(np.percentile(flat, 95)),
                                  max_abs_dq_rad=float(flat.max()), mean_action_std=float(sigma.mean())))
            checks.append(dict(task=task, iteration=int(it), checkpoint=str(ck), checkpoint_sha256=sha(ck), diag_npz_sha256=sha(npz),
                               reference_check_max_abs=j["reference_check_max_abs"], pd_target_minus_qstar_max_abs=j["pd_target_minus_qstar_max_abs"],
                               to_sonic_is_permutation=j["to_sonic_is_permutation"],
                               scale_d=scale.tolist()))
    for name, rows in (("dq_by_checkpoint_joint.csv", dq_rows), ("rollout_by_checkpoint.csv", fall_rows)):
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with (a.out / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    (a.out / "rollout_checks.json").write_text(json.dumps(checks, indent=1) + "\n")
    for r in fall_rows:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    main()
