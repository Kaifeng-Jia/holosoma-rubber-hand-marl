#!/usr/bin/env python3
"""C early training progress vs D trained from scratch at the same iterations (Kevin 2026-11-11).

  * mean_step_reward = metrics `reward_mean` (V3 curve export mapping, codebase_map: mean_step_reward <- reward_mean);
  * Pull tracking part (body + object, no regularisers): dt * sum_i w_i * reward_terms_raw_mean[i] over the 8 tracking
    terms of g1_29dof_plan5_push_reward (the Pull reward); the regulariser part likewise; the residual
    reward_mean - (tracking + regulariser) is reported;
  * Table / Bin: the core4d trainer logs no per-term reward (C and D alike), so the tracking part cannot be computed from
    the logs; the logged tracking-related quantities are reported instead, labelled as such:
    Bucket/raw_reward_mean (the object interaction term, raw) and last_step_agent_ref_position_error_mean_m.
  * values are 100-iteration window means ending at the stated iteration (iterations k-99..k).
D from scratch: Pull V3-25/26/27 (critic50 + joint8000, seeds 721-723); Table T-07 and Bin B-RZ0 paper main result.

    hssim_s1/bin/python -B system1/t03/early_progress.py --c-snapshot logs_system1/T03/early_progress/c_metrics_snapshot_<ts> --out results_system1/T03/early_progress_20261111
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
_B = "/home/kevin/holosoma-core4d-base/logs"
_R = f"{_B}/PaperSupplement20260929/autodl_received/runs"
_V3P = "/home/kevin/holosoma-core4d-base/logs/V3_20261102/cloud_backup_20261104/raw/v3_20261103_pull/logs/V3_20261103_pull_retry01"
DT = 1.0 / 50.0
W_TRACK = {"motion_global_ref_position_error_exp": 0.5, "motion_global_ref_orientation_error_exp": 0.5,
           "motion_relative_body_position_error_exp": 1.0, "motion_relative_body_orientation_error_exp": 1.0,
           "motion_global_body_lin_vel": 1.0, "motion_global_body_ang_vel": 1.0,
           "object_global_ref_position_error_exp": 1.0, "object_global_ref_orientation_error_exp": 1.0}
W_REG = {"action_rate_l2": -0.1, "limits_dof_pos": -10.0, "undesired_contacts": -0.1}
WINDOW = 100
POINTS = (1000, 2000, 3000, 4000)


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()


def runs(snap):
    s = Path(snap)
    r = {("pull", "C", "721"): [s / "c_pull_m20kg_seed721_critic50.metrics.jsonl", s / "c_pull_m20kg_seed721_joint8000.metrics.jsonl"],
         ("table", "C", "721"): [s / "c_table_m5kg_seed721_fresh12000.metrics.jsonl"],
         ("bucket", "C", "721"): [s / "c_bucket_m1kg_seed721_fresh12000.metrics.jsonl"]}
    for i, sd in enumerate(("721", "722", "723")):
        r[("pull", "D", sd)] = [Path(f"{_V3P}/V3-{25 + i}_pull_scratch_seed{sd}_critic50/metrics.jsonl"),
                                Path(f"{_V3P}/V3-{25 + i}_pull_scratch_seed{sd}_joint8000/metrics.jsonl")]
    r[("table", "D", "721")] = [Path(f"{_B}/Core4DSmallTableA/A_no_rel_no_height_5kg_fresh12000_env2048_seed721_cloud_20260920/metrics.jsonl")]
    r[("bucket", "D", "721")] = [Path(f"{_B}/Core4DBucket/A_no_rel_no_height_fresh12000_env2048_seed721_cloud_20260919/metrics.jsonl")]
    for sd in ("722", "723"):
        r[("table", "D", sd)] = [Path(f"{_R}/T07_seed{sd}_fresh12000_cache_isolated02/metrics.jsonl")]
        r[("bucket", "D", sd)] = [Path(f"{_R}/BRZ0_seed{sd}_fresh12000_concurrent_20260930/metrics.jsonl")]
    return r


def series(files, case):
    rows = {}
    for f in files:
        with open(f) as fh:
            for l in fh:
                j = json.loads(l)
                it = int(j["iteration"])
                if it in rows:
                    raise RuntimeError(f"duplicate iteration {it} in {files}")
                x = dict(reward_mean=j["reward_mean"])
                if case == "pull":
                    t = j["reward_terms_raw_mean"]
                    x["tracking"] = DT * sum(w * t[k] for k, w in W_TRACK.items())
                    x["regulariser"] = DT * sum(w * t[k] for k, w in W_REG.items())
                    x["residual"] = j["reward_mean"] - x["tracking"] - x["regulariser"]
                    x["object_tracking"] = DT * sum(W_TRACK[k] * t[k] for k in W_TRACK if k.startswith("object_"))
                else:
                    x["bucket_raw_reward_mean"] = j.get("Bucket/raw_reward_mean")
                    x["agent_ref_position_error_mean_m"] = j.get("last_step_agent_ref_position_error_mean_m")
                rows[it] = x
    its = sorted(rows)
    if its != list(range(its[0], its[-1] + 1)):
        raise RuntimeError(f"gap in iterations {files}")
    return rows


def window(rows, k, key):
    v = [rows[i][key] for i in range(k - WINDOW + 1, k + 1) if i in rows and rows[i].get(key) is not None]
    return float(np.mean(v)) if len(v) == WINDOW else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--c-snapshot", required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    R = runs(a.c_snapshot)
    S = {k: series(v, k[0]) for k, v in R.items()}
    out, prov = [], []
    for k, files in R.items():
        prov += [dict(task=k[0], scheme=k[1], seed=k[2], file=str(f), sha256=sha(f), first_iteration=min(S[k]), last_iteration=max(S[k])) for f in files]
    for case in ("pull", "table", "bucket"):
        c_last = max(S[(case, "C", "721")])
        keys = ["reward_mean"] + (["tracking", "object_tracking", "regulariser", "residual"] if case == "pull"
                                  else ["bucket_raw_reward_mean", "agent_ref_position_error_mean_m"])
        for k_it in [x for x in POINTS if x <= c_last] + [c_last]:
            for (cs, sch, sd), rows in S.items():
                if cs != case:
                    continue
                out.append(dict(task=case, iteration_window_end=k_it, window=WINDOW, scheme=sch, seed=sd,
                                **{key: window(rows, k_it, key) for key in keys}))
    args = a.out
    args.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(f for r in out for f in r))
    with (args / "early_progress_windows.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out)
    with (args / "sources.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(prov[0]))
        w.writeheader()
        w.writerows(prov)
    # residual check of the Pull decomposition (all iterations)
    res = {f"{k[1]}_{k[2]}": float(np.max(np.abs([r["residual"] for r in v.values()]))) for k, v in S.items() if k[0] == "pull"}
    rel = {f"{k[1]}_{k[2]}": float(np.median([abs(r["residual"]) / max(abs(r["reward_mean"]), 1e-9) for r in v.values()])) for k, v in S.items() if k[0] == "pull"}
    (args / "pull_decomposition_check.json").write_text(json.dumps(dict(max_abs_residual=res, median_rel_residual=rel, dt=DT,
                                                                         tracking_weights=W_TRACK, regulariser_weights=W_REG), indent=1) + "\n")
    for r in out:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    print(json.dumps(dict(max_abs_residual=res, median_rel_residual=rel)))


if __name__ == "__main__":
    main()
