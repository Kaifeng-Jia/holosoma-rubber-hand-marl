#!/usr/bin/env python3
"""T03: extreme negative Reward/env_raw_min in the C v2 trainings - which reward term, which agent, what happened -
side by side with the D paper Table runs (seeds 721-723). CPU only, read-only, from the training metrics.jsonl files.

Per run and iteration, two signals are compared (both are logged by every CORE4D pair training, C and D):
  * reward spike:  Reward/env_raw_min < --reward-threshold (normal values are about -0.4 or above);
  * physics spike: Bucket/vector_error_agent{0,1}_m2_mean > --vector-threshold m^2 (normal values are about 0.01-0.05;
    the interaction-vector error is the squared distance between reference and simulated body-object vectors, so
    1e10 m^2 means bodies about 1e5 m away from where they should be, i.e. a physics blow-up in at least one env).
Outputs (results_system1/T03/spikes/): spikes.csv (every iteration with either signal, every run), runs.csv (counts per
run), extreme_spikes_before_after.csv (env_raw_min < -1e8: training statistics 100 iterations before / after); README_CN.md
is written by hand.

    python3 system1/t03/spike_scan.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
C_TRAIN = ROOT / "exports_system1/cloud_sync/logs/train"
D_RUNS = "/home/kevin/holosoma-core4d-base/logs/PaperSupplement20260929/autodl_received/runs"
# paper lineage (logs_system1/post_train/d_curve_catalog.json checkpoints): seed 721 one run; seeds 722 / 723 iterations
# 1-10000 from *_cache_isolated02, 10001-12000 from the resume10000_to12000 clone (it overrides from 10001)
D_TABLE = {
    721: [Path("/home/kevin/holosoma-core4d-base/logs/Core4DSmallTableA/A_no_rel_no_height_5kg_fresh12000_env2048_seed721_cloud_20260920/metrics.jsonl")],
    722: [Path(f"{D_RUNS}/T07_seed722_fresh12000_cache_isolated02/metrics.jsonl"), Path(f"{D_RUNS}/T07_seed722_resume10000_to12000_clone20260930/metrics.jsonl")],
    723: [Path(f"{D_RUNS}/T07_seed723_fresh12000_cache_isolated02/metrics.jsonl"), Path(f"{D_RUNS}/T07_seed723_resume10000_to12000_clone20260930/metrics.jsonl")],
}
KEYS = ("Reward/env_raw_min", "Reward/env_raw_mean", "Bucket/vector_error_agent0_m2_mean", "Bucket/vector_error_agent1_m2_mean",
        "Bucket/raw_reward_min", "Bucket/raw_reward_mean", "physical_failure_count", "reset_count", "value_loss", "critic_grad_norm",
        "actor_grad_norm", "kl", "entropy", "last_step_agent_ref_position_error_max_m")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(paths):
    rows = {}
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            rows[int(r["iteration"])] = r  # a resumed segment overrides the earlier one from its first iteration
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward-threshold", type=float, default=-10.0)
    ap.add_argument("--vector-threshold", type=float, default=1.0)
    ap.add_argument("--out", type=Path, default=ROOT / "results_system1/T03/spikes")
    a = ap.parse_args()
    runs = []
    for d in sorted(C_TRAIN.glob("c*_*_seed72*")):
        if d.is_dir() and (d / "metrics.jsonl").exists() and not d.name.endswith("_critic50"):
            scheme = "C-in" if d.name.startswith("cin_") else "C-out" if d.name.startswith("cout_") else "C v1 (diverged; appendix)"
            runs.append((d.name, scheme, [d / "metrics.jsonl"]))
    for s, p in D_TABLE.items():
        runs.append((f"D_table_paper_T-07_seed{s}", "D", p))
    spikes, summary = [], []
    for name, scheme, paths in runs:
        rows = load(paths)
        its = sorted(rows)
        nr = nv = both = vec_only = rew_only = 0
        worst = None
        for i in its:
            r = rows[i]
            rmin = r.get("Reward/env_raw_min")
            v = max(r.get("Bucket/vector_error_agent0_m2_mean", 0.0) or 0.0, r.get("Bucket/vector_error_agent1_m2_mean", 0.0) or 0.0)
            rs = rmin is not None and rmin < a.reward_threshold
            vs = v > a.vector_threshold
            if not (rs or vs):
                continue
            nr += rs
            nv += vs
            both += rs and vs
            vec_only += vs and not rs
            rew_only += rs and not vs
            if rs and (worst is None or rmin < worst[1]):
                worst = (i, rmin)
            nxt = rows.get(i + 1, {})
            spikes.append(dict(run=name, scheme=scheme, iteration=i, reward_spike=rs, physics_spike=vs,
                               agent=("1" if (r.get("Bucket/vector_error_agent1_m2_mean") or 0) > a.vector_threshold else "")
                               + ("0" if (r.get("Bucket/vector_error_agent0_m2_mean") or 0) > a.vector_threshold else ""),
                               **{k.replace("/", "_"): r.get(k) for k in KEYS},
                               next_env_raw_min=nxt.get("Reward/env_raw_min"), next_value_loss=nxt.get("value_loss"),
                               next_vector_error_max=max(nxt.get("Bucket/vector_error_agent0_m2_mean", 0) or 0,
                                                         nxt.get("Bucket/vector_error_agent1_m2_mean", 0) or 0)))
        summary.append(dict(run=name, scheme=scheme, iterations=len(its), first=its[0], last=its[-1],
                            reward_spike_iterations=nr, physics_spike_iterations=nv, both=both, physics_only=vec_only, reward_only=rew_only,
                            worst_env_raw_min=None if worst is None else worst[1], worst_iteration=None if worst is None else worst[0],
                            has_vector_error_keys="Bucket/vector_error_agent0_m2_mean" in rows[its[-1]],
                            reward_min_of_run=min((rows[i].get("Reward/env_raw_min") for i in its if rows[i].get("Reward/env_raw_min") is not None), default=None),
                            metrics=";".join(str(p) for p in paths), metrics_sha256=";".join(sha(p) for p in paths)))
        print(name, len(its), "reward", nr, "physics", nv, "both", both, "physics_only", vec_only, "reward_only", rew_only, worst, flush=True)
    # extreme spikes (env_raw_min < -1e8): training statistics 100 iterations before / after
    import statistics
    paths_of = {name: paths for name, _, paths in runs}
    before_after = []
    for sp in spikes:
        if sp["Reward_env_raw_min"] is None or sp["Reward_env_raw_min"] >= -1e8:
            continue
        rows, i = load(paths_of[sp["run"]]), sp["iteration"]
        rec = dict(run=sp["run"], scheme=sp["scheme"], iteration=i,
                   samples_estimate=sp["Reward_env_raw_mean"] * 49152 / sp["Reward_env_raw_min"])
        for k, lab in (("reference_completion_fraction_of_resets", "completion_frac"), ("Bucket/raw_reward_mean", "bucket_reward_mean"),
                       ("Episode/reset_length_mean_steps", "reset_len")):
            for tag, lo, hi in (("before100", i - 100, i), ("after100", i + 1, i + 101)):
                v = [rows[j][k] for j in range(lo, hi) if j in rows and rows[j].get(k) is not None]
                rec[f"{lab}_{tag}"] = statistics.mean(v) if v else None
        rec["value_loss_after_1_5"] = ";".join(f"{rows[j]['value_loss']:.4g}" for j in range(i + 1, i + 6) if j in rows)
        rec.update(kl_spike=rows[i]["kl"], entropy_before=rows[i - 1]["entropy"], entropy_after=rows[i + 1]["entropy"])
        before_after.append(rec)
    a.out.mkdir(parents=True, exist_ok=True)
    for fn, rows in (("spikes.csv", spikes), ("runs.csv", summary), ("extreme_spikes_before_after.csv", before_after)):
        keys = []
        for r in rows:
            keys += [k for k in r if k not in keys]
        with (a.out / fn).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
