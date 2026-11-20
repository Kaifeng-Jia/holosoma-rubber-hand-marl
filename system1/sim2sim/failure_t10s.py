#!/usr/bin/env python3
"""T10s failure summary (facts only) of MuJoCo closed-loop runs: per run, termination causes from each episode's last termination
record (robot r torso gravity-z error > 0.8, object position error > 0.25 m, object orientation error > 0.8 rad; several can hold
at once), executed steps, the first step where the object position error exceeds 0.1 m (system1/t10b/phases.divergence_step) and
the reference phase at that step and at the termination step (system1/t10b/phases.label_frames on the reference object of the
same trajectory). Also writes results_system1/T10_sim2sim/isaac_vs_mujoco_all.csv = T10 + T10s rows. Read only, CPU.

    sim2sim_s1/bin/python -B system1/sim2sim/failure_t10s.py --evals logs_system1/t10/mj_eval_other --out results_system1/T10_sim2sim/other_tasks
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from system1.t10b.phases import NAME_ZH, divergence_step, label_frames  # noqa: E402


def save_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def causes(lt):
    g = lt.get("gravity_z_err", [0.0, 0.0])
    out = [f"robot{r}_tilt" for r in range(len(g)) if g[r] > 0.8]
    if lt.get("object_pos_err", 0.0) > 0.25:
        out.append("object_position")
    if lt.get("object_ori_err", 0.0) > 0.8:
        out.append("object_orientation")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    runs, eps = [], []
    for s_path in sorted(a.evals.glob("*/summary.json")):
        s = json.loads(s_path.read_text())
        name = s_path.parent.name
        cnt = collections.Counter()
        div, term_steps, ph_div, ph_term = [], [], collections.Counter(), collections.Counter()
        labels = None
        for e in s["episodes"]:
            if e["completed_reference"]:
                continue
            z = np.load(e["trajectory"])
            if labels is None:  # reference object of this task (same for every state of the run)
                labels = label_frames(z["reference_object_pos_w"], z["reference_object_quat_xyzw"])
            k = divergence_step(z["object_pos_w"], z["reference_object_pos_w"], 0.1)
            c = causes(e.get("last_termination") or {})
            for x in c:
                cnt[x] += 1
            term_steps.append(e["steps"])
            pt = labels[min(e["steps"] - 1, len(labels) - 1)]
            ph_term[pt] += 1
            if k is not None:
                div.append(k)
                ph_div[labels[min(k, len(labels) - 1)]] += 1
            eps.append(dict(run=name, task=s["task"], scheme=s["scheme"], pd_setting=s["pd"], training_seed=s["seed"], perturbation_level_cm=s["level"],
                            state_id=e["state_id"], executed_steps=e["steps"], horizon_steps=e["horizon_steps"], termination_causes="+".join(c),
                            object_divergence_step_0p1m=k, phase_at_divergence=(labels[min(k, len(labels) - 1)] if k is not None else None),
                            phase_at_termination=pt))
        n = len(s["episodes"])
        nf = len(term_steps)
        runs.append(dict(run=name, task=s["task"], scheme=s["scheme"], pd_setting=s["pd"], training_seed=s["seed"], perturbation_level_cm=s["level"],
                         n_episodes_incl_nominal=n, n_failed=nf, **{f"cause_{x}": cnt.get(x, 0) for x in
                         ("robot0_tilt", "robot1_tilt", "object_position", "object_orientation")},
                         termination_step_median=(statistics.median(term_steps) if term_steps else None),
                         termination_step_min=(min(term_steps) if term_steps else None), termination_step_max=(max(term_steps) if term_steps else None),
                         n_failed_with_object_divergence=len(div), object_divergence_step_median=(statistics.median(div) if div else None),
                         phase_at_divergence_counts=json.dumps(dict(ph_div), ensure_ascii=False),
                         phase_at_termination_counts=json.dumps(dict(ph_term), ensure_ascii=False)))
    a.out.mkdir(parents=True, exist_ok=True)
    save_csv(a.out / "failure_summary.csv", runs)
    save_csv(a.out / "failure_episodes.csv", eps)
    (a.out / "phase_labels_zh.json").write_text(json.dumps(NAME_ZH, ensure_ascii=False, indent=1) + "\n")
    # T10 + T10s side by side
    base = ROOT / "results_system1/T10_sim2sim"
    t10 = list(csv.DictReader((base / "isaac_vs_mujoco.csv").open()))
    t10s = list(csv.DictReader((a.out / "isaac_vs_mujoco.csv").open()))
    if list(t10[0]) != list(t10s[0]):
        raise ValueError("T10 and T10s isaac_vs_mujoco.csv columns differ")
    rows = [dict(r, source="T10") for r in t10] + [dict(r, source="T10s") for r in t10s]
    save_csv(base / "isaac_vs_mujoco_all.csv", rows)
    print("runs", len(runs), "failed episodes", len(eps), "all rows", len(rows))


if __name__ == "__main__":
    main()
