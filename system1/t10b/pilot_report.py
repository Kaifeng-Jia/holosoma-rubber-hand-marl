#!/usr/bin/env python3
"""T10b pilot progress report (facts only, read only, CPU), per validation iteration:

1. Training log side by side: each pilot run next to its original run at the same iteration, from metrics.jsonl, over the window
   of the W iterations ending at that iteration (default 100):
   - episode length: mean length of the episodes that ended in the window (Episode/reset_length_mean_steps weighted by
     Episode/reset_count; failed episodes likewise). Training episodes start at a uniformly sampled reference frame
     (PairedA1MotionCommand.reset), so these lengths are not comparable with the evaluation horizon.
   - termination shares: the runs have two termination terms (run_config termination_terms): reference_horizon (= timeout_count,
     reaching the end of the reference) and joint_bad_tracking (= physical_failure_count); share = sum / sum(reset_count).
     joint_bad_tracking is one term (robot torso tilt, object position, object orientation); its parts are not logged in training.
   - reward: reward_mean (includes the PPO timeout bootstrap), Reward/env_raw_mean (sum of all reward terms after weights and dt,
     without the bootstrap), and the bucket_interaction term's logged parts (Bucket/*: raw reward, ungated, position, orientation,
     height, relative, gate, contact_agent0/1). Per-term values of the other 9 reward terms are not in metrics.jsonl.
2. Validation episodes (pilot only): termination step, termination reason by the T10b A rule (system1/t10b/package_friction.py:
   object position error > 0.25 m or orientation error > 0.8 rad at the last saved row, otherwise robot orientation), first step
   with object error > 0.1 m, the reference phase (system1/t10b/phases.py) at those steps, and which robot holds the object in the
   reference there (reference_contact_weights > 0.5, interaction_vectors_v1.npz). Next to it, the original run's completion count at
   the same iteration from results_system1/T03/train_efficiency/curve_points.csv (cloud RTX 4090, same protocol, but on the frozen
   +/-2 TEST states, not the validation states); the original checkpoint's sha256 is checked against that row.

    hssim_s1/bin/python -B system1/t10b/pilot_report.py --pilot logs_system1/t10b/pilot_cloud_mirror --iterations 2000,4000 \
        --out logs_system1/t10b/pilot_report
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from system1.t10b.phases import NAME_ZH, divergence_step, label_frames, quat_angle_xyzw, segments  # noqa: E402

CLOUD_PILOT = "/root/autodl-tmp/system1/logs/t10b_pilot/"
CLOUD_REPO = "/root/autodl-tmp/system1/duet-system1/"
RUNS = {
    "d_table_seed721_rand": dict(
        label="D Table 721", curve_run="D_table_paper_T-07_seed721", original="/home/kevin/holosoma-core4d-base/logs/Core4DSmallTableA/"
        "A_no_rel_no_height_5kg_fresh12000_env2048_seed721_cloud_20260920/metrics.jsonl"),
    "cin_bucket_seed721_rand": dict(
        label="C-in Bin 721", curve_run="cin_bucket_m1kg_seed721_fresh12000", original=str(ROOT / "exports_system1/cloud_sync/logs/train/cin_bucket_m1kg_seed721_fresh12000/metrics.jsonl")),
}
CURVE = ROOT / "results_system1/T03/train_efficiency/curve_points.csv"  # original runs' checkpoint curve (frozen +/-2 TEST states)
BUCKET = ["raw_reward", "raw_ungated", "position", "orientation", "height", "relative", "gate", "contact_agent0", "contact_agent1"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_metrics(path):
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            out[int(r["iteration"])] = r
    return out


def window_stats(m, it, w):
    its = range(it - w + 1, it + 1)
    missing = [i for i in its if i not in m]
    if missing:
        return None
    rs = [m[i] for i in its]
    n = sum(r["reset_count"] for r in rs)
    nc = sum(r["timeout_count"] for r in rs)
    nf = sum(r["physical_failure_count"] for r in rs)
    er = sum(r["Episode/reset_count"] for r in rs)
    ef = sum(r["Episode/failure_count"] for r in rs)
    row = dict(window=f"{it - w + 1}-{it}", resets=n, reference_horizon=nc, joint_bad_tracking=nf,
               share_reference_horizon=nc / n if n else None, share_joint_bad_tracking=nf / n if n else None,
               episode_length_mean_steps=sum(r["Episode/reset_length_mean_steps"] * r["Episode/reset_count"] for r in rs) / er if er else None,
               failed_episode_length_mean_steps=sum(r["Episode/failure_length_mean_steps"] * r["Episode/failure_count"] for r in rs) / ef if ef else None,
               reward_mean=statistics.fmean(r["reward_mean"] for r in rs),
               env_raw_reward_mean=statistics.fmean(r["Reward/env_raw_mean"] for r in rs))
    for k in BUCKET:
        row[f"bucket_{k}_mean"] = statistics.fmean(r[f"Bucket/{k}_mean"] for r in rs)
    return row


def local_path(p):
    p = str(p)
    if p.startswith(CLOUD_REPO):
        return ROOT / p[len(CLOUD_REPO):]
    return ROOT / p if not p.startswith("/") else Path(p)


def holder(cw):
    h = [f"robot{r}" for r in range(2) if cw[r] > 0.5]
    return "+".join(h) if h else "none"


def save_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=Path, required=True, help="local mirror of the cloud t10b_pilot directory")
    ap.add_argument("--iterations", required=True)
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    its = [int(x) for x in a.iterations.split(",")]
    train_rows, ep_rows, val_rows, refs = [], [], [], {}
    for name, info in RUNS.items():
        d = a.pilot / name
        rc = json.loads((d / "train/run_config.json").read_text())
        mp = load_metrics(d / "train/metrics.jsonl")
        mo = load_metrics(info["original"])
        # reference of the task: phase labels and reference holder per frame
        rt = local_path(rc["runtime_reference"])
        iv = local_path(rc["bucket_reference"])
        assert sha(rt) == rc["runtime_reference_sha256"], rt
        z = np.load(rt)
        labels = label_frames(z["object_pos_w"], z["object_quat_w"][:, [1, 2, 3, 0]])
        cw = np.load(iv)["reference_contact_weights"]
        refs[name] = dict(runtime_reference=str(rt.relative_to(ROOT)), runtime_reference_sha256=rc["runtime_reference_sha256"],
                          interaction_vectors=str(iv.relative_to(ROOT)), interaction_vectors_sha256=sha(iv), frames=len(labels),
                          phase_segments=[(l, NAME_ZH[l], s, e) for l, s, e in segments(labels)],
                          holder_segments=[(h, s, e) for h, s, e in segments([holder(c) for c in cw])])
        for it in its:
            for src, m, path in (("pilot", mp, d / "train/metrics.jsonl"), ("original", mo, Path(info["original"]))):
                row = window_stats(m, it, a.window)
                train_rows.append(dict(run=info["label"], source=src, iteration=it, metrics=str(path), last_logged_iteration=max(m),
                                       **(row or dict(window="not reached"))))
            summ = d / "validation" / f"it{it:05d}" / "summary.json"
            if not summ.exists():
                val_rows.append(dict(run=info["label"], iteration=it, completed="not available"))
                continue
            s = json.loads(summ.read_text())
            reasons, ph_term, ph_div, hold_term, steps = collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter(), []
            for e in s["episodes"]:
                row = dict(run=info["label"], iteration=it, state_id=e["state_id"], kind=e["kind"], completed=e["completed_reference"],
                           steps=e["steps"], horizon_steps=e["horizon_steps"], progress=e["progress"])
                if not e["completed_reference"]:
                    t = np.load(local_path(e["trajectory"].replace(CLOUD_PILOT, str(a.pilot.resolve()) + "/")))
                    k = int(t["reference_frame"][-1])
                    pe = float(np.linalg.norm(t["object_pos_w"][-1] - t["reference_object_pos_w"][-1]))
                    oe = float(quat_angle_xyzw(t["object_quat_xyzw"][-1], t["reference_object_quat_xyzw"][-1]))
                    reason = "object_position" if pe > 0.25 else "object_orientation" if oe > 0.8 else "robot_orientation"
                    dv = divergence_step(t["object_pos_w"], t["reference_object_pos_w"], 0.1)
                    dvf = int(t["reference_frame"][dv]) if dv is not None else None
                    row.update(last_reference_frame=k, reason=reason, object_pos_err_last_m=pe, object_ori_err_last_rad=oe,
                               phase_at_termination=labels[min(k, len(labels) - 1)], holder_at_termination=holder(cw[min(k, len(cw) - 1)]),
                               object_divergence_step_0p1m=dv, phase_at_divergence=labels[min(dvf, len(labels) - 1)] if dvf is not None else None)
                    if e["kind"] != "nominal":
                        reasons[reason] += 1
                        ph_term[row["phase_at_termination"]] += 1
                        hold_term[row["holder_at_termination"]] += 1
                        ph_div[row["phase_at_divergence"]] += 1
                        steps.append(e["steps"])
                ep_rows.append(row)
            orig_ck = Path(info["original"]).with_name(f"model_{it:05d}.pt")
            cr = [r for r in csv.DictReader(CURVE.open()) if r["run"] == info["curve_run"] and int(r["iteration"]) == it]
            assert len(cr) == 1 and cr[0]["checkpoint_sha256"] == sha(orig_ck), (name, it)
            pert = [e for e in s["episodes"] if e["kind"] != "nominal"]
            nom = [e for e in s["episodes"] if e["kind"] == "nominal"]
            val_rows.append(dict(run=info["label"], iteration=it, completed=sum(e["completed_reference"] for e in pert), perturbed=len(pert),
                                 nominal_completed=nom[0]["completed_reference"] if nom else None, nominal_steps=nom[0]["steps"] if nom else None,
                                 horizon_steps=pert[0]["horizon_steps"] if pert else None,
                                 failed_steps_median=statistics.median(steps) if steps else None,
                                 failed_steps_min=min(steps) if steps else None, failed_steps_max=max(steps) if steps else None,
                                 reasons=json.dumps(dict(reasons)), phase_at_termination=json.dumps(dict(ph_term), ensure_ascii=False),
                                 holder_at_termination=json.dumps(dict(hold_term)),
                                 phase_at_divergence=json.dumps({str(k): v for k, v in ph_div.items()}, ensure_ascii=False),
                                 checkpoint_sha256=json.loads((summ.parent / "run_manifest.json").read_text()).get("checkpoint_sha256"),
                                 validation_states_sha256=json.loads((summ.parent / "run_manifest.json").read_text()).get("states_sha256"),
                                 original_completed_on_test_states=int(cr[0]["n_completed"]), original_checkpoint_sha256=cr[0]["checkpoint_sha256"],
                                 original_test_states_sha256=cr[0]["states_sha256"], original_eval_dir=cr[0]["eval_dir"]))
    save_csv(a.out / "training_side_by_side.csv", train_rows)
    save_csv(a.out / "validation_summary.csv", val_rows)
    save_csv(a.out / "validation_episodes.csv", ep_rows)
    (a.out / "references.json").write_text(json.dumps(refs, ensure_ascii=False, indent=1) + "\n")
    (a.out / "provenance.json").write_text(json.dumps(dict(
        script_sha256=sha(__file__), window=a.window, iterations=its, curve_points_csv_sha256=sha(CURVE),
        metrics_sha256={str(p): sha(p) for n, i in RUNS.items() for p in (a.pilot / n / "train/metrics.jsonl", Path(i["original"]))}),
        indent=1) + "\n")
    for r in val_rows:
        print(r)
    for r in train_rows:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "metrics"})


if __name__ == "__main__":
    main()
