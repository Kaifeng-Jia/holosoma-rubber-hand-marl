#!/usr/bin/env python3
"""T03 C v2 check 3a (Kevin 2026-11-12): the same checkpoint and states evaluated with the same evaluator on the laptop
RTX 5070 (logs_system1/post_train/recheck/, launched by the pipeline from recheck.json) and on the cloud RTX 4090
(logs_system1/cloud_evals/c2_curve/), +/-2, per state. Checkpoints: C-in Pull it1050, C-in Bin it1000 (copied from the cloud,
sha256 equal). Per state: completed, executed steps, progress, object RMSE, terminal object position error; summary: counts,
paired agreement (both / laptop only / cloud only / neither), states with identical executed steps.

    hssim_s1/bin/python -B system1/t03/c2_recheck_compare.py --out results_system1/T03/c2_checks/a_same_gpu
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PAIRS = [("pull", "cin_pull_m20kg_seed721_joint8000", 1050), ("bucket", "cin_bucket_m1kg_seed721_fresh12000", 1000)]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def term_pos(d, e):
    with np.load(d / Path(e["trajectory"]).name, allow_pickle=False) as z:
        return float(np.linalg.norm(z["object_pos_w"][-1] - z["reference_object_pos_w"][-1]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rows, summ, prov = [], [], []
    for task, label, it in PAIRS:
        dl = ROOT / "logs_system1/post_train/recheck" / label / f"it{it:05d}_pm2"
        dc = ROOT / "logs_system1/cloud_evals/c2_curve" / label / f"it{it:05d}_pm2"
        sl, sc = (json.loads((d / "summary.json").read_text()) for d in (dl, dc))
        ml, mc = (json.loads((d / "run_manifest.json").read_text()) for d in (dl, dc))
        assert sl["complete"] and sc["complete"]
        assert sl["checkpoint_sha256"] == sc["checkpoint_sha256"] and sl["states_sha256"] == sc["states_sha256"]
        launch = json.loads((dl.parent / f"{dl.name}.launch.json").read_text())
        prov.append(dict(task=task, run=label, iteration=it, checkpoint_sha256=sl["checkpoint_sha256"], states_sha256=sl["states_sha256"],
                         laptop_gpu=ml["gpu_name"], cloud_gpu=mc["gpu_name"], laptop_git_head=launch["git_head"],
                         laptop_summary_sha256=sha(dl / "summary.json"), cloud_summary_sha256=sha(dc / "summary.json"),
                         laptop_dir=str(dl.relative_to(ROOT)), cloud_dir=str(dc.relative_to(ROOT))))
        el = {e["state_id"]: e for e in sl["episodes"]}
        ec = {e["state_id"]: e for e in sc["episodes"]}
        for k in ec:
            x, y = el[k], ec[k]
            rows.append(dict(task=task, run=label, iteration=it, state_id=k, kind=y["kind"],
                             laptop_completed=x["completed_reference"], cloud_completed=y["completed_reference"],
                             laptop_steps=x["steps"], cloud_steps=y["steps"], same_steps=x["steps"] == y["steps"],
                             laptop_progress=x["progress"], cloud_progress=y["progress"],
                             laptop_object_rmse_3d_m=x["object_rmse_3d_m"], cloud_object_rmse_3d_m=y["object_rmse_3d_m"],
                             laptop_terminal_object_pos_error_m=term_pos(dl, x), cloud_terminal_object_pos_error_m=term_pos(dc, y),
                             laptop_sonic_fell_any=x.get("sonic_fell_any"), cloud_sonic_fell_any=y.get("sonic_fell_any")))
        pr = [r for r in rows if r["task"] == task and r["kind"] != "nominal"]
        nom = next(r for r in rows if r["task"] == task and r["kind"] == "nominal")
        summ.append(dict(task=task, run=label, iteration=it, n=len(pr),
                         laptop_completed=sum(r["laptop_completed"] for r in pr), cloud_completed=sum(r["cloud_completed"] for r in pr),
                         laptop_progress_mean=float(np.mean([r["laptop_progress"] for r in pr])),
                         cloud_progress_mean=float(np.mean([r["cloud_progress"] for r in pr])),
                         both_completed=sum(r["laptop_completed"] and r["cloud_completed"] for r in pr),
                         laptop_only=sum(r["laptop_completed"] and not r["cloud_completed"] for r in pr),
                         cloud_only=sum(r["cloud_completed"] and not r["laptop_completed"] for r in pr),
                         neither=sum(not r["laptop_completed"] and not r["cloud_completed"] for r in pr),
                         same_executed_steps=sum(r["same_steps"] for r in pr),
                         laptop_fell=sum(bool(r["laptop_sonic_fell_any"]) for r in pr), cloud_fell=sum(bool(r["cloud_sonic_fell_any"]) for r in pr),
                         nominal_laptop_completed=nom["laptop_completed"], nominal_cloud_completed=nom["cloud_completed"],
                         nominal_laptop_steps=nom["laptop_steps"], nominal_cloud_steps=nom["cloud_steps"]))
    for name, rr in (("per_state.csv", rows), ("summary.csv", summ), ("provenance.csv", prov)):
        with (a.out / name).open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rr[0]))
            w.writeheader()
            w.writerows(rr)
    for s in summ:
        print(json.dumps(s))


if __name__ == "__main__":
    raise SystemExit(main())
