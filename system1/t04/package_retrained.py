#!/usr/bin/env python3
"""T04 (v4 addendum): load-capacity table of the per-mass retrained D and C-in policies (seed 721), with the original-mass
points of the same schemes, all evaluated on the laptop RTX 5070 (+/-2 and +/-8, final checkpoints, frozen states).

Rows per evaluation follow the V3 closeout episodes.csv columns with the same builders as system1/t04/package_d_direct.py
(hand error by URDF FK, palm error by the V3 palm post-process for Table / Bin, object errors from the evaluator
summary). Sources (laptop pipeline system1/t03/post_train.py):
  * retrained: logs_system1/post_train/final/{d,cin}_<task>_m<mass>kg_seed721_<segment>_pm{2,8};
  * original-mass D: the paper D seed 721 final checkpoint re-evaluated on the 5070 (d_curve/<D run>/it<final>_pm2,
    d_paper_final/<D run>/it<final>_pm8 - the same files as results_system1/T03/c2_finals/finals.csv);
  * original-mass C-in: the T03 main run cin_<task>_m<orig>kg_seed721 (final/<run>_pm{2,8}).
A missing or incomplete evaluation is written to the summary with n_completed empty and the reason.
Also rebuilds the combined results_system1/T04/mass_sweep_summary.csv: scheme A (all levels), the original-mass D
policy evaluated directly at the other levels (D/mass_sweep_summary_D.csv, unchanged) and these rows.

    hssim_s1/bin/python -B system1/t04/package_retrained.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from system1.package_scheme_a import PALM_OUT, HandMetrics, csvsave, eval_code_sha256, palm_module, read, sha  # noqa: E402

PT = ROOT / "logs_system1/post_train"
OUT = ROOT / "results_system1/T04"
ORIG = dict(pull=20.0, table=5.0, bucket=1.0)
HEAVY = dict(pull=(40.0,), table=(10.0, 20.0), bucket=(2.0, 4.0, 8.0))
SEG = dict(pull="joint8000", table="fresh12000", bucket="fresh12000")
D_ORIG = dict(pull=("D_pull_paper_seed721", 8050), table=("D_table_paper_T-07_seed721", 12000), bucket=("D_bucket_paper_B-RZ0_seed721", 12000))
SCHEME = dict(d="D", cin="C-in")


def mstr(m):
    return f"{m:g}".replace(".", "p")


def sources():
    """(task, scheme, mass, level, retrained label, eval dir, run label)"""
    out = []
    for task in ("pull", "table", "bucket"):
        for lvl in (2, 8):
            run, it = D_ORIG[task]
            sub = "d_curve" if lvl == 2 else "d_paper_final"
            out.append((task, "D", ORIG[task], lvl, "paper result (original mass), 5070 re-evaluation", PT / sub / run / f"it{it:05d}_pm{lvl}", run))
            run = f"cin_{task}_m{mstr(ORIG[task])}kg_seed721_{SEG[task]}"
            out.append((task, "C-in", ORIG[task], lvl, "T03 main run (original mass)", PT / "final" / f"{run}_pm{lvl}", run))
            for k in ("d", "cin"):
                for m in HEAVY[task]:
                    run = f"{k}_{task}_m{mstr(m)}kg_seed721_{SEG[task]}"
                    out.append((task, SCHEME[k], m, lvl, "yes (retrained at this mass)", PT / "final" / f"{run}_pm{lvl}", run))
    return out


def main():
    hm = HandMetrics()
    palm, palm_sha = palm_module()
    rows, summ = [], []
    for task, scheme, m, lvl, retr, d, run in sources():
        s = read(d / "summary.json") if (d / "summary.json").exists() else None
        base = dict(task=task, scheme=scheme, mass_kg=m, original_mass_kg=ORIG[task], mass_ratio=m / ORIG[task], retrained=retr, seed=721,
                    perturbation_level_cm=lvl, run=run)
        if not s or not s.get("complete") or len(s["episodes"]) != 51:
            summ.append(dict(base, n_episodes=50, n_completed=None, status="missing" if not s else "incomplete", eval_dir=str(d.relative_to(ROOT))))
            print("MISSING", task, scheme, m, lvl, d, flush=True)
            continue
        mass = float(s["physics"]["mass"][0][0])
        assert abs(mass - m) < 1e-6, (d, mass, m)
        ssha = sha(d / "summary.json")
        gpu = s.get("gpu_name")
        man = d / "run_manifest.json"
        palm_rows = {}
        if task in ("bucket", "table"):
            palm.process(d / "summary.json")
            key = hashlib.sha256(str(d / "summary.json").encode()).hexdigest()[:12]
            ps = read(PALM_OUT / key / "summary.json")
            assert ps["evaluation_summary_sha256"] == ssha
            palm_rows = {x["state_id"]: x for x in ps["episodes"]}
        hm.ref(task)
        obj_urdf = ROOT / "src/holosoma" / s["object_urdf"]
        er = []
        for e in s["episodes"]:
            nom, done = e["kind"] == "nominal", e["completed_reference"]
            traj = Path(e["trajectory"])
            hand = hm.episode(task, traj) if (done and not nom) else None
            ph = palm_rows.get(e["state_id"])
            er.append(dict(
                experiment_group="System1-T04-retrained", task=task, variant=f"{scheme}-{retr}", run_id=run, training_seed=721,
                model_id=s["model_id"], checkpoint_iteration=s["checkpoint_iteration"], checkpoint_sha256=s["checkpoint_sha256"],
                perturbation_level_cm=lvl, perturbation_level_deg=lvl, state_id=e["state_id"], state_file_sha256=s["states_sha256"],
                is_nominal=nom, completed_reference=done, termination_reason=json.dumps(e["terminal_terms"], sort_keys=True),
                executed_steps=e["steps"], reference_horizon_steps=e["horizon_steps"], object_pos_rmse_cm=e["object_rmse_3d_m"] * 100,
                object_xy_rmse_cm=e["object_rmse_xy_m"] * 100, height_rmse_cm=e["height_rmse_m"] * 100,
                hand_pos_error_cm=hand[0] if hand else None,
                hand_status="complete" if hand else "excluded_nominal" if nom else "not_completed" if not done else "missing",
                palm_error_deg=ph["mean_deg"] if ph else None,
                palm_status=("excluded_nominal" if nom else "not_completed" if not done else "complete" if ph else "missing")
                if task in ("bucket", "table") else ("excluded_nominal" if nom else "missing"),
                n_metric_frames=e["steps"] + 1, palm_valid_hand_frames=ph["count"] if ph else 0, trajectory_path=str(traj),
                trajectory_sha256=sha(traj), config_sha256=sha(man) if man.exists() else None, reference_sha256=hm.refinfo[task]["sha256"],
                object_urdf_sha256=sha(obj_urdf), evaluation_code_sha256=eval_code_sha256(read(d.with_name(d.name + ".launch.json"))),
                summary_path=str(d.relative_to(ROOT)), summary_sha256=ssha, progress=e["progress"], raw_reward_sum_common=e["reward_sum_common"],
                scheme=scheme, mass_kg=mass, original_mass_kg=ORIG[task], mass_ratio=mass / ORIG[task], retrained=retr, seed=721,
                gpu_model=gpu))
        rows += er
        per = [r for r in er if not r["is_nominal"]]
        assert len(per) == 50
        done = [r for r in per if r["completed_reference"]]
        x = dict(base, n_episodes=50, n_completed=len(done), completion_pct=2 * len(done),
                 progress_mean_all50=float(np.mean([r["progress"] for r in per])), status="complete", eval_dir=str(d.relative_to(ROOT)),
                 checkpoint_sha256=s["checkpoint_sha256"], states_sha256=s["states_sha256"], summary_sha256=ssha, gpu_model=gpu)
        for k in ("object_pos_rmse_cm", "hand_pos_error_cm", "palm_error_deg"):
            v = [float(r[k]) for r in done if r[k] not in (None, "")]
            x[k + "_mean"] = float(np.mean(v)) if v else None
            x[k + "_n_episodes"] = len(v)
        summ.append(x)
        print(task, scheme, m, lvl, x["n_completed"], flush=True)
    (OUT / "retrained").mkdir(parents=True, exist_ok=True)
    csvsave(OUT / "retrained/mass_sweep_episodes_retrained.csv", rows)
    summ.sort(key=lambda x: (x["task"], float(x["mass_kg"]), x["scheme"], int(x["perturbation_level_cm"])))
    csvsave(OUT / "retrained/mass_sweep_summary_retrained.csv", summ)
    # combined table: A + D direct (original-mass policy at other masses) + retrained / original-mass points
    keep = ("task", "scheme", "mass_kg", "original_mass_kg", "retrained", "seed", "perturbation_level_cm", "n_episodes", "n_completed",
            "completion_pct", "progress_mean_all50", "object_pos_rmse_cm_mean", "hand_pos_error_cm_mean", "palm_error_deg_mean", "gpu_model")
    comb = []
    for f in (OUT / "A/mass_sweep_summary.csv", OUT / "D/mass_sweep_summary_D.csv"):
        with f.open() as fh:
            for r in csv.DictReader(fh):
                if r["scheme"] == "D" and r["retrained"].startswith("paper"):
                    continue  # V3-row original-mass D point; the 5070 re-evaluation is in the retrained table
                comb.append({k: r.get(k) for k in keep})
    gpu_a = {}
    with (OUT / "mass_sweep_summary.csv").open() as fh:  # GPU model of the A / D-direct cells as packaged before
        for r in csv.DictReader(fh):
            gpu_a[(r["task"], r["scheme"], float(r["mass_kg"]), r["perturbation_level_cm"], r["retrained"])] = r.get("gpu_model")
    for x in comb:
        x["gpu_model"] = x.get("gpu_model") or gpu_a.get((x["task"], x["scheme"], float(x["mass_kg"]), x["perturbation_level_cm"], x["retrained"]))
    comb += [{k: x.get(k) for k in keep} for x in summ]
    comb.sort(key=lambda x: (x["task"], float(x["mass_kg"]), x["scheme"], str(x["retrained"]), int(x["perturbation_level_cm"])))
    csvsave(OUT / "mass_sweep_summary.csv", comb)
    (OUT / "retrained/provenance.json").write_text(json.dumps(dict(
        rows=len(rows), hand_fk_test=hm.fk_test, references=hm.refinfo, palm_postprocess_sha256=palm_sha,
        sources=[dict(task=t, scheme=sc, mass_kg=m, level=l, retrained=r, eval_dir=str(d.relative_to(ROOT)), run=ru)
                 for t, sc, m, l, r, d, ru in sources()]), indent=1, default=str) + "\n")
    print("rows", len(rows), "summary", len(summ))


if __name__ == "__main__":
    main()
