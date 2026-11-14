#!/usr/bin/env python3
"""T04: tables for the direct evaluation of the original-mass paper D policy (seed 721) at other mass levels,
plus the combined A + D mass-sweep table.

Rows follow the V3 closeout episodes.csv columns (same builder rules as system1/package_scheme_a.py: hand error by
URDF FK, palm by the V3 palm post-process, object errors from the evaluator summary). The original-mass D point is
the paper result itself (V3 closeout rows, variant Full, seed 721), copied with its V3 values.

    hssim_s1/bin/python -B system1/t04/package_d_direct.py
Writes results_system1/T04/D/{mass_sweep_episodes_D.csv, mass_sweep_summary_D.csv} and
results_system1/T04/{mass_sweep_episodes.csv, mass_sweep_summary.csv} (A + D).
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from system1 import gpu_provenance  # noqa: E402
from system1.package_scheme_a import PALM_OUT, V3_EPISODES, HandMetrics, csvsave, eval_code_sha256, palm_module, read, sha  # noqa: E402

L = ROOT / "logs_system1/T04/D_direct"
OUT = ROOT / "results_system1/T04"
ORIG = dict(pull=20.0, table=5.0, bucket=1.0)
LEVELS = dict(pull=(5.0, 10.0, 40.0), table=(2.5, 10.0, 20.0), bucket=(2.0, 4.0, 8.0))


def summarize(rs):
    per = [r for r in rs if str(r["is_nominal"]) not in ("True", "true", "1") and r["is_nominal"] is not True]
    assert len(per) == 50, len(per)
    done = [r for r in per if str(r["completed_reference"]) == "True"]
    x = dict(task=rs[0]["task"], scheme=rs[0]["scheme"], mass_kg=float(rs[0]["mass_kg"]), original_mass_kg=float(rs[0]["original_mass_kg"]),
             retrained=rs[0]["retrained"], seed=rs[0]["seed"], perturbation_level_cm=int(rs[0]["perturbation_level_cm"]),
             n_episodes=50, n_completed=len(done), completion_pct=2 * len(done),
             progress_mean_all50=float(np.mean([float(r["progress"]) for r in per])))
    for k in ("object_pos_rmse_cm", "hand_pos_error_cm", "palm_error_deg"):
        v = [float(r[k]) for r in done if r[k] not in (None, "")]
        x[k + "_mean"] = float(np.mean(v)) if v else None
        x[k + "_n_episodes"] = len(v)
    return x


def main():
    hm = HandMetrics()
    palm, palm_sha = palm_module()
    rows = []
    for case, masses in LEVELS.items():
        for m in masses:
            for lvl in (2, 8):
                label = f"{case}_m{m:g}kg".replace(".", "p") + f"_pm{lvl}"
                d = L / label
                ex, la = read(L / "ops" / f"{label}.exit.json"), read(L / "ops" / f"{label}.launch.json")
                s = read(d / "summary.json")
                assert ex["completed"] and s["complete"] and len(s["episodes"]) == 51
                mass = float(s["physics"]["mass"][0][0])
                assert abs(mass - m) < 1e-6 and s.get("system1_checkpoint_mass") == "original"
                assert s["checkpoint_sha256"] == la["checkpoint_sha256"] and s["states_sha256"] == la["states_sha256"]
                ssha = sha(d / "summary.json")
                gpu, gsrc = gpu_provenance.ours(d, L / "ops" / f"{label}.stdout.log")
                assert gpu, d
                palm_rows = {}
                if case in ("bucket", "table"):
                    palm.process(d / "summary.json")
                    key = hashlib.sha256(str(d / "summary.json").encode()).hexdigest()[:12]
                    ps = read(PALM_OUT / key / "summary.json")
                    assert ps["evaluation_summary_sha256"] == ssha
                    palm_rows = {x["state_id"]: x for x in ps["episodes"]}
                hm.ref(case)
                obj_urdf = ROOT / "src/holosoma" / s["object_urdf"]
                for e in s["episodes"]:
                    nom, done = e["kind"] == "nominal", e["completed_reference"]
                    traj = Path(e["trajectory"])
                    hand = hm.episode(case, traj) if (done and not nom) else None
                    ph = palm_rows.get(e["state_id"])
                    rows.append(dict(
                        experiment_group="System1-T04-D-direct", task=case, variant="Full-original-mass-policy",
                        run_id=f"S1D-{case}-m{m:g}kg-pm{lvl}", training_seed=721, model_id=s["model_id"],
                        checkpoint_iteration=s["checkpoint_iteration"], checkpoint_sha256=s["checkpoint_sha256"],
                        perturbation_level_cm=lvl, perturbation_level_deg=lvl, state_id=e["state_id"], state_file_sha256=s["states_sha256"],
                        is_nominal=nom, completed_reference=done, termination_reason=json.dumps(e["terminal_terms"], sort_keys=True),
                        executed_steps=e["steps"], reference_horizon_steps=e["horizon_steps"], object_pos_rmse_cm=e["object_rmse_3d_m"] * 100,
                        object_xy_rmse_cm=e["object_rmse_xy_m"] * 100, height_rmse_cm=e["height_rmse_m"] * 100,
                        hand_pos_error_cm=hand[0] if hand else None,
                        hand_status="complete" if hand else "excluded_nominal" if nom else "not_completed" if not done else "missing",
                        palm_error_deg=ph["mean_deg"] if ph else None,
                        palm_status=("excluded_nominal" if nom else "not_completed" if not done else "complete" if ph else "missing")
                        if case in ("bucket", "table") else ("excluded_nominal" if nom else "missing"),
                        n_metric_frames=e["steps"] + 1, palm_valid_hand_frames=ph["count"] if ph else 0, trajectory_path=str(traj),
                        trajectory_sha256=sha(traj), config_sha256=sha(d / "run_manifest.json"), reference_sha256=hm.refinfo[case]["sha256"],
                        object_urdf_sha256=sha(obj_urdf), evaluation_code_sha256=eval_code_sha256(la),
                        exit_record_path=str(L / "ops" / f"{label}.exit.json"), summary_sha256=ssha, progress=e["progress"],
                        raw_reward_sum_common=e["reward_sum_common"], scheme="D", mass_kg=mass, original_mass_kg=ORIG[case],
                        mass_ratio=mass / ORIG[case], retrained="no (original-mass policy)", seed=721, delta_cm=None,
                        launch_git_head=la["git_head"], gpu_model=gpu, gpu_model_source=gsrc))
                print(label, sum(r["completed_reference"] for r in rows[-51:] if not r["is_nominal"]), flush=True)
    # original-mass D point = paper result (V3 rows, Full, seed 721)
    with V3_EPISODES.open() as f:
        for r in csv.DictReader(f):
            if (r["variant"] == "Full" and r["training_seed"] == "721" and r["task"] in ORIG
                    and r["perturbation_level_cm"] in ("2", "8")):
                rows.append(dict(r, experiment_group="V3-closeout (paper D)", scheme="D", mass_kg=ORIG[r["task"]],
                                 original_mass_kg=ORIG[r["task"]], mass_ratio=1.0, retrained="paper result (original mass)",
                                 seed=721, delta_cm=None, gpu_model=gpu_provenance.v3_row(r)[0]))
    OUT.joinpath("D").mkdir(parents=True, exist_ok=True)
    csvsave(OUT / "D/mass_sweep_episodes_D.csv", rows)
    groups = {}
    for r in rows:
        groups.setdefault((r["task"], float(r["mass_kg"]), int(r["perturbation_level_cm"])), []).append(r)
    summ = [summarize(v) for k, v in sorted(groups.items())]
    csvsave(OUT / "D/mass_sweep_summary_D.csv", summ)
    # combined A + D
    with (OUT / "A/mass_sweep_episodes.csv").open() as f:
        a_rows = list(csv.DictReader(f))
    with (OUT / "A/mass_sweep_summary.csv").open() as f:
        a_summ = list(csv.DictReader(f))
    for r in a_rows:
        r.setdefault("seed", "")
    csvsave(OUT / "mass_sweep_episodes.csv", a_rows + rows)
    keep = ("task", "scheme", "mass_kg", "original_mass_kg", "retrained", "seed", "perturbation_level_cm", "n_episodes", "n_completed",
            "completion_pct", "progress_mean_all50", "object_pos_rmse_cm_mean", "hand_pos_error_cm_mean", "palm_error_deg_mean")
    comb = [{k: x.get(k) for k in keep} for x in a_summ] + [{k: x.get(k) for k in keep} for x in summ]
    gmap = {}
    for r in a_rows + rows:  # GPU model per table cell (Kevin 2026-11-10)
        gmap.setdefault((r["task"], r["scheme"], float(r["mass_kg"]), int(r["perturbation_level_cm"])), set()).add(r["gpu_model"])
    for x in comb:
        x["gpu_model"] = "/".join(sorted(gmap[(x["task"], x["scheme"], float(x["mass_kg"]), int(x["perturbation_level_cm"]))]))
    comb.sort(key=lambda x: (x["task"], float(x["mass_kg"]), x["scheme"], int(x["perturbation_level_cm"])))
    csvsave(OUT / "mass_sweep_summary.csv", comb)
    (OUT / "D/provenance.json").write_text(json.dumps(dict(
        rows=len(rows), hand_fk_test=hm.fk_test, references=hm.refinfo, palm_postprocess_sha256=palm_sha,
        v3_episodes=str(V3_EPISODES), v3_episodes_sha256=sha(V3_EPISODES),
        status_counts={f"{h}|{q}": n for (h, q), n in Counter((r["hand_status"], r["palm_status"]) for r in rows).items()}),
        indent=1, default=str) + "\n")
    print("rows", len(rows))


if __name__ == "__main__":
    main()
