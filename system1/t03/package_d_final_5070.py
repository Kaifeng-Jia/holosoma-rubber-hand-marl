#!/usr/bin/env python3
"""Return pack for the D paper main result on the laptop RTX 5070, Push (main text), Kick and Chair (appendix), seeds 721-723,
+/-2 and +/-8 (Kevin 2026-11-13 23:0x; evaluations by system1/t03/run_d_finals_local.py, final-evaluation protocol).
CPU only, from the existing evaluator outputs in logs_system1/post_train/d_final_5070/. Same builders and columns as
system1/t03/package_c2.py (V3 closeout episodes.csv columns + scheme, seed, gpu_model; hand error by URDF FK; palm error by
the V3 palm post-process for Chair, not applicable for Kick, not computed for Push - as the V3 closeout and scheme A packs).

Outputs (results_system1/T03/d_final_5070/):
  * episodes.csv, per_seed_summary.csv, scheme_summary.csv (mean and sample SD over the 3 seeds);
  * v3_vs_5070.csv - the same checkpoint on the same frozen states in the V3 closeout (paper_handoff_exports/V3_20261102/
    V3_GPU_CLOSEOUT_20261104/results/episodes.csv, variant Full; cloud evaluation) next to the 5070 evaluation: completed
    counts and per-state agreement. Two separate evaluations; the paper table uses the 5070 ones only;
  * configs/ - launch / exit records and run_manifest.json of every evaluation (gzip), provenance.json.

    hssim_s1/bin/python -B system1/t03/package_d_final_5070.py
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from system1.package_scheme_a import PALM_OUT, HandMetrics, csvsave, eval_code_sha256, palm_module, read, sha  # noqa: E402

SRC = ROOT / "logs_system1/post_train/d_final_5070"
OUT = ROOT / "results_system1/T03/d_final_5070"
V3 = Path("/home/kevin/holosoma-core4d-base/paper_handoff_exports/V3_20261102/V3_GPU_CLOSEOUT_20261104/results/episodes.csv")
ORIG = dict(push=20.0, kick=20.0, chair=5.0)
TASKS = ("push", "kick", "chair")
SEEDS = (721, 722, 723)


def sources():
    out = []
    for task in TASKS:
        for seed in SEEDS:
            for lvl in (2, 8):
                run = SRC / f"D_{task}_paper_seed{seed}"
                d = [p for p in run.glob(f"it*_pm{lvl}") if p.is_dir() and ".attempt" not in p.name] if run.exists() else []
                out.append((task, seed, lvl, d[0] if len(d) == 1 else run / f"it?????_pm{lvl}", f"D_{task}_paper_seed{seed}"))
    return out


def v3_states():
    out = {}
    with V3.open() as f:
        for r in csv.DictReader(f):
            if r["task"] in TASKS and r["variant"] == "Full" and r["perturbation_level_cm"] in ("2", "8") and r["is_nominal"] == "False":
                k = (r["task"], int(r["training_seed"]), int(r["perturbation_level_cm"]))
                out.setdefault(k, dict(run_id=r["run_id"], checkpoint_sha256=r["checkpoint_sha256"], state_file_sha256=r["state_file_sha256"],
                                       done={}))["done"][r["state_id"]] = r["completed_reference"] == "True"
    return out


def main():
    hm = HandMetrics()
    palm, palm_sha = palm_module()
    v3 = v3_states()
    rows, per_seed, missing, cmp_rows, cfg = [], [], [], [], {}
    for task, seed, lvl, d, run in sources():
        ex = d.with_name(d.name + ".exit.json")
        s = read(d / "summary.json") if (d / "summary.json").exists() else None
        if not s or not s.get("complete") or len(s["episodes"]) != 51 or not ex.exists() or read(ex)["returncode"] != 0:
            missing.append(dict(task=task, seed=seed, level=lvl, eval_dir=str(d.relative_to(ROOT)), status="missing" if not s else "incomplete"))
            per_seed.append(dict(task=task, variant="D", training_seed=seed, perturbation_level_cm=lvl, run_id=run, n_episodes=50,
                                 n_completed=None, status=missing[-1]["status"], eval_dir=str(d.relative_to(ROOT))))
            print("MISSING", task, seed, lvl, d, flush=True)
            continue
        mass = float(s["physics"]["mass"][0][0])
        assert abs(mass - ORIG[task]) < 1e-6, (d, mass)
        la = d.with_name(d.name + ".launch.json")
        L = read(la)
        assert L["checkpoint_sha256"] == s["checkpoint_sha256"] and L["states_sha256"] == s["states_sha256"], d
        ssha = sha(d / "summary.json")
        gpu = s.get("gpu_name")
        man = d / "run_manifest.json"
        palm_rows = {}
        if task == "chair":
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
            palm_status = ("not_applicable" if task == "kick" else "excluded_nominal" if nom else "not_completed" if not done else
                           "complete" if ph else "missing")
            er.append(dict(
                experiment_group="System1-T03-D-final-5070", task=task, variant="D", run_id=run, training_seed=seed,
                model_id=s["model_id"], checkpoint_iteration=s["checkpoint_iteration"], checkpoint_sha256=s["checkpoint_sha256"],
                perturbation_level_cm=lvl, perturbation_level_deg=lvl, state_id=e["state_id"], state_file_sha256=s["states_sha256"],
                is_nominal=nom, completed_reference=done, termination_reason=json.dumps(e["terminal_terms"], sort_keys=True),
                executed_steps=e["steps"], reference_horizon_steps=e["horizon_steps"], object_pos_rmse_cm=e["object_rmse_3d_m"] * 100,
                object_xy_rmse_cm=e["object_rmse_xy_m"] * 100, height_rmse_cm=e["height_rmse_m"] * 100,
                hand_pos_error_cm=hand[0] if hand else None,
                hand_status="complete" if hand else "excluded_nominal" if nom else "not_completed" if not done else "missing",
                palm_error_deg=ph["mean_deg"] if ph else None, palm_status=palm_status,
                n_metric_frames=e["steps"] + 1, palm_valid_hand_frames=ph["count"] if ph else 0, trajectory_path=str(traj),
                trajectory_sha256=sha(traj), config_sha256=sha(man) if man.exists() else None, reference_sha256=hm.refinfo[task]["sha256"],
                object_urdf_sha256=sha(obj_urdf), evaluation_code_sha256=eval_code_sha256(L),
                summary_path=str(d.relative_to(ROOT)), summary_sha256=ssha, progress=e["progress"], raw_reward_sum_common=e["reward_sum_common"],
                scheme="D", seed=seed, gpu_model=gpu, v3_run_id=L.get("v3_run_id")))
        rows += er
        per = [r for r in er if not r["is_nominal"]]
        assert len(per) == 50
        ok = [r for r in per if r["completed_reference"]]
        x = dict(task=task, variant="D", training_seed=seed, perturbation_level_cm=lvl, run_id=run, n_episodes=50, n_completed=len(ok),
                 completion_pct=2 * len(ok), summary_sha256=ssha, independent_prior_count=None)
        for key in ("object_pos_rmse_cm", "hand_pos_error_cm", "palm_error_deg"):
            v = [float(r[key]) for r in ok if r[key] not in (None, "")]
            x[key + "_mean"] = float(np.mean(v)) if v else None
            x[key + "_n_episodes"] = len(v)
        x.update(progress_mean_all50=float(np.mean([r["progress"] for r in per])), nominal_completed=[r for r in er if r["is_nominal"]][0]["completed_reference"],
                 checkpoint_iteration=s["checkpoint_iteration"], checkpoint_sha256=s["checkpoint_sha256"], states_sha256=s["states_sha256"],
                 gpu_model=gpu, status="complete", eval_dir=str(d.relative_to(ROOT)))
        per_seed.append(x)
        # the V3 closeout evaluation of the same checkpoint on the same states
        w = v3.get((task, seed, lvl))
        mine = {r["state_id"]: bool(r["completed_reference"]) for r in per}
        if w and w["checkpoint_sha256"] == s["checkpoint_sha256"] and w["state_file_sha256"] == s["states_sha256"] and w["done"].keys() == mine.keys():
            cmp_rows.append(dict(task=task, seed=seed, perturbation_level_cm=lvl, checkpoint_sha256=s["checkpoint_sha256"], states_sha256=s["states_sha256"],
                                 v3_run_id=w["run_id"], v3_n_completed=sum(w["done"].values()), laptop_5070_n_completed=len(ok),
                                 difference_5070_minus_v3=len(ok) - sum(w["done"].values()),
                                 states_same_outcome=sum(w["done"][k] == mine[k] for k in mine), states_only_v3_complete=sum(w["done"][k] and not mine[k] for k in mine),
                                 states_only_5070_complete=sum(mine[k] and not w["done"][k] for k in mine)))
        else:
            cmp_rows.append(dict(task=task, seed=seed, perturbation_level_cm=lvl, status="V3 row not matched by checkpoint / states sha256"))
        for f in (la, d.with_name(d.name + ".exit.json"), man):
            cfg[f"{run}__{d.name}__{f.name if f != man else 'run_manifest.json'}"] = f
        print(task, seed, lvl, len(ok), flush=True)
    summ = []
    for task in TASKS:
        for lvl in (2, 8):
            v = [x["completion_pct"] for x in per_seed if x["task"] == task and x["perturbation_level_cm"] == lvl and x.get("n_completed") is not None]
            summ.append(dict(task=task, scheme="D", perturbation_level_cm=lvl, n_seeds=len(v), seeds=";".join(map(str, SEEDS)),
                             completion_pct_per_seed=";".join(map(str, v)), completion_pct_mean=statistics.mean(v) if len(v) == 3 else None,
                             completion_pct_sample_sd=statistics.stdev(v) if len(v) == 3 else None,
                             note="" if len(v) == 3 else f"{3 - len(v)} of 3 seeds missing"))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "configs").mkdir(exist_ok=True)
    for name, f in sorted(cfg.items()):
        (OUT / "configs" / (name + ".gz")).write_bytes(gzip.compress(f.read_bytes(), mtime=0))
    csvsave(OUT / "episodes.csv", rows)
    csvsave(OUT / "per_seed_summary.csv", per_seed)
    csvsave(OUT / "scheme_summary.csv", summ)
    csvsave(OUT / "v3_vs_5070.csv", cmp_rows)
    (OUT / "provenance.json").write_text(json.dumps(dict(
        rows=len(rows), missing=missing, hand_fk_test=hm.fk_test, references=hm.refinfo, palm_postprocess_sha256=palm_sha,
        v3_episodes=str(V3), v3_episodes_sha256=sha(V3), runner="system1/t03/run_d_finals_local.py",
        runner_log_sha256=sha(SRC / "runner.log"),
        sources=[dict(task=t, seed=s, level=l, eval_dir=str(d.relative_to(ROOT)), run=r) for t, s, l, d, r in sources()]),
        indent=1, default=str) + "\n")
    print("rows", len(rows), "per-seed", len(per_seed), "missing", len(missing))


if __name__ == "__main__":
    main()
