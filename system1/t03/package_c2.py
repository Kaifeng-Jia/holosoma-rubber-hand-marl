#!/usr/bin/env python3
"""T03 return pack, C v2 main experiment at the original masses: C-in (seeds 721-723), C-out (seed 721) and D (paper
result, seeds 721-723), final checkpoints, +/-2 and +/-8, all evaluated on the laptop RTX 5070 (the paper-table card).
CPU only, from existing evaluator outputs (laptop pipeline system1/t03/post_train.py):
  * C: logs_system1/post_train/final/<run>_pm{2,8};
  * D: d_curve/<D run>/it<final>_pm2 and d_paper_final/<D run>/it<final>_pm8 (as results_system1/T03/c2_finals).

Outputs (results_system1/T03/c2_main/):
  * episodes.csv - one row per episode, the V3 closeout episodes.csv columns (same builders as system1/t04/package_retrained.py:
    hand error by URDF FK, palm error by the V3 palm post-process for Table / Bin) + scheme, seed, gpu_model;
  * per_seed_summary.csv - one row per (task, scheme, seed, level), V3 per_seed_summary.csv columns + progress / gpu;
  * scheme_summary.csv - completion mean and sample SD over seeds (C-out: one seed, SD empty, marked single seed);
  * mcnemar.csv - preregistered (docs/system1/README_zh.md "C 的主实验"): C paired with D by seed and initial state,
    exact two-sided McNemar. C-in vs D: the 3 seeds pooled (150 pairs, as paper Table III) and each seed alone (50 pairs);
    C-out 721 vs D 721 (50 pairs, single seed). Nominal episodes are not in the pairs (V3 rule: 50 perturbed states).

    hssim_s1/bin/python -B system1/t03/package_c2.py
"""
from __future__ import annotations

import hashlib
import json
import statistics
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from system1.package_scheme_a import PALM_OUT, HandMetrics, csvsave, eval_code_sha256, palm_module, read, sha  # noqa: E402

PT = ROOT / "logs_system1/post_train"
OUT = ROOT / "results_system1/T03/c2_main"
ORIG = dict(pull=20.0, table=5.0, bucket=1.0)
C_RUN = dict(pull="{k}_pull_m20kg_seed{s}_joint8000", table="{k}_table_m5kg_seed{s}_fresh12000", bucket="{k}_bucket_m1kg_seed{s}_fresh12000")
D_RUN = dict(pull=("D_pull_paper_seed{s}", 8050), table=("D_table_paper_T-07_seed{s}", 12000), bucket=("D_bucket_paper_B-RZ0_seed{s}", 12000))
SEEDS = dict(cin=(721, 722, 723), cout=(721,), D=(721, 722, 723))
SCHEME = dict(cin="C-in", cout="C-out", D="D")


def sources():
    """(task, scheme key, seed, level, eval dir, run label)"""
    out = []
    for task in ("pull", "table", "bucket"):
        for lvl in (2, 8):
            for k in ("cin", "cout"):
                for s in SEEDS[k]:
                    run = C_RUN[task].format(k=k, s=s)
                    out.append((task, k, s, lvl, PT / "final" / f"{run}_pm{lvl}", run))
            for s in SEEDS["D"]:
                run, it = D_RUN[task]
                run = run.format(s=s)
                out.append((task, "D", s, lvl, PT / ("d_curve" if lvl == 2 else "d_paper_final") / run / f"it{it:05d}_pm{lvl}", run))
    return out


def mcnemar(a: dict, b: dict):
    """a, b: {(seed, state_id): completed}; same keys. Returns counts and the exact two-sided p."""
    assert a.keys() == b.keys(), (sorted(set(a) ^ set(b)))[:5]
    both = sum(a[k] and b[k] for k in a)
    oa = sum(a[k] and not b[k] for k in a)
    ob = sum(b[k] and not a[k] for k in a)
    bf = sum(not a[k] and not b[k] for k in a)
    p = 1.0 if oa + ob == 0 else float(binomtest(min(oa, ob), oa + ob, 0.5).pvalue)
    return dict(n_pairs=len(a), both_complete=both, only_c_complete=oa, only_d_complete=ob, both_fail=bf, mcnemar_exact_two_sided_p=p)


def main():
    hm = HandMetrics()
    palm, palm_sha = palm_module()
    rows, per_seed, done_map, states_sha, missing = [], [], {}, {}, []
    for task, k, seed, lvl, d, run in sources():
        scheme = SCHEME[k]
        s = read(d / "summary.json") if (d / "summary.json").exists() else None
        if not s or not s.get("complete") or len(s["episodes"]) != 51:
            missing.append(dict(task=task, scheme=scheme, seed=seed, level=lvl, eval_dir=str(d.relative_to(ROOT)),
                                status="missing" if not s else "incomplete"))
            per_seed.append(dict(task=task, variant=scheme, training_seed=seed, perturbation_level_cm=lvl, run_id=run, n_episodes=50,
                                 n_completed=None, status=missing[-1]["status"], eval_dir=str(d.relative_to(ROOT))))
            print("MISSING", task, scheme, seed, lvl, d, flush=True)
            continue
        mass = float(s["physics"]["mass"][0][0])
        assert abs(mass - ORIG[task]) < 1e-6, (d, mass)
        states_sha.setdefault((task, lvl), set()).add(s["states_sha256"])
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
        la = d.with_name(d.name + ".launch.json")
        er = []
        for e in s["episodes"]:
            nom, done = e["kind"] == "nominal", e["completed_reference"]
            traj = Path(e["trajectory"])
            hand = hm.episode(task, traj) if (done and not nom) else None
            ph = palm_rows.get(e["state_id"])
            er.append(dict(
                experiment_group="System1-T03-C2-main", task=task, variant=scheme, run_id=run, training_seed=seed,
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
                object_urdf_sha256=sha(obj_urdf), evaluation_code_sha256=eval_code_sha256(read(la)) if la.exists() else None,
                summary_path=str(d.relative_to(ROOT)), summary_sha256=ssha, progress=e["progress"], raw_reward_sum_common=e["reward_sum_common"],
                scheme=scheme, seed=seed, gpu_model=gpu, sonic_fell_any=e.get("sonic_fell_any")))
            if not nom:
                done_map[(task, k, lvl, seed, e["state_id"])] = bool(done)
        rows += er
        per = [r for r in er if not r["is_nominal"]]
        assert len(per) == 50
        ok = [r for r in per if r["completed_reference"]]
        x = dict(task=task, variant=scheme, training_seed=seed, perturbation_level_cm=lvl, run_id=run, n_episodes=50, n_completed=len(ok),
                 completion_pct=2 * len(ok), summary_sha256=ssha, independent_prior_count=None)
        for key in ("object_pos_rmse_cm", "hand_pos_error_cm", "palm_error_deg"):
            v = [float(r[key]) for r in ok if r[key] not in (None, "")]
            x[key + "_mean"] = float(np.mean(v)) if v else None
            x[key + "_n_episodes"] = len(v)
        x.update(progress_mean_all50=float(np.mean([r["progress"] for r in per])), nominal_completed=[r for r in er if r["is_nominal"]][0]["completed_reference"],
                 checkpoint_iteration=s["checkpoint_iteration"], checkpoint_sha256=s["checkpoint_sha256"], states_sha256=s["states_sha256"],
                 gpu_model=gpu, status="complete", eval_dir=str(d.relative_to(ROOT)))
        per_seed.append(x)
        print(task, scheme, seed, lvl, len(ok), flush=True)
    for key, v in states_sha.items():
        assert len(v) == 1, (key, v)  # every scheme / seed of a task-level used the same frozen states file
    # scheme summary: mean and sample SD over seeds
    summ = []
    for task in ("pull", "table", "bucket"):
        for lvl in (2, 8):
            for k in ("D", "cin", "cout"):
                v = [x["completion_pct"] for x in per_seed if x["task"] == task and x["variant"] == SCHEME[k] and x["perturbation_level_cm"] == lvl
                     and x.get("n_completed") is not None]
                summ.append(dict(task=task, scheme=SCHEME[k], perturbation_level_cm=lvl, n_seeds=len(v), seeds=";".join(map(str, SEEDS[k])),
                                 completion_pct_per_seed=";".join(map(str, v)),
                                 completion_pct_mean=statistics.mean(v) if v else None,
                                 completion_pct_sample_sd=statistics.stdev(v) if len(v) > 1 else None,
                                 note="single seed (preliminary)" if len(SEEDS[k]) == 1 else ""))
    # McNemar, paired by seed and state
    mc = []
    for task in ("pull", "table", "bucket"):
        for lvl in (2, 8):
            def m(k, seeds):
                return {(s, st): v for (t, kk, l, s, st), v in done_map.items() if t == task and kk == k and l == lvl and s in seeds}
            for k, seeds, label in [("cin", (721, 722, 723), "C-in vs D, seeds 721-723 pooled (paper Table III)")] + \
                                   [("cin", (s,), f"C-in vs D, seed {s}") for s in (721, 722, 723)] + \
                                   [("cout", (721,), "C-out vs D, seed 721 (single seed, preliminary)")]:
                c, dd = m(k, seeds), m("D", seeds)
                if len(c) != 50 * len(seeds) or len(dd) != 50 * len(seeds):
                    mc.append(dict(task=task, perturbation_level_cm=lvl, comparison=label, status="missing evaluations"))
                    continue
                r = mcnemar(c, dd)
                mc.append(dict(task=task, perturbation_level_cm=lvl, comparison=label, c_scheme=SCHEME[k], seeds=";".join(map(str, seeds)),
                               **r, c_completion_pct=100 * sum(c.values()) / len(c), d_completion_pct=100 * sum(dd.values()) / len(dd),
                               state_file_sha256=next(iter(states_sha[(task, lvl)])), status="complete"))
    OUT.mkdir(parents=True, exist_ok=True)
    csvsave(OUT / "episodes.csv", rows)
    csvsave(OUT / "per_seed_summary.csv", per_seed)
    csvsave(OUT / "scheme_summary.csv", summ)
    csvsave(OUT / "mcnemar.csv", mc)
    (OUT / "provenance.json").write_text(json.dumps(dict(
        rows=len(rows), missing=missing, hand_fk_test=hm.fk_test, references=hm.refinfo, palm_postprocess_sha256=palm_sha,
        sources=[dict(task=t, scheme=SCHEME[k], seed=s, level=l, eval_dir=str(d.relative_to(ROOT)), run=r) for t, k, s, l, d, r in sources()]),
        indent=1, default=str) + "\n")
    print("rows", len(rows), "per-seed", len(per_seed), "missing", len(missing))


if __name__ == "__main__":
    main()
