#!/usr/bin/env python3
"""Build the scheme B result tables (T02 v2 main setting, B-open control, B mass sweep for T04).

Same columns and the same post-processing as scheme A (system1/package_scheme_a.py, V3 closeout episodes.csv): object
errors from the evaluator summary, hand error by the same URDF FK + reference alignment, palm error by the same V3 palm
post-process. Extra columns: the scheme A NEW_FIELDS plus B_FIELDS (documented in results_system1/T02/README_CN.md).
Every completed evaluation directory is used; attempts that ended without completing (attempt1, CUDA OOM at start-up,
0 episodes) are listed in attempts.csv and not used.

  hssim_s1/bin/python -B system1/t02/package_scheme_b.py
Reads logs_system1/post_train/B/ only; writes results_system1/T02/ (tables) and results_system1/T04/B/ (mass sweep).
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from system1 import gpu_provenance  # noqa: E402
from system1.package_scheme_a import (PALM_OUT, V3_EPISODES, HandMetrics, csvsave, palm_module, r0r1, read,  # noqa: E402
                                      sha)

L = ROOT / "logs_system1"
BROOT = L / "post_train/B"
OUT = ROOT / "results_system1/T02"
OUT_M = ROOT / "results_system1/T04/B"
CASES = ("push", "pull", "chair", "bucket", "table")          # Kick not applicable (T02: planner drops the kick)
VARIANTS = ("B0", "B1-0", "B1-2", "B1-4")
DELTA_CM = {"B-open": None, "B0": None, "B1-0": 0.0, "B1-2": 2.0, "B1-4": 4.0}
FEEDBACK = {"B-open": False, "B0": True, "B1-0": True, "B1-2": True, "B1-4": True}
OBJECT_FEEDBACK = {"B-open": False, "B0": False, "B1-0": True, "B1-2": True, "B1-4": True}
ORIG_MASS = dict(push=20.0, pull=20.0, chair=5.0, bucket=1.0, table=5.0)
SWEEP = dict(pull=(5.0, 10.0, 20.0, 40.0), table=(2.5, 5.0, 10.0, 20.0), bucket=(1.0, 2.0, 4.0, 8.0))
SWEEP_VARIANTS = ("B0", "B1-2")
LAST_WAYPOINT_FRAC = 0.5    # preregistration section 8 (commit eecfbfd)
B_FIELDS = ["b_variant", "delta_cm", "position_feedback", "object_feedback", "last_waypoint_frac", "planner_calls",
            "planner_seconds", "planner_seconds_per_call", "planner_sha256", "contacts_sha256"]
CODE = ("scripts/evaluate_paper_supplement.py", "system1/scheme_b.py", "system1/scheme_b_driver.py", "system1/sonic_controller.py",
        "system1/scheme_b_contacts_v2.py", "system1/mass_variants.py")


def units():
    out = []
    for case in CASES:
        for v in VARIANTS:
            for lvl in (2, 8):
                out.append(dict(group="main", variant=v, case=case, level=lvl, mass=ORIG_MASS[case], sub="main",
                                name=f"{case}_{v}_pm{lvl}"))
        out.append(dict(group="B-open", variant="B-open", case=case, level=2, mass=ORIG_MASS[case], sub="main",
                        name=f"{case}_B-open_pm2"))
    for case, masses in SWEEP.items():
        for m in masses:
            if m == ORIG_MASS[case]:
                continue
            for v in SWEEP_VARIANTS:
                for lvl in (2, 8):
                    tag = f"m{m:g}kg".replace(".", "p")
                    out.append(dict(group="T04", variant=v, case=case, level=lvl, mass=m, sub="mass", name=f"{case}_{v}_{tag}_pm{lvl}"))
    for u in out:
        u["dir"] = BROOT / u["sub"] / u["name"]
        u["ops"] = BROOT / u["sub"]
    return out


def attempts():
    rows = []
    for f in sorted(BROOT.glob("*/*.attempt*.exit.json")):
        ex = read(f)
        log = f.with_name(f.name.replace(".exit.json", ".stdout.log"))
        txt = log.read_text(errors="replace") if log.exists() else ""
        oom = next((ln.strip() for ln in txt.splitlines() if "OutOfMemoryError" in ln), None)
        d = f.with_name(f.name.replace(".exit.json", ""))
        rows.append(dict(attempt=str(f.relative_to(L)), name=ex["name"], returncode=ex["returncode"], completed=ex["completed"],
                         ended=ex.get("ended"), launch_started=read(f.with_name(f.name.replace(".exit", ".launch")))["started"],
                         n_trajectories=len(list(d.glob("*.trajectory.npz"))) if d.exists() else 0,
                         cause=oom[:200] if oom else None, used=False))
    return rows


def load_unit(u):
    ex, la = read(u["ops"] / f"{u['name']}.exit.json"), read(u["ops"] / f"{u['name']}.launch.json")
    s = read(u["dir"] / "summary.json")
    assert ex["returncode"] == 0 and ex["completed"] and s["complete"], u["dir"]
    eps = s["episodes"]
    assert len(eps) == 51 and len({e["state_id"] for e in eps}) == 51 and sum(e["kind"] == "perturbed" for e in eps) == 50
    assert la["states_sha256"] == s["states_sha256"]
    mass = float(s["physics"]["mass"][0][0])
    assert abs(mass - u["mass"]) < 1e-6, (u["dir"], mass, u["mass"])
    assert s["tf32"] is False and s["controller"] == "sonic_planner" and s["hip_pitch_actuator"] == "ours", u["dir"]
    sb = s["scheme_b"]
    assert s["b_variant"] == u["variant"] and sb["position_feedback"] == FEEDBACK[u["variant"]], u["dir"]
    assert abs(sb["last_waypoint_frac"] - LAST_WAYPOINT_FRAC) < 1e-12, u["dir"]
    want_delta = DELTA_CM[u["variant"]]
    if want_delta is None:   # B0, B-open: no object feedback, no contacts
        assert sb.get("delta_m") is None and sb.get("contacts_sha256") is None, (u["dir"], sb.get("delta_m"))
    else:
        assert abs(sb["delta_m"] * 100 - want_delta) < 1e-9 and sb.get("contacts_sha256"), (u["dir"], sb.get("delta_m"))
    assert all(e["b_variant"] == u["variant"] and e["position_feedback"] == FEEDBACK[u["variant"]] for e in eps)
    return dict(u, label=u["name"], summary=s, exit=ex, launch=la, summary_sha=sha(u["dir"] / "summary.json"),
                manifest_sha=sha(u["dir"] / "run_manifest.json"), mass_readback=mass)


def run_id(d):
    return f"S1B-{d['group']}-{d['case']}-{d['variant']}-m{d['mass']:g}kg-pm{d['level']}"


def main():
    data = [load_unit(u) for u in units()]
    hm = HandMetrics()
    palm, palm_src_sha = palm_module()
    robot_urdf = ROOT / "src/holosoma/holosoma/data/robots/g1/main_mesh_collision_rubberhand.urdf"
    git = lambda *x: subprocess.run(["git", "-C", str(ROOT), *x], capture_output=True, text=True).stdout  # noqa: E731
    code_sha_at = {}
    for d in data:
        h = d["launch"]["git_head"]
        if h not in code_sha_at:
            code_sha_at[h] = {p: hashlib.sha256(subprocess.run(["git", "-C", str(ROOT), "show", f"{h}:{p}"],
                                                               capture_output=True).stdout).hexdigest() for p in CODE}
    for d in data:
        d["gpu_model"], d["gpu_model_source"] = gpu_provenance.ours(d["dir"], d["ops"] / f"{d['label']}.stdout.log")
        assert d["gpu_model"], d["dir"]
    rows = []
    for d in data:
        s, case = d["summary"], d["case"]
        palm_rows = {}
        if case in ("chair", "bucket", "table"):
            spath = d["dir"] / "summary.json"
            palm.process(spath)
            key = hashlib.sha256(str(spath).encode()).hexdigest()[:12]
            ps = read(PALM_OUT / key / "summary.json")
            assert ps["evaluation_summary_sha256"] == d["summary_sha"]
            palm_rows = {x["state_id"]: x for x in ps["episodes"]}
        obj_urdf = ROOT / "src/holosoma" / s["object_urdf"]
        hm.ref(case)
        sb = s["scheme_b"]
        for e in s["episodes"]:
            nom, done = e["kind"] == "nominal", e["completed_reference"]
            traj = Path(e["trajectory"])
            hand = hm.episode(case, traj) if (done and not nom) else None
            ph = palm_rows.get(e["state_id"])
            with np.load(traj, allow_pickle=False) as z:  # last saved state = terminal state before autoreset
                t_pos = float(np.linalg.norm(z["object_pos_w"][-1] - z["reference_object_pos_w"][-1]))
                t_ori = float(2 * np.arccos(min(1.0, abs(float(np.dot(z["object_quat_xyzw"][-1], z["reference_object_quat_xyzw"][-1]))))))
            palm_status = ("excluded_nominal" if nom else "not_completed" if not done else "complete" if ph else "missing")
            rows.append(dict(
                experiment_group="System1-B-" + d["group"], task=case, variant="schemeB_" + d["variant"],
                run_id=run_id(d), training_seed=None, model_id=s["model_id"], checkpoint_iteration=None,
                checkpoint_sha256=s["checkpoint_sha256"], perturbation_level_cm=d["level"], perturbation_level_deg=d["level"],
                state_id=e["state_id"], state_file_sha256=s["states_sha256"], is_nominal=nom, completed_reference=done,
                termination_reason=json.dumps(e["terminal_terms"], sort_keys=True), executed_steps=e["steps"],
                reference_horizon_steps=e["horizon_steps"], object_pos_rmse_cm=e["object_rmse_3d_m"] * 100,
                object_xy_rmse_cm=e["object_rmse_xy_m"] * 100, height_rmse_cm=e["height_rmse_m"] * 100,
                hand_pos_error_cm=hand[0] if hand else None,
                hand_status="complete" if hand else "excluded_nominal" if nom else "not_completed" if not done else "missing",
                palm_error_deg=ph["mean_deg"] if ph else None, palm_status=palm_status, n_metric_frames=e["steps"] + 1,
                palm_valid_hand_frames=ph["count"] if ph else 0, trajectory_path=str(traj), trajectory_sha256=sha(traj),
                config_sha256=d["manifest_sha"], reference_sha256=hm.refinfo[case]["sha256"], object_urdf_sha256=sha(obj_urdf),
                evaluation_code_sha256=code_sha_at[d["launch"]["git_head"]]["scripts/evaluate_paper_supplement.py"],
                exit_record_path=str(d["ops"] / f"{d['label']}.exit.json"), summary_sha256=d["summary_sha"],
                progress=e["progress"], raw_reward_sum_common=e["reward_sum_common"],
                scheme="B", mass_kg=d["mass_readback"], original_mass_kg=ORIG_MASS[case],
                mass_ratio=d["mass_readback"] / ORIG_MASS[case], retrained="not_applicable_no_training",
                hip_pitch_actuator=s["hip_pitch_actuator"], tf32=s["tf32"],
                sonic_mpjpe_mm_r0=r0r1(e["sonic_mpjpe_mm_per_robot"], 0), sonic_mpjpe_mm_r1=r0r1(e["sonic_mpjpe_mm_per_robot"], 1),
                sonic_mpjpe_pelvis_rel_mm_r0=r0r1(e["sonic_mpjpe_pelvis_rel_mm_per_robot"], 0),
                sonic_mpjpe_pelvis_rel_mm_r1=r0r1(e["sonic_mpjpe_pelvis_rel_mm_per_robot"], 1),
                sonic_fall_step_r0=e["sonic_fall_step_per_robot"][0], sonic_fall_step_r1=e["sonic_fall_step_per_robot"][1],
                sonic_fell_any=e["sonic_fell_any"],
                sonic_pelvis_dxy_final_m_r0=r0r1(e["sonic_pelvis_dxy_final_m_per_robot"], 0),
                sonic_pelvis_dxy_final_m_r1=r0r1(e["sonic_pelvis_dxy_final_m_per_robot"], 1),
                height_bias_cm=e["height_bias_m"] * 100, pd_target_minus_qstar_max_abs=e["pd_target_minus_qstar_max_abs"],
                launch_git_head=d["launch"]["git_head"], run_manifest_sha256=d["manifest_sha"],
                terminal_object_pos_error_m=t_pos, terminal_object_ori_error_rad=t_ori,
                gpu_model=d["gpu_model"], gpu_model_source=d["gpu_model_source"],
                b_variant=d["variant"], delta_cm=DELTA_CM[d["variant"]], position_feedback=FEEDBACK[d["variant"]],
                object_feedback=OBJECT_FEEDBACK[d["variant"]], last_waypoint_frac=sb["last_waypoint_frac"],
                planner_calls=e["planner_calls"], planner_seconds=e["planner_seconds"],
                planner_seconds_per_call=e["planner_seconds"] / e["planner_calls"] if e["planner_calls"] else None,
                planner_sha256=sb["planner_sha256"], contacts_sha256=sb.get("contacts_sha256")))
        print(d["group"], case, d["variant"], d["mass"], d["level"],
              sum(r["completed_reference"] for r in rows[-51:] if not r["is_nominal"]), flush=True)

    def summarize(rs, d):
        per = [r for r in rs if not r["is_nominal"]]
        assert len(per) == 50
        done = [r for r in per if r["completed_reference"]]
        term = Counter(k for r in per if not r["completed_reference"] for k, v in json.loads(r["termination_reason"]).items() if v)
        x = dict(experiment_group="System1-B-" + d["group"], task=d["case"], variant="schemeB_" + d["variant"], scheme="B",
                 b_variant=d["variant"], delta_cm=DELTA_CM[d["variant"]], position_feedback=FEEDBACK[d["variant"]],
                 object_feedback=OBJECT_FEEDBACK[d["variant"]],
                 mass_kg=d["mass_readback"], original_mass_kg=ORIG_MASS[d["case"]], retrained="not_applicable_no_training",
                 perturbation_level_cm=d["level"], run_id=rs[0]["run_id"], n_episodes=50, n_completed=len(done),
                 completion_pct=len(done) * 2, summary_sha256=d["summary_sha"],
                 progress_mean_all50=float(np.mean([r["progress"] for r in per])),
                 executed_steps_mean_all50=float(np.mean([r["executed_steps"] for r in per])),
                 n_terminated_joint_bad_tracking=term.get("joint_bad_tracking", 0),
                 n_sonic_fell_any=sum(bool(r["sonic_fell_any"]) for r in per))
        fail = [r for r in per if not r["completed_reference"]]
        x.update(n_fail_terminal_object_pos_gt_0p25m=sum(r["terminal_object_pos_error_m"] > 0.25 for r in fail),
                 n_fail_terminal_object_ori_gt_0p8rad=sum(r["terminal_object_ori_error_rad"] > 0.8 for r in fail),
                 n_fail_terminal_object_within_both_thresholds=sum(r["terminal_object_pos_error_m"] <= 0.25 and
                                                                   r["terminal_object_ori_error_rad"] <= 0.8 for r in fail))
        for k in ("object_pos_rmse_cm", "hand_pos_error_cm", "palm_error_deg"):
            v = [r[k] for r in done if r[k] is not None]
            x[k + "_mean"] = float(np.mean(v)) if v else None
            x[k + "_n_episodes"] = len(v)
        for k in ("sonic_mpjpe_mm", "sonic_mpjpe_pelvis_rel_mm"):
            v = [r[k + s] for r in per for s in ("_r0", "_r1") if r[k + s] is not None]
            x[k + "_mean_all50_both_robots"] = float(np.mean(v)) if v else None
        calls = [r["planner_calls"] for r in rs]
        secs = [r["planner_seconds"] for r in rs]
        x.update(planner_calls_mean_per_episode_all51=float(np.mean(calls)), planner_seconds_total_all51=float(np.sum(secs)),
                 planner_seconds_per_call_all51=float(np.sum(secs) / np.sum(calls)) if np.sum(calls) else None)
        nom = [r for r in rs if r["is_nominal"]][0]
        x.update(nominal_completed=nom["completed_reference"], nominal_progress=nom["progress"],
                 nominal_executed_steps=nom["executed_steps"], nominal_sonic_fell_any=nom["sonic_fell_any"],
                 nominal_object_pos_rmse_cm=nom["object_pos_rmse_cm"])
        return x

    by_run = defaultdict(list)
    for r in rows:
        by_run[r["run_id"]].append(r)
    summaries = [summarize(by_run[run_id(d)], d) for d in data]

    # nominal: same initial state in the +/-2 and +/-8 directories of one variant (B-open has +/-2 only)
    nominal_pairs = []
    nom_rows = [r for r in rows if r["is_nominal"]]
    for key in sorted({(r["experiment_group"], r["task"], r["b_variant"], r["mass_kg"]) for r in nom_rows}):
        rr = {r["perturbation_level_cm"]: r for r in nom_rows if (r["experiment_group"], r["task"], r["b_variant"], r["mass_kg"]) == key}
        if set(rr) != {2, 8}:
            continue
        a, b = rr[2], rr[8]
        with np.load(a["trajectory_path"]) as za, np.load(b["trajectory_path"]) as zb:
            same_shape = za["root_pos"].shape == zb["root_pos"].shape
            dmax = float(max(np.max(abs(za[k] - zb[k])) for k in ("root_pos", "dof_pos", "object_pos_w"))) if same_shape else None
        nominal_pairs.append(dict(experiment_group=key[0], task=key[1], b_variant=key[2], mass_kg=key[3], steps_pm2=a["executed_steps"],
                                  steps_pm8=b["executed_steps"], trajectory_max_abs_diff=dmax, identical=same_shape and dmax == 0.0))

    # preregistered (README_zh): B vs D paired by initial state, exact two-sided McNemar against each D seed (V3 Full)
    v3 = defaultdict(dict)
    with V3_EPISODES.open() as f:
        for r in csv.DictReader(f):
            if r["variant"] == "Full" and r["is_nominal"] == "False":
                v3[(r["task"], int(r["perturbation_level_cm"]), int(r["training_seed"]))][r["state_id"]] = r
    vs_d = []
    for d in data:
        if d["group"] not in ("main", "B-open"):
            continue
        b_rows = {r["state_id"]: r for r in by_run[run_id(d)] if not r["is_nominal"]}
        assert len(b_rows) == 50
        d_rates, block = [], []
        for seed in (721, 722, 723):
            drows = v3[(d["case"], d["level"], seed)]
            assert set(drows) == set(b_rows), (d["case"], d["level"], seed)
            assert all(drows[k]["state_file_sha256"] == b_rows[k]["state_file_sha256"] for k in b_rows)
            pa = [(b_rows[k]["completed_reference"], drows[k]["completed_reference"] == "True") for k in sorted(b_rows)]
            d_gpu = {gpu_provenance.v3_row(drows[k])[0] for k in drows}
            assert len(d_gpu) == 1
            d_gpu = d_gpu.pop()
            both = sum(x and y for x, y in pa)
            b_only = sum(x and not y for x, y in pa)
            d_only = sum(y and not x for x, y in pa)
            d_rates.append(2 * sum(y for _, y in pa))
            block.append(dict(task=d["case"], b_variant=d["variant"], perturbation_level_cm=d["level"], d_variant="Full", d_seed=seed,
                              n_pairs=50, b_completion_pct=2 * sum(x for x, _ in pa), d_completion_pct=2 * sum(y for _, y in pa),
                              both_complete=both, only_b_complete=b_only, only_d_complete=d_only, both_fail=50 - both - b_only - d_only,
                              mcnemar_exact_two_sided_p=float(binomtest(min(b_only, d_only), b_only + d_only, .5).pvalue)
                              if b_only + d_only else 1.0,
                              d_checkpoint_sha256=next(iter(drows.values()))["checkpoint_sha256"],
                              b_gpu_model=d["gpu_model"], d_gpu_model=d_gpu, gpu_models_differ=d["gpu_model"] != d_gpu,
                              state_file_sha256=next(iter(b_rows.values()))["state_file_sha256"]))
        for x in block:
            x.update(d_completion_pct_mean_3seeds=float(np.mean(d_rates)), d_completion_pct_sample_sd_3seeds=float(np.std(d_rates, ddof=1)))
        vs_d += block

    # paired against scheme A main (same state files): descriptive, not a preregistered test
    a_rows = {}
    with (ROOT / "results_system1/T01b/schemeA/episodes.csv").open() as f:
        for r in csv.DictReader(f):
            if r["experiment_group"] == "System1-main" and r["is_nominal"] == "False":
                a_rows[(r["task"], int(r["perturbation_level_cm"]), r["state_id"])] = r
    vs_a = []
    for d in data:
        if d["group"] not in ("main", "B-open"):
            continue
        pr = [(r, a_rows[(d["case"], d["level"], r["state_id"])]) for r in by_run[run_id(d)] if not r["is_nominal"]]
        assert len(pr) == 50 and all(b["state_file_sha256"] == a["state_file_sha256"] for b, a in pr)
        bb = [(b["completed_reference"], a["completed_reference"] == "True") for b, a in pr]
        vs_a.append(dict(task=d["case"], b_variant=d["variant"], perturbation_level_cm=d["level"], n_pairs=50,
                         b_completion_pct=2 * sum(x for x, _ in bb), a_completion_pct=2 * sum(y for _, y in bb),
                         only_b_complete=sum(x and not y for x, y in bb), only_a_complete=sum(y and not x for x, y in bb),
                         progress_mean_b=float(np.mean([b["progress"] for b, _ in pr])),
                         progress_mean_a=float(np.mean([float(a["progress"]) for _, a in pr])),
                         n_progress_b_gt_a=sum(b["progress"] > float(a["progress"]) + 1e-12 for b, a in pr),
                         n_progress_b_lt_a=sum(b["progress"] < float(a["progress"]) - 1e-12 for b, a in pr)))

    def flat(dd, p=""):
        o = {}
        for k, v in dd.items():
            if isinstance(v, dict):
                o.update(flat(v, p + k + "/"))
            else:
                o[p + k] = v
        return o
    # manifest diffs: every run against B0 of the same task and level at the original mass (B-open: B0 +/-2)
    base = {(d["case"], d["level"]): flat(read(d["dir"] / "run_manifest.json")) for d in data if d["group"] == "main" and d["variant"] == "B0"}
    diffs = []
    for d in data:
        if d["group"] == "main" and d["variant"] == "B0":
            continue
        new, ref = flat(read(d["dir"] / "run_manifest.json")), base[(d["case"], d["level"])]
        ch = {k: dict(B0=ref.get(k), this=new.get(k)) for k in sorted(set(ref) | set(new)) if ref.get(k) != new.get(k)}
        diffs.append(dict(run=str(d["dir"].relative_to(L)), against=f"post_train/B/main/{d['case']}_B0_pm{d['level']}", fields=ch))
    # and B0 against scheme A main (the scheme change itself)
    a_vs_b0 = []
    for case in CASES:
        for lvl in (2, 8):
            am = flat(read(L / f"T01b/schemeA_main/{case}_pm{lvl}/run_manifest.json"))
            bm = base[(case, lvl)]
            a_vs_b0.append(dict(run=f"post_train/B/main/{case}_B0_pm{lvl}", against=f"T01b/schemeA_main/{case}_pm{lvl}",
                                fields={k: dict(A=am.get(k), B0=bm.get(k)) for k in sorted(set(am) | set(bm)) if am.get(k) != bm.get(k)}))

    OUT.mkdir(parents=True, exist_ok=True)
    OUT_M.mkdir(parents=True, exist_ok=True)
    main_rows = [r for r in rows if r["experiment_group"] in ("System1-B-main", "System1-B-B-open")]
    csvsave(OUT / "episodes.csv", main_rows)
    csvsave(OUT / "per_task_summary.csv", [x for x in summaries if x["experiment_group"] != "System1-B-T04"])
    csvsave(OUT / "nominal_pm2_vs_pm8.csv", nominal_pairs)
    csvsave(OUT / "schemeB_vs_D_mcnemar.csv", vs_d)
    csvsave(OUT / "schemeB_vs_A_paired.csv", vs_a)
    csvsave(OUT / "attempts.csv", attempts())
    sweep = [r for r in rows if r["task"] in SWEEP and r["b_variant"] in SWEEP_VARIANTS]
    for r in sweep:
        r["seed"] = None
    csvsave(OUT_M / "mass_sweep_episodes_B.csv", sweep)
    sweep_summ = sorted([x for x in summaries if x["task"] in SWEEP and x["b_variant"] in SWEEP_VARIANTS],
                        key=lambda x: (x["task"], x["mass_kg"], x["b_variant"], x["perturbation_level_cm"]))
    csvsave(OUT_M / "mass_sweep_summary_B.csv", sweep_summ)
    csvsave(OUT / "mass_sweep/mass_sweep_summary_B.csv", sweep_summ)
    (OUT / "manifest_diffs_vs_B0.json").write_text(json.dumps(diffs, indent=1, default=str) + "\n")
    (OUT / "manifest_diffs_B0_vs_schemeA.json").write_text(json.dumps(a_vs_b0, indent=1, default=str) + "\n")
    configs = [dict(run=str(d["dir"].relative_to(L)), argv=d["launch"]["argv"], started=d["launch"]["started"],
                    ended=d["exit"].get("ended"), git_head=d["launch"]["git_head"], git_status=d["launch"]["git_status"],
                    git_diff_sha256=hashlib.sha256((d["launch"]["git_diff"] or "").encode()).hexdigest(),
                    states=d["launch"]["states"], states_sha256=d["launch"]["states_sha256"],
                    run_manifest_sha256=d["manifest_sha"], summary_sha256=d["summary_sha"], gpu_model=d["gpu_model"],
                    object_mass_readback_kg=d["mass_readback"], object_mass_variant=d["summary"].get("system1_object_mass_variant"),
                    scheme_b=d["summary"]["scheme_b"], sonic_pd=d["summary"]["sonic_pd"],
                    frozen_action_scales=d["summary"]["frozen_action_scales"], effort_limits=d["summary"]["effort_limits"],
                    vel_limits=d["summary"]["vel_limits"], armature=d["summary"]["armature"]) for d in data]
    (OUT / "run_configs.json").write_text(json.dumps(configs, indent=1) + "\n")
    prov = dict(robot_urdf=str(robot_urdf.relative_to(ROOT)), robot_urdf_sha256=sha(robot_urdf),
                sonic_checkpoint_sha256=sorted({d["summary"]["sonic_checkpoint_sha256"] for d in data}),
                sonic_spec_sha256=sorted({d["summary"]["sonic_spec_sha256"] for d in data}),
                planner_sha256=sorted({d["summary"]["scheme_b"]["planner_sha256"] for d in data}),
                contacts_sha256=sorted({str(d["summary"]["scheme_b"].get("contacts_sha256")) for d in data}),
                states_sha256=sorted({(d["case"], d["level"], d["summary"]["states_sha256"]) for d in data}),
                references=hm.refinfo, hand_fk_test=hm.fk_test, palm_postprocess_sha256=palm_src_sha,
                palm_output_dir=str(PALM_OUT.relative_to(ROOT)), evaluation_code_sha256_by_git_head=code_sha_at,
                launch_git_heads=dict(Counter(d["launch"]["git_head"] for d in data)),
                launch_git_status_nonempty=sum(bool(d["launch"]["git_status"]) for d in data),
                git_head_now=git("rev-parse", "HEAD").strip(), units=len(data), episodes=len(rows), main_rows=len(main_rows),
                sweep_rows=len(sweep), gpu_models=dict(Counter(d["gpu_model"] for d in data)),
                hand_status_counts=dict(Counter(r["hand_status"] for r in rows)),
                palm_status_counts=dict(Counter(r["palm_status"] for r in rows)),
                v3_episodes=str(V3_EPISODES), v3_episodes_sha256=sha(V3_EPISODES))
    (OUT / "provenance.json").write_text(json.dumps(prov, indent=1, default=list) + "\n")
    print("units", len(data), "episodes", len(rows), "main rows", len(main_rows), "sweep rows", len(sweep))


if __name__ == "__main__":
    main()
