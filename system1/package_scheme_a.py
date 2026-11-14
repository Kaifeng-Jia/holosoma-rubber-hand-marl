#!/usr/bin/env python3
"""Build the scheme A result tables (T01b main + hip_pitch sensitivity, T04 scheme A mass sweep).

Columns follow the V3 closeout ``results/episodes.csv`` (scripts/finalize_v3_results_20261104.py):
object errors from the evaluator summary, hand error by the same URDF FK + reference alignment,
palm error by the same V3 palm post-process (chair/bucket/table; push/pull targets missing, kick n/a).
Extra columns are listed in NEW_FIELDS and documented in the README.

  hssim_s1/bin/python -B system1/package_scheme_a.py
Reads logs_system1/ only; writes results_system1/T01b/schemeA/ and results_system1/T04/A/.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import sys
import types
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
from system1 import gpu_provenance  # noqa: E402
L = ROOT / "logs_system1"
OUT_A = ROOT / "results_system1/T01b/schemeA"
OUT_M = ROOT / "results_system1/T04/A"
V3RAW = Path("/home/kevin/holosoma-core4d-base/logs/V3_20261102/cloud_backup_20261104/raw")  # read only
PALM_OUT = L / "T01b/schemeA_palm_metrics"
V3_EPISODES = Path("/home/kevin/holosoma-core4d-base/paper_handoff_exports/V3_20261102/V3_GPU_CLOSEOUT_20261104/results/episodes.csv")
CASES = ("push", "pull", "kick", "chair", "bucket", "table")
ORIG_MASS = dict(push=20.0, pull=20.0, kick=20.0, chair=5.0, bucket=1.0, table=5.0)
SWEEP = dict(pull=(5.0, 10.0, 20.0, 40.0), table=(2.5, 5.0, 10.0, 20.0), bucket=(1.0, 2.0, 4.0, 8.0))
NEW_FIELDS = ["scheme", "mass_kg", "original_mass_kg", "mass_ratio", "retrained", "hip_pitch_actuator", "tf32",
              "sonic_mpjpe_mm_r0", "sonic_mpjpe_mm_r1", "sonic_mpjpe_pelvis_rel_mm_r0", "sonic_mpjpe_pelvis_rel_mm_r1",
              "sonic_fall_step_r0", "sonic_fall_step_r1", "sonic_fell_any", "sonic_pelvis_dxy_final_m_r0",
              "sonic_pelvis_dxy_final_m_r1", "height_bias_cm", "pd_target_minus_qstar_max_abs", "launch_git_head",
              "run_manifest_sha256", "terminal_object_pos_error_m", "terminal_object_ori_error_rad"]


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


EVALUATOR = "scripts/evaluate_paper_supplement.py"


def eval_code_sha256(launch: dict) -> str | None:
    """sha256 of the evaluator as it was when an evaluation started, from its launch record (git HEAD then); None if the
    evaluator had uncommitted changes at that time (git_status names it; the launch record keeps git_diff) or the commit is
    not in this clone. Replaces sha256 of the current file, which changes whenever the evaluator is edited later."""
    if EVALUATOR in (launch.get("git_status") or "") or not launch.get("git_head"):
        return None
    r = subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[1]), "show", f"{launch['git_head']}:{EVALUATOR}"],
                       capture_output=True)
    return hashlib.sha256(r.stdout).hexdigest() if r.returncode == 0 else None


def csvsave(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def units():
    """Every scheme A evaluation directory, with its group labels. All of them, no selection."""
    out = []
    for case in CASES:
        for lvl in (2, 8):
            out.append(dict(group="main", variant="schemeA", case=case, level=lvl, mass=ORIG_MASS[case],
                            dir=L / f"T01b/schemeA_main/{case}_pm{lvl}", ops=L / "T01b/schemeA_main/ops"))
    for case in ("push", "pull", "kick"):
        for lvl in (2, 8):
            out.append(dict(group="hip_pitch_sonic", variant="schemeA_hip_pitch_sonic", case=case, level=lvl,
                            mass=ORIG_MASS[case], dir=L / f"T01b/schemeA_hip_pitch_sonic/{case}_pm{lvl}",
                            ops=L / "T01b/schemeA_hip_pitch_sonic/ops"))
    for case, masses in SWEEP.items():
        for m in masses:
            if m == ORIG_MASS[case]:
                continue
            tag = f"{case}_m{m:g}kg".replace(".", "p")
            for lvl in (2, 8):
                out.append(dict(group="T04", variant="schemeA", case=case, level=lvl, mass=m,
                                dir=L / f"T04/A/{tag}/{case}_pm{lvl}", ops=L / f"T04/A/{tag}/ops"))
    return out


def load_unit(u):
    label = u["dir"].name
    ex, la = read(u["ops"] / f"{label}.exit.json"), read(u["ops"] / f"{label}.launch.json")
    s = read(u["dir"] / "summary.json")
    assert ex["returncode"] == 0 and ex["completed"] and s["complete"], u["dir"]
    eps = s["episodes"]
    assert len(eps) == 51 and len({e["state_id"] for e in eps}) == 51 and sum(e["kind"] == "perturbed" for e in eps) == 50
    assert la["states_sha256"] == s["states_sha256"]
    mass = float(s["physics"]["mass"][0][0])
    assert abs(mass - u["mass"]) < 1e-6, (u["dir"], mass, u["mass"])
    assert s["tf32"] is False and s["controller"] == "sonic"
    want_hip = "sonic" if u["variant"] == "schemeA_hip_pitch_sonic" else "ours"
    assert s["hip_pitch_actuator"] == want_hip, (u["dir"], s["hip_pitch_actuator"])
    return dict(u, label=label, summary=s, exit=ex, launch=la, summary_sha=sha(u["dir"] / "summary.json"),
                manifest_sha=sha(u["dir"] / "run_manifest.json"), mass_readback=mass)


class HandMetrics:
    """V3 hand-position error: rubber-hand link origins, 2 robots x 2 hands x all saved frames, cm."""

    def __init__(self):
        import torch
        from prepare_experiments_r2_cpu import BODY, HANDS, BatchFK, reference
        from system1.sonic.task_refs import PUSH_PROOF

        torch.set_num_threads(4)
        self.torch, self.BODY, self.HANDS, self.reference = torch, BODY, HANDS, reference
        self.fk = BatchFK()
        self.fk_test = self.fk.validate()
        self.push_spacing = float(read(PUSH_PROOF)["lateral_spacing_m"])
        self.refs, self.refinfo = {}, {}

    def ref(self, case):
        if case not in self.refs:
            names = self.BODY + self.HANDS
            r, path = self.reference(case, names, spacing=self.push_spacing)
            arr = {k: v.numpy() for k, v in r.sample(self.torch.arange(r.num_frames)).items()}
            arr["body_names"] = names
            self.refs[case] = arr
            self.refinfo[case] = dict(path=str(path), sha256=sha(path),
                                      lateral_spacing_m=self.push_spacing if case == "push" else None)
        return self.refs[case]

    def episode(self, case, traj):
        ref = self.ref(case)
        names = list(map(str, ref["body_names"]))
        hi = [names.index(x) for x in self.HANDS]
        with np.load(traj, allow_pickle=False) as z:
            f = z["reference_frame"].astype(int)
            offset = z["reference_object_pos_w"] - ref["object_pos_w"][f]
            assert np.max(abs(offset - offset[0])) < 3e-6, (traj, "reference alignment")
            expected = ref["agent_body_pos_w"][f][:, :, hi, :] + offset[:, None, None, :]
            actual = self.fk.poses(z["root_pos"], z["root_quat_xyzw"], z["dof_pos"], self.HANDS)[..., :3, 3]
        err = np.linalg.norm(actual - expected, axis=-1) * 100
        return float(err.mean()), len(f)


def palm_module():
    """The V3 palm post-process, unchanged except for its input/output roots (same as the V3 closeout did)."""
    src = V3RAW / "v3_palm_postprocess.py"
    code = src.read_text()
    a, b = "B=Path('/root/autodl-tmp')", "OUT=B/'v3_palm_metrics_20261104'"
    assert a in code and b in code
    code = code.replace(a, "B=Path(" + repr(str(V3RAW)) + ")").replace(b, "OUT=Path(" + repr(str(PALM_OUT)) + ")")
    PALM_OUT.mkdir(parents=True, exist_ok=True)
    mod = types.ModuleType("system1_v3_palm")
    exec(compile(code, str(src), "exec"), mod.__dict__)
    mod.tests()
    return mod, sha(src)


def r0r1(v, i):
    return None if v is None else v[i]


def main():
    data = [load_unit(u) for u in units()]
    hm = HandMetrics()
    palm, palm_src_sha = palm_module()
    robot_urdf = ROOT / "src/holosoma/holosoma/data/robots/g1/main_mesh_collision_rubberhand.urdf"
    eval_code = ROOT / "scripts/evaluate_paper_supplement.py"
    git = lambda *x: subprocess.run(["git", "-C", str(ROOT), *x], capture_output=True, text=True).stdout  # noqa: E731
    code_sha_at = {}
    for d in data:
        h = d["launch"]["git_head"]
        if h not in code_sha_at:
            code_sha_at[h] = {p: hashlib.sha256(subprocess.run(["git", "-C", str(ROOT), "show", f"{h}:{p}"],
                                                               capture_output=True).stdout).hexdigest()
                              for p in ("scripts/evaluate_paper_supplement.py", "system1/scheme_a.py",
                                        "system1/sonic_controller.py", "system1/mass_variants.py")}
    for d in data:  # Kevin 2026-11-10: every evaluation row records the GPU model
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
        for e in s["episodes"]:
            nom, done = e["kind"] == "nominal", e["completed_reference"]
            traj = Path(e["trajectory"])
            hand = hm.episode(case, traj) if (done and not nom) else None
            ph = palm_rows.get(e["state_id"])
            with np.load(traj, allow_pickle=False) as z:  # last saved state = terminal state before autoreset
                t_pos = float(np.linalg.norm(z["object_pos_w"][-1] - z["reference_object_pos_w"][-1]))
                t_ori = float(2 * np.arccos(min(1.0, abs(float(np.dot(z["object_quat_xyzw"][-1], z["reference_object_quat_xyzw"][-1]))))))
            palm_status = ("not_applicable" if case == "kick" else "excluded_nominal" if nom else
                           "not_completed" if not done else "complete" if ph else "missing")
            rows.append(dict(
                experiment_group="System1-" + d["group"], task=case, variant=d["variant"],
                run_id=f"S1A-{d['group']}-{case}-m{d['mass']:g}kg-pm{d['level']}", training_seed=None,
                model_id=s["model_id"], checkpoint_iteration=None, checkpoint_sha256=s["checkpoint_sha256"],
                perturbation_level_cm=d["level"], perturbation_level_deg=d["level"], state_id=e["state_id"],
                state_file_sha256=s["states_sha256"], is_nominal=nom, completed_reference=done,
                termination_reason=json.dumps(e["terminal_terms"], sort_keys=True), executed_steps=e["steps"],
                reference_horizon_steps=e["horizon_steps"], object_pos_rmse_cm=e["object_rmse_3d_m"] * 100,
                object_xy_rmse_cm=e["object_rmse_xy_m"] * 100, height_rmse_cm=e["height_rmse_m"] * 100,
                hand_pos_error_cm=hand[0] if hand else None,
                hand_status="complete" if hand else "excluded_nominal" if nom else "not_completed" if not done else "missing",
                palm_error_deg=ph["mean_deg"] if ph else None, palm_status=palm_status, n_metric_frames=e["steps"] + 1,
                palm_valid_hand_frames=ph["count"] if ph else 0, trajectory_path=str(traj), trajectory_sha256=sha(traj),
                config_sha256=d["manifest_sha"], reference_sha256=hm.refinfo[case]["sha256"],
                object_urdf_sha256=sha(obj_urdf),
                evaluation_code_sha256=code_sha_at[d["launch"]["git_head"]]["scripts/evaluate_paper_supplement.py"],
                exit_record_path=str(d["ops"] / f"{d['label']}.exit.json"), summary_sha256=d["summary_sha"],
                progress=e["progress"], raw_reward_sum_common=e["reward_sum_common"],
                scheme="A", mass_kg=d["mass_readback"], original_mass_kg=ORIG_MASS[case],
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
                gpu_model=d["gpu_model"], gpu_model_source=d["gpu_model_source"]))
        print(d["group"], case, d["mass"], d["level"], sum(r["completed_reference"] for r in rows[-51:] if not r["is_nominal"]), flush=True)

    def summarize(rs, d):
        per = [r for r in rs if not r["is_nominal"]]
        assert len(per) == 50
        done = [r for r in per if r["completed_reference"]]
        term = Counter(k for r in per if not r["completed_reference"] for k, v in json.loads(r["termination_reason"]).items() if v)
        x = dict(experiment_group="System1-" + d["group"], task=d["case"], variant=d["variant"], scheme="A",
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
        nom = [r for r in rs if r["is_nominal"]][0]
        x.update(nominal_completed=nom["completed_reference"], nominal_progress=nom["progress"],
                 nominal_executed_steps=nom["executed_steps"], nominal_sonic_fell_any=nom["sonic_fell_any"],
                 nominal_object_pos_rmse_cm=nom["object_pos_rmse_cm"])
        return x

    by_run = defaultdict(list)
    for r in rows:
        by_run[r["run_id"]].append(r)
    summaries = [summarize(by_run[f"S1A-{d['group']}-{d['case']}-m{d['mass']:g}kg-pm{d['level']}"], d) for d in data]

    # nominal is the same initial state in the +/-2 and +/-8 directories: record whether both runs agree
    nominal_rows = [r for r in rows if r["is_nominal"]]
    nominal_pairs = []
    for key in sorted({(r["experiment_group"], r["task"], r["mass_kg"]) for r in nominal_rows}):
        rr = {r["perturbation_level_cm"]: r for r in nominal_rows if (r["experiment_group"], r["task"], r["mass_kg"]) == key}
        a, b = rr[2], rr[8]
        with np.load(a["trajectory_path"]) as za, np.load(b["trajectory_path"]) as zb:
            same_shape = za["root_pos"].shape == zb["root_pos"].shape
            dmax = float(max(np.max(abs(za[k] - zb[k])) for k in ("root_pos", "dof_pos", "object_pos_w"))) if same_shape else None
        nominal_pairs.append(dict(experiment_group=key[0], task=key[1], mass_kg=key[2], steps_pm2=a["executed_steps"],
                                  steps_pm8=b["executed_steps"], trajectory_max_abs_diff=dmax,
                                  identical=same_shape and dmax == 0.0))

    # hip_pitch sensitivity: paired by state_id against the main setting
    paired = []
    idx = {(r["variant"], r["task"], r["perturbation_level_cm"], r["state_id"]): r for r in rows
           if r["experiment_group"] in ("System1-main", "System1-hip_pitch_sonic") and not r["is_nominal"]}
    for case in ("push", "pull", "kick"):
        for lvl in (2, 8):
            ps = [(idx[("schemeA", case, lvl, f"perturbed_{i:03d}")], idx[("schemeA_hip_pitch_sonic", case, lvl, f"perturbed_{i:03d}")])
                  for i in range(50)]
            assert all(m["state_file_sha256"] == h["state_file_sha256"] for m, h in ps)
            a = sum(m["completed_reference"] and h["completed_reference"] for m, h in ps)
            b = sum(m["completed_reference"] and not h["completed_reference"] for m, h in ps)
            c = sum(not m["completed_reference"] and h["completed_reference"] for m, h in ps)
            paired.append(dict(task=case, perturbation_level_cm=lvl, n_pairs=50, both_complete=a, only_main_complete=b,
                               only_hip_pitch_sonic_complete=c, both_fail=50 - a - b - c,
                               p_exact=float(binomtest(min(b, c), b + c, .5).pvalue) if b + c else 1.0,
                               progress_mean_main=float(np.mean([m["progress"] for m, _ in ps])),
                               progress_mean_hip_pitch_sonic=float(np.mean([h["progress"] for _, h in ps]))))

    # preregistered (README_zh): A vs D paired by initial state, exact two-sided McNemar against each D seed
    v3 = defaultdict(dict)
    with V3_EPISODES.open() as f:
        for r in csv.DictReader(f):
            if r["variant"] == "Full" and r["is_nominal"] == "False":
                v3[(r["task"], int(r["perturbation_level_cm"]), int(r["training_seed"]))][r["state_id"]] = r
    vs_d = []
    for case in CASES:
        for lvl in (2, 8):
            a_rows = {r["state_id"]: r for r in rows if r["experiment_group"] == "System1-main" and r["task"] == case
                      and r["perturbation_level_cm"] == lvl and not r["is_nominal"]}
            assert len(a_rows) == 50
            d_rates = []
            for seed in (721, 722, 723):
                drows = v3[(case, lvl, seed)]
                assert set(drows) == set(a_rows), (case, lvl, seed)
                assert all(drows[k]["state_file_sha256"] == a_rows[k]["state_file_sha256"] for k in a_rows), (case, lvl, seed)
                pa = [(a_rows[k]["completed_reference"], drows[k]["completed_reference"] == "True") for k in sorted(a_rows)]
                a_gpu = {a_rows[k]["gpu_model"] for k in a_rows}
                d_gpu = {gpu_provenance.v3_row(drows[k])[0] for k in drows}
                assert len(a_gpu) == 1 and len(d_gpu) == 1, (case, lvl, seed, a_gpu, d_gpu)
                a_gpu, d_gpu = a_gpu.pop(), d_gpu.pop()
                both = sum(x and y for x, y in pa)
                a_only = sum(x and not y for x, y in pa)
                d_only = sum(y and not x for x, y in pa)
                d_rates.append(2 * sum(y for _, y in pa))
                vs_d.append(dict(task=case, perturbation_level_cm=lvl, d_variant="Full", d_seed=seed, n_pairs=50,
                                 a_completion_pct=2 * sum(x for x, _ in pa), d_completion_pct=2 * sum(y for _, y in pa),
                                 both_complete=both, only_a_complete=a_only, only_d_complete=d_only,
                                 both_fail=50 - both - a_only - d_only,
                                 mcnemar_exact_two_sided_p=float(binomtest(min(a_only, d_only), a_only + d_only, .5).pvalue)
                                 if a_only + d_only else 1.0,
                                 d_checkpoint_sha256=next(iter(drows.values()))["checkpoint_sha256"],
                                 a_gpu_model=a_gpu, d_gpu_model=d_gpu, gpu_models_differ=a_gpu != d_gpu,
                                 state_file_sha256=next(iter(a_rows.values()))["state_file_sha256"]))
            for x in vs_d[-3:]:
                x.update(d_completion_pct_mean_3seeds=float(np.mean(d_rates)),
                         d_completion_pct_sample_sd_3seeds=float(np.std(d_rates, ddof=1)))

    # manifest diffs: each sensitivity / T04 run against the main run of the same task and level
    def flat(d, p=""):
        o = {}
        for k, v in d.items():
            if isinstance(v, dict):
                o.update(flat(v, p + k + "/"))
            else:
                o[p + k] = v
        return o
    base = {(d["case"], d["level"]): flat(read(d["dir"] / "run_manifest.json")) for d in data if d["group"] == "main"}
    diffs = []
    for d in data:
        if d["group"] == "main":
            continue
        new, ref = flat(read(d["dir"] / "run_manifest.json")), base[(d["case"], d["level"])]
        ch = {k: dict(main=ref.get(k), this=new.get(k)) for k in sorted(set(ref) | set(new)) if ref.get(k) != new.get(k)}
        diffs.append(dict(run=f"{d['group']}/{d['dir'].relative_to(L)}", fields=ch))

    OUT_A.mkdir(parents=True, exist_ok=True)
    OUT_M.mkdir(parents=True, exist_ok=True)
    t01b = [r for r in rows if r["experiment_group"] in ("System1-main", "System1-hip_pitch_sonic")]
    csvsave(OUT_A / "episodes.csv", t01b)
    csvsave(OUT_A / "per_task_summary.csv", [x for x in summaries if x["experiment_group"] != "System1-T04"])
    csvsave(OUT_A / "nominal_pm2_vs_pm8.csv", nominal_pairs)
    csvsave(OUT_A / "hip_pitch_sensitivity_paired.csv", paired)
    csvsave(OUT_A / "schemeA_vs_D_mcnemar.csv", vs_d)
    sweep = [r for r in rows if r["task"] in SWEEP and r["variant"] == "schemeA"]
    for r in sweep:
        # T04 column delta = scheme B1 contact offset (T02); not applicable to A (corrected 2026-11-10: was the
        # perturbation level, which is already in perturbation_level_cm)
        r.update(delta_cm=None, seed=None)
    csvsave(OUT_M / "mass_sweep_episodes.csv", sweep)
    csvsave(OUT_M / "mass_sweep_summary.csv",
            sorted([x for x in summaries if x["task"] in SWEEP and x["variant"] == "schemeA"],
                   key=lambda x: (x["task"], x["mass_kg"], x["perturbation_level_cm"])))
    (OUT_A / "manifest_diffs_vs_main.json").write_text(json.dumps(diffs, indent=1) + "\n")
    configs = [dict(run=str(d["dir"].relative_to(L)), argv=d["launch"]["argv"], started=d["launch"]["started"],
                    ended=d["exit"].get("ended"), git_head=d["launch"]["git_head"], git_status=d["launch"]["git_status"],
                    states=d["launch"]["states"], states_sha256=d["launch"]["states_sha256"],
                    run_manifest_sha256=d["manifest_sha"], summary_sha256=d["summary_sha"],
                    object_mass_readback_kg=d["mass_readback"],
                    object_mass_variant=d["summary"].get("system1_object_mass_variant"),
                    sonic_pd=d["summary"]["sonic_pd"], frozen_action_scales=d["summary"]["frozen_action_scales"],
                    effort_limits=d["summary"]["effort_limits"], vel_limits=d["summary"]["vel_limits"],
                    armature=d["summary"]["armature"]) for d in data]
    (OUT_A / "run_configs.json").write_text(json.dumps(configs, indent=1) + "\n")
    prov = dict(robot_urdf=str(robot_urdf.relative_to(ROOT)), robot_urdf_sha256=sha(robot_urdf),
                sonic_checkpoint_sha256=sorted({d["summary"]["sonic_checkpoint_sha256"] for d in data}),
                sonic_spec_sha256=sorted({d["summary"]["sonic_spec_sha256"] for d in data}),
                states_sha256=sorted({(d["case"], d["level"], d["summary"]["states_sha256"]) for d in data}),
                references=hm.refinfo, hand_fk_test=hm.fk_test, palm_postprocess_source=str(V3RAW / "v3_palm_postprocess.py"),
                palm_postprocess_source_sha256=palm_src_sha, palm_output_dir=str(PALM_OUT.relative_to(ROOT)),
                evaluation_code_sha256_by_git_head=code_sha_at, evaluation_code_now_sha256=sha(eval_code),
                git_head_now=git("rev-parse", "HEAD").strip(), units=len(data), episodes=len(rows),
                hand_status_counts=dict(Counter(r["hand_status"] for r in rows)),
                palm_status_counts=dict(Counter(r["palm_status"] for r in rows)),
                v3_episodes=str(V3_EPISODES), v3_episodes_sha256=sha(V3_EPISODES))
    (OUT_A / "provenance.json").write_text(json.dumps(prov, indent=1, default=list) + "\n")
    print("units", len(data), "episodes", len(rows), "T01b rows", len(t01b), "sweep rows", len(sweep))


if __name__ == "__main__":
    main()
