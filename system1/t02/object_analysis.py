#!/usr/bin/env python3
"""T02 preregistration section 9 (written and committed before any of this was computed, commit d6efd67).
CPU only; reads existing trajectories / sonic.npz / references, never simulates.

  push      9.1  object push distance (projection of object displacement on the reference displacement) and ratio;
                 A, B all episodes; D (V3 Full seeds 721-723, T04-D direct seed 721) at +/-2 at matched steps
  objcheck  9.2  A vs B object-error record check on the same task / level / state_id; termination step recompute
  fall      9.3  SONIC fall rule broken down by criterion (pelvis height, orientation, wrist/ankle height); A and B
  pitch     9.3  torso pitch error in steps 1..50 (FK pelvis * Rz Rx Ry of waist joints, checked on the reference)

    hssim_s1/bin/python -B system1/t02/object_analysis.py all --out results_system1/T02
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

A_CSV = ROOT / "results_system1/T01b/schemeA/episodes.csv"
A_MASS_CSV = ROOT / "results_system1/T04/A/mass_sweep_episodes.csv"
D_MASS_CSV = ROOT / "results_system1/T04/D/mass_sweep_episodes_D.csv"
V3 = Path("/home/kevin/holosoma-core4d-base/paper_handoff_exports/V3_20261102/V3_GPU_CLOSEOUT_20261104/results/episodes.csv")
B_ROOT = ROOT / "logs_system1/post_train/B"
ORIG_MASS = dict(push=20.0, pull=20.0, kick=20.0, chair=5.0, bucket=1.0, table=5.0)
B_TASKS = ("push", "pull", "table", "chair", "bucket")
MIN_DREF = 0.05          # m; push_ratio only when |reference displacement| >= this (= drift MIN_DQ)
OBJ_TERM = 0.25          # m; JointBadTrackingZOnly bad_object_pos_threshold
FALL_PEL, FALL_ORI2, FALL_EE = 0.25, 1.0, 0.25
PITCH_STEPS = 50
FK_TOL = 1e-3            # rad
NY = ZoneInfo("America/New_York")


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()


def now():
    return datetime.now(NY).isoformat(timespec="seconds")


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fields})
    print("wrote", path, len(rows), "rows", flush=True)


def fnum(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else float(x)


# ---------------------------------------------------------------- episodes
def _ep(scheme, variant, group, task, mass, level, state_id, is_nominal, completed, steps, traj, seed="", run=""):
    return dict(scheme=scheme, variant=variant, experiment_group=group, task=task, mass_kg=float(mass), level=int(level),
                state_id=state_id, is_nominal=bool(is_nominal), completed=bool(completed), steps=int(steps),
                trajectory=str(traj), seed=str(seed), run=run)


def _csv_eps(path, scheme=None, level=None, seeds=None, variant=None):
    out = []
    for r in csv.DictReader(open(path)):
        if scheme and r["scheme"] != scheme:
            continue
        if variant and r["variant"] != variant:
            continue
        if level is not None and int(r["perturbation_level_cm"]) != level:
            continue
        if seeds and r.get("training_seed", r.get("seed")) not in seeds:
            continue
        out.append(_ep(r["scheme"], r["variant"], r["experiment_group"], r["task"], r["mass_kg"], r["perturbation_level_cm"],
                       r["state_id"], r["is_nominal"] == "True", r["completed_reference"] == "True", r["executed_steps"],
                       r["trajectory_path"], seed=r.get("training_seed") or r.get("seed") or "", run=r.get("run_id", "")))
    return out


def b_episodes(sub):
    out = []
    for s in sorted((B_ROOT / sub).glob("*/summary.json")):
        name = s.parent.name
        if ".attempt" in name:
            continue
        j = json.loads(s.read_text())
        if not j.get("complete"):
            continue
        case, var = name.split("_")[0], name.split("_")[1]
        lvl = int(name.rsplit("_pm", 1)[1])
        m = re.search(r"_m(\d+(?:p\d+)?)kg_", name)
        mass = float(m.group(1).replace("p", ".")) if m else ORIG_MASS[case]
        for e in j["episodes"]:
            out.append(_ep("B", var, f"System1-B-{sub}", case, mass, lvl, e["state_id"], e["kind"] == "nominal",
                           e["completed_reference"], e["steps"], e["trajectory"], run=f"B/{sub}/{name}"))
    return out


def v3_episodes():
    """paper D: V3 closeout Full, +/-2, seeds 721/722/723 (the V3 table has no scheme / mass columns)."""
    out = []
    for r in csv.DictReader(open(V3)):
        if r["variant"] != "Full" or int(r["perturbation_level_cm"]) != 2 or r["training_seed"] not in ("721", "722", "723"):
            continue
        out.append(_ep("D", "Full", "V3-closeout (paper D)", r["task"], ORIG_MASS[r["task"]], 2, r["state_id"], r["is_nominal"] == "True",
                       r["completed_reference"] == "True", r["executed_steps"], r["trajectory_path"], seed=r["training_seed"], run=r["run_id"]))
    return out


def all_episodes(include_b_mass=True):
    seen, eps = set(), []

    def add(lst):
        for e in lst:
            if e["trajectory"] not in seen:
                seen.add(e["trajectory"])
                eps.append(e)
    add(_csv_eps(A_CSV))
    add(_csv_eps(A_MASS_CSV))
    add(b_episodes("main"))
    if include_b_mass:
        add(b_episodes("mass"))
    add(v3_episodes())
    add(_csv_eps(D_MASS_CSV, scheme="D", level=2))
    return eps


def load(path, keys=None):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].astype(np.float64) if z[k].dtype.kind == "f" else z[k] for k in (keys or z.files)}


# ---------------------------------------------------------------- 9.1 push distance
def push_at(obj, ref, k):
    do, dr = obj[k] - obj[0], ref[k] - ref[0]
    n = float(np.linalg.norm(dr))
    pd = float(do @ dr / n) if n > 1e-12 else None
    ratio = pd / n if (pd is not None and n >= MIN_DREF) else None
    return dict(t=int(k), ref_disp_m=n, obj_disp_m=float(np.linalg.norm(do)), push_distance_m=pd, push_ratio=ratio)


def traj_obj(e, cache):
    if e["trajectory"] not in cache:
        z = load(e["trajectory"], ["object_pos_w", "reference_object_pos_w", "episode_step"])
        if len(z["object_pos_w"]) != e["steps"] + 1 or int(z["episode_step"][-1]) != e["steps"]:
            raise RuntimeError(f"rows/steps mismatch {e['trajectory']}")
        cache[e["trajectory"]] = (z["object_pos_w"], z["reference_object_pos_w"], sha(e["trajectory"]))
    return cache[e["trajectory"]]


def mstats(v):
    v = [x for x in v if x is not None]
    return (float(np.mean(v)), float(np.median(v)), len(v)) if v else (None, None, 0)


def cmd_push(out, eps):
    cache, rows = {}, []
    for e in eps:
        obj, ref, h = traj_obj(e, cache)
        rows.append(dict(scheme=e["scheme"], variant=e["variant"], experiment_group=e["experiment_group"], seed=e["seed"],
                         task=e["task"], mass_kg=e["mass_kg"], original_mass_kg=ORIG_MASS[e["task"]], perturbation_level_cm=e["level"],
                         state_id=e["state_id"], is_nominal=e["is_nominal"], completed_reference=e["completed"],
                         executed_steps=e["steps"], **{k: v for k, v in push_at(obj, ref, len(obj) - 1).items() if k != "t"},
                         run=e["run"], trajectory_path=e["trajectory"], trajectory_sha256=h))
    write_csv(out / "object_push/episodes_push.csv", rows)

    # summary per scheme x variant x group x seed x task x mass x level
    grp = defaultdict(list)
    for r in rows:
        grp[(r["scheme"], r["variant"], r["experiment_group"], r["seed"], r["task"], r["mass_kg"], r["perturbation_level_cm"])].append(r)
    summ = []
    for k, rs in sorted(grp.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        pert = [r for r in rs if not r["is_nominal"]]
        nom = [r for r in rs if r["is_nominal"]]
        pm, pmed, _ = mstats([r["push_distance_m"] for r in pert])
        rm, rmed, nr = mstats([r["push_ratio"] for r in pert])
        summ.append(dict(scheme=k[0], variant=k[1], experiment_group=k[2], seed=k[3], task=k[4], mass_kg=k[5], perturbation_level_cm=k[6],
                         n_perturbed=len(pert), n_completed=sum(r["completed_reference"] for r in pert),
                         push_distance_mean_m=pm, push_distance_median_m=pmed, push_ratio_mean=rm, push_ratio_median=rmed,
                         n_ratio_defined=nr, n_ratio_undefined=len(pert) - nr,
                         ref_disp_mean_m=mstats([r["ref_disp_m"] for r in pert])[0], obj_disp_mean_m=mstats([r["obj_disp_m"] for r in pert])[0],
                         executed_steps_mean=float(np.mean([r["executed_steps"] for r in pert])) if pert else None,
                         nominal_push_distance_m=nom[0]["push_distance_m"] if nom else None, nominal_push_ratio=nom[0]["push_ratio"] if nom else None,
                         nominal_executed_steps=nom[0]["executed_steps"] if nom else None))
    write_csv(out / "object_push/summary_push.csv", summ)

    # D at matched steps (+/-2 only)
    dkey = {(e["task"], e["mass_kg"], e["seed"], e["state_id"]): e for e in eps if e["scheme"] == "D"}
    seeds_for = defaultdict(set)
    for (t, m, s, _) in dkey:
        seeds_for[(t, m)].add(s)
    mrows = []
    for e in eps:
        if e["scheme"] == "D" or e["level"] != 2:
            continue
        obj, ref, _ = traj_obj(e, cache)
        x = push_at(obj, ref, len(obj) - 1)
        for s in sorted(seeds_for[(e["task"], e["mass_kg"])]):
            d = dkey.get((e["task"], e["mass_kg"], s, e["state_id"]))
            if d is None:
                raise RuntimeError(f"no D episode for {e['task']} {e['mass_kg']} {s} {e['state_id']}")
            dobj, dref, dh = traj_obj(d, cache)
            k = min(e["steps"], d["steps"])
            dm = push_at(dobj, dref, k)
            mrows.append(dict(compare_scheme=e["scheme"], compare_variant=e["variant"], compare_group=e["experiment_group"],
                              task=e["task"], mass_kg=e["mass_kg"], perturbation_level_cm=2, state_id=e["state_id"], is_nominal=e["is_nominal"],
                              d_seed=s, d_variant=d["variant"], t_end_compare=e["steps"], t_end_d=d["steps"], matched_step=k,
                              d_ended_before=d["steps"] < e["steps"], d_completed_reference=d["completed"],
                              compare_push_distance_m=x["push_distance_m"], compare_push_ratio=x["push_ratio"], compare_ref_disp_m=x["ref_disp_m"],
                              d_push_distance_m=dm["push_distance_m"], d_push_ratio=dm["push_ratio"], d_ref_disp_m=dm["ref_disp_m"],
                              d_trajectory_path=d["trajectory"], d_trajectory_sha256=dh))
    write_csv(out / "object_push/d_matched.csv", mrows)

    g2 = defaultdict(list)
    for r in mrows:
        if not r["is_nominal"]:
            g2[(r["compare_scheme"], r["compare_variant"], r["compare_group"], r["task"], r["mass_kg"])].append(r)
    ms = []
    for k, rs in sorted(g2.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        seeds = sorted({r["d_seed"] for r in rs})
        base = [r for r in rs if r["d_seed"] == seeds[0]]
        row = dict(compare_scheme=k[0], compare_variant=k[1], compare_group=k[2], task=k[3], mass_kg=k[4], perturbation_level_cm=2,
                   n_perturbed=len(base), compare_push_distance_mean_m=mstats([r["compare_push_distance_m"] for r in base])[0],
                   compare_push_ratio_mean=mstats([r["compare_push_ratio"] for r in base])[0],
                   compare_push_ratio_median=mstats([r["compare_push_ratio"] for r in base])[1],
                   compare_n_ratio_defined=mstats([r["compare_push_ratio"] for r in base])[2], d_seeds=" ".join(seeds))
        pdm, prm = [], []
        for s in seeds:
            rr = [r for r in rs if r["d_seed"] == s]
            a, _, _ = mstats([r["d_push_distance_m"] for r in rr])
            b, bmed, nb = mstats([r["d_push_ratio"] for r in rr])
            row.update({f"d{s}_push_distance_mean_m": a, f"d{s}_push_ratio_mean": b, f"d{s}_push_ratio_median": bmed,
                        f"d{s}_n_ratio_defined": nb, f"d{s}_n_ended_before": sum(r["d_ended_before"] for r in rr)})
            pdm.append(a)
            prm.append(b)
        row.update(d_push_distance_mean_over_seeds_m=float(np.mean(pdm)), d_push_distance_sd_over_seeds_m=float(np.std(pdm, ddof=1)) if len(pdm) > 1 else None,
                   d_push_ratio_mean_over_seeds=float(np.mean([p for p in prm if p is not None])) if any(p is not None for p in prm) else None,
                   d_push_ratio_sd_over_seeds=float(np.std([p for p in prm if p is not None], ddof=1)) if sum(p is not None for p in prm) > 1 else None)
        ms.append(row)
    write_csv(out / "object_push/d_matched_summary.csv", ms)
    return dict(n_episodes=len(rows), n_matched_rows=len(mrows))


# ---------------------------------------------------------------- 9.2 object error record check
def first_above(e, thr):
    i = np.nonzero(e > thr)[0]
    return int(i[0]) if len(i) else None


def term_recompute(path):
    """object position error with the reference at reward_reference_frame (termination alignment)."""
    z = load(path, ["object_pos_w", "reference_object_pos_w", "reference_frame", "reward_reference_frame"])
    row_of = {int(f): i for i, f in enumerate(z["reference_frame"])}
    rf = z["reward_reference_frame"].astype(int)
    e = np.full(len(rf), np.nan)
    for t in range(1, len(rf)):
        if int(rf[t]) in row_of:
            e[t] = np.linalg.norm(z["object_pos_w"][t] - z["reference_object_pos_w"][row_of[int(rf[t])]])
    return e


def cmd_objcheck(out, eps):
    A = {(e["task"], e["level"], e["state_id"]): e for e in eps if e["scheme"] == "A" and e["experiment_group"] == "System1-main"
         and e["mass_kg"] == ORIG_MASS[e["task"]]}
    B = [e for e in eps if e["scheme"] == "B" and e["experiment_group"] == "System1-B-main"]
    cache, per, curves = {}, [], []
    for b in B:
        a = A[(b["task"], b["level"], b["state_id"])]
        ao, ar, _ = traj_obj(a, cache)
        bo, br, _ = traj_obj(b, cache)
        K = min(len(ao), len(bo))
        ea, eb = np.linalg.norm(ao - ar, axis=1), np.linalg.norm(bo - br, axis=1)
        per.append(dict(task=b["task"], perturbation_level_cm=b["level"], b_variant=b["variant"], state_id=b["state_id"], is_nominal=b["is_nominal"],
                        a_steps=a["steps"], b_steps=b["steps"], common_rows=K,
                        ref_obj_pos_max_abs_diff_m=float(np.abs(ar[:K] - br[:K]).max()), initial_obj_pos_diff_m=float(np.linalg.norm(ao[0] - bo[0])),
                        obj_err_max_abs_diff_common_m=float(np.abs(ea[:K] - eb[:K]).max()),
                        a_first_step_err_gt_0p05=first_above(ea, 0.05), b_first_step_err_gt_0p05=first_above(eb, 0.05),
                        a_first_step_err_gt_0p10=first_above(ea, 0.10), b_first_step_err_gt_0p10=first_above(eb, 0.10),
                        a_err_terminal_m=float(ea[-1]), b_err_terminal_m=float(eb[-1]),
                        a_obj_disp_terminal_m=float(np.linalg.norm(ao[-1] - ao[0])), b_obj_disp_terminal_m=float(np.linalg.norm(bo[-1] - bo[0]))))
        if b["is_nominal"]:
            for t in range(len(bo)):
                curves.append(dict(task=b["task"], perturbation_level_cm=b["level"], scheme="B", variant=b["variant"], step=t, obj_err_m=float(eb[t]),
                                   obj_disp_m=float(np.linalg.norm(bo[t] - bo[0])), ref_disp_m=float(np.linalg.norm(br[t] - br[0]))))
    for (task, lvl, sid), a in sorted(A.items()):
        if sid != "nominal" or task not in B_TASKS:
            continue
        ao, ar, _ = traj_obj(a, cache)
        ea = np.linalg.norm(ao - ar, axis=1)
        for t in range(len(ao)):
            curves.append(dict(task=task, perturbation_level_cm=lvl, scheme="A", variant="schemeA", step=t, obj_err_m=float(ea[t]),
                               obj_disp_m=float(np.linalg.norm(ao[t] - ao[0])), ref_disp_m=float(np.linalg.norm(ar[t] - ar[0]))))
    write_csv(out / "object_error_check/a_vs_b_per_state.csv", per)
    write_csv(out / "object_error_check/nominal_error_curves.csv", curves)

    term = []
    for e in [x for x in eps if x["scheme"] in ("A", "B") and x["mass_kg"] == ORIG_MASS[x["task"]]
              and x["experiment_group"] in ("System1-main", "System1-B-main") and not x["completed"]]:
        et = term_recompute(e["trajectory"])
        if not (et[-1] > OBJ_TERM):
            continue
        fa = first_above(np.nan_to_num(et, nan=-1.0), OBJ_TERM)
        term.append(dict(scheme=e["scheme"], variant=e["variant"], task=e["task"], perturbation_level_cm=e["level"], state_id=e["state_id"],
                         executed_steps=e["steps"], recomputed_first_step_obj_err_gt_0p25=fa, matches=fa == e["steps"],
                         terminal_err_reward_frame_m=float(et[-1])))
    write_csv(out / "object_error_check/termination_recompute.csv", term)
    failed = [x for x in eps if x["scheme"] in ("A", "B") and x["mass_kg"] == ORIG_MASS[x["task"]]
              and x["experiment_group"] in ("System1-main", "System1-B-main") and not x["completed"]]
    tr = termination_reasons(failed, TorsoRefs())
    write_csv(out / "object_error_check/termination_reasons.csv", tr)
    g = defaultdict(list)
    for r in tr:
        if not r["is_nominal"]:
            g[(r["scheme"], r["variant"], r["task"], r["perturbation_level_cm"])].append(r)
    write_csv(out / "object_error_check/termination_reasons_summary.csv",
              [dict(scheme=k[0], variant=k[1], task=k[2], perturbation_level_cm=k[3], n_failed=len(rs),
                    obj_pos=sum(r["term_obj_pos"] for r in rs), obj_ori=sum(r["term_obj_ori"] for r in rs),
                    robot_torso=sum(r["term_robot_torso"] for r in rs),
                    only_robot_torso=sum(r["term_robot_torso"] and not r["term_obj_pos"] and not r["term_obj_ori"] for r in rs),
                    unexplained=sum(not r["explained"] for r in rs))
               for k, rs in sorted(g.items(), key=lambda kv: tuple(str(x) for x in kv[0]))])
    s = dict(n_pairs=len(per), ref_obj_pos_max_abs_diff_m=max(r["ref_obj_pos_max_abs_diff_m"] for r in per),
             initial_obj_pos_diff_max_m=max(r["initial_obj_pos_diff_m"] for r in per),
             n_termination_checked=len(term), n_termination_match=sum(r["matches"] for r in term),
             termination_mismatch=[r for r in term if not r["matches"]][:20],
             n_failed_reason_checked=len(tr), n_failed_unexplained=sum(not r["explained"] for r in tr))
    (out / "object_error_check/check_summary.json").write_text(json.dumps(s, indent=1) + "\n")
    print(json.dumps({k: v for k, v in s.items() if k != "termination_mismatch"}), flush=True)
    return s


# ---------------------------------------------------------------- 9.3 fall rule breakdown
def cmd_fall(out, eps):
    sel = [e for e in eps if e["scheme"] in ("A", "B") and e["experiment_group"] in ("System1-main", "System1-B-main")
           and e["mass_kg"] == ORIG_MASS[e["task"]]]
    rows, bad = [], 0
    for e in sel:
        sp = e["trajectory"].replace(".trajectory.npz", ".sonic.npz")
        z = load(sp)
        pel, ori, ee, fall = z["pelvis_dz"], z["pelvis_ori_err2"], z["ee_dz"], z["fall"].astype(bool)
        rule = (np.abs(pel) > FALL_PEL) | (ori > FALL_ORI2) | (ee > FALL_EE)
        if not np.array_equal(rule, fall):
            bad += 1
        for r in range(fall.shape[1]):
            i = np.nonzero(fall[:, r])[0]
            i0 = int(i[0]) if len(i) else None
            rows.append(dict(scheme=e["scheme"], variant=e["variant"], task=e["task"], perturbation_level_cm=e["level"], state_id=e["state_id"],
                             is_nominal=e["is_nominal"], robot=r, executed_steps=e["steps"], fell=i0 is not None,
                             first_fall_step=None if i0 is None else i0 + 1,
                             first_pelvis_height=None if i0 is None else bool(abs(pel[i0, r]) > FALL_PEL),
                             first_orientation=None if i0 is None else bool(ori[i0, r] > FALL_ORI2),
                             first_wrist_ankle_height=None if i0 is None else bool(ee[i0, r] > FALL_EE),
                             any_pelvis_height=bool((np.abs(pel[:, r]) > FALL_PEL).any()), any_orientation=bool((ori[:, r] > FALL_ORI2).any()),
                             any_wrist_ankle_height=bool((ee[:, r] > FALL_EE).any()),
                             max_pelvis_dz_m=float(np.abs(pel[:, r]).max()), max_ori_err2=float(ori[:, r].max()), max_ee_dz_m=float(ee[:, r].max()),
                             sonic_npz=sp, sonic_npz_sha256=sha(sp)))
    write_csv(out / "fall_reasons/per_episode_robot.csv", rows)
    g = defaultdict(list)
    for r in rows:
        if not r["is_nominal"]:
            g[(r["scheme"], r["variant"], r["task"], r["perturbation_level_cm"])].append(r)
    summ = []
    for k, rs in sorted(g.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        f = [r for r in rs if r["fell"]]
        summ.append(dict(scheme=k[0], variant=k[1], task=k[2], perturbation_level_cm=k[3], n_episode_robots=len(rs),
                         n_episodes=len({r["state_id"] for r in rs}), n_episodes_fell_any=len({r["state_id"] for r in f}), n_robot_fell=len(f),
                         first_pelvis_height=sum(r["first_pelvis_height"] for r in f), first_orientation=sum(r["first_orientation"] for r in f),
                         first_wrist_ankle_height=sum(r["first_wrist_ankle_height"] for r in f),
                         first_only_wrist_ankle=sum(r["first_wrist_ankle_height"] and not r["first_orientation"] and not r["first_pelvis_height"] for r in f),
                         first_only_orientation=sum(r["first_orientation"] and not r["first_wrist_ankle_height"] and not r["first_pelvis_height"] for r in f),
                         any_pelvis_height=sum(r["any_pelvis_height"] for r in rs), any_orientation=sum(r["any_orientation"] for r in rs),
                         any_wrist_ankle_height=sum(r["any_wrist_ankle_height"] for r in rs),
                         first_fall_step_median=float(np.median([r["first_fall_step"] for r in f])) if f else None,
                         max_pelvis_dz_m_max=max(r["max_pelvis_dz_m"] for r in rs)))
    write_csv(out / "fall_reasons/summary.csv", summ)
    s = dict(n_episodes=len(sel), n_rule_mismatch=bad)
    print(json.dumps(s), flush=True)
    if bad:
        raise RuntimeError(f"fall flag != OR of criteria in {bad} episodes")
    return s


# ---------------------------------------------------------------- 9.3 torso pitch
def qmat(q):
    """xyzw -> rotation matrices [..., 3, 3]."""
    x, y, z, w = (q[..., i] for i in range(4))
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def rot(axis, a):
    c, s = np.cos(a), np.sin(a)
    o, l = np.zeros_like(a), np.ones_like(a)
    if axis == "z":
        m = [[c, -s, o], [s, c, o], [o, o, l]]
    elif axis == "x":
        m = [[l, o, o], [o, c, -s], [o, s, c]]
    else:
        m = [[c, o, s], [o, l, o], [-s, o, c]]
    return np.stack([np.stack(r, -1) for r in m], -2)


def torso_fk(pelvis_q_xyzw, waist):
    return qmat(pelvis_q_xyzw) @ rot("z", waist[..., 0]) @ rot("x", waist[..., 1]) @ rot("y", waist[..., 2])


def pitch(R):
    return np.arcsin(np.clip(-R[..., 2, 0], -1, 1))


class TorsoRefs:
    def __init__(self):
        self.c, self.meta, self.cos_tilt = {}, {}, {}

    def __call__(self, task):
        if task not in self.c:
            import torch
            from system1.sonic.task_refs import load_task_reference, sim_names
            ref, bodies, _, meta = load_task_reference(task)
            a = {k: v.numpy().astype(np.float64) for k, v in ref.sample(torch.arange(ref.num_frames)).items()}
            dofs = sim_names()[1]
            if dofs[12:15] != ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]:
                raise RuntimeError(f"waist dof order {dofs[12:15]}")
            qp, qt = a["agent_body_quat_w"][:, :, bodies.index("pelvis")], a["agent_body_quat_w"][:, :, bodies.index("torso_link")]
            Rt = qmat(qt)
            Rfk = torso_fk(qp, a["agent_joint_pos"][:, :, 12:15])
            cosang = np.clip((np.trace(np.swapaxes(Rfk, -1, -2) @ Rt, axis1=-2, axis2=-1) - 1) / 2, -1, 1)
            fk_err = float(np.arccos(cosang).max())
            self.meta[task] = dict(meta, fk_check_max_angle_rad=fk_err, fk_check_passed=fk_err < FK_TOL)
            if fk_err >= FK_TOL:
                raise RuntimeError(f"torso FK check failed for {task}: {fk_err} rad")
            self.c[task] = pitch(Rt)  # [F, 2]
            self.cos_tilt[task] = Rt[..., 2, 2]  # [F, 2] cos of torso tilt from vertical (termination robot term)
        return self.c[task]


TORSO_GRAV = 0.8  # JointBadTrackingZOnly bad_ref_ori_threshold on |g_z(ref torso) - g_z(robot torso)|


def termination_reasons(eps, refs):
    """Beyond 9.2 (diagnostic): which JointBadTrackingZOnly sub-term holds at the terminal step, with the reference at
    reward_reference_frame: object position > 0.25 m, object orientation > 0.8 rad, robot torso gravity-z error > 0.8."""
    rows = []
    for e in eps:
        refs(e["task"])
        ct = refs.cos_tilt[e["task"]]
        z = load(e["trajectory"], ["object_pos_w", "object_quat_xyzw", "reference_object_pos_w", "reference_object_quat_xyzw",
                                   "reference_frame", "reward_reference_frame", "root_quat_xyzw", "dof_pos"])
        row_of = {int(f): i for i, f in enumerate(z["reference_frame"])}
        rf = int(z["reward_reference_frame"][-1])
        i = row_of[rf]
        pe = float(np.linalg.norm(z["object_pos_w"][-1] - z["reference_object_pos_w"][i]))
        d = float(np.clip(abs(np.sum(z["object_quat_xyzw"][-1] * z["reference_object_quat_xyzw"][i])), 0, 1))
        oe = 2 * float(np.arccos(d))
        R = torso_fk(z["root_quat_xyzw"][-1], z["dof_pos"][-1][:, 12:15])
        ge = np.abs(R[:, 2, 2] - ct[rf])
        rows.append(dict(scheme=e["scheme"], variant=e["variant"], task=e["task"], perturbation_level_cm=e["level"], state_id=e["state_id"],
                         is_nominal=e["is_nominal"], executed_steps=e["steps"], obj_pos_err_m=pe, obj_ori_err_rad=oe,
                         torso_grav_err_r0=float(ge[0]), torso_grav_err_r1=float(ge[1]),
                         term_obj_pos=pe > OBJ_TERM, term_obj_ori=oe > 0.8, term_robot_torso=bool((ge > TORSO_GRAV).any()),
                         explained=pe > OBJ_TERM or oe > 0.8 or bool((ge > TORSO_GRAV).any())))
    return rows


def cmd_pitch(out, eps):
    refs = TorsoRefs()
    sel = [e for e in eps if e["scheme"] in ("A", "B") and e["experiment_group"] in ("System1-main", "System1-B-main")
           and e["mass_kg"] == ORIG_MASS[e["task"]]]
    rows, curves = [], []
    for e in sel:
        th_ref_all = refs(e["task"])
        z = load(e["trajectory"], ["root_quat_xyzw", "dof_pos", "reference_frame"])
        T = min(PITCH_STEPS, e["steps"])
        idx = np.arange(1, T + 1)
        th = pitch(torso_fk(z["root_quat_xyzw"][idx], z["dof_pos"][idx][:, :, 12:15]))       # [T, 2]
        thr = th_ref_all[z["reference_frame"][idx].astype(int)]                                 # [T, 2]
        err = th - thr
        for r in range(2):
            rows.append(dict(scheme=e["scheme"], variant=e["variant"], task=e["task"], perturbation_level_cm=e["level"], state_id=e["state_id"],
                             is_nominal=e["is_nominal"], robot=r, n_steps=T, abs_err_mean_rad=float(np.abs(err[:, r]).mean()),
                             abs_err_max_rad=float(np.abs(err[:, r]).max()), err_mean_rad=float(err[:, r].mean()),
                             ref_pitch_max_rad=float(thr[:, r].max()), robot_pitch_max_rad=float(th[:, r].max()),
                             ref_pitch_at_last_rad=float(thr[-1, r]), robot_pitch_at_last_rad=float(th[-1, r])))
            if e["is_nominal"]:
                for i, t in enumerate(idx):
                    curves.append(dict(scheme=e["scheme"], variant=e["variant"], task=e["task"], perturbation_level_cm=e["level"], robot=r,
                                       step=int(t), robot_pitch_rad=float(th[i, r]), ref_pitch_rad=float(thr[i, r])))
    write_csv(out / "torso_pitch/per_episode_robot.csv", rows)
    write_csv(out / "torso_pitch/nominal_curves.csv", curves)
    g = defaultdict(list)
    for r in rows:
        if not r["is_nominal"]:
            g[(r["scheme"], r["variant"], r["task"], r["perturbation_level_cm"])].append(r)
    summ = [dict(scheme=k[0], variant=k[1], task=k[2], perturbation_level_cm=k[3], n_episode_robots=len(rs),
                 n_steps_mean=float(np.mean([r["n_steps"] for r in rs])),
                 abs_err_mean_rad=float(np.mean([r["abs_err_mean_rad"] for r in rs])), abs_err_max_mean_rad=float(np.mean([r["abs_err_max_rad"] for r in rs])),
                 err_mean_rad=float(np.mean([r["err_mean_rad"] for r in rs])), ref_pitch_max_mean_rad=float(np.mean([r["ref_pitch_max_rad"] for r in rs])),
                 robot_pitch_max_mean_rad=float(np.mean([r["robot_pitch_max_rad"] for r in rs])),
                 ref_pitch_at_last_mean_rad=float(np.mean([r["ref_pitch_at_last_rad"] for r in rs])),
                 robot_pitch_at_last_mean_rad=float(np.mean([r["robot_pitch_at_last_rad"] for r in rs])))
            for k, rs in sorted(g.items(), key=lambda kv: tuple(str(x) for x in kv[0]))]
    write_csv(out / "torso_pitch/summary.csv", summ)
    (out / "torso_pitch/fk_check.json").write_text(json.dumps(refs.meta, indent=1) + "\n")
    return dict(fk=refs.meta)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["push", "objcheck", "fall", "pitch", "all"])
    p.add_argument("--out", type=Path, default=ROOT / "results_system1/T02")
    p.add_argument("--no-b-mass", action="store_true", help="skip B mass-sweep episodes (sweep still running)")
    a = p.parse_args()
    eps = all_episodes(include_b_mass=not a.no_b_mass)
    print(now(), "episodes", len(eps), {s: sum(e["scheme"] == s for e in eps) for s in "ABD"}, flush=True)
    res = dict(started=now(), cmd=a.cmd, argv=sys.argv, n_episodes=len(eps), git_head=__import__("subprocess").run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip())
    for c in (["push", "objcheck", "fall", "pitch"] if a.cmd == "all" else [a.cmd]):
        res[c] = dict(push=cmd_push, objcheck=cmd_objcheck, fall=cmd_fall, pitch=cmd_pitch)[c](a.out, eps)
    res["finished"] = now()
    (a.out / f"object_analysis_{a.cmd}{'_nobmass' if a.no_b_mass else ''}.run.json").write_text(json.dumps(res, indent=1, default=str) + "\n")


if __name__ == "__main__":
    main()
