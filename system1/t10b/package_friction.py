#!/usr/bin/env python3
"""T10b A return pack (results_system1/T10b_friction_margin/): episodes.csv, per_seed_summary.csv (completions and termination-reason
counts), friction_vs_completion.csv (the current hand friction 1.0 = the laptop-5070 finals, results_system1/T03 c2_main /
d_final_5070, next to 0.75 and 0.5), divergence.csv (failed episodes: first step where the object error exceeds 0.1 m, and the
reference phase at that step, system1/t10b/phases.py, as in T10s; the same state's episode at friction 1.0 next to it, and the first
step where the two runs' object positions are more than 0.1 m apart), divergence_summary.csv (per group), configs/ (launch / exit
records, hand_material.json, run_manifest.json, gzip; launch_vs_final.csv: argv and every recorded evaluation setting against the
friction-1.0 final, code versions), SHA256SUMS written separately. Read only. Raises if a run's hand-material read-back is not
'both robots, the two rubber-hand shapes only, at the requested value'.

Termination reason of a failed episode, from its trajectory (the evaluator records only that joint_bad_tracking fired): object
position error > 0.25 m or object orientation error > 0.8 rad at the last saved row, against the reference frame the termination
compared with (reward_reference_frame, one control step behind the row's own reference_frame; system1/t10b/phases.termination_errors),
otherwise 'robot orientation' (torso tilt is not in the trajectory). The divergence step (object error > 0.1 m) keeps the
evaluator's own alignment (row k against reference frame min(k, N-1), as object_rmse and T10s).

    sim2sim_s1/bin/python -B system1/t10b/package_friction.py --evals logs_system1/t10b/friction --out results_system1/T10b_friction_margin
"""
from __future__ import annotations

import argparse
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
from system1.t10b.hand_material import HAND_BODIES as HANDS  # noqa: E402
from system1.t10b.phases import NAME_ZH, divergence_step, label_frames, termination_errors, termination_reason  # noqa: E402


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def finals():
    out = {}
    for f in ("c2_main", "d_final_5070"):
        for r in csv.DictReader((ROOT / f"results_system1/T03/{f}/per_seed_summary.csv").open()):
            if r["perturbation_level_cm"] == "2" and r["variant"] in ("D", "C-in"):
                out[(r["variant"], r["task"], int(r["training_seed"]))] = r
    return out


def reason(traj):
    pe, oe, _ = termination_errors(np.load(traj))
    return termination_reason(pe, oe), pe, oe


SETTING_SKIP = {"episodes", "model_id", "system1_hand_material", "complete", "episode_count"}
RUN_ARGS = ("--model-id", "--output-dir", "--hand-friction")


def hand_only(rec, hf):
    """hand_material.json: both robots, only the two rubber-hand shapes changed, to (hf, hf, 0), every other shape identical."""
    ok = set(rec["per_robot"]) == {"robot", "robot_1"}
    for r in rec["per_robot"].values():
        ok &= r["changed_bodies"] == list(HANDS) and r["unchanged_shapes_identical"] is True
        ok &= all(r["after"][i] == [hf, hf, 0.0] for i in r["hand_shape_indices"])
    return bool(ok)


def strip_run_args(argv):
    """argv without the CPU slot (taskset -c X) and the per-run arguments (model id, output dir, hand friction)."""
    a = list(argv[3:] if argv[0] == "taskset" else argv)
    out, i = [], 0
    while i < len(a):
        if a[i] in RUN_ARGS:
            i += 2
            continue
        out.append(a[i])
        i += 1
    return out


def compare_with_final(d, L, s):
    """This run against the friction-1.0 final it was copied from: argv (other than slot / per-run args) and every recorded
    evaluation setting in summary.json (other than the episodes, model id, hand material); code versions of both."""
    src = ROOT / L["source_final_launch"]
    F = json.loads(src.read_text())
    B = json.loads((src.with_name(src.name[: -len(".launch.json")]) / "summary.json").read_text())
    setting_diff = sorted(k for k in (set(s) | set(B)) - SETTING_SKIP if s.get(k) != B.get(k))
    return dict(name=d.name, source_final_launch=L["source_final_launch"], argv_identical_except_slot_and_run_args=strip_run_args(L["argv"]) == strip_run_args(F["argv"]),
                summary_settings_identical=not setting_diff, summary_setting_differences=json.dumps(setting_diff), cpu_slot_final=F["argv"][2] if F["argv"][0] == "taskset" else "",
                cpu_slot_friction=L["argv"][2], gpu_final=B.get("gpu_name"), gpu_friction=s.get("gpu_name"), git_head_final=F.get("git_head"),
                git_dirty_final=bool(F.get("git_diff")), git_head_friction=L["git_head"], git_dirty_friction=bool(L["git_diff"]),
                git_status_friction=L["git_status"].strip().replace("\n", "; "))


def verify_record(v):
    """From the scene export (--sim2sim-dump) of the check run: the USD physics materials with their combine modes, the USD-bound
    material of every hand / object shape, the PhysX per-shape material table of both robots and the object (which shapes are not
    (1, 1, 0)), and the evaluator's own read-back (hand_material.json)."""
    sc = json.loads((v / "dump" / "scene.json").read_text())
    hm = json.loads((v / "eval" / "hand_material.json").read_text())
    man = json.loads((v / "eval" / "run_manifest.json").read_text())
    shapes = [dict(path=x["path"], body=x["body"], usd_bound_physics_material=x["material"]) for x in sc["shapes"]
              if any(h in str(x["body"]) for h in HANDS) or "/Object/" in x["path"] or x["body"] is None]
    per_shape = {}
    for r in ("robot", "robot_1"):
        m = sc["bodies"][r]["material_per_shape"]
        per_shape[r] = dict(n_shapes=len(m), shapes_not_1_1_0={str(i): x for i, x in enumerate(m) if x != [1.0, 1.0, 0.0]},
                            hand_shape_indices=hm["per_robot"][r]["hand_shape_indices"])
    per_shape["object"] = sc["object"]["material_per_shape"]
    return dict(source=str(v.relative_to(ROOT)), case=man["case"], checkpoint=man["checkpoint"], checkpoint_sha256=man["checkpoint_sha256"], model_id=man["model_id"],
                gpu=man["gpu_name"], hand_friction_requested=man["system1_hand_material"], usd_physics_materials=sc["physics_materials"],
                scene_default_material=sc["scene_default_material"], usd_material_of_hand_object_ground_shapes=shapes, physx_material_per_shape=per_shape,
                evaluator_read_back={r: dict(changed_bodies=x["changed_bodies"], changed_shape_indices=x["changed_shape_indices"],
                                             before=[x["before"][i] for i in x["hand_shape_indices"]], after=[x["after"][i] for i in x["hand_shape_indices"]],
                                             unchanged_shapes_identical=x["unchanged_shapes_identical"]) for r, x in hm["per_robot"].items()},
                scene_sha256=sha(v / "dump" / "scene.json"), hand_material_sha256=sha(v / "eval" / "hand_material.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--verify", type=Path, default=None, help="the --sim2sim-dump check run (D Table 721, nominal, hand friction 0.5): writes hand_material_check.json")
    a = ap.parse_args()
    out = a.out.resolve()
    (out / "configs").mkdir(parents=True, exist_ok=True)
    fin = finals()
    eps, per, missing, div, cmp_rows = [], [], [], [], []
    phase_cache = {}
    groups = {}
    for d in sorted(p for p in a.evals.resolve().iterdir() if p.is_dir()):
        L = json.loads(d.with_name(d.name + ".launch.json").read_text())
        ex = d.with_name(d.name + ".exit.json")
        s = json.loads((d / "summary.json").read_text()) if (d / "summary.json").exists() else None
        if not s or not s.get("complete") or not ex.exists() or json.loads(ex.read_text())["returncode"] != 0:
            missing.append(d.name)
            continue
        task, scheme, seed, hf = L["task"], L["scheme"], L["seed"], L["hand_friction"]
        f0 = fin[(scheme, task, seed)]
        if s["checkpoint_sha256"] != f0["checkpoint_sha256"] or s["states_sha256"] != f0["states_sha256"]:
            raise ValueError(f"{d.name}: checkpoint / states differ from the final evaluation")
        hm = s.get("system1_hand_material") or {}
        if abs(hm.get("static_friction", -1) - hf) > 1e-9:
            raise ValueError(f"{d.name}: hand material not recorded as {hf}")
        hand_ok = hand_only(json.loads((d / "hand_material.json").read_text()), hf)
        if not hand_ok:
            raise ValueError(f"{d.name}: hand material read-back is not 'hands only, value {hf}'")
        cmp_rows.append(compare_with_final(d, L, s))
        rows = []
        for e in s["episodes"]:
            r = dict(task=task, scheme=scheme, training_seed=seed, hand_friction=hf, hand_object_friction=(hf + 0.5) / 2, state_id=e["state_id"],
                     is_nominal=e["kind"] == "nominal", completed_reference=e["completed_reference"], executed_steps=e["steps"], horizon_steps=e["horizon_steps"],
                     terminal_terms=json.dumps(e["terminal_terms"], sort_keys=True), object_pos_rmse_cm=e["object_rmse_3d_m"] * 100,
                     checkpoint_sha256=s["checkpoint_sha256"], states_sha256=s["states_sha256"], summary_sha256=sha(d / "summary.json"), eval_dir=str(d.relative_to(ROOT)))
            if not e["completed_reference"]:
                rr, pe, oe = reason(e["trajectory"])
                r.update(termination_reason=rr, last_object_pos_err_m=pe, last_object_ori_err_rad=oe)
            rows.append(r)
        eps += rows
        pr = [r for r in rows if not r["is_nominal"]]
        fails = [r for r in pr if not r["completed_reference"]]
        per.append(dict(task=task, scheme=scheme, training_seed=seed, hand_friction=hf, hand_object_friction=(hf + 0.5) / 2, n_episodes=len(pr),
                        n_completed=len(pr) - len(fails), completion_pct=100 * (len(pr) - len(fails)) / len(pr),
                        nominal_completed=[r for r in rows if r["is_nominal"]][0]["completed_reference"],
                        fail_object_position=sum(r["termination_reason"] == "object_position" for r in fails),
                        fail_object_orientation=sum(r["termination_reason"] == "object_orientation" for r in fails),
                        fail_robot_orientation=sum(r["termination_reason"] == "robot_orientation" for r in fails),
                        steps_mean_failed=statistics.mean(r["executed_steps"] for r in fails) if fails else None, hand_shapes_only_changed=hand_ok,
                        eval_dir=str(d.relative_to(ROOT))))
        groups[(scheme, task, seed, hf)] = (d, s)
        for f in (d.with_name(d.name + ".launch.json"), ex, d / "hand_material.json", d / "run_manifest.json"):
            (out / "configs" / f"{d.name}__{f.name.replace(d.name + '.', '')}.gz").write_bytes(gzip.compress(f.read_bytes(), mtime=0))
    save_csv(out / "episodes.csv", eps)
    save_csv(out / "per_seed_summary.csv", per)
    # friction vs completion (1.0 = the 5070 finals)
    fv = []
    for (scheme, task, seed), f0 in sorted(fin.items(), key=lambda x: (x[0][1], x[0][0], x[0][2])):
        r = dict(task=task, scheme=scheme, training_seed=seed, completed_hand_1p0=int(f0["n_completed"]), finals_eval_dir=f0["eval_dir"])
        have = False
        for hf in (0.75, 0.5):
            x = next((p for p in per if p["scheme"] == scheme and p["task"] == task and p["training_seed"] == seed and p["hand_friction"] == hf), None)
            r[f"completed_hand_{str(hf).replace('.', 'p')}"] = x["n_completed"] if x else None
            have = have or x is not None
        if have:
            fv.append(r)
    save_csv(out / "friction_vs_completion.csv", fv)
    # divergence of failed episodes, with the same state at friction 1.0
    for (scheme, task, seed, hf), (d, s) in groups.items():
        f0 = fin[(scheme, task, seed)]
        base = {e["state_id"]: e for e in json.loads((ROOT / f0["eval_dir"] / "summary.json").read_text())["episodes"]}
        for e in s["episodes"]:
            if e["kind"] == "nominal" or e["completed_reference"]:
                continue
            z = np.load(e["trajectory"])
            k = divergence_step(z["object_pos_w"], z["reference_object_pos_w"])
            key = (task, f0["eval_dir"])
            if key not in phase_cache:
                zz = np.load(ROOT / f0["eval_dir"] / "nominal.trajectory.npz")
                phase_cache[key] = label_frames(zz["reference_object_pos_w"], zz["reference_object_quat_xyzw"])
            ph = phase_cache[key]
            b = base[e["state_id"]]
            zb = np.load(b["trajectory"])
            kb = divergence_step(zb["object_pos_w"], zb["reference_object_pos_w"])
            n = min(len(z["object_pos_w"]), len(zb["object_pos_w"]))  # same state, friction 1.0: first row where the two object positions differ > 0.1 m
            kx = divergence_step(z["object_pos_w"][:n], zb["object_pos_w"][:n])
            div.append(dict(task=task, scheme=scheme, training_seed=seed, hand_friction=hf, state_id=e["state_id"], executed_steps=e["steps"],
                            divergence_step_0p1m=k, phase_at_divergence=(ph[min(k, len(ph) - 1)] if k is not None else None),
                            phase_zh=(NAME_ZH[ph[min(k, len(ph) - 1)]] if k is not None else None), same_state_hand_1p0_completed=b["completed_reference"],
                            same_state_hand_1p0_divergence_step_0p1m=kb, same_state_hand_1p0_executed_steps=b["steps"],
                            step_object_apart_from_hand_1p0_run_0p1m=kx, phase_at_apart=(ph[min(kx, len(ph) - 1)] if kx is not None else None)))
    save_csv(out / "divergence.csv", div)
    dsum = []
    for key in sorted({(r["task"], r["scheme"], r["training_seed"], r["hand_friction"]) for r in div}, key=lambda k: (k[0], k[1], k[2], -k[3])):
        g = [r for r in div if (r["task"], r["scheme"], r["training_seed"], r["hand_friction"]) == key]
        k = [r["divergence_step_0p1m"] for r in g if r["divergence_step_0p1m"] is not None]
        kb = [r["same_state_hand_1p0_divergence_step_0p1m"] for r in g if r["same_state_hand_1p0_divergence_step_0p1m"] is not None]
        ph = {}
        for r in g:
            ph[r["phase_at_divergence"] or "never_above_0p1m"] = ph.get(r["phase_at_divergence"] or "never_above_0p1m", 0) + 1
        kx = [r["step_object_apart_from_hand_1p0_run_0p1m"] for r in g if r["step_object_apart_from_hand_1p0_run_0p1m"] is not None]
        pa = {}
        for r in g:
            pa[r["phase_at_apart"] or "never_apart_0p1m"] = pa.get(r["phase_at_apart"] or "never_apart_0p1m", 0) + 1
        dsum.append(dict(task=key[0], scheme=key[1], training_seed=key[2], hand_friction=key[3], n_failed=len(g), n_object_error_above_0p1m=len(k),
                         divergence_step_median=statistics.median(k) if k else None, divergence_step_min=min(k) if k else None, divergence_step_max=max(k) if k else None,
                         phase_at_divergence_counts=json.dumps(dict(sorted(ph.items(), key=lambda x: -x[1]))), executed_steps_median=statistics.median(r["executed_steps"] for r in g),
                         same_state_hand_1p0_completed=sum(bool(r["same_state_hand_1p0_completed"]) for r in g),
                         same_state_hand_1p0_divergence_step_median=statistics.median(kb) if kb else None, same_state_hand_1p0_n_above_0p1m=len(kb),
                         n_apart_from_hand_1p0_run_0p1m=len(kx), apart_step_median=statistics.median(kx) if kx else None, apart_step_min=min(kx) if kx else None,
                         apart_step_max=max(kx) if kx else None, phase_at_apart_counts=json.dumps(dict(sorted(pa.items(), key=lambda x: -x[1])))))
    save_csv(out / "divergence_summary.csv", dsum)
    save_csv(out / "configs" / "launch_vs_final.csv", cmp_rows)
    if a.verify is not None:
        (out / "hand_material_check.json").write_text(json.dumps(verify_record(a.verify.resolve()), indent=1) + "\n")
    (out / "provenance.json").write_text(json.dumps(dict(missing=missing, n_evaluations=len(per), n_episodes=len(eps),
                                                         finals_sources=["results_system1/T03/c2_main/per_seed_summary.csv",
                                                                         "results_system1/T03/d_final_5070/per_seed_summary.csv"]), indent=1) + "\n")
    print("evaluations", len(per), "episodes", len(eps), "missing", missing)


if __name__ == "__main__":
    main()
