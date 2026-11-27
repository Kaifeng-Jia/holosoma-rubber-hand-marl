#!/usr/bin/env python3
"""T13 v4 section 5: return package of one finished run (read only on the logs, CPU). Writes results_system1/T13_runs/<run>/:

  episodes.csv   every episode of every evaluation group (Isaac noise off / on, MuJoCo; +/-2, +/-8; nominal + 50 perturbed), in the
                 column set of the T10b package (results_system1/T10b_friction_margin/episodes.csv) plus group / simulator / noise;
  summary.csv    per group: completions of the 50 perturbed states (nominal reported separately, not counted), object-position RMSE
                 of the completed episodes, failure classes;
  comparison.csv completions next to the paper D of the same task (T03 d_final_5070, seeds 721-723; original reference,
                 original observation, no randomization; its test states differ: new reference, states regenerated); Table (no T03
                 row): paper D Table Isaac / MuJoCo from T10 (isaac_vs_mujoco.csv) and the T10b pilot (D Table + randomization);
                 round 2 (c_out_*): also T03 C-out of the same task and the round-1 D run of the same configuration;
  training_diag.csv / .json  internal diagnostics from the copied-back training metrics (train/metrics.jsonl): per robot, the
                 hand-force-in-should share, lost-contact termination signals (term 2) and penalty step share (term 3); last 50
                 iterations and whole run;
  records/       launch / exit records, run manifests, checks (sim2sim), copy-back sha256 verification.

Failure classes (as T10 / T10b): the evaluator's single termination term joint_bad_tracking covers object position error > 0.25 m
("object_position": the object slid / left the reference), object orientation error > 0.8 rad ("object_orientation") and torso
gravity-z error > 0.8 ("robotR_tilt": a robot fell). MuJoCo: from the saved last termination record (several can hold);
Isaac: object errors at the last saved row against the reference frame the term compared with; torso tilt is not saved, so it is
the remaining case ("robot_orientation"). The lost-contact termination (term 2) is a training term and is not part of the
evaluation protocol: no evaluation episode can end by it; it is reported from the training metrics only.

    sonic_s1/bin/python -B system1/t13v4/package_run.py --run d_push
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from system1.sim2sim.failure_t10s import causes as mj_causes  # noqa: E402
from system1.t10b.phases import termination_errors, termination_reason  # noqa: E402

CASE = {"d_push": "push", "d_table": "table", "d_table_pa": "table", "c_out_table": "table", "c_out_push": "push"}  # round 2: c_out_*
CASE.update(d_frontback="table", d_frontback_smoke="table")  # T14 front-back training task (Kevin 2026-11-26); _smoke: pre-training T10 gate


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rel(p):
    p = Path(p).resolve()
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


def groups(run):
    g = []
    for noise in ("off", "on"):
        for lv in (2, 8):
            name = f"{run}_pm{lv}" + ("_noise_on" if noise == "on" else "")
            g.append(dict(group=f"isaac_noise_{noise}_pm{lv}", simulator="isaac", noise=noise, level=lv, dir=ROOT / "logs_system1/t13v4/finals" / name,
                          launch=ROOT / "logs_system1/t13v4/finals" / f"{name}.launch.json", exit=ROOT / "logs_system1/t13v4/finals" / f"{name}.exit.json"))
    for lv in (2, 8):
        g.append(dict(group=f"mujoco_noise_off_pm{lv}", simulator="mujoco", noise="off", level=lv, dir=ROOT / "logs_system1/t13v4/mj_finals" / f"{run}_pm{lv}",
                      launch=None, exit=None))
    return g


def classify(sim, e):
    if e["completed_reference"]:
        return [], None, None
    z = np.load(e["trajectory"])
    if sim == "mujoco":
        lt = e["last_termination"]
        return mj_causes(lt), lt.get("object_pos_err"), lt.get("object_ori_err")
    pe, oe, _ = termination_errors(z)
    return [termination_reason(pe, oe)], pe, oe


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, choices=sorted(CASE))
    a = ap.parse_args()
    task = CASE[a.run]
    out = ROOT / "results_system1/T13_runs" / a.run
    (out / "records").mkdir(parents=True, exist_ok=True)
    ep_rows, summ = [], []
    for g in groups(a.run):
        sp = g["dir"] / "summary.json"
        if not sp.exists():
            summ.append(dict(group=g["group"], simulator=g["simulator"], noise=g["noise"], level_cm=g["level"], status="未完成（没有 summary.json）"))
            continue
        s = json.loads(sp.read_text())
        eps = s["episodes"]
        pert = [e for e in eps if e["kind"] != "nominal"]
        nom = [e for e in eps if e["kind"] == "nominal"]
        cls = collections.Counter()
        for e in eps:
            c, pe, oe = classify(g["simulator"], e)
            if e["kind"] != "nominal":
                for x in c:
                    cls[x] += 1
            ep_rows.append(dict(task=task, run=a.run, group=g["group"], simulator=g["simulator"], observation_noise=g["noise"], level_cm=g["level"],
                                training_seed=721, state_id=e["state_id"], is_nominal=e["kind"] == "nominal", completed_reference=e["completed_reference"],
                                executed_steps=e["steps"], horizon_steps=e["horizon_steps"], terminal_terms=json.dumps(e["terminal_terms"]),
                                object_pos_rmse_cm=100 * e["object_rmse_3d_m"] if e.get("object_rmse_3d_m") is not None else None,
                                checkpoint_sha256=s["checkpoint_sha256"], states_sha256=s["states_sha256"], summary_sha256=sha(sp), eval_dir=rel(g["dir"]),
                                termination_reason="、".join(c) if c else "", last_object_pos_err_m=pe, last_object_ori_err_rad=oe))
        done = [e for e in pert if e["completed_reference"]]
        rm = [100 * e["object_rmse_3d_m"] for e in done if e.get("object_rmse_3d_m") is not None]
        summ.append(dict(group=g["group"], simulator=g["simulator"], noise=g["noise"], level_cm=g["level"], status="complete" if s.get("complete", True) else "incomplete",
                         n_perturbed=len(pert), n_completed=len(done), completion_pct=round(100 * len(done) / len(pert), 1) if pert else None,
                         nominal_completed=nom[0]["completed_reference"] if nom else None,
                         object_pos_rmse_cm_mean_completed=round(float(np.mean(rm)), 3) if rm else None,
                         progress_mean_perturbed=round(float(np.mean([e["progress"] for e in pert])), 4) if pert else None,
                         failures=json.dumps(dict(cls), ensure_ascii=False), checkpoint_sha256=s["checkpoint_sha256"], states_sha256=s["states_sha256"],
                         summary_sha256=sha(sp), eval_dir=rel(g["dir"])))
        m = g["dir"] / "run_manifest.json"
        if m.exists():
            shutil.copy2(m, out / "records" / f"{g['group']}_run_manifest.json")
        for k in ("launch", "exit"):
            if g[k] is not None and g[k].exists():
                shutil.copy2(g[k], out / "records" / f"{g['group']}.{k}.json")
    for name, rows in (("episodes.csv", ep_rows), ("summary.csv", summ)):
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    # comparison with the paper D of the same task (Isaac, laptop 5070, T03 d_final_5070)
    comp = []
    for r in csv.DictReader(open(ROOT / "results_system1/T03/d_final_5070/per_seed_summary.csv")):
        if r["task"] == task and r["variant"] == "D":
            comp.append(dict(source="paper D (T03 d_final_5070; original reference / observation, no randomization)", training_seed=r["training_seed"],
                             level_cm=r["perturbation_level_cm"], n_completed=r["n_completed"], n_episodes=r["n_episodes"], checkpoint_iteration=r["checkpoint_iteration"],
                             states_sha256=r["states_sha256"], note="different test states (new reference, states regenerated)"))
    if task == "table" and not a.run.startswith("d_frontback"):  # task book section 5 item 6: Table has no T03 row; paper D Table from T10, and the T10b randomized pilot (front-back: other task, no paper row)
        for r in csv.DictReader(open(ROOT / "results_system1/T10_sim2sim/isaac_vs_mujoco.csv")):
            if r["task"] == "table" and r["scheme"] == "D":
                for sim, k in (("Isaac", "isaac_5070_n_completed"), ("MuJoCo", "mujoco_n_completed")):
                    comp.append(dict(source=f"paper D Table ({sim}; T10 isaac_vs_mujoco.csv; original reference / observation, no randomization)",
                                     training_seed=r["training_seed"], level_cm=r["perturbation_level_cm"], n_completed=r[k], n_episodes=r["n_episodes"],
                                     checkpoint_iteration=None, states_sha256=r["isaac_states_sha256"], note="different test states (new reference, states regenerated)"))
        for r in csv.DictReader(open(ROOT / "results_system1/T10b_friction_margin/retrain/finals/d_table_seed721_rand/comparison.csv")):
            if r["policy"] == "pilot":
                comp.append(dict(source=f"T10b pilot D Table + randomization ({'Isaac' if r['simulator'] == 'isaac' else 'MuJoCo'}; T10b retrain/finals)",
                                 training_seed=721, level_cm=r["level_cm"], n_completed=r["n_completed"], n_episodes=r["n_episodes"], checkpoint_iteration=14000,
                                 states_sha256=r["states_sha256"], note="original reference / observation; different test states"))
    if a.run.startswith("c_out_"):  # round 2: T03 C-out of the same task (original reference / observation, no randomization), and round-1 D
        for r in csv.DictReader(open(ROOT / "results_system1/T03/c2_main/per_seed_summary.csv")):
            if r["task"] == task and r["variant"] == "C-out":
                comp.append(dict(source="T03 C-out (c2_main, Isaac; original reference / observation, no randomization)", training_seed=r["training_seed"],
                                 level_cm=r["perturbation_level_cm"], n_completed=r["n_completed"], n_episodes=r["n_episodes"],
                                 checkpoint_iteration=r["checkpoint_iteration"], states_sha256=r["states_sha256"], note="different test states"))
        d1 = {"table": "d_table", "push": "d_push"}[task]
        f1 = ROOT / "results_system1/T13_runs" / d1 / "summary.csv"
        if f1.exists():
            for r in csv.DictReader(open(f1)):
                comp.append(dict(source=f"T13 v4 round 1 {d1} ({r['simulator']}, noise {r['noise']}; same configuration, structure D)", training_seed=721,
                                 level_cm=r["level_cm"], n_completed=r.get("n_completed"), n_episodes=r.get("n_perturbed"), checkpoint_iteration=None,
                                 states_sha256=r.get("states_sha256"), note="same test states"))
    for s in summ:
        if "n_completed" in s:
            comp.append(dict(source=f"T13 v4 {a.run} ({'Isaac' if s['simulator'] == 'isaac' else 'MuJoCo'}, noise {s['noise']})", training_seed=721,
                             level_cm=s["level_cm"], n_completed=s["n_completed"], n_episodes=s["n_perturbed"], checkpoint_iteration=None,
                             states_sha256=s["states_sha256"], note=""))
    with open(out / "comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(comp[0]))
        w.writeheader()
        w.writerows(comp)
    # training-time diagnostics
    dec = json.loads((ROOT / "logs_system1/t13v4/runs" / a.run / "decision.json").read_text())
    last_it = int(dec["final_checkpoint_iteration"])  # file iteration of the checkpoint evaluated; later iterations are not used
    met_all = [json.loads(x) for x in open(ROOT / "logs_system1/t13v4/runs" / a.run / "train/metrics.jsonl") if x.strip()]
    met = [m for m in met_all if int(m["iteration"]) <= last_it]
    keys = sorted({k for m in met for k in m if k.startswith("ContactV1/")})
    diag = dict(rule="train/metrics.jsonl rows with iteration <= the evaluated checkpoint's file iteration; last50 = the last 50 of those",
                iterations_used=len(met), first_iteration=int(met[0]["iteration"]), last_iteration=int(met[-1]["iteration"]),
                iterations_in_file=len(met_all), last_iteration_in_file=int(met_all[-1]["iteration"]), keys=keys, last50={}, whole_run={})
    for k in keys:
        v = [m[k] for m in met if isinstance(m.get(k), (int, float))]
        if v:
            diag["last50"][k] = float(np.mean(v[-50:]))
            diag["whole_run"][k] = float(np.mean(v))
    (out / "training_diag.json").write_text(json.dumps(diag, indent=1) + "\n")
    with open(out / "training_diag.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "mean_last50_iterations", "mean_whole_run"])
        for k in keys:
            if k in diag["last50"]:
                w.writerow([k, diag["last50"][k], diag["whole_run"][k]])
    for p in (ROOT / "logs_system1/t13v4/runs" / a.run / "decision.json", ROOT / "logs_system1/t13v4/runs" / a.run / "validation_curve.json"):
        shutil.copy2(p, out / "records" / p.name)
    for p in sorted((ROOT / "logs_system1/t13v4/copyback" / a.run).glob("*")):
        if p.name in ("copyback.txt", "sha256_local_sorted.txt"):
            shutil.copy2(p, out / "records" / f"copyback_{p.name}")
    for p in sorted((ROOT / "logs_system1/t13v4/sim2sim" / f"checks_{a.run}").glob("*")):
        shutil.copy2(p, out / "records" / f"sim2sim_check_{p.name}")
    oc = ROOT / "logs_system1/t13v4/sim2sim" / f"onnx_{a.run}" / "onnx_check.csv"
    if oc.exists():
        shutil.copy2(oc, out / "records" / "onnx_check.csv")
    for s in summ:
        print(s.get("group"), s.get("n_completed"), "/", s.get("n_perturbed"), s.get("failures"), s.get("status"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
