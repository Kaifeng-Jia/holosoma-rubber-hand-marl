#!/usr/bin/env python3
"""T13 C1 return pack (results_system1/T10c_cout_sim2sim/): C-out (seed 721) MuJoCo closed loop next to the C-out Isaac
laptop-5070 finals (T03 c2_main), per episode and per task x level. Facts only. CPU, read only.

Per episode, both simulators:
  completed (all reference steps without joint_bad_tracking), executed steps, termination terms;
  termination causes: MuJoCo = the evaluator's last termination record (robot r torso gravity-z error > 0.8, object position error
  > 0.25 m, object orientation error > 0.8 rad; several can hold); Isaac = recomputed from the saved trajectory (object state of the
  last row against the reference frame the termination used, i.e. the previous row's reference): object_position / object_orientation;
  a terminated Isaac episode with neither object criterion is labelled robot_tilt_inferred (torso tilt is not in the trajectory);
  failure class = (first cause, reference phase at termination); object divergence step = first step with object position error >
  0.1 m (system1/t10b/phases.divergence_step) and the reference phase there (phases.label_frames on the reference object).

    sim2sim_s1/bin/python -B system1/sim2sim/package_t10c.py --evals logs_system1/t10c/mj_eval --out results_system1/T10c_cout_sim2sim
"""
from __future__ import annotations

import argparse
import collections
import csv
import gzip
import json
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from system1.sim2sim.failure_t10s import causes  # noqa: E402
from system1.sim2sim.package_t10 import save_csv, sha  # noqa: E402
from system1.t10b.phases import NAME_ZH, divergence_step, label_frames, quat_angle_xyzw  # noqa: E402

ISAAC_RUNS = ("cout_table:table", "cout_bucket:bucket", "cout_pull:pull")


def isaac_rows():
    out = {}
    for r in csv.DictReader((ROOT / "results_system1/T03/c2_main/per_seed_summary.csv").open()):
        if r["variant"] == "C-out" and int(r["training_seed"]) == 721:
            out[(r["task"], int(r["perturbation_level_cm"]))] = r
    return out


def episode_facts(e, traj_path, labels_cache, key, mujoco):
    z = np.load(traj_path)
    if key not in labels_cache:
        labels_cache[key] = label_frames(z["reference_object_pos_w"], z["reference_object_quat_xyzw"])
    labels = labels_cache[key]
    if e["completed_reference"]:
        c = []
    elif mujoco:
        c = causes(e.get("last_termination") or {})
    else:
        ep = float(np.linalg.norm(z["object_pos_w"][-1] - z["reference_object_pos_w"][-2]))
        eo = float(quat_angle_xyzw(z["object_quat_xyzw"][-1], z["reference_object_quat_xyzw"][-2]))
        c = (["object_position"] if ep > 0.25 else []) + (["object_orientation"] if eo > 0.8 else [])
        c = c or ["robot_tilt_inferred"]
    k = divergence_step(z["object_pos_w"], z["reference_object_pos_w"], 0.1)
    pt = labels[min(e["steps"] - 1, len(labels) - 1)]
    return dict(termination_causes="+".join(c), failure_class=(f"{c[0]}@{pt}" if c else ""), object_divergence_step_0p1m=k,
                phase_at_divergence=(labels[min(k, len(labels) - 1)] if k is not None else None), phase_at_termination=pt if c else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    out, evals = a.out.resolve(), a.evals.resolve()
    (out / "configs").mkdir(parents=True, exist_ok=True)
    isa = isaac_rows()
    eps, summ, prov, missing, labels = [], [], dict(runs=[]), [], {}
    for d in sorted(p for p in evals.iterdir() if p.is_dir()):
        s = json.loads((d / "summary.json").read_text()) if (d / "summary.json").exists() else None
        ex = d.with_name(d.name + ".exit.json")
        if not s or not s.get("complete") or len(s["episodes"]) != 51 or not ex.exists() or json.loads(ex.read_text())["returncode"] != 0:
            missing.append(d.name)
            continue
        task, lvl = s["task"], int(s["level"])
        ir = isa[(task, lvl)]
        idir = ROOT / ir["eval_dir"]
        isum = json.loads((idir / "summary.json").read_text())
        assert isum["checkpoint_sha256"] == s["checkpoint_sha256"] == ir["checkpoint_sha256"], "checkpoint differs from the Isaac final"
        assert isum["states_sha256"] == s["states_sha256"] == ir["states_sha256"], "state file differs from the Isaac final"
        per_sim = {}
        for sim, episodes in (("mujoco", s["episodes"]), ("isaac_5070", isum["episodes"])):
            rows = []
            for e in episodes:
                tp = Path(e["trajectory"]) if sim == "mujoco" else idir / Path(e["trajectory"]).name
                f = episode_facts(e, tp, labels, task, sim == "mujoco")
                row = dict(simulator=sim, task=task, scheme="C-out", training_seed=721, perturbation_level_cm=lvl, state_id=e["state_id"],
                           is_nominal=e["kind"] == "nominal", completed_reference=e["completed_reference"], executed_steps=e["steps"],
                           reference_horizon_steps=e["horizon_steps"], progress=e["progress"], termination_terms=json.dumps(e["terminal_terms"], sort_keys=True),
                           object_pos_rmse_cm=e["object_rmse_3d_m"] * 100, **f)
                if sim == "mujoco":
                    lt = e["last_termination"]
                    row.update(last_object_pos_err_m=lt["object_pos_err"], last_object_ori_err_rad=lt["object_ori_err"],
                               last_torso_gravity_z_err_max=max(lt["gravity_z_err"]))
                rows.append(row)
            eps += rows
            per = [r for r in rows if not r["is_nominal"]]
            fail = [r for r in per if not r["completed_reference"]]
            nom = [r for r in rows if r["is_nominal"]][0]
            cc = collections.Counter(x for r in fail for x in r["termination_causes"].split("+") if x)
            fc = collections.Counter(r["failure_class"] for r in fail)
            div = [r["object_divergence_step_0p1m"] for r in fail if r["object_divergence_step_0p1m"] is not None]
            ts = [r["executed_steps"] for r in fail]
            per_sim[sim] = dict(n_completed=sum(r["completed_reference"] for r in per), nominal_completed=nom["completed_reference"], nominal_steps=nom["executed_steps"],
                                steps_mean_all50=statistics.mean(r["executed_steps"] for r in per),
                                termination_step_median=statistics.median(ts) if ts else None, termination_step_min=min(ts) if ts else None,
                                termination_step_max=max(ts) if ts else None, n_terminated_before_step_50=sum(t < 50 for t in ts),
                                causes=json.dumps(dict(cc), sort_keys=True), failure_classes=json.dumps(dict(fc.most_common()), ensure_ascii=False),
                                object_divergence_step_median=statistics.median(div) if div else None)
        assert per_sim["isaac_5070"]["n_completed"] == int(ir["n_completed"]), "Isaac recount differs from T03 c2_main"
        row = dict(task=task, scheme="C-out", training_seed=721, perturbation_level_cm=lvl, n_episodes=50, reference_horizon_steps=s["episodes"][0]["horizon_steps"])
        for sim in ("isaac_5070", "mujoco"):
            row.update({f"{sim}_{k}": v for k, v in per_sim[sim].items()})
        row.update(difference_mujoco_minus_isaac=per_sim["mujoco"]["n_completed"] - per_sim["isaac_5070"]["n_completed"],
                   checkpoint_sha256=s["checkpoint_sha256"], states_sha256=s["states_sha256"], onnx_sha256=s["onnx_sha256"],
                   isaac_eval_dir=ir["eval_dir"], mujoco_eval_dir=str(d.relative_to(ROOT)), mujoco_wall_s=round(s["wall_s"], 1))
        summ.append(row)
        for f in (d / "run_manifest.json", d.with_name(d.name + ".launch.json"), ex):
            (out / "configs" / f"mujoco__{d.name}__{f.name.replace(d.name + '.', '')}.gz").write_bytes(gzip.compress(f.read_bytes(), mtime=0))
        prov["runs"].append(dict(name=d.name, summary_sha256=sha(d / "summary.json"), onnx=s["onnx"], onnx_sha256=s["onnx_sha256"], checkpoint_sha256=s["checkpoint_sha256"],
                                 states=s["states"], states_sha256=s["states_sha256"], scene_json_sha256=s["scene_json_sha256"], convexes_sha256=s["convexes_sha256"],
                                 isaac_final_summary_sha256=sha(idir / "summary.json")))
    summ.sort(key=lambda r: (r["task"], r["perturbation_level_cm"]))
    save_csv(out / "isaac_vs_mujoco.csv", summ)
    save_csv(out / "episodes.csv", eps)
    (out / "phase_labels_zh.json").write_text(json.dumps(NAME_ZH, ensure_ascii=False, indent=1) + "\n")
    from system1.sim2sim.mj_env import SceneSpec, build_model
    for spec in ISAAC_RUNS:
        run, task = spec.split(":")
        base = ROOT / "logs_system1/t10/isaac" / run
        for f in (base / "launch.json", base / "exit.json", base / "dump/scene.json"):
            (out / "configs" / f"isaac__{run}__{f.name}.gz").write_bytes(gzip.compress(f.read_bytes(), mtime=0))
        prov[f"isaac_{run}"] = dict(scene_json_sha256=sha(base / "dump/scene.json"), convexes_sha256=sha(base / "dump/collision_convexes.npz"),
                                    record_sha256=sha(base / "dump/nominal.sim2sim.npz"), eval_summary_sha256=sha(base / "eval/summary.json"),
                                    runtime_reference_sha256=sha(base / "dump/runtime_reference.npz") if (base / "dump/runtime_reference.npz").exists() else None)
        _, mj, _ = build_model(SceneSpec(base / "dump"), task)
        p = out / "configs" / f"mjcf_{run}.xml"
        p.write_text(mj.to_xml())
        prov[f"mjcf_{run}_sha256"] = sha(p)
    urdf = ROOT / "src/holosoma/holosoma/data/robots/g1/main_mesh_collision_rubberhand.urdf"
    prov.update(robot_urdf=dict(path=str(urdf.relative_to(ROOT)), sha256=sha(urdf)), missing=missing, n_runs=len(summ), n_episode_rows=len(eps))
    (out / "provenance.json").write_text(json.dumps(prov, indent=1) + "\n")
    print("runs", len(summ), "episode rows", len(eps), "missing", missing)
    for r in summ:
        print(r["task"], r["perturbation_level_cm"], "isaac", r["isaac_5070_n_completed"], "mujoco", r["mujoco_n_completed"])


if __name__ == "__main__":
    main()
