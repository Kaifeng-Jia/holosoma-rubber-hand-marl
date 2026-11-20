#!/usr/bin/env python3
"""T10 return pack (results_system1/T10_sim2sim/): episodes.csv, per_seed_summary.csv, isaac_vs_mujoco.csv (Isaac laptop-5070
finals, results_system1/T03/c2_main/per_seed_summary.csv, next to MuJoCo), configs/ (MuJoCo run manifests and launch / exit
records, Isaac export records and scene.json, gzip; the two MJCF scenes as compiled), provenance.json (sha256 of checkpoints,
ONNX, URDF, MJCF, state files, Isaac exports). CPU only, read only.

    sim2sim_s1/bin/python -B system1/sim2sim/package_t10.py --evals logs_system1/t10/mj_eval --out results_system1/T10_sim2sim
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


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()


def save_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--isaac-runs", default="d_table:table,cin_table:table,d_bucket:bucket,cin_bucket:bucket")
    a = ap.parse_args()
    isaac_runs = [tuple(x.split(":")) for x in a.isaac_runs.split(",")]
    out = a.out.resolve()
    a.evals = a.evals.resolve()
    (out / "configs").mkdir(parents=True, exist_ok=True)
    eps, per_seed, missing, prov = [], [], [], dict(runs=[])
    for d in sorted(p for p in a.evals.iterdir() if p.is_dir()):
        s = json.loads((d / "summary.json").read_text()) if (d / "summary.json").exists() else None
        ex = d.with_name(d.name + ".exit.json")
        if not s or not s.get("complete") or len(s["episodes"]) != 51 or not ex.exists() or json.loads(ex.read_text())["returncode"] != 0:
            missing.append(d.name)
            continue
        pd = s["pd"] if s["scheme"] == "C-in" else "paper"
        for e in s["episodes"]:
            eps.append(dict(simulator="mujoco", task=s["task"], scheme=s["scheme"], pd_setting=pd, training_seed=s["seed"], perturbation_level_cm=s["level"],
                            state_id=e["state_id"], is_nominal=e["kind"] == "nominal", completed_reference=e["completed_reference"], executed_steps=e["steps"],
                            reference_horizon_steps=e["horizon_steps"], progress=e["progress"], termination_reason=json.dumps(e["terminal_terms"], sort_keys=True),
                            object_pos_rmse_cm=e["object_rmse_3d_m"] * 100, object_xy_rmse_cm=e["object_rmse_xy_m"] * 100, height_rmse_cm=e["height_rmse_m"] * 100,
                            last_object_pos_err_m=e["last_termination"]["object_pos_err"], last_object_ori_err_rad=e["last_termination"]["object_ori_err"],
                            last_torso_gravity_z_err_max=max(e["last_termination"]["gravity_z_err"]), checkpoint_sha256=s["checkpoint_sha256"],
                            onnx_sha256=s["onnx_sha256"], state_file_sha256=s["states_sha256"], eval_dir=str(d.relative_to(ROOT))))
        per = [e for e in s["episodes"] if e["kind"] != "nominal"]
        ok = [e for e in per if e["completed_reference"]]
        nom = [e for e in s["episodes"] if e["kind"] == "nominal"][0]
        per_seed.append(dict(simulator="mujoco", task=s["task"], scheme=s["scheme"], pd_setting=pd, training_seed=s["seed"], perturbation_level_cm=s["level"],
                             n_episodes=len(per), n_completed=len(ok), completion_pct=100 * len(ok) / len(per),
                             object_pos_rmse_cm_mean_completed=statistics.mean(e["object_rmse_3d_m"] * 100 for e in ok) if ok else None,
                             steps_mean=statistics.mean(e["steps"] for e in per), nominal_completed=nom["completed_reference"], nominal_steps=nom["steps"],
                             terminated_object_pos=sum(1 for e in per if not e["completed_reference"] and e["last_termination"]["object_pos_err"] > 0.25),
                             terminated_object_ori=sum(1 for e in per if not e["completed_reference"] and e["last_termination"]["object_ori_err"] > 0.8),
                             terminated_robot_orientation=sum(1 for e in per if not e["completed_reference"] and max(e["last_termination"]["gravity_z_err"]) > 0.8),
                             wall_s=s.get("wall_s"), checkpoint_sha256=s["checkpoint_sha256"], onnx_sha256=s["onnx_sha256"], eval_dir=str(d.relative_to(ROOT))))
        for f in (d / "run_manifest.json", d.with_name(d.name + ".launch.json"), ex):
            (out / "configs" / f"mujoco__{d.name}__{f.name.replace(d.name + '.', '')}.gz").write_bytes(gzip.compress(f.read_bytes(), mtime=0))
        prov["runs"].append(dict(name=d.name, summary_sha256=sha(d / "summary.json"), onnx=s["onnx"], onnx_sha256=s["onnx_sha256"],
                                 checkpoint_sha256=s["checkpoint_sha256"], states=s["states"], states_sha256=s["states_sha256"],
                                 scene_json_sha256=s["scene_json_sha256"], convexes_sha256=s["convexes_sha256"]))
    save_csv(out / "episodes.csv", eps)
    save_csv(out / "per_seed_summary.csv", per_seed)
    # Isaac (laptop 5070) next to MuJoCo
    isa = {}
    for f in ("c2_main", "d_final_5070"):  # D Push / Kick / Chair finals are in d_final_5070
        for r in csv.DictReader((ROOT / f"results_system1/T03/{f}/per_seed_summary.csv").open()):
            if r["variant"] in ("D", "C-in"):
                isa[(r["task"], r["variant"], int(r["training_seed"]), int(r["perturbation_level_cm"]))] = r
    cmp = []
    for r in per_seed:
        i = isa.get((r["task"], r["scheme"], r["training_seed"], r["perturbation_level_cm"]))
        cmp.append(dict(task=r["task"], scheme=r["scheme"], pd_setting=r["pd_setting"], training_seed=r["training_seed"], perturbation_level_cm=r["perturbation_level_cm"],
                        isaac_5070_n_completed=int(i["n_completed"]) if i else None, mujoco_n_completed=r["n_completed"], n_episodes=50,
                        difference_mujoco_minus_isaac=r["n_completed"] - int(i["n_completed"]) if i else None,
                        isaac_nominal_completed=i["nominal_completed"] if i else None, mujoco_nominal_completed=r["nominal_completed"],
                        isaac_checkpoint_sha256=i["checkpoint_sha256"] if i else None, same_checkpoint=(i["checkpoint_sha256"] == r["checkpoint_sha256"]) if i else None,
                        isaac_states_sha256=i["states_sha256"] if i else None))
    cmp.sort(key=lambda x: (x["task"], x["scheme"], x["pd_setting"], x["perturbation_level_cm"], x["training_seed"]))
    save_csv(out / "isaac_vs_mujoco.csv", cmp)
    # Isaac export records and scene.json; compiled MJCF scenes
    from system1.sim2sim.mj_env import SceneSpec, build_model
    for run, _ in isaac_runs:
        base = ROOT / "logs_system1/t10/isaac" / run
        for f in (base / "launch.json", base / "exit.json", base / "dump/scene.json"):
            (out / "configs" / f"isaac__{run}__{f.name}.gz").write_bytes(gzip.compress(f.read_bytes(), mtime=0))
        prov[f"isaac_{run}"] = dict(scene_json_sha256=sha(base / "dump/scene.json"), convexes_sha256=sha(base / "dump/collision_convexes.npz"),
                                    record_sha256=sha(base / "dump/nominal.sim2sim.npz"), eval_summary_sha256=sha(base / "eval/summary.json"),
                                    runtime_reference_sha256=sha(base / "dump/runtime_reference.npz") if (base / "dump/runtime_reference.npz").exists() else None)
    for run, task in isaac_runs:
        _, spec, _ = build_model(SceneSpec(ROOT / "logs_system1/t10/isaac" / run / "dump"), task)
        p = out / "configs" / f"mjcf_{run}.xml"
        p.write_text(spec.to_xml())
        prov[f"mjcf_{run}_sha256"] = sha(p)
    prov.update(robot_urdf=dict(path="src/holosoma/holosoma/data/robots/g1/main_mesh_collision_rubberhand.urdf",
                                sha256=sha(ROOT / "src/holosoma/holosoma/data/robots/g1/main_mesh_collision_rubberhand.urdf")),
                missing=missing, n_runs=len(per_seed), n_episodes=len(eps))
    (out / "provenance.json").write_text(json.dumps(prov, indent=1) + "\n")
    print("runs", len(per_seed), "episodes", len(eps), "missing", missing)


if __name__ == "__main__":
    main()
