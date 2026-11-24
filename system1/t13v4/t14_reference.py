#!/usr/bin/env python3
"""T13 v4 section 0 item 1: T14 round 6 C references (Kevin 2026-11-24) as training / evaluation inputs, behind a switch.

Table = T14 round 6 C, k = 1.0 (exports_system1/T14_references/r6/table_k1p0_C); Push = T14 round 6 C, m = 1.0, i.e. the
round 6 C single-robot solve placed by the original A1 loader (PairedA1Reference, lateral spacing 0.8780472 m = 2 c(1),
the same construction dual_stretch.py uses for the Push levels). Training object URDFs, robot, physics, rewards and the
loaders are unchanged; only the motion files (and the Table bucket vectors computed from the new motion, as
scripts/prepare_core4d_smalltable_a.py does for the original) change.

Assets (tracked, like the original references): src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/
system1_t14r6C/{table,push}/ with a manifest t14_reference.json each (every file's sha256).

    # offline preparation (hsretargeting_s1 for the Table vectors and the Push conversion; hssim_s1 for the loader channels)
    hsretargeting_s1/bin/python system1/t13v4/t14_reference.py prepare-table
    hsretargeting_s1/bin/python system1/t13v4/t14_reference.py prepare-push
    hssim_s1/bin/python system1/t13v4/t14_reference.py loader-channels   # Push CPU loader channels (+ reproduction check)

Runtime (switch --t14-reference <manifest> in train_core4d_smalltable.py / train_plan5_push.py / evaluate_paper_supplement.py;
off = unchanged): table_experiment(), push_config(), push_runtime_check().
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "src/holosoma"
MOTION = PACKAGE / "holosoma/data/motions/g1_29dof/whole_body_tracking"
ASSETS = MOTION / "system1_t14r6C"
SCHEMA = "system1_t14_reference_v1"

TABLE_SRC = ROOT / "exports_system1/T14_references/r6/table_k1p0_C"
TABLE_OLD = MOTION / "core4d_smalltable5kg_A"
PUSH_SINGLE = ROOT / "logs_system1/t14/r6/single_push_C/push_new_qpos.npz"
PUSH_PAIR = ROOT / "exports_system1/T14_references/r6/push_sizes_C/m1p00/push_m1p00_pair_fps30.npz"
PUSH_OLD_MOTION = MOTION / "rubber_hand_largetable_v1/a1/sub6_largetable_033_a1_mj_fps50_w_obj.npz"
PUSH_OLD_CHANNELS = ROOT / "system1/evidence/checks/push_registered_loader_channels.npz"
PUSH_SPACING_M = 0.8780472  # registered Push spacing = 2 c(1) of dual_stretch.py
PUSH_C_M = 0.7 - 0.5219528 / 2


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rel(p) -> str:
    return str(Path(p).resolve().relative_to(PACKAGE))


def _write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False,
                                     default=lambda x: x.tolist() if isinstance(x, (np.ndarray, np.generic)) else str(x)) + "\n")


# ----------------------------------------------------------------------------------------------------------------- Table
def prepare_table():
    """New Table asset dir: runtime + source pair (copied), the unchanged 5 kg training URDF (copied byte for byte), bucket
    vectors from the new runtime (same algorithm and object-point seed as prepare_core4d_smalltable_a.py) and a promotion
    manifest with the same fields as the original one."""
    sys.path.insert(0, str(ROOT / "src/holosoma_retargeting"))
    from holosoma_retargeting.interaction_mesh_geometry import build_robot_landmarks, landmarks_world, load_box_object_mesh, sample_object_points
    from holosoma_retargeting.interaction_mesh_training_reference import map_landmarks_to_training, training_body_names
    from scipy.spatial.transform import Rotation

    out = ASSETS / "table"
    out.mkdir(parents=True, exist_ok=False)
    src_runtime, src_pair = TABLE_SRC / "core4d_pair_runtime_fps50.npz", TABLE_SRC / "20231030_001_desk001_move_new_fps30.npz"
    man = json.loads((TABLE_SRC / "core4d_pair_runtime_fps50.manifest.json").read_text())
    if man["output_sha256"] != sha(src_runtime) or man["source_sha256"] != sha(src_pair):
        raise ValueError("T14 r6 C Table runtime / source changed since export")
    runtime, pair, urdf = out / "core4d_pair_runtime_fps50.npz", out / src_pair.name, out / "desk001_5kg_training.urdf"
    shutil.copyfile(src_runtime, runtime)
    shutil.copyfile(src_pair, pair)
    shutil.copyfile(TABLE_OLD / "desk001_5kg_training.urdf", urdf)
    old_prom = json.loads((TABLE_OLD / "training_asset_manifest.json").read_text())
    if sha(urdf) != old_prom["training_object_urdf_sha256"]:
        raise ValueError("training URDF differs from the original 5 kg table")
    robot_xml = ROOT / "src/holosoma_retargeting/holosoma_retargeting/models/g1/g1_29dof.xml"
    robot_urdf = PACKAGE / "holosoma/data/robots/g1/main_mesh_collision_rubberhand.urdf"
    specs, landmark_metadata = build_robot_landmarks(robot_xml)
    mapped, mapping_metadata = map_landmarks_to_training(specs, robot_xml, robot_urdf, training_body_names(PACKAGE / "holosoma/config_values/robot.py"))
    with np.load(runtime, allow_pickle=False) as z:
        if z["agent_joint_pos"].shape != (687, 2, 29) or z["fps"].item() != 50:
            raise ValueError("Unexpected runtime layout")
        points = landmarks_world(z["agent_body_pos_w"], z["agent_body_quat_w"], z["body_names"], specs)
        mapped_points = landmarks_world(z["agent_body_pos_w"], z["agent_body_quat_w"], z["body_names"], mapped)
        rotation = Rotation.from_quat(z["object_quat_w"][:, [1, 2, 3, 0]]).as_matrix()
        object_pos = z["object_pos_w"].copy()
    error = float(np.linalg.norm(points - mapped_points, axis=-1).max())
    if points.shape != (687, 2, 19, 3) or error > 1e-6:
        raise ValueError("Fixed-chain landmark mapping mismatch")
    mesh, mesh_meta = load_box_object_mesh(urdf)
    object_points, sampling = sample_object_points(mesh, mesh_meta, 100, seed=42)
    with np.load(TABLE_OLD / "interaction_vectors_v1.npz", allow_pickle=False) as o:
        if not np.array_equal(o["object_points"], object_points):
            raise ValueError("object points differ from the original artifact (same URDF, same seed expected)")
    objects = np.einsum("tij,nj->tni", rotation, object_points) + object_pos[:, None]
    metadata = {
        "version": "core4d_bucket_vectors_v1", "experiment_id": "smalltable5kg_A",
        "runtime_reference_sha256": sha(runtime), "source_pair_sha256": sha(pair),
        "object_urdf_sha256": sha(urdf), "training_robot_urdf_sha256": sha(robot_urdf),
        "num_frames": 687, "num_agents": 2, "fps": 50, "num_body_points": 19,
        "actual_object_points": len(object_points), "sampling": sampling,
        "robot_landmarks": landmark_metadata, "point_mapping": mapping_metadata,
        "fixed_chain_reference_equivalence_max_m": error,
        "contact_target": "not_used_for_smalltable_A",
        "reference_contact_weights_meaning": "all-zero unused compatibility mask; not estimated contact labels; B forbidden",
        "coordinate_frame": "world relative vectors; object points in URDF link frame, not COM",
        "reward_reuse": "unchanged bucket A formulas and weights, table-specific geometry/reference",
        "system1_t14_reference": "T14 round 6 C, k = 1.0 (Kevin 2026-11-24)",
    }
    arrays = {
        "reference_body_points_world": points.astype(np.float32),
        "reference_object_points_world": objects.astype(np.float32),
        "object_points": object_points,
        "point_offsets": np.asarray([s["local_xyz"] for s in mapped], dtype=np.float32),
        "point_body_names": np.asarray([s["body"] for s in mapped]),
        "point_names": np.asarray([s["name"] for s in mapped]),
        "body_point_weights": np.asarray([1 / 3 if s["group"] in ("left_hand", "right_hand") else 1 for s in specs], dtype=np.float32),
        "reference_contact_weights": np.zeros((687, 2), dtype=np.float32),
        "fps": np.asarray(50, dtype=np.int64),
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True, default=lambda x: x.tolist() if isinstance(x, (np.ndarray, np.generic)) else str(x))),
    }
    with np.load(TABLE_OLD / "interaction_vectors_v1.npz", allow_pickle=False) as o:  # same layout as the original artifact
        for k in ("object_points", "point_offsets", "point_body_names", "point_names", "body_point_weights", "reference_contact_weights"):
            if not np.array_equal(o[k], arrays[k]):
                raise ValueError(f"non-reference array {k} differs from the original artifact")
        if set(o.files) != set(arrays):
            raise ValueError("artifact keys differ from the original")
    artifact = out / "interaction_vectors_v1.npz"
    with artifact.open("xb") as f:
        np.savez_compressed(f, **arrays)
    prom = dict(old_prom)
    prom.update(source_pair_sha256=sha(pair), runtime_reference_sha256=sha(runtime), interaction_artifact_sha256=sha(artifact),
                fixed_chain_reference_equivalence_max_m=error, preparation_script_sha256=sha(__file__),
                system1_t14_reference=dict(source="T14 round 6 C, k = 1.0", runtime_source=str(src_runtime.relative_to(ROOT)),
                                           original_manifest_sha256=sha(TABLE_OLD / "training_asset_manifest.json")))
    _write_json(out / "training_asset_manifest.json", prom)
    manifest = dict(schema=SCHEMA, task="table", name="t14_r6C_k1p0", parent_experiment="smalltable5kg_A",
                    decision="Kevin 2026-11-24: Table training reference = T14 round 6 C, k = 1.0",
                    files={k: dict(path=rel(p), sha256=sha(p)) for k, p in dict(
                        runtime_reference=runtime, source_pair=pair, object_urdf=urdf, interaction_artifact=artifact,
                        training_promotion=out / "training_asset_manifest.json").items()},
                    unchanged_vs_original=dict(object_urdf=True, robot_urdf=True, reward_formulas=True, frames=687, fps=50,
                                               object_points_seed42=True),
                    t14_solve_dirs=[str(Path("logs_system1/t14/r6/table_k1p0_C") / p) for p in ("person1", "person2")])
    _write_json(out / "t14_reference.json", manifest)
    print(json.dumps(manifest, indent=1))


def load_manifest(path, task):
    path = Path(path).resolve()
    d = json.loads(path.read_text())
    if d.get("schema") != SCHEMA or d.get("task") != task:
        raise ValueError(f"not a {SCHEMA} manifest for {task}: {path}")
    for k, f in d["files"].items():
        p = PACKAGE / f["path"]
        if not p.is_file() or sha(p) != f["sha256"]:
            raise ValueError(f"T14 reference asset changed or missing: {k} {p}")
    return d, sha(path)


def table_experiment(experiment, manifest_path):
    """Replace the reference identities of smalltable5kg_A (URDF, physics, reward variant unchanged)."""
    if manifest_path is None:
        return experiment, None
    d, msha = load_manifest(manifest_path, "table")
    if experiment.experiment_id != d["parent_experiment"]:
        raise ValueError("T14 Table reference is for smalltable5kg_A only")
    f = d["files"]
    if f["object_urdf"]["sha256"] != experiment.object_urdf_sha256:
        raise ValueError("T14 Table reference must keep the training URDF")
    prom = json.loads((PACKAGE / f["training_promotion"]["path"]).read_text())
    if prom.get("training_ready") is not True or prom["runtime_reference_sha256"] != f["runtime_reference"]["sha256"]:
        raise ValueError("T14 Table promotion manifest inconsistent")
    new = replace(experiment, source_pair_sha256=f["source_pair"]["sha256"],
                  runtime_reference_file=f["runtime_reference"]["path"], runtime_reference_sha256=f["runtime_reference"]["sha256"],
                  training_promotion_file=f["training_promotion"]["path"], training_promotion_sha256=f["training_promotion"]["sha256"])
    return new, dict(manifest=str(Path(manifest_path).resolve().relative_to(ROOT)), manifest_sha256=msha, name=d["name"],
                     runtime_reference_sha256=f["runtime_reference"]["sha256"])


def check_labels(labels_npz, manifest_path, task):
    """The contact label file must have been made from this reference (its json record holds the runtime sha256)."""
    if labels_npz is None or manifest_path is None:
        return None
    d, _ = load_manifest(manifest_path, task)
    want = d["files"]["runtime_reference" if task == "table" else "motion"]["sha256"]
    rec = json.loads(Path(labels_npz).with_suffix(".json").read_text())
    if rec.get("runtime_sha256") != want:
        raise ValueError(f"contact labels {labels_npz} were not made from the T14 {task} reference ({rec.get('runtime_sha256')} != {want})")
    return dict(labels=str(Path(labels_npz).resolve().relative_to(ROOT)), labels_sha256=sha(labels_npz), runtime_sha256=want)


# ------------------------------------------------------------------------------------------------------------------ Push
def prepare_push():
    """Convert the round 6 C single-robot solve with the original converter command (MANIFEST.md of
    rubber_hand_largetable_v1; reproduces the original A1 file to 5e-15) and check that the T14 m = 1.0 pair equals the
    single solve placed at -/+ c(1) along the table's local x (the A1 loader's construction)."""
    out = ASSETS / "push"
    out.mkdir(parents=True, exist_ok=False)
    single = out / "sub6_largetable_033_t14r6C_fixed_object.npz"
    shutil.copyfile(PUSH_SINGLE, single)
    motion = out / "sub6_largetable_033_t14r6C_mj_fps50_w_obj.npz"
    hs = "/home/kevin/.holosoma_deps/miniconda3/envs/hsretargeting_s1/bin/python"
    cmd = [hs, "data_conversion/convert_data_format_mj.py", "--input-file", str(single), "--output-fps", "50", "--output-name", str(motion),
           "--data-format", "smplh", "--object-name", "largetable", "--has-dynamic-object", "--once", "--headless"]
    subprocess.run(cmd, cwd=ROOT / "src/holosoma_retargeting/holosoma_retargeting", check=True, env=dict(__import__("os").environ, CUDA_VISIBLE_DEVICES=""))
    # pair check (30 Hz, source frames): robot r root = single root + (2r - 1) c x_local(table), joints equal, object equal
    s = np.load(single)["qpos"]
    pz = np.load(PUSH_PAIR)
    from scipy.spatial.transform import Rotation
    xl = Rotation.from_quat(s[:, [40, 41, 42, 39]]).apply([1.0, 0.0, 0.0])  # object quat wxyz in qpos[39:43] -> scipy xyzw
    pair_check = dict(c_m=float(pz["c"]), m=float(pz["m"]), spacing_m=2 * float(pz["c"]))
    for r, sign in ((0, -1.0), (1, 1.0)):
        q = pz["robot_qpos"][:, r]
        pair_check[f"robot{r}_root_pos_max_abs_m"] = float(np.abs(q[:, :3] - (s[:, :3] + sign * float(pz["c"]) * xl)).max())
        pair_check[f"robot{r}_rest_max_abs"] = float(np.abs(q[:, 3:36] - s[:, 3:36]).max())
    pair_check["object_max_abs"] = float(np.abs(pz["object_qpos"] - s[:, 36:43]).max())
    if abs(pair_check["spacing_m"] - PUSH_SPACING_M) > 1e-9 or max(v for k, v in pair_check.items() if k.endswith("max_abs_m") or k.endswith("max_abs")) > 1e-9:
        raise ValueError(f"T14 m = 1.0 pair is not the A1-loader placement of the single solve: {pair_check}")
    manifest = dict(schema=SCHEMA, task="push", name="t14_r6C_m1p00", loader="original_PairedA1Reference",
                    decision="Kevin 2026-11-24: Push training reference = T14 round 6 C, m = 1.0",
                    lateral_spacing_m=PUSH_SPACING_M, conversion_command=cmd[1:], pair_check=pair_check,
                    t14_pair=dict(path=str(PUSH_PAIR.relative_to(ROOT)), sha256=sha(PUSH_PAIR)),
                    files={k: dict(path=rel(p), sha256=sha(p)) for k, p in dict(motion=motion, source_qpos=single).items()},
                    object_urdf_unchanged="objects_widetable_plan5_training.urdf (training table, Kevin: 训练用的桌子尺寸不变)")
    _write_json(out / "t14_reference.json", manifest)
    print(json.dumps(manifest, indent=1))


def loader_channels(motion_file, spacing, body_names, joint_names):
    """The original runtime loader on CPU: MotionLoader + PairedA1Reference.from_motion_loader, all frames."""
    import torch
    sys.path.insert(0, str(PACKAGE))
    from holosoma.envs.marl import PairedA1Reference
    from holosoma.managers.command.terms.wbt import MotionLoader
    m = MotionLoader(str(motion_file), list(body_names), list(joint_names), device="cpu")
    ref = PairedA1Reference.from_motion_loader(m, lateral_spacing_m=spacing)
    return {k: v.numpy() for k, v in ref.sample(torch.arange(ref.num_frames)).items()}


def make_loader_channels():
    old = np.load(PUSH_OLD_CHANNELS)
    names, joints = [str(n) for n in old["body_names"]], [str(n) for n in old["joint_names"]]
    rep = loader_channels(PUSH_OLD_MOTION, PUSH_SPACING_M, names, joints)
    repro = {k: float(np.abs(rep[k].astype(float) - old[k].astype(float)).max()) for k in rep}
    if max(repro.values()) > 1e-6:
        raise ValueError(f"CPU loader does not reproduce the original expected channels: {repro}")
    man_path = ASSETS / "push/t14_reference.json"
    d, _ = load_manifest(man_path, "push")
    new = loader_channels(PACKAGE / d["files"]["motion"]["path"], d["lateral_spacing_m"], names, joints)
    out = ASSETS / "push/push_t14r6C_loader_channels.npz"
    np.savez_compressed(out, **new, body_names=old["body_names"], joint_names=old["joint_names"], fps=old["fps"])
    d["files"]["loader_channels"] = dict(path=rel(out), sha256=sha(out))
    d["loader_channels_reproduction_of_original_max_abs"] = repro
    _write_json(man_path, d)
    print(json.dumps(dict(reproduction=repro, channels=d["files"]["loader_channels"]), indent=1))


def push_config(config, manifest_path, placement_provenance):
    """Swap only the A1 motion file of the paired motion command (setup / reset / step terms); spacing must be the
    registered one already set by configure_push_placement."""
    if manifest_path is None:
        return config, placement_provenance
    if placement_provenance is None:
        raise ValueError("--t14-reference for Push goes with --push-placement-proof (registered spacing)")
    d, msha = load_manifest(manifest_path, "push")
    if abs(placement_provenance["lateral_spacing_m"] - d["lateral_spacing_m"]) > 1e-9:
        raise ValueError("registered spacing differs from the T14 m = 1.0 placement")
    motion = d["files"]["motion"]["path"]
    hooks = {}
    for name in ("setup_terms", "reset_terms", "step_terms"):
        terms = dict(getattr(config.command, name))
        term = terms["paired_motion_command"]
        if term.params.get("paired_reference_file") is not None:
            raise ValueError("Do not replace the original Push reference loader")
        mc = term.params["motion_config"]
        terms["paired_motion_command"] = replace(term, params=dict(term.params, motion_config=replace(mc, motion_file=motion)))
        hooks[name] = terms
    config = replace(config, command=replace(config.command, **hooks))
    prov = dict(placement_provenance, t14_reference=dict(manifest=str(Path(manifest_path).resolve().relative_to(ROOT)), manifest_sha256=msha,
                                                       name=d["name"], motion_file=motion, motion_sha256=d["files"]["motion"]["sha256"]))
    return config, prov


def push_runtime_check(env, output, manifest_path):
    """As plan5_registered_placement.export_runtime_reference, against the CPU loader channels of the T14 motion."""
    import torch
    sys.path.insert(0, str(ROOT / "scripts"))
    from plan5_registered_placement import comparison_body_indexes
    d, _ = load_manifest(manifest_path, "push")
    command = env.command_manager.get_state("paired_motion_command")
    samples = command.reference.sample(torch.arange(command.reference.num_frames, device=env.device))
    arrays = {k: v.detach().cpu().numpy() for k, v in samples.items()}
    expected_path = PACKAGE / d["files"]["loader_channels"]["path"]
    errors = {}
    with np.load(expected_path, allow_pickle=False) as expected:
        idx, eidx, names, offline = comparison_body_indexes(env.simulator._body_list, expected["body_names"], command.motion_cfg.body_names_to_track)
        for key, value in arrays.items():
            sel = value[:, :, idx] if key.startswith("agent_body_") else value
            tgt = expected[key][:, :, eidx] if key.startswith("agent_body_") else expected[key]
            if sel.shape != tgt.shape:
                raise ValueError("Runtime reference shape differs: " + key)
            mx = float(np.abs(sel.astype(float) - tgt.astype(float)).max())
            tol = 1e-4 if "vel" in key else 2e-6
            if not np.isfinite(mx) or mx > tol:
                raise ValueError(f"Runtime reference channel differs: {key}: {mx}")
            errors[key] = dict(max_abs=mx, tolerance=tol)
    path = Path(output) / "actual_reference_channels.npz"
    np.savez_compressed(path, **arrays, body_names=env.simulator._body_list, joint_names=env.simulator.dof_names, fps=command.reference.fps)
    return dict(file=path.name, sha256=sha(path), expected_channels_sha256=sha(expected_path), comparison=errors,
                compared_body_names=names, offline_only_body_names=offline, t14_reference=d["name"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("prepare-table", "prepare-push", "loader-channels"))
    a = ap.parse_args()
    dict(**{"prepare-table": prepare_table, "prepare-push": prepare_push, "loader-channels": make_loader_channels})[a.step]()
