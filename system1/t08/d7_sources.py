#!/usr/bin/env python3
"""T08 D7 source records (docs/system1/T08_slides_assets_zh.md, D7 step 1), read only, CPU: the Stage 1B single-robot A1 wide-table
retention check of 2026-08-15 in /home/kevin/holosoma-rubber-hand-marl (report STAGE1B_A1_RETENTION_REPORT_CN.md of commit
d04b312a, later removed from the tree; recordings moved from logs/WholeBodyTracking/stage1b_a1_retention_v1 to
plan5_preflight_v1 by commit ae7ea70b, which also renamed the table asset).

Writes to --out:
  * d7_sources.json - checkpoint (+ conversion manifest and its source), table URDF (git blob of d04b312a), robot URDF, the
    two recordings and their metric JSONs, the A1 source motion, the reference-shift script (git blob), the report (git blob),
    placement and run facts, every file with sha256;
  * d7_attempts.csv - the 20 closed attempts of each end as recorded in formal_wide_{right,left}_trajectory.json;
  * d7_initial_states.json - the saved state at the first row of each attempt (root pose, 29 joint positions, table pose).
    Attempt 0 of both ends is the first environment reset (seed 42); the other attempts start where the analysis closed the
    previous one (reset after a termination, or motion restart after reaching the reference end).

    python3 system1/t08/d7_sources.py --out results_system1/T08_slides   (any python with numpy)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

MARL = Path("/home/kevin/holosoma-rubber-hand-marl")
REC = MARL / "logs/WholeBodyTracking/plan5_preflight_v1"
CKPT = MARL / "logs/WholeBodyTracking/marl_compat_a1_v1/model_07999_actor158.pt"
CONV = MARL / "logs/WholeBodyTracking/marl_compat_a1_v1/conversion_manifest.json"
SRC_MOTION = MARL / "src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/rubber_hand_largetable_v1/a1/sub6_largetable_033_a1_mj_fps50_w_obj.npz"
ROBOT = MARL / "src/holosoma/holosoma/data/robots/g1/main_mesh_collision_rubberhand.urdf"
COMMIT = "d04b312ab4cf47f4cd830ab1d8630ed5f023d534"
BLOBS = dict(table_urdf="src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/objects_widetable_a1_retention.urdf",
             shift_script="scripts/prepare_stage1b_a1_retention.py", report="STAGE1B_A1_RETENTION_REPORT_CN.md",
             robot_urdf="src/holosoma/holosoma/data/robots/g1/main_mesh_collision_rubberhand.urdf")


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()


def blob(path):
    b = subprocess.run(["git", "-C", str(MARL), "show", f"{COMMIT}:{path}"], capture_output=True, check=True).stdout
    return dict(repo=str(MARL), commit=COMMIT, path=path, sha256=hashlib.sha256(b).hexdigest(), bytes=len(b))


def f(p):
    return dict(path=str(p), sha256=sha(p), bytes=Path(p).stat().st_size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    attempts, init, sides = [], {}, {}
    for side in ("right", "left"):
        js, npz = REC / f"formal_wide_{side}_trajectory.json", REC / f"formal_wide_{side}_trajectory.npz"
        rep = json.loads(js.read_text())
        att = rep["attempt_details"][:20]
        with np.load(npz, allow_pickle=False) as z:
            meta = json.loads(str(z["_metadata_json"]))
            init[side] = [dict(attempt=x["attempt"], row=x["start"], root_pos=z["root_pos"][x["start"]].tolist(),
                               root_quat_xyzw=z["root_quat_xyzw"][x["start"]].tolist(), dof_pos=z["dof_pos"][x["start"]].tolist(),
                               object_pos_w=z["object_pos_w"][x["start"]].tolist(), object_quat_xyzw=z["object_quat_xyzw"][x["start"]].tolist(),
                               motion_time_step=int(z["motion_time_step"][x["start"]]), episode_step=int(z["episode_step"][x["start"]])) for x in att]
            init["dof_names"] = meta["dof_names"]
        for x in att:
            attempts.append(dict(table_end=side, **{k: x[k] for k in ("attempt", "start", "end", "length_steps", "duration_s", "completed", "reached_end",
                                                                 "terminated", "timeout", "fall_proxy", "actual_table_displacement_m",
                                                                 "reference_table_displacement_m", "correct_table_displacement_direction",
                                                                 "contact_success")}))
        sides[side] = dict(recording=f(npz), metrics_json=f(js), attempts_summary=rep["attempts"], recording_meta={k: meta[k] for k in (
            "env_id", "dt", "sim_dt", "control_decimation", "urdf_path", "teammate_provider_class", "teammate_observer_side",
            "teammate_lateral_spacing_m", "motion_names", "motion_files", "motion_time_step_total")},
            fall_proxy_rate_per_attempt=rep["stability"]["fall_proxy_rate_per_attempt"])
    keys = list(attempts[0])
    with (a.out / "d7_attempts.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(attempts)
    (a.out / "d7_initial_states.json").write_text(json.dumps(init, indent=1) + "\n")
    conv = json.loads(CONV.read_text())
    rec = dict(
        check="Stage 1B-1 single-robot A1 retention on the 1.4 m table, 2026-08-15 (report: blob 'report' below, section 5.2 / 6)",
        checkpoint=f(CKPT), checkpoint_conversion_manifest=f(CONV), checkpoint_source=dict(path=conv["source"], sha256=conv["source_sha256"],
        note="actor 154 -> 158 inputs, the 4 new teammate columns zero; deterministic output error 0.0 (conversion manifest)"),
        checkpoint_identification="report names model_07999_actor158.pt; commit ad8a80cf (2026-08-17) names "
        "marl_compat_a1_v1/model_07999_actor158.pt as the exact frozen comparison; sha256 equals the conversion manifest output_sha256",
        table_urdf_git=blob(BLOBS["table_urdf"]), table_urdf_in_this_branch="system1/t08/assets/objects_widetable_a1_retention.urdf",
        table_note="1.4 m five-box table, 0.1 kg (the frozen A1 table's mass and contact parameters); renamed objects_widetable_plan5_preflight.urdf "
        "in ae7ea70b with only the robot name and comments changed",
        robot_urdf_git=blob(BLOBS["robot_urdf"]), robot_urdf_now=f(ROBOT),
        placement="robot references = the A1 reference translated by -0.4 e_x(t) (left end) / +0.4 e_x(t) (right end), e_x = table-local X in "
        "world from the table quaternion of each frame; joint angles, orientations and all object channels unchanged; ghost teammate "
        "(no body) follows the other shifted reference; 0.8 m spacing (report section 3, recording metadata)",
        shift_script_git=blob(BLOBS["shift_script"]), a1_source_motion=f(SRC_MOTION),
        run="seed 42, one environment, Isaac Sim CUDA, 6500 policy steps per end, first 20 closed attempts analysed, no training, no ghost "
        "randomisation (report section 5.2); the exact command line is not recorded",
        report_git=blob(BLOBS["report"]), ends=sides,
        initial_states="d7_initial_states.json (first saved row of each attempt, extracted from the recordings); attempt 0 = first reset at seed 42",
        per_attempt_results="d7_attempts.csv",
        note_fall_proxy="report table section 6 lists fall proxy 0 % for both ends; the recordings' metric JSONs give fall_proxy_rate_per_attempt "
        "0.05 (left, attempt 4) and 0.10 (right, attempts 5 and 10). The rendered attempts (0 of each end) have fall_proxy False.",
        code_sha256=sha(__file__))
    (a.out / "d7_sources.json").write_text(json.dumps(rec, indent=1) + "\n")
    print("d7_sources.json, d7_attempts.csv", len(attempts), "rows, d7_initial_states.json")


if __name__ == "__main__":
    main()
