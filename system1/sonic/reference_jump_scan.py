#!/usr/bin/env python3
"""T01b supplement 2: scan every task's paired runtime reference for adjacent-frame jumps. Report only.

For each of the 12 role references (6 tasks x agent 0/1) and the 6 object trajectories, over all
adjacent frame pairs (k-1, k):
  * position jump: max over bodies of ||p_k - p_{k-1}|| (m), and the pelvis alone;
  * orientation jump: max over bodies of the rotation angle between q_{k-1} and q_k (deg), and the pelvis alone;
  * joint jump: max |q_k - q_{k-1}| (rad);
each with the frame index k where it occurs, plus the 99th percentile of the per-frame maxima and the
largest value excluding the reported frame (second-largest frame), so isolated outliers are visible.
References are loaded with the paper environment's loaders and file (system1/sonic/task_refs.py), but with
every body present in the reference file (51; Push's single-person A1 file has 52 including non-robot entries,
kept as loaded), so finger and rubber-hand links are included. Columns prefixed sim_ restrict to the 32 simulator
bodies (the set the environment and SONIC see).

    hssim_s1/bin/python -B system1/sonic/reference_jump_scan.py results_system1/T01b/reference_jumps
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from system1.sonic.task_refs import TASKS, load_task_reference  # noqa: E402


def quat_angle_deg(q0, q1):
    """Rotation angle between unit quaternions (any component order, same for both), degrees."""
    d = np.abs(np.sum(q0 * q1, axis=-1)) / (np.linalg.norm(q0, axis=-1) * np.linalg.norm(q1, axis=-1))
    return np.degrees(2 * np.arccos(np.clip(d, 0.0, 1.0)))


def stat(per_frame):
    """per_frame[k-1] = jump between frames k-1 and k."""
    k = int(np.argmax(per_frame))
    rest = np.delete(per_frame, k)
    return dict(max=float(per_frame[k]), frame=k + 1, p99=float(np.percentile(per_frame, 99)),
                second_max=float(rest.max()) if rest.size else None, second_frame=int(np.argmax(rest) + (1 if np.argmax(rest) < k else 2)) if rest.size else None)


def main():
    out = Path(sys.argv[1])
    rows = []
    from holosoma.envs.marl import PairedA1Reference, PairedMotionReference
    from holosoma.managers.command.terms.wbt import MotionLoader
    from holosoma.utils.path import resolve_data_file_path
    from system1.sonic.task_refs import PUSH_PROOF, reference_file

    for task in TASKS:
        _, sim_bodies, dofs, meta = load_task_reference(task)
        path = resolve_data_file_path(str(reference_file(task)))
        with np.load(path, allow_pickle=True) as z:
            bodies = [str(b) for b in z["body_names"]]
        if task == "push":
            spacing = float(json.loads(PUSH_PROOF.read_text())["lateral_spacing_m"])
            ref = PairedA1Reference.from_motion_loader(MotionLoader(str(reference_file(task)), bodies, dofs, device="cpu"),
                                                       lateral_spacing_m=spacing)
        else:
            ref = PairedMotionReference(str(reference_file(task)), bodies, dofs, device="cpu")
        a = {k: v.numpy() for k, v in ref.sample(torch.arange(ref.num_frames)).items()}
        n = ref.num_frames
        pel = bodies.index("pelvis")
        sim_idx = [bodies.index(b) for b in sim_bodies if b in bodies]
        for r in range(2):
            pos, quat, q = a["agent_body_pos_w"][:, r], a["agent_body_quat_w"][:, r], a["agent_joint_pos"][:, r]
            dp = np.linalg.norm(np.diff(pos, axis=0), axis=-1)            # [N-1, B]
            da = quat_angle_deg(quat[:-1], quat[1:])                      # [N-1, B]
            dq = np.abs(np.diff(q, axis=0))                               # [N-1, J]
            row = dict(task=task, segment=f"{task}_agent{r}", frames=n, n_bodies=len(bodies), n_sim_bodies_found=len(sim_idx),
                       reference_file=meta["file"], reference_sha256=meta["sha256"])
            for name, per in (("body_pos_m", dp.max(1)), ("sim_body_pos_m", dp[:, sim_idx].max(1)), ("pelvis_pos_m", dp[:, pel]),
                              ("body_ori_deg", da.max(1)), ("sim_body_ori_deg", da[:, sim_idx].max(1)),
                              ("pelvis_ori_deg", da[:, pel]), ("joint_rad", dq.max(1))):
                for k, v in stat(per).items():
                    row[f"{name}_{k}"] = v
            row["body_pos_m_body"] = bodies[int(dp[row["body_pos_m_frame"] - 1].argmax())]
            row["body_ori_deg_body"] = bodies[int(da[row["body_ori_deg_frame"] - 1].argmax())]
            row["joint_rad_joint"] = dofs[int(dq[row["joint_rad_frame"] - 1].argmax())]
            rows.append(row)
        op, oq = a["object_pos_w"], a["object_quat_w"]
        row = dict(task=task, segment=f"{task}_object", frames=n, reference_file=meta["file"], reference_sha256=meta["sha256"])
        for name, per in (("body_pos_m", np.linalg.norm(np.diff(op, axis=0), axis=-1)), ("body_ori_deg", quat_angle_deg(oq[:-1], oq[1:]))):
            for k, v in stat(per).items():
                row[f"{name}_{k}"] = v
        rows.append(row)
        print(task, n, flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with out.with_suffix(".csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    out.with_suffix(".json").write_text(json.dumps(dict(
        definition="adjacent frames (k-1,k); position = max over bodies of Euclidean displacement (not per-axis); orientation = max over bodies of "
                   "quaternion rotation angle; joint = max abs joint-angle change; frame = k; p99 over the N-1 frame pairs; "
                   "second_max = largest value at any other frame. body_* use every body in the reference file, sim_body_* the 32 simulator bodies. "
                   "Object rows use the object pose. Report only, nothing modified.",
        fps=50, rows=rows), indent=1) + "\n")


if __name__ == "__main__":
    main()
