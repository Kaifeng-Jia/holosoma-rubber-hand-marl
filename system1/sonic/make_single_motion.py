#!/usr/bin/env python3
"""Build single-robot references for SONIC validation runs in holosoma.

Two sources:
  --sonic-dump DUMP --index E   : motion E of a native SONIC dump (validation a);
  --paired NPZ --agent K        : agent K of a paired runtime reference (validation b).

Outputs (``--out-prefix P``):
  P.holosoma.npz : holosoma MotionLoader file (fps, joint_pos[T,36]=root xyz+wxyz+29,
                   joint_vel[T,35], body_* for every URDF link, quaternions wxyz);
  P.sonic.npz    : SONIC-side reference: dof_pos/dof_vel [T,29] in SONIC joint order,
                   root_pos_w [T,3], root_quat_w [T,4] (wxyz), body_pos_w [T,14,3]
                   (SONIC cmd_body_names), plus provenance.
For the SONIC dump the SONIC-side arrays are copied verbatim; body poses for the
holosoma file come from FK on our rubber-hand URDF. For paired references every
array is taken from the paired file (no FK), joint vectors reordered by name.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = json.loads((ROOT / "system1/sonic/sonic_release_spec.json").read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def wxyz_to_xyzw(q):
    return q[..., [1, 2, 3, 0]]


def xyzw_to_wxyz(q):
    return q[..., [3, 0, 1, 2]]


def ang_vel_from_quat_wxyz(q, fps):
    r = Rotation.from_quat(wxyz_to_xyzw(q))
    d = (r[1:] * r[:-1].inv()).as_rotvec() * fps  # world-frame angular velocity between frames
    w = np.empty((len(q), 3))
    w[1:-1] = 0.5 * (d[1:] + d[:-1])
    w[0], w[-1] = d[0], d[-1]
    return w


def from_sonic_dump(dump, e):
    from prepare_experiments_r2_cpu import JOINTS, BatchFK

    z = np.load(dump)
    s_names = SPEC["joint_names"]
    perm = [s_names.index(n) for n in JOINTS]  # holosoma order from SONIC order
    dof_pos_s, dof_vel_s = z[f"ref{e}/dof_pos"], z[f"ref{e}/dof_vel"]
    root_pos, root_quat = z[f"ref{e}/root_pos_w"].astype(np.float64), z[f"ref{e}/root_quat_w"].astype(np.float64)
    q_h, qd_h = dof_pos_s[:, perm], dof_vel_s[:, perm]
    fps = 50
    fk = BatchFK()
    links = [fk.base] + [j.child for j in fk.joints]
    T = fk.poses(root_pos, wxyz_to_xyzw(root_quat), q_h, links)
    body_pos = T[..., :3, 3]
    body_quat = xyzw_to_wxyz(Rotation.from_matrix(T[..., :3, :3].reshape(-1, 3, 3)).as_quat().reshape(len(q_h), len(links), 4))
    body_lin = np.gradient(body_pos, 1.0 / fps, axis=0)
    body_ang = np.stack([ang_vel_from_quat_wxyz(body_quat[:, i], fps) for i in range(len(links))], axis=1)
    holo = dict(
        fps=np.array([fps]), joint_names=np.array(JOINTS), body_names=np.array(links),
        joint_pos=np.concatenate([root_pos, root_quat, q_h], axis=1),
        joint_vel=np.concatenate([body_lin[:, 0], body_ang[:, 0], qd_h], axis=1),
        body_pos_w=body_pos, body_quat_w=body_quat, body_lin_vel_w=body_lin, body_ang_vel_w=body_ang,
    )
    # FK check against SONIC's own body positions for the 14 tracked bodies.
    cmd = SPEC["cmd_body_names"]
    fk_err = float(np.abs(body_pos[:, [links.index(b) for b in cmd]] - z[f"ref{e}/body_pos_w"]).max())
    sonic = dict(dof_pos=dof_pos_s, dof_vel=dof_vel_s, root_pos_w=root_pos, root_quat_w=root_quat,
                 body_pos_w=z[f"ref{e}/body_pos_w"], fps=np.array(fps))
    prov = dict(source="sonic_native_dump", dump=str(dump), dump_sha256=sha(dump), index=e,
                fk_vs_sonic_body_pos_max_abs_m=fk_err)
    return holo, sonic, prov


def from_paired(path, k):
    d = np.load(path, allow_pickle=False)
    joints = [str(x) for x in d["joint_names"]]
    bodies = [str(x) for x in d["body_names"]]
    s_names = SPEC["joint_names"]
    perm_s = [joints.index(n) for n in s_names]
    fps = int(np.asarray(d["fps"]).reshape(-1)[0])
    bp, bq = d["agent_body_pos_w"][:, k], d["agent_body_quat_w"][:, k]  # wxyz in file
    root = bodies.index("pelvis")
    holo = dict(
        fps=np.array([fps]), joint_names=np.array(joints), body_names=np.array(bodies),
        joint_pos=np.concatenate([bp[:, root], bq[:, root], d["agent_joint_pos"][:, k]], axis=1),
        joint_vel=np.concatenate([d["agent_body_lin_vel_w"][:, k, root], d["agent_body_ang_vel_w"][:, k, root],
                                  d["agent_joint_vel"][:, k]], axis=1),
        body_pos_w=bp, body_quat_w=bq, body_lin_vel_w=d["agent_body_lin_vel_w"][:, k],
        body_ang_vel_w=d["agent_body_ang_vel_w"][:, k],
    )
    cmd = SPEC["cmd_body_names"]
    sonic = dict(dof_pos=d["agent_joint_pos"][:, k][:, perm_s], dof_vel=d["agent_joint_vel"][:, k][:, perm_s],
                 root_pos_w=bp[:, root], root_quat_w=bq[:, root], body_pos_w=bp[:, [bodies.index(b) for b in cmd]],
                 fps=np.array(fps))
    prov = dict(source="paired_runtime_reference", file=str(path), sha256=sha(path), agent=k)
    return holo, sonic, prov


def from_task(task, k):
    """Agent k of the task's reference, loaded with the paper env's loader (system1/sonic/task_refs.py)."""
    import torch

    sys.path.insert(0, str(ROOT))
    from system1.sonic.task_refs import load_task_reference

    ref, bodies, dofs, meta = load_task_reference(task)
    smp = ref.sample(torch.arange(ref.num_frames))
    g = lambda key: smp[key][:, k].double().numpy()  # noqa: E731
    bp, bq = g("agent_body_pos_w"), xyzw_to_wxyz(g("agent_body_quat_w"))
    root = bodies.index("pelvis")
    holo = dict(
        fps=np.array([int(ref.fps)]), joint_names=np.array(dofs), body_names=np.array(bodies),
        joint_pos=np.concatenate([bp[:, root], bq[:, root], g("agent_joint_pos")], axis=1),
        joint_vel=np.concatenate([g("agent_body_lin_vel_w")[:, root], g("agent_body_ang_vel_w")[:, root], g("agent_joint_vel")], axis=1),
        body_pos_w=bp, body_quat_w=bq, body_lin_vel_w=g("agent_body_lin_vel_w"), body_ang_vel_w=g("agent_body_ang_vel_w"),
    )
    s_names = SPEC["joint_names"]
    perm = [dofs.index(n) for n in s_names]
    cmd = SPEC["cmd_body_names"]
    sonic = dict(dof_pos=g("agent_joint_pos")[:, perm], dof_vel=g("agent_joint_vel")[:, perm], root_pos_w=bp[:, root],
                 root_quat_w=bq[:, root], body_pos_w=bp[:, [bodies.index(b) for b in cmd]], fps=np.array(int(ref.fps)))
    prov = dict(source="task_reference_loader", agent=k, **meta)
    return holo, sonic, prov


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--sonic-dump", type=Path)
    g.add_argument("--paired", type=Path)
    g.add_argument("--task")
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--agent", type=int, default=0)
    p.add_argument("--out-prefix", type=Path, required=True)
    a = p.parse_args()
    if a.sonic_dump:
        holo, sonic, prov = from_sonic_dump(a.sonic_dump, a.index)
    elif a.task:
        holo, sonic, prov = from_task(a.task, a.agent)
    else:
        holo, sonic, prov = from_paired(a.paired, a.agent)
    a.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    np.savez(f"{a.out_prefix}.holosoma.npz", **holo)
    np.savez(f"{a.out_prefix}.sonic.npz", **sonic, provenance=np.array(json.dumps(prov)))
    print(json.dumps(prov, indent=1))


if __name__ == "__main__":
    main()
