#!/usr/bin/env python3
"""T14 (Kevin 2026-11-24): contact-position error as a 3-D distance; replaces the surface-only version of contact_gap.py
(which compared the two points' projections onto the surface and could not see a hand hovering above it).

  * frames: every frame in which the person's hand touches the object (any SMPL-X hand vertex within 1.5 cm of the captured
    mesh); CORE4D 15 Hz frame t <-> reference frame 2t, robot r <-> person r + 1; Push / Pull 30 Hz, robot 1 inherits the
    source (Pull mirrored across the table's local XY plane, hands swapped);
  * target = the person's palm point -> nearest point on the captured object mesh (object frame), moved to the level's object
    by the level's transform (CORE4D x s, s = s_bar k; Push / Pull: the OMOMO capture table the hands touch, moved to the
    training table by the exact similarity of omomo_table.py (round 6; earlier recomputations used the InterMimic table), then
    the role-isometry translation +-c(m), Pull robot 1 mirrored);
    n = the mesh face normal at that point, oriented toward the person's palm point (outside the object);
  * robot = the robot palm point (palm point of the simulator hand hull, simulator FK) in the reference's object frame;
  * error vector e = robot - target: distance |e|; normal component e.n (signed: + = outside / hovering, - = inside the object
    surface); tangential component |e - (e.n) n|.
Per robot and hand over the contact frames: mean and 95th percentile of the distance, of the tangential component and of
|normal|, and the mean signed normal component (cm).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "system1/t14"))
sys.path.insert(0, str(ROOT / "system1/t12v2"))
sys.path.insert(0, str(ROOT / "src/holosoma_retargeting"))
import contact_gap as cg  # noqa: E402

S_BAR = cg.S_BAR


def _targets(mesh, palm_local):
    """nearest mesh points and face normals oriented toward the palm point; palm_local [..., 3]."""
    P = palm_local.reshape(-1, 3)
    cp, _, tri = trimesh.proximity.closest_point(mesh, P)
    n = mesh.face_normals[tri]
    s = np.sign(np.einsum("ij,ij->i", P - cp, n))
    n = n * np.where(s == 0, 1.0, s)[:, None]
    return cp.reshape(palm_local.shape), n.reshape(palm_local.shape)


def human_core4d(task):
    import table_shared as tsh
    cfg = tsh.TASKS[task]
    tsh.SEQ_RAW, tsh.OBJ_KEY = tsh.CORE4D / "CORE4D_Real/human_object_motions" / cfg["raw"], cfg["obj_key"]
    _, contact, palm_local = tsh.source_palms_and_contacts(np.eye(3))
    mesh_path = next((tsh.CORE4D / "CORE4D_Real/object_models").rglob(f"{cfg['obj_key'].split('_', 1)[1]}_m.obj"))
    mesh = trimesh.load(mesh_path, force="mesh")
    T = len(contact[(1, "left")])
    flags, target, normal = np.zeros((T, 2, 2), bool), np.zeros((T, 2, 2, 3)), np.zeros((T, 2, 2, 3))
    for p in (1, 2):
        for h, side in enumerate(("left", "right")):
            flags[:, p - 1, h] = contact[(p, side)]
            target[:, p - 1, h], normal[:, p - 1, h] = _targets(mesh, palm_local[(p, side)])
    return dict(flags=flags, target=target, normal=normal, mesh=str(mesh_path))


def human_omomo(task):
    import dual_stretch as ds
    import single_shared as ss
    from holosoma_retargeting.src.utils import load_intermimic_data
    seq = ds.SEQ[task]
    palms, _ = ss.omomo_palms(seq)
    _, obj = load_intermimic_data(str(ss.OMOMO_NEW / f"{seq}.pt"))
    R = cg.cs.pose_mat(obj[:, 4:7], obj[:, [1, 2, 3, 0]])
    local = np.einsum("tji,thj->thi", R[:, :3, :3], palms - obj[:, None, 4:7])
    import omomo_table as ot  # the persons touch the OMOMO capture (round 6 check): target on it, moved to the training table
    cpt, normal = _targets(ot.relation(seq)["mesh"], local)
    target = ot.to_training(seq, cpt)
    h_sd, _ = ds.human_hand_sd(seq)
    return dict(flags=h_sd >= cg.CONTACT, target=target, normal=normal, mesh="OMOMO capture x obj_scale (omomo_table.py), moved by the similarity k p + o")


def _stats(e, n, f):
    e, n = e[f], n[f]
    if not len(e):
        return dict(frames=0)
    nc = np.einsum("ij,ij->i", e, n)
    tg = np.linalg.norm(e - nc[:, None] * n, axis=1)
    d = np.linalg.norm(e, axis=1)
    r = lambda x: round(100 * float(x), 2)
    return dict(frames=int(len(e)), dist_mean_cm=r(d.mean()), dist_p95_cm=r(np.percentile(d, 95)), tangential_mean_cm=r(tg.mean()),
                tangential_p95_cm=r(np.percentile(tg, 95)), normal_abs_mean_cm=r(np.abs(nc).mean()), normal_abs_p95_cm=r(np.percentile(np.abs(nc), 95)),
                normal_signed_mean_cm=r(nc.mean()))


def core4d(task, robot, offs, human, pair, obj_q, k):
    t15 = np.arange(min(len(human["flags"]), (len(pair) + 1) // 2))
    P = cg.robot_palms_local(robot, pair[2 * t15], obj_q[2 * t15], offs)
    s = S_BAR[task] * k
    e = P - s * human["target"][t15]
    nrm = human["normal"][t15]
    return {f"r{r}_{side}": _stats(e[:, r, h], nrm[:, r, h], human["flags"][t15, r, h]) for r in (0, 1) for h, side in enumerate(("left", "right"))}


def omomo(task, robot, offs, human, pair, obj_q, m):
    import dual_stretch as ds
    n = min(len(human["flags"]), len(pair))
    c = ds.offset_c(m)
    ax = np.eye(3)[ds.TABLE[task][1]]
    tg, nm, fl = np.zeros((n, 2, 2, 3)), np.zeros((n, 2, 2, 3)), np.zeros((n, 2, 2), bool)
    T, N, F = human["target"][:n], human["normal"][:n], human["flags"][:n]
    if task == "push":
        tg[:, 0], tg[:, 1] = T - c * ax, T + c * ax
        nm[:, 0] = nm[:, 1] = N
        fl[:, 0] = fl[:, 1] = F
    else:
        M = np.array([1.0, 1.0, -1.0])
        tg[:, 0], tg[:, 1] = T - c * ax, T[:, ::-1] * M + c * ax
        nm[:, 0], nm[:, 1] = N, N[:, ::-1] * M
        fl[:, 0], fl[:, 1] = F, F[:, ::-1]
    P = cg.robot_palms_local(robot, pair[:n], obj_q[:n], offs)
    e = P - tg
    return {f"r{r}_{side}": _stats(e[:, r, h], nm[:, r, h], fl[:, r, h]) for r in (0, 1) for h, side in enumerate(("left", "right"))}
