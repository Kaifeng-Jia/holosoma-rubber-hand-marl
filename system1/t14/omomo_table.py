"""T14 round 6 (Kevin 2026-11-24): the table the Push / Pull persons' hands touch is the OMOMO capture
(largetable_cleaned_simplified.obj x obj_scale), not the InterMimic largetable.obj (check_omomo_table.py: palm 0.7-0.9 cm from
the capture top vs 4.6-4.9 cm from the InterMimic one for Push).

Relation (computed per sequence, checked here): in the InterMimic object frame the capture table sits at a constant offset
(rotation identity, translation t_M from the two pose tracks), and the InterMimic table is an exact similarity of it:
    p_intermimic = k * p_capture + o      (k = 0.9327, o = (0, -0.0322, 0); vertex residual < 1e-8 m, same topology).
The training table at the robot's end is the InterMimic table (its box proxy), so a point on the capture table is moved to the
training table by this similarity ("the contact position scales with the object"); the training table itself is unchanged.
"""
from __future__ import annotations

import functools
import sys
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "system1/t14"))
sys.path.insert(0, str(ROOT / "system1/t12v2"))
sys.path.insert(0, str(ROOT / "src/holosoma_retargeting"))
INTERMIMIC = ROOT / "src/holosoma_retargeting/holosoma_retargeting/models/largetable/largetable.obj"


@functools.lru_cache(maxsize=4)
def relation(seq):
    import joblib
    import single_shared as ss
    from contact_stats import pose_mat
    from holosoma_retargeting.src.utils import load_intermimic_data
    v = None
    for split in ("train", "test"):
        d = joblib.load(ss.OMOMO_P / f"{split}_diffusion_manip_seq_joints24.p")
        c = [x for x in d.values() if x["seq_name"] == seq]
        del d
        if c:
            v = c[0]
            break
    _, fit = ss.omomo_palms(seq)
    K = np.eye(4)
    K[:3, :3], K[:3, 3] = np.array(fit["rotation"]), np.array(fit["translation"])
    _, obj = load_intermimic_data(str(ss.OMOMO_NEW / f"{seq}.pt"))
    Pi = pose_mat(obj[:, 4:7], obj[:, [1, 2, 3, 0]])
    Ms = []
    for f in range(len(obj)):
        Tc = np.eye(4)
        Tc[:3, :3], Tc[:3, 3] = v["obj_rot"][f], v["obj_trans"][f, :, 0]
        Ms.append(np.linalg.inv(Pi[f]) @ K @ Tc)
    Ms = np.array(Ms)
    M = Ms[0]
    dev_t = float(np.abs(Ms[:, :3, 3] - M[:3, 3]).max())
    dev_r = float(np.degrees(max(np.arccos(np.clip((np.trace(m[:3, :3].T @ M[:3, :3]) - 1) / 2, -1, 1)) for m in Ms)))
    sc = float(v["obj_scale"][0])
    cap = trimesh.load(ss.OMOMO_P / "captured_objects/largetable_cleaned_simplified.obj", force="mesh")
    A = np.asarray(cap.vertices) * sc @ M[:3, :3].T + M[:3, 3]
    B = np.asarray(trimesh.load(INTERMIMIC, force="mesh").vertices)
    Am, Bm = A.mean(0), B.mean(0)
    k = float(((A - Am) * (B - Bm)).sum() / ((A - Am) ** 2).sum())
    o = Bm - k * Am
    res = float(np.linalg.norm(k * A + o - B, axis=1).max())
    mesh = trimesh.Trimesh(A, cap.faces, process=False)
    return dict(k=k, o=o, mesh=mesh, M=M, obj_scale=sc, frame_dev_translation_m=dev_t, frame_dev_rotation_deg=dev_r, vertex_residual_m=res)


def to_training(seq, p):
    r = relation(seq)
    return r["k"] * np.asarray(p, float) + r["o"]


def from_training(seq, p):
    r = relation(seq)
    return (np.asarray(p, float) - r["o"]) / r["k"]


def record(seq):
    r = relation(seq)
    return dict(k=round(r["k"], 6), o=np.round(r["o"], 6).tolist(), capture_in_intermimic_translation=np.round(r["M"][:3, 3], 5).tolist(),
                pose_relation_max_dev_translation_m=r["frame_dev_translation_m"], pose_relation_max_dev_rotation_deg=round(r["frame_dev_rotation_deg"], 4),
                vertex_residual_max_m=r["vertex_residual_m"], obj_scale=r["obj_scale"])
