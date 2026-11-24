"""T14 step 7 (Kevin 2026-11-24 part 3): per-frame person / object quantities of an OMOMO sequence (original release,
train / test_diffusion_manip_seq_joints24.p; captured_objects meshes; 30 Hz), used by the class screen.

* person joints: SMPL-X body joints from the stored parameters (sequence gender model, 16 betas, root_orient, pose_body,
  trans), joints only (smplx.lbs.batch_rigid_transform on the shaped rest joints; no skinning);
* heading: horizontal part of (left hip - right hip) x up (SMPL-X joints 1, 2), normalised;
* object: captured mesh (<name>_cleaned_simplified.obj) placed by obj_scale, obj_rot, obj_trans; bottom = lowest convex-hull
  vertex z; position = mesh centroid (trimesh centre of mass of the hull) in the world; frames with an invalid stored pose
  (scale <= 0 or non-finite) are flagged and filled from the nearest valid frame.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

OMOMO = Path("/home/kevin/holosoma/data")
SMPLX_DIR = Path("/home/kevin/CS6758Project/data/smplx_models")
FPS = 30.0
_MODELS: dict = {}
_HULLS: dict = {}


def smplx_joints(v):
    """[T, 22, 3] SMPL-X body joints (pelvis .. right wrist) in the OMOMO world."""
    import smplx
    import torch
    from smplx.lbs import batch_rigid_transform, batch_rodrigues, vertices2joints
    g = str(v["gender"])
    if g not in _MODELS:
        _MODELS[g] = smplx.create(str(SMPLX_DIR), model_type="smplx", gender=g, num_betas=16, use_pca=False, flat_hand_mean=True, batch_size=1)
    m = _MODELS[g]
    with torch.no_grad():
        betas = torch.tensor(v["betas"], dtype=torch.float32)
        shaped = m.v_template[None] + torch.einsum("bl,mkl->bmk", betas, m.shapedirs[:, :, :16])
        J = vertices2joints(m.J_regressor, shaped)[0, :22]  # rest joints of the shaped body
        T = len(v["trans"])
        aa = torch.cat([torch.tensor(v["root_orient"], dtype=torch.float32), torch.tensor(v["pose_body"], dtype=torch.float32)], 1).reshape(T * 22, 3)
        R = batch_rodrigues(aa).reshape(T, 22, 3, 3)
        posed, _ = batch_rigid_transform(R, J[None].expand(T, 22, 3).contiguous(), m.parents[:22])
        out = posed + torch.tensor(v["trans"], dtype=torch.float32)[:, None]
    return out.numpy().astype(np.float64)


def hull(name):
    if name not in _HULLS:
        import trimesh
        m = trimesh.load(OMOMO / f"captured_objects/{name}_cleaned_simplified.obj", force="mesh", process=True)
        h = m.convex_hull
        _HULLS[name] = dict(v=np.asarray(h.vertices, float), c=np.asarray(h.center_mass, float))
    return _HULLS[name]


def object_track(v, name):
    s, R, t = np.asarray(v["obj_scale"], float), np.asarray(v["obj_rot"], float), np.asarray(v["obj_trans"], float)[:, :, 0]
    ok = np.isfinite(s) & (s > 0) & np.isfinite(R).all(axis=(1, 2)) & np.isfinite(t).all(axis=1)
    if not ok.all():
        good = np.flatnonzero(ok)
        near = good[np.abs(np.arange(len(s))[:, None] - good[None]).argmin(1)]
        s, R, t = s[near], R[near], t[near]
    h = hull(name)
    Vw = np.einsum("tij,vj->tvi", R * s[:, None, None], h["v"]) + t[:, None]
    return dict(bottom=Vw[..., 2].min(1), centre=np.einsum("tij,j->ti", R * s[:, None, None], h["c"]) + t, R=R, s=s, t=t, hull_world=Vw,
                invalid_frames=int((~ok).sum()))


def heading(J):
    left = J[:, 1] - J[:, 2]
    f = np.cross(left, np.array([0.0, 0.0, 1.0]))
    f[:, 2] = 0
    return f / np.linalg.norm(f, axis=1, keepdims=True)


def load_sequences(objects):
    """{seq_name: dict(only the needed keys, split)} for the given object names, both splits."""
    import joblib
    keys = ("seq_name", "betas", "gender", "trans", "root_orient", "pose_body", "obj_scale", "obj_rot", "obj_trans")
    out = {}
    for split in ("train", "test"):
        d = joblib.load(OMOMO / f"{split}_diffusion_manip_seq_joints24.p")
        for v in d.values():
            if v["seq_name"].split("_")[1] in objects:
                out[v["seq_name"]] = {k: v[k] for k in keys} | dict(split=split)
        del d
    return out
