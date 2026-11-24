#!/usr/bin/env python3
"""T14 (Kevin 2026-11-24): which table surface do the Push / Pull persons' hands actually touch - the OMOMO capture
(largetable_cleaned_simplified.obj x obj_scale at the OMOMO object pose, height 0.459 m) or the InterMimic table
(largetable.obj at the OMOMO_new object pose, height 0.428 m; used by the retargeting scene)?

Same person in both: OMOMO SMPL-X (flat hands) in the OMOMO world; mapped into the OMOMO_new world by the rigid fit of the
22 body joints (single_shared.omomo_palms; residual < 1e-6 m). Per frame and hand: hand signed distance = largest signed
distance of the SMPL-X hand vertices to the mesh surface (+ = inside), and the palm point's signed distance (mean of the 67
palm-side vertices). Contact frames = frames where either mesh gives hand signed distance >= -1.5 cm (the T14 rule).
Signed distance: nearest point of dense surface samples (1.5 mm) with the sample's face normal (outward) for the sign.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "system1/t14"))
sys.path.insert(0, str(ROOT / "system1/t12v2"))
sys.path.insert(0, str(ROOT / "src/holosoma_retargeting"))


class SD:
    def __init__(self, mesh, spacing=0.0015):
        mesh = mesh.copy()
        mesh.fix_normals()
        n = int(mesh.area / spacing ** 2)
        self.p, fid = trimesh.sample.sample_surface(mesh, n, seed=0)
        self.n = mesh.face_normals[fid]
        self.tree = cKDTree(self.p)

    def __call__(self, x):
        d, i = self.tree.query(x)
        return -np.sign(np.einsum("ij,ij->i", x - self.p[i], self.n[i])) * d  # + inside


def main():
    import joblib
    import smplx
    import torch
    import contact_parts as cp
    import dual_stretch as ds
    import single_shared as ss
    from table_shared import palm_vertex_ids
    from holosoma_retargeting.src.utils import load_intermimic_data
    hv, pv = cp.hand_vertex_ids(), palm_vertex_ids()
    cap = trimesh.load(ss.OMOMO_P / "captured_objects/largetable_cleaned_simplified.obj", force="mesh")
    inter = trimesh.load(ROOT / "src/holosoma_retargeting/holosoma_retargeting/models/largetable/largetable.obj", force="mesh")
    out = {}
    for task in ("push", "pull"):
        seq = ds.SEQ[task]
        v = None
        for split in ("train", "test"):
            d = joblib.load(ss.OMOMO_P / f"{split}_diffusion_manip_seq_joints24.p")
            c = [x for x in d.values() if x["seq_name"] == seq]
            del d
            if c:
                v = c[0]
                break
        T = len(v["trans"])
        m = smplx.create(str(ss.SMPLX_DIR), model_type="smplx", gender=str(v["gender"]), num_betas=16, use_pca=False, flat_hand_mean=True, batch_size=T)
        with torch.no_grad():
            V = m(betas=torch.tensor(np.repeat(v["betas"], T, 0), dtype=torch.float32), global_orient=torch.tensor(v["root_orient"], dtype=torch.float32),
                  body_pose=torch.tensor(v["pose_body"], dtype=torch.float32), transl=torch.tensor(v["trans"], dtype=torch.float32)).vertices.numpy().astype(np.float64)
        _, fit = ss.omomo_palms(seq)
        R, t = np.array(fit["rotation"]), np.array(fit["translation"])
        _, obj = load_intermimic_data(str(ss.OMOMO_NEW / f"{seq}.pt"))
        from contact_stats import pose_mat
        Pi = pose_mat(obj[:, 4:7], obj[:, [1, 2, 3, 0]])
        sc = float(v["obj_scale"][0])
        sd_cap = SD(trimesh.Trimesh(np.asarray(cap.vertices) * sc, cap.faces, process=False))
        sd_int = SD(inter)
        res = {}
        H = {}
        for hi, side in enumerate(("left", "right")):
            hc = np.zeros(T)
            hin = np.zeros(T)
            pc = np.zeros(T)
            pin = np.zeros(T)
            for f in range(T):
                # capture: OMOMO world, object frame = (x - obj_trans) @ obj_rot, mesh x obj_scale
                loc_c = (V[f] - v["obj_trans"][f, :, 0]) @ v["obj_rot"][f]
                # InterMimic: OMOMO_new world (rigid fit), object frame of the InterMimic pose
                Vn = V[f] @ R.T + t
                loc_i = (Vn - Pi[f, :3, 3]) @ Pi[f, :3, :3]
                hc[f] = sd_cap(loc_c[hv[side]]).max()
                hin[f] = sd_int(loc_i[hv[side]]).max()
                pc[f] = sd_cap(loc_c[pv[side]].mean(0)[None])[0]
                pin[f] = sd_int(loc_i[pv[side]].mean(0)[None])[0]
            H[side] = (hc, hin, pc, pin)
            contact = (hc >= -0.015) | (hin >= -0.015)
            q = lambda x: dict(median_cm=round(100 * float(np.median(x[contact])), 2), p10_cm=round(100 * float(np.percentile(x[contact], 10)), 2),
                               p90_cm=round(100 * float(np.percentile(x[contact], 90)), 2), within_1cm_share=round(float((np.abs(x[contact]) <= 0.01).mean()), 3))
            res[side] = dict(contact_frames=int(contact.sum()), contact_frames_capture_rule=int((hc >= -0.015).sum()), contact_frames_intermimic_rule=int((hin >= -0.015).sum()),
                             hand_sd_capture=q(hc), hand_sd_intermimic=q(hin), palm_sd_capture=q(pc), palm_sd_intermimic=q(pin))
        out[task] = dict(sequence=seq, obj_scale=sc, capture_height_m=round(float(np.ptp(np.asarray(cap.vertices)[:, 1]) * sc), 4),
                         intermimic_height_m=round(float(np.ptp(np.asarray(inter.vertices)[:, 1])), 4), hands=res)
        print(task, json.dumps(out[task]), flush=True)
    p = ROOT / "results_system1/T14_retargeting/step6/omomo_table_check.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dict(rules=__doc__, tasks=out), indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
