#!/usr/bin/env python3
"""T07 step 1: per-frame human finger prior for the finger reference (Kevin 2026-11-13, task book 2e1cdb0):
    loss += sum_t w_t |q_f(t) - q_f^human(t)|^2,   w_t = w_max * w_attach * r_t   (w_max relative to the attach weight)

Sources (CORE4D, the raw SMPL-X outputs of the two persons, person k = agent k; 15 Hz; joints and object poses in the
same world frame):
  * bucket: CORE4D_Real/human_object_motions/20231020/071 (150 frames), object bucket003_m.obj;
  * table:  CORE4D_Real/human_object_motions/20231030/001 (207 frames), object desk001_m.obj.
  Push / Pull (OMOMO) have no finger motion: r_t = 0 (not produced here).

Anatomical correspondence (same-named joints of the same finger), from joint positions in each hand's own palm frame
(A1 construction: forward = wrist -> middle MCP, across = index - pinky MCP, normal = forward x across):
  index / middle / ring / pinky:  MCP_FE = atan2(p.N, p.F), MCP_AA = atan2(p.A, |p.(F,N)|) of the proximal phalanx p,
                                  PIP = angle(proximal, middle phalanx), DIP = angle(middle, distal phalanx),
                                  pinky CMC: no human counterpart -> 0;
  thumb: CMC_FE / CMC_AA = the two angles of the metacarpal, MCP_FE / MCP_AA = the two angles of the proximal phalanx
         in the metacarpal's frame, IP = angle(proximal, distal phalanx).
  The same functions are evaluated on the Sharpa forward kinematics; per finger the linear relation
  angles = G q + b around the open pose (finite differences) is inverted (least squares) and the result is clipped
  to the Sharpa joint limits. G and b are written to the output (the mapping table of the README).
Reliability r_t per hand and frame: d_t = distance of the closest of {palm point, 5 fingertips} to the object surface
(negative inside counts as 0); r_t = clip((0.05 - d_t) / 0.03, 0, 1); frames where a mapped finger joint changes by
more than 0.1 rad from the previous runtime frame get r_t / 2. Runtime frame f <- source time f / 50 s (linear
interpolation of the mapped angles and of d_t), as in the existing contact diagnostics.

    hssim_s1/bin/python -B system1/hand/human_prior.py --case bucket --out logs_system1/t07/human_prior/bucket.npz
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from system1.hand.human_fingers import SHARPA_KP, SMPLX_BASE, SMPLX_FINGER_OFFSET, SMPLX_TIP, SMPLX_WRIST  # noqa: E402
from system1.hand.sharpa import FINGERS, SIDES, SharpaHand, palm_basis_from_landmarks  # noqa: E402

CORE4D = Path("/home/kevin/datasets/CORE4D-V1-full/CORE4D_Real")
SOURCES = dict(bucket=dict(seq=CORE4D / "human_object_motions/20231020/071", mesh=CORE4D / "object_models/bucket/bucket003_m.obj"),
               table=dict(seq=CORE4D / "human_object_motions/20231030/001", mesh=CORE4D / "object_models/desk/desk001_m.obj"))
RUNTIME = dict(bucket=ROOT / "src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/core4d_bucket003_20231020_071_a1/core4d_pair_runtime_fps50.npz",
               table=ROOT / "src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/core4d_smalltable5kg_A/core4d_pair_runtime_fps50.npz")
FPS_SRC = 15.0
D_FULL, D_ZERO, JITTER = 0.02, 0.05, 0.1


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _ang(u, v):
    c = np.clip((u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1)), -1, 1)
    return np.arccos(c)


def _dir_angles(d, F, A, N):
    """(flex toward N, abduction toward A) of direction d in the frame (F, A, N)."""
    f, a, n = d @ F, d @ A, d @ N
    return np.arctan2(n, f), np.arctan2(a, np.sqrt(f * f + n * n))


def anatomical(kp: dict, B: np.ndarray) -> dict:
    """kp[finger] (4, 3) keypoints, B palm basis columns [F, A, N] -> {finger: angle vector} in the order of FEATURES."""
    F, A, N = B[:, 0], B[:, 1], B[:, 2]
    out = {}
    for f in ("index", "middle", "ring", "pinky"):
        p = kp[f]
        d1, d2, d3 = p[1] - p[0], p[2] - p[1], p[3] - p[2]
        fl, ab = _dir_angles(d1 / np.linalg.norm(d1), F, A, N)
        out[f] = np.array([fl, ab, _ang(d1, d2), _ang(d2, d3)])
    p = kp["thumb"]
    m, d1, d2 = p[1] - p[0], p[2] - p[1], p[3] - p[2]
    m = m / np.linalg.norm(m)
    c1, c2 = _dir_angles(m, F, A, N)
    # metacarpal frame: Fm = m, Nm = component of N orthogonal to m, Am = Nm x Fm
    Nm = N - (N @ m) * m
    Nm = Nm / np.linalg.norm(Nm)
    Am = np.cross(Nm, m)
    m1, m2 = _dir_angles(d1 / np.linalg.norm(d1), m, Am, Nm)
    out["thumb"] = np.array([c1, c2, m1, m2, _ang(d1, d2)])
    return out


FEATURES = dict(index=("MCP_FE", "MCP_AA", "PIP", "DIP"), middle=("MCP_FE", "MCP_AA", "PIP", "DIP"),
                ring=("MCP_FE", "MCP_AA", "PIP", "DIP"), pinky=("MCP_FE", "MCP_AA", "PIP", "DIP"),
                thumb=("CMC_FE", "CMC_AA", "MCP_FE", "MCP_AA", "IP"))


def sharpa_keypoints(hand: SharpaHand, q: np.ndarray):
    T = hand.fk(torch.as_tensor(q, dtype=torch.float64))
    s = hand.side
    kp = {f: np.stack([T[f"{s}_{n}"][:3, 3].numpy() for n in SHARPA_KP[f]]) for f in FINGERS}
    return kp


class Calibration:
    """Per finger: angles = G q_f + b around the open pose (finite differences on the Sharpa FK)."""

    def __init__(self, hand: SharpaHand, eps=1e-4):
        self.hand = hand
        self.B = hand.palm_basis()
        s = hand.side
        self.names = hand.dof_names
        q0 = np.zeros(22)
        a0 = anatomical(sharpa_keypoints(hand, q0), self.B)
        self.cal = {}
        for f in FINGERS:
            idx = [self.names.index(f"{s}_{f}_{j}") for j in FEATURES[f]]
            G = np.zeros((len(idx), len(idx)))
            for k, i in enumerate(idx):
                q = q0.copy()
                q[i] += eps
                G[:, k] = (anatomical(sharpa_keypoints(hand, q), self.B)[f] - a0[f]) / eps
            self.cal[f] = dict(idx=idx, G=G, b=a0[f], cond=float(np.linalg.cond(G)))

    def to_sharpa(self, ang: dict) -> np.ndarray:
        q = np.zeros(22)
        for f, c in self.cal.items():
            q[c["idx"]] = np.linalg.lstsq(c["G"], ang[f] - c["b"], rcond=None)[0]
        lo, hi = self.hand.lower.numpy(), self.hand.upper.numpy()
        return np.clip(q, lo, hi)

    def table(self) -> dict:
        return {f: dict(joints=[self.names[i] for i in c["idx"]], features=list(FEATURES[f]), G=c["G"].round(4).tolist(),
                        b=c["b"].round(4).tolist(), cond=round(c["cond"], 3)) for f, c in self.cal.items()}


def human_keypoints(J, side):
    b = SMPLX_BASE[side]
    kp = {f: np.concatenate([J[b + SMPLX_FINGER_OFFSET[f]: b + SMPLX_FINGER_OFFSET[f] + 3], J[SMPLX_TIP[side][f]][None]]) for f in FINGERS}
    return J[SMPLX_WRIST[side]], kp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=tuple(SOURCES), required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    src = SOURCES[a.case]
    import trimesh
    mesh = trimesh.load(src["mesh"], force="mesh")
    pq = trimesh.proximity.ProximityQuery(mesh)
    objp = np.load(src["seq"] / "smooth_objposes.npy", allow_pickle=True)  # (T_src, 4, 4)
    persons = [np.load(src["seq"] / f"person{k}_poses.npz", allow_pickle=True)["arr_0"].item()["joints"] for k in (1, 2)]
    T_src = min(len(objp), *(len(p) for p in persons))
    ref = np.load(RUNTIME[a.case], allow_pickle=True)
    T = ref["agent_joint_pos"].shape[0]
    fps = float(np.asarray(ref["fps"]).reshape(-1)[0])
    t_src = np.arange(T_src) / FPS_SRC
    t_run = np.clip(np.arange(T) / fps, 0, t_src[-1])
    q_h = np.zeros((T, 2, 44))
    r = np.zeros((T, 2, 2))
    dist = np.zeros((T, 2, 2))
    cal_tables, diag = {}, {}
    for hi, side in enumerate(SIDES):
        hand = SharpaHand(side)
        cal = Calibration(hand)
        cal_tables[side] = cal.table()
        for ag in range(2):
            J = np.asarray(persons[ag][:T_src], float)
            qs = np.zeros((T_src, 22))
            ds = np.zeros(T_src)
            for t in range(T_src):
                w, kp = human_keypoints(J[t], side)
                B = palm_basis_from_landmarks(w, kp["index"][0], kp["middle"][0], kp["pinky"][0], side)
                qs[t] = cal.to_sharpa(anatomical(kp, B))
                palm = np.mean([w, kp["index"][0], kp["middle"][0], kp["pinky"][0]], axis=0)
                pts = np.vstack([palm[None], np.stack([kp[f][3] for f in FINGERS])])
                Ti = np.linalg.inv(objp[t])
                local = (Ti[:3, :3] @ pts.T).T + Ti[:3, 3]
                sd = pq.signed_distance(local)  # trimesh: positive inside
                ds[t] = max(0.0, float((-sd).min()))
            for j in range(22):
                q_h[:, ag, hi * 22 + j] = np.interp(t_run, t_src, qs[:, j])
            dist[:, ag, hi] = np.interp(t_run, t_src, ds)
            rr = np.clip((D_ZERO - dist[:, ag, hi]) / (D_ZERO - D_FULL), 0, 1)
            jump = np.r_[False, np.abs(np.diff(q_h[:, ag, hi * 22:(hi + 1) * 22], axis=0)).max(1) > JITTER]
            rr[jump] *= 0.5
            r[:, ag, hi] = rr
            diag[f"agent{ag + 1}_{side}"] = dict(frames_r_gt_0=int((rr > 0).sum()), frames_r_1=int((rr >= 1).sum()), r_mean=float(rr.mean()),
                                                 jitter_frames=int(jump.sum()), dist_min_m=float(dist[:, ag, hi].min()),
                                                 dist_p50_m=float(np.median(dist[:, ag, hi])),
                                                 q_range={n: [float(qs[:, k].min()), float(qs[:, k].max())] for k, n in enumerate(hand.dof_names)})
            print(a.case, ag + 1, side, json.dumps({k: v for k, v in diag[f"agent{ag + 1}_{side}"].items() if k != "q_range"}), flush=True)
    names = [n for s in SIDES for n in SharpaHand(s).dof_names]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, q_human=q_h, r=r, dist_m=dist, finger_joint_names=np.array(names), fps=fps, case=np.array(a.case))
    meta = dict(case=a.case, output=str(a.out), output_sha256=sha256(a.out), source_dir=str(src["seq"]), mesh=str(src["mesh"]), mesh_sha256=sha256(src["mesh"]),
                person_files_sha256={k: sha256(src["seq"] / f"person{k}_poses.npz") for k in (1, 2)}, objposes_sha256=sha256(src["seq"] / "smooth_objposes.npy"),
                runtime_reference=str(RUNTIME[a.case].relative_to(ROOT)), runtime_reference_sha256=sha256(RUNTIME[a.case]),
                source_frames=T_src, source_fps=FPS_SRC, runtime_frames=T, reliability=dict(full_m=D_FULL, zero_m=D_ZERO, jitter_rad=JITTER),
                person_to_agent="person k = agent k", calibration=cal_tables, hands=diag)
    a.out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
