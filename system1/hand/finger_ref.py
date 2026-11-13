#!/usr/bin/env python3
"""T06 step 2: retarget-free finger reference for the Bin task (CPU only).

Body: the existing D Bin reference (core4d_bucket003_20231020_071_a1/core4d_pair_runtime_fps50.npz) is kept, except
the six wrist joints, which are re-solved for the Sharpa palm frame with the same target as the A1 wrist-only solver:
the A1 output matches the rubber-hand palm basis B_rub to the demonstrated palm (error < 1e-11 deg), so the target palm
orientation is P(t) = R_wrist_yaw_ref(t) B_rub, and the Sharpa wrist must satisfy R_wrist_yaw_new(t) B_sharpa = P(t).
With the elbow pose of the reference fixed, R_elbow^T R_wrist_yaw_new = Rx(roll) Ry(pitch) Rz(yaw) (G1 wrist joints:
x, y, z axes, rpy = 0 origins) is solved in closed form (intrinsic XYZ Euler), branch closest to the previous frame,
checked against the joint limits.

Fingers, per hand and frame:
  * contact frames (same definition as the existing pipeline, source_contact_diagnostics.npz: unscaled human palm point
    to bucket-surface distance, 15 Hz -> 50 Hz) d < outer band 0.04 m; non-contact frames: open pose;
  * transitions: 0.2 s (10 frames) smoothstep ramp before each contact interval and after it,
    q(t) = q_open + s(t) (q*(t) - q_open) with q* the optimised pose (held from the nearest contact frame in a ramp);
  * objective on contact frames (all frames of a hand jointly):
      attach  w_a * sum_groups relu(min_points sd)^2         groups = palm (C_MC) and each finger's links
      pen     w_p * sum_points relu(-sd)^2                   sd < 0 inside the object
      smooth  w_s * sum_t |q_t - q_{t-1}|^2
      open    w_o * |q - q_open|^2
      human   w_h * |q - q_human|^2                          (w_h = 0 by default: no human finger data used)
    joint limits are hard (projection after every step);
  * locked mode: the open pose in every frame (wrist as above).
Diagnostics per hand and frame: max penetration (all collision samples), palm-only penetration, number of fingers and
palm within the contact tolerance 5 mm, joint limits. Gate (T06): max penetration <= 1 cm; in contact frames palm and
at least two fingers in contact.

    hssim_s1/bin/python -B system1/hand/finger_ref.py --mode track --out logs_system1/t06/finger_ref/track
    hssim_s1/bin/python -B system1/hand/finger_ref.py --mode locked --out logs_system1/t06/finger_ref/locked
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from system1.hand.sdf import GridSDF  # noqa: E402
from system1.hand.sharpa import FINGERS, SIDES, SharpaHand, mount_transform, rpy_matrix, rubber_palm_basis  # noqa: E402

BIN_DIR = ROOT / "src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/core4d_bucket003_20231020_071_a1"
REFERENCE = BIN_DIR / "core4d_pair_runtime_fps50.npz"
CONTACT = BIN_DIR / "source_contact_diagnostics.npz"
MESH = BIN_DIR / "bucket003_m.obj"
G1_URDF = ROOT / "src/holosoma/holosoma/data/robots/g1/main_mesh_collision_rubberhand.urdf"
CONTACT_OUTER_M = 0.04          # existing soft band outer edge (source_contact_diagnostics.json)
RAMP_S = 0.2                    # T06: ~0.2 s transitions
CONTACT_TOL_M = 0.005           # a palm / finger group "touches" if its closest sample is within 5 mm
GATE_PEN_M = 0.01
DEFAULT_W = dict(attach=1e4, pen=1e6, smooth=10.0, open=0.01, human=0.0)


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def quat_wxyz_to_matrix(q):
    q = np.asarray(q, float)
    return Rotation.from_quat(np.concatenate([q[..., 1:], q[..., :1]], -1)).as_matrix()


def g1_wrist_chain(side: str):
    r = ET.parse(G1_URDF).getroot()
    out = []
    for n in ("roll", "pitch", "yaw"):
        j = r.find(f"joint[@name='{side}_wrist_{n}_joint']")
        o, lim = j.find("origin"), j.find("limit")
        rpy = [float(x) for x in o.get("rpy").split()]
        if any(abs(v) > 1e-12 for v in rpy):
            raise ValueError("closed-form wrist solve assumes rpy = 0 joint origins")
        out.append(dict(xyz=np.array([float(x) for x in o.get("xyz").split()]), axis=np.array([float(x) for x in j.find("axis").get("xyz").split()]),
                        lower=float(lim.get("lower")), upper=float(lim.get("upper"))))
    if [tuple(c["axis"]) for c in out] != [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
        raise ValueError("unexpected G1 wrist axes")
    return out


def wrist_fk(R_elbow, p_elbow, ang, chain):
    """wrist_yaw_link pose from the elbow pose and (roll, pitch, yaw)."""
    R, p = R_elbow.copy(), p_elbow.copy()
    for k, c in enumerate(chain):
        p = p + R @ c["xyz"]
        R = R @ Rotation.from_rotvec(c["axis"] * ang[k]).as_matrix()
    return R, p


def solve_wrist(ref, agent: int, side: str, B_sh_wy: np.ndarray):
    """Closed-form Sharpa wrist angles for one agent / side; returns angles (T, 3) and diagnostics."""
    names = [str(n) for n in ref["body_names"]]
    jn = [str(n) for n in ref["joint_names"]]
    ie, iw = names.index(f"{side}_elbow_link"), names.index(f"{side}_wrist_yaw_link")
    jw = [jn.index(f"{side}_wrist_{n}_joint") for n in ("roll", "pitch", "yaw")]
    chain = g1_wrist_chain(side)
    Rel = quat_wxyz_to_matrix(ref["agent_body_quat_w"][:, agent, ie])
    pel = np.asarray(ref["agent_body_pos_w"][:, agent, ie], float)
    Rwy_ref = quat_wxyz_to_matrix(ref["agent_body_quat_w"][:, agent, iw])
    pwy_ref = np.asarray(ref["agent_body_pos_w"][:, agent, iw], float)
    ang_ref = np.asarray(ref["agent_joint_pos"][:, agent, jw], float)
    B_rub = rubber_palm_basis(side)
    T = len(Rel)
    fk_err_rot, fk_err_pos = [], []
    for t in range(T):  # the reference is self-consistent: FK(elbow, wrist angles) = wrist_yaw pose
        R, p = wrist_fk(Rel[t], pel[t], ang_ref[t], chain)
        fk_err_rot.append(np.degrees(np.linalg.norm(Rotation.from_matrix(R.T @ Rwy_ref[t]).as_rotvec())))
        fk_err_pos.append(np.linalg.norm(p - pwy_ref[t]))
    ang = np.empty((T, 3))
    prev = ang_ref[0]
    lo = np.array([c["lower"] for c in chain]); hi = np.array([c["upper"] for c in chain])
    for t in range(T):
        P = Rwy_ref[t] @ B_rub                      # demonstrated palm (A1 target) in world
        Rnew = P @ B_sh_wy.T                        # wrist_yaw orientation that puts the Sharpa palm on it
        local = Rel[t].T @ Rnew
        a = Rotation.from_matrix(local).as_euler("XYZ")
        cands = [a, np.array([a[0] + math.pi, math.pi - a[1], a[2] + math.pi])]
        best = None
        for c in cands:
            c = (c + math.pi) % (2 * math.pi) - math.pi
            c = c + 2 * math.pi * np.round((prev - c) / (2 * math.pi))
            key = (bool(np.all((c >= lo) & (c <= hi))), -np.linalg.norm(c - prev))
            if best is None or key > best[0]:
                best = (key, c)
        ang[t] = best[1]
        prev = ang[t]
    res, pos_shift = [], []
    for t in range(T):
        R, p = wrist_fk(Rel[t], pel[t], ang[t], chain)
        res.append(np.degrees(np.linalg.norm(Rotation.from_matrix((R @ B_sh_wy).T @ (Rwy_ref[t] @ B_rub)).as_rotvec())))
        pos_shift.append(np.linalg.norm(p - pwy_ref[t]))
    within = (ang >= lo) & (ang <= hi)
    diag = dict(fk_consistency_rot_deg_max=float(np.max(fk_err_rot)), fk_consistency_pos_m_max=float(np.max(fk_err_pos)),
                palm_orientation_error_deg_max=float(np.max(res)), wrist_change_deg_max=np.degrees(np.abs(ang - ang_ref).max(0)).tolist(),
                wrist_change_deg_mean=np.degrees(np.abs(ang - ang_ref).mean(0)).tolist(), within_limits_frames=int(within.all(1).sum()),
                frames=T, wrist_yaw_position_shift_m_max=float(np.max(pos_shift)))
    return ang, diag, (Rel, pel, chain)


def ramp_weights(contact: np.ndarray, fps: float) -> np.ndarray:
    """s(t) in [0, 1]: 1 on contact frames, smoothstep ramps of RAMP_S before each interval and after it."""
    n = max(1, int(round(RAMP_S * fps)))
    T = len(contact)
    s = contact.astype(float).copy()
    idx = np.flatnonzero(contact)
    if len(idx) == 0:
        return s
    for t in range(T):
        if contact[t]:
            continue
        d = np.min(np.abs(idx - t))
        if d <= n:
            x = 1.0 - d / (n + 1)
            s[t] = x * x * (3 - 2 * x)
    return s


def finger_groups(hand: SharpaHand, links: list[str]) -> list[tuple[str, np.ndarray]]:
    s = hand.side
    groups = [("palm", np.array([i for i, l in enumerate(links) if l == f"{s}_hand_C_MC"]))]
    for f in FINGERS:
        groups.append((f, np.array([i for i, l in enumerate(links) if l.startswith(f"{s}_{f}_")])))
    return groups


def optimise_hand(hand: SharpaHand, base_obj: torch.Tensor, sdf: GridSDF, frames: np.ndarray, w: dict, q_human=None,
                  iters: int = 400, lr: float = 0.03, log=None):
    """q* (len(frames), 22) for the hand base poses base_obj (T, 4, 4) in the object frame."""
    T = len(frames)
    q_open = hand.open_pose()
    q = q_open.expand(T, 22).clone().requires_grad_(True)
    opt = torch.optim.Adam([q], lr=lr)
    B = base_obj[frames]
    _, links = hand.points_world(q_open[None], torch.eye(4, dtype=torch.float64)[None])
    groups = finger_groups(hand, links)
    for it in range(iters):
        opt.zero_grad()
        P, _ = hand.points_world(q, B)
        sd = sdf(P)                                   # (T, N)
        attach = sum((torch.relu(sd[:, g].amin(dim=1)) ** 2).sum() for _, g in groups)  # closest sample of each group
        pen = (torch.relu(-sd) ** 2).sum()
        smooth = ((q[1:] - q[:-1]) ** 2).sum() if T > 1 else q.sum() * 0
        reg = ((q - q_open) ** 2).sum()
        loss = w["attach"] * attach + w["pen"] * pen + w["smooth"] * smooth + w["open"] * reg
        if w.get("human", 0) and q_human is not None:
            loss = loss + w["human"] * ((q - q_human[frames]) ** 2).sum()
        loss.backward()
        opt.step()
        with torch.no_grad():
            q.copy_(torch.maximum(torch.minimum(q, hand.upper), hand.lower))
        if log is not None and (it % 100 == 0 or it == iters - 1):
            log(f"    it {it:4d} loss {loss.item():.4f} attach {attach.item():.5f} pen {pen.item():.3e} smooth {smooth.item():.4f}")
    return q.detach()


def evaluate_hand(hand: SharpaHand, q: torch.Tensor, base_obj: torch.Tensor, sdf: GridSDF):
    with torch.no_grad():
        P, links = hand.points_world(q, base_obj)
        sd = sdf(P)
    groups = finger_groups(hand, links)
    gmin = torch.stack([sd[:, g].min(1).values for _, g in groups], 1).numpy()  # (T, 6)
    pen = torch.relu(-sd).max(1).values.numpy()
    pen_palm = torch.relu(-sd[:, groups[0][1]]).max(1).values.numpy()
    touching = gmin <= CONTACT_TOL_M
    return dict(max_pen=pen, palm_pen=pen_palm, palm_touch=touching[:, 0], fingers_touching=touching[:, 1:].sum(1), group_min_sd=gmin)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("track", "locked"), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--human-weight", type=float, default=0.0, help="w of |q - q_human|^2 (0: no human finger data)")
    ap.add_argument("--frames", type=str, default=None, help="optional 'start:stop' subset (smoke tests)")
    a = ap.parse_args()
    if a.human_weight:
        raise NotImplementedError("w > 0 needs q_human from CORE4D finger keypoints (separate demonstration reference)")
    a.out.mkdir(parents=True, exist_ok=True)
    logf = (a.out / "log.txt").open("w")
    def log(*x):
        s = " ".join(str(v) for v in x)
        print(s, flush=True)
        logf.write(s + "\n")
    t0 = time.time()
    ref = dict(np.load(REFERENCE, allow_pickle=True))
    con = np.load(CONTACT, allow_pickle=True)
    dist = np.asarray(con["runtime_hand_surface_distances_m"], float)  # (T, agent, hand[left, right])
    fps = float(ref["fps"])
    T = ref["agent_joint_pos"].shape[0]
    sel = np.arange(T) if a.frames is None else np.arange(T)[slice(*[int(x) for x in a.frames.split(":")])]
    sdf = GridSDF(MESH, resolution=0.003, cache_dir=ROOT / "logs_system1/t06/cache")
    Robj = quat_wxyz_to_matrix(ref["object_quat_w"])
    pobj = np.asarray(ref["object_pos_w"], float)
    hands = {s: SharpaHand(s) for s in SIDES}
    mounts = {s: mount_transform(s, hands[s]) for s in SIDES}
    jn = [str(n) for n in ref["joint_names"]]
    new_joint = np.asarray(ref["agent_joint_pos"], float).copy()
    finger = np.zeros((T, 2, 44))
    blend = np.zeros((T, 2, 2))
    summary = dict(mode=a.mode, frames=int(T), evaluated_frames=[int(sel[0]), int(sel[-1])], fps=fps, weights=DEFAULT_W,
                   contact_definition=f"source human palm distance < {CONTACT_OUTER_M} m (source_contact_diagnostics.npz)",
                   contact_tolerance_m=CONTACT_TOL_M, ramp_s=RAMP_S, gate_max_penetration_m=GATE_PEN_M,
                   inputs=dict(reference=str(REFERENCE.relative_to(ROOT)), reference_sha256=sha256(REFERENCE),
                               contact=str(CONTACT.relative_to(ROOT)), contact_sha256=sha256(CONTACT),
                               mesh=str(MESH.relative_to(ROOT)), mesh_sha256=sha256(MESH),
                               g1_urdf_sha256=sha256(G1_URDF),
                               sharpa_urdf_sha256={s: sha256(hands[s].path) for s in SIDES}),
                   mount={s: mounts[s][1] for s in SIDES}, hands={})
    per_frame = []
    for agent in range(2):
        for hi, side in enumerate(SIDES):
            hand = hands[side]
            M = mounts[side][0]
            B_sh_wy = M[:3, :3] @ hand.palm_basis()
            ang, wdiag, (Rel, pel, chain) = solve_wrist(ref, agent, side, B_sh_wy)
            new_joint[:, agent, [jn.index(f"{side}_wrist_{n}_joint") for n in ("roll", "pitch", "yaw")]] = ang
            base = np.empty((T, 4, 4))
            for t in range(T):
                R, p = wrist_fk(Rel[t], pel[t], ang[t], chain)
                Twy = np.eye(4); Twy[:3, :3], Twy[:3, 3] = R, p
                Tob = np.eye(4); Tob[:3, :3], Tob[:3, 3] = Robj[t], pobj[t]
                base[t] = np.linalg.inv(Tob) @ Twy @ M          # hand flange in the object frame
            base_t = torch.as_tensor(base)
            contact = (dist[:, agent, hi] < CONTACT_OUTER_M) & np.isin(np.arange(T), sel)
            s = ramp_weights(contact, fps) * np.isin(np.arange(T), sel)
            q_open = hand.open_pose()
            qfull = q_open.expand(T, 22).clone()
            cf = np.flatnonzero(contact)
            log(f"agent {agent} {side}: wrist {json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in wdiag.items()})}; contact frames {len(cf)}")
            if a.mode == "track" and len(cf):
                qc = optimise_hand(hand, base_t, sdf, cf, DEFAULT_W, iters=a.iters, log=log)
                qstar = q_open.expand(T, 22).clone()
                qstar[cf] = qc
                ramp = np.flatnonzero((s > 0) & ~contact)
                for t in ramp:  # hold the nearest contact frame's pose, blended toward open
                    qstar[t] = qc[np.argmin(np.abs(cf - t))]
                qfull = q_open + torch.as_tensor(s)[:, None] * (qstar - q_open)
            ev = evaluate_hand(hand, qfull, base_t, sdf)
            ev_open = evaluate_hand(hand, q_open.expand(T, 22), base_t, sdf)
            finger[:, agent, hi * 22:(hi + 1) * 22] = qfull.numpy()
            blend[:, agent, hi] = s
            lim_ok = bool(((qfull >= hand.lower - 1e-9) & (qfull <= hand.upper + 1e-9)).all())
            ok_contact = ev["palm_touch"][cf] & (ev["fingers_touching"][cf] >= 2) if len(cf) else np.array([], bool)
            evm = np.isin(np.arange(T), sel)
            summary["hands"][f"agent{agent + 1}_{side}"] = dict(
                wrist=wdiag, contact_frames=int(len(cf)), ramp_frames=int(((s > 0) & ~contact).sum()),
                max_penetration_m=float(ev["max_pen"][evm].max()), frames_penetration_over_1cm=int((ev["max_pen"][evm] > GATE_PEN_M).sum()),
                palm_only_max_penetration_m=float(ev["palm_pen"][evm].max()),
                open_pose_max_penetration_m=float(ev_open["max_pen"][evm].max()),
                contact_frames_palm_and_2_fingers=int(ok_contact.sum()),
                contact_frames_palm_touching=int(ev["palm_touch"][cf].sum()) if len(cf) else 0,
                contact_frames_fingers_touching_mean=float(ev["fingers_touching"][cf].mean()) if len(cf) else None,
                palm_min_distance_in_contact_frames_m=dict(p50=float(np.median(ev["group_min_sd"][cf, 0])) if len(cf) else None,
                                                           max=float(ev["group_min_sd"][cf, 0].max()) if len(cf) else None),
                joint_limits_ok=lim_ok)
            for t in sel:
                per_frame.append(dict(agent=agent + 1, side=side, frame=int(t), contact=bool(contact[t]), blend=float(s[t]),
                                      max_penetration_m=float(ev["max_pen"][t]), palm_penetration_m=float(ev["palm_pen"][t]),
                                      palm_touching=bool(ev["palm_touch"][t]), fingers_touching=int(ev["fingers_touching"][t]),
                                      **{f"min_sd_{g}": float(ev["group_min_sd"][t, k]) for k, g in enumerate(("palm",) + FINGERS)}))
            log(f"agent {agent} {side}: " + json.dumps(summary["hands"][f"agent{agent + 1}_{side}"], default=float)[:600])
    hs = summary["hands"].values()
    summary["gate"] = dict(max_penetration_m=max(h["max_penetration_m"] for h in hs),
                           penetration_ok=all(h["max_penetration_m"] <= GATE_PEN_M for h in hs),
                           contact_ok=all(h["contact_frames_palm_and_2_fingers"] == h["contact_frames"] for h in hs),
                           joint_limits_ok=all(h["joint_limits_ok"] for h in hs))
    summary["seconds"] = time.time() - t0
    finger_names = [n for s in SIDES for n in hands[s].dof_names]
    np.savez_compressed(a.out / "finger_reference.npz", agent_joint_pos=new_joint, agent_finger_joint_pos=finger,
                        finger_joint_names=np.array(finger_names), joint_names=ref["joint_names"], contact_blend=blend,
                        fps=ref["fps"], mode=np.array(a.mode))
    import csv
    with (a.out / "per_frame.csv").open("w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(per_frame[0]))
        wtr.writeheader()
        wtr.writerows(per_frame)
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    log("gate", json.dumps(summary["gate"]), "seconds %.0f" % summary["seconds"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
