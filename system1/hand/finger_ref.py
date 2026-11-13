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

--variant A (Kevin 2026-11-13, after the v1 gate failed; default v1 = the behaviour above, unchanged):
  * contact frames from the robot hand: the robot palm point (mean of the Sharpa wrist and index / middle / pinky MCP
    joint origins, the human palm-point construction) has signed distance to the bucket surface < 0.04 m (same outer
    band as the existing pipeline);
  * one optimisation over all frames of a hand: the attach term is weighted per frame by the 0.2 s smoothstep ramp
    weight s(t) (1 on contact frames, 0 away from them), the penetration, smoothness and open terms act on every
    frame, so fingers bend away from the object wherever the open pose would penetrate; the optimised q is used
    directly (no post-hoc blend toward the open pose);
  * the rigid part of the hand (hand_wrist and hand_C_MC: fixed to the flange, placed by the body reference) is
    reported separately; the 1 cm gate is on the finger links (every link of the five finger chains, including the
    thumb and pinky metacarpals). Palm + 2 fingers in contact frames is reported as in v1, plus fingers-only counts.

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
WBT = ROOT / "src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking"
# T07: the four paper tasks. reference: two-agent runtime reference (Push: the frozen registered pair the evaluator checks
# its runtime reference against, quaternions xyzw); object: collision geometry the simulator uses.
CASES = dict(
    bucket=dict(reference=REFERENCE, quat="wxyz", object=("mesh", MESH), contact=CONTACT,
                object_urdf=BIN_DIR / "bucket003_training.urdf"),
    table=dict(reference=WBT / "core4d_smalltable5kg_A/core4d_pair_runtime_fps50.npz", quat="wxyz",
               object=("boxes", WBT / "core4d_smalltable5kg_A/desk001_5kg_training.urdf"), contact=None,
               object_urdf=WBT / "core4d_smalltable5kg_A/desk001_5kg_training.urdf"),
    pull=dict(reference=WBT / "rubber_hand_largetable_v1/pull/plan5_attempt08_mirrored_pair_runtime.npz", quat="wxyz",
              object=("boxes", WBT / "objects_widetable_plan5_pull_training.urdf"), contact=None,
              object_urdf=WBT / "objects_widetable_plan5_pull_training.urdf"),
    push=dict(reference=ROOT / "system1/evidence/checks/push_registered_loader_channels.npz", quat="xyzw",
              object=("boxes", WBT / "objects_widetable_plan5_training.urdf"), contact=None,
              object_urdf=WBT / "objects_widetable_plan5_training.urdf"),
)


def load_reference(case: str) -> dict:
    """Two-agent reference with quaternions converted to wxyz (the convention of the rest of this module)."""
    c = CASES[case]
    ref = dict(np.load(c["reference"], allow_pickle=True))
    if c["quat"] == "xyzw":
        for k in ("agent_body_quat_w", "object_quat_w"):
            ref[k] = np.asarray(ref[k])[..., [3, 0, 1, 2]]
    ref["fps"] = np.asarray(float(np.asarray(ref["fps"]).reshape(-1)[0]))
    return ref


def object_sdf(case: str):
    kind, path = CASES[case]["object"]
    if kind == "mesh":
        return GridSDF(path, resolution=0.003, cache_dir=ROOT / "logs_system1/t06/cache")
    from system1.hand.sdf import BoxUnionSDF
    return BoxUnionSDF(path)
CONTACT_OUTER_M = 0.04          # existing soft band outer edge (source_contact_diagnostics.json)
RAMP_S = 0.2                    # T06: ~0.2 s transitions
CONTACT_TOL_M = 0.005           # a palm / finger group "touches" if its closest sample is within 5 mm
GATE_PEN_M = 0.01
DEFAULT_W = dict(attach=1e4, pen=1e6, smooth=10.0, open=0.01, human=0.0)
RIGID = ("hand_wrist", "hand_C_MC")  # links fixed to the flange (variant A: reported separately)
PALM_GROUP_A = ("hand_C_MC", "thumb_MC", "hand_wrist")  # task book 2026-11-13: palm + thumb_MC (+ wrist housing, see finger_groups)
GATE_CONTACT_FRAC = 0.8
GATE_EXCESS_M = 0.01      # second revision (2026-11-13 15:30): palm group may be at most 1 cm deeper than the rubber hand
GATE_STEP_RAD = 0.1       # second revision, gate 5: every finger joint changes by at most 0.1 rad per frame


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


def finger_groups(hand: SharpaHand, links: list[str], variant: str = "v1") -> list[tuple[str, np.ndarray]]:
    """v1: palm = hand_C_MC, each finger = all its links. A (task book 2026-11-13 revision): palm group = hand_C_MC +
    thumb_MC + hand_wrist (the wrist housing, rigid with the palm, is added here; the task book names palm + thumb_MC),
    thumb = the thumb links without thumb_MC."""
    s = hand.side
    palm = {f"{s}_hand_C_MC"} if variant == "v1" else {f"{s}_{x}" for x in PALM_GROUP_A}
    groups = [("palm", np.array([i for i, l in enumerate(links) if l in palm]))]
    for f in FINGERS:
        groups.append((f, np.array([i for i, l in enumerate(links) if l.startswith(f"{s}_{f}_") and l not in palm])))
    return groups


def dense_samples(items, spacing: float = 0.003):
    """items: [(link, mesh path, 4x4 origin)] -> (link per point, (N, 3) local points): every vertex plus area / spacing^2
    uniform surface samples per mesh (same density for the Sharpa and the rubber hand)."""
    import trimesh
    links, pts = [], []
    for k, (link, path, origin) in enumerate(items):
        m = trimesh.load(path, force="mesh")
        n = max(64, int(math.ceil(m.area / spacing ** 2)))
        sp, _ = trimesh.sample.sample_surface(m, n, seed=k)
        p = np.vstack([sp, m.vertices])
        p = (origin[:3, :3] @ p.T).T + origin[:3, 3]
        links += [link] * len(p)
        pts.append(p)
    return links, np.vstack(pts)


def evaluate_gate_A(hand: SharpaHand, q: torch.Tensor, base_obj: torch.Tensor, sdf: GridSDF, chunk: int = 25):
    """Dense (3 mm) per-frame penetration and contact for the revised gate: palm group (C_MC + thumb_MC + wrist),
    palm core (C_MC + thumb_MC, the task book's palm group), other fingers; touching = closest point within 5 mm."""
    links, local = dense_samples([(c.link, c.mesh, c.origin) for c in hand.collisions])
    s = hand.side
    L = np.array(links)
    core = np.isin(L, [f"{s}_hand_C_MC", f"{s}_thumb_MC"])
    pg = core | (L == f"{s}_hand_wrist")
    groups = [("palm", np.flatnonzero(pg))] + [(f, np.flatnonzero(np.char.startswith(L, f"{s}_{f}_") & ~pg)) for f in FINGERS]
    fing = np.concatenate([g for _, g in groups[1:]])
    loc = torch.as_tensor(local)
    out = {k: [] for k in ("palm_group_pen", "palm_core_pen", "finger_pen", "gmin", "core_min")}
    with torch.no_grad():
        for a in range(0, len(q), chunk):
            T = hand.fk(q[a:a + chunk], base_obj[a:a + chunk])
            P = torch.empty(len(q[a:a + chunk]), len(L), 3, dtype=torch.float64)
            for l in set(links):
                ii = torch.as_tensor(np.flatnonzero(L == l))
                P[:, ii] = (T[l][:, None, :3, :3] @ loc[ii][..., None]).squeeze(-1) + T[l][:, None, :3, 3]
            sd = sdf(P)
            out["palm_group_pen"].append(torch.relu(-sd[:, pg]).max(1).values)
            out["palm_core_pen"].append(torch.relu(-sd[:, core]).max(1).values)
            out["finger_pen"].append(torch.relu(-sd[:, fing]).max(1).values)
            out["gmin"].append(torch.stack([sd[:, g].min(1).values for _, g in groups], 1))
            out["core_min"].append(sd[:, core].min(1).values)
    o = {k: torch.cat(v).numpy() for k, v in out.items()}
    touch = o["gmin"] <= CONTACT_TOL_M
    return dict(palm_group_pen=o["palm_group_pen"], palm_core_pen=o["palm_core_pen"], finger_pen=o["finger_pen"],
                palm_group_touch=touch[:, 0], palm_core_touch=o["core_min"] <= CONTACT_TOL_M, fingers_touching=touch[:, 1:].sum(1),
                group_min_sd=o["gmin"], samples=int(len(L)))


def rubber_hand_penetration(ref: dict, agent: int, side: str, sdf: GridSDF, chunk: int = 50) -> np.ndarray:
    """Per-frame max penetration of the original D reference's rubber hand (collision mesh of <side>_rubber_hand_link,
    the reference pose of that body), dense 3 mm samples like evaluate_gate_A."""
    r = ET.parse(G1_URDF).getroot()
    col = r.find(f"link[@name='{side}_rubber_hand_link']/collision")
    o = col.find("origin")
    origin = np.eye(4)
    origin[:3, :3] = rpy_matrix(*[float(x) for x in o.get("rpy").split()]) if o is not None else np.eye(3)
    origin[:3, 3] = [float(x) for x in o.get("xyz").split()] if o is not None else 0
    _, local = dense_samples([("rubber", G1_URDF.parent / col.find("geometry/mesh").get("filename"), origin)])
    names = [str(n) for n in ref["body_names"]]
    ib = names.index(f"{side}_rubber_hand_link")
    Rh = quat_wxyz_to_matrix(ref["agent_body_quat_w"][:, agent, ib])
    ph = np.asarray(ref["agent_body_pos_w"][:, agent, ib], float)
    Ro = quat_wxyz_to_matrix(ref["object_quat_w"])
    po = np.asarray(ref["object_pos_w"], float)
    out = []
    for a in range(0, len(ph), chunk):
        w = np.einsum("tij,nj->tni", Rh[a:a + chunk], local) + ph[a:a + chunk, None]
        ob = np.einsum("tji,tnj->tni", Ro[a:a + chunk], w - po[a:a + chunk, None])  # R^T (w - p)
        out.append(torch.relu(-sdf(torch.as_tensor(ob))).max(1).values.numpy())
    return np.concatenate(out)


def optimise_hand(hand: SharpaHand, base_obj: torch.Tensor, sdf: GridSDF, frames: np.ndarray, w: dict, q_human=None,
                  iters: int = 400, lr: float = 0.03, log=None, attach_frame_w=None, variant: str = "v1", max_step=None,
                  rate_weight: float = 0.0, rate_limit: float = 0.1):
    """q* (len(frames), 22) for the hand base poses base_obj (T, 4, 4) in the object frame.
    attach_frame_w: optional per-frame weight of the attach term (len(frames)); None = 1 on every frame (v1)."""
    T = len(frames)
    aw = torch.ones(T, dtype=torch.float64) if attach_frame_w is None else torch.as_tensor(np.asarray(attach_frame_w, float))
    q_open = hand.open_pose()
    q = q_open.expand(T, 22).clone().requires_grad_(True)
    opt = torch.optim.Adam([q], lr=lr)
    B = base_obj[frames]
    _, links = hand.points_world(q_open[None], torch.eye(4, dtype=torch.float64)[None])
    groups = finger_groups(hand, links, variant)
    for it in range(iters):
        opt.zero_grad()
        P, _ = hand.points_world(q, B)
        sd = sdf(P)                                   # (T, N)
        attach = sum((aw * torch.relu(sd[:, g].amin(dim=1)) ** 2).sum() for _, g in groups)  # closest sample of each group
        pen = (torch.relu(-sd) ** 2).sum()
        smooth = ((q[1:] - q[:-1]) ** 2).sum() if T > 1 else q.sum() * 0
        reg = ((q - q_open) ** 2).sum()
        loss = w["attach"] * attach + w["pen"] * pen + w["smooth"] * smooth + w["open"] * reg
        if rate_weight and T > 1:  # hinge on the per-frame change above rate_limit (gate 5), lets the optimiser anticipate
            loss = loss + rate_weight * (torch.relu((q[1:] - q[:-1]).abs() - rate_limit) ** 2).sum()
        if w.get("human", 0) and q_human is not None:
            loss = loss + w["human"] * ((q - q_human[frames]) ** 2).sum()
        loss.backward()
        opt.step()
        with torch.no_grad():
            q.copy_(torch.maximum(torch.minimum(q, hand.upper), hand.lower))
            if max_step is not None:  # hard per-frame rate limit (forward pass; consecutive frames of `frames`)
                for t in range(1, T):
                    q[t] = torch.maximum(torch.minimum(q[t], q[t - 1] + max_step), q[t - 1] - max_step)
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
    fing = np.concatenate([g for _, g in groups[1:]])
    rigid = np.array([i for i, l in enumerate(links) if l in {f"{hand.side}_{r}" for r in RIGID}])
    pen_f = torch.relu(-sd[:, fing]).max(1).values.numpy()
    pen_r = torch.relu(-sd[:, rigid]).max(1).values.numpy()
    touching = gmin <= CONTACT_TOL_M
    return dict(max_pen=pen, palm_pen=pen_palm, finger_pen=pen_f, rigid_pen=pen_r, palm_touch=touching[:, 0],
                fingers_touching=touching[:, 1:].sum(1), group_min_sd=gmin)


def palm_point_flange(hand: SharpaHand) -> np.ndarray:
    """Robot palm point in the flange frame: mean of wrist and index / middle / pinky MCP (human palm-point construction)."""
    lm = hand.palm_landmarks()
    return np.mean([lm["wrist"], lm["index"], lm["middle"], lm["pinky"]], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("track", "locked"), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--human-weight", type=float, default=0.0, help="w of |q - q_human|^2 (0: no human finger data)")
    ap.add_argument("--human-ref", type=Path, default=None,
                    help="npz with q_human (T, 2, 44) in finger_joint_names order (system1/hand/human_fingers.py); needed for w > 0")
    ap.add_argument("--smooth-weight", type=float, default=None, help=f"smoothness weight (default {DEFAULT_W['smooth']})")
    ap.add_argument("--rate-weight", type=float, default=0.0,
                    help="variant A: weight of sum relu(|q_t - q_t-1| - 0.1 rad)^2 (default 0 = off)")
    ap.add_argument("--max-step-rad", type=float, default=None,
                    help="variant A: hard limit on every finger joint's change per frame, applied after each optimiser step (default off)")
    ap.add_argument("--frames", type=str, default=None, help="optional 'start:stop' subset (smoke tests)")
    ap.add_argument("--case", choices=tuple(CASES), default="bucket", help="T07 task (default bucket = T06)")
    ap.add_argument("--variant", choices=("v1", "A"), default="v1",
                    help="v1: human-palm contact frames, contact frames optimised (default, unchanged); "
                         "A: robot-palm contact frames, all frames optimised, rigid hand part reported separately")
    a = ap.parse_args()
    if a.human_weight and a.human_ref is None:
        raise NotImplementedError("w > 0 needs q_human from CORE4D finger keypoints (--human-ref, system1/hand/human_fingers.py)")
    if a.human_weight and a.variant != "A":
        raise NotImplementedError("w > 0 is implemented for --variant A only")
    W = dict(DEFAULT_W)
    if a.smooth_weight is not None:
        W["smooth"] = a.smooth_weight
    W["human"] = a.human_weight
    q_hum_all = None
    if a.human_weight:
        hz = np.load(a.human_ref, allow_pickle=True)
        q_hum_all = np.asarray(hz["q_human"], float)
    a.out.mkdir(parents=True, exist_ok=True)
    logf = (a.out / "log.txt").open("w")
    def log(*x):
        s = " ".join(str(v) for v in x)
        print(s, flush=True)
        logf.write(s + "\n")
    t0 = time.time()
    case = CASES[a.case]
    ref = load_reference(a.case)
    T0 = ref["agent_joint_pos"].shape[0]
    if case["contact"] is not None:
        con = np.load(case["contact"], allow_pickle=True)
        dist = np.asarray(con["runtime_hand_surface_distances_m"], float)  # (T, agent, hand[left, right])
    else:
        if a.variant == "v1":
            raise ValueError("v1 needs the human contact diagnostics, which exist only for bucket")
        dist = np.full((T0, 2, 2), np.nan)
    fps = float(ref["fps"])
    T = ref["agent_joint_pos"].shape[0]
    sel = np.arange(T) if a.frames is None else np.arange(T)[slice(*[int(x) for x in a.frames.split(":")])]
    sdf = object_sdf(a.case)
    Robj = quat_wxyz_to_matrix(ref["object_quat_w"])
    pobj = np.asarray(ref["object_pos_w"], float)
    hands = {s: SharpaHand(s) for s in SIDES}
    mounts = {s: mount_transform(s, hands[s]) for s in SIDES}
    jn = [str(n) for n in ref["joint_names"]]
    new_joint = np.asarray(ref["agent_joint_pos"], float).copy()
    finger = np.zeros((T, 2, 44))
    blend = np.zeros((T, 2, 2))
    summary = dict(mode=a.mode, variant=a.variant, frames=int(T), evaluated_frames=[int(sel[0]), int(sel[-1])], fps=fps, weights=W,
                   human_ref=None if a.human_ref is None else dict(path=str(a.human_ref), sha256=sha256(a.human_ref)),
                   max_step_rad=a.max_step_rad, rate_weight=a.rate_weight, iters=a.iters,
                   contact_definition=(f"source human palm distance < {CONTACT_OUTER_M} m (source_contact_diagnostics.npz)" if a.variant == "v1"
                                       else f"robot palm point (mean of Sharpa wrist, index/middle/pinky MCP) signed distance < {CONTACT_OUTER_M} m"),
                   rigid_links=list(RIGID) if a.variant == "A" else None,
                   contact_tolerance_m=CONTACT_TOL_M, ramp_s=RAMP_S, gate_max_penetration_m=GATE_PEN_M,
                   case=a.case,
                   inputs=dict(reference=str(case["reference"].relative_to(ROOT)), reference_sha256=sha256(case["reference"]),
                               contact=None if case["contact"] is None else str(case["contact"].relative_to(ROOT)),
                               contact_sha256=None if case["contact"] is None else sha256(case["contact"]),
                               object=str(case["object"][1].relative_to(ROOT)), object_kind=case["object"][0],
                               object_sha256=sha256(case["object"][1]),
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
            pp = palm_point_flange(hand)
            palm_sd = sdf(torch.as_tensor(base[:, :3, :3] @ pp + base[:, :3, 3])).numpy()  # robot palm point, object frame
            d_contact = dist[:, agent, hi] if a.variant == "v1" else palm_sd
            contact = (d_contact < CONTACT_OUTER_M) & np.isin(np.arange(T), sel)
            s = ramp_weights(contact, fps) * np.isin(np.arange(T), sel)
            q_open = hand.open_pose()
            qfull = q_open.expand(T, 22).clone()
            cf = np.flatnonzero(contact)
            log(f"agent {agent} {side}: wrist {json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in wdiag.items()})}; contact frames {len(cf)}")
            if a.mode == "track" and a.variant == "A":
                qh = None if q_hum_all is None else torch.as_tensor(q_hum_all[:, agent, hi * 22:(hi + 1) * 22])
                qs = optimise_hand(hand, base_t, sdf, sel, W, q_human=qh, iters=a.iters, log=log, attach_frame_w=s[sel], variant="A",
                                   max_step=a.max_step_rad, rate_weight=a.rate_weight)
                qfull[sel] = qs
            elif a.mode == "track" and len(cf):
                qc = optimise_hand(hand, base_t, sdf, cf, W, iters=a.iters, log=log)
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
            dqj = np.abs(np.diff(qfull[sel].numpy(), axis=0)) if len(sel) > 1 else np.zeros((1, 22))
            dq = dqj.max(1)
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
                joint_limits_ok=lim_ok,
                finger_max_penetration_m=float(ev["finger_pen"][evm].max()),
                frames_finger_penetration_over_1cm=int((ev["finger_pen"][evm] > GATE_PEN_M).sum()),
                contact_frames_finger_penetration_over_1cm=int((ev["finger_pen"][cf] > GATE_PEN_M).sum()) if len(cf) else 0,
                rigid_max_penetration_m=float(ev["rigid_pen"][evm].max()),
                frames_rigid_penetration_over_1cm=int((ev["rigid_pen"][evm] > GATE_PEN_M).sum()),
                contact_frames_2_fingers=int((ev["fingers_touching"][cf] >= 2).sum()) if len(cf) else 0,
                robot_palm_point_sd_m=dict(min=float(palm_sd[evm].min()), p50=float(np.median(palm_sd[evm]))),
                finger_step_max_rad=float(dq.max()), finger_speed_max_rad_s=float(dq.max() * fps),
                finger_steps_over_0p1_rad=int((dq > 0.1).sum()), finger_steps_over_0p5_rad=int((dq > 0.5).sum()),
                finger_step_max_rad_per_joint={n: float(v) for n, v in zip(hand.dof_names, dqj.max(0))})
            gA = None
            if a.variant == "A":  # revised gate (task book 2026-11-13), dense 3 mm samples
                gA = evaluate_gate_A(hand, qfull, base_t, sdf)
                rub = rubber_hand_penetration(ref, agent, side, sdf)
                rev = {}
                for nm, pen_key, touch_key in (("palm_group", "palm_group_pen", "palm_group_touch"),
                                               ("palm_core", "palm_core_pen", "palm_core_touch")):
                    exc = (gA[pen_key] - rub)[evm]
                    okc = gA[touch_key][cf] & (gA["fingers_touching"][cf] >= 2) if len(cf) else np.array([], bool)
                    rev[nm] = dict(max_penetration_m=float(gA[pen_key][evm].max()), frames_deeper_than_rubber=int((exc > 0).sum()),
                                   frames_deeper_than_rubber_by_1mm=int((exc > 0.001).sum()), max_excess_over_rubber_m=float(exc.max()),
                                   contact_frames_palm_and_2_fingers=int(okc.sum()),
                                   contact_fraction_palm_and_2_fingers=float(okc.mean()) if len(cf) else None)
                inh = rub > GATE_PEN_M                          # frames where the original rubber hand is > 1 cm inside
                fin = gA["finger_pen"]
                summary["hands"][f"agent{agent + 1}_{side}"]["gate_revised2"] = dict(
                    palm_group_max_excess_over_rubber_m=rev["palm_group"]["max_excess_over_rubber_m"],
                    palm_group_frames_excess_over_1cm=int(((gA["palm_group_pen"] - rub)[evm] > GATE_EXCESS_M).sum()),
                    finger_max_penetration_where_rubber_within_1cm_m=float(fin[evm & ~inh].max()) if (evm & ~inh).any() else 0.0,
                    frames_finger_over_1cm_where_rubber_within_1cm=int((fin[evm & ~inh] > GATE_PEN_M).sum()),
                    frames_rubber_over_1cm=int((evm & inh).sum()),
                    inherited_frames_finger_over_1cm=int((fin[evm & inh] > GATE_PEN_M).sum()),
                    inherited_finger_max_penetration_m=float(fin[evm & inh].max()) if (evm & inh).any() else 0.0,
                    contact_fraction_palm_and_2_fingers=rev["palm_group"]["contact_fraction_palm_and_2_fingers"],
                    finger_step_max_rad=float(dq.max()), frames_step_over_0p1_rad=int((dq > GATE_STEP_RAD + 1e-12).sum()))
                summary["hands"][f"agent{agent + 1}_{side}"]["gate_revised"] = dict(
                    dense_samples_per_hand=gA["samples"], rubber_max_penetration_m=float(rub[evm].max()),
                    rubber_frames_over_1cm=int((rub[evm] > GATE_PEN_M).sum()),
                    finger_max_penetration_m=float(gA["finger_pen"][evm].max()), frames_finger_over_1cm=int((gA["finger_pen"][evm] > GATE_PEN_M).sum()),
                    contact_frames_finger_over_1cm=int((gA["finger_pen"][cf] > GATE_PEN_M).sum()) if len(cf) else 0,
                    contact_frames_2_fingers=int((gA["fingers_touching"][cf] >= 2).sum()) if len(cf) else 0, **rev)
            for t in sel:
                per_frame.append(dict(agent=agent + 1, side=side, frame=int(t), contact=bool(contact[t]), blend=float(s[t]),
                                      max_penetration_m=float(ev["max_pen"][t]), palm_penetration_m=float(ev["palm_pen"][t]),
                                      finger_penetration_m=float(ev["finger_pen"][t]), rigid_penetration_m=float(ev["rigid_pen"][t]),
                                      robot_palm_point_sd_m=float(palm_sd[t]), human_palm_distance_m=float(dist[t, agent, hi]),
                                      palm_touching=bool(ev["palm_touch"][t]), fingers_touching=int(ev["fingers_touching"][t]),
                                      **{f"min_sd_{g}": float(ev["group_min_sd"][t, k]) for k, g in enumerate(("palm",) + FINGERS)},
                                      **({} if gA is None else dict(
                                          rev_palm_group_penetration_m=float(gA["palm_group_pen"][t]), rev_palm_core_penetration_m=float(gA["palm_core_pen"][t]),
                                          rev_finger_penetration_m=float(gA["finger_pen"][t]), rubber_hand_penetration_m=float(rub[t]),
                                          rev_palm_group_touching=bool(gA["palm_group_touch"][t]), rev_palm_core_touching=bool(gA["palm_core_touch"][t]),
                                          rev_fingers_touching=int(gA["fingers_touching"][t]),
                                          **{f"rev_min_sd_{g}": float(gA["group_min_sd"][t, k]) for k, g in enumerate(("palm_group",) + FINGERS)}))))
            log(f"agent {agent} {side}: " + json.dumps(summary["hands"][f"agent{agent + 1}_{side}"], default=float)[:600])
    hs = summary["hands"].values()
    summary["gate"] = dict(max_penetration_m=max(h["max_penetration_m"] for h in hs),
                           penetration_ok=all(h["max_penetration_m"] <= GATE_PEN_M for h in hs),
                           contact_ok=all(h["contact_frames_palm_and_2_fingers"] == h["contact_frames"] for h in hs),
                           joint_limits_ok=all(h["joint_limits_ok"] for h in hs))
    if a.variant == "A":  # revised gate (task book 2026-11-13): 1. palm group per frame <= rubber hand; 2. other fingers
        # <= 1 cm; 3. >= 80 % of robot-defined contact frames with palm + 2 fingers (hands without contact frames: n/a)
        g = [h["gate_revised"] for h in hs]
        cfr = [x["palm_group"]["contact_fraction_palm_and_2_fingers"] for x in g]
        summary["gate_revised"] = dict(
            palm_group_links=list(PALM_GROUP_A), palm_core_links=["hand_C_MC", "thumb_MC"],
            c1_palm_group_not_deeper_than_rubber=all(x["palm_group"]["frames_deeper_than_rubber"] == 0 for x in g),
            c1_frames_deeper=[x["palm_group"]["frames_deeper_than_rubber"] for x in g],
            c1_palm_core_not_deeper_than_rubber=all(x["palm_core"]["frames_deeper_than_rubber"] == 0 for x in g),
            c2_fingers_within_1cm=all(x["finger_max_penetration_m"] <= GATE_PEN_M for x in g),
            c2_finger_max_penetration_m=max(x["finger_max_penetration_m"] for x in g),
            c3_contact_fraction_palm_and_2_fingers=cfr,
            c3_at_least_80pct=all(c is None or c >= GATE_CONTACT_FRAC for c in cfr),
            c3_palm_core_at_least_80pct=all(x["palm_core"]["contact_fraction_palm_and_2_fingers"] is None
                                            or x["palm_core"]["contact_fraction_palm_and_2_fingers"] >= GATE_CONTACT_FRAC for x in g),
            joint_limits_ok=summary["gate"]["joint_limits_ok"])
        g2 = [h["gate_revised2"] for h in hs]
        summary["gate_revised2"] = dict(  # task book second revision (2026-11-13 15:30)
            c1_palm_group_excess_within_1cm=all(x["palm_group_max_excess_over_rubber_m"] <= GATE_EXCESS_M for x in g2),
            c1_max_excess_m=max(x["palm_group_max_excess_over_rubber_m"] for x in g2),
            c2_fingers_within_1cm_where_rubber_within_1cm=all(x["frames_finger_over_1cm_where_rubber_within_1cm"] == 0 for x in g2),
            c2_finger_max_where_rubber_within_1cm_m=max(x["finger_max_penetration_where_rubber_within_1cm_m"] for x in g2),
            c2_inherited_frames_finger_over_1cm=[x["inherited_frames_finger_over_1cm"] for x in g2],
            c3_at_least_80pct=summary["gate_revised"]["c3_at_least_80pct"],
            c3_contact_fraction_palm_and_2_fingers=summary["gate_revised"]["c3_contact_fraction_palm_and_2_fingers"],
            c5_finger_step_within_0p1_rad=all(x["finger_step_max_rad"] <= GATE_STEP_RAD + 1e-12 for x in g2),
            c5_finger_step_max_rad=max(x["finger_step_max_rad"] for x in g2),
            joint_limits_ok=summary["gate"]["joint_limits_ok"])
    summary["seconds"] = time.time() - t0
    finger_names = [n for s in SIDES for n in hands[s].dof_names]
    np.savez_compressed(a.out / "finger_reference.npz", agent_joint_pos=new_joint, agent_finger_joint_pos=finger,
                        finger_joint_names=np.array(finger_names), joint_names=ref["joint_names"], contact_blend=blend,
                        fps=ref["fps"], mode=np.array(a.mode), variant=np.array(a.variant), case=np.array(a.case),
                        pelvis_pos_w=np.asarray(ref["agent_body_pos_w"])[:, :, [str(x) for x in ref["body_names"]].index("pelvis")],
                        pelvis_quat_w=np.asarray(ref["agent_body_quat_w"])[:, :, [str(x) for x in ref["body_names"]].index("pelvis")],
                        object_pos_w=np.asarray(ref["object_pos_w"]), object_quat_w=np.asarray(ref["object_quat_w"]))
    import csv
    with (a.out / "per_frame.csv").open("w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(per_frame[0]))
        wtr.writeheader()
        wtr.writerows(per_frame)
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    log("gate", json.dumps(summary["gate"]), "seconds %.0f" % summary["seconds"])
    if "gate_revised2" in summary:
        log("gate_revised2", json.dumps(summary["gate_revised2"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
