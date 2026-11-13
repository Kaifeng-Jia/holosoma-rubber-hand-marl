#!/usr/bin/env python3
"""T06 step 2 demonstration input: q_f^human for the w > 0 finger reference (Bin, CORE4D 20231020/071).

Source: source_canonical.npz of the Bin reference (human_joints_full: SMPL-X 127 joints per person and frame, 15 Hz;
person k = agent k as in source_contact_diagnostics). Per hand, 4 keypoints per finger: SMPL-X finger joints 1-3
(index 25-27, middle 28-30, pinky 31-33, ring 34-36, thumb 37-39; right hand +15) and the fingertip vertices-joints
66-70 (left thumb, index, middle, ring, pinky) / 71-75 (right).
Sharpa keypoints at the same anatomical places: the origin of the child link of each joint (thumb CMC / MCP / IP =
thumb_CMC_VL / thumb_MCP_VL / thumb_DP; finger MCP / PIP / DIP = <f>_MCP_VL / <f>_MP / <f>_DP) and <f>_fingertip.

Retargeting is scale-free: the 15 segment directions (3 per finger) are expressed in each hand's own palm frame (the A1
construction: forward = wrist -> middle MCP, across = index - pinky, normal = forward x across) and
  min_q  sum_segments |d_sharpa(q) - d_human|^2 + 1e-3 |q|^2,    joint limits hard (projection), Adam 400 steps,
per frame independently. q at 15 Hz is interpolated linearly to the 50 Hz runtime frames (t = frame / 50 s, the same
mapping as the contact diagnostics). Residual angles per segment are written for the report.

    taskset -c 0-15 hssim_s1/bin/python -B system1/hand/human_fingers.py --out logs_system1/t06/human_fingers/q_human.npz
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
from system1.hand.sharpa import FINGERS, SIDES, SharpaHand, palm_basis_from_landmarks  # noqa: E402

BIN_DIR = ROOT / "src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/core4d_bucket003_20231020_071_a1"
SOURCE = BIN_DIR / "source_canonical.npz"
REFERENCE = BIN_DIR / "core4d_pair_runtime_fps50.npz"
SMPLX_BASE = dict(left=25, right=40)                      # index1 of each hand
SMPLX_FINGER_OFFSET = dict(index=0, middle=3, pinky=6, ring=9, thumb=12)
SMPLX_TIP = dict(left=dict(thumb=66, index=67, middle=68, ring=69, pinky=70), right=dict(thumb=71, index=72, middle=73, ring=74, pinky=75))
SMPLX_WRIST = dict(left=20, right=21)
SHARPA_KP = dict(thumb=("thumb_CMC_VL", "thumb_MCP_VL", "thumb_DP", "thumb_fingertip"))
for _f in ("index", "middle", "ring", "pinky"):
    SHARPA_KP[_f] = (f"{_f}_MCP_VL", f"{_f}_MP", f"{_f}_DP", f"{_f}_fingertip")


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def human_keypoints(J: np.ndarray, side: str):
    """J (T, 127, 3) -> wrist (T, 3), {finger: (T, 4, 3)}."""
    b = SMPLX_BASE[side]
    kp = {}
    for f in FINGERS:
        o = b + SMPLX_FINGER_OFFSET[f]
        kp[f] = np.concatenate([J[:, o:o + 3], J[:, SMPLX_TIP[side][f]][:, None]], axis=1)
    return J[:, SMPLX_WRIST[side]], kp


def human_directions(J: np.ndarray, side: str) -> np.ndarray:
    """(T, 5, 3, 3): unit segment directions per finger (FINGERS order) in the human palm frame."""
    w, kp = human_keypoints(J, side)
    out = np.empty((len(J), len(FINGERS), 3, 3))
    for t in range(len(J)):
        B = palm_basis_from_landmarks(w[t], kp["index"][t, 0], kp["middle"][t, 0], kp["pinky"][t, 0], side)
        for k, f in enumerate(FINGERS):
            d = np.diff(kp[f][t], axis=0) @ B          # rows: segments, in palm coordinates
            out[t, k] = d / np.linalg.norm(d, axis=1, keepdims=True)
    return out


def sharpa_directions(hand: SharpaHand, q: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """(T, 5, 3, 3) unit segment directions in the Sharpa palm frame B (3, 3, flange coordinates)."""
    T = hand.fk(q)
    s = hand.side
    outs = []
    for f in FINGERS:
        P = torch.stack([T[f"{s}_{n}"][..., :3, 3] for n in SHARPA_KP[f]], dim=-2)  # (T, 4, 3)
        d = (P[..., 1:, :] - P[..., :-1, :]) @ B
        outs.append(d / torch.linalg.norm(d, dim=-1, keepdim=True))
    return torch.stack(outs, dim=-3)


def solve(hand: SharpaHand, dh: np.ndarray, iters: int = 400, lr: float = 0.05):
    B = torch.as_tensor(hand.palm_basis())
    target = torch.as_tensor(dh)
    q = torch.zeros(len(dh), 22, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([q], lr=lr)
    for _ in range(iters):
        opt.zero_grad()
        loss = ((sharpa_directions(hand, q, B) - target) ** 2).sum() + 1e-3 * (q ** 2).sum()
        loss.backward()
        opt.step()
        with torch.no_grad():
            q.copy_(torch.maximum(torch.minimum(q, hand.upper), hand.lower))
    with torch.no_grad():
        ds = sharpa_directions(hand, q, B).numpy()
    res = np.degrees(np.arccos(np.clip((ds * dh).sum(-1), -1, 1)))  # (T, 5, 3)
    return q.detach().numpy(), res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--iters", type=int, default=400)
    a = ap.parse_args()
    src = np.load(SOURCE, allow_pickle=True)
    J = np.asarray(src["human_joints_full"], float)          # (150, 2, 127, 3)
    fps_src = float(src["fps"])
    ref = np.load(REFERENCE, allow_pickle=True)
    T50, fps = ref["agent_joint_pos"].shape[0], float(ref["fps"])
    t_src = np.arange(J.shape[0]) / fps_src
    t_run = np.clip(np.arange(T50) / fps, t_src[0], t_src[-1])
    q_human = np.zeros((T50, 2, 44))
    q_src = np.zeros((J.shape[0], 2, 44))
    report = {}
    names = []
    for side in SIDES:
        names += SharpaHand(side).dof_names
    for agent in range(2):
        for hi, side in enumerate(SIDES):
            hand = SharpaHand(side)
            dh = human_directions(J[:, agent], side)
            q, res = solve(hand, dh, iters=a.iters)
            q_src[:, agent, hi * 22:(hi + 1) * 22] = q
            for j in range(22):
                q_human[:, agent, hi * 22 + j] = np.interp(t_run, t_src, q[:, j])
            dq = np.abs(np.diff(q_human[:, agent, hi * 22:(hi + 1) * 22], axis=0))
            report[f"agent{agent + 1}_{side}"] = dict(
                residual_deg=dict(mean=float(res.mean()), p95=float(np.percentile(res, 95)), max=float(res.max()),
                                  per_finger_mean={f: float(res[:, k].mean()) for k, f in enumerate(FINGERS)}),
                q_range_rad={n: [float(q[:, j].min()), float(q[:, j].max())] for j, n in enumerate(hand.dof_names)},
                runtime_step_max_rad=float(dq.max()), at_limit_fraction=float(((q <= hand.lower.numpy() + 1e-6) | (q >= hand.upper.numpy() - 1e-6)).mean()))
            print(agent + 1, side, json.dumps(report[f"agent{agent + 1}_{side}"]["residual_deg"]), "step max", round(report[f"agent{agent + 1}_{side}"]["runtime_step_max_rad"], 4))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, q_human=q_human, q_human_source_15hz=q_src, finger_joint_names=np.array(names), fps=fps,
                        source=str(SOURCE.relative_to(ROOT)), source_sha256=sha256(SOURCE))
    a.out.with_suffix(".json").write_text(json.dumps(dict(source=str(SOURCE.relative_to(ROOT)), source_sha256=sha256(SOURCE),
                                                          reference_sha256=sha256(REFERENCE), output_sha256=sha256(a.out),
                                                          method="segment directions in each palm frame, per-frame Adam, linear 15 -> 50 Hz",
                                                          hands=report), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
