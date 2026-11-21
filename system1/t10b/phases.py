"""Phase labels of a paired reference, from the reference object's motion only (facts, no interpretation of the policy):
per frame (50 Hz), with z0 = the object's initial height and speeds from 0.1 s central differences:
  rotate   object angular speed > 0.5 rad/s
  lift     vertical speed > +0.05 m/s
  lower    vertical speed < -0.05 m/s
  carry    height > z0 + 0.05 m, otherwise (held up: carrying / handing over)
  push     height <= z0 + 0.05 m and planar speed > 0.05 m/s (pushing / pulling / sliding on the ground)
  still    otherwise
Used by the T10s / T10b failure summaries: 'the step where the object error starts to grow' is reported with the label of the
reference phase at that step.
"""
from __future__ import annotations

import numpy as np

LABELS = ("rotate", "lift", "lower", "carry", "push", "still")
NAME_ZH = dict(rotate="翻转/转动", lift="抬起", lower="放下", carry="举起中（搬运/交接）", push="贴地推行/拉行", still="静止")


def quat_angle_xyzw(a, b):
    d = np.abs(np.sum(a * b, axis=-1))
    return 2 * np.arccos(np.clip(d, -1.0, 1.0))


def label_frames(obj_pos, obj_quat_xyzw, fps=50, h=5):
    n = len(obj_pos)
    z0 = obj_pos[0, 2]
    out = []
    for k in range(n):
        a, b = max(0, k - h), min(n - 1, k + h)
        dt = (b - a) / fps if b > a else 1.0
        v = (obj_pos[b] - obj_pos[a]) / dt
        w = quat_angle_xyzw(obj_quat_xyzw[a], obj_quat_xyzw[b]) / dt
        if w > 0.5:
            out.append("rotate")
        elif v[2] > 0.05:
            out.append("lift")
        elif v[2] < -0.05:
            out.append("lower")
        elif obj_pos[k, 2] > z0 + 0.05:
            out.append("carry")
        elif np.hypot(v[0], v[1]) > 0.05:
            out.append("push")
        else:
            out.append("still")
    return out


def divergence_step(obj_pos, ref_pos, threshold=0.1):
    """First saved row where the object position error exceeds ``threshold`` m (None if never)."""
    e = np.linalg.norm(obj_pos - ref_pos, axis=-1)
    idx = np.flatnonzero(e > threshold)
    return int(idx[0]) if idx.size else None


def termination_errors(z):
    """Object position / orientation error of an evaluator trajectory (scripts/evaluate_paper_supplement.py) at its last saved row,
    against the reference frame the termination term compared with: the motion command's frame at that row, saved as
    reward_reference_frame (one control step behind the row's own reference_frame; row k holds reference frame min(k, N-1)).
    Returns (position error m, orientation error rad, that frame)."""
    f = int(z["reward_reference_frame"][-1])
    rows = np.flatnonzero(np.asarray(z["reference_frame"]) == f)
    if rows.size == 0:
        raise ValueError(f"reference frame {f} not saved in the trajectory")
    j = int(rows[0])
    pe = float(np.linalg.norm(z["object_pos_w"][-1] - z["reference_object_pos_w"][j]))
    oe = float(quat_angle_xyzw(z["object_quat_xyzw"][-1], z["reference_object_quat_xyzw"][j]))
    return pe, oe, f


def termination_reason(pe, oe, pos_threshold=0.25, ori_threshold=0.8):
    """joint_bad_tracking's object parts in order; otherwise the robot part (torso tilt, not saved in the trajectory)."""
    return "object_position" if pe > pos_threshold else "object_orientation" if oe > ori_threshold else "robot_orientation"


def segments(labels):
    """[(label, first_frame, last_frame)] runs of equal labels."""
    out = []
    for k, l in enumerate(labels):
        if out and out[-1][0] == l:
            out[-1][2] = k
        else:
            out.append([l, k, k])
    return [tuple(x) for x in out]
