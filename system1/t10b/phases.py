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


def segments(labels):
    """[(label, first_frame, last_frame)] runs of equal labels."""
    out = []
    for k, l in enumerate(labels):
        if out and out[-1][0] == l:
            out[-1][2] = k
        else:
            out.append([l, k, k])
    return [tuple(x) for x in out]
