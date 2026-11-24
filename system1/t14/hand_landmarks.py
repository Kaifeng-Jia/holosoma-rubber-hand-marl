"""T14 round 4 (section 9 item 2): hand landmarks for the interaction mesh and hand-orientation metrics.

Human (SMPL-X): per hand Index1 and Pinky1 joints (index / pinky MCP; SMPL-X joints 25 / 31 left, 40 / 46 right) and the
middle fingertip (SMPL-X vertex 5058 left, 7794 right; smplx.vertex_ids 'lmiddle' / 'rmiddle').

Robot (rubber hand, the simulator's PhysX hull of <side>_rubber_hand_link): the corresponding points are derived from the
hull the same way for every point, by proportional placement in the hand's bounding box:
  * human hand frame (SMPL-X neutral template, flat hand): origin wrist joint; a = wrist -> Middle1 (long axis);
    n = palm normal = sign * normalize((Index1 - wrist) x (Pinky1 - wrist)), sign chosen so that n points out of the palm
    (the same construction as table_shared.palm_vertex_ids); a orthogonalised against n; l = n x a;
    hand vertices = template vertices whose largest skinning weight is the wrist or a finger joint of that hand;
    u(p) = (coords(p) - bbox_min) / (bbox_max - bbox_min) in (a, l, n) coordinates;
  * robot hand frame (link frame): a = +x (the hull's long axis, wrist -> fingers), n = the palm normal used for the palm
    point (table_shared.palm_point), a orthogonalised against n, l = n x a; robot point = bbox_min + u * (bbox_max - bbox_min)
    of the hull vertices in (a, l, n) coordinates.
  The thumb lies on the same side (sign of the l coordinate) for the human template and the robot hull (checked).

Orientation (metrics, term B): palm normal = n of the hand frame (human: from the posed Index1 / Pinky1 / wrist joints with
the template's sign; robot: link rotation @ n); finger direction: human wrist -> middle fingertip, robot link rotation @ a;
hand long axis (secondary): human wrist -> Middle1.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

SMPLX_NEUTRAL = Path("/home/kevin/CS6758Project/data/smplx_models/smplx/SMPLX_NEUTRAL.npz")
JOINTS = dict(left=dict(wrist=20, index1=25, middle1=28, pinky1=31, thumb1=37, fingers=range(25, 40)),
              right=dict(wrist=21, index1=40, middle1=43, pinky1=46, thumb1=52, fingers=range(40, 55)))
TIP_VERTEX = dict(left=5058, right=7794)
LANDMARKS = ("Index1", "Pinky1", "MiddleTip")
PALM_NORMAL_LOCAL = dict(left=np.array([-0.0751, -0.9954, -0.0594]), right=np.array([-0.0751, 0.9954, -0.0593]))  # = table_shared.palm_point
THUMB_MARKER_LOCAL = np.array([0.08, 0.0, 0.05])  # retargeting MJCF thumb marker sphere in the rubber-hand frame (side check only)


def keys(side):
    return [("L_" if side == "left" else "R_") + k for k in LANDMARKS]


def _frame(a, n):
    n = n / np.linalg.norm(n)
    a = a - (a @ n) * n
    a = a / np.linalg.norm(a)
    return np.stack([a, np.cross(n, a), n])  # rows a, l, n


def template_frames():
    z = np.load(SMPLX_NEUTRAL)
    V = z["v_template"]
    J = z["J_regressor"] @ V
    lab = z["weights"].argmax(1)
    out = {}
    for side, ix in JOINTS.items():
        w = J[ix["wrist"]]
        raw = np.cross(J[ix["index1"]] - w, J[ix["pinky1"]] - w)
        sign = -1.0 if raw[1] > 0 else 1.0  # palm side = down in the T-pose template (table_shared.palm_vertex_ids)
        F = _frame(J[ix["middle1"]] - w, sign * raw)
        hv = np.flatnonzero(np.isin(lab, [ix["wrist"], *ix["fingers"]]))
        C = (V[hv] - w) @ F.T
        lo, hi = C.min(0), C.max(0)
        pts = dict(Index1=J[ix["index1"]], Pinky1=J[ix["pinky1"]], MiddleTip=V[TIP_VERTEX[side]], Thumb1=J[ix["thumb1"]])
        u = {k: ((p - w) @ F.T - lo) / (hi - lo) for k, p in pts.items()}
        out[side] = dict(sign=sign, frame=F, u=u, bbox_lo=lo, bbox_hi=hi)
    return out


def robot_offsets(hull_vertices, side, tf=None):
    """Landmark offsets in the rubber-hand link frame from the simulator hull vertices."""
    tf = tf or template_frames()
    F = _frame(np.array([1.0, 0.0, 0.0]), PALM_NORMAL_LOCAL[side])
    C = np.asarray(hull_vertices, float) @ F.T
    lo, hi = C.min(0), C.max(0)
    u = tf[side]["u"]
    off = {k: (lo + u[k] * (hi - lo)) @ F for k in LANDMARKS}
    # thumb on the same side: human Thumb1 vs the robot thumb marker, l coordinate relative to the bbox middle
    human_side = np.sign(u["Thumb1"][1] - 0.5)
    robot_side = np.sign(THUMB_MARKER_LOCAL @ F[1] - 0.5 * (lo[1] + hi[1]))
    assert human_side == robot_side, (side, u["Thumb1"], THUMB_MARKER_LOCAL @ F[1], lo[1], hi[1])
    return off, dict(frame_rows_a_l_n=F.round(6).tolist(), hull_bbox_lo=lo.round(6).tolist(), hull_bbox_hi=hi.round(6).tolist(),
                     u={k: np.round(v, 4).tolist() for k, v in u.items()})


def human_points(joints, vertices, side):
    """[T, 3 (Index1, Pinky1, MiddleTip), 3] from posed SMPL-X joints [T, >=55, 3] and vertices [T, 10475, 3]."""
    ix = JOINTS[side]
    return np.stack([joints[:, ix["index1"]], joints[:, ix["pinky1"]], vertices[:, TIP_VERTEX[side]]], 1)


def human_directions(joints, vertices, side, tf=None):
    """Per frame: palm normal, finger direction (wrist -> middle tip), hand long axis (wrist -> Middle1); unit vectors."""
    tf = tf or template_frames()
    ix = JOINTS[side]
    w = joints[:, ix["wrist"]]
    n = tf[side]["sign"] * np.cross(joints[:, ix["index1"]] - w, joints[:, ix["pinky1"]] - w)
    unit = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)
    return dict(palm_normal=unit(n), finger=unit(vertices[:, TIP_VERTEX[side]] - w), long_axis=unit(joints[:, ix["middle1"]] - w))


def robot_directions(R, side):
    """R [..., 3, 3] rubber-hand link rotations -> palm normal and finger direction (unit, world)."""
    F = _frame(np.array([1.0, 0.0, 0.0]), PALM_NORMAL_LOCAL[side])
    return dict(palm_normal=R @ F[2], finger=R @ F[0])


if __name__ == "__main__":
    import json
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "system1/t12v2"))
    import contact_stats as cs
    shapes = dict(cs.dump_shapes("table", "/World/envs/env_0/Robot/")[0])
    tf = template_frames()
    for side in ("left", "right"):
        off, rec = robot_offsets(shapes[f"{side}_rubber_hand_link"].vertices, side, tf)
        print(side, json.dumps({k: np.round(v, 4).tolist() for k, v in off.items()}), json.dumps(rec["u"]))
