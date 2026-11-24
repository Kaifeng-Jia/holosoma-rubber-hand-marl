"""T14 step 9 (Kevin 2026-11-24 18:40, item 3): the "front-back" partner of a class-a (carry and walk forward) demonstration;
the partner in front walks backwards. Built in the solver's (preprocessed, person-scaled) world, on the clip = the carry
episode of the screen (lift episode start .. stop).

* Walking direction = the person's pelvis net horizontal displacement over the clip. Lengthening axis k = the horizontal box
  axis (box frame at the clip's first frame) closest to it, sign sigma so that +sigma e_k points forward. The target = the box
  lengthened along k only (width and height kept) to L = m x 2 w (1.0 x = two copies end to end); role 0's region = the
  source box itself (its end), the target extends forward; target centre c_T = sigma (L / 2 - w / 2) e_k in the box frame;
  target frame = box frame moved to c_T.
* Partner region map: T180 = rotation by 180 deg about the target's vertical axis through its centre (role 0's region -> the
  far end, facing back). In the step-8 notation M0(x) = x - c_T, M1(x) = R180 (x - c_T).
* Object: role 0's view A(t) = O(t) Trans(c_T); role 1's view B(t) = S D0(t) S^-1 A(0) with S = the 180-deg map at frame 0.
  Shared P(t): translation = A(t)'s (the demonstration's; both walk the same way), rotation = slerp midpoint of A and B
  (one-sided tilts removed).
* Partner person (role 1):
  - upper body (every joint except Pelvis, hips, knees, ankles, toes) and palms follow the corresponding object locations:
    x1(t) = P(t) T180 A(t)^-1 x0(t);
  - legs (Pelvis, hips, knees, ankles, toes) = the demonstration's legs played backwards in time over the clip, turned by
    180 deg about the vertical (backward walking, CooHOI-style reversed carry), translated horizontally by the least-squares
    offset that puts the reversed pelvis on the object-attached pelvis (heights unchanged: the feet stay on the ground);
  - contact flags: role 0's at the same time (the hands follow the object in time); palm targets rotated with the hands;
  - support: from its own (reversed) toes (role_solve's support rule).
  Diagnostic (reported, not used): D(t) = reversed-leg pelvis - object-attached pelvis, horizontal.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

LOWER = ("Pelvis", "L_Hip", "L_Knee", "L_Ankle", "L_Toe", "R_Hip", "R_Knee", "R_Ankle", "R_Toe")


def rz180():
    return np.diag([-1.0, -1.0, 1.0])


def affine(R, t):
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = R, t
    return M


def walking_axis(pelvis, O0):
    wd = pelvis[-1, :2] - pelvis[0, :2]
    wd = wd / np.linalg.norm(wd)
    best = None
    for k in (0, 1):
        ax = O0[:3, k][:2]
        n = np.linalg.norm(ax)
        c = float(ax @ wd / n)
        if best is None or abs(c) > abs(best[1]):
            best = (k, c)
    k, c = best
    return k, (1.0 if c > 0 else -1.0), float(np.degrees(np.arccos(min(1.0, abs(c))))), wd


def build(human0, names, O, size, m):
    """human0 [T, J, 3] role 0 (solver world), names [J], O [T, 4, 4] own object (box frame) poses, size [3] box (solver units),
    m level. Returns dict with P, A, cand (Q0, c0, Q1, c1, axis, L1, Lm), human1, rot1 [T, 3, 3] (hands' rotation), D, k, sigma."""
    T = len(O)
    il = [names.index(n) for n in LOWER]
    iu = [i for i in range(len(names)) if i not in il]
    pel = human0[:, names.index("Pelvis")]
    k, sigma, ang, wd = walking_axis(pel, O[0])
    w = float(size[k])
    L1, Lm = 2 * w, m * 2 * w
    e = np.eye(3)[k]
    cT = sigma * (Lm / 2 - w / 2) * e
    A = O @ affine(np.eye(3), cT)
    R180 = rz180()
    T180 = affine(R180, np.zeros(3))
    S = A[0] @ T180 @ np.linalg.inv(A[0])
    D0 = O @ np.linalg.inv(O[0])
    B = S @ D0 @ np.linalg.inv(S) @ A[0]
    P = A.copy()
    for t in range(T):
        P[t, :3, :3] = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([A[t, :3, :3], B[t, :3, :3]])))([0.5]).as_matrix()[0]
    G = P @ T180 @ np.linalg.inv(A)  # world -> world map carrying role 0's object-attached points to the partner
    human1 = np.zeros_like(human0)
    human1[:, iu] = np.einsum("tij,tkj->tki", G[:, :3, :3], human0[:, iu]) + G[:, None, :3, 3]
    rev = human0[::-1][:, il]  # legs played backwards
    Rw = R180
    legs = rev @ Rw.T
    pel_att = np.einsum("tij,tj->ti", G[:, :3, :3], pel) + G[:, :3, 3]
    off = (pel_att[:, :2] - legs[:, 0, :2]).mean(0)  # least-squares horizontal offset (index 0 of LOWER = Pelvis)
    legs[..., :2] += off
    human1[:, il] = legs
    D = legs[:, 0, :2] - pel_att[:, :2]
    Q0, c0 = np.eye(3), -cT
    Q1, c1 = R180, -R180 @ cT
    cand = dict(Q0=Q0.tolist(), c0=c0.tolist(), Q1=Q1.tolist(), c1=c1.tolist(), axis=k, L1=L1, parity0=1, parity1=1, id=f"front-back axis{k} sigma{sigma:+.0f}")
    return dict(P=P, A=A, B=B, S=S, G=G, cand=cand, human1=human1, rot1=G[:, :3, :3], D=D, k=k, sigma=sigma, walk_axis_angle_deg=ang, walk_dir=wd,
                L1=L1, Lm=Lm, cT=cT, leg_offset=off)


def diag(D, s):
    d = np.linalg.norm(D, axis=1)
    return dict(pelvis_mismatch_max_m=round(float(d.max()), 4), pelvis_mismatch_mean_m=round(float(d.mean()), 4),
                pelvis_mismatch_max_person_scale_m=round(float(d.max()) / s, 4))
