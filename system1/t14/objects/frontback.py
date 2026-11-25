"""T14 step 9 (Kevin 2026-11-24 18:40, item 3): the "front-back" partner of a class-a (carry and walk forward) demonstration;
the partner in front walks backwards. Built in the solver's (preprocessed, person-scaled) world, on the clip = the carry
episode of the screen (lift episode start .. stop).

* Walking direction = the person's pelvis net horizontal displacement over the clip. Lengthening axis k = the horizontal box
  axis (box frame at the clip's first frame) closest to it, sign sigma so that +sigma e_k points forward. The target = the box
  lengthened along k only (width and height kept) to L = m x 2 w (1.0 x = two copies end to end); role 0's region = the
  source box itself (its end), the target extends forward; target centre c_T = sigma (L / 2 - w / 2) e_k in the box frame;
  target frame = box frame moved to c_T and turned by R_t (identity for sigma = +1, 180 deg about the box's vertical for
  sigma = -1), so that role 0's region is always at the -e_k end of the target (the step-8 level rule partner.region_maps:
  role 0 keeps the -e_k end, role 1 the +e_k end). The box is symmetric, so R_t changes no geometry.
* Partner region map: T180 = rotation by 180 deg about the target's vertical axis through its centre (role 0's region -> the
  far end, facing back). In the step-8 notation M0(x) = R_t^T (x - c_T), M1(x) = R180 M0(x); the candidate record holds the
  level-1.0 maps (Q0 = R_t^T, c0 = -w/2 e_k, Q1 = R180 R_t^T, c1 = +w/2 e_k) and partner.region_maps gives level m.
* Object: role 0's view A(t) = O(t) Trans(c_T) R_t; role 1's view B(t) = S D0(t) S^-1 A(0) with S = the 180-deg map at frame 0.
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

Step 9c options (Kevin 2026-11-25 01:52; defaults = the step-9b construction above):
  turn="world"  the partner's 180-deg turn S is about the WORLD vertical through the target centre at frame 0 (gravity
                deviation 0): S = Trans(c) Rz180 Trans(-c), c = A(0) origin; in the target frame T_w = A(0)^-1 S A(0) replaces
                T180 in role 1's object view B(t) = S D0(t) S^-1 A(0) and in the upper-body map G(t) = P(t) T_w A(t)^-1. The hand
                targets (region maps M0, M1 of the candidate) stay the box's own symmetry (M1 = R180 M0); the small angle between
                the partner's hands and the box face is left to the palm-orientation term.
  legs="walk"   only the walking segment of the demonstration is reversed: walking segment [w0, w1) = first swing start to
                last swing end (screen_omomo.steps_in swing rule: toe horizontal speed > 0.25 m/s after Gaussian sigma 2,
                >= 3 frames, displacement > 0.10 m; SMPL-X toes 10 / 11 of the OMOMO original release, the clip's frames).
                Partner legs at clip frame t = demonstration legs at frame clip(w0 + w1 - 1 - t, w0, w1 - 1), turned by 180 deg
                about the vertical, + the least-squares horizontal offset (over the clip, as before): reversed inside the
                walking segment, and before / after it the partner stands still with the legs of its boundary frame (pose and
                position).
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


def swing_segments(toe_full, s, e):
    """screen_omomo.steps_in's swing segments of the clip [s, e) (sequence frames), returned as clip-relative [a, b) per foot.
    toe_full [T_seq, 2, 3] = SMPL-X joints 10 / 11 (OMOMO original release, real scale)."""
    import screen_omomo as so
    from scipy.ndimage import gaussian_filter1d
    R_ = so.RULES
    x = gaussian_filter1d(np.asarray(toe_full, float)[..., :2], R_["step_sigma_frames"], axis=0, mode="nearest")
    v = np.zeros(x.shape[:2])
    v[1:] = np.linalg.norm(np.diff(x, axis=0), axis=-1) * R_["fps"]
    out = []
    for k in range(2):
        for a, b in so.runs(v[s:e, k] > R_["step_speed_mps"]):
            if b - a >= R_["step_min_frames"] and np.linalg.norm(x[s + b - 1, k] - x[s + a, k]) > R_["step_disp_m"]:
                out.append(dict(foot=("left", "right")[k], start=int(a), stop=int(b)))
    return sorted(out, key=lambda d: d["start"])


def walking_segment(v, s, e):
    """clip-relative walking segment [w0, w1) of an OMOMO sequence dict v (omomo_motion.load_sequences) and its swings."""
    import omomo_motion as om
    sw = swing_segments(om.smplx_joints(v)[:, [10, 11]], s, e)
    if not sw:
        return None
    return dict(w0=min(d["start"] for d in sw), w1=max(d["stop"] for d in sw), swings=sw)


def build(human0, names, O, size, m, turn="box", legs="all", walk=None):
    """human0 [T, J, 3] role 0 (solver world), names [J], O [T, 4, 4] own object (box frame) poses, size [3] box (solver units),
    m level. Returns dict with P, A, cand (Q0, c0, Q1, c1, axis, L1, Lm), human1, rot1 [T, 3, 3] (hands' rotation), D, k, sigma.
    turn: "box" (step 9b) or "world" (step 9c); legs: "all" (step 9b) or "walk" (step 9c, walk = walking_segment(...))."""
    T = len(O)
    il = [names.index(n) for n in LOWER]
    iu = [i for i in range(len(names)) if i not in il]
    pel = human0[:, names.index("Pelvis")]
    k, sigma, ang, wd = walking_axis(pel, O[0])
    w = float(size[k])
    L1, Lm = 2 * w, m * 2 * w
    e = np.eye(3)[k]
    cT = sigma * (Lm / 2 - w / 2) * e
    R180 = rz180()
    Rt = np.eye(3) if sigma > 0 else R180
    A = O @ affine(Rt, cT)
    if turn == "world":
        c = A[0][:3, 3]
        S = affine(R180, c - R180 @ c)  # world vertical through the target centre at frame 0
        T180 = np.linalg.inv(A[0]) @ S @ A[0]  # the same turn in the target frame
    else:
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
    if legs == "walk":
        w0, w1 = int(walk["w0"]), int(walk["w1"])
        idx = np.clip(w0 + w1 - 1 - np.arange(T), w0, w1 - 1)  # reversed inside [w0, w1), held before / after
        rev = human0[idx][:, il]
    else:
        idx = np.arange(T)[::-1]
        rev = human0[::-1][:, il]  # legs played backwards
    Rw = R180
    leg_xyz = rev @ Rw.T
    pel_att = np.einsum("tij,tj->ti", G[:, :3, :3], pel) + G[:, :3, 3]
    off = (pel_att[:, :2] - leg_xyz[:, 0, :2]).mean(0)  # least-squares horizontal offset (index 0 of LOWER = Pelvis)
    leg_xyz[..., :2] += off
    human1[:, il] = leg_xyz
    D = leg_xyz[:, 0, :2] - pel_att[:, :2]
    Q0, Q1 = Rt.T, R180 @ Rt.T
    cand = dict(Q0=Q0.tolist(), c0=(-w / 2 * e).tolist(), Q1=Q1.tolist(), c1=(w / 2 * e).tolist(), axis=k, L1=L1, parity0=1, parity1=1,
                id=f"front-back axis{k} sigma{sigma:+.0f}")
    M0 = affine(Q0, -Rt.T @ cT)  # level-m maps (box frame -> target frame)
    M1 = affine(Q1, -Q1 @ cT)
    return dict(P=P, A=A, B=B, S=S, G=G, cand=cand, M0=M0, M1=M1, human1=human1, rot1=G[:, :3, :3], D=D, k=k, sigma=sigma, walk_axis_angle_deg=ang,
                walk_dir=wd, L1=L1, Lm=Lm, cT=cT, Rt=Rt, leg_offset=off, turn=turn, legs=legs, walk=walk, leg_source_frame=idx)


def box_sd(p, M, size):
    """signed distance of points p [N, 3] to the oriented box of full size `size` at pose M (negative inside)."""
    q = (p - M[:3, 3]) @ M[:3, :3]
    d = np.abs(q) - np.asarray(size) / 2
    return np.linalg.norm(np.maximum(d, 0), axis=-1) + np.minimum(d.max(-1), 0)


LEG_POINTS = ("L_Knee", "L_Ankle", "L_Toe", "R_Knee", "R_Ankle", "R_Toe")


def leg_box(human0, names, O, size, C):
    """per frame: smallest signed distance of the leg joints (knees, ankles, toes) to the person's own box; role 1 = the
    partner vs G O, role 0 = the demonstration vs O. Summary: minimum, its frame, frames inside, first / last 5 frames."""
    il = [names.index(n) for n in LEG_POINTS]
    T = len(O)
    d1 = np.array([box_sd(C["human1"][t, il], C["G"][t] @ O[t], size).min() for t in range(T)])
    d0 = np.array([box_sd(human0[t, il], O[t], size).min() for t in range(T)])
    f = lambda d: dict(min_m=round(float(d.min()), 4), min_frame=int(d.argmin()), frames_inside=int((d < 0).sum()),  # noqa: E731
                       first5_m=np.round(d[:5], 3).tolist(), last5_m=np.round(d[-5:], 3).tolist())
    return dict(partner=f(d1), demo=f(d0)), d1, d0


def diag(D, s):
    d = np.linalg.norm(D, axis=1)
    return dict(pelvis_mismatch_max_m=round(float(d.max()), 4), pelvis_mismatch_mean_m=round(float(d.mean()), 4),
                pelvis_mismatch_max_person_scale_m=round(float(d.max()) / s, 4))
