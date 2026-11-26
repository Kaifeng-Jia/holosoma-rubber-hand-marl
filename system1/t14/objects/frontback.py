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
  legs="walk_scopy"  (step 9d, Kevin 2026-11-25 03:50) the walking segment exactly as legs="walk"; before it (lift) and after it
                (put down) the partner's legs (Pelvis, hips, knees, ankles, toes) = the S copy of the demonstration's legs at the
                same frame (its knee bend and stepping), moved horizontally as a whole so that its feet meet the reversed walking
                segment: pre = S x(t) + d_pre, d_pre = least-squares horizontal offset of the four foot points (ankles, toes) of
                S x(w0 - 1) onto the walking legs at w0; post = S x(t) + d_post with S x(w1) onto the walking legs at w1 - 1. The
                upper body stays object-attached (G), as in every variant.
  legs="full_scopy"  (step 9e, Kevin 2026-11-25 15:10) the walking segment exactly as legs="walk_scopy" (reversed legs,
                object-attached upper body); before it (lift) and after it (put down) the WHOLE partner person (every joint,
                upper body and palms included) = the S copy of the demonstration at the same frame, moved by the same d_pre /
                d_post of legs="walk_scopy" (four foot points, least squares). The partner's own object moves with its person:
                person map Gp(t) = G(t) inside the walking segment, Trans(d_pre) S before it, Trans(d_post) S after it; own
                object Gp O, palm targets rotated by Gp (role_solve), so that in lift / put down the S copy holds its own box as
                the demonstration holds the source box. For every other option Gp = G.
  legs="reverse_warp"  (step 9f, Kevin 2026-11-25 17:10; turn="world" only) the partner = the WHOLE demonstration played
                backwards in time (every joint, palms included; one source, no splicing), turned by S, plus ONE constant horizontal
                offset d. Time alignment: demonstration segments lift [0, w0), walk [w0, w1), put down [w1, T); the reversed
                put-down / walk / lift are each stretched or compressed linearly onto the demonstration's lift / walk / put-down
                frames: partner frame t <- demonstration time tau(t) (fractional; segment end frames map onto end frames;
                positions interpolated linearly, rotations by slerp; an empty source segment holds its boundary frame).
                Partner's own box own1(t) = Trans(d) S O(tau(t)); role 1's object view B(t) = S A(tau(t)) T_w^-1 (rotation),
                shared P(t): translation = A(t)'s, rotation = slerp midpoint of A(t) and B(t) (as step 9c). d = least-squares
                horizontal offset putting own1 on the far end of the shared target: mean_t (P(t) M1 - S O(tau(t))) origin, xy.
                Hand targets: the region maps M0 / M1 of the box's own symmetry (as step 9c). Gp = own1 O^-1 (own box = Gp O).
  legs="mirror"  (step 9g, Kevin 2026-11-26 01:40, task book section 12 step 9g: upper body spatially symmetric, lower body
                temporally symmetric; turn is not used). Walking direction e = the demonstration's pelvis displacement over the walking
                segment, horizontal. Mirror plane {x : x.e = p(t)} perpendicular to e; p(t) = p0 + (pelvis(clip(t, w0, w1-1)) -
                pelvis(w0)).e: inside the walking segment it moves forward with the demonstration's pelvis by the same distance, fixed
                before / after it; p0 = least squares (over the clip) of the target centre attached to the demonstration's box (the
                lengthened target's centre). Mirror Sm(t): x -> Ref x + 2 p(t) e, Ref = I - 2 e e^T (improper).
                Rule 1 (upper body): the partner = Sm(t) applied to the demonstration at the SAME frame, left / right exchanged
                (every joint and the palms; pelvis and hips too).
                Rule 2 (lower body): walking segment: ankles and toes = the demonstration's walking legs played backwards with the
                playback set by walked distance (distance_tau: x(tau) = x(w0) + x(w1-1) - x(t), x = pelvis along e), mirrored about
                the fixed plane p_f = p0 + (x(w1-1) - x(w0)) / 2 (then the reversed walk's pelvis meets the mirrored pelvis along e
                at every frame), left / right exchanged; lift / put down: ankles and toes held at the reversed walk's first / last
                stance (source frames tau(w0) = w1-1 / tau(w1-1) = w0); pelvis and hips = the mirror (rule 1); knees in every
                segment by two-segment IK from the mirrored hip to the ankle (the demonstration's own thigh / shank lengths; bend
                side = the reversed knee in the walking segment, the mirrored knee in lift / put down). Overextension (hip-ankle
                distance > thigh + shank) is counted in diag9g.
                Rule 3 (box): the target = the demonstration's box levelled (long axis k along e, its frame-0 vertical axis vertical,
                centre kept; Ob), lengthened to m x 2 w; shared P(t): rotation constant (level), centre = the attached target centre
                with its coordinate along e replaced by p(t) (on the plane; lateral position and height = the demonstration's).
                Hand targets: M1 = (the target's own mirror about its centre plane) M0 (improper; partner hands exchanged).
                Partner's own box = Sm(t) Ob(t) (role_solve appends the local flip F as for every improper map); role 0's own box in
                the solver = Ob (level), so that both person-side frames differ from the shared target by a translation only.
Every variant also returns r1_source: per partner frame t the source frame and the rigid map (R, t) that carried the
demonstration's legs there (reversed / held legs: R180 and the least-squares offset; S copies: S and d_pre / d_post), for
the support-drift source of the partner (step9_eval.py --r1-source construction).
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


def warp_tau(T, w0, w1):
    """step 9f: partner frame t -> demonstration time tau(t) (float). Partner segments [0, w0) / [w0, w1) / [w1, T) take the
    demonstration's reversed put-down [T-1 .. w1] / reversed walk [w1-1 .. w0] / reversed lift [w0-1 .. 0], each mapped linearly
    end frame onto end frame (one partner frame: the middle of the source range; empty source range: its boundary frame held)."""
    tau = np.zeros(T)
    segs = []
    for (a0, a1), (hi, lo), name in (((0, w0), (T - 1, w1), "lift <- reversed put down"), ((w0, w1), (w1 - 1, w0), "walk <- reversed walk"),
                                     ((w1, T), (w0 - 1, 0), "put down <- reversed lift")):
        La, Lb = a1 - a0, hi - lo + 1
        if La <= 0:
            segs.append(dict(segment=name, partner_frames=0, source_frames=max(Lb, 0), scale=None))
            continue
        if Lb <= 0:  # no source frames (e.g. no lift in the demonstration): hold the boundary frame
            tau[a0:a1] = min(max(lo, 0), T - 1) if name.startswith("put") else max(min(hi, T - 1), 0)
            segs.append(dict(segment=name, partner_frames=La, source_frames=0, scale=None, note="source segment empty: boundary frame held"))
            continue
        tau[a0:a1] = (hi + lo) / 2.0 if La == 1 else hi - np.arange(La) * (hi - lo) / (La - 1)
        segs.append(dict(segment=name, partner_frames=La, source_frames=Lb, scale=round(La / Lb, 4),
                         note="partner frames / source frames (>1 stretched, <1 compressed)"))
    return tau, segs


def lerp_at(X, tau):
    """X [T, ...] at fractional times tau [K] (linear)."""
    X = np.asarray(X, float)
    i0 = np.clip(np.floor(tau).astype(int), 0, len(X) - 1)
    i1 = np.clip(i0 + 1, 0, len(X) - 1)
    f = (tau - i0).reshape(-1, *([1] * (X.ndim - 1)))
    return X[i0] * (1 - f) + X[i1] * f


def slerp_at(Rm, tau):
    """rotation matrices Rm [T, ..., 3, 3] at fractional times tau [K] (slerp per element)."""
    Rm = np.asarray(Rm, float)
    T = len(Rm)
    i0 = np.clip(np.floor(tau).astype(int), 0, T - 1)
    i1 = np.clip(i0 + 1, 0, T - 1)
    f = tau - i0
    out = np.empty((len(tau),) + Rm.shape[1:])
    flat = Rm.reshape(T, -1, 3, 3)
    for k in range(len(tau)):
        for j in range(flat.shape[1]):
            if i0[k] == i1[k] or f[k] == 0:
                out.reshape(len(tau), -1, 3, 3)[k, j] = flat[i0[k], j]
            else:
                r = Rotation.from_matrix(np.stack([flat[i0[k], j], flat[i1[k], j]]))
                out.reshape(len(tau), -1, 3, 3)[k, j] = Slerp([0.0, 1.0], r)([f[k]]).as_matrix()[0]
    return out


def poses_at(M, tau):
    """4x4 poses M [T, 4, 4] at fractional times (translation linear, rotation slerp)."""
    out = np.tile(np.eye(4), (len(tau), 1, 1))
    out[:, :3, :3] = slerp_at(M[:, :3, :3], tau)
    out[:, :3, 3] = lerp_at(M[:, :3, 3], tau)
    return out


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
    if legs == "reverse_warp":
        if turn != "world":
            raise ValueError("legs='reverse_warp' is defined with turn='world'")
        return _reverse_warp(human0, names, O, size, m, walk, A, S, T180, k, sigma, ang, wd, w, e, L1, Lm, cT, Rt, R180, pel)
    if legs == "mirror":
        return _mirror_9g(human0, names, O, size, m, walk, k, sigma, ang, wd, w, L1, Lm, Rt, R180, pel)
    D0 = O @ np.linalg.inv(O[0])
    B = S @ D0 @ np.linalg.inv(S) @ A[0]
    P = A.copy()
    for t in range(T):
        P[t, :3, :3] = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([A[t, :3, :3], B[t, :3, :3]])))([0.5]).as_matrix()[0]
    G = P @ T180 @ np.linalg.inv(A)  # world -> world map carrying role 0's object-attached points to the partner
    human1 = np.zeros_like(human0)
    human1[:, iu] = np.einsum("tij,tkj->tki", G[:, :3, :3], human0[:, iu]) + G[:, None, :3, 3]
    if legs in ("walk", "walk_scopy", "full_scopy"):
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
    d_pre = d_post = None
    Gp = G.copy()  # the partner person's map (its own object moves with it); = G except legs="full_scopy" lift / put down
    src_frame = idx.copy()
    src_R = np.repeat(Rw[None], T, 0)
    src_t = np.zeros((T, 3))
    src_t[:, :2] = off
    if legs in ("walk_scopy", "full_scopy"):
        feet = [LOWER.index(n) for n in ("L_Ankle", "L_Toe", "R_Ankle", "R_Toe")]
        Sx_all = human0 @ S[:3, :3].T + S[:3, 3]  # the S copy of the demonstration, every frame
        Sx = Sx_all[:, il]  # its legs
        segs = []
        if w0 > 0:
            d_pre = (leg_xyz[w0, feet, :2] - Sx[w0 - 1, feet, :2]).mean(0)
            leg_xyz[:w0] = Sx[:w0]
            leg_xyz[:w0, :, :2] += d_pre
            segs.append((slice(0, w0), d_pre))
        if w1 < T:
            d_post = (leg_xyz[w1 - 1, feet, :2] - Sx[w1, feet, :2]).mean(0)
            leg_xyz[w1:] = Sx[w1:]
            leg_xyz[w1:, :, :2] += d_post
            segs.append((slice(w1, T), d_post))
        for sl, d in segs:
            Md = affine(np.eye(3), np.r_[d, 0.0]) @ S
            src_frame[sl] = np.arange(T)[sl]
            src_R[sl], src_t[sl] = Md[:3, :3], Md[:3, 3]
            if legs == "full_scopy":
                human1[sl] = Sx_all[sl]
                human1[sl, :, :2] += d
                Gp[sl] = Md
    human1[:, il] = leg_xyz
    D = leg_xyz[:, 0, :2] - pel_att[:, :2]
    Q0, Q1 = Rt.T, R180 @ Rt.T
    cand = dict(Q0=Q0.tolist(), c0=(-w / 2 * e).tolist(), Q1=Q1.tolist(), c1=(w / 2 * e).tolist(), axis=k, L1=L1, parity0=1, parity1=1,
                id=f"front-back axis{k} sigma{sigma:+.0f}")
    M0 = affine(Q0, -Rt.T @ cT)  # level-m maps (box frame -> target frame)
    M1 = affine(Q1, -Q1 @ cT)
    return dict(P=P, A=A, B=B, S=S, G=G, Gp=Gp, cand=cand, M0=M0, M1=M1, human1=human1, rot1=Gp[:, :3, :3], D=D, k=k, sigma=sigma, walk_axis_angle_deg=ang,
                r1_source=dict(frame=src_frame, R=src_R, t=src_t),
                walk_dir=wd, L1=L1, Lm=Lm, cT=cT, Rt=Rt, leg_offset=off, turn=turn, legs=legs, walk=walk, leg_source_frame=idx,
                pre_offset=None if d_pre is None else d_pre.tolist(), post_offset=None if d_post is None else d_post.tolist())


def _reverse_warp(human0, names, O, size, m, walk, A, S, T180, k, sigma, ang, wd, w, e, L1, Lm, cT, Rt, R180, pel):
    """step 9f construction (legs="reverse_warp"); see the module docstring."""
    T = len(O)
    w0, w1 = int(walk["w0"]), int(walk["w1"])
    tau, segs = warp_tau(T, w0, w1)
    Ow = poses_at(O, tau)  # demonstration box at tau(t)
    Aw = poses_at(A, tau)
    Rs = S[:3, :3]
    B = A.copy()
    B[:, :3, :3] = np.einsum("ij,tjk,kl->til", Rs, Aw[:, :3, :3], T180[:3, :3].T)  # role 1's object view in role 0's convention
    P = A.copy()
    for t in range(T):
        P[t, :3, :3] = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([A[t, :3, :3], B[t, :3, :3]])))([0.5]).as_matrix()[0]
    Q0, Q1 = Rt.T, R180 @ Rt.T
    cand = dict(Q0=Q0.tolist(), c0=(-w / 2 * e).tolist(), Q1=Q1.tolist(), c1=(w / 2 * e).tolist(), axis=k, L1=L1, parity0=1, parity1=1,
                id=f"front-back axis{k} sigma{sigma:+.0f}")
    M0 = affine(Q0, -Rt.T @ cT)
    M1 = affine(Q1, -Q1 @ cT)
    SO = S @ Ow  # the S copy of the demonstration's box at tau(t)
    far = P @ M1  # the far end of the shared target as a box pose
    d = (far[:, :2, 3] - SO[:, :2, 3]).mean(0)  # one constant horizontal offset (least squares)
    Md = affine(np.eye(3), np.r_[d, 0.0]) @ S
    own1 = Md @ Ow
    human1 = lerp_at(human0, tau) @ Rs.T + Md[:3, 3]
    Gp = own1 @ np.linalg.inv(O)
    G_att = P @ T180 @ np.linalg.inv(A)  # step-9c object-attached map, diagnostic only
    pel_att = np.einsum("tij,tj->ti", G_att[:, :3, :3], pel) + G_att[:, :3, 3]
    D = human1[:, names.index("Pelvis"), :2] - pel_att[:, :2]
    return dict(P=P, A=A, B=B, S=S, G=Gp, Gp=Gp, cand=cand, M0=M0, M1=M1, human1=human1, rot1=np.repeat(Rs[None], T, 0), D=D, k=k, sigma=sigma,
                walk_axis_angle_deg=ang, walk_dir=wd, L1=L1, Lm=Lm, cT=cT, Rt=Rt, leg_offset=d, turn="world", legs="reverse_warp", walk=walk,
                leg_source_frame=tau, pre_offset=None, post_offset=None, tau=tau, warp_segments=segs, own1=own1, offset_d=d.tolist(),
                own_vs_far_end_m=np.linalg.norm(own1[:, :3, 3] - far[:, :3, 3], axis=1),
                r1_source=dict(frame=tau, R=np.repeat(Rs[None], T, 0), t=np.repeat(Md[:3, 3][None], T, 0)))


def swap_lr(names):
    """names with L_ / R_ exchanged (the partner's left joint = the mirrored demonstration's right joint)."""
    return [("R_" + n[2:]) if n.startswith("L_") else (("L_" + n[2:]) if n.startswith("R_") else n) for n in names]


def distance_tau(x, w0, w1):
    """step 9g rule 2 (Kevin 2026-11-26): walking-segment source times by walked distance. x [T] = the demonstration's pelvis
    along the walking direction; inside [w0, w1) made non-decreasing (running maximum). Partner frame t in [w0, w1) <- the
    demonstration time tau(t) with x(tau) = x(w0) + x(w1 - 1) - x(t) (the reversed walk has covered the same distance as the
    demonstration), fractional, linear inverse; tau(w0) = w1 - 1, tau(w1 - 1) = w0. Returns tau [w1 - w0], xs (the monotone x)."""
    xs = np.maximum.accumulate(np.asarray(x, float)[w0:w1])
    n = w1 - w0
    if n == 1:
        return np.array([float(w0)]), xs
    xs_ = xs + 1e-9 * np.arange(n)  # strictly increasing for the inverse (flat runs: the first frame of the run)
    tgt = xs_[0] + xs_[-1] - xs_
    return w0 + np.interp(tgt, xs_, np.arange(n, dtype=float)), xs


def leg_ik(H, A, a, b, K0):
    """two-segment leg IK per frame: hip H [T, 3], ankle A [T, 3], thigh a, shank b (scalars), bend hint K0 [T, 3] (the knee lies
    in the plane of H, A, K0, on K0's side). Hip-ankle distance >= a + b (overextension): the knee on the line H-A at the
    fraction a / (a + b), bones stretched; counted. Returns K [T, 3], over [T] bool, excess [T] = |A - H| - (a + b)."""
    dv = A - H
    d = np.linalg.norm(dv, axis=1)
    u = dv / np.maximum(d, 1e-9)[:, None]
    h = K0 - H
    n = h - np.sum(h * u, axis=1)[:, None] * u
    nn = np.linalg.norm(n, axis=1)
    fallback = np.cross(u, np.array([0.0, 0.0, 1.0]))
    n = np.where(nn[:, None] > 1e-9, n / np.maximum(nn, 1e-12)[:, None], fallback / np.maximum(np.linalg.norm(fallback, axis=1), 1e-12)[:, None])
    over = d >= a + b
    along = np.where(over, d * a / (a + b), (a * a - b * b + d * d) / (2 * np.maximum(d, 1e-9)))
    r = np.where(over, 0.0, np.sqrt(np.maximum(a * a - along * along, 0.0)))
    return H + along[:, None] * u + r[:, None] * n, over, d - (a + b)


def _mirror_9g(human0, names, O, size, m, walk, k, sigma, ang, wd, w, L1, Lm, Rt, R180, pel):
    """step 9g construction (legs="mirror"); see the module docstring."""
    T = len(O)
    w0, w1 = int(walk["w0"]), int(walk["w1"])
    if w1 - w0 < 2:
        raise ValueError("walking segment shorter than 2 frames")
    dv = pel[w1 - 1, :2] - pel[w0, :2]
    e2 = dv / np.linalg.norm(dv)
    e = np.r_[e2, 0.0]  # walking direction: the demonstration's pelvis over the walking segment, horizontal
    ang_e_clip = float(np.degrees(np.arccos(np.clip(e2 @ wd, -1, 1))))
    R0 = O[0][:3, :3]
    v = int(np.argmax(np.abs(R0[2, :])))  # the box axis closest to the vertical at frame 0
    if v == k:
        raise ValueError("lengthening axis is the vertical box axis")
    u_ = 3 - k - v
    Rb = np.zeros((3, 3))
    Rb[:, k] = sigma * e  # +sigma e_k forward, as the lengthening rule
    Rb[:, v] = np.sign(R0[2, v]) * np.array([0.0, 0.0, 1.0])
    Rb[:, u_] = np.cross(Rb[:, (u_ + 1) % 3], Rb[:, (u_ + 2) % 3])
    assert abs(np.linalg.det(Rb) - 1) < 1e-9
    Ob = O.copy()
    Ob[:, :3, :3] = Rb  # the demonstration's box levelled (rule 3: level, long axis along the walking direction), centre kept
    ek = np.eye(3)[k]
    cT = sigma * (Lm / 2 - w / 2) * ek
    A = Ob @ affine(Rt, cT)  # the level target attached to the demonstration's box (role 0's region at the target's -e_k end)
    RA = A[0, :3, :3].copy()
    xp = pel @ e
    tt = np.clip(np.arange(T), w0, w1 - 1)
    dp = xp[tt] - xp[w0]  # the plane moves with the demonstration's pelvis inside the walking segment only (rule 1)
    cen = A[:, :3, 3]
    p0 = float(np.mean(cen @ e - dp))  # least squares: the plane path closest to the attached target centre over the clip
    p = p0 + dp
    P = A.copy()
    P[:, :3, :3] = RA
    P[:, :3, 3] = cen + (p - cen @ e)[:, None] * e  # rule 3: centre on the plane; lateral position and height = the attached target's
    Ref = np.eye(3) - 2 * np.outer(e, e)
    Sm = np.tile(np.eye(4), (T, 1, 1))
    Sm[:, :3, :3] = Ref
    Sm[:, :3, 3] = 2 * p[:, None] * e
    perm = [names.index(x) for x in swap_lr(names)]
    mir = (human0 @ Ref.T + Sm[:, None, :3, 3])[:, perm]  # rule 1: the mirror at the same frame, left / right exchanged
    human1 = mir.copy()  # upper body, palms, pelvis and hips stay the mirror
    # rule 2: legs. Walking segment: the demonstration's walking legs reversed by walked distance, mirrored about the FIXED plane
    # p_f (the reversed walk then advances exactly with the moving plane; its pelvis meets the mirrored pelvis along e);
    # lift / put down: feet held at the reversed walk's first / last stance; knees by IK from the mirrored hips.
    tau_w, xs = distance_tau(xp, w0, w1)
    tau = np.empty(T)
    tau[:w0], tau[w0:w1], tau[w1:] = tau_w[0], tau_w, tau_w[-1]
    pf = p0 + (xs[-1] - xs[0]) / 2
    src = lerp_at(human0, tau)
    legm = (src @ Ref.T + 2 * pf * e)[:, perm]  # the reversed (and held) source legs mirrored about p_f, left / right exchanged
    il = {n: names.index(n) for n in LOWER}
    for sd in ("L", "R"):
        for j in ("Ankle", "Toe"):
            human1[:, il[f"{sd}_{j}"]] = legm[:, il[f"{sd}_{j}"]]
    ikrec = {}
    seg = np.where(np.arange(T) < w0, 0, np.where(np.arange(T) < w1, 1, 2))
    for sd in ("L", "R"):
        H, Kd, Ad = mir[:, il[f"{sd}_Hip"]], mir[:, il[f"{sd}_Knee"]], mir[:, il[f"{sd}_Ankle"]]
        a_ = float(np.median(np.linalg.norm(Kd - H, axis=1)))  # the (mirrored) demonstration's own thigh / shank lengths
        b_ = float(np.median(np.linalg.norm(Ad - Kd, axis=1)))
        hint = np.where((seg == 1)[:, None], legm[:, il[f"{sd}_Knee"]], Kd)  # bend side: the reversed knee (walk) / the mirrored knee (lift, put down)
        K, over, exc = leg_ik(H, human1[:, il[f"{sd}_Ankle"]], a_, b_, hint)
        human1[:, il[f"{sd}_Knee"]] = K
        hip_gap = np.linalg.norm(H - legm[:, il[f"{sd}_Hip"]], axis=1)  # mirrored hip (t) vs the reversed walk's hip (tau): what the IK absorbs
        knee_move = np.linalg.norm(K - hint, axis=1)
        ikrec[sd] = dict(thigh_m=round(a_, 4), shank_m=round(b_, 4),
                         **{ph: dict(frames=int((seg == q).sum()), overextended_frames=int((over & (seg == q)).sum()),
                                     max_excess_m=round(float(exc[seg == q].max()), 4) if (seg == q).any() else None,
                                     hip_vs_reversed_hip_max_m=round(float(hip_gap[seg == q].max()), 4) if (seg == q).any() else None,
                                     knee_vs_hint_max_m=round(float(knee_move[seg == q].max()), 4) if (seg == q).any() else None)
                            for q, ph in enumerate(("lift", "walk", "put_down"))})
    Q0 = Rt.T
    Rk = np.eye(3)
    Rk[k, k] = -1.0  # the target's own mirror plane (perpendicular to its long axis, through its centre)
    Q1 = Rk @ Q0
    cand = dict(Q0=Q0.tolist(), c0=(-w / 2 * ek).tolist(), Q1=Q1.tolist(), c1=(w / 2 * ek).tolist(), axis=k, L1=L1, parity0=1, parity1=-1,
                id=f"front-back mirror axis{k} sigma{sigma:+.0f}")
    M0 = affine(Q0, -Q0 @ cT)
    M1 = affine(Rk, np.zeros(3)) @ M0
    own1 = Sm @ Ob  # the partner's own box: the mirror of the levelled demonstration box (improper; role_solve appends F)
    Gp = own1 @ np.linalg.inv(O)
    legpel = legm[:, names.index("Pelvis")]
    D = legpel[:, :2] - mir[:, names.index("Pelvis"), :2]  # reversed-walk pelvis (p_f mirror) - mirrored pelvis, horizontal (diagnostic)
    att_dev = (A[:, :3, 3] - P[:, :3, 3]) @ e  # the attached target centre (the demonstration's box) minus the plane, along e
    diag9g = dict(walk_dir=e2.tolist(), walk_dir_vs_clip_net_deg=round(ang_e_clip, 3), vertical_box_axis=v, plane_p0=round(p0, 5), plane_fixed_legs=round(float(pf), 5),
                  plane_travel_m=round(float(dp[-1]), 5),
                  attached_vs_plane_along_e_m={ph: dict(max_abs=round(float(np.abs(att_dev[seg == q]).max()), 4), mean=round(float(att_dev[seg == q].mean()), 4))
                                               for q, ph in enumerate(("lift", "walk", "put_down")) if (seg == q).any()},
                  demo_box_vs_level_box_deg=dict(  # rotation angle between the demonstration's box and the levelled one; tilt = its vertical axis vs world vertical
                      rotation_max=round(float(np.degrees(np.arccos(np.clip((np.einsum("tij,ij->t", O[:, :3, :3], Rb) - 1) / 2, -1, 1))).max()), 3),
                      tilt_max=round(float(np.degrees(np.arccos(np.clip(np.sign(R0[2, v]) * O[:, 2, v], -1, 1))).max()), 3)),
                  tau_walk=dict(first=round(float(tau_w[0]), 4), last=round(float(tau_w[-1]), 4),
                                speed_ratio_max=round(float(np.abs(np.diff(tau_w)).max()), 4) if len(tau_w) > 1 else None,
                                held_lift_source_frame=round(float(tau_w[0]), 4), held_put_down_source_frame=round(float(tau_w[-1]), 4)),
                  leg_ik=ikrec)
    return dict(P=P, A=A, B=A.copy(), S=Sm[0].copy(), G=Gp, Gp=Gp, cand=cand, M0=M0, M1=M1, human1=human1, rot1=np.repeat(Ref[None], T, 0), D=D, k=k, sigma=sigma,
                walk_axis_angle_deg=ang, walk_dir=wd, L1=L1, Lm=Lm, cT=cT, Rt=Rt, leg_offset=(2 * pf * e)[:2], turn="mirror", legs="mirror", walk=walk,
                leg_source_frame=tau, pre_offset=None, post_offset=None, tau=tau, Ob=Ob, own1=own1, Sm=Sm, plane=p, Ref=Ref, diag9g=diag9g,
                r1_source=dict(frame=tau, R=np.repeat(Ref[None], T, 0), t=np.repeat((2 * pf * e)[None], T, 0), swap_lr=True))


def box_sd(p, M, size):
    """signed distance of points p [N, 3] to the oriented box of full size `size` at pose M (negative inside)."""
    q = (p - M[:3, 3]) @ M[:3, :3]
    d = np.abs(q) - np.asarray(size) / 2
    return np.linalg.norm(np.maximum(d, 0), axis=-1) + np.minimum(d.max(-1), 0)


LEG_POINTS = ("L_Knee", "L_Ankle", "L_Toe", "R_Knee", "R_Ankle", "R_Toe")


def leg_box(human0, names, O, size, C):
    """per frame: smallest signed distance of the leg joints (knees, ankles, toes) to the person's own box; role 1 = the
    partner vs Gp O (= G O except legs="full_scopy"), role 0 = the demonstration vs O. Summary: minimum, its frame, frames inside, first / last 5 frames."""
    il = [names.index(n) for n in LEG_POINTS]
    T = len(O)
    d1 = np.array([box_sd(C["human1"][t, il], C["Gp"][t] @ O[t], size).min() for t in range(T)])
    d0 = np.array([box_sd(human0[t, il], O[t], size).min() for t in range(T)])
    f = lambda d: dict(min_m=round(float(d.min()), 4), min_frame=int(d.argmin()), frames_inside=int((d < 0).sum()),  # noqa: E731
                       first5_m=np.round(d[:5], 3).tolist(), last5_m=np.round(d[-5:], 3).tolist())
    return dict(partner=f(d1), demo=f(d0)), d1, d0


def diag(D, s):
    d = np.linalg.norm(D, axis=1)
    return dict(pelvis_mismatch_max_m=round(float(d.max()), 4), pelvis_mismatch_mean_m=round(float(d.mean()), 4),
                pelvis_mismatch_max_person_scale_m=round(float(d.max()) / s, 4))
