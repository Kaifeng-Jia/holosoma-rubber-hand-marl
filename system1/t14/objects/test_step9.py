"""Smoke tests for T14 step 9 (frontback.py, step8_rejudge.added_drift). Run: sonic_s1/bin/python -B system1/t14/objects/test_step9.py"""
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
import frontback as fb  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import step8_rejudge as rj  # noqa: E402

NAMES = list(fb.LOWER) + ["Torso", "L_Wrist", "R_Wrist"]


def straight_carry(T=61, turn_deg=0.0):
    """person walks 1 m along +x with a time-symmetric profile carrying a 0.3 m box 0.4 m ahead; optional yaw turn of the box."""
    tt = np.linspace(0, 1, T)
    x = 0.5 - 0.5 * np.cos(np.pi * tt)  # symmetric in time
    H = np.zeros((T, len(NAMES), 3))
    for j, n in enumerate(NAMES):
        y = 0.1 if n.startswith("L_") else (-0.1 if n.startswith("R_") else 0.0)
        z = 0.05 if "Toe" in n or "Ankle" in n else 0.9
        H[:, j] = np.stack([x, np.full(T, y), np.full(T, z)], 1)
    H[:, NAMES.index("L_Wrist"), 0] += 0.25
    H[:, NAMES.index("R_Wrist"), 0] += 0.25
    O = np.tile(np.eye(4), (T, 1, 1))
    for t in range(T):
        c, s = np.cos(np.radians(turn_deg * tt[t])), np.sin(np.radians(turn_deg * tt[t]))
        O[t, :3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        O[t, :3, 3] = [x[t] + 0.4, 0.0, 0.8]
    return H, O


def test_straight_symmetric_walk_is_consistent():
    H, O = straight_carry()
    C = fb.build(H, NAMES, O, np.array([0.3, 0.4, 0.3]), 1.0)
    assert C["k"] == 0 and C["sigma"] == 1.0 and np.isclose(C["Lm"], 0.6)
    assert np.linalg.norm(C["D"], axis=1).max() < 1e-9  # reversed legs coincide with the object-attached pelvis
    pel1 = C["human1"][:, 0]
    assert np.isclose(pel1[-1, 0] - pel1[0, 0], 1.0)  # the partner moves forward with the box (walking backwards)
    assert np.allclose(C["P"][:, :3, 3], C["A"][:, :3, 3])  # translation = the demonstration's
    w = C["human1"][0, NAMES.index("L_Wrist")]
    # target centre at x = 0.4 + 0.15; the wrist (x = 0.25, y = 0.1) turned 180 deg about it -> (0.85, -0.1): the far end
    assert np.allclose(w[:2], [0.85, -0.1], atol=1e-9)


def test_turn_breaks_consistency():
    H, O = straight_carry(turn_deg=180.0)
    C = fb.build(H, NAMES, O, np.array([0.3, 0.4, 0.3]), 1.0)
    assert np.linalg.norm(C["D"], axis=1).max() > 0.3


def test_region_maps_both_directions():
    """level maps of the record (partner.region_maps on the level-1.0 candidate) = the construction's maps, for a box frame
    whose walking axis points forward (sigma +1) and backwards (sigma -1); role 0's own object point and its target point coincide in the world (A M0 = O), the
    partner's object-attached map G O = P M1."""
    import partner as pt
    rng = np.random.default_rng(0)
    for flip in (False, True):
        H, O = straight_carry()
        if flip:  # the box frame turned by 180 deg about its vertical (same box): its +x axis points backwards -> sigma = -1
            O = O @ pt.affine(np.diag([-1.0, -1.0, 1.0]), np.zeros(3))
        for m in (1.0, 1.5):
            C = fb.build(H, NAMES, O, np.array([0.3, 0.4, 0.3]), m)
            Q0, c0, Q1, c1 = pt.region_maps(C["cand"], C["k"], C["L1"], C["Lm"])
            assert np.allclose(pt.affine(Q0, c0), C["M0"]) and np.allclose(pt.affine(Q1, c1), C["M1"])
            x = np.c_[rng.normal(size=(5, 3)) * 0.1, np.ones(5)]
            for t in (0, 30, 60):
                assert np.allclose((C["A"][t] @ C["M0"] @ x.T).T, (O[t] @ x.T).T)
                assert np.allclose((C["G"][t] @ O[t] @ x.T).T, (C["P"][t] @ C["M1"] @ x.T).T)
            # role 0's region is at the -e_k end of the target, role 1's at the +e_k end
            assert C["M0"][C["k"], 3] < 0 < C["M1"][C["k"], 3]
        assert C["sigma"] == (-1.0 if flip else 1.0)


def test_world_turn_is_upright():
    """step 9c: a box tilted at frame 0 -> the box-axis turn tilts S (2 x tilt), the world-vertical turn does not; the hand
    targets (M0, M1) are the same box symmetry for both; G(0) = S for the world turn."""
    import partner as pt
    H, O = straight_carry()
    tilt = np.radians(6.0)
    Rx = np.array([[1, 0, 0], [0, np.cos(tilt), -np.sin(tilt)], [0, np.sin(tilt), np.cos(tilt)]])
    O = O @ pt.affine(Rx, np.zeros(3))
    Cb = fb.build(H, NAMES, O, np.array([0.3, 0.4, 0.3]), 1.0, turn="box")
    Cw = fb.build(H, NAMES, O, np.array([0.3, 0.4, 0.3]), 1.0, turn="world")
    assert np.isclose(pt.gravity_deg(Cb["S"]), 12.0, atol=1e-6)
    assert np.isclose(pt.gravity_deg(Cw["S"]), 0.0, atol=1e-9)
    assert np.allclose(Cb["M0"], Cw["M0"]) and np.allclose(Cb["M1"], Cw["M1"])
    assert np.allclose(Cw["G"][0], Cw["S"])
    c = Cw["A"][0][:3, 3]
    assert np.allclose(Cw["S"][:3, :3] @ c + Cw["S"][:3, 3], c)  # the turn axis passes through the target centre (fixed point)


def test_walk_legs_hold_and_reverse():
    H, O = straight_carry(T=61)
    walk = dict(w0=10, w1=50)
    C = fb.build(H, NAMES, O, np.array([0.3, 0.4, 0.3]), 1.0, legs="walk", walk=walk)
    src = C["leg_source_frame"]
    assert (src[:10] == 49).all() and (src[50:] == 10).all() and src[10] == 49 and src[49] == 10 and (np.diff(src[10:50]) == -1).all()
    il = [NAMES.index(n) for n in fb.LOWER]
    legs = C["human1"][:, il]
    assert np.allclose(legs[:10], legs[10]) and np.allclose(legs[50:], legs[49])  # standing still before / after the walk
    assert C["legs"] == "walk"


def test_swing_segments():
    """two feet, each one 0.3 m step at 0.6 m/s (15 frames) and one 3-frame shuffle of 3 cm (too short a displacement)."""
    T = 120
    toe = np.zeros((T, 2, 3))
    toe[30:45, 0, 0] = np.linspace(0, 0.3, 15)
    toe[45:, 0, 0] = 0.3
    toe[60:75, 1, 0] = np.linspace(0, 0.3, 15)
    toe[75:, 1, 0] = 0.3
    toe[90:93, 1, 0] += np.linspace(0, 0.03, 3)
    toe[93:, 1, 0] += 0.03
    sw = fb.swing_segments(toe, 20, 110)
    assert [d["foot"] for d in sw] == ["left", "right"], sw
    assert 8 <= sw[0]["start"] <= 12 and 23 <= sw[0]["stop"] <= 28 and 38 <= sw[1]["start"] <= 42, sw


def test_walk_scopy_lift_and_put_down():
    """step 9d: outside the walking segment the partner's legs are the S copy of the demonstration's legs plus one horizontal
    offset each (pre / post); across both boundaries the mean displacement of the four foot points is zero (least squares)."""
    H, O = straight_carry(T=61)
    rng = np.random.default_rng(1)
    H = H + rng.normal(scale=0.01, size=H.shape)  # some leg motion in the lift / put-down segments
    walk = dict(w0=10, w1=50)
    C = fb.build(H, NAMES, O, np.array([0.3, 0.4, 0.3]), 1.0, turn="world", legs="walk_scopy", walk=walk)
    C9c = fb.build(H, NAMES, O, np.array([0.3, 0.4, 0.3]), 1.0, turn="world", legs="walk", walk=walk)
    il = [NAMES.index(n) for n in fb.LOWER]
    legs, S = C["human1"][:, il], C["S"]
    Sx = H[:, il] @ S[:3, :3].T + S[:3, 3]
    assert np.allclose(legs[10:50], C9c["human1"][10:50][:, il])  # walking segment as step 9c
    dp, dq = np.r_[C["pre_offset"], 0.0], np.r_[C["post_offset"], 0.0]
    assert np.allclose(legs[:10], Sx[:10] + dp) and np.allclose(legs[50:], Sx[50:] + dq)
    feet = [fb.LOWER.index(n) for n in ("L_Ankle", "L_Toe", "R_Ankle", "R_Toe")]
    assert np.allclose((legs[10, feet, :2] - legs[9, feet, :2]).mean(0), 0) and np.allclose((legs[50, feet, :2] - legs[49, feet, :2]).mean(0), 0)


def test_full_scopy_whole_body():
    """step 9e: outside the walking segment the WHOLE partner (upper body too) is the S copy + d_pre / d_post of step 9d, the
    person map Gp is Trans(d) S there and G inside the walk; the legs and the walking segment equal step 9d's; Gp = G for 9d;
    r1_source rebuilds the partner's legs from the demonstration's at every frame (reversed walk, S copies outside)."""
    H, O = straight_carry(T=61)
    rng = np.random.default_rng(2)
    H = H + rng.normal(scale=0.01, size=H.shape)
    walk = dict(w0=10, w1=50)
    size = np.array([0.3, 0.4, 0.3])
    C = fb.build(H, NAMES, O, size, 1.0, turn="world", legs="full_scopy", walk=walk)
    C9d = fb.build(H, NAMES, O, size, 1.0, turn="world", legs="walk_scopy", walk=walk)
    S = C["S"]
    Sx = H @ S[:3, :3].T + S[:3, 3]
    dp, dq = np.r_[C["pre_offset"], 0.0], np.r_[C["post_offset"], 0.0]
    assert np.allclose(C["pre_offset"], C9d["pre_offset"]) and np.allclose(C["post_offset"], C9d["post_offset"])
    assert np.allclose(C["human1"][:10], Sx[:10] + dp) and np.allclose(C["human1"][50:], Sx[50:] + dq)
    assert np.allclose(C["human1"][10:50], C9d["human1"][10:50])  # walking segment unchanged (reversed legs, object-attached upper body)
    il = [NAMES.index(n) for n in fb.LOWER]
    assert np.allclose(C["human1"][:, il], C9d["human1"][:, il])  # legs as step 9d everywhere
    T_ = lambda d: fb.affine(np.eye(3), d)  # noqa: E731
    assert np.allclose(C["Gp"][:10], T_(dp) @ S) and np.allclose(C["Gp"][50:], T_(dq) @ S) and np.allclose(C["Gp"][10:50], C["G"][10:50])
    assert np.allclose(C9d["Gp"], C9d["G"]) and np.allclose(C["G"], C9d["G"])
    # the S copy holds its own box Gp O as the demonstration holds O (whole-body relation kept)
    for t in (0, 5, 55):
        rel0 = (H[t] - O[t][:3, 3]) @ O[t][:3, :3]
        G1 = C["Gp"][t] @ O[t]
        rel1 = (C["human1"][t] - G1[:3, 3]) @ G1[:3, :3]
        assert np.allclose(rel0, rel1)
    for CC in (C, C9d, fb.build(H, NAMES, O, size, 1.0, turn="world", legs="walk", walk=walk), fb.build(H, NAMES, O, size, 1.0)):
        rs = CC["r1_source"]
        rebuilt = np.einsum("tij,tkj->tki", rs["R"], H[rs["frame"]][:, il]) + rs["t"][:, None]
        assert np.allclose(rebuilt, CC["human1"][:, il]), CC["legs"]


def test_reverse_warp_time_map():
    """step 9f: tau maps the reversed put-down / walk / lift end-to-end onto lift / walk / put-down; integer tau reproduces frames."""
    tau, segs = fb.warp_tau(61, 10, 50)
    assert tau[0] == 60 and np.isclose(tau[9], 50) and np.isclose(tau[10], 49) and np.isclose(tau[49], 10) and np.isclose(tau[50], 9) and tau[60] == 0
    assert (np.diff(tau) < 0).all()
    assert [g["scale"] for g in segs] == [round(10 / 11, 4), 1.0, round(11 / 10, 4)]
    tau2, segs2 = fb.warp_tau(30, 0, 25)  # no lift: the put-down part holds the demonstration's first frame
    assert np.allclose(tau2[25:], 0) and segs2[0]["partner_frames"] == 0
    X = np.random.default_rng(3).normal(size=(61, 4, 3))
    assert np.allclose(fb.lerp_at(X, np.array([3.0, 7.0])), X[[3, 7]])
    R = Rotation.random(61, random_state=4).as_matrix()
    assert np.allclose(fb.slerp_at(R, np.array([5.0, 5.5]))[0], R[5])


def test_reverse_warp_whole_body():
    """step 9f: the partner is S applied to the demonstration at tau(t) plus ONE constant horizontal offset (every joint), its own
    box moves with it (relation person / own box = the demonstration's at tau), the offset is the least-squares fit of the own box
    onto the far end of the shared target, and r1_source rebuilds the legs."""
    H, O = straight_carry(T=61)
    rng = np.random.default_rng(5)
    H = H + rng.normal(scale=0.01, size=H.shape)
    walk = dict(w0=10, w1=50)
    size = np.array([0.3, 0.4, 0.3])
    C = fb.build(H, NAMES, O, size, 1.25, turn="world", legs="reverse_warp", walk=walk)
    S, tau = C["S"], C["tau"]
    d = np.r_[C["offset_d"], 0.0]
    exp = fb.lerp_at(H, tau) @ S[:3, :3].T + S[:3, 3] + d
    assert np.allclose(C["human1"], exp)
    off = C["human1"] - (fb.lerp_at(H, tau) @ S[:3, :3].T)  # the same translation at every frame and joint
    assert np.allclose(off, off[0, 0])
    for t in (0, 9, 10, 30, 50, 60):
        Ow = fb.poses_at(O, tau[t:t + 1])[0]
        rel0 = (fb.lerp_at(H, tau[t:t + 1])[0] - Ow[:3, 3]) @ Ow[:3, :3]
        rel1 = (C["human1"][t] - C["own1"][t][:3, 3]) @ C["own1"][t][:3, :3]
        assert np.allclose(rel0, rel1)
    far = C["P"] @ C["M1"]
    assert np.allclose((far[:, :2, 3] - C["own1"][:, :2, 3]).mean(0), 0, atol=1e-9)  # least squares: zero mean residual
    assert np.allclose(C["Gp"] @ O, C["own1"])
    il = [NAMES.index(n) for n in fb.LOWER]
    rs = C["r1_source"]
    rebuilt = np.einsum("tij,tkj->tki", rs["R"], fb.lerp_at(H, rs["frame"])[:, il]) + rs["t"][:, None]
    assert np.allclose(rebuilt, C["human1"][:, il])
    # the straight symmetric carry: reversed and time-aligned partner's box rotation equals the demonstration's (no one-sided tilt)
    assert np.allclose(C["P"][:, :3, :3], C["A"][:, :3, :3], atol=1e-9)


NAMES9G = list(fb.LOWER) + ["Torso", "L_Wrist", "R_Wrist", "L_Palm", "R_Palm"]


def carry_9g(T=80, w0=12, w1=66, dirdeg=20.0):
    """a person with leg geometry standing [0, w0), walking 1.2 m with a non-uniform speed [w0, w1), standing after; walking
    direction dirdeg from +x; a 0.3 m box carried rigidly 0.45 m ahead of the pelvis, tilted 15 deg during the lift (frames < w0)."""
    th = np.radians(dirdeg)
    e = np.array([np.cos(th), np.sin(th), 0.0])
    lat = np.array([-np.sin(th), np.cos(th), 0.0])
    u = np.clip((np.arange(T) - w0) / (w1 - 1 - w0), 0, 1)
    x = 1.2 * (u + 0.15 * np.sin(2 * np.pi * u) / (2 * np.pi))  # monotone, speed varies
    loc = {"Pelvis": (0, 0, 0.95), "L_Hip": (0, 0.1, 0.9), "R_Hip": (0, -0.1, 0.9), "L_Knee": (0.06, 0.1, 0.5), "R_Knee": (0.06, -0.1, 0.5),
           "L_Ankle": (0, 0.1, 0.1), "R_Ankle": (0, -0.1, 0.1), "L_Toe": (0.13, 0.1, 0.03), "R_Toe": (0.13, -0.1, 0.03), "Torso": (0, 0, 1.2),
           "L_Wrist": (0.3, 0.18, 1.0), "R_Wrist": (0.3, -0.14, 1.0), "L_Palm": (0.33, 0.16, 1.0), "R_Palm": (0.33, -0.12, 1.0)}
    H = np.zeros((T, len(NAMES9G), 3))
    for j, n in enumerate(NAMES9G):
        a, b, c = loc[n]
        H[:, j] = (x[:, None] + a) * e + b * lat + np.array([0, 0, c])
    H[:, :, 2] += 0.01 * np.sin(np.arange(T) / 3.0)[:, None]  # small bob
    O = np.tile(np.eye(4), (T, 1, 1))
    for t in range(T):
        tilt = np.radians(15.0) if t < w0 else 0.0
        O[t, :3, :3] = Rotation.from_euler("zy", [dirdeg, np.degrees(tilt)], degrees=True).as_matrix()
        O[t, :3, 3] = (x[t] + 0.45) * e + np.array([0, 0, 0.9])
    return H, O, e, x


def test_distance_tau():
    x = np.r_[np.zeros(5), np.cumsum(np.linspace(0.01, 0.05, 30)), np.full(5, 1.0)]
    w0, w1 = 5, 35
    tau, xs = fb.distance_tau(x, w0, w1)
    assert np.isclose(tau[0], w1 - 1) and np.isclose(tau[-1], w0)
    xi = np.interp(tau, np.arange(len(x)), x)
    assert np.allclose(xi, x[w0] + x[w1 - 1] - x[w0:w1], atol=1e-6)  # the reversed walk has covered the demonstration's distance
    assert np.all(np.diff(tau) <= 1e-12)


def test_leg_ik():
    T = 4
    Hh = np.zeros((T, 3))
    A = np.array([[0, 0, -0.7], [0.1, 0, -0.6], [0, 0, -0.85], [0, 0, -0.95]])
    K0 = np.tile([0.2, 0, -0.4], (T, 1))
    K, over, exc = fb.leg_ik(Hh, A, 0.42, 0.42, K0)
    ok = ~over
    assert np.allclose(np.linalg.norm(K[ok] - Hh[ok], axis=1), 0.42) and np.allclose(np.linalg.norm(K[ok] - A[ok], axis=1), 0.42)
    assert np.all(K[ok, 0] > 0)  # bends toward the hint
    assert over.tolist() == [False, False, True, True] and exc[3] > 0


def test_mirror_construction():
    """step 9g rules: upper body = the mirror at the same frame (left / right exchanged); plane moves with the pelvis inside the
    walk only; box level with its centre on the plane; walk feet = the reversed walk by distance about the fixed plane; lift /
    put-down feet held; knees by IK; hand targets of the partner = the mirror of role 0's; r1_source rebuilds the feet."""
    H, O, e, x = carry_9g()
    T, w0, w1 = len(O), 12, 66
    walk = dict(w0=w0, w1=w1)
    size = np.array([0.3, 0.4, 0.3])
    C = fb.build(H, NAMES9G, O, size, 1.25, legs="mirror", walk=walk)
    Ref, p = C["Ref"], C["plane"]
    assert np.allclose(Ref @ e, -e) and np.linalg.det(C["S"][:3, :3]) < 0
    sw = [NAMES9G.index(n) for n in fb.swap_lr(NAMES9G)]
    mir = (H @ Ref.T + 2 * p[:, None, None] * e)[:, sw]
    for n in ("Pelvis", "L_Hip", "R_Hip", "Torso", "L_Wrist", "R_Wrist", "L_Palm", "R_Palm"):
        j = NAMES9G.index(n)
        assert np.allclose(C["human1"][:, j], mir[:, j]), n
    pe = H[:, 0] @ e
    assert np.allclose(p[:w0 + 1], p[w0]) and np.allclose(p[w1 - 1:], p[w1 - 1])
    assert np.allclose(p[w0:w1] - p[w0], pe[w0:w1] - pe[w0])
    R = C["P"][:, :3, :3]
    assert np.allclose(R, R[0]) and np.allclose(np.abs(R[0][2]), np.eye(3)[np.argmax(np.abs(R[0][2]))])  # level, constant
    assert np.allclose(C["P"][:, :3, 3] @ e, p)  # centre on the plane
    assert np.allclose(C["P"][:, 2, 3], C["A"][:, 2, 3])  # height = the demonstration's
    feet = [NAMES9G.index(n) for n in ("L_Ankle", "L_Toe", "R_Ankle", "R_Toe")]
    assert np.allclose(C["human1"][:w0, feet], C["human1"][0, feet]) and np.allclose(C["human1"][w1:, feet], C["human1"][-1, feet])
    tau = C["tau"]
    pf = C["diag9g"]["plane_fixed_legs"]
    exp = (fb.lerp_at(H, tau) @ Ref.T + 2 * pf * e)[:, sw]
    assert np.allclose(C["human1"][:, feet], exp[:, feet], atol=1e-6)
    # the reversed walk's pelvis meets the mirrored pelvis along e (monotone walk)
    assert np.allclose(exp[:, 0] @ e, C["human1"][:, 0] @ e, atol=1e-4)
    for sd in ("L", "R"):
        hj, kj, aj = (NAMES9G.index(f"{sd}_{n}") for n in ("Hip", "Knee", "Ankle"))
        th = np.linalg.norm(C["human1"][:, kj] - C["human1"][:, hj], axis=1)
        sh = np.linalg.norm(C["human1"][:, aj] - C["human1"][:, kj], axis=1)
        ok = np.linalg.norm(C["human1"][:, aj] - C["human1"][:, hj], axis=1) < C["diag9g"]["leg_ik"][sd]["thigh_m"] + C["diag9g"]["leg_ik"][sd]["shank_m"]
        assert ok.any() and np.allclose(th[ok], C["diag9g"]["leg_ik"][sd]["thigh_m"], atol=1e-4) and np.allclose(sh[ok], C["diag9g"]["leg_ik"][sd]["shank_m"], atol=1e-4)
    assert np.linalg.det(C["M1"][:3, :3]) < 0
    import partner as pt
    Q0, c0, Q1, c1 = pt.region_maps(C["cand"], C["cand"]["axis"], C["L1"], C["Lm"])
    assert np.allclose(pt.affine(Q0, c0), C["M0"]) and np.allclose(pt.affine(Q1, c1), C["M1"])
    loc = np.array([0.05, -0.1, 0.07])
    for t in (0, 30, 70):  # rigid carry: the mirror of role 0's contact point is the partner's (same frame)
        a0 = C["P"][t] @ C["M0"] @ np.r_[loc, 1]
        a1 = C["P"][t] @ C["M1"] @ np.r_[loc, 1]
        assert np.allclose(Ref @ a0[:3] + 2 * p[t] * e, a1[:3], atol=1e-9)
    assert np.allclose(C["own1"], np.einsum("tij,tjk->tik", C["Sm"], C["Ob"])) and np.allclose(C["own1_real"], np.einsum("tij,tjk->tik", C["Sm"], O))
    for t in (0, 30, 70):  # the mirrored person holds the mirrored real box as the demonstration holds its box
        jj = [NAMES9G.index(n) for n in fb.swap_lr(NAMES9G)]
        rel0 = (H[t] - O[t][:3, 3]) @ O[t][:3, :3]
        rel1 = (C["human1"][t] - C["own1_real"][t][:3, 3]) @ C["own1_real"][t][:3, :3]
        up = [NAMES9G.index(n) for n in ("Torso", "L_Wrist", "R_Wrist", "L_Palm", "R_Palm", "Pelvis", "L_Hip", "R_Hip")]
        assert np.allclose(rel1[up], rel0[jj][up])
    rs = C["r1_source"]
    assert rs["swap_lr"]
    rebuilt = (np.einsum("tij,tkj->tki", rs["R"], fb.lerp_at(H, rs["frame"])) + rs["t"][:, None])[:, sw]
    assert np.allclose(rebuilt[:, feet], C["human1"][:, feet], atol=1e-6)
    # person-side frames: role 0 (Ob M0^-1) and role 1 (Sm Ob F (M1 F)^-1) differ from the shared target by a translation only
    F = np.diag([1.0, 1.0, -1.0, 1.0])
    f0 = C["Ob"] @ np.linalg.inv(C["M0"])
    f1 = C["own1"] @ F @ np.linalg.inv(C["M1"] @ F)
    assert np.allclose(f0[:, :3, :3], R) and np.allclose(f1[:, :3, :3], R)


def test_added_drift():
    T = 20
    rob, src = np.zeros((T, 3)), np.zeros((T, 3))
    rob[5:10, 0] = np.linspace(0, 0.03, 5)
    flag = np.zeros(T, bool)
    flag[5:10] = True
    v, nw, nf = rj.added_drift(rob, src, flag)
    assert np.isclose(v, 0.03) and nw == 1 and nf == 5
    assert rj.added_drift(rob, src, np.zeros(T, bool))[0] is None


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
            print("ok", k)
