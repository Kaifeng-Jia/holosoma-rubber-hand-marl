"""T14 step 8 (Kevin 2026-11-24, step-7 decisions item 4): the new partner construction for single-person demonstrations.

* Roles: role 0 = the original demonstration; role 1 = the demonstration copied in the world by a fixed isometry S.
  S comes from one candidate pair of the existing role check (the same candidate set and order as step 7: per role, the
  region map M_r(x) = Q_r x + c_r from the source object frame to the lengthened target's frame); in the source object frame
  S_local = M0^-1 o M1 (role 0's region -> role 1's region), and in the world S = O(0) S_local O(0)^-1 with O(0) the object pose
  at the clip's first frame (object at rest), so the copy's feet start on the ground.
* Object: role 0 sees the demonstration's object motion D0(t) = O(t) O(0)^-1, role 1 its copy D1(t) = S D0(t) S^-1. The shared
  target (the lengthened object of the level) starts at P(0) = O(0) o M0^-1 (role 0's region on the source object, role 1's
  region on S of it) and moves to the midpoint of D0(t) P(0) and D1(t) P(0): translation = mean of the two positions of the
  target's origin, rotation = slerp midpoint. No parameter.
* Everything is computed in the solver's (preprocessed) world, where the person is scaled by s about the origin and the
  object trajectory is a similarity of the raw one (scale s), so midpoints and isometries commute with it; lengths of the
  main-motion rule are divided by s (screen thresholds are on the person's real scale).
* Gates before solving (existing numbers only): V0 relative_gravity_deviation_deg (5 deg) on S's rotation; main motion of the
  shared trajectory with the screen thresholds: class b the target origin moves >= 0.5 m horizontally from frame 0; class c the
  target's lowest corner is > 0.10 m above its rest height (lowest over the clip after a 5-frame running median) for >= 15
  frames.
Families: "box" (step-7 objects, robot object = oriented box x s, r = 1, nominal poses) and "table" (Push / Pull: round-6
objects, the training table at its real size, phys poses).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "system1/t14"))
sys.path.insert(0, str(ROOT / "src/holosoma_retargeting"))
sys.path.insert(0, str(HERE))

V0_GRAVITY_DEG = 5.0  # reference_geometry_round2_checks.V0["relative_gravity_deviation_deg"]
MAIN = dict(b_travel_m=0.5, c_lift_m=0.10, c_min_frames=15, rest_median_frames=5)  # screen_omomo.RULES
WBT = ROOT / "src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking"
TABLES = dict(push=dict(seq="sub6_largetable_033", target=WBT / "objects_widetable_plan5_training.urdf", axis=0, single=ROOT / "logs_system1/t14/r6/single_push_C"),
              pull=dict(seq="sub3_largetable_010", target=WBT / "objects_widetable_plan5_pull_training.urdf", axis=2, single=ROOT / "logs_system1/t14/r6/single_pull_C"))
SOURCE_TABLE = WBT / "objects_largetable.urdf"


def mj_to_mat(q):
    """MuJoCo free-joint object qpos (x y z qw qx qy qz) [T, 7] -> [T, 4, 4]."""
    q = np.asarray(q, float)
    M = np.tile(np.eye(4), (len(q), 1, 1))
    M[:, :3, :3] = Rotation.from_quat(q[:, [4, 5, 6, 3]]).as_matrix()
    M[:, :3, 3] = q[:, :3]
    return M


def mat_to_mj(M):
    q = Rotation.from_matrix(M[:, :3, :3]).as_quat()[:, [3, 0, 1, 2]]
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    return np.concatenate([M[:, :3, 3], q], 1)


def affine(R, t):
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = R, t
    return M


def region_maps(cand, axis, L1, Lm):
    """level maps: role r keeps its end region: c_r(m) = c_r -/+ (L(m) - L(1)) / 2 along the axis."""
    e = np.eye(3)[axis]
    Q0, Q1 = np.asarray(cand["Q0"], float), np.asarray(cand["Q1"], float)
    c0 = np.asarray(cand["c0"], float) - (Lm - L1) / 2 * e
    c1 = np.asarray(cand["c1"], float) + (Lm - L1) / 2 * e
    return Q0, c0, Q1, c1


def construction(O, cand, axis, L1, Lm):
    """O [T, 4, 4] own (source) object poses in the solver world (object frame = the source object frame of the candidates).
    Returns dict(S [4, 4] world isometry role 0 -> role 1, S_local, P [T, 4, 4] shared target poses, P0_role (D0 P0), P1_role)."""
    Q0, c0, Q1, c1 = region_maps(cand, axis, L1, Lm)
    M0, M1 = affine(Q0, c0), affine(Q1, c1)
    S_local = np.linalg.inv(M0) @ M1
    O0 = O[0]
    S = O0 @ S_local @ np.linalg.inv(O0)
    P0 = O0 @ np.linalg.inv(M0)  # target frame -> world at frame 0
    D0 = O @ np.linalg.inv(O0)
    D1 = S @ D0 @ np.linalg.inv(S)
    A, B = D0 @ P0, D1 @ P0
    P = np.tile(np.eye(4), (len(O), 1, 1))
    P[:, :3, 3] = (A[:, :3, 3] + B[:, :3, 3]) / 2
    for t in range(len(O)):
        r = Rotation.from_matrix(np.stack([A[t, :3, :3], B[t, :3, :3]]))
        P[t, :3, :3] = Slerp([0.0, 1.0], r)([0.5]).as_matrix()[0]
    return dict(S=S, S_local=S_local, P=P, role0=A, role1=B, M0=M0, M1=M1)


def gravity_deg(S):
    """V0 relative gravity deviation of the fixed map: angle between S's linear part applied to the vertical and the vertical."""
    v = S[:3, :3] @ np.array([0.0, 0.0, 1.0])
    return float(np.degrees(np.arccos(np.clip(v[2], -1.0, 1.0))))


def main_motion(P, corners_local, cls, s):
    """main-motion rule on the shared trajectory, lengths / s (person's real scale)."""
    from scipy.ndimage import median_filter
    origin = P[:, :3, 3]
    travel = float(np.linalg.norm((origin[:, :2] - origin[0, :2]), axis=1).max()) / s
    world = np.einsum("tij,kj->tki", P[:, :3, :3], corners_local) + origin[:, None]
    bottom = median_filter(world[..., 2].min(1), MAIN["rest_median_frames"], mode="nearest")
    lift = (bottom - bottom.min()) / s
    above = lift > MAIN["c_lift_m"]
    f = np.concatenate([[False], above, [False]])
    e = np.flatnonzero(np.diff(f.astype(int)))
    longest = int(max((b - a for a, b in zip(e[0::2], e[1::2])), default=0))
    if cls == "b":
        ok = travel >= MAIN["b_travel_m"]
    else:
        ok = longest >= MAIN["c_min_frames"]
    return dict(cls=cls, passed=bool(ok), travel_m=round(travel, 4), lift_max_m=round(float(lift.max()), 4), frames_lift_above_0p10=int(above.sum()),
                longest_run_above_0p10=longest)


def box_corners(size):
    h = np.asarray(size, float) / 2
    return np.array([[sx * h[0], sy * h[1], sz * h[2]] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])


def urdf_corners(urdf):
    from reference_geometry_math import BoxUnion
    b = BoxUnion(urdf)
    lo, hi = np.min([p["lo"] for p in b.parts], 0), np.max([p["hi"] for p in b.parts], 0)
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


def candidate_pairs(q, obj_mj, src_urdf, family, axis=None, target_urdf=None, case=None):
    """the existing role check's candidate pairs (step-7 order) for a source robot solve: q [T, 36], obj_mj [T, 7] (MuJoCo order).
    box: role_check_box.construct_box on both horizontal box axes (target = 2 x along the axis); table: the original round-2
    construct() with the training table as target. Returns list of dict(id, axis, Q0, c0, Q1, c1, parity0, parity1, fit_rmse)."""
    from reference_geometry_math import BoxUnion
    from reference_geometry_pilot import RobotFK
    from reference_geometry_round2_correspondence import tie_safe_weights, fit_candidates, construct
    import role_check_box as rcb
    obj = np.concatenate([obj_mj[:, :3], obj_mj[:, 3:7]], 1)  # (pos, wxyz) = the check's convention
    src = BoxUnion(src_urdf)
    fk = RobotFK()
    inter, curves = rcb.source_interactions(q, obj, fk, src, 30)
    weighted = tie_safe_weights(inter, curves)
    out, specs = [], []
    if family == "box":
        size = src.parts[0]["size"]
        for ax in (0, 1):
            tsize = size.copy()
            tsize[ax] *= 2
            import tempfile
            with tempfile.TemporaryDirectory() as d:
                target = BoxUnion(rcb.box_urdf(Path(d) / "t.urdf", "b", tsize))
                spec = rcb.construct_box(src, target, weighted, ax)
            specs.append((ax, spec, float(tsize[ax])))
    else:
        target = BoxUnion(target_urdf)
        spec = construct(case, src, target, weighted)
        wide = next(p for p in target.parts if p["kind"] == "tabletop")
        specs.append((axis, spec, float(wide["size"][axis])))
    for ax, spec, L1 in specs:
        cands = fit_candidates(spec)
        r0 = [c for c in cands if c["role"] == 0]
        r1 = [c for c in cands if c["role"] == 1]
        for a in r0:
            for b in r1:
                if a["fit"]["status"] != "solved" or b["fit"]["status"] != "solved":
                    continue
                out.append(dict(id=a["id"] + " || " + b["id"], axis=ax, L1=L1, Q0=a["fit"]["Q"].tolist(), c0=a["fit"]["c"].tolist(), Q1=b["fit"]["Q"].tolist(),
                                c1=b["fit"]["c"].tolist(), parity0=a["parity"], parity1=b["parity"],
                                fit_rmse_m=float(np.sqrt((a["fit"]["weighted_rmse_m"] ** 2 + b["fit"]["weighted_rmse_m"] ** 2) / 2))))
    windows = {r["hand"]: [(w["face_id"], w["start"], w["end_exclusive"]) for w in r["stable_face_windows"]] for r in inter["records"]}
    return out, dict(windows=windows, accepted_per_role=[s[1]["accepted_per_role"] for s in specs], attempted_per_role=[s[1]["attempted_per_role"] for s in specs])
