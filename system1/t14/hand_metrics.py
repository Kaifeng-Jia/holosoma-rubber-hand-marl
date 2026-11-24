#!/usr/bin/env python3
"""T14 round 4 (section 9 item 3) hand metrics of two-robot references. All directions are compared in the object frame
(human: source object frame; robot: the reference's object frame), so the role isometry and the CORE4D scale / shift do not
enter. Human-contact frames = the source contact rule (any SMPL-X hand vertex within 1.5 cm of the captured mesh);
CORE4D 15 Hz frame t <-> reference frame 2t, robot r <-> person r + 1; Push / Pull 30 Hz, robot 1 inherits the source (Pull:
mirrored across the table's local XY plane, hands swapped).

Like the human (contact frames, per robot and hand, mean and 95th percentile, degrees):
  palm_normal_angle   robot palm normal vs the person's palm normal (hand_landmarks: Index1 / Pinky1 / wrist plane);
  finger_angle        robot finger direction (rubber-hand long axis) vs the person's wrist -> middle fingertip;
  long_axis_angle     (secondary) robot long axis vs the person's wrist -> Middle1.
Plausible:
  palm_surface_angle  (contact frames) robot palm normal vs the inward object-surface normal at the point of the simulator
                      collision surface nearest to the robot palm point;
  wrist_near_limit    fraction of all frames in which a wrist joint (roll / pitch / yaw) is within 2 deg of a URDF limit,
                      per joint and "any of the three";
  wrist_speed_peak    largest |dq/dt| (deg/s, 30 Hz central differences) over the three wrist joints.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "system1/t14"))
sys.path.insert(0, str(ROOT / "system1/t12v2"))
sys.path.insert(0, str(ROOT / "src/holosoma_retargeting"))
import contact_stats as cs  # noqa: E402
import hand_landmarks as hl  # noqa: E402

WRIST = ("wrist_roll", "wrist_pitch", "wrist_yaw")


def _ang(a, b):
    return np.degrees(np.arccos(np.clip(np.einsum("...i,...i->...", a, b), -1.0, 1.0)))


def _stats(x):
    x = np.asarray(x, float)
    return dict(frames=int(len(x)), mean_deg=round(float(x.mean()), 1), p95_deg=round(float(np.percentile(x, 95)), 1)) if len(x) else dict(frames=0, mean_deg=None, p95_deg=None)


# ------------------------------------------------------------------------------------------------ human side
def human_core4d(task):
    """flags [T15, 2 persons, 2 hands], directions in the raw object frame {name: [T15, 2, 2, 3]}."""
    import contact_gap as cg
    import table_shared as tsh
    flags, _, _ = cg.human_core4d(task)
    seq = tsh.CORE4D / "CORE4D_Real/human_object_motions" / tsh.TASKS[task]["raw"]
    To = np.load(seq / "smooth_objposes.npy")
    tf = hl.template_frames()
    D = {k: np.zeros((len(To), 2, 2, 3)) for k in ("palm_normal", "finger", "long_axis")}
    for p in (1, 2):
        P = np.load(seq / f"person{p}_poses.npz", allow_pickle=True)["arr_0"].item()
        J, V = P["joints"].astype(np.float64), P["vertices"].astype(np.float64)
        n = min(len(J), len(To))
        for hi, side in enumerate(("left", "right")):
            d = hl.human_directions(J[:n], V[:n], side, tf)
            for k in D:
                D[k][:n, p - 1, hi] = np.einsum("tji,tj->ti", To[:n, :3, :3], d[k])  # R^T d (raw object frame)
    return flags, D


def human_omomo(task):
    import contact_gap as cg
    import dual_stretch as ds
    import single_shared as ss
    from holosoma_retargeting.src.utils import load_intermimic_data
    seq = ds.SEQ[task]
    ss.omomo_palms(seq)
    J, tips = ss.omomo_palms.last["joints_omomo_new"], ss.omomo_palms.last["tips_omomo_new"]  # OMOMO_new world (rigid fit)
    flags, _, _ = cg.human_omomo(task)
    _, obj = load_intermimic_data(str(ss.OMOMO_NEW / f"{seq}.pt"))
    Ro = cs.pose_mat(obj[:, 4:7], obj[:, [1, 2, 3, 0]])[:, :3, :3]
    tf = hl.template_frames()
    T = len(J)
    D = {k: np.zeros((T, 2, 3)) for k in ("palm_normal", "finger", "long_axis")}
    for hi, side in enumerate(("left", "right")):
        V = np.zeros((T, max(hl.TIP_VERTEX.values()) + 1, 3))
        V[:, hl.TIP_VERTEX[side]] = tips[side]
        d = hl.human_directions(J, V, side, tf)
        for k in D:
            D[k][:, hi] = np.einsum("tji,tj->ti", Ro, d[k])
    return flags, D


# ------------------------------------------------------------------------------------------------ robot side
class RobotHands:
    def __init__(self):
        self.robot = cs.Robot("physx")
        import yourdfpy
        u = yourdfpy.URDF.load(str(cs.ROBOT_URDF), load_meshes=False, build_scene_graph=False)
        jn = {j.name: j for j in u.robot.joints}
        self.wrist = {}
        for side in ("left", "right"):
            idx, lo, hi = [], [], []
            for w in WRIST:
                n = f"{side}_{w}_joint"
                idx.append(self.robot.dofs.index(n))
                lo.append(jn[n].limit.lower)
                hi.append(jn[n].limit.upper)
            self.wrist[side] = (np.array(idx), np.array(lo), np.array(hi))

    def directions_local(self, pair, obj_q):
        """robot palm normal / finger direction in the object frame: {name: [T, 2 robots, 2 hands, 3]}."""
        tl = self.robot.fk(pair[:, :, :3], np.concatenate([pair[:, :, 4:7], pair[:, :, 3:4]], -1), pair[:, :, 7:])
        Ro = cs.pose_mat(obj_q[:, :3], np.concatenate([obj_q[:, 4:7], obj_q[:, 3:4]], -1))[:, :3, :3]
        out = {k: np.zeros(pair.shape[:2] + (2, 3)) for k in ("palm_normal", "finger")}
        for hi, side in enumerate(("left", "right")):
            d = hl.robot_directions(tl[f"{side}_rubber_hand_link"][..., :3, :3], side)
            for k in out:
                out[k][:, :, hi] = np.einsum("tji,trj->tri", Ro, d[k])
        return out

    def wrist_stats(self, pair, fps=30):
        out = {}
        for r in (0, 1):
            for side in ("left", "right"):
                idx, lo, hi = self.wrist[side]
                q = pair[:, r, 7 + idx]
                near = (q - lo < np.radians(2.0)) | (hi - q < np.radians(2.0))
                dq = np.degrees(np.abs(np.gradient(q, axis=0)) * fps)
                out[f"r{r}_{side}"] = dict(near_limit_any=round(float(near.any(1).mean()), 4),
                                           near_limit_per_joint={w: round(float(near[:, i].mean()), 4) for i, w in enumerate(WRIST)},
                                           speed_peak_dps=round(float(dq.max()), 1), speed_peak_joint=WRIST[int(dq.max(0).argmax())],
                                           speed_peak_frame=int(dq.max(1).argmax()))
        return out


def summarize(robot_dirs, human_dirs, flags, palm_surface_angle, wrist):
    """robot_dirs / human_dirs [T, 2, 2, 3] aligned frame by frame and robot by robot; flags [T, 2, 2]."""
    res = {}
    for r in (0, 1):
        for hi, side in enumerate(("left", "right")):
            f = flags[:, r, hi]
            res[f"r{r}_{side}"] = dict(
                palm_normal_angle=_stats(_ang(robot_dirs["palm_normal"][:, r, hi], human_dirs["palm_normal"][:, r, hi])[f]),
                finger_angle=_stats(_ang(robot_dirs["finger"][:, r, hi], human_dirs["finger"][:, r, hi])[f]),
                long_axis_angle=_stats(_ang(robot_dirs["finger"][:, r, hi], human_dirs["long_axis"][:, r, hi])[f]),
                palm_surface_angle=_stats(palm_surface_angle[:, r, hi][f]),
                **wrist[f"r{r}_{side}"])
    return res


def core4d(task, rh, human, pair, obj_q, k):
    import contact_gap as cg
    from object_surface import UnionSurface
    from sim_scene import object_pieces
    flags15, H = human
    t15 = np.arange(min(len(flags15), (len(pair) + 1) // 2))
    P, O = pair[2 * t15], obj_q[2 * t15]
    rd = rh.directions_local(P, O)
    Pl = cg.robot_palms_local(rh.robot, P, O, cg.palm_offsets())
    _, _, nrm = UnionSurface(object_pieces(task, k)).nearest(Pl, normals=True)
    psa = _ang(rd["palm_normal"], -nrm)
    return summarize(rd, {k2: v[t15] for k2, v in H.items()}, flags15[t15], psa, rh.wrist_stats(pair))


def omomo(task, rh, human, pair, obj_q, urdf):
    import contact_gap as cg
    import dual_stretch as ds
    from eval_core4d_hulls import urdf_box_meshes
    from object_surface import UnionSurface
    flags, H = human
    n = min(len(pair), len(flags))
    P, O = pair[:n], obj_q[:n]
    rd = rh.directions_local(P, O)
    Pl = cg.robot_palms_local(rh.robot, P, O, cg.palm_offsets())
    _, _, nrm = UnionSurface(urdf_box_meshes(urdf)).nearest(Pl, normals=True)
    psa = _ang(rd["palm_normal"], -nrm)
    hd = {k: np.zeros((n, 2, 2, 3)) for k in H}
    fl = np.zeros((n, 2, 2), bool)
    for k in H:
        hd[k][:, 0] = H[k][:n]
        if task == "push":
            hd[k][:, 1] = H[k][:n]
        else:
            hd[k][:, 1] = H[k][:n, ::-1] * np.array([1.0, 1.0, -1.0])  # mirror across the table's local XY plane, hands swapped
    fl[:, 0] = flags[:n]
    fl[:, 1] = flags[:n] if task == "push" else flags[:n, ::-1]
    return summarize(rd, hd, fl, psa, rh.wrist_stats(pair))
