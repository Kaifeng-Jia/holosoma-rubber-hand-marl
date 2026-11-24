#!/usr/bin/env python3
"""T14 step 8: verdict and metrics of one candidate pair of the new partner construction (two robots solved around the shared
trajectory by role_solve.py). Read only, CPU.

Verdict (existing numbers only, Kevin 2026-11-24 step-7 decision 4):
  * V0 (scripts/reference_geometry_round2_checks.py, frozen) items named by Kevin:
      gravity            relative_gravity_deviation_deg <= 5 (S's linear part vs the vertical; partner.gravity_deg);
      support drift      <side>_support_added_drift_m <= 0.01 m per robot (V0 support windows of the source robot: foot bottom
                         <= 4 cm, foot speed <= 0.15 m/s for >= 0.1 s; drift of the robot minus the source robot's; role 1's
                         source = the source robot moved by S, whose drift is the same);
      non-hand penetration  additional_nonhand_proxy_penetration_m <= 0.01 m per robot (round-2 CollisionProxy: robot vs the
                         shared target minus the source robot vs the source object);
      two-robot clearance   robot_robot_clearance_m >= -0.01 m (CollisionProxy, cross-robot OBB proxies);
    (V0 support added speed is reported, not part of Kevin's list);
  * section-4 contact acceptance: CoCarry contact retention >= 0.99 for each robot (person's hand within 0.10 m of the captured
    object and the shared object moving faster than 0.03 m/s on the person's scale; robot hand signed distance >= -0.02 m,
    exact convex distance to the target boxes). Section 4's penetration criterion compares with "OmniRetarget as is", which has
    no result for these sequences; it is not used (reported numbers only);
  * main motion of the shared trajectory (partner.main_motion, screen thresholds) — evaluated before solving.
Metrics as rounds 6 / step 7: exact max penetration, robot-robot exact distance, 3-D contact-position error (person's palm ->
nearest captured-surface point, moved by the role's region map; table family: capture -> training similarity first), hand
metrics, foot slide (posture_metrics rule), posture vs the person (posture_metrics.analyse on each robot's solve).

    sonic_s1/bin/python -B system1/t14/objects/shared_eval.py --cand-dir <dir with r0/ r1/> --source <source solve> --out <json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for p in (ROOT / "scripts", ROOT / "system1/t14", ROOT / "system1/t12v2", ROOT / "src/holosoma_retargeting", HERE):
    sys.path.insert(0, str(p))
import contact_gap as cg  # noqa: E402
import dual_stretch as ds  # noqa: E402
import eval_core4d_hulls as eh  # noqa: E402
import hand_metrics as hm  # noqa: E402
import partner as pt  # noqa: E402
from reference_geometry_math import BoxUnion  # noqa: E402
from reference_geometry_pilot import RobotFK  # noqa: E402
from reference_geometry_round2_checks import V0, CollisionProxy, support_metrics, finite_max  # noqa: E402

CACHE = {}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ctx():
    if not CACHE:
        import posture_metrics as pm
        CACHE.update(fk=RobotFK(), robot=eh.cs.Robot("physx"), rh=hm.RobotHands(), ffk=pm.FullFK(), offs=cg.palm_offsets())
        CACHE["pc"] = ds.PairCollision(CACHE["robot"], depth="exact")
    return CACHE


def feet(fk, q):
    names = ["left_ankle_roll_link", "right_ankle_roll_link"]
    poses = fk.poses(q, names)
    out = {}
    for side, n in zip(("left", "right"), names):
        v = fk.meshes[n].vertices
        p = np.einsum("tij,pj->tpi", poses[n][:, :3, :3], v) + poses[n][:, :3, 3][:, None]
        out[side + "_foot_bottom_z_m"] = p[:, :, 2].min(1)
        out[side + "_foot_position_m"] = poses[n][:, :3, 3]
    return out


def stats(x):
    x = np.asarray(x, float)
    return dict(frames=int(len(x)), mean_cm=round(100 * float(x.mean()), 2), p95_cm=round(100 * float(np.percentile(x, 95)), 2)) if len(x) else None


def human_targets(rec):
    """person's palm -> nearest captured-surface point (source object frame, robot units) and its outward normal, per source hand."""
    import trimesh
    from scipy.spatial.transform import Rotation
    import single_shared as ss
    seq = rec["sequence"]
    palms, fit = ss.omomo_palms(seq)
    if rec["family"] == "box":
        import box_assets as ba
        import omomo_motion as om
        v = om.load_sequences({rec["object"]})[seq]
        bf = ba.box_frame(rec["object"], v)
        poses, _ = ba.poses_omomo_new(v, bf, fit)
        mesh = ba.mesh_in_box_frame(rec["object"], bf)
    else:
        import omomo_table as ot
        from holosoma_retargeting.src.utils import load_intermimic_data
        _, poses = load_intermimic_data(str(ss.OMOMO_NEW / f"{seq}.pt"))
        mesh = ot.relation(seq)["mesh"]
    R = Rotation.from_quat(poses[:, [1, 2, 3, 0]]).as_matrix()
    local = np.einsum("tji,thj->thi", R, palms - poses[:, None, 4:7])
    tg, nm = np.zeros_like(local), np.zeros_like(local)
    for h in (0, 1):
        c, _, tri = trimesh.proximity.closest_point(mesh, local[:, h])
        nn = mesh.face_normals[tri]
        nn = nn * np.where(np.einsum("ij,ij->i", local[:, h] - c, nn) < 0, -1.0, 1.0)[:, None]
        tg[:, h], nm[:, h] = c, nn
    if rec["family"] == "table":
        import omomo_table as ot
        tg = ot.to_training(seq, tg.reshape(-1, 3)).reshape(tg.shape)
    unit = rec["smpl_scale"] if rec["family"] == "box" else 1.0
    return tg * unit, nm


def evaluate(cand_dir, source_dir, gates):
    C = ctx()
    fk, robot = C["fk"], C["robot"]
    recs = [json.loads((cand_dir / f"r{r}" / "run_record.json").read_text()) for r in (0, 1)]
    zs = [np.load(cand_dir / f"r{r}" / f"{recs[r]['tag']}_new_qpos.npz") for r in (0, 1)]
    q = [z["qpos"].astype(np.float64)[:, :36] for z in zs]
    obj = zs[0]["object_qpos_mj"].astype(np.float64)  # (pos, wxyz): MuJoCo order = the check's order
    assert np.abs(obj - zs[1]["object_qpos_mj"]).max() < 1e-9
    T = min(len(q[0]), len(q[1]))
    q = [x[:T] for x in q]
    obj = obj[:T]
    pair = np.stack(q, 1)
    rec = recs[0]
    S = np.asarray(rec["S"])
    # source robot and geometry
    if rec["family"] == "box":
        srec = json.loads((source_dir / "run_record.json").read_text())
        sz = np.load(source_dir / f"{srec['tag']}_new_qpos.npz")
        src_geom = BoxUnion(srec["target"]["urdf"])
    else:
        srec = json.loads((source_dir / "run_record.json").read_text())
        sz = np.load(source_dir / f"{srec['task']}_new_qpos.npz")
        src_geom = BoxUnion(pt.SOURCE_TABLE)
    sq = sz["qpos"].astype(np.float64)[:T]
    s_robot, s_obj = sq[:, :36], sq[:, 36:43]
    tgt_urdf = rec["target"]["urdf"]
    tgt_geom = BoxUnion(tgt_urdf)
    # ---------------- V0
    src_feet = feet(fk, s_robot)
    src_engine = CollisionProxy(fk, src_geom)
    src_poses = fk.poses(s_robot, list(fk.meshes))
    src_coll, _ = src_engine.evaluate([src_poses], s_obj)
    src_pen = np.maximum(0, -src_coll["nonhand_object_clearance_m"][:, 0])
    engine = CollisionProxy(fk, tgt_geom)
    poses = [fk.poses(x, list(fk.meshes)) for x in q]
    coll, wit = engine.evaluate(poses, obj)
    v0 = dict(gravity=dict(value_deg=gates["gravity_deg"], limit=V0["relative_gravity_deviation_deg"], passed=gates["gravity_deg"] <= V0["relative_gravity_deviation_deg"] + 1e-10))
    for r in (0, 1):
        sm = support_metrics(feet(fk, q[r]), src_feet, 30)
        drift = max(finite_max(sm["left_support_added_drift_m"], -np.inf), finite_max(sm["right_support_added_drift_m"], -np.inf))
        spd = max(finite_max(sm["left_support_added_speed_m_s"], -np.inf), finite_max(sm["right_support_added_speed_m_s"], -np.inf))
        v0[f"support_drift_r{r}"] = dict(value_m=None if not np.isfinite(drift) else round(drift, 5), limit=V0["additional_support_drift_m"],
                                         passed=bool(not np.isfinite(drift) or drift <= V0["additional_support_drift_m"] + 1e-10),
                                         support_frames=int(sm["left_source_support_mask"].sum() + sm["right_source_support_mask"].sum()))
        v0[f"support_speed_r{r}_info"] = dict(value_mps=None if not np.isfinite(spd) else round(spd, 5), limit=V0["additional_support_speed_m_s"])
        add = float((np.maximum(0, -coll["nonhand_object_clearance_m"][:, r]) - src_pen).max())
        v0[f"nonhand_penetration_r{r}"] = dict(value_m=round(add, 5), limit=V0["additional_nonhand_proxy_penetration_m"], passed=add <= V0["additional_nonhand_proxy_penetration_m"] + 1e-10)
    rr_proxy = float(coll["robot_robot_clearance_m"].min())
    v0["robot_robot_clearance"] = dict(value_m=round(rr_proxy, 5), limit=-V0["pair_proxy_penetration_m"], passed=rr_proxy >= -V0["pair_proxy_penetration_m"] - 1e-10)
    # ---------------- contact acceptance and metrics (exact)
    ev = eh.ObjectRobotDistance(eh.urdf_box_meshes(tgt_urdf), robot, depth="exact")
    Hd, Pn = ev.frames(pair, obj)
    hc = [np.load(cand_dir / f"r{r}" / "human_contact.npz") for r in (0, 1)]
    ret, cond_frames = {}, {}
    for r in (0, 1):
        h_sd, spd = hc[r]["h_sd"][:T], hc[r]["object_speed"][:T]
        cond = (h_sd >= -0.10) & (spd[:, None] > 0.03)
        cond_frames[f"r{r}"] = int(cond.sum())
        ret[f"r{r}"] = round(float((Hd[:, r] >= -0.02)[cond].mean()), 4) if cond.any() else None
    contact_ok = all(v is not None and v >= 0.99 for v in ret.values())
    rr = C["pc"].min_distance(pair)
    # 3-D contact-position error
    tg, nm = human_targets(rec)
    Pl = cg.robot_palms_local(robot, pair, obj, C["offs"])
    ce = {}
    Ms = [np.asarray(recs[r]["candidate"]["Q0" if r == 0 else "Q1"]) for r in (0, 1)]
    for r in (0, 1):
        R_, c_ = (np.asarray(x) for x in pt.region_maps(rec["candidate"], rec["axis"], rec["L1"], rec["Lm"])[2 * r:2 * r + 2])
        hidx = [0, 1] if (r == 0 or not recs[r]["improper"]) else [1, 0]
        t_r = tg[:T][:, hidx] @ R_.T + c_
        n_r = nm[:T][:, hidx] @ R_.T
        flags = hc[r]["h_sd"][:T] >= -0.015
        e = Pl[:, r] - t_r
        nc = np.einsum("thi,thi->th", e, n_r)
        dist = np.linalg.norm(e, axis=-1)
        ce[f"r{r}"] = dict(distance=stats(dist[flags]), normal_signed_mean_cm=round(100 * float(nc[flags].mean()), 2) if flags.any() else None)
    # foot slide (posture_metrics rule) and posture
    import posture_metrics as pm
    import support
    from holosoma_retargeting.config_types.data_type import SMPLH_DEMO_JOINTS
    names = list(SMPLH_DEMO_JOINTS) + ["_"] * 64
    post, foot = {}, {}
    for r in (0, 1):
        d = cand_dir / f"r{r}"
        post[f"r{r}"] = pm.analyse(d / f"{recs[r]['tag']}_new_qpos.npz", d / "run_record.json", C["ffk"], names, ["L_Toe", "R_Toe"])
        foot[f"r{r}"] = {k: post[f"r{r}"][k] for k in ("foot_slide_left", "foot_slide_right")}
    verdict_items = [v0["gravity"]["passed"], v0["support_drift_r0"]["passed"], v0["support_drift_r1"]["passed"], v0["nonhand_penetration_r0"]["passed"],
                     v0["nonhand_penetration_r1"]["passed"], v0["robot_robot_clearance"]["passed"], contact_ok, gates["main_motion"]["passed"]]
    failed = [n for n, ok in zip(("V0 gravity", "V0 support drift r0", "V0 support drift r1", "V0 non-hand penetration r0", "V0 non-hand penetration r1",
                                  "V0 robot-robot clearance", "section-4 contact retention >= 0.99", "main motion"), verdict_items) if not ok]
    return dict(applicable=not failed, failed=failed, V0=v0, contact_retention=ret, condition_frames=cond_frames, contact_acceptance_passed=contact_ok,
                main_motion=gates["main_motion"], max_penetration_cm={f"r{r}": round(100 * float(max(Pn[:, r].max(), 0)), 3) for r in (0, 1)},
                frames_penetrating={f"r{r}": round(float((Pn[:, r] > 0).mean()), 4) for r in (0, 1)},
                robot_robot_min_distance_cm=round(100 * float(rr.min()), 3), robot_robot_intersect_frames=int((rr < 0).sum()),
                contact_position_error_3d=ce, foot_slide=foot, posture={k: {kk: v[kk] for kk in ("torso_tilt_deg", "pelvis_height_cm", "elbow_angle_left_deg", "elbow_angle_right_deg",
                                                                                         "elbow_position_error_left_cm", "elbow_position_error_right_cm", "contact_frames")} for k, v in post.items()},
                solves={f"r{r}": dict(seconds=recs[r]["seconds"], finite=recs[r]["finite"], shift_xy_m=recs[r]["shift_xy_m"], object_locked_max_err=recs[r]["object_locked_max_err"]) for r in (0, 1)},
                target_urdf=tgt_urdf, target_urdf_sha256=sha(tgt_urdf), frames=T)


def write_pair(cand_dir, out):
    """two solves -> one pair npz in the step-7 format (robot_qpos [T, 2, 36], object_qpos (pos, wxyz))."""
    recs = [json.loads((cand_dir / f"r{r}" / "run_record.json").read_text()) for r in (0, 1)]
    zs = [np.load(cand_dir / f"r{r}" / f"{recs[r]['tag']}_new_qpos.npz") for r in (0, 1)]
    T = min(len(zs[0]["qpos"]), len(zs[1]["qpos"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, robot_qpos=np.stack([zs[0]["qpos"][:T, :36], zs[1]["qpos"][:T, :36]], 1), object_qpos=zs[0]["object_qpos_mj"][:T], fps=np.int64(30),
             m=recs[0]["m"], S=np.asarray(recs[0]["S"]), box_or_table_urdf=recs[0]["target"]["urdf"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand-dir", type=Path, required=True)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--gates", type=Path, required=True, help="json with gravity_deg and main_motion of this candidate / level")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pair-out", type=Path, default=None)
    a = ap.parse_args()
    res = evaluate(a.cand_dir, a.source, json.loads(a.gates.read_text()))
    if a.pair_out is not None:
        p = write_pair(a.cand_dir, a.pair_out)
        res.update(pair=str(p), pair_sha256=sha(p))
    res.update(rules=__doc__, command=" ".join(sys.argv))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)) + "\n")
    print(json.dumps(dict(applicable=res["applicable"], failed=res["failed"], retention=res["contact_retention"], rr=res["robot_robot_min_distance_cm"])), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
