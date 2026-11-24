#!/usr/bin/env python3
"""T14 step 7, layer 1: the existing role-isometry check (scripts/reference_geometry_pilot.py source interactions +
scripts/reference_geometry_round2_*.py correspondence, per-role screen and pair test, frozen thresholds V0) applied to a
single-robot solve of a box-approximated object. Only the correspondence construction is generalised (the original
construct() is written for the Push / Pull tables: a "tabletop" part plus legs, stretch axis fixed per task):

  * source = the solve's box (BoxUnion of box_assets' URDF, one part) and robot / object qpos; the source interactions (hand
    sample distances, stable face windows: >= 5 % of 256 hand samples within 3 cm for >= 0.1 s) and the tie-safe weights are
    the pilot's and round 2's, unchanged;
  * target at 1.0 x = the box lengthened along one horizontal box axis k to twice its length (two copies end to end); role 0's
    region = the low end [lo, lo + w], role 1's = the high end [hi - w, hi] (w = the source length along k); both horizontal
    axes (box x, box y) are tried; vertical stretching is not;
  * candidates per role = every proper signed permutation O (parity +1: two robots of the same chirality; mirrored partners
    are not tried, as in the Push case) whose box dimensions match the region, whose active source faces map onto real
    exterior faces of the clipped target (no cut walls) with matching corners; fit = weighted Procrustes (round 2);
  * per-role screen = round 2's audit_role with V0; pair test = both roles pass and the cross-robot convex-proxy clearance
    is >= -0.01 m (V0 pair_proxy_penetration_m); selection = round 2's rule (passing pairs: smallest joint weighted fit
    error, ties by ID; none passing: diagnostic, fewest failures).

    sonic_s1/bin/python -B system1/t14/objects/role_check_box.py --solve logs_system1/t14/step7/solves/single_largeboxc_C --out <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src/holosoma_retargeting"))
from reference_geometry_math import BoxUnion, map_robot, signed_permutations, continuous_windows  # noqa: E402
from reference_geometry_pilot import RobotFK, G1_29DOF_JOINT_NAMES, g1_sagittal_joint_map, default, THRESHOLDS  # noqa: E402
from reference_geometry_round2_correspondence import clipped_faces, tie_safe_weights, fit_candidates  # noqa: E402
from reference_geometry_round2_checks import V0, CollisionProxy, surface_curves, support_metrics, motion_metrics, summaries  # noqa: E402
from run_reference_geometry_round2 import audit_role  # noqa: E402
from holosoma_retargeting.dual_a1_layout import quaternion_wxyz_to_matrix  # noqa: E402


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, default=default, ensure_ascii=False, indent=1, allow_nan=True) + "\n")


def box_urdf(path, name, size):
    path.parent.mkdir(parents=True, exist_ok=True)
    s = " ".join(f"{x:.9g}" for x in size)
    path.write_text(f'<?xml version="1.0" ?>\n<robot name="{name}"><link name="{name}_link"><visual name="{name}_box"><origin xyz="0 0 0" rpy="0 0 0"/>'
                    f'<geometry><box size="{s}"/></geometry></visual><collision name="{name}_box"><origin xyz="0 0 0" rpy="0 0 0"/>'
                    f'<geometry><box size="{s}"/></geometry></collision></link></robot>\n')  # visual: for the figures (BoxUnion reads collisions only)
    return path


def source_interactions(q, obj, fk, src, fps):
    """the pilot's analyze() loop: hand fit samples vs the source boxes."""
    transforms = fk.poses(q, fk.hands)
    rot = quaternion_wxyz_to_matrix(obj[:, 3:])
    curves, records = {}, []
    for hand in fk.hands:
        tf = transforms[hand]
        world = np.einsum("tij,pj->tpi", tf[:, :3, :3], fk.fit[hand]) + tf[:, :3, 3][:, None]
        local = np.einsum("tji,tpj->tpi", rot, world - obj[:, :3][:, None])  # R^T (x - p), as the pilot
        unsigned, signed, face, near, ties = src.distances(local, with_ties=True)
        frac = np.mean(unsigned <= THRESHOLDS["unsigned_near_m"], axis=1)
        windows = continuous_windows(frac >= THRESHOLDS["near_sample_fraction"], int(np.ceil(THRESHOLDS["min_duration_s"] * fps)))
        face_records = []
        for idx, f in enumerate(src.faces):
            fraction = np.mean((unsigned <= THRESHOLDS["unsigned_near_m"]) & ((ties & (np.uint64(1) << np.uint64(idx))) != 0), axis=1)
            for a, b in continuous_windows(fraction >= THRESHOLDS["near_sample_fraction"], int(np.ceil(THRESHOLDS["min_duration_s"] * fps))):
                face_records.append(dict(face_index=idx, face_id=f["id"], part_id=f["part_id"], start=a, end_exclusive=b, duration_s=(b - a) / fps,
                                         mean_sample_fraction=float(fraction[a:b].mean()), anchors=f["corners"]))
        records.append(dict(hand=hand, source_windows=windows, stable_face_windows=face_records, min_unsigned_distance_m=float(unsigned.min()),
                            max_penetration_m=float(max(0, -signed.min())), max_near_fraction=float(frac.max())))
        curves[hand + "_signed_distance_m"] = signed
        curves[hand + "_competing_face_bits"] = ties
    return dict(records=records, thresholds=THRESHOLDS), curves


def construct_box(src, target, weighted, axis):
    """round-2 construct() for one source box and its lengthened target (see module doc)."""
    slab, wide = src.parts[0], target.parts[0]
    width = slab["size"][axis]
    intervals = [[wide["lo"][axis], wide["lo"][axis] + width], [wide["hi"][axis] - width, wide["hi"][axis]]]
    active_ids = sorted({w["face_id"] for w in weighted["windows"]})
    source_faces = {f["id"]: f for f in src.faces}
    attempts, valid, regions = [], [], []
    for role, interval in enumerate(intervals):
        rf = clipped_faces(target, axis, interval)
        region_center = wide["center"].copy()
        region_center[axis] = sum(interval) / 2
        region_size = wide["size"].copy()
        region_size[axis] = width
        regions.append(dict(role=role, axis=axis, interval_m=interval, parity=1, faces=rf))
        for order, O in enumerate(signed_permutations(1)):
            cid = f"axis{axis}/role{role}/axes{order:02d}"
            rejected = []
            if not np.allclose(abs(O) @ slab["size"], region_size, atol=1e-10):
                rejected.append("box_metric_dimensions")
            offset = region_center - O @ slab["center"]
            face_map, target_anchors = {}, np.empty_like(weighted["source_anchors"])
            if not rejected:
                for fid in active_ids:
                    sf = source_faces[fid]
                    normal = O @ sf["normal"]
                    matches = [f for f in rf if np.allclose(f["normal"], normal, atol=1e-10)]
                    if len(matches) != 1:
                        rejected.append("missing_real_exterior_face:" + fid)
                        continue
                    tf = matches[0]
                    expected = sf["corners"] @ O.T + offset
                    dist = np.linalg.norm(expected[:, None] - tf["corners"][None], axis=-1)
                    idx = dist.argmin(1)
                    if len(set(idx.tolist())) != 4 or dist[np.arange(4), idx].max() > 1e-9:
                        rejected.append("finite_corner_or_adjacency_mismatch:" + fid)
                        continue
                    face_map[fid] = dict(target_face_id=tf["id"], source_normal=sf["normal"], target_normal=tf["normal"], source_corners=sf["corners"],
                                         target_corners=tf["corners"][idx], corner_permutation=idx, target_patches=tf["patches"])
                if not rejected:
                    for w in weighted["windows"]:
                        s0, e0 = w["anchor_slice"]
                        target_anchors[s0:e0] = face_map[w["face_id"]]["target_corners"]
            attempts.append(dict(id=cid, role=role, parity=1, O=O, rejection_reasons=rejected,
                                 status="geometry_correspondence_accepted" if not rejected else "geometry_pruned"))
            if not rejected:
                valid.append(dict(id=cid, role=role, parity=1, part_map={slab["id"]: wide["id"]}, face_map=face_map, target_anchors=target_anchors, region_index=role, O=O))
    return dict(axis=axis, source_length_m=float(width), regions=regions, weighted_source=weighted, attempts=attempts, candidates=valid,
                attempted_per_role=[sum(a["role"] == i for a in attempts) for i in (0, 1)], accepted_per_role=[sum(a["role"] == i for a in valid) for i in (0, 1)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solve", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rec = json.loads((a.solve / "run_record.json").read_text())
    name, tag = rec["object"], rec["tag"]
    z = np.load(a.solve / f"{tag}_new_qpos.npz")
    qq = z["qpos"].astype(np.float64)
    source, obj, fps = qq[:, :36], qq[:, 36:43], int(z["fps"])  # obj = (pos, quat wxyz)
    a.out.mkdir(parents=True, exist_ok=True)
    src = BoxUnion(rec["box"]["urdf"])
    size = src.parts[0]["size"]
    fk = RobotFK()
    interactions, icurves = source_interactions(source, obj, fk, src, fps)
    write(a.out / "source_interactions.json", interactions)
    weighted = tie_safe_weights(interactions, icurves)
    index, sign = g1_sagittal_joint_map(G1_29DOF_JOINT_NAMES)
    result = dict(object=name, tag=tag, solve=str(a.solve), rules=__doc__, V0=V0, box_size_m=size.tolist(), axes={})
    all_pairs, all_cands = [], {}
    for axis in (0, 1):
        tsize = size.copy()
        tsize[axis] *= 2
        turdf = box_urdf(a.out / f"target_axis{axis}.urdf", name, tsize)
        target = BoxUnion(turdf)
        spec = construct_box(src, target, weighted, axis)
        spec["weighted_source"] = weighted
        cands = fit_candidates(spec)
        source_curves, source_poses = surface_curves(source, obj, fk, src, spec)
        src_engine = CollisionProxy(fk, src)
        source_collision, _ = src_engine.evaluate([source_poses], obj)
        source_curves.update(support_metrics(source_curves, source_curves, fps))
        source_curves.update(motion_metrics(source, source, obj, fps, np.eye(3)))
        source_curves["nonhand_object_proxy_clearance_m"] = source_collision["nonhand_object_clearance_m"][:, 0]
        source_curves["hand_object_proxy_clearance_m"] = source_collision["hand_object_convex_clearance_m"][:, 0]
        engine = CollisionProxy(fk, target)
        audits = {}
        for c in cands:
            if c["fit"]["status"] != "solved":
                audits[c["id"]] = dict(id=c["id"], passed=False, failures=["degenerate_fit"])
                continue
            q = map_robot(source, obj, c["fit"]["Q"], c["fit"]["c"], c["parity"], index, sign)
            audit, curves, poses = audit_role(q, source, obj, fps, fk, target, engine, spec, c, source_curves, source_collision, source_poses)
            audit["fit"] = dict(Q=c["fit"]["Q"], c=c["fit"]["c"], weighted_rmse_m=c["fit"]["weighted_rmse_m"])
            audits[c["id"]] = audit
            all_cands[c["id"]] = (c, poses, audit)
            print(name, c["id"], "PASS" if audit["passed"] else "FAIL " + ",".join(audit["failures"]), flush=True)
        pairs = []
        for c0, c1 in product([c for c in cands if c["role"] == 0], [c for c in cands if c["role"] == 1]):
            item = dict(id=c0["id"] + " || " + c1["id"], axis=axis, roles=[c0["id"], c1["id"]],
                        joint_weighted_rmse_m=float(np.sqrt((c0["fit"]["weighted_rmse_m"] ** 2 + c1["fit"]["weighted_rmse_m"] ** 2) / 2)), failures=[])
            for c in (c0, c1):
                if not audits[c["id"]]["passed"]:
                    item["failures"].append("role_screen_failed:" + c["id"])
            if not item["failures"]:
                coll, _ = engine.evaluate([all_cands[c0["id"]][1], all_cands[c1["id"]][1]], obj)
                item["robot_robot_min_proxy_clearance_m"] = float(coll["robot_robot_clearance_m"].min())
                if item["robot_robot_min_proxy_clearance_m"] < -V0["pair_proxy_penetration_m"]:
                    item["failures"].append("robot_robot_convex_proxy_penetration")
            item["passed"] = not item["failures"]
            item["role_failures"] = {k: audits[k]["failures"] for k in item["roles"]}
            pairs.append(item)
        all_pairs += pairs
        result["axes"][str(axis)] = dict(target_size_m=tsize.tolist(), target_urdf=str(turdf), attempted_per_role=spec["attempted_per_role"],
                                         accepted_per_role=spec["accepted_per_role"], attempts=[dict(id=x["id"], O=x["O"], status=x["status"], rejection_reasons=x["rejection_reasons"]) for x in spec["attempts"]],
                                         candidates={k: dict(passed=v["passed"], failures=v["failures"], fit=v.get("fit")) for k, v in audits.items()},
                                         pairs=len(pairs), passing_pairs=sum(p["passed"] for p in pairs), pair_records=pairs)
    ok = [p for p in all_pairs if p["passed"]]
    if ok:
        chosen = sorted(ok, key=lambda p: (round(p["joint_weighted_rmse_m"] / V0["fit_tie_tolerance_m"]), p["id"]))[0]
        status = "passes_source_relative_geometry"
    elif all_pairs:
        chosen = sorted(all_pairs, key=lambda p: (sum(len(all_cands[k][2]["failures"]) for k in p["roles"] if k in all_cands) + len(p["failures"]),
                                                 round(p["joint_weighted_rmse_m"] / V0["fit_tie_tolerance_m"]), p["id"]))[0]
        status = "diagnostic_only"
    else:
        chosen, status = None, "no_candidate_pair"
    result.update(status=status, pairs_total=len(all_pairs), passing_pairs_total=len(ok), selected=chosen,
                  selected_roles=None if chosen is None else [dict(id=k, Q=all_cands[k][0]["fit"]["Q"], c=all_cands[k][0]["fit"]["c"], parity=1) for k in chosen["roles"]],
                  source_interaction_windows={r["hand"]: [(w["face_id"], w["start"], w["end_exclusive"]) for w in r["stable_face_windows"]] for r in interactions["records"]})
    write(a.out / "role_check.json", result)
    print(name, "status", status, "pairs", len(all_pairs), "passing", len(ok), "selected", None if chosen is None else chosen["id"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
