#!/usr/bin/env python3
"""T12-diag: why robot 0's hands mostly hover above the table edge (D Table, paper policy). Read only, CPU, saved data only.

Parts (rules fixed here before numbers were looked at):
1. Force check: the Isaac trajectories of Table and Bin carry hand_object_normal_force_w [row, robot, hand(L, R), xyz] = the
   object-hand contact sensor's filtered force matrix (filters: Robot left / right hand, then Robot_1 left / right hand; the
   evaluator saves it after each physics step together with the states). A hand is in force contact when |F| > 1 N (the
   threshold of the reward's contact detection, run_config bucket_reward_contract.contact_threshold_n). Per episode, over the
   T12-v2 window (reference first hand contact .. last executed row): per-robot fraction of rows with any hand in force contact,
   both robots at once, and the row-by-row agreement with the T12-v2 distance rule (logs_system1/t12v2/per_step[_physx],
   distance >= -1.5 cm; URDF and PhysX geometry). MuJoCo trajectories have no force record.
2. Reference: Table reference (core4d_pair_runtime_fps50.npz through the paper loader), every frame of the reference contact
   window (first hand contact .. last frame): signed distance (m, positive = penetration) of each robot's left / right hand to
   the table (contact_stats geometry: URDF hands vs union of the 5 boxes; and PhysX hand hulls), the closest-hand distance per
   robot, percentiles, share >= -1.5 cm, share > 0 and the max penetration; per reference phase (T10b phase labels) as well.
2b. Where the hands are relative to the tabletop (object frame: y is up; tabletop = the first URDF box, top face at
   y = 0.2705 m, bottom face at y = 0.2357 m): for each hand, the hand sample point closest to the tabletop box and the face of
   the box nearest to it (top / bottom / side), the hand point's height above the top face, and whether it lies over the
   tabletop footprint; rows where the hand is within 1.5 cm of the table (URDF rule) in the reference's active phases (lift,
   push, carry, lower) and over the same frames in the Isaac episodes.
3. Tracking error: every Table episode (Isaac and MuJoCo, seeds 721-723, +/-2 and +/-8), every executed row of the window: hand
   link origin (FK of the saved state) vs the reference hand link origin (FK of the reference at frame min(k, N-1), moved by the
   constant world offset saved reference object - reference object, checked constant): 3-D error and vertical error
   (actual - reference z), in the world and in the object frame (hand relative to the actual table vs reference hand relative
   to the reference table); also the pelvis and the mean over all 23 collision links for comparison. Per episode: mean over
   the window; summary per simulator x level x robot.

    OMP_NUM_THREADS=1 hssim_s1/bin/python -B system1/t12diag/diag_table.py --out results_system1/T12_diag --workers 24
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "system1/t12v2"))
import contact_stats as cs  # noqa: E402

FORCE_N = 1.0
G = {}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def pct(x, qs=(5, 25, 50, 75, 95)):
    x = np.asarray(x, dtype=float)
    return {f"p{q}": round(float(np.percentile(x, q)), 4) for q in qs}


def force_job(job):
    """part 1 for one Isaac run."""
    s = json.loads((job["eval_dir"] / "summary.json").read_text())
    tag = f"{job['task']}_seed{job['seed']}_pm{job['level']}_isaac"
    flags = {g: np.load(ROOT / f"logs_system1/t12v2/{d}/{tag}.npz") for g, d in (("urdf", "per_step"), ("physx", "per_step_physx"))}
    links = list(flags["urdf"]["links"])
    assert links[:2] == list(cs.HANDS)
    k0 = G["k0"][job["task"]]
    rows = []
    for e in s["episodes"]:
        with np.load(e["trajectory"]) as z:
            F = z["hand_object_normal_force_w"]
        n = cs.executed_rows(e, len(F))
        Fm = np.linalg.norm(F[:n], axis=-1)  # (n, robot, hand)
        fc = Fm > FORCE_N
        w = slice(k0, n)
        if n <= k0:
            continue
        r = dict(task=job["task"], seed=job["seed"], level_cm=job["level"], state_id=e["state_id"], kind=e["kind"], completed=e["completed_reference"],
                 window_rows=n - k0)
        for ri in range(2):
            r[f"force_frac_r{ri}_hand"] = round(float(fc[w, ri].any(-1).mean()), 4)
            on = Fm[w, ri][fc[w, ri]]
            r[f"force_mean_N_r{ri}_when_contact"] = round(float(on.mean()), 2) if on.size else None
            r[f"force_p95_N_r{ri}"] = round(float(np.percentile(Fm[w, ri].max(-1), 95)), 2)
        r["force_frac_both"] = round(float((fc[w, 0].any(-1) & fc[w, 1].any(-1)).mean()), 4)
        for g in ("urdf", "physx"):
            dc = flags[g][e["state_id"]][:, :, :2]  # (n, robot, hand) distance contact
            assert len(dc) == n
            for ri in range(2):
                a, b = fc[w, ri].any(-1), dc[w, ri].any(-1)
                r[f"dist_{g}_frac_r{ri}_hand"] = round(float(b.mean()), 4)
                r[f"agree_{g}_r{ri}"] = round(float((a == b).mean()), 4)
                r[f"force_without_dist_{g}_r{ri}"] = round(float((a & ~b).mean()), 4)
                r[f"dist_without_force_{g}_r{ri}"] = round(float((~a & b).mean()), 4)
        rows.append(r)
    return rows


def track_job(job):
    """part 3 for one Table run."""
    robot, obj, R = G["robot"], G["obj"], G["ref"]
    s = json.loads((job["eval_dir"] / "summary.json").read_text())
    k0 = G["k0"]["table"]
    N = R["N"]
    rows = []
    for e in s["episodes"]:
        with np.load(e["trajectory"]) as z:
            a = {k: z[k] for k in ("root_pos", "root_quat_xyzw", "dof_pos", "object_pos_w", "object_quat_xyzw", "reference_object_pos_w", "reference_frame")}
        n = cs.executed_rows(e, len(a["dof_pos"]))
        if n <= k0:
            continue
        f = np.minimum(np.arange(n), N - 1)
        off = a["reference_object_pos_w"][:n] - R["obj_pos"][f]
        assert np.abs(off - off[0]).max() < 1e-4, "world offset not constant"
        off = off[0]
        Ta = robot.fk(a["root_pos"][:n], a["root_quat_xyzw"][:n], a["dof_pos"][:n])
        Tr = robot.fk(R["root_pos"][f] + off, R["root_quat"][f], R["dof"][f])
        Oa = np.linalg.inv(cs.pose_mat(a["object_pos_w"][:n], a["object_quat_xyzw"][:n]))[:, None]
        Or = np.linalg.inv(cs.pose_mat(R["obj_pos"][f] + off, R["obj_quat"][f]))[:, None]
        w = slice(k0, n)
        r = dict(simulator=job["simulator"], seed=job["seed"], level_cm=job["level"], state_id=e["state_id"], kind=e["kind"], completed=e["completed_reference"],
                 window_rows=n - k0)
        for ri in range(2):
            for side, h in zip(("left", "right"), cs.HANDS):
                pa, pr = Ta[h][w, ri, :3, 3], Tr[h][w, ri, :3, 3]
                qa, qr = (Oa @ Ta[h])[w, ri, :3, 3], (Or @ Tr[h])[w, ri, :3, 3]
                r[f"r{ri}_{side}_hand_err_world_cm"] = round(100 * float(np.linalg.norm(pa - pr, axis=-1).mean()), 2)
                r[f"r{ri}_{side}_hand_dz_world_cm"] = round(100 * float((pa[:, 2] - pr[:, 2]).mean()), 2)
                r[f"r{ri}_{side}_hand_err_objframe_cm"] = round(100 * float(np.linalg.norm(qa - qr, axis=-1).mean()), 2)
            r[f"r{ri}_pelvis_err_world_cm"] = round(100 * float(np.linalg.norm(Ta["pelvis"][w, ri, :3, 3] - Tr["pelvis"][w, ri, :3, 3], axis=-1).mean()), 2)
            r[f"r{ri}_all_links_err_world_cm"] = round(100 * float(np.mean([np.linalg.norm(Ta[l][w, ri, :3, 3] - Tr[l][w, ri, :3, 3], axis=-1).mean() for l in robot.links])), 2)
            r[f"r{ri}_hands_err_world_cm"] = round((r[f"r{ri}_left_hand_err_world_cm"] + r[f"r{ri}_right_hand_err_world_cm"]) / 2, 2)
            r[f"r{ri}_hands_err_objframe_cm"] = round((r[f"r{ri}_left_hand_err_objframe_cm"] + r[f"r{ri}_right_hand_err_objframe_cm"]) / 2, 2)
            r[f"r{ri}_hands_dz_world_cm"] = round((r[f"r{ri}_left_hand_dz_world_cm"] + r[f"r{ri}_right_hand_dz_world_cm"]) / 2, 2)
        rows.append(r)
    return rows


def face_stats(robot, obj, root_pos, root_quat, dof, obj_pos, obj_quat):
    """per row, robot, hand: (face id 0 top / 1 bottom / 2 side, height of the closest hand point above the top face (m),
    over-footprint flag, signed distance to the whole table (m))."""
    c, h = obj.box_c[0], obj.box_h[0]
    T = robot.fk(root_pos, root_quat, dof)
    Oi = np.linalg.inv(cs.pose_mat(obj_pos, obj_quat))[:, None]
    n = len(dof)
    face = np.zeros((n, 2, 2), int)
    hgt = np.zeros((n, 2, 2))
    over = np.zeros((n, 2, 2), bool)
    sd = np.zeros((n, 2, 2))
    for hi, hand in enumerate(cs.HANDS):
        M = Oi @ T[hand]
        P = np.einsum("nrij,pj->nrpi", M[..., :3, :3], robot.points[hand]) + M[..., None, :3, 3]  # (n, 2, P, 3) object frame
        d = np.abs(P - c) - h
        box_sd = -(np.linalg.norm(np.maximum(d, 0), axis=-1) + np.minimum(d.max(-1), 0))
        j = box_sd.argmax(-1)
        q = np.take_along_axis(P, j[..., None, None], axis=2)[..., 0, :]  # closest point (n, 2, 3)
        rel = q - c
        top, bot = c[1] + h[1], c[1] - h[1]
        inside_xz = (np.abs(rel[..., 0]) <= h[0]) & (np.abs(rel[..., 2]) <= h[2])
        over[:, :, hi] = inside_xz
        hgt[:, :, hi] = q[..., 1] - top
        face[:, :, hi] = np.where(inside_xz & (q[..., 1] >= c[1]), 0, np.where(inside_xz, 1, 2))
        sd[:, :, hi] = obj.sdf(P).max(-1)
    return face, hgt, over, sd


def summarize_faces(face, hgt, over, sd, rows):
    out = {}
    for ri in range(2):
        rec = {}
        for hi, side in enumerate(("left", "right")):
            m = sd[rows, ri, hi] >= cs.CONTACT_M
            f = face[rows, ri, hi][m]
            rec[side] = dict(rows=int(len(rows)), rows_within_1p5cm=int(m.sum()),
                             face_share_top=round(float(np.mean(f == 0)), 3) if m.any() else None,
                             face_share_bottom=round(float(np.mean(f == 1)), 3) if m.any() else None,
                             face_share_side=round(float(np.mean(f == 2)), 3) if m.any() else None,
                             median_height_above_top_cm_all_rows=round(100 * float(np.median(hgt[rows, ri, hi])), 2),
                             share_over_footprint_all_rows=round(float(np.mean(over[rows, ri, hi])), 3))
        out[f"robot{ri}"] = rec
    return out


def reference_part(robot_u, robot_p, obj_u, obj_p, R, k0, phases):
    out = dict(frames=R["N"], window=[k0, R["N"] - 1])
    labels = [None] * R["N"]
    for name, _, s, e in phases:
        for k in range(s, e + 1):
            labels[k] = name
    per_frame = {}
    for g, rb, ob in (("urdf", robot_u, obj_u), ("physx", robot_p, obj_p)):
        D = cs.link_distances(rb, ob, R["root_pos"], R["root_quat"], R["dof"], R["obj_pos"], R["obj_quat"])[:, :, :2]  # (N, robot, hand)
        D = np.where(np.isfinite(D), D, -1.0)  # culled = far (> 1.5 cm); shown as -1 m
        per_frame[g] = D
        res = {}
        for ri in range(2):
            closest = D[k0:, ri].max(-1)
            rec = dict(closest_hand=dict(**pct(closest), share_within_1p5cm=round(float(np.mean(closest >= cs.CONTACT_M)), 4),
                                         share_penetrating=round(float(np.mean(closest > 0)), 4), max_penetration_cm=round(100 * float(closest.max()), 2)))
            for hi, side in enumerate(("left", "right")):
                d = D[k0:, ri, hi]
                rec[f"{side}_hand"] = dict(**pct(d), share_within_1p5cm=round(float(np.mean(d >= cs.CONTACT_M)), 4), share_penetrating=round(float(np.mean(d > 0)), 4),
                                           max_penetration_cm=round(100 * float(d.max()), 2))
            ph = {}
            for name in dict.fromkeys(l for l in labels[k0:] if l):
                idx = [k for k in range(k0, R["N"]) if labels[k] == name]
                c = D[idx, ri].max(-1)
                ph[name] = dict(frames=len(idx), median_cm=round(100 * float(np.median(c)), 2), share_within_1p5cm=round(float(np.mean(c >= cs.CONTACT_M)), 4),
                                share_penetrating=round(float(np.mean(c > 0)), 4))
            rec["per_phase_closest_hand"] = ph
            res[f"robot{ri}"] = rec
        out[g] = res
    return out, per_frame, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    t0 = time.time()
    a.out.mkdir(parents=True, exist_ok=True)
    refs = cs.load_references()
    robot_u, robot_p = cs.Robot("urdf"), cs.Robot("physx")
    runs = cs.runs()
    obj_urdf = {t: cs.DATA / json.loads((next(r for r in runs if r["task"] == t)["eval_dir"] / "summary.json").read_text())["object_urdf"] for t in ("table", "bucket")}
    obj_u = cs.Object(obj_urdf["table"])
    obj_p = cs.Object(obj_urdf["table"], geometry="physx", task="table")
    k0 = {}
    for t in ("table", "bucket"):  # the T12-v2 windows (URDF rule)
        prov = json.loads((ROOT / "results_system1/T12_videos_v2/contact_provenance.json").read_text())
        k0[t] = prov["references"][t]["first_contact_frame"]
    G.update(robot=robot_u, obj=obj_u, ref=refs["table"], k0=k0)
    # part 2
    phases = json.loads((ROOT / "results_system1/T10b_friction_margin/retrain/pilot_progress/references.json").read_text())["d_table_seed721_rand"]["phase_segments"]
    ref_rec, per_frame, labels = reference_part(robot_u, robot_p, obj_u, obj_p, refs["table"], k0["table"], phases)
    ref_rec.update(reference_file=refs["table"]["meta"]["file"], reference_sha256=refs["table"]["meta"]["sha256"], phase_rule="T10b phase labels (references.json d_table)",
                   distance_convention="signed distance in m, positive = penetration into the table; culled hands (farther than about 1.5 cm + link radius) = -1 m")
    (a.out / "reference_hand_table_distance.json").write_text(json.dumps(ref_rec, indent=1) + "\n")
    with (a.out / "reference_hand_table_distance_per_frame.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "t_s", "phase", *[f"{g}_r{ri}_{side}_cm" for g in ("urdf", "physx") for ri in range(2) for side in ("left", "right")]])
        for k in range(refs["table"]["N"]):
            w.writerow([k, round(k / 50, 2), labels[k], *[round(100 * float(per_frame[g][k, ri, hi]), 2) for g in ("urdf", "physx") for ri in range(2) for hi in range(2)]])
    active = [k for k in range(k0["table"], refs["table"]["N"]) if labels[k] in ("lift", "push", "carry", "lower")]
    R = refs["table"]
    fr = face_stats(robot_u, obj_u, R["root_pos"], R["root_quat"], R["dof"], R["obj_pos"], R["obj_quat"])
    faces = dict(rule="2b in the docstring; active reference phases lift / push / carry / lower", active_frames=len(active),
                 reference=summarize_faces(*fr, active), isaac={})
    for r in [r for r in runs if r["task"] == "table" and r["simulator"] == "isaac"]:
        s = json.loads((r["eval_dir"] / "summary.json").read_text())
        for e in s["episodes"]:
            if e["kind"] != "perturbed":
                continue
            with np.load(e["trajectory"]) as z:
                tr = {k: z[k] for k in ("root_pos", "root_quat_xyzw", "dof_pos", "object_pos_w", "object_quat_xyzw")}
            n = cs.executed_rows(e, len(tr["dof_pos"]))
            rows = [k for k in active if k < n]
            if not rows:
                continue
            f = face_stats(robot_u, obj_u, tr["root_pos"][:n], tr["root_quat_xyzw"][:n], tr["dof_pos"][:n], tr["object_pos_w"][:n], tr["object_quat_xyzw"][:n])
            faces["isaac"].setdefault(f"pm{r['level']}", []).append((f, rows))
    agg = {}
    for lvl, items in faces["isaac"].items():
        F = [np.concatenate([x[i][rows] for x, rows in items]) for i in range(4)]
        agg[lvl] = summarize_faces(*F, np.arange(len(F[0])))
        agg[lvl]["episodes"] = len(items)
    faces["isaac"] = agg
    (a.out / "hand_face_on_tabletop.json").write_text(json.dumps(faces, indent=1) + "\n")
    print(f"{time.time() - t0:.0f}s reference done", flush=True)
    # parts 1 and 3
    f_jobs = [r for r in runs if r["task"] in ("table", "bucket") and r["simulator"] == "isaac"]
    t_jobs = [r for r in runs if r["task"] == "table"]
    with Pool(a.workers) as pool:
        frows = [x for rs in pool.map(force_job, f_jobs) for x in rs]
        trows = [x for rs in pool.map(track_job, t_jobs) for x in rs]
    for name, rows in (("force_vs_distance_episodes.csv", frows), ("tracking_error_episodes.csv", trows)):
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with (a.out / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    summ = []
    for t in ("table", "bucket"):
        for lvl in (2, 8):
            for grp in ("completed", "failed", "all"):
                sel = [r for r in frows if r["task"] == t and r["level_cm"] == lvl and r["kind"] == "perturbed" and (grp == "all" or r["completed"] == (grp == "completed"))]
                row = dict(part="force", task=t, simulator="isaac", level_cm=lvl, episodes=grp, n=len(sel))
                for k in [k for k in (frows[0] if frows else {}) if k.startswith(("force_", "dist_", "agree_"))]:
                    v = [r[k] for r in sel if r.get(k) is not None]
                    row[k] = round(float(np.mean(v)), 4) if v else None
                summ.append(row)
    tsum = []
    for sim in ("isaac", "mujoco"):
        for lvl in (2, 8):
            for grp in ("completed", "failed", "all"):
                sel = [r for r in trows if r["simulator"] == sim and r["level_cm"] == lvl and r["kind"] == "perturbed" and (grp == "all" or r["completed"] == (grp == "completed"))]
                row = dict(part="tracking", task="table", simulator=sim, level_cm=lvl, episodes=grp, n=len(sel))
                for k in [k for k in trows[0] if k.startswith(("r0_", "r1_"))]:
                    v = [r[k] for r in sel]
                    row[k] = round(float(np.mean(v)), 3) if v else None
                tsum.append(row)
    for name, rows in (("force_vs_distance_summary.csv", summ), ("tracking_error_summary.csv", tsum)):
        with (a.out / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    (a.out / "diag_provenance.json").write_text(json.dumps(dict(script=str(Path(__file__).relative_to(ROOT)), script_sha256=sha(__file__),
                                                                 contact_stats_sha256=sha(ROOT / "system1/t12v2/contact_stats.py"), force_threshold_n=FORCE_N,
                                                                 windows_first_contact_frame=k0, reference=refs["table"]["meta"],
                                                                 runs=[dict(task=r["task"], seed=r["seed"], level_cm=r["level"], simulator=r["simulator"], eval_dir=str(r["eval_dir"].relative_to(ROOT)),
                                                                            summary_sha256=sha(r["eval_dir"] / "summary.json")) for r in runs if r["task"] in ("table", "bucket")],
                                                                 wall_s=round(time.time() - t0)), indent=1) + "\n")
    print(f"{time.time() - t0:.0f}s done: force episodes {len(frows)}, tracking episodes {len(trows)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
