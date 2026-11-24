#!/usr/bin/env python3
"""T14 round 4, section 三 diagnostics (CPU, read only). Same geometry and measures as eval_r3 (exact distances).

  fail-frames  For each reference: every CoCarry condition frame (person's hand within 0.10 m of the captured mesh and the
               object faster than 0.03 m/s) in which the robot hand is not retained (hand signed distance < -0.02 m):
               robot, hand, robot hand gap (cm, exact hull distance), human hand gap (cm; "真接触" <= 1.5 cm, "靠近"
               1.5-10 cm), phase of the object motion at that source frame:
                 抬起 v_z > +0.05 m/s; 放下 v_z < -0.05 m/s; 转动 angular speed > 30 deg/s (and |v_z| <= 0.05); 搬运 otherwise.
  k08          Table: why the contact-position error (tangential distance of the two surface projections, human-contact
               frames only) is small while retention is low: per robot / hand in human-contact frames the palm point's
               distance to the surface (normal distance, which the contact-position error ignores), and how many failing
               retention frames are "靠近" frames (not counted by the contact-position error).
  a1           Old Bin / Chair references before and after the A1 wrist post-process: max penetration (exact, simulator
               geometry of the 1.0x training object), per robot, with the link, frame, and the frames where A1 changed it.

    sonic_s1/bin/python -B system1/t14/diag_r4.py fail-frames --out <dir>
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "system1/t14"))
sys.path.insert(0, str(ROOT / "system1/t12v2"))
import eval_core4d_hulls as eh  # noqa: E402

R3 = ROOT / "exports_system1/T14_references/r3"
PAIR = dict(table="20231030_001_desk001_move_new_fps30.npz", bin="bin_new_fps30.npz", chair="chair_new_fps30.npz")
PREV = Path("/home/kevin/holosoma-core4d-base/logs/Core4DPreviews")  # read only


def phases(task):
    from scipy.spatial.transform import Rotation
    seq = Path("/home/kevin/datasets/CORE4D-V1-full/CORE4D_Real/human_object_motions") / eh.TASKS[task]["raw"]
    To = np.load(seq / "smooth_objposes.npy")
    x = To[:, :3, 3]
    v = np.gradient(x, axis=0) * 15.0
    R = Rotation.from_matrix(To[:, :3, :3])
    w = np.zeros(len(To))
    w[1:-1] = (R[2:] * R[:-2].inv()).magnitude() / (2 / 15.0)
    w[0], w[-1] = w[1], w[-2]
    # CORE4D raw is Y-up
    vz = v[:, 1]
    lab = np.where(vz > 0.05, "抬起", np.where(vz < -0.05, "放下", np.where(np.degrees(w) > 30.0, "转动", "搬运")))
    return lab, vz, np.degrees(w)


def hand_sd(task, path, k):
    z = np.load(path)
    pair, oq = z["robot_qpos"].astype(np.float64), z["object_qpos"].astype(np.float64)
    H, P = eh.ObjectRobotDistance(eh.dump_object_meshes(task, k), eh.cs.Robot("physx"), depth="exact").frames(pair, oq)
    return H, P, pair, oq


def fail_frames(a):
    refs = [(t, k, R3 / f"{t}_k{k.replace('.', 'p')}_a10" / PAIR[t]) for t in ("bin", "chair") for k in ("0.8", "0.9", "1.0", "1.1", "1.2")]
    refs += [("table", k, R3 / f"table_k{k.replace('.', 'p')}_a10" / PAIR["table"]) for k in ("0.8", "0.9")]
    rows, summary = [], {}
    cache = {}
    for task, k, path in refs:
        if task not in cache:
            cache[task] = (eh.human_hand_sd(task), phases(task))
        (h_sd, speed), (lab, vz, wdeg) = cache[task]
        H, _, _, _ = hand_sd(task, path, float(k))
        T = len(H)
        t15 = np.arange(min(len(h_sd), (T + 1) // 2))
        cond = (h_sd[t15] >= -0.10) & (speed[t15, None, None] > 0.03)
        robo = H[2 * t15] >= -0.02
        fail = cond & ~robo
        s = dict(condition_frames=int(cond.sum()), failing=int(fail.sum()), retention=round(float(robo[cond].mean()), 4), by=[])
        for r in (0, 1):
            for hi, side in enumerate(("left", "right")):
                for t in np.flatnonzero(fail[:, r, hi]):
                    hg = -100 * h_sd[t, r, hi]
                    rows.append(dict(task=task, level=k, frame_15hz=int(t), frame_30hz=int(2 * t), time_s=round(t / 15, 3), robot=r, hand=side,
                                     robot_hand_gap_cm=round(-100 * float(H[2 * t, r, hi]), 2), human_hand_gap_cm=round(float(hg), 2),
                                     human_bucket="真接触" if hg <= 1.5 else "靠近", phase=str(lab[t]), object_vz_mps=round(float(vz[t]), 3),
                                     object_angular_dps=round(float(wdeg[t]), 1)))
                f_r = [x for x in rows if x["task"] == task and x["level"] == k and x["robot"] == r and x["hand"] == side]
                if f_r:
                    s["by"].append(dict(robot=r, hand=side, failing=len(f_r), condition=int(cond[:, r, hi].sum()),
                                        真接触=sum(x["human_bucket"] == "真接触" for x in f_r), 靠近=sum(x["human_bucket"] == "靠近" for x in f_r),
                                        phase={p: sum(x["phase"] == p for x in f_r) for p in ("抬起", "搬运", "转动", "放下")},
                                        robot_gap_cm_median=round(float(np.median([x["robot_hand_gap_cm"] for x in f_r])), 2),
                                        robot_gap_cm_max=round(float(np.max([x["robot_hand_gap_cm"] for x in f_r])), 2),
                                        frames_s=[round(min(x["time_s"] for x in f_r), 2), round(max(x["time_s"] for x in f_r), 2)]))
        summary[f"{task}_k{k}"] = s
        print(task, k, json.dumps(s, ensure_ascii=False), flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    with open(a.out / "failing_frames.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (a.out / "failing_frames_summary.json").write_text(json.dumps(dict(rules=__doc__, refs={k: str(p) for _, k, p in refs}, summary=summary), indent=1, ensure_ascii=False) + "\n")


def k08(a):
    import contact_gap as cg
    from object_surface import UnionSurface
    from sim_scene import object_pieces
    robot = eh.cs.Robot("physx")
    offs = cg.palm_offsets()
    human = cg.human_core4d("table")
    h_sd, speed = eh.human_hand_sd("table")
    out = {}
    for k in ("0.8", "1.0"):
        path = R3 / f"table_k{k.replace('.', 'p')}_a10" / PAIR["table"]
        H, _, pair, oq = hand_sd("table", path, float(k))
        flags15 = human[0]
        t15 = np.arange(min(len(flags15), (len(pair) + 1) // 2))
        Pl = cg.robot_palms_local(robot, pair[2 * t15], oq[2 * t15], offs)
        surf = UnionSurface(object_pieces("table", float(k)))
        near, dist = surf.nearest(Pl)
        gap, _ = cg.core4d_arrays("table", robot, offs, human, pair, oq, float(k))
        cond = (h_sd[t15] >= -0.10) & (speed[t15, None, None] > 0.03)
        robo = H[2 * t15] >= -0.02
        res = {}
        for r in (0, 1):
            for hi, side in enumerate(("left", "right")):
                c = flags15[t15, r, hi]
                f = cond[:, r, hi] & ~robo[:, r, hi]
                hg = -h_sd[t15, r, hi]
                res[f"r{r}_{side}"] = dict(human_contact_frames=int(c.sum()), palm_to_surface_cm_median_in_contact=round(100 * float(np.median(dist[:, r, hi][c])), 2) if c.any() else None,
                                           palm_to_surface_cm_p90_in_contact=round(100 * float(np.percentile(dist[:, r, hi][c], 90)), 2) if c.any() else None,
                                           contact_position_error_mean_cm=gap[f"r{r}_{side}"]["mean_cm"], retention_condition_frames=int(cond[:, r, hi].sum()),
                                           failing_frames=int(f.sum()), failing_in_human_contact=int((f & (hg <= 0.015)).sum()), failing_in_approach=int((f & (hg > 0.015)).sum()),
                                           robot_hand_gap_cm_median_failing=round(-100 * float(np.median(H[2 * t15][:, r, hi][f])), 2) if f.any() else None)
        out[f"k{k}"] = dict(path=str(path), retention=round(float(robo[cond].mean()), 4), hands=res)
        print(k, json.dumps(out[f"k{k}"], ensure_ascii=False), flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "table_k08_gap_vs_retention.json").write_text(json.dumps(dict(rules=__doc__, results=out), indent=1, ensure_ascii=False) + "\n")


def a1(a):
    import subprocess
    pairs = dict(bin=(PREV / "bucket003_20231020_071_20260918/two_stage/core4d_pair_reference.npz", PREV / "bucket003_20231020_071_20260918/a1_wrist_only/core4d_pair_reference.npz"),
                 chair=(PREV / "chair021_20231020_074_baseline_20260908/shared_small_preview/core4d_pair_reference.npz",
                        PREV / "chair021_20231020_074_baseline_20260908/shared_small_a1_preview_tol6/core4d_pair_reference.npz"))
    cur = dict(bin=ROOT / "src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/core4d_bucket003_20231020_071_a1/core4d_pair_compact_fps30.npz",
               chair=ROOT / "src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/core4d_chair021_20231020_074_a1/core4d_pair_compact_fps30.npz")
    import hashlib
    out = {}
    for task, (pre, post) in pairs.items():
        zp, zq = np.load(pre), np.load(post)
        assert np.allclose(zp["object_qpos"], zq["object_qpos"]), "object differs between pre and post A1"
        changed = np.abs(zp["robot_qpos"] - zq["robot_qpos"]).max(axis=(0, 1))
        res = dict(pre=str(pre), post=str(post), post_equals_current=bool(np.array_equal(zq["robot_qpos"], np.load(cur[task])["robot_qpos"])),
                   pre_sha256=hashlib.sha256(pre.read_bytes()).hexdigest(), post_sha256=hashlib.sha256(post.read_bytes()).hexdigest(),
                   qpos_indices_changed_by_a1=[int(i) for i in np.flatnonzero(changed > 1e-12)])
        for name, p in (("pre_a1", pre), ("post_a1", post)):
            r = subprocess.run([sys.executable, "-B", str(ROOT / "system1/t14/diag/pen_where.py"), str(p), f"dump:{task}:1.0"], capture_output=True, text=True, check=True)
            res[name] = json.loads(r.stdout.strip().splitlines()[-1])
        out[task] = res
        print(task, json.dumps(res), flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "a1_penetration.json").write_text(json.dumps(dict(rules=__doc__, results=out), indent=1) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("fail-frames", "k08", "a1"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    dict(**{"fail-frames": fail_frames, "k08": k08, "a1": a1})[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
