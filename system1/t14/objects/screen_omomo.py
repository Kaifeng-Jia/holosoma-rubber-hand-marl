#!/usr/bin/env python3
"""T14 step 7, part 3 item 1 (Kevin 2026-11-24): screen OMOMO sequences of largebox, smallbox, plasticbox, trashcan, suitcase,
smalltable into three classes of motion. Read only, CPU. All thresholds are in RULES and were fixed before any count was made.

Data: OMOMO original release (train + test pickles, 30 Hz; captured_objects meshes), quantities from omomo_motion.py. Hand
contact = the T14 stop-1 per-frame contact flags (contact_parts.py rule 4, the T12 / T14 rule: a hand (SMPL-X wrist + finger
vertices, flat hands) is in contact when its largest vertex signed distance to the captured mesh is >= -1.5 cm; computed every
3rd frame, 10 Hz; logs_system1/t14/contact_parts/per_frame_flags.npy); the contact face normal of a hand-frame = the outward
normal of the mesh surface sample nearest to its deepest vertex (obj_cache dense samples).

Per frame:
  lift(t)      = object bottom z (lowest convex-hull vertex) - rest height; rest height = the lowest bottom z of the sequence
                 after a 5-frame running median (the object rests on the floor at some time in every sequence);
  pelvis(t)    = SMPL-X pelvis joint, horizontal part; heading(t) = horizontal (left hip - right hip) x up;
  obj speed(t) = horizontal speed of the object centre after Gaussian smoothing (sigma 4 frames).
Episodes:
  lift episode = maximal run of lift > 0.03 m that contains >= 15 frames (0.5 s) of lift > 0.10 m; "picked up and put down" =
                 the run starts after the first frame and ends before the last frame (the object is on the floor before and
                 after it inside the sequence; added after the smoke run, where some smallbox / suitcase sequences start with
                 the object already in the hands).
  a (carry and walk forward): lift episode, put down, pelvis net horizontal displacement over the episode > 1.0 m, angle between
                 that displacement and the circular mean heading over the episode < 45 deg.
  c (lift in place): lift episode, put down, largest horizontal distance of the pelvis from its position at the episode start
                 < 0.30 m.
  b (push / pull / drag): maximal run of moving frames (obj speed > 0.03 m/s, the CoCarry moving threshold; gaps <= 15 frames
                 merged) in which lift < 0.03 m in every frame (a frame with lift >= 0.03 m splits the run); the object's largest
                 horizontal distance from its position at the run start > 0.5 m; both hands in contact in >= 50 % of the run's
                 10 Hz contact samples.
Reported per episode:
  face of a hand-frame: the contact point (mesh surface sample nearest to the hand's deepest vertex) relative to the object's
     box in the person's frame (forward = heading, left, up): the object's convex-hull vertices give the box (centre and half
     extents along these three axes at that frame); the axis with the largest |offset / half extent| and its sign give
     front (toward the person) / back, side-left / side-right, bottom / top. (Position, not the surface normal: on the thin
     walls of plasticbox / trashcan the nearest sample can be on the inner wall, whose normal points the other way; first
     written with the normal, changed after the 15-sequence smoke run.) A hand's face in an episode = majority over its
     contact samples.
  a: grip "两侧夹住" = left hand side-left and right hand side-right; "前面或底部托住" = both hands in {front, bottom};
     otherwise "其他" with the two faces;
     steps = swing phases of the two toes (SMPL-X joints 10, 11) inside the episode: runs of horizontal toe speed > 0.25 m/s
     (Gaussian sigma 2 frames) of >= 3 frames whose start-to-end horizontal toe displacement is > 0.10 m.
  b: same side = the two hands have the same face; also the
     direction of travel relative to the heading (push = object moves along the heading, pull = against it; angle at 90 deg
     = sideways), the faces, the largest object tilt.
Typical picks (fixed before looking at any picture): within a class, episodes whose both-hand contact share is >= 0.8 (a, c)
or >= 0.5 (b); distance to the class median in normalised features (a: travel, duration, peak lift; b: travel, duration;
c: peak lift, duration; each divided by the class inter-quartile range); best 3, at most one per object.

    sonic_s1/bin/python -B system1/t14/objects/screen_omomo.py --out results_system1/T14_retargeting/step7/screen
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import omomo_motion as om  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
FLAGS = ROOT / "logs_system1/t14/contact_parts/per_frame_flags.npy"
CACHE = ROOT / "logs_system1/t14/obj_cache"
OBJECTS = ("largebox", "smallbox", "plasticbox", "trashcan", "suitcase", "smalltable")
RULES = dict(fps=30, contact_rule="hand vertex signed distance >= -0.015 m (T12 / T14), 10 Hz samples (contact_parts.py)",
             rest_median_frames=5, lift_low_m=0.03, lift_high_m=0.10, lift_high_min_frames=15,
             a_pelvis_net_m=1.0, a_heading_deg=45.0, c_pelvis_max_m=0.30,
             b_lift_max_m=0.03, b_speed_mps=0.03, b_speed_sigma_frames=4, b_gap_merge_frames=15, b_travel_m=0.5, b_both_hands_share=0.5,
             face="contact point vs the object's box in the person's frame (largest |offset / half extent|)", step_speed_mps=0.25, step_sigma_frames=2, step_min_frames=3, step_disp_m=0.10,
             pick_both_hands_share=dict(a=0.8, b=0.5, c=0.8), pick_features=dict(a=["travel_m", "duration_s", "peak_lift_m"],
                                                                                  b=["travel_m", "duration_s"], c=["peak_lift_m", "duration_s"]))
R_ = RULES


def runs(mask):
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    e = np.flatnonzero(np.diff(m.astype(int)))
    return list(zip(e[0::2], e[1::2]))


def contact_arrays(entry, T):
    """per 30 Hz frame: sampled (bool), contact [T, 2], sample index [T, 2] (-1)."""
    sampled = np.zeros(T, bool)
    sampled[::3] = True
    con = np.zeros((T, 2), bool)
    samp = np.full((T, 2), -1, int)
    for hi, side in enumerate(("left", "right")):
        for rec in entry["people"].get(f"person1_{side}", []):
            t = rec[0]
            if t < T:
                con[t, hi] = True
                samp[t, hi] = rec[3]
    return sampled, con, samp


FACES = (("front", "back"), ("side-right", "side-left"), ("bottom", "top"))


def face_label(q, hull_world, fwd):
    """q: contact point (world); hull_world: the object's hull vertices (world) at that frame; fwd: heading."""
    axes = np.stack([fwd, np.cross(np.array([0.0, 0.0, 1.0]), fwd), np.array([0.0, 0.0, 1.0])])
    P = hull_world @ axes.T
    mid, half = (P.max(0) + P.min(0)) / 2, np.maximum((P.max(0) - P.min(0)) / 2, 1e-6)
    x = (q @ axes.T - mid) / half
    k = int(np.argmax(np.abs(x)))
    return FACES[k][int(x[k] > 0)]


def majority(labels):
    if not labels:
        return None
    k, c = np.unique(labels, return_counts=True)
    return str(k[np.argmax(c)])


def steps_in(toe, s, e):
    x = gaussian_filter1d(toe[..., :2], R_["step_sigma_frames"], axis=0, mode="nearest")
    v = np.zeros(x.shape[:2])
    v[1:] = np.linalg.norm(np.diff(x, axis=0), axis=-1) * R_["fps"]
    n = 0
    for k in range(2):
        for a, b in runs(v[s:e, k] > R_["step_speed_mps"]):
            if b - a >= R_["step_min_frames"] and np.linalg.norm(x[s + b - 1, k] - x[s + a, k]) > R_["step_disp_m"]:
                n += 1
    return n


def analyse(name, v, entry, points):
    T = len(v["trans"])
    J = om.smplx_joints(v)
    ob = om.object_track(v, name)
    bottom = median_filter(ob["bottom"], R_["rest_median_frames"], mode="nearest")
    lift = bottom - bottom.min()
    pel = J[:, 0, :2]
    hd = om.heading(J)
    cen = ob["centre"][:, :2]
    xs = gaussian_filter1d(cen, R_["b_speed_sigma_frames"], axis=0, mode="nearest")
    spd = np.zeros(T)
    spd[1:] = np.linalg.norm(np.diff(xs, axis=0), axis=1) * R_["fps"]
    sampled, con, samp = contact_arrays(entry, T)
    up_local = ob["R"][0].T @ np.array([0.0, 0.0, 1.0])  # tilt = angle of the object's frame-0 up direction from the vertical
    tilt = np.degrees(np.arccos(np.clip((ob["R"] @ up_local)[:, 2], -1, 1)))
    seq = dict(object=name, sequence=v["seq_name"], split=v["split"], frames=T, invalid_object_frames=ob["invalid_frames"],
               max_lift_m=round(float(lift.max()), 4), contact_samples_left=int(con[sampled, 0].sum()), contact_samples_right=int(con[sampled, 1].sum()),
               samples=int(sampled.sum()))
    eps = []

    def hand_faces(s, e):
        out = []
        for hi in range(2):
            lab = []
            for t in range(s, e):
                if sampled[t] and con[t, hi] and samp[t, hi] >= 0:
                    q = ob["s"][t] * (ob["R"][t] @ points[samp[t, hi]]) + ob["t"][t]
                    lab.append(face_label(q, ob["hull_world"][t], hd[t]))
            out.append(majority(lab))
        return out

    def both_share(s, e):
        m = sampled[s:e]
        return float((con[s:e][m].all(1)).mean()) if m.any() else 0.0

    # lift episodes (a, c)
    for s, e in runs(lift > R_["lift_low_m"]):
        hi = lift[s:e] > R_["lift_high_m"]
        if max((b - a for a, b in runs(hi)), default=0) < R_["lift_high_min_frames"]:
            continue
        put_down = bool(s > 0 and e < T)  # picked up and put down inside the sequence
        d = pel[e - 1] - pel[s]
        net = float(np.linalg.norm(d))
        mh = hd[s:e, :2].sum(0)
        mh = mh / np.linalg.norm(mh)
        ang = float(np.degrees(np.arccos(np.clip(d @ mh / max(net, 1e-9), -1, 1)))) if net > 1e-6 else None
        maxdev = float(np.linalg.norm(pel[s:e] - pel[s], axis=1).max())
        fl, fr = hand_faces(s, e)
        if fl == "side-left" and fr == "side-right":
            grip = "两侧夹住"
        elif fl in ("front", "bottom") and fr in ("front", "bottom"):
            grip = "前面或底部托住"
        else:
            grip = f"其他（左 {fl}，右 {fr}）"
        ep = dict(kind="lift", start=int(s), stop=int(e), duration_s=round((e - s) / R_["fps"], 3), peak_lift_m=round(float(lift[s:e].max()), 4),
                  put_down=bool(put_down), pelvis_net_m=round(net, 4), pelvis_max_from_start_m=round(maxdev, 4),
                  travel_vs_heading_deg=None if ang is None else round(ang, 1), both_hands_share=round(both_share(s, e), 3),
                  left_face=fl, right_face=fr, grip=grip, steps=steps_in(J[:, [10, 11]], s, e))
        ep["class_a"] = bool(put_down and net > R_["a_pelvis_net_m"] and ang is not None and ang < R_["a_heading_deg"])
        ep["class_c"] = bool(put_down and maxdev < R_["c_pelvis_max_m"])
        ep["travel_m"] = ep["pelvis_net_m"]
        eps.append(ep)
    # moving-on-the-floor episodes (b)
    mv = spd > R_["b_speed_mps"]
    for a, b in runs(~mv):  # merge short gaps
        if 0 < a and b < T and b - a <= R_["b_gap_merge_frames"]:
            mv[a:b] = True
    for s0, e0 in runs(mv):
        for s, e in runs(lift[s0:e0] < R_["b_lift_max_m"]):
            s, e = s0 + s, s0 + e
            if e - s < 2:
                continue
            travel = float(np.linalg.norm(cen[s:e] - cen[s], axis=1).max())
            share = both_share(s, e)
            fl, fr = hand_faces(s, e)
            same = None if fl is None or fr is None else bool(fl == fr)
            d = cen[e - 1] - cen[s]
            mh = hd[s:e, :2].sum(0)
            mh = mh / np.linalg.norm(mh)
            ang = float(np.degrees(np.arccos(np.clip(d @ mh / max(np.linalg.norm(d), 1e-9), -1, 1))))
            ep = dict(kind="floor", start=int(s), stop=int(e), duration_s=round((e - s) / R_["fps"], 3), travel_m=round(travel, 4),
                      max_lift_m=round(float(lift[s:e].max()), 4), both_hands_share=round(share, 3), left_face=fl, right_face=fr, same_side=same,
                      object_travel_vs_heading_deg=round(ang, 1), push_pull="推" if ang < 45 else ("拉" if ang > 135 else "侧向"),
                      max_tilt_deg=round(float(tilt[s:e].max()), 1), pelvis_net_m=round(float(np.linalg.norm(pel[e - 1] - pel[s])), 4))
            ep["class_b"] = bool(travel > R_["b_travel_m"] and share >= R_["b_both_hands_share"])
            eps.append(ep)
    for k in ("a", "b", "c"):
        seq[f"class_{k}"] = any(ep.get(f"class_{k}", False) for ep in eps)
    return seq, eps


def pick(eps, cls):
    cand = [e for e in eps if e.get(f"class_{cls}") and e["both_hands_share"] >= R_["pick_both_hands_share"][cls]]
    if not cand:
        return [], []
    feats = R_["pick_features"][cls]
    X = np.array([[e[f] for f in feats] for e in cand], float)
    med = np.median(X, 0)
    iqr = np.subtract(*np.percentile(X, [75, 25], 0))
    iqr[iqr <= 0] = 1.0
    dist = np.linalg.norm((X - med) / iqr, axis=1)
    order = np.argsort(dist, kind="stable")
    out, seen = [], set()
    for i in order:
        if cand[i]["object"] in seen:
            continue
        seen.add(cand[i]["object"])
        out.append(dict(cand[i], typical_distance=round(float(dist[i]), 4)))
        if len(out) == 3:
            break
    per_object = {}
    for i in order:  # most typical per object (layer-1 choice uses it for b and c)
        per_object.setdefault(cand[i]["object"], dict(cand[i], typical_distance=round(float(dist[i]), 4)))
    return out, list(per_object.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None, help="smoke only: first N sequences per object")
    a = ap.parse_args()
    t0 = time.time()
    a.out.mkdir(parents=True, exist_ok=True)
    seqs = om.load_sequences(OBJECTS)
    flags = {r["sequence"]: r for r in np.load(FLAGS, allow_pickle=True) if r["dataset"] == "OMOMO" and r["sequence"].split("_")[1] in OBJECTS}
    points = {n: np.load(CACHE / f"omomo_{n}.npz")["dense_p"].astype(np.float64) for n in OBJECTS}  # mesh units (obj_scale applies)
    rows, episodes = [], []
    count = {}
    for name in sorted(seqs, key=lambda s: (s.split("_")[1], seqs[s]["split"], s)):
        obj = name.split("_")[1]
        count[obj] = count.get(obj, 0) + 1
        if a.limit and count[obj] > a.limit:
            continue
        if name not in flags:
            raise SystemExit(f"no contact flags for {name}")
        seq, eps = analyse(obj, seqs[name], flags[name], points[obj])
        rows.append(seq)
        for e in eps:
            episodes.append(dict(object=obj, sequence=name, **e))
    keys = list(rows[0])
    with open(a.out / "sequences.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    ekeys = []
    for e in episodes:
        ekeys += [k for k in e if k not in ekeys]
    with open(a.out / "episodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ekeys)
        w.writeheader()
        w.writerows(episodes)
    summary = {}
    for obj in OBJECTS:
        r = [x for x in rows if x["object"] == obj]
        e = [x for x in episodes if x["object"] == obj]
        summary[obj] = dict(sequences=len(r), **{f"class_{k}_sequences": sum(x[f"class_{k}"] for x in r) for k in "abc"},
                            **{f"class_{k}_episodes": sum(bool(x.get(f"class_{k}")) for x in e) for k in "abc"},
                            a_grip={g: sum(1 for x in e if x.get("class_a") and x["grip"].startswith(g)) for g in ("两侧夹住", "前面或底部托住", "其他")},
                            b_same_side=dict(same=sum(1 for x in e if x.get("class_b") and x["same_side"] is True),
                                             different=sum(1 for x in e if x.get("class_b") and x["same_side"] is False)),
                            b_push_pull={p: sum(1 for x in e if x.get("class_b") and x["push_pull"] == p) for p in ("推", "拉", "侧向")})
    picks, per_object = {}, {}
    for k in "abc":
        picks[k], per_object[k] = pick(episodes, k)
    rec = dict(rules=RULES, doc=__doc__, summary=summary, picks=picks, most_typical_per_object=per_object, sequences_total=len(rows),
               episodes_total=len(episodes), inputs=dict(per_frame_flags=str(FLAGS), per_frame_flags_sha256=hashlib.sha256(FLAGS.read_bytes()).hexdigest(),
                                                         pickles={s: hashlib.sha256((om.OMOMO / f"{s}_diffusion_manip_seq_joints24.p").read_bytes()).hexdigest() for s in ("train", "test")},
                                                         meshes={n: hashlib.sha256((om.OMOMO / f"captured_objects/{n}_cleaned_simplified.obj").read_bytes()).hexdigest() for n in OBJECTS}),
               seconds=round(time.time() - t0, 1), command=" ".join(sys.argv))
    (a.out / "screen.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False, default=lambda x: x.item() if hasattr(x, "item") else str(x)) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    for k in "abc":
        print(k, [(p["sequence"], p["start"], p["stop"]) for p in picks[k]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
