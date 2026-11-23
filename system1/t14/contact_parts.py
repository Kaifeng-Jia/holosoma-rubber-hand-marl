#!/usr/bin/env python3
"""T14 stop 1, rule 2 (section 3): does the human hand touch the object with the palm on a surface, or do the fingers have
to go around a part thinner than the palm? Per CORE4D category / object and per OMOMO object. Read only, CPU.

Rules (fixed here before any number was computed; the verdict cut-offs in rule 8 are Claude's proposal, no external
source, for Kevin to confirm or change at the stop point):

1. Palm width W = 0.1044 m: the middle extent of the oriented bounding box of the robot's rubber hand collision mesh
   (main_mesh_collision_rubberhand.urdf, left_rubber_hand_link: 0.044 x 0.104 x 0.142 m). The robot palm is what has to
   lie on the surface.
2. Hand: SMPL-X vertices whose largest skinning weight is that hand's wrist or finger joints (wrist 20 / 21 and the 15
   finger joints of each hand; 780 / 778 vertices), as T12-diag source_trace.
   * CORE4D: the stored per-frame SMPL-X vertices of person1 and person2, every frame (15 Hz).
   * OMOMO: SMPL-X vertices from the stored parameters (the sequence's gender model, 16 betas, root_orient, pose_body,
     trans; flat hands: OMOMO has no finger capture), every 3rd frame (10 Hz).
3. Object: the dataset mesh placed by the stored pose. CORE4D object_models/<category>/<name>_m.obj (name lower-cased),
   smooth_objposes.npy. OMOMO captured_objects/<name>_cleaned_simplified.obj scaled by obj_scale, obj_rot, obj_trans;
   mop and vacuum: the _top part with obj_*, the _bottom part with obj_bottom_*, union of the two.
   Meshes with inconsistent winding get trimesh.repair.fix_normals in memory (the files are not touched); listed.
3b. (Added after the first full run stopped on it.) OMOMO frames in which a part's stored pose is invalid (obj_scale <= 0
   or a non-finite scale / rotation / translation; 39 sequences have such frames, mostly the vacuum's bottom part): that
   part is left out of that frame; a frame with no valid part is not used. Counted per sequence.
4. Signed distance of a point (positive inside, as T12): to the nearest of dense surface samples of the mesh (area-
   uniform, seed 0, about one per (3 mm)^2, at most 400 000), signed by that sample's outward face normal.
   A hand is in contact in a frame when its largest vertex signed distance is >= -1.5 cm (T12 rule). Its contact
   vertices are the vertices with signed distance >= -1.5 cm.
5. Thin-part ("rod") points, object geometry only: on coarse surface samples (about one per cm^2, at most 50 000), take the
   outward normals of all coarse samples within W of the point (a cross-section narrower than W is then always inside
   the neighbourhood, from any point on it; first written as W / 2, changed after the 12-sequence smoke run because from an
   off-centre point of stick006, 3.7 x 7.2 cm, the far side was out of a W / 2 ball); axis = the eigenvector of the smallest eigenvalue of
   the normals' scatter matrix; project the normals onto the plane normal to the axis; the point is a rod point when the
   largest angular gap between the projected normals is < 105 deg (the surface turns all the way round the axis) and the
   extent of the neighbourhood points across the axis is < W: the part's cross-section is smaller than the palm (stick,
   pole, leg, handle, bail, rim with a lip, narrow bar). (The extent condition was added after the 30-sequence smoke run:
   without it the turn alone also fired where a table top's edge has an apron below it, cross-section 11-17 cm.) A face wider than the palm has one normal direction; the edge of a plate (table top, board, seat) has
   at most half a turn (top, side, bottom: gap 180 deg); a box corner about 120 deg. Dense samples take the flag of the
   nearest coarse sample. Rod size = the extent of the neighbourhood points across the axis (about the part diameter).
6. A contact hand-frame is a rod contact when more than half of its contact vertices have their nearest surface sample on
   a rod point.
7. Secondary, behavioural: opposing contact = among (up to 64 evenly chosen) contact vertices of the hand, two whose nearest
   surface normals point against each other (dot < -0.5) and whose surface points are less than W apart along that
   normal: the hand is on two opposite faces of something thinner than the palm (fingers and palm around it; for the
   flat OMOMO hands, the hand cuts through it).
8. Verdict per CORE4D category / OMOMO object (proposal): rod-contact share of all contact hand-frames < 20 %: applicable;
   > 50 %: not applicable; otherwise undecided. Per-object rows are reported under each category.

    sonic_s1/bin/python -B system1/t14/contact_parts.py --out results_system1/T14_retargeting/stop1 --workers 24
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
CORE4D = Path("/home/kevin/datasets/CORE4D-V1-full/CORE4D_Real")
OMOMO = Path("/home/kevin/holosoma/data")
SMPLX_DIR = Path("/home/kevin/CS6758Project/data/smplx_models")
CACHE = ROOT / "logs_system1/t14/obj_cache"
W = 0.1044
RADIUS = W  # rule 5 neighbourhood
CONTACT = -0.015
GAP_DEG = 105.0
DENSE_SPACING, DENSE_MAX = 0.003, 400_000
COARSE_SPACING, COARSE_MAX = 0.01, 50_000
OMOMO_STRIDE = 3
SPLIT_PARTS = ("mop", "vacuum")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def hand_vertex_ids():
    z = np.load(SMPLX_DIR / "smplx/SMPLX_NEUTRAL.npz", allow_pickle=False)
    lab = z["weights"].argmax(1)
    return dict(left=np.flatnonzero(np.isin(lab, [20] + list(range(25, 40)))), right=np.flatnonzero(np.isin(lab, [21] + list(range(40, 55)))))


# ------------------------------------------------------------------------------------------------- object geometry
def rod_flags(P, N, radius):
    """Rule 5 on points P with outward normals N (same units as radius). Returns flag, max gap (deg), cross size."""
    tree = cKDTree(P)
    nb = tree.query_ball_point(P, radius)
    flag = np.zeros(len(P), bool)
    gap = np.full(len(P), 360.0)
    size = np.full(len(P), np.nan)
    for i, idx in enumerate(nb):
        n = N[idx]
        if len(idx) < 4:
            continue
        S = n.T @ n
        w, v = np.linalg.eigh(S)
        ax = v[:, 0]
        q = n - np.outer(n @ ax, ax)
        ok = np.linalg.norm(q, axis=1) > 0.2  # normals almost along the axis say nothing about the turn
        if ok.sum() < 3:
            continue
        e1 = v[:, 2]
        e2 = np.cross(ax, e1)
        ang = np.sort(np.degrees(np.arctan2(q[ok] @ e2, q[ok] @ e1)) % 360)
        g = np.max(np.diff(np.concatenate([ang, ang[:1] + 360])))
        gap[i] = g
        x = P[idx] - P[i]
        x = x - np.outer(x @ ax, ax)
        x = x if len(x) <= 120 else x[np.linspace(0, len(x) - 1, 120).astype(int)]
        size[i] = float(np.max(np.linalg.norm(x[:, None] - x[None], axis=-1)))
        flag[i] = g < GAP_DEG and size[i] < radius
    return flag, gap, size


def build_object(key, mesh_path, scale):
    """Samples in the mesh's own units; scale converts mesh units to metres (1 for CORE4D, median obj_scale for OMOMO)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / f"{key}.npz"
    if out.exists():
        return out
    m = trimesh.load(mesh_path, force="mesh", process=True)
    fixed = False
    if not m.is_winding_consistent:
        trimesh.repair.fix_normals(m)
        fixed = True
    area_m2 = m.area * scale ** 2
    nd = int(min(DENSE_MAX, max(20_000, area_m2 / DENSE_SPACING ** 2)))
    nc = int(min(COARSE_MAX, max(5_000, area_m2 / COARSE_SPACING ** 2)))
    pd, fd = trimesh.sample.sample_surface(m, nd, seed=0)
    pc, fc = trimesh.sample.sample_surface(m, nc, seed=1)
    nrm_d, nrm_c = m.face_normals[fd], m.face_normals[fc]
    flag, gap, size = rod_flags(pc, nrm_c, RADIUS / scale)
    _, near = cKDTree(pc).query(pd)
    np.savez(out, dense_p=pd.astype(np.float32), dense_n=nrm_d.astype(np.float32), dense_rod=flag[near], dense_rod_size_m=(size[near] * scale).astype(np.float32),
             coarse_p=pc.astype(np.float32), coarse_n=nrm_c.astype(np.float32), coarse_rod=flag, coarse_gap_deg=gap.astype(np.float32),
             coarse_rod_size_m=(size * scale).astype(np.float32), scale=scale, fixed_normals=fixed, mesh_sha256=sha(mesh_path), mesh=str(mesh_path),
             watertight=m.is_watertight, area_m2=area_m2, extents_m=m.extents * scale)
    return out


_OBJ = {}
OBJ_CACHE_MAX = 3  # objects kept per worker (memory; jobs are grouped by object)


def peak_rss_gb():
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return round(int(line.split()[1]) / 1024 ** 2, 3)
    return None


def obj(key):
    if key not in _OBJ:
        while len(_OBJ) >= OBJ_CACHE_MAX:
            _OBJ.pop(next(iter(_OBJ)))
        z = np.load(CACHE / f"{key}.npz")
        _OBJ[key] = dict(tree=cKDTree(z["dense_p"]), p=z["dense_p"].astype(np.float64), n=z["dense_n"].astype(np.float64), rod=z["dense_rod"],
                         size=z["dense_rod_size_m"], scale=float(z["scale"]), c=z["dense_p"].mean(0), r=float(np.linalg.norm(z["dense_p"] - z["dense_p"].mean(0), axis=1).max()))
    return _OBJ[key]


# ------------------------------------------------------------------------------------------------- per hand-frame
def hand_frame(local_parts, keys, rng_ids):
    """local_parts: list of (points in that part's mesh units [n,3], key). Returns dict or None (no contact)."""
    best = None
    for pl, key in local_parts:
        o = obj(key)
        s = o["scale"]
        if np.linalg.norm(pl.mean(0) - o["c"]) * s > o["r"] * s + 0.25:
            continue
        d, j = o["tree"].query(pl)
        sgn = -np.sign(np.einsum("ij,ij->i", pl - o["p"][j], o["n"][j]))
        sd = sgn * d * s
        if best is None or sd.max() > best[0].max():
            best = (sd, j, key)
    if best is None or best[0].max() < CONTACT:
        return None
    sd, j, key = best
    o = obj(key)
    c = sd >= CONTACT
    jc = j[c]
    rod_share = float(o["rod"][jc].mean())
    sel = jc if len(jc) <= 64 else jc[np.linspace(0, len(jc) - 1, 64).astype(int)]
    n = o["n"][sel]
    p = o["p"][sel] * o["scale"]
    dots = n @ n.T
    sep = np.abs(np.einsum("ijk,ik->ij", p[None] - p[:, None], n))
    opposing = bool(np.any((dots < -0.5) & (sep < W)))
    k = int(np.argmax(sd))
    return dict(rod=rod_share > 0.5, rod_share=rod_share, opposing=opposing, depth=float(sd.max()), part_key=key,
                rod_size=float(np.nanmedian(o["size"][jc][o["rod"][jc]])) if o["rod"][jc].any() else np.nan, closest_sample=int(j[k]))


def summarize(records):
    n = len(records)
    if n == 0:
        return dict(contact_hand_frames=0)
    rod = np.array([r["rod"] for r in records])
    opp = np.array([r["opposing"] for r in records])
    sz = np.array([r["rod_size"] for r in records if r["rod"]])
    return dict(contact_hand_frames=n, rod_contacts=int(rod.sum()), opposing_contacts=int(opp.sum()), rod_and_opposing=int((rod & opp).sum()),
                rod_size_sum=float(np.nansum(sz)), rod_size_n=int(np.isfinite(sz).sum()))


# ------------------------------------------------------------------------------------------------- CORE4D
def core4d_job(args):
    seq, key = args
    hv = hand_vertex_ids()
    To = np.load(CORE4D / "human_object_motions" / seq / "smooth_objposes.npy")
    out = dict(dataset="CORE4D", sequence=seq, part=key, frames=0, mismatch=None, people={})
    for pi in (1, 2):
        V = np.load(CORE4D / "human_object_motions" / seq / f"person{pi}_poses.npz", allow_pickle=True)["arr_0"].item()["vertices"]
        T = min(len(V), len(To))
        if len(V) != len(To):
            out["mismatch"] = [len(To), len(V)]
        out["frames"] = T
        for side in ("left", "right"):
            recs = []
            for t in range(T):
                R, x = To[t, :3, :3], To[t, :3, 3]
                pl = (V[t, hv[side]].astype(np.float64) - x) @ R
                r = hand_frame([(pl, key)], None, None)
                if r is not None:
                    r["t"] = t
                    recs.append(r)
            out["people"][f"person{pi}_{side}"] = dict(summary=summarize(recs), frames=[(r["t"], r["rod"], r["opposing"], r["closest_sample"]) for r in recs])
        del V
    out["worker_peak_rss_gb"] = peak_rss_gb()
    return out


def core4d_sequences():
    seqs = []
    for d in sorted(glob.glob(str(CORE4D / "human_object_motions/*/*"))):
        m = json.loads(Path(d, "object_metadata.json").read_text())
        cat = m["obj_model_path"].split("/")[-3]
        name = m["obj_name"].lower()
        seqs.append((str(Path(d).relative_to(CORE4D / "human_object_motions")), cat, name))
    return seqs


# ------------------------------------------------------------------------------------------------- OMOMO
_BM = {}
SMPLX_BATCH = 64  # one model per gender, fixed batch (memory); the last chunk is padded


def smplx_vertices(v, idx):
    import smplx
    import torch
    torch.set_num_threads(1)
    g = str(v["gender"])
    if g not in _BM:
        _BM[g] = smplx.create(str(SMPLX_DIR), model_type="smplx", gender=g, num_betas=16, use_pca=False, flat_hand_mean=True, batch_size=SMPLX_BATCH)
    m = _BM[g]
    out = []
    for a in range(0, len(idx), SMPLX_BATCH):
        k = idx[a:a + SMPLX_BATCH]
        pad = np.concatenate([k, np.repeat(k[-1:], SMPLX_BATCH - len(k))])
        with torch.no_grad():
            V = m(betas=torch.tensor(np.repeat(v["betas"], SMPLX_BATCH, 0), dtype=torch.float32), global_orient=torch.tensor(v["root_orient"][pad], dtype=torch.float32),
                  body_pose=torch.tensor(v["pose_body"][pad], dtype=torch.float32), transl=torch.tensor(v["trans"][pad], dtype=torch.float32)).vertices.numpy()
        out.append(V[: len(k)].astype(np.float64))
    return np.concatenate(out)


def omomo_job(v):
    import torch
    hv = hand_vertex_ids()
    name = v["seq_name"].split("_")[1]
    idx = np.arange(0, len(v["trans"]), OMOMO_STRIDE)
    T = len(idx)
    V = smplx_vertices(v, idx)
    parts = [(f"omomo_{name}_top" if name in SPLIT_PARTS else f"omomo_{name}", v["obj_scale"][idx], v["obj_rot"][idx], v["obj_trans"][idx, :, 0])]
    if name in SPLIT_PARTS:
        parts.append((f"omomo_{name}_bottom", v["obj_bottom_scale"][idx], v["obj_bottom_rot"][idx], v["obj_bottom_trans"][idx, :, 0]))
    valid = [np.isfinite(s) & (s > 0) & np.isfinite(R).all(axis=(1, 2)) & np.isfinite(x).all(axis=1) for _, s, R, x in parts]
    any_valid = np.any(valid, axis=0)
    out = dict(dataset="OMOMO", sequence=v["seq_name"], split=v["split"], frames=int(any_valid.sum()), people={},
               frames_invalid_object_pose=int((~any_valid).sum()), part_frames_invalid=[int((~ok).sum()) for ok in valid])
    for side in ("left", "right"):
        recs = []
        for k in range(T):
            p = V[k, hv[side]]
            loc = []
            for (key, s, R, x), ok in zip(parts, valid):
                if ok[k]:  # rule 3b: a part with an invalid pose in this frame is left out
                    loc.append((((p - x[k]) @ R[k]) / s[k], key))  # world = s R v + t  ->  mesh units
            if not loc:
                continue
            r = hand_frame(loc, None, None)
            if r is not None:
                r["t"] = int(idx[k])
                recs.append(r)
        out["people"][f"person1_{side}"] = dict(summary=summarize(recs), frames=[(r["t"], r["rod"], r["opposing"], r["closest_sample"], r["part_key"]) for r in recs])
    out["worker_peak_rss_gb"] = peak_rss_gb()
    return out


def omomo_sequences():
    import joblib
    seqs = []
    for split in ("train", "test"):
        d = joblib.load(OMOMO / f"{split}_diffusion_manip_seq_joints24.p")
        for v in d.values():
            keep = {k: v[k] for k in v if k not in ("rest_offsets", "trans2joint", "obj_com_pos")}
            keep["split"] = split
            seqs.append(keep)
        del d
    seqs.sort(key=lambda v: (v["seq_name"].split("_")[1], v["split"], v["seq_name"]))  # grouped by object (worker object cache)
    return seqs


# ------------------------------------------------------------------------------------------------- main
def verdict(share):
    return "applicable" if share < 0.20 else ("not_applicable" if share > 0.50 else "undecided")


def add(acc, s):
    for k, v in s.items():
        acc[k] = acc.get(k, 0) + v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--only", choices=("core4d", "omomo"), default=None)
    ap.add_argument("--limit", type=int, default=None, help="smoke: first N sequences per dataset")
    ap.add_argument("--chunksize", type=int, default=2)
    a = ap.parse_args()
    t0 = time.time()
    pool = Pool(a.workers)  # forked before the OMOMO pickles are loaded: jobs are sent to the workers, not inherited
    a.out.mkdir(parents=True, exist_ok=True)
    logs = ROOT / "logs_system1/t14/contact_parts"
    logs.mkdir(parents=True, exist_ok=True)
    results = []
    objects = {}
    missing_mesh = []
    if a.only in (None, "core4d"):
        seqs = core4d_sequences()[: a.limit]
        missing_mesh = [dict(sequence=s, category=c, object=n) for s, c, n in seqs if not (CORE4D / "object_models" / c / f"{n}_m.obj").exists()]
        seqs = [x for x in seqs if (CORE4D / "object_models" / x[1] / f"{x[2]}_m.obj").exists()]
        for c, n in sorted({(c, n) for _, c, n in seqs}):
            objects[f"core4d_{n}"] = dict(dataset="CORE4D", category=c, object=n, mesh=CORE4D / "object_models" / c / f"{n}_m.obj", scale=1.0)
        jobs_c = sorted([(s, f"core4d_{n}") for s, c, n in seqs], key=lambda j: (j[1], j[0]))  # grouped by object (worker object cache)
        print("CORE4D sequences without a local mesh (not evaluated):", len(missing_mesh), sorted({m["object"] for m in missing_mesh}), flush=True)
    if a.only in (None, "omomo"):
        oseqs = omomo_sequences()
        if a.limit:
            oseqs = [v for i, v in enumerate(oseqs) if i % max(1, len(oseqs) // a.limit) == 0][: a.limit]
        scales = {}
        for v in oseqs:
            name = v["seq_name"].split("_")[1]
            scales.setdefault(name, []).append(float(np.median(v["obj_scale"])))
            if name in SPLIT_PARTS:
                scales.setdefault(name + "_bottom", []).append(float(np.median(v["obj_bottom_scale"])))
        for name in sorted({v["seq_name"].split("_")[1] for v in oseqs}):
            for part in (("_top", "_bottom") if name in SPLIT_PARTS else ("",)):
                sc = float(np.median(scales[name + ("_bottom" if part == "_bottom" else "")]))
                objects[f"omomo_{name}{part}"] = dict(dataset="OMOMO", category=name, object=name + part, mesh=OMOMO / f"captured_objects/{name}_cleaned_simplified{part}.obj", scale=sc)
    print(f"building {len(objects)} object caches", flush=True)
    pool.starmap(build_object, [(k, o["mesh"], o["scale"]) for k, o in objects.items()], chunksize=1)
    for k, o in objects.items():
        z = np.load(CACHE / f"{k}.npz")
        o.update(mesh=str(o["mesh"]), mesh_sha256=str(z["mesh_sha256"]), watertight=bool(z["watertight"]), fixed_normals=bool(z["fixed_normals"]),
                 extents_m=[round(float(x), 4) for x in z["extents_m"]], area_m2=round(float(z["area_m2"]), 4), rod_area_share=round(float(z["coarse_rod"].mean()), 4),
                 dense_samples=int(len(z["dense_p"])), coarse_samples=int(len(z["coarse_p"])))
    print(f"objects done {time.time() - t0:.0f}s", flush=True)
    if a.only in (None, "core4d"):
        for i, r in enumerate(pool.imap(core4d_job, jobs_c, chunksize=a.chunksize)):
            results.append(r)
            if i % 100 == 0:
                print("core4d", i, len(jobs_c), f"{time.time() - t0:.0f}s", flush=True)
    if a.only in (None, "omomo"):
        for i, r in enumerate(pool.imap(omomo_job, oseqs, chunksize=a.chunksize)):
            results.append(r)
            if i % 500 == 0:
                print("omomo", i, len(oseqs), f"{time.time() - t0:.0f}s", flush=True)
    pool.close()
    pool.join()
    # per sequence rows and the per-frame flags (logs only)
    np.save(logs / "per_frame_flags.npy", np.array([dict(dataset=r["dataset"], sequence=r["sequence"], split=r.get("split"), people={k: v["frames"] for k, v in r["people"].items()}) for r in results], dtype=object), allow_pickle=True)
    mem = dict(parent_peak_rss_gb=peak_rss_gb(), worker_peak_rss_gb_max=max(r.get("worker_peak_rss_gb") or 0 for r in results) if results else None, workers=a.workers)
    print("memory", mem, flush=True)
    seq_rows = []
    for r in results:
        key = r.get("part") or (f"omomo_{r['sequence'].split('_')[1]}")
        cat = objects[key if key in objects else key + "_top"]["category"]
        objname = objects[key if key in objects else key + "_top"]["object"].replace("_top", "")
        row = dict(dataset=r["dataset"], category=cat, object=objname, sequence=r["sequence"], frames_used=r["frames"], frame_count_mismatch=r.get("mismatch"),
                   frames_invalid_object_pose=r.get("frames_invalid_object_pose", 0))
        tot = {}
        for k, v in r["people"].items():
            add(tot, v["summary"])
        row.update(tot)
        seq_rows.append(row)
    fields = ["dataset", "category", "object", "sequence", "frames_used", "frame_count_mismatch", "frames_invalid_object_pose", "contact_hand_frames", "rod_contacts", "opposing_contacts", "rod_and_opposing", "rod_size_sum", "rod_size_n"]
    with (a.out / "rule2_per_sequence.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fields, extrasaction="ignore")
        w.writeheader()
        for row in seq_rows:
            w.writerow({k: row.get(k, 0) for k in fields})
    # summaries: per category and per object
    summ = []
    for level in ("category", "object"):
        groups = {}
        for row in seq_rows:
            groups.setdefault((row["dataset"], row["category"], row[level] if level == "object" else ""), []).append(row)
        for (ds, cat, ob), rows in sorted(groups.items()):
            tot = {}
            for row in rows:
                add(tot, {k: row.get(k, 0) for k in fields[6:]})
            n = tot.get("contact_hand_frames", 0)
            sh = tot.get("rod_contacts", 0) / n if n else float("nan")
            summ.append(dict(level=level, dataset=ds, category=cat, object=ob, sequences=len(rows), sequences_with_contact=sum(1 for r in rows if r.get("contact_hand_frames", 0) > 0),
                             frames=sum(r["frames_used"] for r in rows), frames_invalid_object_pose=tot.get("frames_invalid_object_pose", 0), contact_hand_frames=n, rod_contact_share=round(sh, 4),
                             opposing_contact_share=round(tot.get("opposing_contacts", 0) / n, 4) if n else "", rod_and_opposing_share=round(tot.get("rod_and_opposing", 0) / n, 4) if n else "",
                             mean_rod_size_cm=round(100 * tot["rod_size_sum"] / tot["rod_size_n"], 1) if tot.get("rod_size_n") else "",
                             proposed_verdict=verdict(sh) if (level == "category" and n) else ""))
    sf = ["level", "dataset", "category", "object", "sequences", "sequences_with_contact", "frames", "frames_invalid_object_pose", "contact_hand_frames", "rod_contact_share", "opposing_contact_share", "rod_and_opposing_share", "mean_rod_size_cm", "proposed_verdict"]
    with (a.out / "rule2_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, sf)
        w.writeheader()
        for row in summ:
            w.writerow(row)
    prov = dict(script="system1/t14/contact_parts.py", script_sha256=sha(__file__), rules=__doc__, palm_width_m=W, contact_threshold_m=CONTACT, rod_gap_deg=GAP_DEG,
                omomo_stride=OMOMO_STRIDE, core4d_sequences_without_mesh=missing_mesh, objects=objects, smplx_models={g: sha(SMPLX_DIR / f"smplx/SMPLX_{g}.npz") for g in ("MALE", "FEMALE", "NEUTRAL")},
                omomo_pickles={s: sha(OMOMO / f"{s}_diffusion_manip_seq_joints24.p") for s in ("train", "test")} if a.only in (None, "omomo") else None,
                sequences=len(results), seconds=round(time.time() - t0, 1), limit=a.limit, memory=mem)
    (a.out / "rule2_provenance.json").write_text(json.dumps(prov, indent=1, default=str) + "\n")
    print(f"done {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
