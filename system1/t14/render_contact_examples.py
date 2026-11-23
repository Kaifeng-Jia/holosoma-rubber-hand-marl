#!/usr/bin/env python3
"""T14 stop 1: one example contact frame per CORE4D category and per OMOMO object, for the rule-2 list. Read only, CPU.

Example choice (fixed before rendering): sequences in CORE4D path order, OMOMO split order (test, then train) and name
order; the first sequence of that category / object with at least one contact hand-frame; among its
contact hand-frames ordered by (frame, person, hand), the middle one (index n // 2).

Picture: the dataset person(s) (SMPL-X mesh, CORE4D both persons, OMOMO the one person) and the object at that frame,
object light blue, the object's rule-5 thin-part ("rod") surface in salmon, chequered floor at the lowest foot vertex.
Two views of the same frame: close (0.55 m from the contact hand's closest vertex) and context (1.8 m), azimuth from
the object's centre towards the contact point plus 35 deg, elevation 25 deg. Contact sheets: jpg, each <= 1 MB.

    CUDA_VISIBLE_DEVICES= MUJOCO_GL=egl PYOPENGL_PLATFORM=egl __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json \
      sonic_s1/bin/python -B system1/t14/render_contact_examples.py --out results_system1/T14_retargeting/stop1/examples
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import contact_parts as cp  # noqa: E402

ROOT = cp.ROOT
FLAGS = ROOT / "logs_system1/t14/contact_parts/per_frame_flags.npy"
YUP_TO_ZUP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)
VW, VH = 520, 390


def choose(results, dataset):
    """First sequence (processing order) per category with contact; its middle contact hand-frame."""
    out = {}
    rows = [r for r in results if r["dataset"] == dataset]
    rows.sort(key=lambda r: (r.get("split") or "", r["sequence"]))
    for r in rows:
        cat = r["category"]
        if cat in out:
            continue
        hf = sorted((f[0], who) + tuple(f[1:]) for who, fr in r["people"].items() for f in fr)
        if hf:
            out[cat] = dict(sequence=r["sequence"], pick=hf[len(hf) // 2], n_contact=len(hf))
    return out


def faces_split(mesh, key):
    z = np.load(cp.CACHE / f"{key}.npz")
    _, j = cKDTree(z["coarse_p"]).query(mesh.triangles_center)
    rod = z["coarse_rod"][j]
    return mesh.submesh([np.flatnonzero(~rod)], append=True) if (~rod).any() else None, mesh.submesh([np.flatnonzero(rod)], append=True) if rod.any() else None


def render(meshes, focus, view_dir, dist_close, dist_far, floor_z):
    import mujoco
    tmp = Path(tempfile.mkdtemp(dir=ROOT / "logs_system1/t14"))
    assets, geoms = [], []
    for i, (m, rgba) in enumerate(meshes):
        if m is None or len(m.faces) == 0:
            continue
        p = tmp / f"m{i}.obj"
        m.export(p)
        assets.append(f'<mesh name="m{i}" file="{p}" inertia="shell"/>')
        geoms.append(f'<geom type="mesh" mesh="m{i}" rgba="{rgba}" contype="0" conaffinity="0"/>')
    xml = f"""<mujoco><visual><global offwidth="{VW}" offheight="{VH}"/><headlight ambient="0.35 0.35 0.35" diffuse="0.5 0.5 0.5"/><quality shadowsize="2048"/></visual>
    <asset><texture name="grid" type="2d" builtin="checker" rgb1="0.82 0.82 0.82" rgb2="0.62 0.62 0.62" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="8 8"/>{''.join(assets)}</asset>
    <worldbody><light pos="{focus[0]} {focus[1]} {floor_z + 4}" dir="0 0 -1" diffuse="0.6 0.6 0.6"/>
    <geom type="plane" size="5 5 0.01" pos="{focus[0]} {focus[1]} {floor_z}" material="grid" contype="0" conaffinity="0"/>{''.join(geoms)}</worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    imgs = []
    with mujoco.Renderer(model, VH, VW) as r:
        for d in (dist_close, dist_far):
            cam = mujoco.MjvCamera()
            cam.lookat[:] = focus
            cam.distance = d
            cam.azimuth = float(np.degrees(np.arctan2(view_dir[1], view_dir[0]))) + 180 + 35
            cam.elevation = -25
            r.update_scene(data, cam)
            imgs.append(r.render().copy())
    for f in tmp.iterdir():
        f.unlink()  # our own temporary meshes, written above in this call
    tmp.rmdir()
    return np.concatenate(imgs, axis=1)


def caption(img, lines):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    d.rectangle([0, 0, im.width, 22 * len(lines) + 6], fill=(255, 255, 255))
    for i, t in enumerate(lines):
        d.text((6, 4 + 22 * i), t, fill=(0, 0, 0), font=font)
    return np.asarray(im)


def core4d_frame(seq, cat, name, pick):
    t, who = pick[0], pick[1]
    base = cp.CORE4D / "human_object_motions" / seq
    To = np.load(base / "smooth_objposes.npy")[t]
    faces = np.load(cp.SMPLX_DIR / "smplx/SMPLX_NEUTRAL.npz")["f"]
    people, shown = [], []
    for pi in (1, 2):
        Vall = np.load(base / f"person{pi}_poses.npz", allow_pickle=True)["arr_0"].item()["vertices"]
        if t < len(Vall):  # the two persons' files can differ in length; a person without this frame is not drawn
            people.append(Vall[t].astype(np.float64) @ YUP_TO_ZUP.T)
            shown.append(pi)
    mesh = trimesh.load(cp.CORE4D / "object_models" / cat / f"{name}_m.obj", force="mesh", process=True)
    plain, rod = faces_split(mesh, f"core4d_{name}")
    T = np.eye(4)
    T[:3, :3] = YUP_TO_ZUP @ To[:3, :3]
    T[:3, 3] = YUP_TO_ZUP @ To[:3, 3]
    for m in (plain, rod):
        if m is not None:
            m.apply_transform(T)
    pi = shown.index(int(who[6]))
    side = who.split("_")[1]
    hv = cp.hand_vertex_ids()[side]
    return people, faces, (plain, rod), people[pi][hv], mesh.centroid @ T[:3, :3].T + T[:3, 3], shown


def omomo_frame(v, pick):
    import smplx
    import torch
    t, who = pick[0], pick[1]
    name = v["seq_name"].split("_")[1]
    m = smplx.create(str(cp.SMPLX_DIR), model_type="smplx", gender=str(v["gender"]), num_betas=16, use_pca=False, flat_hand_mean=True, batch_size=1)
    with torch.no_grad():
        V = m(betas=torch.tensor(v["betas"], dtype=torch.float32), global_orient=torch.tensor(v["root_orient"][t:t + 1], dtype=torch.float32),
              body_pose=torch.tensor(v["pose_body"][t:t + 1], dtype=torch.float32), transl=torch.tensor(v["trans"][t:t + 1], dtype=torch.float32)).vertices[0].numpy().astype(np.float64)
    parts = [("", "obj_")] if name not in cp.SPLIT_PARTS else [("_top", "obj_"), ("_bottom", "obj_bottom_")]
    meshes, cents = [], []
    for suf, pre in parts:
        mesh = trimesh.load(cp.OMOMO / f"captured_objects/{name}_cleaned_simplified{suf}.obj", force="mesh", process=True)
        key = f"omomo_{name}{suf}"
        plain, rod = faces_split(mesh, key)
        T = np.eye(4)
        T[:3, :3] = v[pre + "rot"][t] * v[pre + "scale"][t]
        T[:3, 3] = v[pre + "trans"][t, :, 0]
        for mm in (plain, rod):
            if mm is not None:
                mm.apply_transform(T)
        meshes += [plain, rod]
        cents.append(mesh.centroid @ T[:3, :3].T + T[:3, 3])
    faces = np.load(cp.SMPLX_DIR / "smplx/SMPLX_NEUTRAL.npz")["f"]
    side = who.split("_")[1]
    return [V], faces, tuple(meshes), V[cp.hand_vertex_ids()[side]], np.mean(cents, axis=0)


def tile(people, faces, objmeshes, hand, centre, lines):
    floor_z = min(float(P[:, 2].min()) for P in people)
    focus = hand.mean(0)
    view = focus - centre
    view[2] = 0
    if np.linalg.norm(view) < 1e-6:
        view = np.array([1.0, 0, 0])
    meshes = [(trimesh.Trimesh(P, faces, process=False), "0.80 0.70 0.62 1") for P in people]
    for i, m in enumerate(objmeshes):
        meshes.append((m, "0.62 0.80 0.95 1" if i % 2 == 0 else "0.98 0.55 0.45 1"))
    return caption(render(meshes, focus, view / np.linalg.norm(view), 0.55, 1.8, floor_z), lines)


def sheet(tiles, path, cols=2):
    rows = (len(tiles) + cols - 1) // cols
    h, w = tiles[0].shape[:2]
    S = np.full((rows * h, cols * w, 3), 255, np.uint8)
    for i, t in enumerate(tiles):
        S[(i // cols) * h:(i // cols + 1) * h, (i % cols) * w:(i % cols + 1) * w] = t[..., :3]
    im = Image.fromarray(S)
    q = 88
    while True:
        im.save(path, quality=q)
        if path.stat().st_size <= 1_000_000 or q <= 40:
            break
        q -= 8
    return dict(file=path.name, bytes=path.stat().st_size, jpeg_quality=q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    assert os.environ.get("MUJOCO_GL") == "egl"
    a.out.mkdir(parents=True, exist_ok=True)
    flags = list(np.load(FLAGS, allow_pickle=True))
    seqcsv = {(r["dataset"], r["sequence"]): r for r in __import__("csv").DictReader((a.out.parent / "rule2_per_sequence.csv").open())}
    for r in flags:
        r["category"] = seqcsv[(r["dataset"], r["sequence"])]["category"]
        r["object"] = seqcsv[(r["dataset"], r["sequence"])]["object"]
    record = []
    tiles = []
    for cat, c in sorted(choose(flags, "CORE4D").items()):
        name = next(r["object"] for r in flags if r["dataset"] == "CORE4D" and r["sequence"] == c["sequence"])
        people, faces, objm, hand, centre, shown = core4d_frame(c["sequence"], cat, name, c["pick"])
        t, who, rod, opp = c["pick"][:4]
        extra = "" if len(shown) == 2 else f"  (only person{shown[0]} has this frame)"
        lines = [f"CORE4D {cat} ({name})  {c['sequence']}  frame {t} (15 Hz)  {who}{extra}", f"rod contact: {bool(rod)}   opposing contact: {bool(opp)}   salmon = rule-5 thin part"]
        tiles.append(tile(people, faces, objm, hand, centre, lines))
        record.append(dict(dataset="CORE4D", category=cat, object=name, sequence=c["sequence"], frame=int(t), hand=who, rod=bool(rod), opposing=bool(opp), contact_hand_frames_in_sequence=c["n_contact"]))
        print(record[-1], flush=True)
    sheets = [sheet(tiles, a.out / "core4d_examples.jpg")]
    import joblib
    om = {}
    for split in ("train", "test"):
        for v in joblib.load(cp.OMOMO / f"{split}_diffusion_manip_seq_joints24.p").values():
            om[v["seq_name"]] = v
    tiles = []
    for cat, c in sorted(choose(flags, "OMOMO").items()):
        v = om[c["sequence"]]
        people, faces, objm, hand, centre = omomo_frame(v, c["pick"])
        t, who, rod, opp = c["pick"][:4]
        lines = [f"OMOMO {cat}  {c['sequence']}  frame {t} (30 Hz)  {who.replace('person1_', '')} hand", f"rod contact: {bool(rod)}   opposing contact: {bool(opp)}   salmon = rule-5 thin part"]
        tiles.append(tile(people, faces, objm, hand, centre, lines))
        record.append(dict(dataset="OMOMO", category=cat, object=cat, sequence=c["sequence"], frame=int(t), hand=who, rod=bool(rod), opposing=bool(opp), contact_hand_frames_in_sequence=c["n_contact"]))
        print(record[-1], flush=True)
    for k in range(0, len(tiles), 8):
        sheets.append(sheet(tiles[k:k + 8], a.out / f"omomo_examples_{k // 8 + 1}.jpg"))
    (a.out / "examples.json").write_text(json.dumps(dict(rule=__doc__, examples=record, sheets=sheets), indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
