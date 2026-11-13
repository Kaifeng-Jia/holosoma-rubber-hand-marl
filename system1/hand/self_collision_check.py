#!/usr/bin/env python3
"""T06 step 1 (CPU part): initial interpenetration between the Sharpa hands and the G1 body, collision meshes only.

The assembled URDF (system1/hand/assets/g1_29dof_sharpa.urdf) is posed with yourdfpy at (a) the zero body pose and
(b) frame 0 of the Bin reference for each agent (body joints with the Sharpa wrist re-solve of system1/hand/finger_ref.py),
fingers open, and every Sharpa collision mesh is tested against every G1 collision mesh with FCL (trimesh
CollisionManager): pairs in contact and the deepest penetration. The flange face sits on <side>_wrist_yaw_link by design;
that pair is reported separately. Same-hand pairs are not tested (they are filtered in the official USD).

    taskset -c 0-15 hssim_s1/bin/python -B system1/hand/self_collision_check.py --out results_system1/T06/step1/self_collision.json

--all-frames <finger_reference.npz>: every frame of a finger reference (both agents) instead of the three poses, and the
same check for the original D reference with the rubber-hand G1 (D Bin training enables robot self-collisions,
core4d_smalltable_experiment.py, and resets at uniformly random reference frames, wbt.py reset), written as per-frame CSV.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
URDF = ROOT / "system1/hand/assets/g1_29dof_sharpa.urdf"


def collision_meshes(u, urdf=None):
    import trimesh
    urdf = URDF if urdf is None else Path(urdf)
    out = {}
    for name, link in u.link_map.items():
        ms = []
        for c in link.collisions:
            g = c.geometry
            if g.mesh is None:
                continue
            fn = Path(g.mesh.filename)
            fn = fn if fn.is_absolute() else (urdf.parent / fn).resolve()
            m = trimesh.load(fn, force="mesh")
            if g.mesh.scale is not None:
                m.apply_scale(g.mesh.scale)
            if c.origin is not None:
                m.apply_transform(c.origin)
            ms.append(m)
        if ms:
            out[name] = trimesh.util.concatenate(ms)
    return out


def check(u, meshes, cfg: dict, hand_links: set[str]):  # noqa: D103
    import trimesh
    u.update_cfg(cfg)
    mh, mb = trimesh.collision.CollisionManager(), trimesh.collision.CollisionManager()
    for n, m in meshes.items():
        T = u.get_transform(n, u.base_link)
        (mh if n in hand_links else mb).add_object(n, m, transform=T)
    hit, _names, contacts = mh.in_collision_other(mb, return_names=True, return_data=True)
    pairs = {}
    for c in contacts:
        a, b = sorted(c.names)
        pairs[(a, b)] = max(pairs.get((a, b), 0.0), float(c.depth))
    return pairs


def all_frames(a):
    import csv
    import yourdfpy
    ref0 = np.load(ROOT / "src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/core4d_bucket003_20231020_071_a1/core4d_pair_runtime_fps50.npz", allow_pickle=True)
    fr = np.load(a.all_frames, allow_pickle=True)
    jn = [str(x) for x in fr["joint_names"]]
    fn = [str(x) for x in fr["finger_joint_names"]]
    rub_urdf = ROOT / "src/holosoma/holosoma/data/robots/g1/main_mesh_collision_rubberhand.urdf"
    rows, summ = [], {}
    for name, path in (("sharpa", URDF), ("rubber", rub_urdf)):
        u = yourdfpy.URDF.load(str(path), build_scene_graph=True, load_meshes=False, build_collision_scene_graph=False)
        meshes = collision_meshes(u, path)
        hand = {n for n in meshes if any(n.startswith(f"{s}_{p}") for s in ("left", "right")
                                         for p in ("hand_", "thumb_", "index_", "middle_", "ring_", "pinky_", "rubber_hand"))}
        names = u.actuated_joint_names
        for agent in range(2):
            for t in range(fr["agent_joint_pos"].shape[0]):
                cfg = {n: 0.0 for n in names}
                if name == "sharpa":
                    cfg.update({n: float(fr["agent_joint_pos"][t, agent, i]) for i, n in enumerate(jn)})
                    cfg.update({n: float(fr["agent_finger_joint_pos"][t, agent, i]) for i, n in enumerate(fn)})
                else:
                    cfg.update({n: float(ref0["agent_joint_pos"][t, agent, i]) for i, n in enumerate(jn)})
                pairs = check(u, meshes, cfg, hand)
                other = {k: d for k, d in pairs.items() if not (k[0].endswith("wrist_yaw_link") or k[1].endswith("wrist_yaw_link"))}
                deep = max(other.items(), key=lambda kv: kv[1]) if other else (("", ""), 0.0)
                rows.append(dict(model=name, agent=agent + 1, frame=t, pairs=len(other), max_depth_m=float(deep[1]), deepest_pair="|".join(deep[0])))
            r = [x for x in rows if x["model"] == name and x["agent"] == agent + 1]
            summ[f"{name}_agent{agent + 1}"] = dict(frames=len(r), frames_with_contact=sum(x["pairs"] > 0 for x in r),
                                                    frames_over_1cm=sum(x["max_depth_m"] > 0.01 for x in r),
                                                    max_depth_m=max(x["max_depth_m"] for x in r))
            print(name, agent + 1, json.dumps(summ[f"{name}_agent{agent + 1}"]), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.with_suffix(".csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    a.out.write_text(json.dumps(dict(finger_reference=str(a.all_frames), finger_reference_sha256=__import__("hashlib").sha256(Path(a.all_frames).read_bytes()).hexdigest(),
                                     sharpa_urdf=str(URDF.relative_to(ROOT)), rubber_urdf=str(rub_urdf.relative_to(ROOT)), summary=summ), indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--all-frames", type=Path, default=None, help="finger_reference.npz: check every frame (see module doc)")
    a = ap.parse_args()
    if a.all_frames is not None:
        return all_frames(a)
    import yourdfpy
    u = yourdfpy.URDF.load(str(URDF), build_scene_graph=True, load_meshes=False, build_collision_scene_graph=False)
    meshes = collision_meshes(u)
    hand_links = {n for n in meshes if any(n.startswith(f"{s}_{p}") for s in ("left", "right")
                                           for p in ("hand_", "thumb_", "index_", "middle_", "ring_", "pinky_"))}
    names = u.actuated_joint_names
    res = {}
    zero = {n: 0.0 for n in names}
    cases = {"zero_pose": zero}
    ref = np.load(ROOT / "logs_system1/t06/finger_ref/locked/finger_reference.npz", allow_pickle=True)
    jn = [str(x) for x in ref["joint_names"]]
    fn = [str(x) for x in ref["finger_joint_names"]]
    for agent in range(2):
        cfg = dict(zero)
        cfg.update({n: float(ref["agent_joint_pos"][0, agent, i]) for i, n in enumerate(jn)})
        cfg.update({n: float(ref["agent_finger_joint_pos"][0, agent, i]) for i, n in enumerate(fn)})
        cases[f"bin_reference_frame0_agent{agent + 1}"] = cfg
    for k, cfg in cases.items():
        pairs = check(u, meshes, cfg, hand_links)
        mount = {f"{a}|{b}": d for (a, b), d in pairs.items() if (a.endswith("wrist_yaw_link") or b.endswith("wrist_yaw_link"))}
        other = {f"{a}|{b}": d for (a, b), d in pairs.items() if not (a.endswith("wrist_yaw_link") or b.endswith("wrist_yaw_link"))}
        res[k] = dict(hand_vs_body_pairs_in_contact=len(pairs), mount_face_pairs=mount, other_pairs=other,
                      max_depth_other_m=max(other.values(), default=0.0), max_depth_mount_m=max(mount.values(), default=0.0))
        print(k, json.dumps(res[k])[:600])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(dict(urdf=str(URDF.relative_to(ROOT)), hand_links=sorted(hand_links), cases=res), indent=1))


if __name__ == "__main__":
    raise SystemExit(main())
