#!/usr/bin/env python3
"""T13 v4 section 2: per-frame contact labels for the contact items (managers/reward/terms/contact_v1.py).

Rules (T13 v4, Kevin 2026-11-23):
  * should_contact: per robot and hand, from the original human data frame by frame: any SMPL-X hand vertex within 1.5 cm
    of the object surface (signed distance >= -1.5 cm, the T12 / T14 stop-1 rule). Runtime frame f (50 Hz, t = f / 50) <-> the
    nearest source frame (CORE4D 15 Hz, OMOMO 30 Hz). CORE4D: robot r <-> person r + 1. Push (OMOMO single person): both
    robots inherit the source labels (role isometry = translation, no left / right swap).
  * contact_point: the reference robot's palm point (the rubber-hand palm point of T14 on the simulator hand hull, simulator
    URDF FK of the runtime reference) projected to the simulator collision surface of the object, in the object frame.
  * human_far: per robot and non-hand link, the corresponding human body part (LINK_SEGMENTS: SMPL-X skinning segments,
    from the retargeting keypoint correspondence) is farther than 0.1 m from the captured object surface.
Read only; CPU.

    sonic_s1/bin/python -B system1/t13v4/contact_labels.py --task table --runtime <core4d_pair_runtime_fps50.npz> --out <npz>
    sonic_s1/bin/python -B system1/t13v4/contact_labels.py --task push --runtime <single fps50 npz> --out <npz>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in ("system1/t14", "system1/t12v2", "src/holosoma_retargeting"):
    sys.path.insert(0, str(ROOT / p))

CONTACT_M, FAR_M = -0.015, 0.10
PUSH_SPACING_M = 0.8780472  # PairedA1Reference lateral spacing of the paper D Push (codebase_map)
# robot link -> human body part: system1/t14/body_parts.py (shared with the T14 Push leg check; retargeting keypoint
# correspondence, SMPL-X skinning segments; rubber hands excluded)
from body_parts import LINK_TO_SMPLX, smplx_part_vertices  # noqa: E402
LINK_SEGMENTS = LINK_TO_SMPLX
LINKS = list(LINK_SEGMENTS)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def segment_vertex_ids():
    parts = smplx_part_vertices()
    return {link: parts[(name, joints)] if (name, joints) in parts else parts[name] for link, (name, joints) in LINK_SEGMENTS.items()}


def part_max_sd(obj, V_local, scale, seg):
    """[L] largest signed distance (positive = inside) of each segment's vertices to the object samples."""
    out = np.empty(len(LINKS))
    for i, link in enumerate(LINKS):
        pl = V_local[seg[link]]
        d, j = obj["tree"].query(pl)
        out[i] = (-np.sign(np.einsum("ij,ij->i", pl - obj["p"][j], obj["n"][j])) * d).max() * scale
    return out


# ------------------------------------------------------------------------------------------------- human side
def human_core4d(task):
    """per source frame (15 Hz): hand max signed distance [T, 2 persons, 2 hands], part max signed distance [T, 2, L]."""
    import contact_parts as cp
    import table_shared as tsh
    cfg = tsh.TASKS[task]
    seq = tsh.CORE4D / "CORE4D_Real/human_object_motions" / cfg["raw"]
    To = np.load(seq / "smooth_objposes.npy")
    obj = cp.obj(cfg["obj_key"])
    hv, seg = cp.hand_vertex_ids(), segment_vertex_ids()
    T = len(To)
    hand = np.full((T, 2, 2), -np.inf)
    part = np.full((T, 2, len(LINKS)), -np.inf)
    for pi in (1, 2):
        V = np.load(seq / f"person{pi}_poses.npz", allow_pickle=True)["arr_0"].item()["vertices"].astype(np.float64)
        for t in range(min(len(V), T)):
            R, x = To[t, :3, :3], To[t, :3, 3]
            Vl = (V[t] - x) @ R
            for hi, side in enumerate(("left", "right")):
                pl = Vl[hv[side]]
                d, j = obj["tree"].query(pl)
                hand[t, pi - 1, hi] = (-np.sign(np.einsum("ij,ij->i", pl - obj["p"][j], obj["n"][j])) * d).max()
            part[t, pi - 1] = part_max_sd(obj, Vl, 1.0, seg)
    return hand, part, 15, dict(sequence=str(seq), object_key=cfg["obj_key"])


def human_omomo(seq):
    """per source frame (30 Hz): [T, 1 person, 2 hands], [T, 1, L]."""
    import joblib
    import smplx
    import torch
    import contact_parts as cp
    for split in ("train", "test"):
        d = joblib.load(cp.OMOMO / f"{split}_diffusion_manip_seq_joints24.p")
        c = [x for x in d.values() if x["seq_name"] == seq]
        del d
        if c:
            v = c[0]
            break
    T = len(v["trans"])
    mdl = smplx.create(str(cp.SMPLX_DIR), model_type="smplx", gender=str(v["gender"]), num_betas=16, use_pca=False, flat_hand_mean=True, batch_size=T)
    with torch.no_grad():
        V = mdl(betas=torch.tensor(np.repeat(v["betas"], T, 0), dtype=torch.float32), global_orient=torch.tensor(v["root_orient"], dtype=torch.float32),
                body_pose=torch.tensor(v["pose_body"], dtype=torch.float32), transl=torch.tensor(v["trans"], dtype=torch.float32)).vertices.numpy().astype(np.float64)
    obj = cp.obj("omomo_largetable")
    hv, seg = cp.hand_vertex_ids(), segment_vertex_ids()
    hand = np.full((T, 1, 2), -np.inf)
    part = np.full((T, 1, len(LINKS)), -np.inf)
    for t in range(T):
        sc = v["obj_scale"][t]
        Vl = ((V[t] - v["obj_trans"][t, :, 0]) @ v["obj_rot"][t]) / sc
        for hi, side in enumerate(("left", "right")):
            pl = Vl[hv[side]]
            d, j = obj["tree"].query(pl)
            hand[t, 0, hi] = (-np.sign(np.einsum("ij,ij->i", pl - obj["p"][j], obj["n"][j])) * d).max() * sc
        part[t, 0] = part_max_sd(obj, Vl, sc, seg)
    return hand, part, 30, dict(sequence=seq, object_key="omomo_largetable")


# ------------------------------------------------------------------------------------------------- robot side
def quat_wxyz_to_xyzw(q):
    return np.concatenate([q[..., 1:4], q[..., 0:1]], -1)


def robot_palms_obj(robot, offs, root_pos, root_quat_xyzw, dof, obj_pos, obj_quat_xyzw):
    """palm points in the object frame [F, R, 2, 3]."""
    import contact_stats as cs
    tl = robot.fk(root_pos, root_quat_xyzw, dof)
    Ti = np.linalg.inv(cs.pose_mat(obj_pos, obj_quat_xyzw))[:, None]
    out = []
    for side in ("left", "right"):
        M = Ti @ tl[f"{side}_rubber_hand_link"]
        out.append(np.einsum("...ij,j->...i", M[..., :3, :3], offs[side]) + M[..., :3, 3])
    return np.stack(out, -2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=("table", "push"), required=True)
    ap.add_argument("--runtime", type=Path, required=True, help="table: paired runtime fps50 npz; push: single-robot fps50 npz")
    ap.add_argument("--object-k", type=float, default=1.0, help="table: object scale of the reference (PhysX pieces x k)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    import contact_gap as cg
    import contact_stats as cs
    from object_surface import UnionSurface
    robot = cs.Robot("physx")
    offs = cg.palm_offsets()
    z = np.load(a.runtime, allow_pickle=True)
    if a.task == "table":
        import eval_core4d_hulls as eh
        names = [str(n) for n in z["joint_names"]]
        dof = z["agent_joint_pos"][:, :, [names.index(n) for n in robot.dofs]].astype(np.float64)   # [F, 2, 29]
        bi = [str(n) for n in z["body_names"]].index("pelvis")
        root_pos = z["agent_body_pos_w"][:, :, bi].astype(np.float64)
        root_q = quat_wxyz_to_xyzw(z["agent_body_quat_w"][:, :, bi].astype(np.float64))
        obj_pos, obj_q = z["object_pos_w"].astype(np.float64), quat_wxyz_to_xyzw(z["object_quat_w"].astype(np.float64))
        palms = robot_palms_obj(robot, offs, root_pos, root_q, dof, obj_pos, obj_q)              # [F, 2, 2, 3]
        surf = UnionSurface(eh.dump_object_meshes("table", a.object_k))
        hand, part, fps_src, hinfo = human_core4d("table")
        persons = [0, 1]
        geometry = f"T10 PhysX pieces of the Table training object x {a.object_k}"
    else:
        import dual_stretch as ds
        from eval_core4d_hulls import urdf_box_meshes
        jp = z["joint_pos"].astype(np.float64)                     # [F, 36]: root xyz, quat wxyz, 29 dofs (checked vs the 30 Hz source in T14)
        names = [str(n) for n in z["joint_names"]]
        dof = jp[:, 7:][:, [names.index(n) for n in robot.dofs]]
        obj_pos, obj_q = z["object_pos_w"].astype(np.float64), quat_wxyz_to_xyzw(z["object_quat_w"].astype(np.float64))
        single = robot_palms_obj(robot, offs, jp[:, None, :3], quat_wxyz_to_xyzw(jp[:, None, 3:7]), dof[:, None], obj_pos, obj_q)[:, 0]
        ex = np.array([1.0, 0.0, 0.0])
        palms = np.stack([single - 0.5 * PUSH_SPACING_M * ex, single + 0.5 * PUSH_SPACING_M * ex], 1)  # PairedA1Reference offsets
        surf = UnionSurface(urdf_box_meshes(ds.TABLE["push"][0]))
        hand, part, fps_src, hinfo = human_omomo(ds.SEQ["push"])
        persons = [0, 0]
        geometry = f"training table URDF boxes {ds.TABLE['push'][0].name}"
    F = len(palms)
    src = np.clip(np.round(np.arange(F) / 50.0 * fps_src).astype(int), 0, len(hand) - 1)
    should = np.stack([hand[src, p] >= CONTACT_M for p in persons], 1)            # [F, 2, 2]
    far = np.stack([part[src, p] < -FAR_M for p in persons], 1)                   # [F, 2, L]
    point, dist = surf.nearest(palms)
    # the simulator tracks wrist_yaw_link, not the rubber hand (fixed joint): palm point also in the wrist_yaw_link frame
    wrist = []
    for side in ("left", "right"):
        j = next(j for j in robot.u.robot.joints if j.child == f"{side}_rubber_hand_link")
        assert j.parent == f"{side}_wrist_yaw_link" and j.type == "fixed", (j.parent, j.type)
        M = np.eye(4) if j.origin is None else np.asarray(j.origin)
        wrist.append(M[:3, :3] @ offs[side] + M[:3, 3])
    np.savez(a.out, should_contact=should, contact_point=point.astype(np.float32), human_far=far, link_names=np.array(LINKS),
             palm_offset=np.stack([offs["left"], offs["right"]]).astype(np.float32), palm_offset_wrist=np.stack(wrist).astype(np.float32),
             source_frame=src.astype(np.int32),
             palm_to_surface_m=dist.astype(np.float32))
    rec = dict(task=a.task, runtime=str(a.runtime), runtime_sha256=sha(a.runtime), frames=F, source_fps=fps_src, human=hinfo,
               object_collision=geometry, should_contact_fraction={f"r{r}_{s}": round(float(should[:, r, h].mean()), 4) for r in (0, 1) for h, s in enumerate("LR")},
               palm_to_surface_in_contact_m=dict(median=float(np.median(dist[should])), p95=float(np.percentile(dist[should], 95))) if should.any() else None,
               human_far_fraction={l: round(float(far[:, :, i].mean()), 4) for i, l in enumerate(LINKS)},
               link_segments={l: dict(part=n, smplx_joints=list(j)) for l, (n, j) in LINK_SEGMENTS.items()}, body_parts_source="system1/t14/body_parts.py",
               out=str(a.out), out_sha256=sha(a.out),
               rules=__doc__)
    a.out.with_suffix(".json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({k: rec[k] for k in ("frames", "should_contact_fraction", "palm_to_surface_in_contact_m")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
