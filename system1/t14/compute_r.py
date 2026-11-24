#!/usr/bin/env python3
"""T14 section 11 item 1: r = training object size / (size of the object the person used x s), per task, person and axis.

  * s = the person's human-to-robot scale used by the retargeting (CORE4D: human_to_robot_scale of
    prepare_core4d_person_retarget_input; OMOMO: smpl_scale = robot height / person height);
  * the person's object = the captured mesh at the size the person interacted with (CORE4D object_models <name>_m.obj;
    OMOMO: the OMOMO capture largetable_cleaned_simplified.obj x obj_scale, and, for reference, the InterMimic largetable.obj
    that the OMOMO_new motion and the retargeting scene use);
  * the training object = the simulator collision shape of the 1.0x training object (Table / Bin / Chair: the T10 PhysX pieces;
    Push / Pull: the training-table URDF boxes);
  * axes in the object frame: height = the local axis closest to the world vertical at the first frame; length = the longer
    and width = the shorter of the other two (Push / Pull: length = the stretch axis);
  * also: the table-top / object-top height above the floor the person stands on (first frame; floor = the lowest SMPL-X
    vertex), x s, next to the training object's height.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "system1/t14"))
sys.path.insert(0, str(ROOT / "system1/t12v2"))
sys.path.insert(0, str(ROOT / "src/holosoma_retargeting"))


def extents(points, up_local, length_axis=None):
    lo, hi = points.min(0), points.max(0)
    e = hi - lo
    h = int(np.argmax(np.abs(up_local)))
    rest = [i for i in range(3) if i != h]
    if length_axis is None:
        length_axis = max(rest, key=lambda i: e[i])
    width_axis = [i for i in rest if i != length_axis][0]
    return dict(height=float(e[h]), width=float(e[width_axis]), length=float(e[length_axis]), axes=dict(height=h, width=width_axis, length=length_axis))


def core4d(task):
    import table_shared as ts
    import eval_core4d_hulls as eh
    from holosoma_retargeting.data_utils.core4d_adapter import load_canonical_core4d_sequence, resample_core4d_pair_sequence
    from holosoma_retargeting.data_utils.core4d_retarget import prepare_core4d_person_retarget_input
    cfg = ts.TASKS[task]
    seq_dir = ts.CORE4D / "CORE4D_Real/human_object_motions" / cfg["raw"]
    To = np.load(seq_dir / "smooth_objposes.npy")
    up_local = To[0, :3, :3].T @ np.array([0.0, 1.0, 0.0])  # raw CORE4D is Y-up
    mesh_path = next((ts.CORE4D / "CORE4D_Real/object_models").rglob(f"{cfg['obj_key'].split('_', 1)[1]}_m.obj"))
    raw = trimesh.load(mesh_path, force="mesh")
    hum = extents(np.asarray(raw.vertices), up_local)
    train = extents(np.concatenate([m.vertices for m in eh.dump_object_meshes(task, 1.0)]), up_local, hum["axes"]["length"])
    seq = resample_core4d_pair_sequence(load_canonical_core4d_sequence(cfg["canon"]), 30)
    out = dict(object=dict(person_object=str(mesh_path), person_object_dims_m={k: round(v, 4) for k, v in hum.items() if k != "axes"},
                           training_object="T10 PhysX pieces x 1.0", training_dims_m={k: round(v, 4) for k, v in train.items() if k != "axes"}, axes_local=hum["axes"]), persons={})
    # top height above the person's floor (first frame, raw Y-up world)
    top = float((np.asarray(raw.vertices) @ To[0, :3, :3].T + To[0, :3, 3])[:, 1].max())
    for p in (1, 2):
        pin = prepare_core4d_person_retarget_input(seq, p - 1)
        s = float(pin.human_to_robot_scale)
        V = np.load(seq_dir / f"person{p}_poses.npz", allow_pickle=True)["arr_0"].item()["vertices"][0]
        floor = float(V[:, 1].min())
        r = {k: round(train[k] / (hum[k] * s), 4) for k in ("height", "width", "length")}
        out["persons"][f"person{p}"] = dict(s=round(s, 5), r=r, object_top_above_floor_m=round(top - floor, 4),
                                            object_top_above_floor_x_s_m=round((top - floor) * s, 4), training_height_m=round(train["height"], 4))
    return out


def omomo(task):
    import joblib
    import dual_stretch as ds
    import single_shared as ss
    from eval_core4d_hulls import urdf_box_meshes
    seq = ds.SEQ[task]
    v = None
    for split in ("train", "test"):
        d = joblib.load(ss.OMOMO_P / f"{split}_diffusion_manip_seq_joints24.p")
        c = [x for x in d.values() if x["seq_name"] == seq]
        del d
        if c:
            v = c[0]
            break
    cap = trimesh.load(ss.OMOMO_P / "captured_objects/largetable_cleaned_simplified.obj", force="mesh")
    capv = np.asarray(cap.vertices) * float(v["obj_scale"][0])
    inter = trimesh.load(ROOT / "src/holosoma_retargeting/holosoma_retargeting/models/largetable/largetable.obj", force="mesh")
    up = v["obj_rot"][0].T @ np.array([0.0, 0.0, 1.0])  # OMOMO world is Z-up
    src_urdf, axis = ds.TABLE[task]
    train = extents(np.concatenate([m.vertices for m in urdf_box_meshes(src_urdf)]), up, axis)
    hum_cap = extents(capv, up, axis)
    hum_int = extents(np.asarray(inter.vertices), up, axis)
    s = float(np.load(ROOT / f"logs_system1/t14/r5/single_{task}_C/{task}_new_qpos.npz")["smpl_scale"])
    # top height above the floor in the OMOMO world (first frame): object top vs lowest body joint of the capture
    top = float((capv @ v["obj_rot"][0].T + v["obj_trans"][0, :, 0])[:, 2].max())
    import smplx
    import torch
    m = smplx.create(str(ss.SMPLX_DIR), model_type="smplx", gender=str(v["gender"]), num_betas=16, use_pca=False, flat_hand_mean=True, batch_size=1)
    with torch.no_grad():
        V = m(betas=torch.tensor(v["betas"][:1], dtype=torch.float32), global_orient=torch.tensor(v["root_orient"][:1], dtype=torch.float32),
              body_pose=torch.tensor(v["pose_body"][:1], dtype=torch.float32), transl=torch.tensor(v["trans"][:1], dtype=torch.float32)).vertices.numpy()[0]
    floor = float(V[:, 2].min())
    rr = lambda hum: {k: round(train[k] / (hum[k] * s), 4) for k in ("height", "width", "length")}
    return dict(object=dict(person_object_capture_dims_m={k: round(x, 4) for k, x in hum_cap.items() if k != "axes"},
                            person_object_intermimic_dims_m={k: round(x, 4) for k, x in hum_int.items() if k != "axes"},
                            training_object=str(src_urdf.relative_to(ROOT)), training_dims_m={k: round(x, 4) for k, x in train.items() if k != "axes"}, axes_local=train["axes"]),
                persons=dict(person=dict(s=round(s, 5), r_vs_capture=rr(hum_cap), r_vs_intermimic=rr(hum_int), object_top_above_floor_m=round(top - floor, 4),
                                         object_top_above_floor_x_s_m=round((top - floor) * s, 4), training_height_m=round(train["height"], 4))))


def main():
    out = {t: core4d(t) for t in ("table", "chair", "bin")} | {t: omomo(t) for t in ("push", "pull")}
    p = ROOT / "results_system1/T14_retargeting/step6/r_ratios.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dict(rules=__doc__, tasks=out), indent=1) + "\n")
    for t, x in out.items():
        print(t, json.dumps(x["object"]["training_dims_m"]), json.dumps(x["persons"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
