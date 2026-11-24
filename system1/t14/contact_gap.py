#!/usr/bin/env python3
"""T14 round 3: contact-position error (Kevin 2026-11-23; reported only, not an acceptance criterion).

Definition (Kevin's text; the HOI-Retarget contact-point gap / ReCHOIR contact distance are named as the reference style):
  * frames: every frame in which the person's hand touches the object (the source contact rule used throughout T14: any
    SMPL-X hand vertex within 1.5 cm of the captured object mesh);
  * target: the person's palm point (mean of the 67 palm-side SMPL-X vertices) -> nearest point on the captured object mesh,
    in the object frame -> moved to the level's object by the same transform that made this level's reference:
      CORE4D: x s (s = s_bar k; Bin s_bar = 1), robot r <-> person r + 1;
      Push / Pull: the role isometry of the level (table frame): Push robot 0 / 1 = source -/+ c(m) along table x, same
      hands; Pull robot 0 = source - c(m) along table z, robot 1 = source mirrored across the table's local XY plane + c(m),
      left / right hands swapped (the G1 sagittal mirror of dual_pull_reference);
  * measured: the robot palm point (the rubber-hand palm point on the simulator's hand hull, simulator FK) -> nearest point
    on the simulator collision surface of that level's object (object_surface.UnionSurface: Table / Push / Pull boxes,
    Bin / Chair PhysX pieces x k); error = distance between that point and the target;
  * per robot and hand: frame count, mean and 95th percentile (cm). CORE4D reference frame of a 15 Hz source frame t = 2t.

    sonic_s1/bin/python -B system1/t14/contact_gap.py --task table --ref new_k1.0=<pair npz>@1.0 ... --out <json>
    sonic_s1/bin/python -B system1/t14/contact_gap.py --task push --ref new_m1.0=<single qpos npz>@1.0 --ref current=<fps50 npz>@1.0@fps50 ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "system1/t14"))
sys.path.insert(0, str(ROOT / "system1/t12v2"))
sys.path.insert(0, str(ROOT / "src/holosoma_retargeting"))
import contact_stats as cs  # noqa: E402
import table_shared as tsh  # noqa: E402
from object_surface import UnionSurface  # noqa: E402

CONTACT = -0.015
S_BAR = dict(table=0.7412283856, chair=0.744193062691, bin=1.0)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def palm_offsets():
    shapes = dict(cs.dump_shapes("table", "/World/envs/env_0/Robot/")[0])
    return {s: tsh.palm_point(shapes[f"{s}_rubber_hand_link"].vertices, s) for s in ("left", "right")}


def robot_palms_local(robot, pair, obj_q, offs):
    """pair [T, 2, 36] wxyz, obj_q [T, 7] wxyz -> palm points in the object frame [T, 2 robots, 2 hands, 3]."""
    tl = robot.fk(pair[:, :, :3], np.concatenate([pair[:, :, 4:7], pair[:, :, 3:4]], -1), pair[:, :, 7:])
    To = cs.pose_mat(obj_q[:, :3], np.concatenate([obj_q[:, 4:7], obj_q[:, 3:4]], -1))
    Ti = np.linalg.inv(To)[:, None]
    out = []
    for side in ("left", "right"):
        M = Ti @ tl[f"{side}_rubber_hand_link"]
        out.append(np.einsum("...ij,j->...i", M[..., :3, :3], offs[side]) + M[..., :3, 3])
    return np.stack(out, 2)


# ------------------------------------------------------------------------------------------------------- human side
def human_core4d(task):
    cfg = tsh.TASKS[task]
    tsh.SEQ_RAW, tsh.OBJ_KEY = tsh.CORE4D / "CORE4D_Real/human_object_motions" / cfg["raw"], cfg["obj_key"]
    _, contact, palm_local = tsh.source_palms_and_contacts(np.eye(3))
    mesh_path = next((tsh.CORE4D / "CORE4D_Real/object_models").rglob(f"{cfg['obj_key'].split('_', 1)[1]}_m.obj"))
    raw = trimesh.load(mesh_path, force="mesh")
    T = len(contact[(1, "left")])
    flags, target = np.zeros((T, 2, 2), bool), np.zeros((T, 2, 2, 3))
    for p in (1, 2):
        for h, side in enumerate(("left", "right")):
            flags[:, p - 1, h] = contact[(p, side)]
            target[:, p - 1, h] = trimesh.proximity.closest_point(raw, palm_local[(p, side)])[0]
    return flags, target, dict(mesh=str(mesh_path), mesh_sha256=sha(mesh_path), fps=15)


def human_omomo(task):
    import single_shared as ss
    import dual_stretch as ds
    from holosoma_retargeting.src.utils import load_intermimic_data
    seq = ds.SEQ[task]
    palms, fit = ss.omomo_palms(seq)  # OMOMO_new world [T, 2, 3]
    _, obj = load_intermimic_data(str(ss.OMOMO_NEW / f"{seq}.pt"))  # [T, 7] qw qx qy qz x y z
    R = cs.pose_mat(obj[:, 4:7], obj[:, [1, 2, 3, 0]])
    local = np.einsum("tji,thj->thi", R[:, :3, :3], palms - obj[:, None, 4:7])
    mesh_path = ROOT / "src/holosoma_retargeting/holosoma_retargeting/models/largetable/largetable.obj"
    mesh = trimesh.load(mesh_path, force="mesh")
    target = np.stack([trimesh.proximity.closest_point(mesh, local[:, h])[0] for h in (0, 1)], 1)
    h_sd, _ = ds.human_hand_sd(seq)
    return h_sd >= CONTACT, target, dict(mesh=str(mesh_path), mesh_sha256=sha(mesh_path), fps=30, palm_fit_residual_max_m=fit["fit_residual_max_m"])


# ------------------------------------------------------------------------------------------------------- per reference
def summarize(err, flags):
    out = {}
    for r in (0, 1):
        for h, side in enumerate(("left", "right")):
            e = err[:, r, h][flags[:, r, h]]
            out[f"r{r}_{side}"] = (dict(frames=int(len(e)), mean_cm=round(100 * float(e.mean()), 2), p95_cm=round(100 * float(np.percentile(e, 95)), 2))
                                   if len(e) else dict(frames=0, mean_cm=None, p95_cm=None))
        e = err[:, r][flags[:, r]]
        out[f"r{r}"] = dict(frames=int(len(e)), mean_cm=round(100 * float(e.mean()), 2), p95_cm=round(100 * float(np.percentile(e, 95)), 2)) if len(e) else None
    return out


def core4d_ref(task, robot, offs, human, path, k):
    z = np.load(path)
    s, info = core4d_arrays(task, robot, offs, human, z["robot_qpos"].astype(np.float64), z["object_qpos"].astype(np.float64), k)
    return s, dict(path=str(path), sha256=sha(path), **info)


def core4d_arrays(task, robot, offs, human, pair, obj_q, k):
    flags15, target_raw, _ = human
    from sim_scene import object_pieces
    surf = UnionSurface(object_pieces(task, k))
    s = S_BAR[task] * k
    t15 = np.arange(min(len(flags15), (len(pair) + 1) // 2))
    P = robot_palms_local(robot, pair[2 * t15], obj_q[2 * t15], offs)
    near, _ = surf.nearest(P)
    err = np.linalg.norm(near - s * target_raw[t15], axis=-1)
    return summarize(err, flags15[t15]), dict(k=k, s=s)


def omomo_single(path, fps50):
    z = np.load(path)
    if not fps50:
        return z["qpos"].astype(np.float64)
    jp, op, oq = z["joint_pos"], z["object_pos_w"], z["object_quat_w"]
    T30 = int(round((len(jp) - 1) * 30 / 50)) + 1
    idx = np.minimum(np.round(np.arange(T30) * 50 / 30).astype(int), len(jp) - 1)  # nearest 50 Hz frame of each 30 Hz frame
    return np.concatenate([jp[idx], op[idx], oq[idx]], 1).astype(np.float64)


def omomo_ref(task, robot, offs, human, path, m, fps50):
    import dual_stretch as ds
    q = omomo_single(path, fps50)
    pair, obj_q = ds.make_pair(task, q, m)
    src_urdf, axis = ds.TABLE[task]
    urdf = ds.stretch_table(src_urdf, axis, m, ROOT / f"logs_system1/t14/r3/contact_gap_urdf/{task}_m{m:g}.urdf")
    s, info = omomo_arrays(task, robot, offs, human, pair, obj_q, m, urdf)
    return s, dict(path=str(path), sha256=sha(path), input_fps50_nearest_frames=fps50, **info)


def omomo_arrays(task, robot, offs, human, pair, obj_q, m, urdf):
    import dual_stretch as ds
    from eval_core4d_hulls import urdf_box_meshes
    flags, target_src, _ = human
    axis = ds.TABLE[task][1]
    surf = UnionSurface(urdf_box_meshes(urdf))
    c = ds.offset_c(m)
    n = min(len(flags), len(pair))
    tgt = np.zeros((n, 2, 2, 3))
    fl = np.zeros((n, 2, 2), bool)
    e = np.eye(3)[axis]
    if task == "push":
        tgt[:, 0], tgt[:, 1] = target_src[:n] - c * e, target_src[:n] + c * e
        fl[:, 0], fl[:, 1] = flags[:n], flags[:n]
    else:
        Mz = np.diag([1.0, 1.0, -1.0])
        tgt[:, 0] = target_src[:n] - c * e
        tgt[:, 1] = target_src[:n, ::-1] @ Mz + c * e  # robot 1 left <- source right mirrored
        fl[:, 0], fl[:, 1] = flags[:n], flags[:n, ::-1]
    P = robot_palms_local(robot, pair[:n], obj_q[:n], offs)
    near, _ = surf.nearest(P)
    err = np.linalg.norm(near - tgt, axis=-1)
    return summarize(err, fl), dict(m=m, c_m=round(c, 7), urdf=str(urdf))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=("table", "bin", "chair", "push", "pull"), required=True)
    ap.add_argument("--ref", action="append", required=True, help="name=path@k (CORE4D pair) | name=path@m[@fps50] (Push / Pull single)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    robot = cs.Robot("physx")
    offs = palm_offsets()
    omomo = a.task in ("push", "pull")
    human = human_omomo(a.task) if omomo else human_core4d(a.task)
    res = {}
    for spec in a.ref:
        name, rest = spec.split("=", 1)
        parts = rest.split("@")
        if omomo:
            s, info = omomo_ref(a.task, robot, offs, human, parts[0], float(parts[1]), len(parts) > 2 and parts[2] == "fps50")
        else:
            s, info = core4d_ref(a.task, robot, offs, human, parts[0], float(parts[1]))
        res[name] = dict(**info, contact_position_error=s)
        print(name, json.dumps(s), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(dict(task=a.task, rules=__doc__, human=human[2], human_contact_frames=int(human[0].sum()),
                                     palm_offsets_m={k: v.round(6).tolist() for k, v in offs.items()}, refs=res), indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
