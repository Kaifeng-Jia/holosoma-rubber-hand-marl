#!/usr/bin/env python3
"""T06 step 3: runtime reference for the G1 + Sharpa robot (73 joints), built from the D Bin runtime reference and a
step 2 finger reference (system1/hand/finger_ref.py output).

What changes relative to core4d_pair_runtime_fps50.npz (everything else is copied bit for bit):
  * joint_names: the 29 G1 joints + the 44 Sharpa finger joints (finger_reference.npz order);
    agent_joint_pos: body joints from the finger reference (= D reference with the six wrist joints re-solved for the
    Sharpa palm), finger joints from the finger reference (--fingers open: all zeros, the locked mode);
    agent_joint_vel: np.gradient of agent_joint_pos at the reference fps (the method of the original file, checked:
    max difference 0 on the original joints);
  * body poses of <side>_wrist_{roll,pitch,yaw}_link: forward kinematics of the assembled URDF with the new wrist
    joints, anchored at the reference pose of <side>_elbow_link (the URDF and the MJCF the reference was made with
    differ by ~1 cm in the torso chain; anchoring keeps the original elbow); their velocities = original velocity + (np.gradient of the new pose -
    np.gradient of the original pose), i.e. the original velocity definition plus the finite-difference change;
  * appended bodies: every link of the assembled URDF that is not in the original body list (the Sharpa links),
    poses by forward kinematics anchored at the elbow as above, linear velocity np.gradient(position), angular velocity from consecutive rotations
    (central rotation-vector difference, world frame);
  * provenance: JSON with the inputs, sha256 and this method.
Self-check: forward kinematics of the original bodies with the original joints reproduces the original body poses
(reported; the reference was generated from the MuJoCo model of the same robot).

    taskset -c 0-15 hssim_s1/bin/python -B system1/hand/sharpa_reference.py \
        --fingers logs_system1/t06/finger_ref/<track run>/finger_reference.npz --out logs_system1/t06/reference/track.npz
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
BIN_DIR = ROOT / "src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/core4d_bucket003_20231020_071_a1"
REFERENCE = BIN_DIR / "core4d_pair_runtime_fps50.npz"
URDF = ROOT / "system1/hand/assets/g1_29dof_sharpa.urdf"
RUBBER_URDF = ROOT / "src/holosoma/holosoma/data/robots/g1/main_mesh_collision_rubberhand.urdf"
WRIST_LINKS = [f"{s}_wrist_{n}_link" for s in ("left", "right") for n in ("roll", "pitch", "yaw")]


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def pose(p, q_wxyz):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(np.asarray(q_wxyz)[[1, 2, 3, 0]]).as_matrix()
    T[:3, 3] = p
    return T


def fk_all(urdf_path: Path, ref: dict, joints: np.ndarray, joint_names: list[str], links: list[str], anchored: bool = False):
    """World poses (T, 2, L, 4, 4) of `links` for every frame / agent. Root = reference pelvis pose; anchored=True:
    each <side>_* link is placed relative to the reference pose of <side>_elbow_link (only the elbow -> link part of
    the URDF chain is used, so the URDF / MJCF offset differences of the torso chain do not enter)."""
    import yourdfpy
    u = yourdfpy.URDF.load(str(urdf_path), build_scene_graph=True, load_meshes=False, build_collision_scene_graph=False)
    names = list(u.actuated_joint_names)
    idx = [joint_names.index(n) for n in names]
    bn = [str(b) for b in ref["body_names"]]
    ip = bn.index("pelvis")
    assert u.base_link == "pelvis", u.base_link
    T, A = joints.shape[:2]
    out = np.empty((T, A, len(links), 4, 4))
    for a in range(A):
        for t in range(T):
            u.update_cfg(joints[t, a, idx])
            Tb = pose(ref["agent_body_pos_w"][t, a, ip], ref["agent_body_quat_w"][t, a, ip])
            anchors = {}
            if anchored:
                for s in ("left", "right"):
                    ie = bn.index(f"{s}_elbow_link")
                    anchors[s] = pose(ref["agent_body_pos_w"][t, a, ie], ref["agent_body_quat_w"][t, a, ie]) @ np.linalg.inv(u.get_transform(f"{s}_elbow_link", u.base_link))
            for k, l in enumerate(links):
                side = l.split("_", 1)[0]
                base = anchors[side] if (anchored and side in anchors) else Tb  # G1 links without a side (logo, head, imu): pelvis
                out[t, a, k] = base @ u.get_transform(l, u.base_link)
    return out, u


def ang_vel_world(R: np.ndarray, fps: float) -> np.ndarray:
    """R (T, ..., 3, 3) -> world angular velocity (T, ..., 3), central differences (one-sided at the ends)."""
    T = R.shape[0]
    w = np.zeros(R.shape[:-2] + (3,))
    for t in range(T):
        a, b = max(t - 1, 0), min(t + 1, T - 1)
        dR = R[b] @ np.swapaxes(R[a], -1, -2)
        w[t] = Rotation.from_matrix(dR.reshape(-1, 3, 3)).as_rotvec().reshape(dR.shape[:-2] + (3,)) * fps / (b - a)
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fingers", type=Path, required=True, help="finger_reference.npz (body joints with the Sharpa wrist + fingers)")
    ap.add_argument("--open", action="store_true", help="finger joints all zero (locked mode); body joints still from --fingers")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    ref = dict(np.load(REFERENCE, allow_pickle=False))
    fr = np.load(a.fingers, allow_pickle=True)
    fps = float(ref["fps"])
    jn = [str(x) for x in ref["joint_names"]]
    assert [str(x) for x in fr["joint_names"]] == jn
    fn = [str(x) for x in fr["finger_joint_names"]]
    body_q = np.asarray(fr["agent_joint_pos"], float)
    finger_q = np.zeros_like(fr["agent_finger_joint_pos"], dtype=float) if a.open else np.asarray(fr["agent_finger_joint_pos"], float)
    q73 = np.concatenate([body_q, finger_q], axis=2)
    names73 = jn + fn
    bn = [str(b) for b in ref["body_names"]]

    # self-check: original joints + rubber-hand URDF reproduce the original body poses
    chk = [b for b in bn if b not in ("world",)]
    import yourdfpy
    ur = yourdfpy.URDF.load(str(RUBBER_URDF), build_scene_graph=True, load_meshes=False, build_collision_scene_graph=False)
    chk = [b for b in chk if b in ur.link_map]
    P0, _ = fk_all(RUBBER_URDF, ref, np.asarray(ref["agent_joint_pos"], float), jn, chk)
    ib = [bn.index(b) for b in chk]
    pos_err = np.linalg.norm(P0[..., :3, 3] - ref["agent_body_pos_w"][:, :, ib], axis=-1)
    Rref = Rotation.from_quat(ref["agent_body_quat_w"][:, :, ib].reshape(-1, 4)[:, [1, 2, 3, 0]]).as_matrix().reshape(P0.shape[:3] + (3, 3))
    rot_err = np.degrees(np.linalg.norm(Rotation.from_matrix((np.swapaxes(Rref, -1, -2) @ P0[..., :3, :3]).reshape(-1, 3, 3)).as_rotvec(), axis=-1))
    selfcheck = dict(bodies=len(chk), pos_err_m_max=float(pos_err.max()), rot_err_deg_max=float(rot_err.max()),
                     note="root-anchored URDF FK vs the MJCF-made reference: torso/waist offsets differ (~1 cm); original bodies are kept as they are")
    # anchored self-check on the wrist links (original wrist joints): this is the construction used below
    Pw, _ = fk_all(RUBBER_URDF, ref, np.asarray(ref["agent_joint_pos"], float), jn, WRIST_LINKS + ["left_rubber_hand_link", "right_rubber_hand_link"], anchored=True)
    iw = [bn.index(b) for b in WRIST_LINKS + ["left_rubber_hand_link", "right_rubber_hand_link"]]
    selfcheck["elbow_anchored_wrist_hand_pos_err_m_max"] = float(np.linalg.norm(Pw[..., :3, 3] - ref["agent_body_pos_w"][:, :, iw], axis=-1).max())
    Rw = Rotation.from_quat(ref["agent_body_quat_w"][:, :, iw].reshape(-1, 4)[:, [1, 2, 3, 0]]).as_matrix().reshape(Pw.shape[:3] + (3, 3))
    selfcheck["elbow_anchored_wrist_hand_rot_err_deg_max"] = float(np.degrees(np.linalg.norm(Rotation.from_matrix((np.swapaxes(Rw, -1, -2) @ Pw[..., :3, :3]).reshape(-1, 3, 3)).as_rotvec(), axis=-1)).max())
    print("self-check:", selfcheck, flush=True)

    # new robot: wrist links + all Sharpa links
    us = yourdfpy.URDF.load(str(URDF), build_scene_graph=True, load_meshes=False, build_collision_scene_graph=False)
    new_links = [l for l in us.link_map if l not in bn]
    links = WRIST_LINKS + new_links
    P, _ = fk_all(URDF, ref, q73, names73, links, anchored=True)
    out = {k: v for k, v in ref.items()}
    out["joint_names"] = np.array(names73)
    out["agent_joint_pos"] = q73.astype(ref["agent_joint_pos"].dtype)
    out["agent_joint_vel"] = np.gradient(q73, 1.0 / fps, axis=0).astype(ref["agent_joint_vel"].dtype)
    pos = np.concatenate([ref["agent_body_pos_w"], np.zeros(ref["agent_body_pos_w"].shape[:2] + (len(new_links), 3))], axis=2)
    quat = np.concatenate([ref["agent_body_quat_w"], np.zeros(ref["agent_body_quat_w"].shape[:2] + (len(new_links), 4))], axis=2)
    lin = np.concatenate([ref["agent_body_lin_vel_w"], np.zeros(pos.shape[:2] + (len(new_links), 3))], axis=2)
    ang = np.concatenate([ref["agent_body_ang_vel_w"], np.zeros(pos.shape[:2] + (len(new_links), 3))], axis=2)
    Rn = P[..., :3, :3]
    pn = P[..., :3, 3]
    qn = Rotation.from_matrix(Rn.reshape(-1, 3, 3)).as_quat().reshape(Rn.shape[:3] + (4,))[..., [3, 0, 1, 2]]
    wrist_change = {}
    for k, l in enumerate(links):
        if l in WRIST_LINKS:
            i = bn.index(l)
            Rold = Rotation.from_quat(ref["agent_body_quat_w"][:, :, i].reshape(-1, 4)[:, [1, 2, 3, 0]]).as_matrix().reshape(pn.shape[:2] + (3, 3))
            lin[:, :, i] = ref["agent_body_lin_vel_w"][:, :, i] + np.gradient(pn[:, :, k], 1.0 / fps, axis=0) - np.gradient(ref["agent_body_pos_w"][:, :, i], 1.0 / fps, axis=0)
            ang[:, :, i] = ref["agent_body_ang_vel_w"][:, :, i] + ang_vel_world(Rn[:, :, k], fps) - ang_vel_world(Rold, fps)
            wrist_change[l] = dict(pos_m_max=float(np.linalg.norm(pn[:, :, k] - ref["agent_body_pos_w"][:, :, i], axis=-1).max()),
                                   rot_deg_max=float(np.degrees(np.linalg.norm(Rotation.from_matrix((np.swapaxes(Rold, -1, -2) @ Rn[:, :, k]).reshape(-1, 3, 3)).as_rotvec(), axis=-1)).max()))
            pos[:, :, i] = pn[:, :, k]
            quat[:, :, i] = qn[:, :, k]
        else:
            i = len(bn) + new_links.index(l)
            pos[:, :, i] = pn[:, :, k]
            quat[:, :, i] = qn[:, :, k]
            lin[:, :, i] = np.gradient(pn[:, :, k], 1.0 / fps, axis=0)
            ang[:, :, i] = ang_vel_world(Rn[:, :, k], fps)
    dt = ref["agent_body_pos_w"].dtype
    out.update(body_names=np.array(bn + new_links), agent_body_pos_w=pos.astype(dt), agent_body_quat_w=quat.astype(dt),
               agent_body_lin_vel_w=lin.astype(dt), agent_body_ang_vel_w=ang.astype(dt))
    prov = dict(json.loads(str(ref["provenance"])), system1_t06=dict(
        generator="system1/hand/sharpa_reference.py", base_reference=str(REFERENCE.relative_to(ROOT)), base_reference_sha256=sha256(REFERENCE),
        finger_reference=str(a.fingers), finger_reference_sha256=sha256(a.fingers), fingers="open (zeros)" if a.open else "finger reference",
        robot_urdf=str(URDF.relative_to(ROOT)), robot_urdf_sha256=sha256(URDF), appended_bodies=len(new_links), self_check=selfcheck,
        wrist_link_change=wrist_change))
    out["provenance"] = np.array(json.dumps(prov))
    for k, v in out.items():
        assert v.dtype != object, k
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, **out)
    meta = dict(output=str(a.out), output_sha256=sha256(a.out), joints=len(names73), bodies=len(bn) + len(new_links), **prov["system1_t06"])
    a.out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k != "wrist_link_change"}), flush=True)
    print("wrist link change", json.dumps(wrist_change))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
