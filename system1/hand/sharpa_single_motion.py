#!/usr/bin/env python3
"""T07 step 0: single-robot holosoma motion (<prefix>.holosoma.npz, the SONIC sample walk of T01) for the G1 + Sharpa
robot with the fingers locked open, so that system1/sonic/validate_single.py --sharpa can load it.

What changes (everything else is copied bit for bit):
  * joint_names: the 29 G1 joints + the 44 Sharpa finger joints (system1/hand/robot_config.finger_joint_names order);
    joint_pos / joint_vel: the finger columns are 0 (open pose, not moving), appended after the original columns;
  * appended bodies: every link of the assembled G1 + Sharpa URDF that is not in the original body list (the Sharpa
    links). With the fingers fixed at 0 each of them is rigid to <side>_wrist_yaw_link, so its pose is
    T_ref(<side>_wrist_yaw_link) @ T_urdf(<side>_wrist_yaw_link -> link, q = 0), its linear velocity
    v_wrist + w_wrist x (p_link - p_wrist) and its angular velocity w_wrist (exact rigid-body relations);
  * the rubber-hand bodies stay in the list (the G1 + Sharpa robot does not have them; the motion loader only
    looks up the robot's bodies, extra bodies are not used).
The SONIC reference (<prefix>.sonic.npz: 29 joints, 14 SONIC bodies) does not change and is copied next to the output.

    hssim_s1/bin/python -B system1/hand/sharpa_single_motion.py --prefix logs_system1/T01/sonic/refs/sonic_sample_A001 \
        --out-prefix logs_system1/t07/step0/refs/sonic_sample_A001_sharpa
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
URDF = ROOT / "system1/hand/assets/g1_29dof_sharpa.urdf"


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", type=Path, required=True)
    ap.add_argument("--out-prefix", type=Path, required=True)
    a = ap.parse_args()
    from system1.hand.robot_config import finger_joint_names
    import yourdfpy
    src = Path(f"{a.prefix}.holosoma.npz")
    z = dict(np.load(src, allow_pickle=False))
    jn = [str(x) for x in z["joint_names"]]
    bn = [str(x) for x in z["body_names"]]
    fn = finger_joint_names()
    assert len(jn) == 29 and z["joint_pos"].shape[1] == 29 + 7 and z["joint_vel"].shape[1] == 29 + 6
    T = z["joint_pos"].shape[0]
    u = yourdfpy.URDF.load(str(URDF), build_scene_graph=True, load_meshes=False, build_collision_scene_graph=False)
    assert all(n in u.actuated_joint_names for n in fn) and set(jn) <= set(u.actuated_joint_names)
    u.update_cfg(np.zeros(len(u.actuated_joint_names)))
    new = [l for l in u.link_map if l not in bn]
    out = dict(z)
    out["joint_names"] = np.array(jn + fn)
    out["joint_pos"] = np.concatenate([z["joint_pos"], np.zeros((T, len(fn)), z["joint_pos"].dtype)], axis=1)
    out["joint_vel"] = np.concatenate([z["joint_vel"], np.zeros((T, len(fn)), z["joint_vel"].dtype)], axis=1)
    P = np.zeros((T, len(new), 3)); Q = np.zeros((T, len(new), 4)); V = np.zeros((T, len(new), 3)); W = np.zeros((T, len(new), 3))
    for k, l in enumerate(new):
        side = l.split("_", 1)[0]
        assert side in ("left", "right"), l
        wy = f"{side}_wrist_yaw_link"
        iw = bn.index(wy)
        X = u.get_transform(l, wy)                         # link pose in the wrist-yaw frame (fingers at 0)
        Rw = Rotation.from_quat(z["body_quat_w"][:, iw][:, [1, 2, 3, 0]])  # wxyz -> xyzw
        pw = z["body_pos_w"][:, iw]
        P[:, k] = pw + Rw.apply(X[:3, 3])
        Q[:, k] = (Rw * Rotation.from_matrix(X[:3, :3])).as_quat()[:, [3, 0, 1, 2]]
        W[:, k] = z["body_ang_vel_w"][:, iw]
        V[:, k] = z["body_lin_vel_w"][:, iw] + np.cross(W[:, k], P[:, k] - pw)
    for key, add in (("body_pos_w", P), ("body_quat_w", Q), ("body_lin_vel_w", V), ("body_ang_vel_w", W)):
        out[key] = np.concatenate([z[key], add.astype(z[key].dtype)], axis=1)
    out["body_names"] = np.array(bn + new)
    a.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    dst = Path(f"{a.out_prefix}.holosoma.npz")
    np.savez(dst, **out)
    shutil.copyfile(f"{a.prefix}.sonic.npz", f"{a.out_prefix}.sonic.npz")
    # self-check: the Sharpa flange and fingertip of each side are where the URDF chain from the wrist puts them
    chk = {}
    for side in ("left", "right"):
        l = f"{side}_index_fingertip" if f"{side}_index_fingertip" in new else new[0]
        k = new.index(l)
        d = np.linalg.norm(P[:, k] - z["body_pos_w"][:, bn.index(f"{side}_wrist_yaw_link")], axis=-1)
        chk[side] = dict(link=l, distance_to_wrist_yaw_m=[float(d.min()), float(d.max())])
    meta = dict(source=str(src), source_sha256=sha256(src), output=str(dst), output_sha256=sha256(dst),
                sonic_reference_sha256=sha256(f"{a.out_prefix}.sonic.npz"), robot_urdf=str(URDF.relative_to(ROOT)), robot_urdf_sha256=sha256(URDF),
                frames=int(T), joints=len(jn) + len(fn), appended_bodies=len(new), fingers="0 (open, locked)", rigid_check=chk)
    Path(f"{a.out_prefix}.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
