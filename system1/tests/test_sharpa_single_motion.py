#!/usr/bin/env python3
"""Smoke test (CPU) for system1/hand/sharpa_single_motion.py: the 73-joint SONIC sample walk for G1 + Sharpa.

    hssim_s1/bin/python -B system1/tests/test_sharpa_single_motion.py
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SRC = ROOT / "logs_system1/T01/sonic/refs/sonic_sample_A001"


def main():
    from system1.hand.robot_config import finger_body_names, finger_joint_names
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "walk_sharpa"
        r = subprocess.run([sys.executable, "-B", str(ROOT / "system1/hand/sharpa_single_motion.py"), "--prefix", str(SRC),
                            "--out-prefix", str(out)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]
        z0 = np.load(f"{SRC}.holosoma.npz")
        z = np.load(f"{out}.holosoma.npz")
        jn = [str(x) for x in z["joint_names"]]
        assert jn == [str(x) for x in z0["joint_names"]] + finger_joint_names() and len(jn) == 73
        assert np.array_equal(z["joint_pos"][:, :36], z0["joint_pos"]) and np.abs(z["joint_pos"][:, 36:]).max() == 0
        assert np.array_equal(z["joint_vel"][:, :35], z0["joint_vel"]) and np.abs(z["joint_vel"][:, 35:]).max() == 0
        nb0 = z0["body_pos_w"].shape[1]
        for k in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
            assert np.array_equal(z[k][:, :nb0], z0[k]), k
        bn = [str(x) for x in z["body_names"]]
        assert set(finger_body_names()) <= set(bn)  # every body of the G1 + Sharpa robot config is in the motion
        q = z["body_quat_w"][:, nb0:]
        assert np.allclose(np.linalg.norm(q, axis=-1), 1, atol=1e-5)
        for side in ("left", "right"):  # rigid to the wrist: constant distance, velocity = finite difference of position
            i, w = bn.index(f"{side}_index_fingertip"), bn.index(f"{side}_wrist_yaw_link")
            dist = np.linalg.norm(z["body_pos_w"][:, i] - z["body_pos_w"][:, w], axis=-1)
            assert dist.max() - dist.min() < 1e-6, (side, dist.min(), dist.max())
            fps = float(np.asarray(z["fps"]).reshape(-1)[0])
            fd = np.gradient(z["body_pos_w"][:, i], 1.0 / fps, axis=0)
            assert np.median(np.linalg.norm(fd - z["body_lin_vel_w"][:, i], axis=-1)) < 0.05
        assert np.array_equal(np.load(f"{out}.sonic.npz")["dof_pos"], np.load(f"{SRC}.sonic.npz")["dof_pos"])
    print("ok")


if __name__ == "__main__":
    raise SystemExit(main())
