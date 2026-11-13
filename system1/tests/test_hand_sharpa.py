#!/usr/bin/env python3
"""Smoke test for T06 system1/hand (CPU only): Sharpa model / FK / palm frame / mount, the assembled G1+Sharpa URDF,
the object SDF, the closed-form wrist solve and a short finger-reference run on a few frames.

    taskset -c 0-15 hssim_s1/bin/python -B system1/tests/test_hand_sharpa.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from system1.hand import finger_ref as fr  # noqa: E402
from system1.hand.sdf import GridSDF  # noqa: E402
from system1.hand.sharpa import SIDES, SharpaHand, mount_transform, pd_table, rubber_palm_basis  # noqa: E402


def main():
    torch.set_num_threads(4)
    for side in SIDES:
        h = SharpaHand(side)
        assert len(h.dof_names) == 22 and abs(h.mass - 1.31) < 0.01, (len(h.dof_names), h.mass)
        q0 = torch.zeros(22, dtype=torch.float64)
        T = h.fk(q0)
        assert torch.allclose(T[h.root], torch.eye(4, dtype=torch.float64))
        # FK is batched and consistent
        qb = (torch.rand(5, 22, dtype=torch.float64) * (h.upper - h.lower) + h.lower)
        Tb = h.fk(qb)
        T2 = h.fk(qb[2])
        assert torch.allclose(Tb[f"{side}_index_fingertip"][2], T2[f"{side}_index_fingertip"])
        # palm frame: orthonormal, right-handed, normal on the palmar side (finger flexion moves tips along +normal)
        B = h.palm_basis()
        assert np.allclose(B.T @ B, np.eye(3), atol=1e-9) and np.linalg.det(B) > 0
        for f in ("index", "middle", "ring"):
            q = q0.clone(); q[h.dof_names.index(f"{side}_{f}_MCP_FE")] = 0.8
            d = (h.fk(q)[f"{side}_{f}_fingertip"][:3, 3] - T[f"{side}_{f}_fingertip"][:3, 3]).numpy()
            assert d @ B[:, 2] > 0.03, (side, f, d @ B[:, 2])
        # mount: flange +z along wrist +x, flange origin at the rubber-hand palm joint, palm normal close to the rubber one
        M, info = mount_transform(side, h)
        assert np.allclose(M[:3, :3] @ [0, 0, 1], [1, 0, 0], atol=1e-9) and np.allclose(M[:3, 3], [0.0415, 0.003 if side == "left" else -0.003, 0])
        assert info["palm_normal_angle_to_rubber_deg"] < 10 and np.allclose(rubber_palm_basis(side).T @ rubber_palm_basis(side), np.eye(3))
        pd = pd_table(side)
        assert len(pd) == 22 and all(v["kp"] > 0 and v["kd"] > 0 and 0 < v["action_scale"] < 0.5 for v in pd.values())
        pts, links = h.points_world(qb)
        assert pts.shape[:2] == (5, len(links)) and torch.isfinite(pts).all()

    # assembled URDF: 29 body + 44 finger joints, the two rubber hands gone
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "sub" / "g1_29dof_sharpa.urdf"
        r = subprocess.run([sys.executable, "-B", str(ROOT / "system1/hand/assemble.py"), "--out", str(out)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]
        man = json.loads(out.with_suffix(".manifest.json").read_text())
        assert man["movable_joints"] == 73 and abs(man["mass_difference_kg"] - (2.622 - 0.34)) < 0.01
        txt = out.read_text()
        assert "rubber_hand_link" not in txt and txt.count("_hand_flange\"") >= 2
        import yourdfpy
        u = yourdfpy.URDF.load(str(out), build_scene_graph=True, load_meshes=False)
        assert len(u.actuated_joint_names) == 73

    # SDF: sign and value near the surface of a unit-ish box
    import trimesh
    with tempfile.TemporaryDirectory() as d:
        box = Path(d) / "box.obj"
        trimesh.creation.box(extents=(0.2, 0.2, 0.2)).export(box)
        s = GridSDF(box, resolution=0.004, margin=0.05)
        v = s(torch.tensor([[0.0, 0.0, 0.0], [0.11, 0.0, 0.0], [0.09, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=torch.float64)).numpy()
        assert abs(v[0] + 0.1) < 0.005 and abs(v[1] - 0.01) < 0.002 and abs(v[2] + 0.01) < 0.002 and v[3] > 0.35, v

    # closed-form wrist solve on the real Bin reference: self-consistent, exact palm, within limits
    ref = dict(np.load(fr.REFERENCE, allow_pickle=True))
    h = SharpaHand("right")
    M, _ = mount_transform("right", h)
    ang, diag, _ = fr.solve_wrist(ref, 1, "right", M[:3, :3] @ h.palm_basis())
    assert diag["fk_consistency_rot_deg_max"] < 1e-6 and diag["palm_orientation_error_deg_max"] < 1e-6, diag
    assert diag["within_limits_frames"] == diag["frames"] and ang.shape == (diag["frames"], 3)
    # identical palm basis -> identical wrist angles (the solver reproduces the A1 rubber-hand solution)
    ang_rub, d2, _ = fr.solve_wrist(ref, 1, "right", fr.rubber_palm_basis("right"))
    jn = [str(n) for n in ref["joint_names"]]
    jw = [jn.index(f"right_wrist_{n}_joint") for n in ("roll", "pitch", "yaw")]
    assert np.abs(ang_rub - ref["agent_joint_pos"][:, 1, jw]).max() < 1e-6, np.abs(ang_rub - ref["agent_joint_pos"][:, 1, jw]).max()

    # ramp: 0.2 s smoothstep around contact
    c = np.zeros(60, bool); c[20:30] = True
    s = fr.ramp_weights(c, 50.0)
    assert s[20:30].min() == 1 and s[:9].max() == 0 and s[41:].max() == 0 and 0 < s[15] < 1 and np.all(np.diff(s[9:21]) >= 0)

    # short finger-reference run (track and locked) on a few frames
    with tempfile.TemporaryDirectory() as d:
        for mode in ("track", "locked"):
            r = subprocess.run([sys.executable, "-B", str(ROOT / "system1/hand/finger_ref.py"), "--mode", mode, "--out", f"{d}/{mode}",
                                "--frames", "230:236", "--iters", "5"], capture_output=True, text=True)
            assert r.returncode == 0, r.stderr[-3000:]
            z = np.load(f"{d}/{mode}/finger_reference.npz", allow_pickle=True)
            assert z["agent_finger_joint_pos"].shape == (497, 2, 44) and z["agent_joint_pos"].shape == (497, 2, 29)
            summ = json.loads(Path(f"{d}/{mode}/summary.json").read_text())
            assert set(summ["gate"]) >= {"penetration_ok", "contact_ok", "joint_limits_ok"}
            if mode == "locked":
                assert np.abs(z["agent_finger_joint_pos"]).max() == 0.0  # open pose everywhere
            # body joints other than the six wrists are unchanged
            keep = [i for i, n in enumerate(jn) if "wrist" not in n]
            assert np.array_equal(z["agent_joint_pos"][:, :, keep], ref["agent_joint_pos"][:, :, keep])
    print("ok")


if __name__ == "__main__":
    raise SystemExit(main())
