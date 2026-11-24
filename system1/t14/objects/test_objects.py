"""Smoke tests for T14 step 7 (system1/t14/objects). Run: sonic_s1/bin/python -B system1/t14/objects/test_objects.py
(or pytest). Data-heavy components (single_object.py, render_*.py) are smoke-run on real data instead (README)."""
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "scripts"))


def test_runs_steps_heading():
    import omomo_motion as om
    import screen_omomo as so
    assert so.runs([0, 1, 1, 0, 1]) == [(1, 3), (4, 5)]
    J = np.zeros((3, 22, 3))
    J[:, 1, 0], J[:, 2, 0] = -0.1, 0.1  # left hip at -x, right hip at +x -> facing +y
    assert np.allclose(om.heading(J), [[0, 1, 0]] * 3)
    T = 90
    toe = np.zeros((T, 2, 3))
    toe[30:40, 0, 0] = np.linspace(0, 0.3, 10)  # one 0.3 m swing of the left toe in 1/3 s
    toe[40:, 0, 0] = 0.3
    assert so.steps_in(toe, 0, T) == 1


def test_face_label():
    import screen_omomo as so
    hull = np.array([[x, y, z] for x in (-0.2, 0.2) for y in (1.0, 1.4) for z in (0.0, 0.3)])  # box ahead of a person facing +y
    fwd = np.array([0.0, 1.0, 0.0])
    assert so.face_label(np.array([0.0, 1.0, 0.15]), hull, fwd) == "front"
    assert so.face_label(np.array([0.0, 1.4, 0.15]), hull, fwd) == "back"
    assert so.face_label(np.array([-0.2, 1.2, 0.15]), hull, fwd) == "side-left"
    assert so.face_label(np.array([0.2, 1.2, 0.15]), hull, fwd) == "side-right"
    assert so.face_label(np.array([0.0, 1.2, 0.0]), hull, fwd) == "bottom"


def test_pick_one_per_object():
    import screen_omomo as so
    eps = [dict(object=o, sequence=f"s_{o}_{i}", class_c=True, both_hands_share=1.0, peak_lift_m=0.5 + 0.01 * i, duration_s=3.0)
           for o in ("a", "b") for i in range(3)]
    picks, per = so.pick(eps, "c")
    assert len(picks) == 2 and {p["object"] for p in picks} == {"a", "b"} and len(per) == 2


def test_box_assets_and_poses():
    import box_assets as ba
    bf = dict(R_obb=np.eye(3), c_obb=np.array([1.0, 0.0, 0.0]), extents_m=np.array([0.3, 0.4, 0.2]), scale=0.5, mesh_sha256="x")
    with tempfile.TemporaryDirectory() as d:
        import xml.etree.ElementTree as ET
        ba.quality = lambda name, bf: {}  # no captured mesh in the test
        rec = ba.write_assets("testbox", bf, Path(d))
        root = ET.parse(rec["scene"]).getroot()
        body = [b for b in root.iter("body") if b.get("name") == "testbox_link"][0]
        g = body.find("geom")
        assert g.get("type") == "box" and np.allclose([float(x) for x in g.get("size").split()], [0.15, 0.2, 0.1])
        from reference_geometry_math import BoxUnion
        assert np.allclose(BoxUnion(rec["urdf"]).parts[0]["size"], [0.3, 0.4, 0.2])
    v = dict(obj_scale=np.array([0.5, 0.5]), obj_rot=np.stack([np.eye(3)] * 2), obj_trans=np.array([[[0.0], [0.0], [1.0]], [[1.0], [0.0], [1.0]]]))
    fit = dict(rotation=np.eye(3).tolist(), translation=[0.0, 0.0, 0.0])
    p, invalid = ba.poses_omomo_new(v, bf, fit)
    assert invalid == 0 and np.allclose(p[:, :4], [1, 0, 0, 0]) and np.allclose(p[:, 4:], [[0.5, 0, 1], [1.5, 0, 1]])  # centre = s * c_obb + t


def test_role_construct_box():
    import role_check_box as rc
    from reference_geometry_math import BoxUnion
    with tempfile.TemporaryDirectory() as d:
        src = BoxUnion(rc.box_urdf(Path(d) / "s.urdf", "b", [0.3, 0.4, 0.2]))
        tgt = BoxUnion(rc.box_urdf(Path(d) / "t.urdf", "b", [0.6, 0.4, 0.2]))
        face = next(f for f in src.faces if f["axis"] == 1 and f["sign"] == -1)  # hands on the -y face
        weighted = dict(source_anchors=face["corners"].copy(), weights=np.full(4, 0.25),
                        windows=[dict(face_id=face["id"], anchor_slice=[0, 4], hand="left_rubber_hand_link", start=0, end_exclusive=10)])
        spec = rc.construct_box(src, tgt, weighted, 0)
        ids = {c["id"] for c in spec["candidates"]}
        # distinct box dimensions: the 4 proper diagonal sign maps; all keep the face a real exterior one (the -y face maps to
        # -y or +y); 2 keep the vertical (identity, 180 deg yaw), 2 turn the box upside down (removed later by the gravity screen)
        assert spec["accepted_per_role"] == [4, 4], spec["accepted_per_role"]
        for c in spec["candidates"]:
            assert np.isclose(np.linalg.det(c["O"]), 1.0) and np.allclose(abs(c["O"]), np.eye(3))
        assert sum(c["O"][2, 2] > 0 for c in spec["candidates"]) == 4 and len(ids) == 8


def test_level_offsets():
    # dual_box.level: role r keeps its end region: c_r(m) = c_r -/+ (L(m) - L(1)) / 2 along the axis
    w, m = 0.3, 1.5
    L1, Lm = 2 * w, m * 2 * w
    c0, c1 = np.array([-w / 2, 0, 0]), np.array([w / 2, 0, 0])
    e = np.eye(3)[0]
    n0, n1 = c0 - (Lm - L1) / 2 * e, c1 + (Lm - L1) / 2 * e
    assert np.isclose(n1[0] - n0[0], Lm - w)  # region centres at +-(L / 2 - w / 2)


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
            print("ok", k)
