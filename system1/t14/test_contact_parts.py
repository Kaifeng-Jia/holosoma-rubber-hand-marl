"""Smoke tests for system1/t14/contact_parts.py (rule 5 rod flag, rule 4 signed distance, rule 7 opposing contact)."""
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
import contact_parts as cp  # noqa: E402


def flags_at(mesh, pts, n=6000):
    P, f = trimesh.sample.sample_surface(mesh, n, seed=1)
    N = mesh.face_normals[f]
    flag, gap, size = cp.rod_flags(P, N, cp.RADIUS)
    j = np.array([np.argmin(np.linalg.norm(P - q, axis=1)) for q in pts])
    return flag[j], gap[j], size[j]


def test_rod_shapes():
    cyl = trimesh.creation.cylinder(radius=0.015, height=1.0, sections=48)
    f, g, s = flags_at(cyl, [[0.015, 0, 0]])
    assert f[0] and 0.02 < s[0] < 0.05, (f, g, s)
    bar = trimesh.creation.box((0.03, 0.03, 1.0))
    f, g, s = flags_at(bar, [[0.015, 0, 0]])
    assert f[0], (f, g)
    plate = trimesh.creation.box((0.6, 0.035, 0.8))  # table top: face centre, long edge, nothing is a rod
    f, g, s = flags_at(plate, [[0, 0.0175, 0], [0.3, 0, 0], [0, 0, 0.4]])
    assert not f.any(), (f, g)
    box = trimesh.creation.box((0.3, 0.3, 0.3))  # face centre, edge, corner
    f, g, s = flags_at(box, [[0.15, 0, 0], [0.15, 0.15, 0], [0.149, 0.149, 0.149]])
    assert not f.any(), (f, g)
    strip = trimesh.creation.box((0.06, 0.015, 1.0))  # strip narrower than the palm
    f, g, s = flags_at(strip, [[0, 0.0075, 0]])
    assert f[0], (f, g)
    apron = trimesh.util.concatenate([trimesh.creation.box((0.8, 0.03, 0.6)),  # table top with an apron 4 cm in from the edge
                                      trimesh.creation.box((0.02, 0.08, 0.6)).apply_translation((0.36 - 0.01, -0.055, 0))])
    f, g, s = flags_at(apron, [[0.4, 0, 0], [0.38, 0.015, 0]])
    assert not f.any(), (f, g, s)
    flat = trimesh.creation.box((0.072, 0.037, 1.0))  # stick006-like: off-centre point of the wide face
    f, g, s = flags_at(flat, [[0.02, 0.0185, 0]])
    assert f[0], (f, g)


def test_signed_distance_and_opposing(tmp_path, monkeypatch):
    box = trimesh.creation.box((0.3, 0.3, 0.3))
    path = tmp_path / "box.obj"
    box.export(path)
    monkeypatch.setattr(cp, "CACHE", tmp_path)
    cp._OBJ.clear()
    cp.build_object("box", path, 1.0)
    inside = np.array([[0.14, 0, 0], [0.145, 0.01, 0]])
    r = cp.hand_frame([(inside, "box")], None, None)
    assert r is not None and 0.004 < r["depth"] < 0.012 and not r["rod"] and not r["opposing"]
    far = np.array([[0.2, 0, 0]])
    assert cp.hand_frame([(far, "box")], None, None) is None
    thin = trimesh.creation.box((0.02, 0.3, 0.3))
    path = tmp_path / "thin.obj"
    thin.export(path)
    cp.build_object("thin", path, 1.0)
    pinch = np.array([[0.012, 0, 0], [-0.012, 0, 0]])  # one point each side of a 2 cm plate
    r = cp.hand_frame([(pinch, "thin")], None, None)
    assert r is not None and r["opposing"]
