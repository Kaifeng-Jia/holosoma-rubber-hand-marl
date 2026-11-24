"""T14 round 3: nearest point on the surface of an object given as a union of convex pieces (the simulator's collision
shape: Table / Push / Pull boxes, Bin / Chair PhysX pieces), used by the step-2 anchors and the contact-position error.

The union's boundary is represented by uniform surface samples of every piece (spacing ~1.5 mm by default) with the samples
strictly inside another piece removed; nearest = the nearest kept sample (error <= about half the spacing)."""
from __future__ import annotations

import numpy as np
import trimesh
from scipy.spatial import ConvexHull, cKDTree


class UnionSurface:
    def __init__(self, meshes, spacing=0.0015, seed=0):
        rng = np.random.default_rng(seed)
        eqs = [ConvexHull(np.asarray(m.vertices)).equations for m in meshes]
        keep, nrm = [], []
        for i, m in enumerate(meshes):
            n = max(200, int(np.ceil(m.area / spacing ** 2)))
            pts, fid = trimesh.sample.sample_surface(m, n, seed=int(rng.integers(1 << 31)))
            ok = np.ones(len(pts), bool)
            for j, e in enumerate(eqs):
                if j != i:
                    ok &= ~np.all(pts @ e[:, :3].T + e[:, 3] < -1e-6, axis=1)
            keep.append(pts[ok])
            e = eqs[i]  # outward unit normal: the facet plane of this convex piece the sample lies on
            nrm.append(e[np.argmin(np.abs(pts[ok] @ e[:, :3].T + e[:, 3]), axis=1), :3])
        self.points = np.concatenate(keep)
        self.normals = np.concatenate(nrm)  # outward unit normal of the piece face each sample lies on
        self.tree = cKDTree(self.points)
        self.spacing = spacing

    def nearest(self, P, normals=False):
        d, i = self.tree.query(np.asarray(P, float).reshape(-1, 3))
        out = (self.points[i].reshape(np.shape(P)), d.reshape(np.shape(P)[:-1]))
        return out + (self.normals[i].reshape(np.shape(P)),) if normals else out
