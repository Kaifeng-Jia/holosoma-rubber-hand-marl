"""Signed distance field of an object mesh on a regular grid (T06 finger optimisation), CPU.

Convention: negative inside the mesh, positive outside (metres). The grid is computed once with open3d's
RaycastingScene (exact closest-triangle distance, sign from ray parity; the mesh must be watertight) and sampled with
trilinear interpolation in torch, so the value is differentiable in the query points. Outside the grid the distance to
the grid box is added (a conservative positive value). The bucket mesh of the Bin task is a closed solid
(results_system1/T06/feasibility), the same mesh the simulator decomposes into convex parts.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch


class GridSDF:
    def __init__(self, mesh_path: Path, resolution: float = 0.003, margin: float = 0.06, cache_dir: Path | None = None):
        import trimesh
        self.mesh_path = Path(mesh_path)
        self.mesh_sha256 = hashlib.sha256(self.mesh_path.read_bytes()).hexdigest()
        m = trimesh.load(self.mesh_path, force="mesh")
        if not m.is_watertight:
            raise ValueError(f"{mesh_path} is not watertight: the sign of the distance is undefined")
        lo, hi = m.bounds[0] - margin, m.bounds[1] + margin
        n = np.ceil((hi - lo) / resolution).astype(int) + 1
        self.lo, self.res, self.n = lo, resolution, n
        cache = None
        if cache_dir is not None:
            cache = Path(cache_dir) / f"sdf_{self.mesh_sha256[:16]}_{resolution:.4f}_{margin:.3f}.npy"
        if cache is not None and cache.exists():
            grid = np.load(cache)
        else:
            import open3d as o3d
            scene = o3d.t.geometry.RaycastingScene()
            tm = o3d.t.geometry.TriangleMesh()
            tm.vertex.positions = o3d.core.Tensor(np.asarray(m.vertices, np.float32))
            tm.triangle.indices = o3d.core.Tensor(np.asarray(m.faces, np.int32))
            scene.add_triangles(tm)
            axes = [lo[i] + resolution * np.arange(n[i]) for i in range(3)]
            P = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3).astype(np.float32)
            grid = np.empty(len(P), np.float32)
            for s in range(0, len(P), 2_000_000):
                grid[s:s + 2_000_000] = scene.compute_signed_distance(o3d.core.Tensor(P[s:s + 2_000_000]), nsamples=3).numpy()
            grid = grid.reshape(*n)
            if cache is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache, grid)
        self.grid = torch.as_tensor(grid, dtype=torch.float64)
        self.lo_t = torch.as_tensor(lo, dtype=torch.float64)
        self.hi_t = torch.as_tensor(lo + resolution * (n - 1), dtype=torch.float64)

    def __call__(self, p: torch.Tensor) -> torch.Tensor:
        """Signed distance at points p (..., 3) in the mesh frame."""
        p = torch.as_tensor(p, dtype=torch.float64)
        pc = torch.maximum(torch.minimum(p, self.hi_t), self.lo_t)
        outside = torch.linalg.norm(p - pc, dim=-1)
        u = (pc - self.lo_t) / self.res
        i0 = torch.clamp(torch.floor(u).long(), min=torch.zeros(3, dtype=torch.long), max=torch.as_tensor(self.n - 2))
        f = u - i0
        g = self.grid
        out = 0.0
        for dx in (0, 1):
            wx = f[..., 0] if dx else 1 - f[..., 0]
            for dy in (0, 1):
                wy = f[..., 1] if dy else 1 - f[..., 1]
                for dz in (0, 1):
                    wz = f[..., 2] if dz else 1 - f[..., 2]
                    out = out + wx * wy * wz * g[i0[..., 0] + dx, i0[..., 1] + dy, i0[..., 2] + dz]
        return out + outside
