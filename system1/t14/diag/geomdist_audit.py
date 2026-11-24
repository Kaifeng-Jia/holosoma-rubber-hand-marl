"""T14 round 3: audit mj_geomDistance of the running MuJoCo against the exact convex distance (Minkowski difference of the
two vertex sets: origin inside -> -depth to the nearest facet; outside -> distance to the hull) for every robot-object
candidate pair (mj_collision, margin 0.1) of a solved reference in its retargeting scene.
    python diag/geomdist_audit.py <scene xml> <qpos npz> <object body> [--box-as-mesh] [--stride N]"""
import sys, json
import numpy as np, mujoco, trimesh
from scipy.spatial import ConvexHull
scene, qpath, objb = sys.argv[1:4]
stride = int(sys.argv[sys.argv.index("--stride") + 1]) if "--stride" in sys.argv else 1
if "--box-as-mesh" in sys.argv:
    import xml.etree.ElementTree as ET
    from pathlib import Path
    tree = ET.parse(scene); root = tree.getroot(); asset = root.find("asset")
    out = Path(scene).with_name(Path(scene).stem + "_boxmesh_audit.xml")
    for b in root.iter("body"):
        if b.get("name") != objb:
            continue
        for g in b.findall("geom"):
            if g.get("type") == "box" and not (g.get("contype") == "0" and g.get("conaffinity") == "0"):
                hs = np.array([float(x) for x in g.get("size").split()])
                p = Path(scene).parent / f"auditbox_{g.get('name')}.obj"
                trimesh.creation.box(extents=2 * hs).export(p)
                ET.SubElement(asset, "mesh", name=f"auditbox_{g.get('name')}", file=str(p.resolve()))
                g.set("type", "mesh"); g.set("mesh", f"auditbox_{g.get('name')}"); g.attrib.pop("size")
    tree.write(out); scene = str(out)
m = mujoco.MjModel.from_xml_path(scene); d = mujoco.MjData(m)
q = np.load(qpath)["qpos"]
ob = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, objb)
obj = {g for g in range(m.ngeom) if m.geom_bodyid[g] == ob}
names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(m.ngeom)]
def verts(g):
    if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
        c = np.array([[a, b, e] for a in (-1, 1) for b in (-1, 1) for e in (-1, 1)]) * m.geom_size[g]
    else:
        mid = m.geom_dataid[g]; c = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
    return c @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
def exact(g1, g2):
    H = ConvexHull((verts(g1)[:, None] - verts(g2)[None]).reshape(-1, 3)); b = H.equations[:, 3]
    if np.all(b <= 1e-12):
        return float(np.max(b))  # -depth
    mh = trimesh.Trimesh(H.points, H.simplices); mh.fix_normals()
    return float(trimesh.proximity.closest_point(mh, np.zeros((1, 3)))[1][0])
ft = np.zeros(6); n = 0; worst = 0.0; bad = []
for t in range(0, len(q), stride):
    d.qpos[:] = q[t]; mujoco.mj_forward(m, d)
    sv = m.geom_margin.copy(); m.geom_margin[:] = 0.1; mujoco.mj_collision(m, d); m.geom_margin[:] = sv
    cand = {(min(c.geom1, c.geom2), max(c.geom1, c.geom2)) for c in d.contact[:d.ncon]}
    for g1, g2 in cand:
        if not ((g1 in obj) ^ (g2 in obj)):
            continue
        if mujoco.mjtGeom.mjGEOM_PLANE in (m.geom_type[g1], m.geom_type[g2]):  # object-ground: not a constraint pair
            continue
        dm = mujoco.mj_geomDistance(m, d, g1, g2, 0.1, ft)
        if dm >= 0.1:
            continue
        ex = exact(g1, g2); n += 1
        err = abs(dm - ex)
        if err > 1e-5:
            bad.append((t, names[g1], names[g2], round(dm, 5), round(ex, 5)))
        worst = max(worst, err)
print(json.dumps(dict(mujoco=mujoco.__version__, scene=scene, pairs_checked=n, pairs_wrong=len(bad), worst_abs_err_m=worst, wrong_examples=bad[:8])))
