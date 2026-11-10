#!/usr/bin/env python3
"""T04: tabulate the PhysX read-back object mass / inertia / COM for every scheme A mass level
(logs_system1/T04/A_readback/*/object_physics_readback.json) against the URDF values."""
import csv, json, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from system1.mass_variants import read_mass_inertia  # noqa: E402
rows = []
for p in sorted((ROOT / "logs_system1/T04/A_readback").glob("*/object_physics_readback.json")):
    d = json.loads(p.read_text()); ph = d["physics"]
    urdf = ROOT / "src/holosoma" / d["object_urdf"]
    m_urdf, I_urdf = read_mass_inertia(urdf.read_text())
    I = [float(x) for x in ph["inertia_row_major"][0][0]] if isinstance(ph["inertia_row_major"][0][0], list) else [float(x) for x in ph["inertia_row_major"][0]]
    want = [I_urdf["ixx"], I_urdf["ixy"], I_urdf["ixz"], I_urdf["ixy"], I_urdf["iyy"], I_urdf["iyz"], I_urdf["ixz"], I_urdf["iyz"], I_urdf["izz"]]
    m = float(ph["mass"][0][0])
    rows.append(dict(case=d["case"], run=p.parent.name, object_mass_kg_arg=d["object_mass_kg_arg"], object_urdf=d["object_urdf"],
                     object_urdf_sha256=d["object_urdf_sha256"], urdf_mass_kg=m_urdf, physx_mass_kg=m,
                     mass_abs_diff=abs(m - m_urdf),
                     urdf_inertia_row_major=json.dumps(want), physx_inertia_row_major=json.dumps(I),
                     inertia_max_abs_diff=max(abs(a - b) for a, b in zip(I, want)),
                     inertia_max_rel_diff=max(abs(a - b) / max(abs(b), 1e-12) for a, b in zip(I, want) if abs(b) > 0),
                     physx_com_pose_xyzw=json.dumps(ph["com_pose_xyzw"]), physx_material=json.dumps(ph["material"])))
out = ROOT / "results_system1/T04/A/object_physics_readback.csv"
with out.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows:
    print(r["run"], r["urdf_mass_kg"], r["physx_mass_kg"], f"{r['inertia_max_abs_diff']:.3g}", f"{r['inertia_max_rel_diff']:.3g}")
