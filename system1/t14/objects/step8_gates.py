#!/usr/bin/env python3
"""T14 step 8: pre-solve gates (V0 gravity on S, main motion of the shared trajectory) of one candidate at level m, and for the
table family the level's training-table URDF (dual_stretch.stretch_table). Writes <cdir>/gates_m<tag>.json.

    sonic_s1/bin/python -B system1/t14/objects/step8_gates.py --cdir <.../cNN> --cands <candidates.json> --m 1.25
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import partner as pt  # noqa: E402
from step8_candidates import gates_for  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cdir", type=Path, required=True)
ap.add_argument("--cands", type=Path, required=True)
ap.add_argument("--m", type=float, required=True)
a = ap.parse_args()
cs = json.loads(a.cands.read_text())
cand = json.loads((a.cdir / "cand.json").read_text())
tag = f"m{a.m:.2f}".replace(".", "p")
if cs["family"] == "box":
    rec = json.loads((Path(cs["source"]) / "run_record.json").read_text())
    z = np.load(Path(cs["source"]) / f"{rec['tag']}_new_qpos.npz")
    size = np.array(rec["target"]["size"], float)

    def cf(L):
        sz = size.copy()
        sz[int(cand["axis"])] = L
        return pt.box_corners(sz)
    urdf = None
else:
    import dual_stretch as ds
    T = pt.TABLES[cs["task"]]
    z = np.load(T["single"] / f"{cs['task']}_new_qpos.npz")
    urdf = ds.stretch_table(T["target"], T["axis"], a.m, a.cdir / tag / f"table_{tag}.urdf")

    def cf(L):
        return pt.urdf_corners(urdf)
O = pt.mj_to_mat(z["qpos"][:, 36:43].astype(np.float64))
g = gates_for(O, cand, a.m, cf, cs["cls"], float(cs["smpl_scale"]))
if urdf is not None:
    g["target_urdf"] = str(urdf)
(a.cdir / f"gates_{tag}.json").write_text(json.dumps(g, indent=1) + "\n")
print(json.dumps({k: g[k] for k in ("m", "gravity_deg", "passed")}), g["main_motion"])
