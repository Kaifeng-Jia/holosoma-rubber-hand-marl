#!/usr/bin/env python3
"""T14 step 8: candidate pairs of the existing role check (partner.candidate_pairs; step-7 order) for one sequence and the
pre-solve gates of the new construction at a level (partner.construction): V0 gravity on S and the main-motion rule on the
shared trajectory. Writes <out>/candidates.json and per candidate <out>/c<NN>/cand.json + gates_m<level>.json.

    sonic_s1/bin/python -B system1/t14/objects/step8_candidates.py --family box --source logs_system1/t14/step8/src/smallboxb --cls b --out <dir>
    sonic_s1/bin/python -B system1/t14/objects/step8_candidates.py --family table --task push --cls b --out <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import partner as pt  # noqa: E402


def gates_for(O, cand, Lm_factor, corners_fn, cls, s):
    L1 = float(cand["L1"])
    C = pt.construction(O, cand, int(cand["axis"]), L1, Lm_factor * L1)
    g = pt.gravity_deg(C["S"])
    mm = pt.main_motion(C["P"], corners_fn(Lm_factor * L1), cls, s)
    return dict(m=Lm_factor, gravity_deg=round(g, 4), gravity_passed=g <= pt.V0_GRAVITY_DEG + 1e-10, main_motion=mm,
                passed=bool(g <= pt.V0_GRAVITY_DEG + 1e-10 and mm["passed"]), improper=bool(np.linalg.det(C["S"][:3, :3]) < 0),
                S=np.round(C["S"], 6).tolist(), S_translation_horizontal_m=round(float(np.linalg.norm(C["S"][:2, 3])), 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=("box", "table"), required=True)
    ap.add_argument("--source", type=Path, default=None)
    ap.add_argument("--task", choices=("push", "pull"), default=None)
    ap.add_argument("--cls", choices=("b", "c"), required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.family == "box":
        rec = json.loads((a.source / "run_record.json").read_text())
        z = np.load(a.source / f"{rec['tag']}_new_qpos.npz")
        src_urdf = rec["target"]["urdf"]
        size = np.array(rec["target"]["size"], float)
        cands, info = pt.candidate_pairs(z["qpos"][:, :36].astype(np.float64), z["qpos"][:, 36:43].astype(np.float64), src_urdf, "box")
        s = float(rec["smpl_scale"])
        source = str(a.source)
    else:
        T = pt.TABLES[a.task]
        rec = json.loads((T["single"] / "run_record.json").read_text())
        z = np.load(T["single"] / f"{a.task}_new_qpos.npz")
        cands, info = pt.candidate_pairs(z["qpos"][:, :36].astype(np.float64), z["qpos"][:, 36:43].astype(np.float64), pt.SOURCE_TABLE, "table",
                                         axis=T["axis"], target_urdf=T["target"], case=a.task)
        s = float(rec["smpl_scale"])
        source = str(T["single"])
    O = pt.mj_to_mat(z["qpos"][:, 36:43].astype(np.float64))
    a.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, c in enumerate(cands):
        if a.family == "box":
            def cf(L, c=c):
                sz = size.copy()
                sz[int(c["axis"])] = L
                return pt.box_corners(sz)
        else:
            import tempfile
            import dual_stretch as ds

            def cf(L, c=c):
                with tempfile.TemporaryDirectory() as d:
                    u = ds.stretch_table(T["target"], T["axis"], L / c["L1"], Path(d) / "t.urdf")
                    return pt.urdf_corners(u)
        g = gates_for(O, c, 1.0, cf, a.cls, s)
        d = a.out / f"c{i:02d}"
        d.mkdir(exist_ok=True)
        (d / "cand.json").write_text(json.dumps(c, indent=1) + "\n")
        (d / "gates_m1p00.json").write_text(json.dumps(g, indent=1) + "\n")
        rows.append(dict(index=i, id=c["id"], axis=c["axis"], parities=[c["parity0"], c["parity1"]], fit_rmse_m=c["fit_rmse_m"], **{k: g[k] for k in ("gravity_deg", "gravity_passed", "main_motion", "passed", "improper", "S_translation_horizontal_m")}))
        print(i, c["id"], "gravity %.2f" % g["gravity_deg"], "main", g["main_motion"], "PASS" if g["passed"] else "gated out", flush=True)
    (a.out / "candidates.json").write_text(json.dumps(dict(family=a.family, task=a.task, cls=a.cls, source=source, smpl_scale=s, pilot=info, candidates=rows,
                                                           rules=pt.__doc__, command=" ".join(sys.argv)), indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
