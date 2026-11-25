#!/usr/bin/env python3
"""T14 step 9c: which partner links make the V0 non-hand penetration fail? Diagnostic only (read only, CPU).

For a level directory (r0/ r1/ solves): the V0 round-2 CollisionProxy (the evaluation's engine) of each robot vs the shared
target; per robot: frames with non-hand proxy penetration (clearance < 0), the deepest value, the frame ranges, and the robot
links of the witnesses counted over those frames (scripts/reference_geometry_round2_checks.CollisionProxy witnesses).
The walking segment of the level's gates file is given for reference.

    sonic_s1/bin/python -B system1/t14/objects/step9_nonhand.py --levels <tag>=<level dir>@<gates json> ... --out <json>
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import shared_eval as se  # noqa: E402  (import paths)
from step9_intersect import ranges  # noqa: E402


def main():
    from reference_geometry_math import BoxUnion
    from reference_geometry_round2_checks import CollisionProxy
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    fk = se.ctx()["fk"]
    res = {}
    for item in a.levels:
        tag, rest = item.split("=", 1)
        d, _, gates = rest.partition("@")
        d = Path(d)
        recs = [json.loads((d / f"r{r}/run_record.json").read_text()) for r in (0, 1)]
        zs = [np.load(d / f"r{r}/{recs[r]['tag']}_new_qpos.npz") for r in (0, 1)]
        obj = zs[0]["object_qpos_mj"].astype(np.float64)
        eng = CollisionProxy(fk, BoxUnion(recs[0]["target"]["urdf"]))
        out = {}
        for r in (0, 1):
            q = zs[r]["qpos"].astype(np.float64)[:, :36]
            coll, wit = eng.evaluate([fk.poses(q, list(fk.meshes))], obj)
            c = coll["nonhand_object_clearance_m"][:, 0]
            bad = np.flatnonzero(c < 0)
            links = Counter(eng.names[w["geom_indices"][0] % eng.n] for w in wit if w["kind"] == "nonhand_object")
            out[f"r{r}"] = dict(frames_penetrating=int(len(bad)), deepest_m=round(float(-c.min()), 4) if len(bad) else 0.0, deepest_frame=int(c.argmin()),
                                ranges=ranges(bad), links=links.most_common(6))
        walk = json.loads(Path(gates).read_text())["diag"].get("walk") if gates else None
        res[tag] = dict(level_dir=str(d), frames=int(len(obj)), walk=None if walk is None else [walk["w0"], walk["w1"]], robots=out)
        print(tag, "walk", res[tag]["walk"], {r: (v["frames_penetrating"], v["deepest_m"], v["deepest_frame"], v["ranges"][:4], v["links"][:3]) for r, v in out.items()}, flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(dict(rules=__doc__, levels=res), indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
