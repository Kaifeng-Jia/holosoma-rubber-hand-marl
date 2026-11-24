#!/usr/bin/env python3
"""T14 step 8: per sequence, the candidates' evaluations -> results tables (candidates.csv rows) and, for the picture, a levels.json
in the step-7 format (render_box.py). Applicable sequence: the selected candidate and its levels. Not applicable: the solved
candidate with the fewest failed items at 1.0, labelled "不适用，仅示意" (no reference is made from it).

    sonic_s1/bin/python -B system1/t14/objects/step8_levels_json.py --tag smallboxb --out <dir>
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
L = ROOT / "logs_system1/t14/step8"


def conv(e, m, cdir, rec):
    foot = {}
    for r in (0, 1):
        for side in ("left", "right"):
            foot[f"r{r}_{side}"] = e["foot_slide"][f"r{r}"][f"foot_slide_{side}"]
    ret = e["contact_retention"]
    pooled = None if None in ret.values() else round((ret["r0"] + ret["r1"]) / 2, 4)
    return dict(m=m, pair=e.get("pair"), urdf=e["target_urdf"], length_along_axis_m=rec["Lm"], contact_retention=dict(ret, pooled=pooled),
                max_penetration_cm=e["max_penetration_cm"], robot_robot_min_distance_cm=e["robot_robot_min_distance_cm"],
                robot_robot_intersect_frames=e["robot_robot_intersect_frames"], foot_slide=foot, contact_position_error_3d=e["contact_position_error_3d"],
                applicable=e["applicable"], failed=e["failed"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    cd = L / f"cands/{a.tag}"
    cs = json.loads((cd / "candidates.json").read_text())
    summ = {}
    for f in sorted(L.glob("driver*.json"), key=lambda x: x.stat().st_mtime):  # re-runs (driver_rerun*.json) override the main run
        summ.update(json.loads(f.read_text()).get("summary", {}))
    summ = summ.get(a.tag, {})
    evals = {}
    for c in cs["candidates"]:
        f = cd / f"c{c['index']:02d}/m1p00/eval.json"
        if f.exists():
            evals[c["index"]] = json.loads(f.read_text())
    sel = summ.get("selected")
    if sel is None:
        if not evals:
            print("no solved candidate")
            return 0
        sel = sorted(evals, key=lambda i: (len(evals[i]["failed"]), i))[0]
        status = f"不适用，仅示意：失败项最少的已求解候选 c{sel:02d}（{'、'.join(evals[sel]['failed'])}）"
        levels = [1.0]
    else:
        status = f"适用：候选 c{sel:02d}"
        levels = sorted({1.0, 1.25, 1.5} | ({summ["a"]} if summ.get("a") else set()))
    cdir = cd / f"c{sel:02d}"
    out = {}
    for m in levels:
        d = cdir / f"m{m:.2f}".replace(".", "p")
        if (d / "eval.json").exists():
            rec = json.loads((d / "r0/run_record.json").read_text())
            out[f"{m:.2f}"] = conv(json.loads((d / "eval.json").read_text()), m, cdir, rec)
    rec0 = json.loads((cdir / "m1p00/r0/run_record.json").read_text())
    size = rec0["target"].get("size") or []
    if not size:  # table family: overall extents of the training table of the level
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import partner as pt
        cc = pt.urdf_corners(rec0["target"]["urdf"])
        size = (cc.max(0) - cc.min(0)).round(4).tolist()
    lv = dict(tag=a.tag, role_status=status, applicable=summ.get("selected") is not None, selected=sel, a=summ.get("a"), first_free_level_above_1=None,
              report_levels=[float(k) for k in out], levels=out, source_box_m=size if size else [0, 0, 0], solve_r0=str(cdir / "m1p00/r0"))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "levels.json").write_text(json.dumps(lv, indent=1, ensure_ascii=False) + "\n")
    print(a.tag, status, list(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
