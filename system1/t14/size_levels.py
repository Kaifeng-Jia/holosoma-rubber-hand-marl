#!/usr/bin/env python3
"""T14: compose the levels.json consumed by render_sizes.py (contact moments + one line of numbers per level).

Numbers per level: contact retention (CoCarry definition, pooled over both robots), max penetration (convex penetration
depth between any robot link shape and the object, either robot, simulator geometry: hull_penetration.py), solve success (finite, object locked; per-person solves for
the two-person tasks, the single solve for the single-person ones), smallest robot-robot distance (exact convex distance
between the two robots' simulator link shapes, dual_stretch.PairCollision; "> 50 cm" when no pair is within 50 cm).

    sonic_s1/bin/python -B system1/t14/size_levels.py --task table --out logs_system1/t14/levels/table.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "system1/t14"))
REF = ROOT / "exports_system1/T14_references"


def moments(flags):
    """flags [T] bool (30 Hz) -> [first, middle, last] contact frames."""
    i = np.flatnonzero(flags)
    return [int(i[0]), int(round((i[0] + i[-1]) / 2)), int(i[-1])]


def numbers(ret, pen_cm, ok, rr_cm):
    rr = "> 50 cm" if not np.isfinite(rr_cm) else (f"{10 * rr_cm:.2f} mm（不相交）" if 0 <= rr_cm < 0.1 else f"{rr_cm:.1f} cm")
    return [f"接触保持率 {ret:.3f}", f"最大穿透 {pen_cm:.2f} cm", f"求解成功：{'是' if ok else '否'}", f"机器人最小间隙 {rr}"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=("table", "push", "pull", "bin", "chair"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    levels = []
    hp = json.loads((ROOT / "results_system1/T14_retargeting/step2_sizes/hull_penetration.json").read_text())["refs"]
    import dual_stretch as ds
    pc = ds.PairCollision(ds.cs.Robot("physx"))
    if a.task == "table":
        import eval_table_refs as et
        h, _ = et.human_hand_sd()
        f15 = (h >= -0.015).any(axis=(1, 2))
        frames = [2 * x for x in moments(f15)]
        m = json.loads((ROOT / "results_system1/T14_retargeting/step2_sizes/table/metrics.json").read_text())["refs"]
        rr = json.loads((ROOT / "results_system1/T14_retargeting/step2_sizes/table/robot_robot_hull_distance.json").read_text())
        for k in ("0.8", "0.9", "1.0", "1.1", "1.2"):
            tag = "k" + k.replace(".", "p")
            r = m["k" + k]
            recs = [json.loads((ROOT / f"logs_system1/t14/table/{'new' if k == '1.0' else 'new_' + tag}/person{p}/run_record.json").read_text()) for p in (1, 2)]
            ok = all(x["finite"] and x["object_locked_max_err"] == 0.0 for x in recs)
            pen = max(hp[f"table_{tag}"]["max_penetration_cm"].values())
            urdf = REF / f"table_{tag}" / f"desk001_{tag}.urdf"
            levels.append(dict(label=f"k = {k}", size=float(k), pair=str((REF / f"table_{tag}/20231030_001_desk001_move_new_fps30.npz").relative_to(ROOT)),
                               urdf=str(urdf.relative_to(ROOT)), numbers=numbers(r["contact_retention"]["pooled"], pen, ok, rr[tag[1:]]["min_distance_cm"]),
                               values=dict(retention=r["contact_retention"]["pooled"], max_penetration_cm=pen, success=ok, robot_robot_min_distance_cm=rr[tag[1:]]["min_distance_cm"])))
    elif a.task in ("bin", "chair"):
        import eval_core4d_hulls as eh
        h, _ = eh.human_hand_sd(a.task)
        frames = [2 * x for x in moments((h >= -0.015).any(axis=(1, 2)))]
        m = json.loads((ROOT / f"results_system1/T14_retargeting/step2_sizes/{a.task}/metrics.json").read_text())["refs"]
        import re
        for k in ("0.8", "0.9", "1.0", "1.1", "1.2"):
            tag = "k" + k.replace(".", "p")
            rp = [ROOT / f"logs_system1/t14/core4d/{a.task}_{tag}/person{p}/run_record.json" for p in (1, 2)]
            if not all(x.exists() for x in rp):  # a person's solve failed: the level has no two-robot reference
                why = []
                for p, x in zip((1, 2), rp):
                    if not x.exists():
                        log = (ROOT / f"logs_system1/t14/core4d/{a.task}_{tag}_p{p}.log").read_text(errors="replace")
                        prog = re.findall(r"(\d+)/(\d+) \[", log)
                        err = "Clarabel 求解失败" if "SolverError" in log else "未完成"
                        why.append(f"机器人 {p - 1}：{err}" + (f"（第 {prog[-1][0]}/{prog[-1][1]} 帧）" if prog else ""))
                levels.append(dict(label=f"k = {k}", size=float(k), failed="；".join(why), numbers=["求解成功：否"] + why,
                                   values=dict(retention=None, max_penetration_cm=None, success=False, robot_robot_min_distance_cm=None)))
                continue
            r = m["k" + k]
            recs = [json.loads(x.read_text()) for x in rp]
            ok = all(x["finite"] and x["object_locked_max_err"] == 0.0 for x in recs)
            pen = max(r["max_penetration_cm"].values())
            name = "原尺寸桶" if (a.task == "bin" and k == "1.0") else ""
            levels.append(dict(label=f"k = {k}" + (f"（{name}）" if name else ""), size=float(k),
                               pair=str((REF / f"{a.task}_{tag}/{a.task}_new_fps30.npz").relative_to(ROOT)),
                               urdf=str((REF / f"{a.task}_{tag}/{a.task}_{tag}.urdf").relative_to(ROOT)),
                               numbers=numbers(r["contact_retention"]["pooled"], pen, ok, r["robot_robot_min_distance_cm"]),
                               values=dict(retention=r["contact_retention"]["pooled"], max_penetration_cm=pen, success=ok,
                                           robot_robot_min_distance_cm=r["robot_robot_min_distance_cm"])))
    else:
        h, _ = ds.human_hand_sd(ds.SEQ[a.task])
        frames = moments((h >= -0.015).any(axis=1))
        sz = json.loads((REF / f"{a.task}_sizes/sizes.json").read_text())
        rec = json.loads((ROOT / f"logs_system1/t14/single/{a.task}/run_record.json").read_text())
        ok = rec["finite"] and rec["object_locked_max_err"] == 0.0
        for mlev in sz["report_levels"]:
            r = sz["levels"][str(float(mlev))]
            tag = f"m{mlev:.2f}".replace(".", "p")
            pen = max(hp[f"{a.task}_{tag}"]["max_penetration_cm"].values())
            name = f"a = {mlev:.2f}" if mlev == sz["a"] else f"m = {mlev:.1f}"
            rr = pc.min_distance(np.load(REF / f"{a.task}_sizes/{tag}/{a.task}_{tag}_pair_fps30.npz")["robot_qpos"])
            r["robot_robot_min_distance_cm"] = float(100 * rr.min())  # unrounded (sizes.json keeps 2 decimals in cm)
            levels.append(dict(label=f"{name}（桌长 {1.4 * mlev:.2f} m）", size=float(mlev), pair=str((REF / f"{a.task}_sizes/{tag}/{a.task}_{tag}_pair_fps30.npz").relative_to(ROOT)),
                               urdf=str((REF / f"{a.task}_sizes/{tag}/table_{tag}.urdf").relative_to(ROOT)),
                               numbers=numbers(r["contact_retention"]["pooled"], pen, ok, r["robot_robot_min_distance_cm"]),
                               values=dict(retention=r["contact_retention"]["pooled"], max_penetration_cm=pen, success=ok, robot_robot_min_distance_cm=r["robot_robot_min_distance_cm"])))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(dict(task=a.task, contact_frames_30hz=frames, levels=levels), indent=1, ensure_ascii=False) + "\n")
    print(a.task, frames, [(L["label"], L["numbers"]) for L in levels])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
